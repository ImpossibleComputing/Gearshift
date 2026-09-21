#!/usr/bin/env python3
"""One bounded on-pod setup, followed by an independently detached supervisor.

Only setup requires networking. There is no provider lifecycle API or credential
input here. The already armed, separate lease guard owns allocation shutdown.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gearshift.coding_coverage_v2_lease import (
    atomic_json, control_root, linux_process_identity, load_lease, sha256_file)
from scripts.coding_coverage_v2_supervisor import scoped, verify_plan

PACKAGES = ('transformers==4.57.6', 'accelerate==1.15.0', 'numpy==2.2.6',
            'psutil==7.2.2', 'safetensors==0.7.0', 'huggingface-hub==0.36.2', 'matplotlib==3.11.2')
CACHE = '/workspace/hf'


def read(path):
    return json.loads(Path(path).read_text())


def child_environment(*, offline=False):
    # An allowlist also excludes credentials hidden in proxy/package-index URLs.
    allowed = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'LIBRARY_PATH',
               'CUDA_HOME', 'CUDA_PATH', 'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES')
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env.update(HF_HOME=CACHE, HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_OFFLINE='1' if offline else '0',
               TRANSFORMERS_OFFLINE='1' if offline else '0', HF_DATASETS_OFFLINE='1' if offline else '0',
               HF_HUB_ETAG_TIMEOUT='30', HF_HUB_DOWNLOAD_TIMEOUT='60',
               TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONUNBUFFERED='1', PIP_DISABLE_PIP_VERSION_CHECK='1', DEBIAN_FRONTEND='noninteractive')
    return env


def command_plan(repo, gpu_count, base_python=sys.executable):
    python = str(Path(repo) / '.pilot-venv/bin/python')
    runtime_check = (
        'import importlib.metadata as m,json,torch; '
        'expected=' + repr(dict(package.split('==') for package in PACKAGES)) + '; '
        'actual={k:m.version(k) for k in expected}; assert actual==expected,(actual,expected); '
        'assert torch.__version__.startswith("2.8.0"),torch.__version__; '
        f'assert torch.cuda.device_count()=={gpu_count},torch.cuda.device_count(); '
        'gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; '
        'assert all("H200" in name for name in gpus),gpus; '
        'print(json.dumps({"packages":actual,"torch":torch.__version__,"cuda":torch.version.cuda,"gpus":gpus}),flush=True)')
    return [
        ('apt_update', ['apt-get', 'update'], 1200),
        ('sandbox_packages', ['apt-get', 'install', '-y', '--no-install-recommends', 'python3', 'python3-venv', 'libseccomp2'], 1200),
        ('sandbox_probe', ['/usr/bin/python3', 'scripts/coding_sandbox_probe.py'], 300),
        ('venv', [base_python, '-m', 'venv', '--system-site-packages', '.pilot-venv'], 300),
        ('pinned_packages', [python, '-m', 'pip', 'install', *PACKAGES], 1200),
        ('runtime_check', [python, '-c', runtime_check], 180),
        ('report_dependency_check', [python, '-c',
            'import io,json,numpy as np,matplotlib; matplotlib.use("Agg"); '
            'import matplotlib.pyplot as plt; import scripts.coding_coverage_v2_report as report; '
            'assert callable(report.render) and callable(report.cluster_statistics); '
            'assert np.quantile(np.array([0.,1.]),.5)==.5; '
            'fig,ax=plt.subplots(); ax.plot([0,1],[0,1]); buffer=io.BytesIO(); '
            'fig.savefig(buffer,format="png"); plt.close(fig); '
            'assert buffer.getvalue().startswith(bytes([137,80,78,71])); '
            'print(json.dumps({"report_import":True,"headless_png_render":True,"numpy":np.__version__,"matplotlib":matplotlib.__version__}),flush=True)'], 180),
        ('package_inventory', [python, '-m', 'pip', 'freeze'], 120),
        ('pinned_model_download', [python, 'scripts/coding_download_models.py'], 5400),
    ]


def mounted_volume(repo, volume_id, mountinfo=Path('/proc/self/mountinfo')):
    if not isinstance(volume_id, str) or not volume_id:
        raise ValueError('Provider network-volume identity is required')
    repo = Path(repo).resolve()
    repo.relative_to(Path('/workspace'))
    matches = []
    for line in Path(mountinfo).read_text().splitlines():
        before, after = line.split(' - ', 1)
        left, right = before.split(), after.split()
        mount = Path(left[4].replace('\\040', ' ').replace('\\011', '\t').replace('\\134', '\\'))
        if repo.is_relative_to(mount):
            matches.append((len(mount.parts), mount, left[2], right[0], right[1]))
    if not matches:
        raise ValueError('No mounted filesystem covers the scientific repository')
    _, mount, device, filesystem, source = max(matches)
    if not mount.is_relative_to(Path('/workspace')):
        raise ValueError('Repository is on the temporary container filesystem, not the /workspace volume')
    return {'network_volume_id': volume_id, 'mount_point': str(mount),
            'device_major_minor': device, 'filesystem': filesystem,
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'repository': str(repo), 'separate_workspace_mount_verified': True}


def probe_volume(folder, python=sys.executable):
    """Verify atomic commit/durability operations and cross-process lock behavior."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    data = os.urandom(32768); temporary = folder / 'atomic.tmp'; final = folder / 'atomic.bin'
    with temporary.open('wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.rename(temporary, final)
    directory = os.open(folder, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    if final.read_bytes() != data or temporary.exists():
        raise ValueError('Mounted volume atomic rename/fsync verification failed')
    lock = folder / 'queue.lock'
    contender = ('import fcntl,sys; f=open(sys.argv[1],"a"); '
                 '\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
                 '\nexcept BlockingIOError: sys.exit(7)'
                 '\nelse: sys.exit(0)')
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocked = subprocess.run([python, '-c', contender, str(lock)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=child_environment(), timeout=15)
        if blocked.returncode != 7:
            raise ValueError('Mounted volume does not exclude a competing process lock')
    released = subprocess.run([python, '-c', contender, str(lock)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=child_environment(), timeout=15)
    if released.returncode != 0:
        raise ValueError('Mounted volume lock did not release')
    return {'atomic_rename_verified': True, 'file_fsync_verified': True, 'directory_fsync_verified': True,
            'cross_process_flock_exclusion_verified': True, 'flock_release_verified': True,
            'probe_bytes': len(data), 'probe_sha256': sha256_file(final)}


class Bootstrap:
    def __init__(self, repo, plan_path, lease_path):
        self.repo = Path(repo).resolve()
        self.plan_path, self.lease_path = scoped(self.repo, plan_path), scoped(self.repo, lease_path)
        self.plan = read(self.plan_path)
        if scoped(self.repo, self.plan['allocation_lease_path']) != self.lease_path:
            raise ValueError('Bootstrap lease differs from frozen allocation plan')
        self.lease = load_lease(self.lease_path, self.plan['allocation_lease_sha256'])
        self.root = verify_plan(self.repo, self.plan, self.lease)
        self.deadline = self.lease['deadline_epoch'] - 120
        self.allocation_control = control_root(self.lease)
        self.execution_role = self.plan.get('execution_role', 'primary')
        if self.execution_role not in ('primary', 'evaluation_helper'):
            raise ValueError('Unknown allocation execution role')
        self.local_gpu_count = self.plan.get('local_gpu_count', self.lease['gpu_count'])
        if type(self.local_gpu_count) is not int or self.local_gpu_count < 1:
            raise ValueError('Invalid local GPU population')
        self.attempt_id = self.lease['pod_id'] + '_attempt_' + str(time.time_ns())
        self.control = self.allocation_control / 'bootstrap'; self.control.mkdir(parents=True, exist_ok=True)
        self.attempt = self.control / self.attempt_id; self.attempt.mkdir(exist_ok=False)
        self.rows = []

    def guard(self):
        if time.time() >= self.deadline or (self.allocation_control / 'lease_guard/STOP').exists():
            raise TimeoutError('Bootstrap reached immutable allocation deadline or local STOP')

    def record(self, state, **details):
        row = {'state': state, 'epoch': time.time(), 'attempt_id': self.attempt_id,
               'experiment_id': self.plan['experiment_id'], 'pod_id': self.lease['pod_id'], **details}
        atomic_json(self.attempt / 'status.json', row)
        atomic_json(self.control / 'latest.json', row)
        return row

    def verify_guard(self):
        guard = read(self.allocation_control / 'lease_guard/status.json')
        if guard.get('state') != 'armed' or any(guard.get(k) != self.lease[k] for k in ('experiment_id', 'pod_id', 'deadline_epoch')):
            raise ValueError('Independent allocation lease guard is not armed for this exact lease')
        os.kill(guard['pid'], 0)
        self.guard()
        return {'guard_pid': guard['pid'], 'deadline_epoch': guard['deadline_epoch'],
                'guard_independent_of_bootstrap': True}

    def existing_supervisor(self):
        path = self.allocation_control / 'lease_guard/supervisor_registration.json'
        if not path.exists(): return None
        registration = read(path)
        if any(registration.get(k) != self.lease[k] for k in ('experiment_id', 'pod_id')):
            raise ValueError('Existing supervisor registration differs from this allocation')
        try: process = linux_process_identity(registration['pid'])
        except FileNotFoundError: return None
        if all(process[k] == registration.get(k) for k in ('pid', 'pgid', 'start_ticks', 'boot_id')):
            return registration
        return None

    def run_command(self, label, command, maximum_seconds):
        self.guard(); started = time.time()
        timeout = min(maximum_seconds, self.deadline - started)
        if timeout <= 0: raise TimeoutError('No allocation time remains for bootstrap')
        log = self.attempt / (f'{len(self.rows):02d}_' + label + '.log')
        self.record('setting_up', command=command, label=label, log=str(log.relative_to(self.root)))
        with log.open('ab') as output:
            completed = subprocess.run(command, cwd=self.repo, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, env=child_environment(offline=self.execution_role == 'evaluation_helper'), timeout=timeout)
            output.flush(); os.fsync(output.fileno())
        row = {'label': label, 'command': command, 'returncode': completed.returncode,
               'started_epoch': started, 'seconds': time.time() - started,
               'log': str(log.relative_to(self.root)), 'log_sha256': sha256_file(log), 'log_bytes': log.stat().st_size}
        self.rows.append(row); atomic_json(self.attempt / 'commands.json', self.rows)
        if completed.returncode:
            # Keep the complete command log; the failure receipt remains small.
            raise RuntimeError('Bootstrap command failed: ' + label + ' (exit ' + str(completed.returncode) + ')')
        self.guard()
        return row

    def verify_receipts(self):
        evidence = self.repo / 'evidence/coding_pilot_v1'
        destination = self.attempt / 'download_receipts'; destination.mkdir(exist_ok=True)
        models = read(self.repo / 'configs/coding_pilot_v1/pilot.json')['models']
        verification = read(evidence / 'weight_verification.json')
        if {r['role'] for r in verification['models']} != set(models):
            raise ValueError('Downloaded model verification population differs')
        records = []
        for role, model in models.items():
            pin_path = evidence / (role + '_weight_pins.json'); pins = read(pin_path)
            if pins['revision'] != model['revision'] or not pins['sha256']:
                raise ValueError('Downloaded model revision or shard pins differ')
            previous = evidence / 'coverage_v2_input_provenance' / pin_path.name
            if previous.exists() and read(previous) != pins:
                raise ValueError('Fresh pinned model hashes differ from preserved historical pins')
            entry = next(r for r in verification['models'] if r['role'] == role)
            if entry['revision'] != model['revision'] or entry['shards_verified'] != len(pins['sha256']) or entry['bytes'] <= 0:
                raise ValueError('Model download did not verify all pinned shards')
            # The unchanged downloader already rereads every shard for SHA-256.
            # The model index closure below rejects absent/unpinned shard names.
            snapshot = Path(CACHE) / 'hub' / ('models--' + model['id'].replace('/', '--')) / 'snapshots' / model['revision']
            index = read(snapshot / 'model.safetensors.index.json')
            required = set(index['weight_map'].values())
            if required != set(pins['sha256']) or any(not (snapshot / name).is_file() for name in required):
                raise ValueError('Pinned model shard index is incomplete')
            shutil.copyfile(pin_path, destination / pin_path.name)
            records.append({'role': role, 'model_id': model['id'], 'revision': model['revision'],
                'cache_snapshot': str(snapshot), 'shards_verified': len(required), 'bytes': entry['bytes'],
                'fresh_weight_pins_sha256': sha256_file(pin_path), 'historical_pins_compared': previous.exists()})
        shutil.copyfile(evidence / 'weight_verification.json', destination / 'weight_verification.json')
        sandbox = read(evidence / 'sandbox_gate.json')
        if sandbox.get('passed') is not True or not sandbox.get('checks') or not all(sandbox['checks'].values()):
            raise ValueError('Generated-code sandbox probe did not pass all checks')
        if sandbox['implementation_sha256'] != sha256_file(self.repo / 'scripts/coding_sandbox_child.py'):
            raise ValueError('Sandbox probe implementation differs')
        shutil.copyfile(evidence / 'sandbox_gate.json', self.attempt / 'sandbox_gate.json')
        proof = {'models': records, 'sandbox_passed': True,
                 'sandbox_gate_sha256': sha256_file(self.attempt / 'sandbox_gate.json'),
                 'model_download_script_sha256': sha256_file(self.repo / 'scripts/coding_download_models.py'),
                 'weight_verification_sha256': sha256_file(destination / 'weight_verification.json'),
                 'runtime_packages': list(PACKAGES), 'workers_run_offline': True}
        atomic_json(self.attempt / 'setup_verification.json', proof)
        return proof

    def shared_setup_contract(self):
        return {'schema_version': 1, 'experiment_id': self.plan['experiment_id'],
                'code_commit': self.plan['code_commit'], 'declaration_sha256': self.plan['declaration_sha256'],
                'input_manifest_sha256': hashlib.sha256(json.dumps(self.plan['input_manifest'], sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                'network_volume_id': self.plan['network_volume_id'], 'repository': str(self.repo),
                'runtime_packages': list(PACKAGES), 'hf_home': CACHE,
                'environment_path': str(self.repo / '.pilot-venv')}

    def verify_shared_setup(self, receipt):
        payload = receipt['payload']
        observed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if receipt.get('payload_sha256') != observed:
            raise ValueError('Shared setup receipt payload hash differs')
        if any(payload.get(key) != value for key, value in self.shared_setup_contract().items()):
            raise ValueError('Shared setup scientific/runtime identity differs')
        files = payload.get('files')
        if not isinstance(files, list) or not files or len({r['path'] for r in files}) != len(files):
            raise ValueError('Shared setup immutable evidence manifest is missing or duplicated')
        for row in files:
            path = scoped(self.repo, row['path'])
            if not path.is_file() or path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
                raise ValueError('Shared setup evidence hash differs: ' + row['path'])
        required = {'setup_verification.json', 'sandbox_gate.json', 'source_weight_pins.json',
                    'receiver_weight_pins.json', 'weight_verification.json', 'pyvenv.cfg'}
        if not required <= {Path(r['path']).name for r in files}:
            raise ValueError('Shared setup is missing environment/model/sandbox evidence')
        proof_path = scoped(self.repo, payload['setup_verification_path'])
        if payload['setup_verification_path'] not in {r['path'] for r in files}:
            raise ValueError('Shared setup proof is unbound')
        proof = read(proof_path)
        if proof.get('sandbox_passed') is not True or proof.get('workers_run_offline') is not True or proof.get('runtime_packages') != list(PACKAGES):
            raise ValueError('Shared setup proof failed controls')
        return payload

    def publish_shared_setup(self):
        if self.execution_role != 'primary':
            raise ValueError('Only primary may publish shared installation/download readiness')
        paths = [self.attempt / 'setup_verification.json', self.attempt / 'sandbox_gate.json',
                 self.repo / '.pilot-venv/pyvenv.cfg']
        paths.extend(sorted((self.attempt / 'download_receipts').glob('*.json')))
        paths.extend(self.root / row['log'] for row in self.rows
                     if row['label'] in ('runtime_check', 'report_dependency_check', 'package_inventory'))
        files = []
        for path in paths:
            with path.open('rb') as stream: os.fsync(stream.fileno())
            files.append({'path': str(path.relative_to(self.repo)), 'sha256': sha256_file(path), 'bytes': path.stat().st_size})
        payload = {**self.shared_setup_contract(), 'creator_pod_id': self.lease['pod_id'],
                   'setup_verification_path': str((self.attempt / 'setup_verification.json').relative_to(self.repo)),
                   'files': files}
        receipt = {'payload': payload, 'payload_sha256': hashlib.sha256(json.dumps(payload,
                   sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()}
        self.verify_shared_setup(receipt)
        path = self.root / 'bootstrap/shared_setup_ready.json'
        if path.exists():
            existing = read(path); self.verify_shared_setup(existing)
            if existing != receipt:
                raise ValueError('Immutable shared setup receipt already exists; do not mutate a live shared environment')
        else:
            atomic_json(path, receipt)
        return receipt

    def wait_for_shared_setup(self):
        if self.execution_role != 'evaluation_helper':
            raise ValueError('Only an evaluation helper waits on shared primary setup')
        path = self.root / 'bootstrap/shared_setup_ready.json'
        self.record('waiting_for_primary_shared_setup', receipt_path=str(path.relative_to(self.root)))
        while not path.exists():
            self.guard(); time.sleep(2)
        self.guard(); receipt = read(path); payload = self.verify_shared_setup(receipt)
        atomic_json(self.attempt / 'shared_setup_verified.json', {'shared_receipt_sha256': sha256_file(path),
            'payload_sha256': receipt['payload_sha256'], 'creator_pod_id': payload['creator_pod_id'],
            'shared_environment_and_models_reused': True, 'network_required': False,
            'pip_download_sandbox_mutations_performed': False})
        # Preserve a local setup summary; all model/sandbox evidence remains
        # bound through the immutable primary receipt, not rewritten by helpers.
        atomic_json(self.attempt / 'setup_verification.json', {'shared_receipt_sha256': sha256_file(path),
            'primary_setup_verification_path': payload['setup_verification_path'],
            'helper_reuses_shared_setup': True, 'runtime_packages': list(PACKAGES), 'workers_run_offline': True})
        return receipt

    def launch_supervisor(self):
        self.guard()
        command = [str(self.repo / '.pilot-venv/bin/python'), 'scripts/coding_coverage_v2_supervisor.py',
                   '--plan', str(self.plan_path.relative_to(self.repo)), '--lease', str(self.lease_path.relative_to(self.repo))]
        log = self.attempt / 'supervisor_launch.log'
        with log.open('ab') as output:
            process = subprocess.Popen(command, cwd=self.repo, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, env=child_environment(offline=True), start_new_session=True)
        end = min(time.monotonic() + 45, time.monotonic() + max(0., self.deadline - time.time()))
        while time.monotonic() < end:
            if process.poll() is not None:
                raise RuntimeError('Scientific supervisor exited during independent startup (see preserved launch log)')
            registration = self.existing_supervisor()
            if registration and registration['pid'] == process.pid and registration['pgid'] == process.pid:
                result = {'command': command, 'pid': process.pid, 'pgid': process.pid,
                          'registration': registration, 'detached_new_session': True,
                          'credential_environment_inherited': False, 'workers_offline': True,
                          'launcher_liveness_required_after_dispatch': False}
                atomic_json(self.attempt / 'supervisor_started.json', result)
                return result
            self.guard(); time.sleep(.25)
        # The independent lease guard remains responsible for this process.
        # Preserve the PID without creating an unregistered duplicate attempt.
        atomic_json(self.attempt / 'supervisor_startup_unconfirmed.json', {'pid': process.pid, 'command': command})
        raise TimeoutError('Supervisor registration was not confirmed within45 seconds')

    def run(self):
        try:
            guard = self.verify_guard()
            running = self.existing_supervisor()
            if running:
                return self.record('already_running', supervisor=running)
            for name, receipt in self.plan['input_manifest'].items():
                path = scoped(self.repo, name)
                if path.is_symlink() or not path.is_file() or path.stat().st_size != receipt['bytes'] or sha256_file(path) != receipt['sha256']:
                    raise ValueError('Immutable staged input changed: ' + name)
            volume_id = self.plan.get('network_volume_id')
            if volume_id != self.lease.get('network_volume_id'):
                raise ValueError('Plan and lease network-volume identities differ')
            mount = mounted_volume(self.repo, volume_id)
            probe = probe_volume(self.attempt / 'volume_probe')
            atomic_json(self.attempt / 'volume_verification.json', {**mount, **probe})
            atomic_json(self.attempt / 'guard_verification.json', guard)
            if self.execution_role == 'evaluation_helper':
                self.wait_for_shared_setup()
                commands = [row for row in command_plan(self.repo, self.local_gpu_count)
                            if row[0] in ('runtime_check', 'report_dependency_check', 'package_inventory')]
                for label, command, maximum_seconds in commands:
                    self.run_command(label, command, maximum_seconds)
            else:
                # A verified shared environment must never be modified while
                # helper readers might be active, including on bootstrap retry.
                shared = self.root / 'bootstrap/shared_setup_ready.json'
                if shared.exists():
                    self.verify_shared_setup(read(shared))
                    original = read(scoped(self.repo, read(shared)['payload']['setup_verification_path']))
                    atomic_json(self.attempt / 'setup_verification.json', original)
                else:
                    for label, command, maximum_seconds in command_plan(self.repo, self.local_gpu_count):
                        self.run_command(label, command, maximum_seconds)
                    self.verify_receipts()
                    self.publish_shared_setup()
            self.record('setup_complete', proof_sha256=sha256_file(self.attempt / 'setup_verification.json'))
            supervisor = self.launch_supervisor()
            return self.record('dispatched', supervisor=supervisor,
                               setup_verification_sha256=sha256_file(self.attempt / 'setup_verification.json'))
        except BaseException as error:
            failure = {'exception': type(error).__name__, 'error': str(error)[:2048],
                       'traceback': traceback.format_exc()[-8192:], 'epoch': time.time(),
                       'attempt_id': self.attempt_id, 'experiment_id': self.plan['experiment_id'],
                       'completed_commands': len(self.rows), 'provider_lifecycle_action_taken': False}
            atomic_json(self.attempt / 'failure.json', failure)
            self.record('failed', failure_receipt=str((self.attempt / 'failure.json').relative_to(self.root)),
                        exception=type(error).__name__)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--lease', required=True)
    args = parser.parse_args()
    bootstrap = Bootstrap(REPO, args.plan, args.lease)
    with (bootstrap.control / 'bootstrap.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bootstrap.run()


if __name__ == '__main__': main()
