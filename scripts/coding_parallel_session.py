#!/usr/bin/env python3
"""Studio-owned one-shot stage dispatch to at most eight isolated H200s.

No inference choice is made here. Every stage requires hash-bound scientific
proofs, an exact uploaded file manifest and a fixed worker command/task list.
Use --validate-only before installing a controller. No automatic run retry.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tomllib
import traceback
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import PREFIX, decision, sha, write
from gearshift.coding_capacity_retry import confirmed_no_capacity
from gearshift.coding_parallel import (check_dispatch_budget, checkpoint_files,
    file_in, validate_plan, verify_worker_status, worker_input_files, worker_spec)
from gearshift.coding_snapshot import verify_archive
from coding_cloud_guard import cli, tick
from coding_cloud_session import quote, ssh_args

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'evidence/coding_pilot_v1/control'
REMOTE = '/workspace/Gearshift'
APPROVAL = C / 'parallel_500h_approved.json'
ALLOCATION_LOCK = threading.Lock()


def remote(conn, command, timeout=120):
    p = subprocess.run(ssh_args(conn) + [command], capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError('Remote command failed: ' + p.stderr[-1500:] + p.stdout[-1500:])
    return p.stdout


def collect(conn, dest, spec=None):
    """Each worker has an independent archive, extracted tree and acknowledgments."""
    dest.mkdir(parents=True, exist_ok=True)
    temp = dest / 'latest.tar.gz.tmp'
    command = 'cd ' + REMOTE + ' && timeout --signal=TERM --kill-after=5s 150s python3 scripts/coding_snapshot.py'
    if spec is not None:
        command += ' --worker-spec ' + shlex.quote(spec['worker_root'] + '/worker_spec.json')
    with temp.open('wb') as f:
        p = subprocess.run(ssh_args(conn) + [command],
            stdout=f, stderr=subprocess.PIPE, timeout=180)
        if p.returncode:
            raise RuntimeError('Compact backup failed: ' + p.stderr.decode()[-500:])
        f.flush(); os.fsync(f.fileno())
    stage = Path(tempfile.mkdtemp(prefix='verified-', dir=dest))
    try:
        manifest = verify_archive(temp, stage)
        if spec is not None:
            scope = manifest.get('scope', {})
            expected = {key: spec[key] for key in ['run_id', 'worker_id', 'stage_identity', 'result_root', 'worker_root']}
            if scope.get('kind') != 'worker' or any(scope.get(key) != value for key, value in expected.items()):
                raise ValueError('Backup worker scope differs from allocated identity')
            spec_path = stage / spec['worker_root'] / 'worker_spec.json'
            if not spec_path.is_file() or sha(spec_path) != scope.get('worker_spec_sha256'):
                raise ValueError('Snapshot spec identity did not verify')
            if sha(spec_path) != sha(ROOT / spec['worker_root'] / 'worker_spec.json'):
                raise ValueError('Snapshot specification differs from frozen Studio input')
    except BaseException:
        shutil.rmtree(stage); raise
    previous = dest / 'previous'
    if previous.exists(): shutil.rmtree(previous)
    if (dest / 'latest').exists(): os.replace(dest / 'latest', previous)
    os.replace(stage, dest / 'latest')
    os.replace(temp, dest / 'latest.tar.gz')
    write(dest / 'backup_ack.json', {'epoch': time.time(), 'archive_sha256': sha(dest / 'latest.tar.gz'),
        'verified_files': len(manifest['files']), 'snapshot_epoch': manifest['epoch'], 'scope': manifest.get('scope'),
        'capture_seconds': manifest.get('capture_seconds'), 'uncompressed_bytes': manifest.get('uncompressed_bytes')})
    return dest / 'latest'


def copy_checkpoints(conn, saved, dest, spec):
    """Copy committed mapper/paired-feature hashes; never full model weights."""
    files, manifests = {}, {}
    for name in ['checkpoint_manifest.json', 'feature_manifest.json']:
        manifest_path = saved / spec['result_root'] / name
        if not manifest_path.exists(): continue
        manifest = json.loads(manifest_path.read_text())
        files.update(checkpoint_files(manifest, spec['result_root'], features=name == 'feature_manifest.json'))
        manifests[name] = sha(manifest_path)
    if not files: return {}
    copies = {}
    for rel, expected in files.items():
        target = dest / 'mapper_checkpoints' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != expected: raise ValueError('Immutable mapper checkpoint changed')
        else:
            # Remote size check occurs before a bounded read. No shell path expansion.
            code = ('import pathlib,sys; p=pathlib.Path(' + repr(REMOTE + '/' + rel) + '); '
                'assert not p.is_symlink() and p.is_file(); '
                'assert p.stat().st_size <= 1024**3; '
                'f=p.open("rb"); import shutil; shutil.copyfileobj(f,sys.stdout.buffer)')
            temp = target.with_suffix(target.suffix + '.tmp')
            with temp.open('wb') as f:
                p = subprocess.run(ssh_args(conn) + ['python3 -c ' + shlex.quote(code)],
                    stdout=f, stderr=subprocess.PIPE, timeout=240)
                f.flush(); os.fsync(f.fileno())
            if p.returncode or temp.stat().st_size > 1024**3 or sha(temp) != expected:
                temp.unlink(missing_ok=True)
                raise ValueError('Mapper checkpoint transfer did not verify')
            os.replace(temp, target)
        copies[rel] = {'sha256': expected, 'bytes': target.stat().st_size,
            'local_path': str(target.relative_to(ROOT))}
    if sum(v['bytes'] for v in copies.values()) > 8 * 1024**3:
        raise ValueError('Mapper checkpoint set exceeds compact stage bound')
    write(dest / 'mapper_backup_ack.json', {'epoch': time.time(), 'files': copies,
        'manifest_sha256': manifests})
    return copies


def upload(conn, files, package):
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    manifest_path = package.with_suffix('.manifest.json')
    write(manifest_path, {'files': hashes})
    with tarfile.open(package, 'w') as tar:
        for p in files + [manifest_path]:
            tar.add(p, arcname=p.relative_to(ROOT), recursive=False)
    with package.open('rb') as f:
        p = subprocess.run(ssh_args(conn) + ['mkdir -p ' + REMOTE +
            ' && tar --no-same-owner --no-same-permissions -xf - -C ' + REMOTE],
            stdin=f, capture_output=True, timeout=3600)
    if p.returncode: raise RuntimeError('Worker upload failed: ' + p.stderr.decode()[-500:])
    # Large source-trajectory manifests exceed SSH/exec single-argument limits.
    # Transfer the manifest as data and keep the verifier command bounded.
    code = ('import json,pathlib,sys; r=pathlib.Path(' + repr(REMOTE) + '); sys.path.insert(0,str(r)); '
        'from gearshift.coding_control import sha; files=json.loads((r/' +
        repr(str(manifest_path.relative_to(ROOT))) + ').read_text())["files"]; '
        'assert all(sha(r/f)==h for f,h in files.items())')
    remote(conn, 'python3 -c ' + shlex.quote(code), 180)
    return hashes


def own_resources():
    return {kind: [r for r in cli(command, 'list') if r.get('name', '').startswith(PREFIX)]
        for kind, command in [('pods', 'pod'), ('volumes', 'network-volume')]}


def sanitized_pod(pod):
    """Keep allocation evidence without provider environment or public keys."""
    fields = ['id', 'name', 'gpuCount', 'imageName', 'costPerHr', 'adjustedCostPerHr',
        'networkVolumeId', 'machineId', 'desiredStatus', 'volumeMountPath', 'containerDiskInGb']
    result = {key: pod[key] for key in fields if key in pod}
    if isinstance(pod.get('networkVolume'), dict):
        result['networkVolume'] = {key: pod['networkVolume'][key]
            for key in ['id', 'name', 'size', 'dataCenterId'] if key in pod['networkVolume']}
    if isinstance(pod.get('machine'), dict):
        result['machine'] = {key: pod['machine'][key] for key in ['id', 'gpuTypeId', 'gpuDisplayName', 'dataCenterId']
            if key in pod['machine']}
        if isinstance(pod['machine'].get('gpuType'), dict):
            result['machine']['gpuType'] = {key: pod['machine']['gpuType'][key] for key in ['id', 'displayName']
                if key in pod['machine']['gpuType']}
    return result


def check_pod(pod, volume_id, image, *, require_machine=False):
    nested_volume = (pod.get('networkVolume') or {}).get('id')
    volume = pod.get('networkVolumeId') or nested_volume
    if nested_volume is not None and pod.get('networkVolumeId') is not None and nested_volume != pod['networkVolumeId']:
        raise ValueError('Provider volume associations disagree')
    if pod.get('gpuCount') != 1 or volume != volume_id:
        raise ValueError('Provider allocation differs from single-GPU request: '
            f'gpuCount={pod.get("gpuCount")!r}, volume={volume!r}, expected_volume={volume_id!r}')
    if not 0 < float(pod.get('costPerHr') or 0) <= 5.5:
        raise ValueError('Provider price exceeds approved H200 ceiling')
    if pod.get('imageName') != image:
        raise ValueError('Provider image differs from pinned image')
    machine = pod.get('machine') or {}
    gpu_type = machine.get('gpuTypeId') or (machine.get('gpuType') or {}).get('id') or machine.get('gpuDisplayName')
    if (require_machine or gpu_type is not None) and gpu_type not in ('NVIDIA H200', 'H200', 'H200 SXM', 'NVIDIA H200 SXM'):
        raise ValueError('Provider machine is not the approved H200')


def pod_detail_rest(pod_id):
    """Avoid installed CLI 2.1.9 dropping the network-volume response fields."""
    if not isinstance(pod_id, str) or not pod_id.isalnum() or len(pod_id) > 80:
        raise ValueError('Invalid provider pod ID')
    auth = tomllib.loads((Path.home() / '.runpod/config.toml').read_text())
    url = 'https://rest.runpod.io/v1/pods/' + pod_id + '?includeNetworkVolume=true&includeMachine=true'
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + auth['apikey'],
        'User-Agent': 'gearshift-pilot'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def verify_created_pod(created, volume_id, image, evidence_path=None):
    # Runpod documents these GET query fields. Older runpodctl versions discard
    # them while decoding JSON (upstream fix #322), so use the raw read-only API.
    detail = pod_detail_rest(created['id'])
    sanitized = sanitized_pod(detail)
    if evidence_path is not None:
        write(evidence_path, {'epoch': time.time(), 'source': 'documented_rest_get',
            'response_field_names': sorted(detail), 'pod': sanitized})
    if detail.get('id') != created['id'] or detail.get('name') != created['name']:
        raise ValueError('Fresh provider pod identity differs from creation receipt')
    check_pod(detail, volume_id, image, require_machine=True)
    return sanitized


def process_alive(conn, pid, spec_path):
    # A zombie has no command line; kill(0) alone would wait forever for it.
    code = ('import pathlib; p=pathlib.Path("/proc/' + str(int(pid)) + '/cmdline"); '
        'b=p.read_bytes() if p.exists() else b""; '
        'print("alive" if b"coding_parallel_session.py" in b and ' + repr(spec_path.encode()) + ' in b else "")')
    return remote(conn, 'python3 -c ' + shlex.quote(code), 30).strip() == 'alive'


def controller_heartbeat(controller_id, state):
    receipt = {'epoch': time.time(), 'controller_id': controller_id, 'pid': os.getpid(), 'state': state}
    write(C / 'controller_heartbeats' / (controller_id + '.json'), receipt)
    write(C / 'controller_heartbeat.json', receipt)


def allocation_names(plan, worker):
    base = PREFIX + plan['run_id'] + '-' + worker['worker_id']
    if 'region_candidates' not in worker:
        return [base]
    return [base + '-capacity' + str(i + 1).zfill(2) for i in range(len(worker['region_candidates']))]


def verify_empty_capacity_attempt(volume, name, control, index, failure, stage_identity):
    """Clean and prove an unallocated capacity attempt before trying a region."""
    if not confirmed_no_capacity(failure):
        raise ValueError('Only an exact no-instance response can authorize capacity retry')
    pods = cli('pod', 'list')
    if any(row.get('name') == name for row in pods):
        raise ValueError('Capacity response is contradicted by an allocated pod')
    receipts = C / 'resource_receipts'
    history = [json.loads(path.read_text()) for path in receipts.glob('*.json')]
    ledger_path = C / 'ledger.json'
    if ledger_path.exists(): history.extend(json.loads(ledger_path.read_text()).get('resources', []))
    for receipt in history:
        if receipt.get('kind') == 'pod' and receipt.get('name') == name:
            raise ValueError('A GPU was already recorded for this capacity attempt')
    cli('network-volume', 'delete', volume['id'])
    volumes = cli('network-volume', 'list')
    if any(row.get('id') == volume['id'] for row in volumes):
        raise ValueError('Temporary capacity volume is not confirmed absent')
    now = time.time()
    path = receipts / (volume['id'] + '.json')
    receipt = json.loads(path.read_text()); receipt['absent_epoch'] = now; write(path, receipt)
    # Reconcile confirmed volume deletion before the next dispatch decision.
    # A stale registration-pending observation must not block a safe region retry.
    tick()
    write(control / f'capacity_{index:02d}_verified.json', {'epoch': now, 'capacity_only': True,
        'failure': failure, 'name': name, 'volume_id': volume['id'], 'all_resources_absent': True,
        'no_gpu_ever_allocated': True, 'region_retry_allowed': True, 'stage_identity': stage_identity})


def launch_worker(plan, worker, identity, stop_event):
    run_id, worker_id = plan['run_id'], worker['worker_id']
    control = C / 'parallel' / run_id / worker_id
    control.mkdir(parents=True, exist_ok=False)
    dest = ROOT / 'evidence/coding_pilot_v1/parallel_backups' / run_id / worker_id
    conn = pod = volume = None
    verified = False
    started = time.time()
    deadline = started + worker['maximum_seconds']
    status = {'state': 'allocating', 'epoch': started, 'stage_identity': identity}
    write(control / 'status.json', status)
    try:
        with ALLOCATION_LOCK:
            if stop_event.is_set() or (C / 'STOP').exists(): raise RuntimeError('Dispatch stopped')
            ledger = json.loads((C / 'ledger.json').read_text())
            current = decision(ledger)
            if current['stop'] or current.get('dispatch_blocked'):
                raise RuntimeError('Budget/watchdog dispatch stop')
            # Regions are frozen before dispatch. A retry exists only when the
            # provider explicitly allocated no GPU and temporary storage is gone.
            regions = worker.get('region_candidates', [worker['region']])
            for index, (region, name) in enumerate(zip(regions, allocation_names(plan, worker)), 1):
                if stop_event.is_set() or (C / 'STOP').exists() or time.time() >= deadline - 180:
                    raise RuntimeError('Stopped before capacity attempt')
                check = decision(json.loads((C / 'ledger.json').read_text()))
                if check['stop'] or check.get('dispatch_blocked'):
                    raise RuntimeError('Watchdog blocks new capacity dispatch: ' + str(check.get('stop_reasons', []) + check.get('dispatch_block_reasons', [])))
                attempt_started = time.time()
                write(control / f'allocation_intent_{index:02d}.json', {'epoch': attempt_started,
                    'controller_id': run_id, 'name': name, 'region': region,
                    'stage_identity': identity, 'image_digest': plan['image_digest']})
                # Keep the conventional first/current intent for status readers.
                write(control / 'allocation_intent.json', {'epoch': attempt_started, 'controller_id': run_id,
                    'name': name, 'stage_identity': identity, 'region': region, 'image_digest': plan['image_digest']})
                volume = cli('network-volume', 'create', '--name', name + '-storage', '--size', '250', '--data-center-id', region)
                write(control / f'volume_{index:02d}.json', volume); write(control / 'volume.json', volume)
                write(C / 'resource_receipts' / (volume['id'] + '.json'), {'kind': 'volume', 'id': volume['id'],
                    'name': volume['name'], 'started_epoch': attempt_started, 'upper_rate_usd': .04, 'controller_id': run_id})
                try:
                    pod = cli('pod', 'create', '--name', name, '--image', plan['image_digest'],
                        '--gpu-id', 'NVIDIA H200', '--gpu-count', '1', '--cloud-type', 'SECURE',
                        '--data-center-ids', region, '--container-disk-in-gb', '40',
                        '--network-volume-id', volume['id'], '--ports', '22/tcp', '--ssh')
                except RuntimeError as exc:
                    failure = {'state': 'failed', 'epoch': time.time(), 'error': str(exc)}
                    write(control / f'capacity_{index:02d}_failure.json', failure)
                    if not confirmed_no_capacity(failure) or index == len(regions): raise
                    verify_empty_capacity_attempt(volume, name, control, index, failure, identity)
                    volume = None
                    continue
                write(control / 'pod.json', sanitized_pod(pod))
                write(C / 'resource_receipts' / (pod['id'] + '.json'), {'kind': 'pod', 'id': pod['id'],
                    'name': pod['name'], 'started_epoch': attempt_started, 'upper_rate_usd': 5.52,
                    'controller_id': run_id, 'gpu_count': 1, 'gpu_type': 'NVIDIA H200'})
                verified_pod = verify_created_pod(pod, volume['id'], plan['image_digest'], control / 'pod_detail_observed.json')
                write(control / 'pod_verified.json', {'epoch': time.time(), 'passed': True, 'pod': verified_pod})
                break
        for _ in range(60):
            if stop_event.is_set() or (C / 'STOP').exists() or time.time() >= deadline - 180:
                raise RuntimeError('Stopped before worker connection')
            try:
                info = cli('ssh', 'info', pod['id']); candidate = info.get('connection', info)
                if candidate and candidate.get('ip'):
                    remote(candidate, 'true', 30); conn = candidate; break
            except Exception: pass
            stop_event.wait(10)
        if conn is None: raise TimeoutError('Pod startup exceeded ten minutes')
        write(control / 'connection.json', conn)
        worker_rel = f'evidence/coding_pilot_v1/parallel/{run_id}/{worker_id}'
        allocation_rel = worker_rel + '/allocation.json'
        allocation = {'pod_id': pod['id'], 'volume_id': volume['id'], 'started_epoch': started,
            'deadline_epoch': deadline, 'preflight_deadline_epoch': deadline,
            'remaining_authorized_seconds': deadline - started, 'image_digest': plan['image_digest'],
            'cap_usd': 1000, 'approval_sha256': plan['approval_sha256'], 'stage_identity': identity,
            'controller_id': run_id, 'worker_id': worker_id,
            'prior_usage': decision(json.loads((C / 'ledger.json').read_text()))}
        spec = worker_spec(plan, worker, identity, deadline, allocation_rel)
        spec['command'] = worker['command']
        write(ROOT / allocation_rel, allocation)
        spec_path = ROOT / worker_rel / 'worker_spec.json'; write(spec_path, spec)
        selected_files = worker_input_files(plan, worker)
        files = [file_in(ROOT, rel) for rel in selected_files]
        if any(sha(path) != selected_files[str(path.relative_to(ROOT))] for path in files):
            raise ValueError('Frozen upload input changed after allocation; stop without generation')
        files += [ROOT / allocation_rel, spec_path]
        hashes = upload(conn, files, control / 'upload.tar')
        if any(hashes[rel] != expected for rel, expected in selected_files.items()):
            raise ValueError('Uploaded input differs from immutable dispatch identity')
        write(control / 'upload_manifest.json', {'epoch': time.time(), 'files': hashes})
        code = ('import pathlib,subprocess,os; r=pathlib.Path(' + repr(REMOTE) + '); '
            'e=r/' + repr(worker_rel) + '; f=(e/"bootstrap.log").open("ab"); '
            'p=subprocess.Popen(["python3","scripts/coding_parallel_session.py","--pod-bootstrap",' +
            repr(str(spec_path.relative_to(ROOT))) + '],cwd=r,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)')
        pid = int(remote(conn, 'python3 -c ' + shlex.quote(code), 30).strip())
        write(control / 'process.json', {'pid': pid, 'epoch': time.time()})
        last_backup = time.time()
        terminal = False
        while time.time() < deadline - 120 and not stop_event.is_set() and not (C / 'STOP').exists():
            try:
                saved = collect(conn, dest, spec)
                copy_checkpoints(conn, saved, dest, spec)
                last_backup = time.time()
                write(C / 'backups_ack' / (volume['id'] + '.json'), {'verified': True,
                    'volume_id': volume['id'], 'epoch': last_backup,
                    'archive_sha256': sha(dest / 'latest.tar.gz')})
                path = saved / spec['worker_status_path']
                if path.exists():
                    status = json.loads(path.read_text()); state = verify_worker_status(status, spec)
                    write(control / 'worker_status.json', status)
                    terminal = state in ('complete', 'failed')
                bootstrap = saved / worker_rel / 'bootstrap_status.json'
                if bootstrap.exists() and json.loads(bootstrap.read_text()).get('state') == 'failed':
                    status = json.loads(bootstrap.read_text()); terminal = True
                    write(control / 'worker_status.json', status)
                if terminal and not process_alive(conn, pid, str(spec_path.relative_to(ROOT))):
                    break
                write(control / 'heartbeat.json', {'epoch': time.time(), 'last_backup_epoch': last_backup})
            except Exception as exc:
                write(control / 'backup_retry.json', {'epoch': time.time(), 'error': str(exc), 'last_backup_epoch': last_backup})
                if time.time() - last_backup > 600: raise RuntimeError('Ten minutes without verified worker backup')
            stop_event.wait(30)
        remote(conn, 'touch ' + REMOTE + '/' + worker_rel + '/STOP ' + REMOTE + '/evidence/coding_pilot_v1/STOP', 30)
        for _ in range(12):
            if not process_alive(conn, pid, str(spec_path.relative_to(ROOT))): break
            time.sleep(5)
        else: raise RuntimeError('Worker did not drain; storage retained')
        saved = collect(conn, dest, spec)
        checkpoints = copy_checkpoints(conn, saved, dest, spec)
        latest = dest / 'latest.tar.gz'; second = dest / 'final.tar.gz'
        shutil.copy2(latest, second)
        if sha(latest) != sha(second): raise ValueError('Final backup copies differ')
        receipt = {'epoch': time.time(), 'verified': True, 'pod_id': pod['id'], 'volume_id': volume['id'],
            'worker_confirmed_absent': True, 'copies': [str(p.relative_to(ROOT)) for p in [latest, second]],
            'sha256': sha(latest), 'mapper_checkpoints': checkpoints, 'stage_identity': identity}
        write(control / 'backup_verified.json', receipt)
        write(C / 'backups_verified' / (volume['id'] + '.json'), receipt)
        verified = True
        status_path = saved / spec['worker_status_path']
        status = json.loads(status_path.read_text()) if status_path.exists() else {'state': 'failed', 'error': 'Worker did not complete'}
        if status.get('state') != 'complete':
            raise RuntimeError('Stage worker stopped before completion: ' + str(status))
        verify_worker_status(status, spec)
        return {'worker_id': worker_id, 'passed': True, 'status': status, 'backup_sha256': sha(latest)}
    except BaseException as exc:
        status = {'state': 'failed', 'epoch': time.time(), 'error': str(exc)}
        write(control / 'error.json', status)
        # A shard failure blocks the next stage but does not discard other
        # independent shards. Global budget/watchdog stops still stop all work.
        raise
    finally:
        if pod:
            try: cli('pod', 'delete', pod['id'])
            except Exception as exc: write(control / 'pod_cleanup_pending.json', {'epoch': time.time(), 'error': str(exc)})
        # No connected storage is deleted without a verified final checkpoint.
        if volume and (verified or conn is None):
            try: cli('network-volume', 'delete', volume['id'])
            except Exception as exc: write(control / 'volume_cleanup_pending.json', {'epoch': time.time(), 'error': str(exc)})
        write(control / 'status.json', {'epoch': time.time(), 'state': 'stopped_backed_up' if verified else 'stopped_inspect_backup',
            'worker_state': status.get('state'), 'stage_identity': identity})


def bootstrap_command_timeout(command, remaining_seconds):
    """Allow slow pinned downloads within, never beyond, the same allocation."""
    limit = 3600 if command[-1] == 'scripts/coding_download_models.py' else 1800
    return min(limit, remaining_seconds)


def bootstrap(spec_path):
    """Pod-only bounded bootstrap and child supervisor, without provider access."""
    spec = json.loads((ROOT / spec_path).read_text())
    work = ROOT / spec['worker_root']; work.mkdir(parents=True, exist_ok=True)
    deadline = spec['deadline_epoch'] - 120
    env = dict(os.environ, HF_HOME='/workspace/hf', HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        GEARSHIFT_WORKER_SPEC=str(ROOT / spec_path), GEARSHIFT_WORKER_ROOT=str(work), GEARSHIFT_REPO_ROOT=str(ROOT))
    py = str(ROOT / '.pilot-venv/bin/python')
    commands = [['apt-get', 'update'], ['apt-get', 'install', '-y', '--no-install-recommends', 'python3', 'python3-venv', 'libseccomp2'],
        ['/usr/bin/python3', 'scripts/coding_sandbox_probe.py'],
        [sys.executable, '-m', 'venv', '--system-site-packages', '.pilot-venv'],
        [py, '-m', 'pip', 'install', 'transformers==4.57.6', 'accelerate==1.15.0', 'numpy==2.2.6', 'psutil==7.2.2', 'safetensors==0.7.0'],
        [py, '-c', 'import torch; assert torch.__version__.startswith("2.8.0"); assert torch.cuda.device_count()==1; assert "H200" in torch.cuda.get_device_name(0)'],
        [py, 'scripts/coding_download_models.py']]
    child = None
    try:
        for command in commands:
            if (work / 'STOP').exists() or time.time() >= deadline: raise TimeoutError('Bootstrap deadline or stop')
            write(work / 'bootstrap_status.json', {'state': 'bootstrapping', 'epoch': time.time(), 'command': command})
            subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=bootstrap_command_timeout(command, deadline - time.time()))
        child = subprocess.Popen([py] + spec['command'], cwd=ROOT, env=env, start_new_session=True)
        while child.poll() is None:
            if (work / 'STOP').exists() or (ROOT / 'evidence/coding_pilot_v1/STOP').exists() or time.time() >= deadline:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise TimeoutError('Worker deadline or stop')
            write(work / 'bootstrap_status.json', {'state': 'running', 'epoch': time.time(), 'child_pid': child.pid})
            time.sleep(5)
        if child.returncode: raise RuntimeError('Worker process failed: ' + str(child.returncode))
        status = json.loads((ROOT / spec['worker_status_path']).read_text())
        if verify_worker_status(status, spec) != 'complete': raise RuntimeError('Worker did not publish complete identity')
        write(work / 'bootstrap_status.json', {'state': 'complete', 'epoch': time.time()})
    except BaseException as exc:
        write(work / 'bootstrap_status.json', {'state': 'failed', 'epoch': time.time(), 'error': str(exc)})
        traceback.print_exc()
        raise


def verify_idle_resources(plan, resources):
    """Retain only explicitly bound, twice-backed-up volumes between stages.

    This permits preserving an inspected failed attempt's storage while a new
    isolated worker uses fresh storage. Any active pod or unaccounted volume
    still blocks dispatch; retained storage remains in the cumulative ledger.
    """
    if resources['pods']:
        raise ValueError('Previous task pods must be stopped before a new stage')
    expected={}
    for ref in plan.get('retained_storage',[]):
        if ref['volume_id'] in expected or not ref['name'].startswith(PREFIX):
            raise ValueError('Invalid retained-volume identity')
        path=file_in(ROOT,ref['recovery_receipt'])
        if sha(path)!=ref['sha256'] or plan['files'].get(ref['recovery_receipt'])!=ref['sha256']:
            raise ValueError('Retained-volume recovery proof changed')
        proof=json.loads(path.read_text())
        copies=proof.get('copies',[])
        if (proof.get('volume_id')!=ref['volume_id'] or proof.get('both_copies_verified') is not True
            or len(copies)!=2 or len(set(copies))!=2):
            raise ValueError('Retained volume needs two verified recovery copies')
        for rel in copies:
            if sha(file_in(ROOT,rel))!=proof['archive_sha256'] or plan['files'].get(rel)!=proof['archive_sha256']:
                raise ValueError('Retained-volume backup changed')
        expected[ref['volume_id']]=ref['name']
    actual={v['id']:v['name'] for v in resources['volumes']}
    if len(actual)!=len(resources['volumes']) or actual!=expected:
        raise ValueError('Unaccounted task storage blocks dispatch')


def run(plan_path, validate_only=False):
    plan = json.loads(plan_path.read_text())
    identity = validate_plan(plan, ROOT, sha(APPROVAL))
    ledger = json.loads((C / 'ledger.json').read_text())
    reservation = check_dispatch_budget(plan, decision(ledger))
    if validate_only:
        print(json.dumps({'validated': True, 'stage_identity': identity, **reservation})); return
    with (C / 'recovery_controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_control = C / 'parallel' / plan['run_id']
        if run_control.exists(): raise ValueError('One-shot dispatch already exists; no automatic retry')
        resources = own_resources()
        verify_idle_resources(plan, resources)
        if (C / 'STOP').exists(): raise ValueError('Existing stop requires inspection; never clear it automatically')
        probe = subprocess.run(['launchctl', 'print', f'gui/{os.getuid()}/com.gearshift.coding-pilot.watchdog'], capture_output=True, text=True)
        if probe.returncode or 'state = running' not in probe.stdout: raise ValueError('Independent watchdog must be running')
        actual_quote = quote()
        controller_heartbeat(plan['run_id'], 'validating')
        tick()
        reservation = check_dispatch_budget(plan, decision(json.loads((C / 'ledger.json').read_text())))
        validate_plan(plan, ROOT, sha(APPROVAL))
        run_control.mkdir(parents=True, exist_ok=False)
        write(run_control / 'plan.json', plan)
        write(run_control / 'dispatch.json', {'epoch': time.time(), 'stage_identity': identity, 'quote': actual_quote, **reservation})
        names = [name for worker in plan['workers'] for name in allocation_names(plan, worker)]
        write(C / 'allocation_intents' / (plan['run_id'] + '.json'), {'controller_id': plan['run_id'],
            'started_epoch': time.time(), 'resource_names': names + [name + '-storage' for name in names],
            'gpu_type': 'NVIDIA H200', 'gpu_count': 1})
        stop_event = threading.Event(); finished = threading.Event()
        def pulse():
            while not finished.is_set():
                controller_heartbeat(plan['run_id'], 'running'); finished.wait(10)
        heartbeat = threading.Thread(target=pulse, daemon=True); heartbeat.start()
        results, errors = [], []
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan['workers'])) as pool:
                futures = [pool.submit(launch_worker, plan, worker, identity, stop_event) for worker in plan['workers']]
                for future in concurrent.futures.as_completed(futures):
                    try: results.append(future.result())
                    except BaseException as exc: errors.append(str(exc))
        finally:
            finished.set(); heartbeat.join(timeout=15)
            write(run_control / 'complete.json', {'epoch': time.time(), 'stage_identity': identity,
                'passed': not errors and len(results) == len(plan['workers']), 'workers': results, 'errors': errors})
            tick()
        if errors: raise RuntimeError('Bounded stage incomplete: ' + '; '.join(errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--pod-bootstrap')
    args = parser.parse_args()
    if args.pod_bootstrap: bootstrap(args.pod_bootstrap)
    elif args.plan: run(args.plan.resolve(), args.validate_only)
    else: parser.error('--plan or --pod-bootstrap required')
