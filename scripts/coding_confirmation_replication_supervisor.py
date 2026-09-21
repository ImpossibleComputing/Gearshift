#!/usr/bin/env python3
"""Detached paired replication, bounded resume retries and verified GPU release.

Only the independently armed confirmation lease guard owns provider credentials
and pod deletion. This process needs no network or controller heartbeat.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from gearshift.coding_control import digest
from gearshift.coding_confirmation_lease import (
    atomic_json, control_root, linux_process_identity, load_lease, sha256_file,
    verify_release_receipt)
from scripts.coding_confirmation_replication_stage import verify_manifest
from scripts.coding_confirmation_replication import (
    ARMS, CHECKPOINTS, DECLARATION, EXPERIMENT, RESULT, make_checkpoint_identity)


def read(path):
    return json.loads(Path(path).read_text())


def scoped(repo, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected scoped repository-relative path')
    result = Path(repo) / path
    if result.is_symlink(): raise ValueError('Linked plan input')
    result.resolve().relative_to(Path(repo).resolve())
    return result


def child_environment(lease, gpu):
    allowed = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'LIBRARY_PATH',
               'CUDA_HOME', 'CUDA_PATH', 'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES')
    env = {k: os.environ[k] for k in allowed if k in os.environ}
    env.update(RUNPOD_POD_ID=lease['pod_id'], CUDA_VISIBLE_DEVICES=str(gpu),
               HF_HOME='/workspace/hf', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_DISABLE_TELEMETRY='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONUNBUFFERED='1')
    return env


def checkpoint_files(folder, identity):
    """Validate committed metadata and full byte hashes without loading GPU code."""
    folder = Path(folder)
    if folder.is_symlink() or folder.name.startswith('.'):
        raise ValueError('Partial or linked checkpoint')
    manifest = read(folder / 'manifest.json')
    if (manifest.get('checkpoint_identity') != identity
        or manifest.get('checkpoint_identity_sha256') != digest(identity)
        or manifest.get('arm') != identity['arm']
        or manifest.get('complete_resumable') is not True
        or manifest.get('verified_roundtrip') is not True
        or manifest.get('step') != manifest.get('schedule_position')
        or folder.name != f"step_{manifest.get('step', -1):04d}"
        or manifest.get('step') not in CHECKPOINTS
        or set(manifest.get('files', {})) != {'mapper.pt', 'full.pt'}):
        raise ValueError('Checkpoint identity, schedule or completeness differs')
    required = ['mapper', 'optimizer', 'scheduler', 'update', 'schedule_position',
                'python_rng', 'numpy_rng', 'torch_cpu_rng', 'torch_cuda_rng',
                'scaler_explicitly_none_for_bf16', 'checkpoint_identity', 'training_log']
    if manifest.get('state_components') != required:
        raise ValueError('Checkpoint lacks complete resumable state')
    for name, receipt in manifest['files'].items():
        path = folder / name
        if path.is_symlink() or path.stat().st_size != receipt['bytes'] or sha256_file(path) != receipt['sha256']:
            raise ValueError('Checkpoint bytes changed')
    return [folder / 'manifest.json', folder / 'full.pt', folder / 'mapper.pt']


def prepare_restart(arm_root, identity):
    """Restart only the latest committed state; retain an interrupted preflight."""
    arm_root = Path(arm_root)
    existing = sorted((arm_root / 'checkpoints').glob('step_*'))
    if existing:
        checkpoint_files(existing[-1], identity)
        return existing[-1]
    preserved = arm_root / 'attempts' / ('preflight_interrupted_' + str(time.time_ns()))
    names = ['resume_preflight', 'resume_preflight.json', 'pristine_restore.json', 'optimizer_reset.json']
    for name in names:
        old = arm_root / name
        if old.exists():
            preserved.mkdir(parents=True, exist_ok=True)
            old.rename(preserved / name)
    return None


def file_receipt(root, path):
    path = Path(path)
    if path.is_symlink(): raise ValueError('Cannot preserve linked evidence')
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def copy_verified(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    with source.open('rb') as src, temporary.open('xb') as out:
        shutil.copyfileobj(src, out, 8 * 1024 * 1024); out.flush(); os.fsync(out.fileno())
    if temporary.stat().st_size != expected['bytes'] or sha256_file(temporary) != expected['sha256']:
        raise ValueError('Durable backup checksum differs')
    os.rename(temporary, destination)
    directory = os.open(destination.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def verify_archive(path, expected):
    rows = {x['path']: x for x in expected}
    with tarfile.open(path, 'r:gz') as archive:
        names = archive.getnames()
        if len(names) != len(set(names)) or set(names) != set(rows) | {'COMPACT_MANIFEST.json'}:
            raise ValueError('Compact backup member set differs')
        for member in archive.getmembers():
            if not member.isfile(): raise ValueError('Compact backup contains a non-file')
            raw = archive.extractfile(member).read()
            if member.name == 'COMPACT_MANIFEST.json':
                if json.loads(raw) != expected: raise ValueError('Compact backup manifest differs')
            elif len(raw) != rows[member.name]['bytes'] or hashlib.sha256(raw).hexdigest() != rows[member.name]['sha256']:
                raise ValueError('Compact backup content differs')


def durable_release(root, lease, identities, outcome, supervisor_folder):
    root = Path(root); control = control_root(lease)
    # Other confirmation pods may still be writing. Only this pair's closed
    # training roots, own worker attempts and closed supervisor logs are scanned.
    scopes = [root / 'replication', Path(supervisor_folder) / 'attempts']
    scopes.extend(root / 'workers' / (lease['pod_id'] + '_replication_' + arm) for arm in ARMS)
    for arm in ARMS:
        arm_root = root / 'replication' / 'arms' / arm
        folders = sorted((arm_root / 'checkpoints').glob('step_*'))
        if outcome == 'complete':
            if {p.name for p in folders} != {f'step_{s:04d}' for s in CHECKPOINTS}:
                raise ValueError('Both arms must retain every declared checkpoint')
            complete = read(arm_root / 'training_complete.json')
            if complete.get('full_target_completed') is not True or complete.get('completed_updates') != 1024:
                raise ValueError('Training stopped before the frozen endpoint')
        for folder in folders:
            try: checkpoint_files(folder, identities[arm])
            except Exception:
                if outcome == 'complete': raise
                # Preserve corrupt/incomplete failure evidence as actual bytes;
                # it cannot be advertised as a valid resumable checkpoint.
                atomic_json(Path(supervisor_folder) / ('failure_checkpoint_' + str(time.time_ns()) + '.json'),
                    {'arm': arm, 'path': str(folder.relative_to(root)), 'checkpoint_valid': False,
                     'outcome': 'failed_preserved', 'traceback': traceback.format_exc()})
    files = []
    for scope in scopes:
        for path in sorted(scope.rglob('*')):
            if path.is_symlink(): raise ValueError('Linked training output cannot authorize release')
            if path.is_file() and not path.name.endswith('.lock'):
                files.append(path)
    # A failed pre-initialization launch still has a closed failure record.
    files.extend(sorted(Path(supervisor_folder).glob('failure_*.json')))
    files = sorted(set(files))
    records = [file_receipt(root, path) for path in files]
    if not records: raise ValueError('No closed durable evidence to preserve')
    backup = control / 'release_backups' / str(time.time_ns()); backup.mkdir(parents=True)
    heavy_copies = []
    for row in records:
        if Path(row['path']).suffix == '.pt':
            target = backup / 'full_states' / row['path']
            copy_verified(root / row['path'], target, row)
            heavy_copies.append(file_receipt(root, target))
    compact = [x for x in records if Path(x['path']).suffix != '.pt' and x['bytes'] <= 32 * 1024 * 1024]
    first = backup / 'compact_a.tar.gz'; second = backup / 'compact_b.tar.gz'
    with tarfile.open(first, 'w:gz', compresslevel=1) as archive:
        for row in compact: archive.add(root / row['path'], arcname=row['path'], recursive=False)
        raw = json.dumps(compact, sort_keys=True, allow_nan=False).encode()
        info = tarfile.TarInfo('COMPACT_MANIFEST.json'); info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    with first.open('rb') as stream: os.fsync(stream.fileno())
    verify_archive(first, compact)
    copy_verified(first, second, file_receipt(root, first)); verify_archive(second, compact)
    records.extend(heavy_copies + [file_receipt(root, first), file_receipt(root, second)])
    manifest = {'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id'],
                'outcome': outcome, 'files': records, 'full_checkpoint_copies_verified': True,
                'compact_archive_copies_verified': 2, 'off_pod_backup_verified': False,
                'network_volume_deletion_authorized': False}
    manifest_path = control / 'release_manifest.json'; atomic_json(manifest_path, manifest)
    receipt = {'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id'], 'outcome': outcome,
               'gpu_release_verified': True, 'all_gpu_workers_stopped': True,
               'durable_backup_verified': True, 'manifest_path': str(manifest_path.relative_to(root)),
               'manifest_sha256': sha256_file(manifest_path), 'network_volume_deletion_authorized': False}
    atomic_json(control / 'gpu_release_verified.json', receipt)
    if not verify_release_receipt(lease): raise ValueError('New confirmation release verification failed')
    return receipt


class Supervisor:
    def __init__(self, repo, plan_path):
        self.repo = Path(repo); self.plan_path = scoped(repo, plan_path); self.plan = read(self.plan_path)
        self.lease = load_lease(scoped(repo, self.plan['lease_path']), self.plan['lease_sha256'])
        self.root = scoped(repo, self.plan['result_root'])
        if self.plan['experiment_id'] != EXPERIMENT or self.lease['experiment_id'] != EXPERIMENT:
            raise ValueError('Wrong replication experiment')
        if self.root.resolve() != Path(self.lease['allowed_result_root']).resolve() or self.lease['gpu_count'] != 2:
            raise ValueError('Lease root or paired GPU count differs')
        if self.root / 'replication' != self.repo / RESULT:
            raise ValueError('Wrong replication namespace')
        self.control = control_root(self.lease); self.folder = self.control / 'replication_supervisor'
        self.folder.mkdir(parents=True, exist_ok=True)
        d = read(self.repo / DECLARATION)
        self.identities = {a: make_checkpoint_identity(d, self.plan['code_commit'], a) for a in ARMS}
        self.children = {}; self.attempts = {a: 0 for a in ARMS}; self.complete = set(); self.stopping = False

    def event(self, state, **details):
        row = {'epoch': time.time(), 'state': state, 'pod_id': self.lease['pod_id'], **details}
        atomic_json(self.folder / 'status.json', row)
        with (self.folder / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n'); stream.flush(); os.fsync(stream.fileno())

    def guard(self):
        if self.stopping or time.time() >= self.lease['deadline_epoch'] - 120 or (self.control / 'lease_guard/STOP').exists():
            raise TimeoutError('Local reserved allocation deadline or stop')

    def start(self, arm):
        self.guard()
        if arm in self.children: raise ValueError('Arm already has a live worker')
        attempt = self.attempts[arm] + 1
        if attempt > 3: raise RuntimeError('Two resume restarts exhausted: ' + arm)
        root = self.root / 'replication/arms' / arm
        checkpoint = prepare_restart(root, self.identities[arm])
        folder = self.folder / 'attempts' / arm / f'attempt_{attempt:02d}'
        folder.mkdir(parents=True, exist_ok=False)
        worker_id = self.lease['pod_id'] + '_replication_' + arm
        command = [str(self.repo / '.pilot-venv/bin/python'), 'scripts/coding_confirmation_replication_worker.py',
                   '--plan', str(self.plan_path.relative_to(self.repo)), '--arm', arm, '--worker-id', worker_id]
        if checkpoint: command.extend(['--resume-checkpoint', str(checkpoint)])
        with (folder / 'worker.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=self.repo, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, env=child_environment(self.lease, ARMS.index(arm)),
                start_new_session=False)
        if os.getpgid(process.pid) != os.getpgrp():
            process.terminate(); process.wait(timeout=30); raise RuntimeError('Worker escaped registered group')
        self.attempts[arm] = attempt
        self.children[arm] = (process, folder)
        atomic_json(folder / 'started.json', {'epoch': time.time(), 'pid': process.pid,
            'pgid': os.getpgrp(), 'arm': arm, 'attempt': attempt, 'command': command,
            'checkpoint': str(checkpoint) if checkpoint else None})
        self.event('worker_started', arm=arm, attempt=attempt, pid=process.pid)

    def poll(self):
        for arm, (process, folder) in list(self.children.items()):
            code = process.poll()
            if code is None: continue
            atomic_json(folder / 'exited.json', {'epoch': time.time(), 'returncode': code})
            del self.children[arm]; self.event('worker_exited', arm=arm, returncode=code)
            if code == 0: self.complete.add(arm)
            elif not self.stopping: self.start(arm)

    def stop_children(self):
        self.stopping = True
        for process, _ in self.children.values():
            if process.poll() is None: process.terminate()
        end = time.monotonic() + 90
        while self.children and time.monotonic() < end:
            self.poll(); time.sleep(.5)
        for process, _ in self.children.values():
            if process.poll() is None: process.kill()
            process.wait(timeout=10)
        self.poll()

    def run(self):
        if os.getpid() != os.getpgrp(): raise RuntimeError('Supervisor must own a detached process group')
        status = read(self.control / 'lease_guard/status.json')
        if any(status.get(k) != self.lease[k] for k in ('experiment_id', 'pod_id', 'deadline_epoch')) or status.get('state') != 'armed':
            raise ValueError('Independent confirmation lease guard is not armed')
        os.kill(status['pid'], 0)
        pod = os.environ.get('RUNPOD_POD_ID')
        if not pod:
            pod = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x).get(b'RUNPOD_POD_ID', b'').decode()
        if pod != self.lease['pod_id']: raise ValueError('Supervisor is on a different pod')
        atomic_json(self.control / 'lease_guard/supervisor_registration.json', {
            **linux_process_identity(os.getpid()), 'experiment_id': EXPERIMENT, 'pod_id': pod})
        def stop(signum, frame): self.stopping = True
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        outcome = 'complete'
        try:
            manifest_path = scoped(self.repo, self.plan['stage_manifest_path'])
            if sha256_file(manifest_path) != self.plan['stage_manifest_sha256']:
                raise ValueError('Staged input inventory changed')
            verify_manifest(self.repo, read(manifest_path))
            self.event('started', code_commit=self.plan['code_commit'], plan_sha256=sha256_file(self.plan_path))
            for arm in ARMS: self.start(arm)
            while self.children:
                self.guard(); self.poll(); time.sleep(2)
            if self.complete != set(ARMS): raise RuntimeError('Both replication workers must finish')
        except BaseException as error:
            outcome = 'failed_preserved'
            atomic_json(self.folder / ('failure_' + str(time.time_ns()) + '.json'), {
                'epoch': time.time(), 'exception': type(error).__name__, 'error': str(error),
                'traceback': traceback.format_exc()})
            self.event('failure_draining', error=str(error)); self.stop_children()
        if self.children: raise RuntimeError('Cannot release allocation with live children')
        self.event('verifying_durable_backup', outcome=outcome)
        receipt = durable_release(self.root, self.lease, self.identities, outcome, self.folder)
        self.event('release_authorized', outcome=outcome)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan', required=True)
    args = parser.parse_args(argv); supervisor = Supervisor(ROOT, args.plan)
    with (supervisor.folder / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor.run()


if __name__ == '__main__': main()
