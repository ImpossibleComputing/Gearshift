#!/usr/bin/env python3
"""Detached secondary-only workers with bounded retries and verified idle release."""
import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_confirmation_lease import (
    atomic_json, control_root, linux_process_identity, load_lease, sha256_file, verify_release_receipt)
from scripts.coding_confirmation_replication_supervisor import (
    child_environment, copy_verified, file_receipt, scoped, verify_archive)
from scripts.coding_confirmation_generation_supervisor import closed_files as primary_closed_files, validate_guard
from scripts import coding_confirmation_secondary_generate as secondary


def read(path): return json.loads(Path(path).read_text())


def closed_files(root, folder, pod_id):
    root = Path(root)
    files = set(primary_closed_files(root, folder, pod_id))
    # Reuse receipts and these stopped workers are absent from the primary scanner.
    files.update((root / 'secondary/tasks').glob('*/control_reuse.json'))
    for base in (root / 'workers').glob(pod_id + '_secondary_*'):
        files.update(p for p in base.rglob('*') if p.is_file() and not p.name.endswith('.lock'))
    return sorted(files)


def preserve_partial_jobs(root, folder, pod_id):
    """Copy only this allocation's unfinished secondary task while holding its lock."""
    root, folder = Path(root), Path(folder)
    for owner_path in sorted((root / 'secondary/claims').glob('*.owner.json')):
        job = owner_path.name[:-len('.owner.json')]
        with owner_path.with_name(job + '.lock').open('a') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: continue
            owner = read(owner_path)
            if not owner.get('attempt_id', '').startswith(pod_id + '_secondary_'): continue
            if not job.startswith('receiver_'): raise ValueError('Unknown secondary claim')
            if (root / 'secondary/jobs' / job / 'complete.json').exists(): continue
            task = job[len('receiver_'):]
            if not task or Path(task).name != task: raise ValueError('Unsafe secondary task claim')
            base = root / 'secondary/tasks' / task
            selected = [owner_path, *(p for p in base.rglob('*') if p.is_file())]
            for source in selected:
                if source.name.endswith('.lock'): continue
                row = file_receipt(root, source)
                copy_verified(source, folder / 'partial_snapshots' / row['path'], row)


def release(root, lease, folder, outcome):
    root = Path(root); control = control_root(lease)
    preserve_partial_jobs(root, folder, lease['pod_id'])
    rows = [file_receipt(root, p) for p in closed_files(root, folder, lease['pod_id'])]
    if not rows: raise ValueError('No stopped secondary evidence to preserve')
    backup = control / 'release_backups' / str(time.time_ns()); backup.mkdir(parents=True)
    first, second = backup / 'compact_a.tar.gz', backup / 'compact_b.tar.gz'
    with tarfile.open(first, 'w:gz', compresslevel=1) as archive:
        for row in rows: archive.add(root / row['path'], arcname=row['path'], recursive=False)
        payload = json.dumps(rows, sort_keys=True).encode()
        info = tarfile.TarInfo('COMPACT_MANIFEST.json'); info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with first.open('rb') as source: os.fsync(source.fileno())
    verify_archive(first, rows)
    copy_verified(first, second, file_receipt(root, first)); verify_archive(second, rows)
    rows += [file_receipt(root, first), file_receipt(root, second)]
    manifest = control / 'release_manifest.json'
    atomic_json(manifest, {'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id'],
        'cohort': 'secondary', 'outcome': outcome, 'files': rows,
        'network_volume_deletion_authorized': False,
        'scope': 'Immutable commits and this stopped allocation; other active samplers remain protected on shared volume.'})
    atomic_json(control / 'gpu_release_verified.json', {'experiment_id': lease['experiment_id'],
        'pod_id': lease['pod_id'], 'outcome': outcome, 'gpu_release_verified': True,
        'all_gpu_workers_stopped': True, 'durable_backup_verified': True,
        'manifest_path': str(manifest.relative_to(root)), 'manifest_sha256': sha256_file(manifest)})
    if not verify_release_receipt(lease): raise ValueError('Secondary release evidence did not verify')


def run(plan_path):
    plan_path = scoped(ROOT, plan_path); plan = read(plan_path)
    lease = load_lease(scoped(ROOT, plan['lease_path']), plan['lease_sha256'])
    root = scoped(ROOT, plan['result_root']); control = control_root(lease)
    if plan['experiment_id'] != lease['experiment_id'] or str(root.resolve()) != lease['allowed_result_root']:
        raise ValueError('Secondary allocation and dispatch plan differ')
    if os.getpid() != os.getpgrp(): raise ValueError('Secondary supervisor requires a detached process group')
    folder = control / 'secondary_supervisor'; folder.mkdir(parents=True)
    validate_guard(lease, control)
    atomic_json(control / 'lease_guard/supervisor_registration.json', {
        **linux_process_identity(os.getpid()), 'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id']})
    children = {}; attempts = {}; stop = False; ownership_verified = True; outcome = 'ready_secondary_jobs_drained'

    def stopping(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping); signal.signal(signal.SIGINT, stopping)

    def start(gpu):
        nonlocal ownership_verified
        attempt = attempts.get(gpu, 0) + 1
        if attempt > 3: raise RuntimeError('Bounded secondary restart allowance exhausted')
        attempts[gpu] = attempt
        path = folder / f'gpu_{gpu}_attempt_{attempt}'; path.mkdir()
        command = [str(ROOT / '.pilot-venv/bin/python'), 'scripts/coding_confirmation_secondary_generate.py',
            'run', '--plan', str(plan_path.relative_to(ROOT)), '--worker-id', lease['pod_id'] + '_secondary_' + str(gpu)]
        with (path / 'worker.log').open('ab') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=child_environment(lease, gpu),
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=False)
            children[gpu] = (proc, path)
        try: group = os.getpgid(proc.pid)
        except ProcessLookupError:
            if proc.poll() is None:
                ownership_verified = False; raise RuntimeError('Live secondary process group could not be verified')
        except OSError:
            ownership_verified = False; raise
        else:
            if group != os.getpgrp():
                ownership_verified = False; raise RuntimeError('Secondary worker escaped supervisor process group')
        atomic_json(path / 'started.json', {'pid': proc.pid, 'gpu': gpu, 'attempt': attempt, 'epoch': time.time()})

    try:
        secondary.validate_secondary(ROOT, plan, verify_weights=True)
        if type(lease['gpu_count']) is not int or lease['gpu_count'] < 1: raise ValueError('Secondary GPU allocation required')
        for gpu in range(lease['gpu_count']): start(gpu)
        while children:
            if stop or time.time() >= lease['deadline_epoch'] - 120 or (control / 'lease_guard/STOP').exists():
                raise TimeoutError('Reserved secondary allocation deadline or stop')
            for gpu, (proc, path) in list(children.items()):
                code = proc.poll()
                if code is None: continue
                atomic_json(path / 'exited.json', {'returncode': code, 'epoch': time.time()})
                del children[gpu]
                if code:
                    if code in (64, 65, 66): raise RuntimeError('Secondary scientific identity/control failure; retry refused')
                    start(gpu)
            time.sleep(2)
    except BaseException as exc:
        outcome = 'failed_preserved'
        atomic_json(folder / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
    finally:
        for proc, _ in children.values():
            if proc.poll() is None: proc.terminate()
        deadline = time.monotonic() + 90
        for proc, _ in children.values():
            try: proc.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)
        atomic_json(folder / 'workers_stopped.json', {'outcome': outcome, 'epoch': time.time(),
            'all_registered_workers_stopped': True, 'process_group_ownership_verified': ownership_verified})
        if ownership_verified: release(root, lease, folder, outcome)
        else: atomic_json(folder / 'release_withheld.json', {
            'reason': 'Secondary worker group ownership could not be verified', 'immutable_lease_guard_retains_authority': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True)
    run(parser.parse_args().plan)
