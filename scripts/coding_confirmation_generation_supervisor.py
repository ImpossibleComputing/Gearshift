#!/usr/bin/env python3
"""Detached public-generation workers and verified idle GPU release."""
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


def read(path): return json.loads(Path(path).read_text())


def closed_files(root, control, pod_id):
    """Only immutable commits or this stopped allocation's files may be copied."""
    root, control = Path(root), Path(control)
    files = set()
    def sampler(folder, kind):
        receipt = read(folder / 'complete.json')
        record = folder / ('source_history.json' if kind == 'reasoning' else 'answer_record.json')
        if receipt['record_file_sha256'] != sha256_file(record):
            raise ValueError('Completed sampler bytes changed before backup')
        timing = read(folder / 'completion_timing.json')
        if (timing.get('identity_sha256') != receipt['identity_sha256'] or
                timing.get('record_sha256') != receipt['record_sha256']):
            raise ValueError('Completed sampler timing identity differs before backup')
        files.update(folder / n for n in ['identity.json', 'resume.json', 'complete.json',
                                        'completion_timing.json', record.name])
    for cohort in ('primary', 'secondary'):
        base = root / cohort
        for path in base.rglob('draw_complete.json'):
            row = read(path); folder = path.parent
            if (row['answer_sha256'] != sha256_file(folder / 'answer.json') or
                    row['sampler_complete_sha256'] != sha256_file(folder / 'sampler/complete.json')):
                raise ValueError('Committed draw changed before backup')
            files.update([path, folder / 'answer.json', folder / 'draw_identity.json'])
            sampler(folder / 'sampler', 'answer')
        for path in base.rglob('history_ready.json'):
            if read(path)['sha256'] != sha256_file(path.parent / 'source_history.json'):
                raise ValueError('Committed history changed before backup')
            files.add(path); sampler(path.parent, 'reasoning')
        for path in base.glob('jobs/*/complete.json'):
            row = read(path)
            for key in ('runtime_path', 'setup_path'):
                files.add(scoped(root, row[key]))
        for pattern in ['jobs/*/complete.json', 'tasks/*/controls/*.json', 'tasks/*/mapper_loads/*.json',
                        'tasks/*/prompt_only_template.json', 'generation_closure.json']:
            files.update(base.glob(pattern))
    for base in [control, *sorted((root / 'workers').glob(pod_id + '_generation_*'))]:
        for path in base.rglob('*'):
            if path.is_file() and not path.name.endswith('.lock'):
                files.add(path)
    return sorted(files)


def preserve_partial_jobs(root, folder, pod_id):
    """Snapshot this stopped allocation's unfinished work while holding its lock.

    Another pod may already have reclaimed a job. Never snapshot that pod's
    mutable sampler files; they remain durable on the protected shared volume.
    """
    root = Path(root); folder = Path(folder)
    for owner_path in sorted((root / 'claims').glob('*.owner.json')):
        job_id = owner_path.name[:-len('.owner.json')]
        lock_path = owner_path.with_name(job_id + '.lock')
        with lock_path.open('a') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: continue
            owner = read(owner_path)
            if not owner.get('attempt_id', '').startswith(pod_id + '_generation_'): continue
            kind, task = job_id.split('_', 1)
            if kind not in ('source', 'small', 'receiver'): raise ValueError('Unknown claimed generation job')
            complete = root / 'primary/jobs' / job_id / 'complete.json'
            if complete.exists(): continue
            names = {'source': ['large_history', 'A'], 'small': ['small_history', 'B'],
                     'receiver': ['D', 'P', 'FIXED_M', 'ROTATING_M', 'FIXED_H', 'ROTATING_H',
                                  'controls', 'prompt_only_template.json']}[kind]
            selected = [owner_path]
            for name in names:
                scope = root / 'primary/tasks' / task / name
                if scope.is_dir(): selected.extend(p for p in scope.rglob('*') if p.is_file())
                elif scope.is_file(): selected.append(scope)
            for source in selected:
                if source.name.endswith('.lock'): continue
                row = file_receipt(root, source)
                target = folder / 'partial_snapshots' / row['path']
                copy_verified(source, target, row)


def validate_guard(lease, control):
    status = read(Path(control) / 'lease_guard/status.json')
    if (status['state'] != 'armed' or status['pod_id'] != lease['pod_id'] or
            status['experiment_id'] != lease['experiment_id'] or
            status['deadline_epoch'] != lease['deadline_epoch']):
        raise ValueError('Independent guard is not armed for this allocation')
    os.kill(status['pid'], 0)
    pod = os.environ.get('RUNPOD_POD_ID')
    if not pod:
        pod = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x).get(b'RUNPOD_POD_ID', b'').decode()
    if pod != lease['pod_id']: raise ValueError('Supervisor is on a different pod')
    return status


def release(root, lease, folder, outcome):
    root = Path(root); control = control_root(lease)
    preserve_partial_jobs(root, folder, lease['pod_id'])
    rows = [file_receipt(root, p) for p in closed_files(root, folder, lease['pod_id'])]
    if not rows: raise ValueError('No stopped worker evidence to preserve')
    backup = control / 'release_backups' / str(time.time_ns()); backup.mkdir(parents=True)
    first, second = backup / 'compact_a.tar.gz', backup / 'compact_b.tar.gz'
    with tarfile.open(first, 'w:gz', compresslevel=1) as archive:
        for row in rows: archive.add(root / row['path'], arcname=row['path'], recursive=False)
        data = json.dumps(rows, sort_keys=True).encode()
        info = tarfile.TarInfo('COMPACT_MANIFEST.json'); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with first.open('rb') as source: os.fsync(source.fileno())
    verify_archive(first, rows)
    copy_verified(first, second, file_receipt(root, first)); verify_archive(second, rows)
    rows += [file_receipt(root, first), file_receipt(root, second)]
    manifest_path = control / 'release_manifest.json'
    atomic_json(manifest_path, {'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id'],
        'outcome': outcome, 'files': rows, 'network_volume_deletion_authorized': False,
        'scope': 'Immutable committed science plus this stopped allocation; active sampler progress remains on protected volume.'})
    atomic_json(control / 'gpu_release_verified.json', {
        'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id'], 'outcome': outcome,
        'gpu_release_verified': True, 'all_gpu_workers_stopped': True, 'durable_backup_verified': True,
        'manifest_path': str(manifest_path.relative_to(root)), 'manifest_sha256': sha256_file(manifest_path)})
    if not verify_release_receipt(lease): raise ValueError('Release evidence did not verify')


def run(plan_path):
    plan_path = scoped(ROOT, plan_path); plan = read(plan_path)
    lease = load_lease(scoped(ROOT, plan['lease_path']), plan['lease_sha256'])
    root = scoped(ROOT, plan['result_root']); control = control_root(lease)
    if plan['experiment_id'] != lease['experiment_id'] or str(root.resolve()) != lease['allowed_result_root']:
        raise ValueError('Allocation and generation plan differ')
    if os.getpid() != os.getpgrp(): raise ValueError('Supervisor requires its own detached process group')
    folder = control / 'generation_supervisor'; folder.mkdir(parents=True)
    validate_guard(lease, control)
    atomic_json(control / 'lease_guard/supervisor_registration.json', {
        **linux_process_identity(os.getpid()), 'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id']})
    stop = False; children = {}; attempts = {}; outcome = 'ready_jobs_drained'
    ownership_verified = True

    def stopping(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping); signal.signal(signal.SIGINT, stopping)

    def start(gpu):
        nonlocal ownership_verified
        attempt = attempts.get(gpu, 0) + 1
        if attempt > 3: raise RuntimeError('Bounded generation restart allowance exhausted')
        attempts[gpu] = attempt
        path = folder / f'gpu_{gpu}_attempt_{attempt}'; path.mkdir()
        worker_id = lease['pod_id'] + '_generation_' + str(gpu)
        command = [str(ROOT / '.pilot-venv/bin/python'), 'scripts/coding_confirmation_generate.py',
            '--plan', str(plan_path.relative_to(ROOT)), '--worker-id', worker_id,
            '--preference', plan['preferences'][gpu]]
        with (path / 'worker.log').open('ab') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=child_environment(lease, gpu),
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=False)
            children[gpu] = (proc, path)
        # Track every spawned process before any fallible inspection or receipt
        # write. A startup failure must never leave a live, unowned GPU worker
        # outside the shutdown path and then authorize pod deletion.
        try:
            group = os.getpgid(proc.pid)
        except ProcessLookupError:
            if proc.poll() is None:
                ownership_verified = False
                raise RuntimeError('Live worker process group could not be verified')
        except OSError:
            ownership_verified = False
            raise
        else:
            if group != os.getpgrp():
                ownership_verified = False
                raise RuntimeError('Worker escaped supervisor process group')
        atomic_json(path / 'started.json', {'pid': proc.pid, 'gpu': gpu, 'attempt': attempt, 'epoch': time.time()})

    try:
        if len(plan['preferences']) != lease['gpu_count'] or not all(p in ('source', 'small', 'receiver') for p in plan['preferences']):
            raise ValueError('Worker preferences do not match the reserved GPUs')
        for gpu in range(lease['gpu_count']): start(gpu)
        while children:
            if stop or time.time() >= lease['deadline_epoch'] - 120 or (control / 'lease_guard/STOP').exists():
                raise TimeoutError('Reserved local allocation deadline or stop')
            for gpu, (proc, path) in list(children.items()):
                code = proc.poll()
                if code is None: continue
                atomic_json(path / 'exited.json', {'returncode': code, 'epoch': time.time()})
                del children[gpu]
                if code:
                    if code in (64, 65, 66): raise RuntimeError('Scientific identity/control failure; automatic retry refused')
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
            try: proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)
        atomic_json(folder / 'workers_stopped.json', {'outcome': outcome, 'epoch': time.time(),
            'all_registered_workers_stopped': True, 'process_group_ownership_verified': ownership_verified})
        if ownership_verified:
            release(root, lease, folder, outcome)
        else:
            # The direct child was stopped, but escaped descendants cannot be
            # proven stopped. Preserve the volume and leave deletion to the
            # independent immutable lease deadline, never a false idle receipt.
            atomic_json(folder / 'release_withheld.json', {
                'reason': 'Worker process-group ownership could not be verified',
                'immutable_lease_guard_retains_authority': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True)
    args = parser.parse_args(); run(args.plan)
