#!/usr/bin/env python3
"""Detached secondary regional workers; independent allocation and exact scientific identity."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_confirmation_lease import atomic_json, control_root, linux_process_identity, load_lease, sha256_file
from scripts.coding_confirmation_replication_supervisor import child_environment, scoped, copy_verified, file_receipt
from scripts.coding_confirmation_secondary_supervisor import release as secondary_release
from scripts.coding_confirmation_generation_supervisor import validate_guard
from scripts import coding_confirmation_secondary_sharded_generate_v2 as sharded

def read(path): return json.loads(Path(path).read_text())


def release(root, lease, folder, outcome, plan_path):
    plan = read(plan_path)
    for source in [plan_path, *(root / 'secondary/sharding').rglob('*.json')]:
        relative = source.relative_to(ROOT)
        receipt = {'bytes': source.stat().st_size, 'sha256': sha256_file(source)}
        copy_verified(source, folder / 'operational_provenance' / relative, receipt)
    partition = read(ROOT / plan['partition_path'])
    paths = [plan['partition_path'], plan['adapter_source_manifest_path'], partition['quiescence_path'],
        plan['lease_path'], plan['provider_mount_receipt_path']]
    quiescence = read(ROOT / partition['quiescence_path'])
    mount = read(ROOT / plan['provider_mount_receipt_path'])
    paths.append(mount['provider_inspection_path'])
    paths += [p['path'] for row in quiescence['allocation_receipts'] for p in row['files'].values()]
    paths += [row['lease_path'] for row in partition['original_generation_allocations']]
    paths += [plan['secondary_declaration_path'], plan['secondary_shard_source_manifest_path']]
    secondary = read(ROOT / plan['secondary_declaration_path'])
    paths += [cp['manifest_path'] for cp in secondary['checkpoints'].values()]
    for relative in sorted(set(paths)):
        source = sharded.secondary.primary.scoped_file(ROOT, relative)
        receipt = {'bytes': source.stat().st_size, 'sha256': sha256_file(source)}
        copy_verified(source, folder / 'operational_provenance' / relative, receipt)
    secondary_release(root, lease, folder, outcome)


def refuse_live_registration(control):
    path = control / 'lease_guard/supervisor_registration.json'
    if not path.exists(): return
    old = read(path)
    try: identity = linux_process_identity(old['pid'])
    except (ProcessLookupError, FileNotFoundError): return
    if all(identity.get(k) == old.get(k) for k in ('pid', 'pgid', 'start_ticks', 'boot_id')):
        raise ValueError('Allocation already has a live registered supervisor; primary/training ownership cannot be replaced')


def run(plan_path, plan_sha256):
    plan_path = scoped(ROOT, plan_path)
    if sha256_file(plan_path) != plan_sha256: raise ValueError('Pinned sharded dispatch plan differs')
    plan = read(plan_path)
    if plan.get('execution_mode') != 'generation' or plan.get('worker_role') != sharded.ROLE or plan.get('exclusive_gpu_allocation') is not True:
        raise ValueError('Supervisor requires a dedicated secondary generation dispatch')
    lease = load_lease(scoped(ROOT, plan['lease_path']), plan['lease_sha256'])
    root = scoped(ROOT, plan['result_root']); control = control_root(lease)
    if plan['experiment_id'] != lease['experiment_id'] or str(root.resolve()) != lease['allowed_result_root']:
        raise ValueError('Sharded allocation and dispatch plan differ')
    if os.getpid() != os.getpgrp(): raise ValueError('Sharded supervisor requires a detached process group')
    folder = control / 'secondary_sharded_supervisor'; folder.mkdir(parents=True)
    validate_guard(lease, control)
    refuse_live_registration(control)
    atomic_json(control / 'lease_guard/supervisor_registration.json', {
        **linux_process_identity(os.getpid()), 'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id']})
    children = {}; attempts = {}; stop = False; ownership_verified = True; outcome = 'ready_secondary_sharded_jobs_drained'

    def stopping(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping); signal.signal(signal.SIGINT, stopping)

    def start(gpu):
        nonlocal ownership_verified
        attempt = attempts.get(gpu, 0) + 1
        if attempt > 3: raise RuntimeError('Bounded sharded restart allowance exhausted')
        attempts[gpu] = attempt
        path = folder / f'gpu_{gpu}_attempt_{attempt}'; path.mkdir()
        command = [str(ROOT / '.pilot-venv/bin/python'), 'scripts/coding_confirmation_secondary_sharded_generate_v2.py',
            'run', '--plan', str(plan_path.relative_to(ROOT)), '--plan-sha256', plan_sha256,
            '--worker-id', lease['pod_id'] + '_secondary_' + str(gpu)]
        with (path / 'worker.log').open('ab') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=child_environment(lease, gpu),
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=False)
            children[gpu] = (proc, path)
        try: group = os.getpgid(proc.pid)
        except ProcessLookupError:
            if proc.poll() is None:
                ownership_verified = False; raise RuntimeError('Live sharded process group could not be verified')
        except OSError:
            ownership_verified = False; raise
        else:
            if group != os.getpgrp():
                ownership_verified = False; raise RuntimeError('Sharded worker escaped supervisor process group')
        atomic_json(path / 'started.json', {'pid': proc.pid, 'gpu': gpu, 'attempt': attempt, 'epoch': time.time()})

    try:
        public = sharded.public_context(ROOT, str(plan_path.relative_to(ROOT)), plan_sha256, verify_weights=True)
        sharded.verify_allocation(public, verify_local=True)
        sharded.verify_dataset(public)
        if type(lease['gpu_count']) is not int or lease['gpu_count'] < 1: raise ValueError('Sharded GPU allocation required')
        for gpu in range(lease['gpu_count']): start(gpu)
        while children:
            if stop or time.time() >= lease['deadline_epoch'] - 120 or (control / 'lease_guard/STOP').exists():
                raise TimeoutError('Reserved sharded allocation deadline or stop')
            for gpu, (proc, path) in list(children.items()):
                code = proc.poll()
                if code is None: continue
                atomic_json(path / 'exited.json', {'returncode': code, 'epoch': time.time()})
                del children[gpu]
                if code:
                    if code in (64, 65, 66): raise RuntimeError('Sharded scientific identity/control failure; retry refused')
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
        if ownership_verified: release(root, lease, folder, outcome, plan_path)
        else: atomic_json(folder / 'release_withheld.json', {
            'reason': 'Sharded worker group ownership could not be verified', 'immutable_lease_guard_retains_authority': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args(); run(args.plan, args.plan_sha256)
