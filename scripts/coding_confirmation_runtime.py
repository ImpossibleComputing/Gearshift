#!/usr/bin/env python3
"""Local durable worker context; no console heartbeat or private test dependency."""
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import digest, sha, write
from gearshift.coding_confirmation_lease import load_lease, control_root


def build_context(plan_path, role, worker_id, arm=None):
    from gearshift.coding_recovery import Telemetry
    plan_path = Path(plan_path)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    plan = json.loads(plan_path.read_text())
    lease = load_lease(ROOT / plan['lease_path'], plan['lease_sha256'])
    if lease['experiment_id'] != plan['experiment_id']:
        raise ValueError('Plan/lease experiment mismatch')
    if os.environ.get('RUNPOD_POD_ID') != lease['pod_id']:
        raise ValueError('Worker is not on its reserved pod')
    top = ROOT / plan['result_root']
    if top.resolve() != Path(lease['allowed_result_root']).resolve():
        raise ValueError('Plan/lease result root mismatch')
    if not worker_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Unsafe worker ID')
    attempt = worker_id + '_' + str(time.time_ns())
    status = top / 'workers' / worker_id / 'attempts' / attempt
    status.mkdir(parents=True, exist_ok=False)
    root = top
    if role == 'replication_train':
        if arm not in ('FIXED', 'ROTATING'):
            raise ValueError('Missing training arm')
        root = top / 'replication' / 'arms' / arm
    root.mkdir(parents=True, exist_ok=True)
    current = {'state': 'running', 'stage': 'starting', 'role': role,
               'worker_id': worker_id, 'attempt_id': attempt, 'pod_id': lease['pod_id']}

    def publish(**values):
        current.update(values)
        row = {**current, 'epoch': time.time()}
        write(status / 'status.json', row)
        write(top / 'workers' / worker_id / 'latest.json', row)

    def guard():
        if time.time() >= lease['deadline_epoch'] - 120:
            raise TimeoutError('Reserved allocation deadline')
        if (control_root(lease) / 'lease_guard' / 'STOP').exists():
            raise TimeoutError('Local allocation stop marker')

    def stop(signum, frame):
        raise TimeoutError('Controlled process termination')

    signal.signal(signal.SIGTERM, stop)
    identity = {'experiment_id': plan['experiment_id'], 'plan_sha256': sha(plan_path),
                'code_commit': plan['code_commit'], 'pod_id': lease['pod_id'],
                'role': role, 'worker_id': worker_id, 'attempt_id': attempt}
    publish()
    return {'repo_root': ROOT, 'root': root, 'top': top, 'status_root': status,
            'plan': plan, 'spec': plan.copy(), 'identity': identity,
            'code_commit': plan['code_commit'], 'attempt_id': attempt,
            'guard': guard, 'publish': publish, 'telemetry': Telemetry(status),
            'lease': lease, 'allocation_control': control_root(lease)}
