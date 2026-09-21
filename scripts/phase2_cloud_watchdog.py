#!/usr/bin/env python3
"""Independent Runpod guard. No model service, laptop, or third-party dependency."""
import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'evidence/phase2/studio_transfer/cloud'
PREFIX = 'gearshift-phase2-20260914-'
START = dt.datetime.fromisoformat('2026-09-14T08:51:00+00:00').timestamp()
sys.path.insert(0, str(ROOT / 'scripts'))
from phase2_execution_authorization import DEADLINE, SHUTDOWN_UTC, AUTHORIZATION_END_UTC
POLICY = dict(prefix=PREFIX, authorization_start_utc='2026-09-14T08:51:00Z',
    shutdown_deadline_utc=SHUTDOWN_UTC, authorization_end_utc=AUTHORIZATION_END_UTC,
    hard_cap_usd=1000, shutdown_threshold_usd=900, max_allocated_hourly_usd=20,
    storage_reserve_hourly_per_pod_usd=0.25, poll_seconds=15, api_timeout_seconds=20,
    cost_method='Conservative upper bound: highest observed allocation rate plus storage reserve, charged continuously from authorization start until absence confirmed. Actual billing reconciled separately. No top-ups authorized.')


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def evaluate(ledger, pods, now, list_ok=True):
    """Pure decision function; all prefixed resources are discovered, including stopped pods."""
    ledger = json.loads(json.dumps(ledger))
    known = ledger.setdefault('pods', {})
    active = set()
    unsafe = False
    if list_ok:
        for pod in pods:
            if not str(pod.get('name', '')).startswith(PREFIX):
                continue
            pid = pod['id']; active.add(pid)
            item = known.setdefault(pid, dict(id=pid, name=pod['name'], first_seen=now,
                charged_from=START, rate_usd_hour=0, terminated_confirmed_at=None))
            item['terminated_confirmed_at'] = None
            try:
                rate = float(pod['costPerHr'])
                if not math.isfinite(rate) or rate <= 0: raise ValueError('invalid rate')
            except (KeyError, TypeError, ValueError):
                unsafe = True; rate = 20.0
            item['rate_usd_hour'] = max(item['rate_usd_hour'], rate + 0.25)
            item['last_seen'] = now
            item['desired_status'] = pod.get('desiredStatus')
        for pid, item in known.items():
            if pid not in active and item['terminated_confirmed_at'] is None:
                item['terminated_confirmed_at'] = now
    else:
        active = {pid for pid, x in known.items() if x['terminated_confirmed_at'] is None}
    cost = sum(max(0, (x['terminated_confirmed_at'] or now) - x['charged_from']) / 3600 * x['rate_usd_hour'] for x in known.values())
    rate = sum(known[p]['rate_usd_hour'] for p in active)
    reasons = []
    if now >= DEADLINE: reasons.append('deadline')
    if cost + rate * 120 / 3600 >= 900: reasons.append('budget_with_two_minute_margin')
    if rate > 20: reasons.append('allocated_rate')
    if unsafe: reasons.append('unknown_rate')
    if not list_ok: reasons.append('provider_listing_failure')
    ledger.update(updated_epoch=now, estimated_upper_bound_usd=cost, allocated_upper_bound_usd_hour=rate)
    return ledger, sorted(active) if reasons else [], reasons


def cli(*args):
    result = subprocess.run(['/opt/homebrew/bin/runpodctl', *args, '--output', 'json'],
        text=True, capture_output=True, timeout=20)
    if result.returncode: raise RuntimeError('Runpod command failed: ' + ' '.join(args[:2]))
    return json.loads(result.stdout or 'null')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--once', action='store_true'); args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE / 'watchdog.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy_file = STATE / 'policy.json'
    if policy_file.exists() and json.loads(policy_file.read_text()) != POLICY:
        raise RuntimeError('Cloud policy changed; refusing silent replacement')
    write(policy_file, POLICY)
    if os.uname().sysname == 'Darwin': subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    while True:
        now = time.time(); error = None
        try:
            pods = cli('pod', 'list', '--all')
            if not isinstance(pods, list): raise ValueError('Invalid pod listing')
            list_ok = True
        except Exception as exc:
            pods = []; list_ok = False; error = type(exc).__name__
        corrupt = False
        try:
            ledger = json.loads((STATE / 'ledger.json').read_text()) if (STATE / 'ledger.json').exists() else {'pods': {}}
            ledger, targets, reasons = evaluate(ledger, pods, now, list_ok)
        except Exception as exc:
            corrupt = True; error = 'ledger_' + type(exc).__name__
            ledger, targets, reasons = evaluate({'pods': {}}, pods, now, list_ok)
            targets = [p['id'] for p in pods if str(p.get('name', '')).startswith(PREFIX)]
            reasons.append('ledger_corrupt')
        if (STATE / 'STOP_CLOUD.json').exists() or (STATE.parent / 'CONTROL_DONE.json').exists():
            reasons.append('explicit_shutdown')
            targets = [pid for pid, x in ledger['pods'].items() if x['terminated_confirmed_at'] is None]
        if corrupt and (STATE / 'ledger.json').exists():
            (STATE / 'ledger.json').rename(STATE / f'ledger_corrupt_{time.time_ns()}.json')
        write(STATE / 'ledger.json', ledger)  # Persist discovered IDs BEFORE any cleanup call.
        actions = []
        for pid in targets:
            try:
                cli('pod', 'delete', pid)
                actions.append(dict(pod_id=pid, action='delete_requested', ok=True))
            except Exception as exc:
                actions.append(dict(pod_id=pid, action='delete_requested', ok=False, error=type(exc).__name__))
        if actions or reasons:
            with (STATE / 'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=now, reasons=reasons, actions=actions)) + '\n'); f.flush(); os.fsync(f.fileno())
        status = dict(pid=os.getpid(), ppid=os.getppid(), host=os.uname().nodename, heartbeat_epoch=time.time(),
            state='shutdown_enforcing' if reasons else 'healthy', reasons=reasons, provider_list_ok=list_ok, error=error,
            task_pod_ids=[p for p, x in ledger['pods'].items() if x['terminated_confirmed_at'] is None],
            estimated_upper_bound_usd=ledger['estimated_upper_bound_usd'],
            allocated_upper_bound_usd_hour=ledger['allocated_upper_bound_usd_hour'], actions=actions,
            shutdown_deadline_utc=POLICY['shutdown_deadline_utc'])
        write(STATE / 'watchdog_status.json', status)
        if args.once: break
        time.sleep(15)


if __name__ == '__main__': main()
