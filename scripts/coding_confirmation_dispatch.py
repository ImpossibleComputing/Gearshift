#!/usr/bin/env python3
"""Bounded console provisioning; live work never requires this console."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_confirmation_lease import atomic_json, sha256_file, verify_lease

EXPERIMENT = 'confirmation_01_20260919T094418Z'
EVIDENCE = ROOT / 'evidence/coding_pilot_v1' / EXPERIMENT / 'resources'
REMOTE = '/workspace/GearshiftConfirmation'
RESULT = 'results/coding_pilot_v1/' + EXPERIMENT
IMAGE = 'runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'
CLI = str(Path.home() / '.local/bin/runpodctl-2.14.0')
# A conservative envelope, not a GPU-hour limit. Reconcile before enlarging it.
ENVELOPE = 20 * 12 * 5.55 + 24 + 5
BASELINE = 519.4707858258678


def key():
    return tomllib.loads((Path.home() / '.runpod/config.toml').read_text())['apikey']


def api(path, method='GET', body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request('https://api.runpod.io/v2/' + path, data=data, method=method,
        headers={'Authorization': 'Bearer ' + key(), 'User-Agent': 'runpodctl/2.14.0',
                 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
        return json.loads(raw) if raw else {}


def safe(pod):
    return {k: pod[k] for k in ('id', 'name', 'status', 'image', 'gpu', 'cpu', 'cost',
            'dataCenterId', 'mounts', 'ssh', 'createdAt', 'startedAt', 'disk') if k in pod}


def allocate(name, kind, count):
    if not name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Unsafe allocation name')
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE / 'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        folder = EVIDENCE / name
        if folder.exists():
            raise ValueError('Allocation exists; reconcile instead of duplicating')
        existing = list(EVIDENCE.glob('*/lease.json'))
        leases = [json.loads(p.read_text()) for p in existing]
        if kind == 'gpu' and sum(x['gpu_count'] for x in leases) + count > 20:
            raise ValueError('Reconcile and enlarge fleet reservation before new allocation')
        if kind == 'cpu' and any(x['gpu_count'] == 0 for x in leases):
            raise ValueError('One dedicated scoring allocation is already reserved')
        epoch = time.time()
        rate = 5.55 * count if kind == 'gpu' else 1.0
        duration = 12 if kind == 'gpu' else 24
        lease = {'experiment_id': EXPERIMENT, 'pod_id': 'reservation_pending',
            'allocation_epoch': epoch, 'deadline_epoch': epoch + duration * 3600,
            'upper_hourly_usd': rate, 'gpu_count': count if kind == 'gpu' else 0,
            'other_reserved_usd': ENVELOPE - duration * rate,
            'baseline_usd': BASELINE, 'baseline_gpu_hours': 91.24181750045884,
            'total_cap_usd': 2500, 'total_cap_gpu_hours': None,
            'cleanup_reserve_usd': 40, 'allowed_result_root': REMOTE + '/' + RESULT}
        reservation = verify_lease(lease)
        folder.mkdir()
        atomic_json(folder / 'reservation.json', {'lease': lease, 'reservation': reservation})
        if kind == 'gpu':
            volume = 'o0ndo35b3f'
            region = 'AP-JP-1'
        else:
            region = 'EU-RO-1'
            volume_row = json.loads(subprocess.check_output([CLI, 'network-volume', 'create',
                '--name', 'gearshift-confirmation-private-scoring', '--size', '20',
                '--data-center-id', region]))
            atomic_json(folder / 'volume.json', volume_row)
            volume = volume_row['id']
        request = {'name': 'gearshift-confirmation-' + name, 'image': IMAGE, 'disk': 40,
            'cloud': 'SECURE', 'dataCenterIds': [region], 'ports': ['22/tcp'], 'startSsh': True,
            'mounts': {'network': [{'volumeId': volume, 'path': '/workspace'}]}}
        if kind == 'gpu':
            request['gpu'] = {'id': 'NVIDIA H200', 'count': count,
                              'minRamPerGpu': 96, 'minVcpuCountPerGpu': 8}
        else:
            request['cpu'] = {'id': 'cpu5g', 'vcpuCount': 16}
        atomic_json(folder / 'create_request.json', request)
        try:
            pod = api('pods', 'POST', request)
        except Exception as exc:
            atomic_json(folder / 'ambiguous_creation.json', {
                'error_type': type(exc).__name__, 'http_status': getattr(exc, 'code', None),
                'reconciliation_required_before_retry': True, 'epoch': time.time()})
            raise
        lease.update(pod_id=pod['id'], network_volume_id=volume,
                     control_relative='allocations/' + pod['id'])
        atomic_json(folder / 'lease.json', lease)
        atomic_json(folder / 'pod.json', safe(pod))
        print(json.dumps({'allocation': name, 'pod': safe(pod), 'reservation': reservation}))


def ssh_args(pod):
    d = pod['ssh']['direct']
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
            '-o', 'StrictHostKeyChecking=accept-new', '-i', str(Path.home() / '.ssh/id_ed25519'),
            '-p', str(d['port']), d.get('username', 'root') + '@' + d['host']]


def arm(name):
    folder = EVIDENCE / name
    lease_path = folder / 'lease.json'
    lease = json.loads(lease_path.read_text())
    verify_lease(lease)
    pod = api('pods/' + lease['pod_id'])
    atomic_json(folder / 'pod.json', safe(pod))
    if not pod.get('ssh', {}).get('direct'):
        raise RuntimeError('Pod has no direct SSH endpoint yet')
    control = lease['allowed_result_root'] + '/' + lease['control_relative']
    remote_lease = REMOTE + '/' + str(lease_path.relative_to(ROOT))
    guard_source = (ROOT / 'gearshift/coding_confirmation_lease.py').read_text()
    payload = {'lease': lease_path.read_text(), 'guard': guard_source, 'key': key()}
    # The API key travels only on encrypted stdin, never in process arguments/results.
    setup = """import json,os,pathlib,subprocess,sys
p=json.load(sys.stdin)
env=dict(x.split(b'=',1) for x in pathlib.Path('/proc/1/environ').read_bytes().split(b'\\0') if b'=' in x)
assert (os.environ.get('RUNPOD_POD_ID') or env.get(b'RUNPOD_POD_ID',b'').decode())==POD
for path,text,mode in [(LEASE,p['lease'],0o644),(GUARD,p['guard'],0o644),('/root/.gearshift-confirmation-provider-key',p['key'],0o600)]:
 f=pathlib.Path(path);f.parent.mkdir(parents=True,exist_ok=True);f.write_text(text);f.chmod(mode)
pathlib.Path(CONTROL+'/lease_guard').mkdir(parents=True,exist_ok=True)
log=open(CONTROL+'/lease_guard/launcher.log','ab')
q=subprocess.Popen(['python3',GUARD,'--lease',LEASE,'--lease-sha256',SHA,'--key-file','/root/.gearshift-confirmation-provider-key'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps({'guard_pid':q.pid}))
"""
    constants = '\n'.join(k + '=' + repr(v) for k, v in {
        'POD': lease['pod_id'], 'LEASE': remote_lease,
        'GUARD': REMOTE + '/gearshift/coding_confirmation_lease.py',
        'CONTROL': control, 'SHA': sha256_file(lease_path)}.items())
    result = subprocess.run(ssh_args(pod) + ['python3 -c ' + shlex.quote(constants + '\n' + setup)],
        input=json.dumps(payload), text=True, capture_output=True, timeout=40, check=True)
    receipt = json.loads(result.stdout)
    atomic_json(folder / 'guard_dispatch.json', {**receipt, 'epoch': time.time(),
        'lease_sha256': sha256_file(lease_path), 'guard_source_sha256': sha256_file(ROOT / 'gearshift/coding_confirmation_lease.py')})
    print(json.dumps(receipt))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['allocate', 'arm'])
    p.add_argument('--name', required=True)
    p.add_argument('--kind', choices=['cpu', 'gpu'], default='gpu')
    p.add_argument('--count', type=int, choices=[1, 2, 4, 8], default=1)
    a = p.parse_args()
    if a.action == 'allocate': allocate(a.name, a.kind, a.count)
    else: arm(a.name)
