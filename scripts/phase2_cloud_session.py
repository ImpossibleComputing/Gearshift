#!/usr/bin/env python3
"""Studio-owned upload, bootstrap, continuous collection and terminal pod cleanup.

Launch only for an already registered task pod. Does not allocate resources.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from phase2_cloud_watchdog import DEADLINE, PREFIX, cli, write
from phase2_execution_authorization import SHUTDOWN_UTC

ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / 'evidence/phase2/studio_transfer/cloud'
REMOTE = '/workspace/Gearshift'
RESULT_DIRS = ['characterization', 'training', 'confirmation_1p7_to_0p6', 'confirmation_4b_to_0p6', 'controls_cuda', 'controls_4b_cuda']


def ssh_args(connection):
    args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
        '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile=' + str(CLOUD / 'known_hosts')]
    key = connection.get('ssh_key', {}).get('path')
    if key:
        args += ['-i', key]
    return args + ['-p', str(int(connection['port'])), 'root@' + connection['ip']]


def verify_task(pod, pod_id):
    if pod['id'] != pod_id or not pod['name'].startswith(PREFIX):
        raise ValueError('Refusing unrelated pod')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pod_id')
    p.add_argument('--payload')
    p.add_argument('--resume-existing', action='store_true', help='Collect an existing worker without uploading or restarting it')
    a = p.parse_args()
    if not a.resume_existing and not a.payload:
        p.error('--payload is required for a new worker')
    os.chdir(ROOT)
    folder = CLOUD / a.pod_id
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'session.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pod = cli('pod', 'get', a.pod_id)
    verify_task(pod, a.pod_id)  # Never enter the cleanup path for an unrelated pod.
    subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    start = time.time()
    ssh = None
    def state(label, **extra):
        write(folder / 'session_status.json', dict(state=label, pod_id=a.pod_id, pid=os.getpid(), heartbeat_epoch=time.time(), shutdown_deadline_utc=SHUTDOWN_UTC, **extra))
    def remote(command, timeout=60):
        return subprocess.run([*ssh, command], text=True, capture_output=True, timeout=timeout, check=True)
    def sync(update_status=True):
        # No --delete. Pull only CUDA-owned stage namespaces and immutable source snapshots.
        sources = ['evidence/phase2/cuda_execution', 'evidence/phase2/source_snapshots']
        sources += ['results/phase2_v1/' + name for name in RESULT_DIRS]
        for source in sources:
            exists = subprocess.run([*ssh, 'test -d ' + shlex.quote(REMOTE + '/' + source)], capture_output=True, timeout=25)
            if exists.returncode == 1:
                continue
            if exists.returncode:
                raise RuntimeError('SSH unavailable during collection')
            dest = ROOT / source
            dest.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(['rsync', '-az', '--exclude=*.tmp', '--exclude=*.lock', '-e', shlex.join(ssh[:-1]),
                ssh[-1] + ':' + REMOTE + '/' + source + '/', str(dest) + '/'], capture_output=True, timeout=240)
            if result.returncode not in (0, 24):
                raise RuntimeError('Result sync failed with code ' + str(result.returncode))
        # Replication creates this validation-selected config on the GPU host.
        source = 'configs/phase2_4b_cuda.json'
        exists = subprocess.run([*ssh, 'test -f ' + shlex.quote(REMOTE + '/' + source)], capture_output=True, timeout=25)
        if exists.returncode == 0:
            subprocess.run(['rsync', '-az', '-e', shlex.join(ssh[:-1]),
                ssh[-1] + ':' + REMOTE + '/' + source, str(ROOT / source)], check=True, capture_output=True, timeout=30)
        elif exists.returncode != 1:
            raise RuntimeError('SSH unavailable during replication config collection')
        if update_status:
            state('collecting', last_successful_sync_epoch=time.time())
    try:
        while time.time() < min(DEADLINE, start + 900):
            state('waiting_for_ssh')
            conn = cli('ssh', 'info', a.pod_id)
            if conn.get('ip') and conn.get('port'):
                ssh = ssh_args(conn)
                try:
                    remote('true', timeout=20)
                    write(folder / 'connection.json', conn)
                    break
                except (subprocess.SubprocessError, OSError):
                    pass
            time.sleep(15)
        else:
            raise RuntimeError('Pod SSH startup deadline exceeded')
        if a.resume_existing:
            current = json.loads(remote('cat ' + REMOTE + '/evidence/phase2/cuda_execution/status.json').stdout)
            if current.get('state') not in ('running', 'complete'):
                raise RuntimeError('Existing worker is not healthy for collection resume')
            write(folder / 'collection_resume.json', dict(epoch=time.time(), worker=current, uploaded=False, restarted_worker=False))
        else:
            state('uploading')
            remote('mkdir -p ' + REMOTE)
            with Path(a.payload).open('rb') as stream:
                subprocess.run([*ssh, 'tar --no-same-owner --no-same-permissions -xf - -C ' + REMOTE], stdin=stream, check=True, timeout=900)
            remote('command -v rsync || (apt-get update -qq && apt-get install -y -qq rsync)', timeout=180)
            launch = "import os,subprocess; os.chdir('/workspace/Gearshift'); os.makedirs('evidence/phase2/cuda_execution',exist_ok=True); f=open('evidence/phase2/cuda_execution/bootstrap.txt','w'); p=subprocess.Popen(['python3','scripts/phase2_cuda_bootstrap.py'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)"
            receipt = remote('python3 -c ' + shlex.quote(launch))
            write(folder / 'worker_launch.json', dict(remote_bootstrap_pid=int(receipt.stdout.strip()), epoch=time.time()))
        bootstrap_start = time.time()
        while time.time() < DEADLINE:
            sync()
            for name in ['status.json', 'bootstrap_status.json']:
                path = ROOT / 'evidence/phase2/cuda_execution' / name
                if not path.exists():
                    continue
                status = json.loads(path.read_text())
                if status['state'] == 'failed' or (name == 'status.json' and status['state'] == 'complete'):
                    state('worker_terminal', worker=status)
                    return
            bootstrap = ROOT / 'evidence/phase2/cuda_execution/bootstrap_status.json'
            if time.time() - bootstrap_start > 3600 and (not bootstrap.exists() or json.loads(bootstrap.read_text())['state'] != 'complete'):
                raise RuntimeError('One-hour setup limit exceeded')
            worker = ROOT / 'evidence/phase2/cuda_execution/status.json'
            if worker.exists():
                status = json.loads(worker.read_text())
                if status['state'] == 'running' and time.time() - status['heartbeat_epoch'] > 300:
                    raise RuntimeError('Worker heartbeat stale')
            time.sleep(30)
        state('deadline')
    except BaseException as exc:
        state('failed', error=repr(exc))
        raise
    finally:
        if ssh is not None and time.time() < DEADLINE:
            try:
                sync(update_status=False)
            except Exception:
                pass
        # The separate watchdog keeps retrying cleanup if this request fails.
        cli('pod', 'delete', a.pod_id)
        write(folder / 'cleanup_requested.json', dict(pod_id=a.pod_id, epoch=time.time(), action='delete'))


if __name__ == '__main__':
    main()
