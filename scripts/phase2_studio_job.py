#!/usr/bin/env python3
"""Deadline-bounded durable Studio job, independent of controller/model availability."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from phase2_cloud_watchdog import write, DEADLINE
from phase2_execution_authorization import SHUTDOWN_UTC
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('spec');a=p.parse_args()
    os.chdir(ROOT);spec=json.loads(Path(a.spec).read_text())
    folder=ROOT/'evidence/phase2/studio_transfer/jobs'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime=json.loads((ROOT/'configs/phase2_studio_runtime.json').read_text())
    env=dict(os.environ,**runtime['environment'])
    subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    base=dict(pid=os.getpid(),host=os.uname().nodename,started_epoch=time.time(),spec=spec,deadline_utc=SHUTDOWN_UTC)
    for index,command in enumerate(spec['commands']):
        if time.time()>=DEADLINE:write(folder/'status.json',dict(base,state='deadline'));return
        if spec.get('included_usage_before_each'):
            os.environ.update(runtime['environment'])
            from phase2_usage_guard import included_usage
            usage=included_usage();write(folder/f'usage_{index:02d}.json',usage)
            if not usage['allowed']:write(folder/'status.json',dict(base,state='included_usage_unavailable'));return
        stem=folder/f'command_{index:02d}'
        if stem.with_suffix('.json').exists():raise RuntimeError('Preserve prior attempts; use a new job name')
        with stem.with_suffix('.txt').open('w') as log:
            process=subprocess.Popen([sys.executable,*command],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                write(folder/'status.json',dict(base,state='running',command_index=index,command=command,child_pid=process.pid,heartbeat_epoch=time.time()))
                if time.time()>=DEADLINE:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    break
                try:process.wait(timeout=min(15,max(0.1,DEADLINE-time.time())))
                except subprocess.TimeoutExpired:pass
        write(stem.with_suffix('.json'),dict(command=command,returncode=process.returncode,finished_epoch=time.time()))
        if process.returncode:write(folder/'status.json',dict(base,state='failed',command_index=index,returncode=process.returncode));return
    write(folder/'status.json',dict(base,state='complete',finished_epoch=time.time()))


if __name__=='__main__':main()
