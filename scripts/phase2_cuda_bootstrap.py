#!/usr/bin/env python3
"""Bounded bootstrap followed directly by the autonomous scientific worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from phase2_cloud_watchdog import DEADLINE, write

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'evidence/phase2/cuda_execution'


def main():
    os.chdir(ROOT)
    STATE.mkdir(parents=True, exist_ok=True)
    write(STATE / 'bootstrap_status.json', dict(state='running', pid=os.getpid(), started_epoch=time.time()))
    uv = str(ROOT / '.bootstrap-tools/bin/uv')
    commands = [
        ['git', 'init'], ['git', 'fetch', 'source.bundle', 'HEAD'],
        ['git', 'update-ref', 'refs/heads/main', 'FETCH_HEAD'], ['git', 'symbolic-ref', 'HEAD', 'refs/heads/main'],
        [sys.executable, '-m', 'pip', 'install', '--target', str(ROOT / '.bootstrap-tools'),
         '--no-deps', '--only-binary=:all:', 'uv==0.10.12'],
        [uv, 'python', 'install', '3.12.4'],
        [uv, 'venv', '--seed', '--python', '3.12.4', '.venv'],
        [uv, 'pip', 'install', '--python', '.venv/bin/python', '-r', 'requirements.lock.txt'],
        ['.venv/bin/python', 'scripts/phase2_cuda_setup.py'],
    ]
    try:
        for command in commands:
            write(STATE / 'bootstrap_status.json', dict(state='running', pid=os.getpid(),
                heartbeat_epoch=time.time(), command=command))
            remaining = DEADLINE - time.time()
            if remaining <= 0:
                raise RuntimeError('Shutdown deadline')
            subprocess.run(command, check=True, timeout=min(1800, remaining))
        write(STATE / 'bootstrap_status.json', dict(state='complete', finished_epoch=time.time()))
        os.execv('.venv/bin/python', ['.venv/bin/python', 'scripts/phase2_cuda_worker.py'])
    except BaseException as exc:
        write(STATE / 'bootstrap_status.json', dict(state='failed', error=repr(exc), finished_epoch=time.time()))
        raise


if __name__ == '__main__':
    main()
