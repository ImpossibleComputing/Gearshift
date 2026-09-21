#!/usr/bin/env python3
"""Run the already-declared local stages in order while keeping separate logs."""
from pathlib import Path
import subprocess
import time
import sys
import os


def main():
    # The current ablation is already running in a separate local process.
    while not Path('results/phase2_v1/ablation/complete.json').exists():
        time.sleep(15)
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    commands=[('memory',[sys.executable,'scripts/phase2_memory.py']),
              ('development',[sys.executable,'scripts/phase2.py','development','--tasks','results/phase2_v1/tasks/development.json',
                               '--conditions','results/phase2_v1/tasks/development_conditions.json']),
              ('development_scoring',[sys.executable,'scripts/phase2_score.py','development','--tasks','results/phase2_v1/tasks/development.json'])]
    for label,cmd in commands:
        with open(f'evidence/phase2/{label}.txt','w') as output:
            result=subprocess.run(cmd,env=env,stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(f'{label} failed with {result.returncode}; inspect preserved log')


if __name__=='__main__':main()
