#!/usr/bin/env python3
"""Independent bounded judge queue; local model execution may proceed meanwhile."""
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read,write


def main():
    from gearshift.phase2_pause import require_unpaused
    require_unpaused()
    root=Path('results/phase2_v1');policy=read(root/'judging_execution_policy.json');stages=policy['primary_stage_order']
    for round_name,scope in [('priority','primary'),('primary','primary'),('all','all')]:
        for stage in stages:
            require_unpaused()
            while not (root/stage/'judging/condition_key.json').exists():time.sleep(15)
            folder=Path('evidence/phase2/judging');folder.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'scripts/phase2_judge_resume.py','run','--stage',stage,'--scope',scope]
            if round_name=='priority':command+=['--task-ids',str(root/'judging_priority_cases'/f'{stage}.json')]
            with (folder/f'{stage}_{round_name}.txt').open('a') as out:
                result=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
            if result.returncode:raise SystemExit(f'Judge runtime failed for {stage}; retained logs')
            status=read(root/stage/'judging/judging_status.json')
            if status.get('usage_guard_stop') or status.get('service_blocked'):
                write(root/'judging_queue_status.json',dict(state='stopped_with_pending_packets',stage=stage,scope=scope,status=status));return
    write(root/'judging_queue_status.json',dict(state='all_exported_packets_attempted',stages=stages,
        note='Invalid judgments, if any, remain unscored. No human labels are implied.'))


if __name__=='__main__':main()
