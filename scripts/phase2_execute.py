#!/usr/bin/env python3
"""Bounded local stage execution with durable commands, timing and failure logs."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read,write,immutable,artifact,snapshot

ROOT=Path('results/phase2_v1')


def run_group(group,commands):
    folder=Path('evidence/phase2/execution')/group;folder.mkdir(parents=True,exist_ok=True)
    immutable(folder/'commands.json',dict(commands=commands,executor_source=snapshot(['scripts/phase2_execute.py']),
        policy='Sequential local inference; no concurrent scientific MPS jobs. A failed command stops this group. Reruns retain attempt logs and stage identity validation.'))
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    for label,args in commands:
        previous=sorted(folder.glob(label+'_attempt*.json'))
        if previous and read(previous[-1])['returncode']==0:continue
        index=len(previous);stem=folder/f'{label}_attempt{index:02d}';command=[sys.executable,*args]
        start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(group,label,'started',utc,flush=True)
        with stem.with_suffix('.txt').open('w') as output:
            result=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT)
        write(stem.with_suffix('.json'),dict(command=command,started_utc=utc,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start,
            returncode=result.returncode,stdout_stderr=artifact(stem.with_suffix('.txt'))))
        print(group,label,'finished',result.returncode,flush=True)
        if result.returncode:raise SystemExit(f'{label} failed; inspect {stem.with_suffix(".txt")}')


def main():
    p=argparse.ArgumentParser();p.add_argument('group',choices=['characterization','extensions','adaptation','replication']);a=p.parse_args()
    if not (ROOT/'final_protocol_lock.json').exists():raise ValueError('Final protocol must be locked first')
    if a.group=='characterization':
        commands=[('inference',['scripts/phase2.py','characterization','--config','configs/phase2_final.json','--tasks',str(ROOT/'tasks/characterization.json'),'--conditions',str(ROOT/'tasks/characterization_conditions.json')]),
                  ('scores',['scripts/phase2_score.py','characterization','--tasks',str(ROOT/'tasks/characterization.json')]),
                  ('packets',['scripts/phase2_judge.py','export','--stage','characterization','--tasks',str(ROOT/'tasks/characterization.json'),'--secondary'])]
    elif a.group=='extensions':
        commands=[]
        for stage in ['long_outputs','branching']:
            commands += [(stage,['scripts/phase2_extended.py',stage,'--config','configs/phase2_final.json']),
                         (stage+'_scores',['scripts/phase2_score.py',stage,'--tasks',str(ROOT/stage/'tasks.json')]),
                         (stage+'_packets',['scripts/phase2_judge.py','export','--stage',stage,'--tasks',str(ROOT/stage/'tasks.json'),'--secondary'])]
        commands += [('timing',['scripts/phase2_extended.py','timing','--config','configs/phase2_final.json'])]
    elif a.group=='adaptation':
        commands=[('preparation',['scripts/phase2_train.py','prepare','--config','configs/phase2_final.json']),('training',['scripts/phase2_train.py','train','--config','configs/phase2_final.json']),
                  ('confirmation',['scripts/phase2_confirmation.py','--config','configs/phase2_final.json']),
                  ('scores',['scripts/phase2_score.py','confirmation_1p7_to_0p6','--tasks',str(ROOT/'confirmation_1p7_to_0p6/tasks.json')]),
                  ('packets',['scripts/phase2_judge.py','export','--stage','confirmation_1p7_to_0p6','--tasks',str(ROOT/'confirmation_1p7_to_0p6/tasks.json')])]
    else:
        selected=read(ROOT/'training/1p7_to_0p6/selection.json');cfg=read('configs/phase2_final.json')
        cfg.update(source='Qwen/Qwen3-4B',source_revision='1cfa9a7208912126459214e8b04321603b3df60c',training_objectives=[selected['selected_objective']])
        cfg['training']['seeds']=[selected['representative_seed']];immutable('configs/phase2_4b.json',cfg)
        common=['--config','configs/phase2_4b.json','--pair','4b_to_0p6']
        commands=[('controls',['scripts/phase2_controls.py','--config','configs/phase2_4b.json','--attempt','controls_4b']),
                  ('preparation',['scripts/phase2_train.py','prepare',*common]),
                  ('affine_fit',['scripts/phase2_fit_replication.py',*common]),
                  ('training',['scripts/phase2_train.py','train',*common,'--initialization',str(ROOT/'training/4b_to_0p6/affine_initialization.pt')]),
                  ('confirmation',['scripts/phase2_confirmation.py',*common]),
                  ('scores',['scripts/phase2_score.py','confirmation_4b_to_0p6','--tasks',str(ROOT/'confirmation_4b_to_0p6/tasks.json')]),
                  ('packets',['scripts/phase2_judge.py','export','--stage','confirmation_4b_to_0p6','--tasks',str(ROOT/'confirmation_4b_to_0p6/tasks.json')])]
    run_group(a.group,commands)


if __name__=='__main__':main()
