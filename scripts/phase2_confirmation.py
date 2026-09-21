#!/usr/bin/env python3
"""One shared live source trajectory for frozen and adapted mapper comparisons."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gearshift.phase2_io import read,write,immutable,artifact,digest,snapshot,validate_transaction
from gearshift.phase2_tasks import from_dict,apply_budgets
from gearshift.phase2_inference import backends,stage_manifest,source_history,adapter_from,measure_condition,commit_task,complete_stage


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase2_v1.json');p.add_argument('--pair',default='1p7_to_0p6');args=p.parse_args()
    cfg=read(args.config);root=Path(cfg['output']);policy=read(root/'tasks/adaptation_reservation.json');source,target=backends(cfg)
    selected=read(root/'training'/args.pair/'selection.json')
    tasks=apply_budgets(cfg,[from_dict(t) for t in read(root/'tasks/confirmation_evaluation.json')])
    replica=args.pair!='1p7_to_0p6'
    if replica:tasks=[t for t in tasks if t.task_id in policy['replication_task_ids']]
    first=selected['representative_seed'];extra_ids=set(policy['all_seed_task_ids'])
    paths={}
    if not replica:paths['M/frozen']='results/followup_v1/selected_mapper.pt'
    else:paths['M/initial']=str(root/'training'/args.pair/'affine_initialization.pt')
    objectives=cfg.get('training_objectives',['ordinary','boundary'])
    for seed in selected['seeds']:
        for objective in objectives:paths[f'M/{objective}/{seed}']=str(root/'training'/args.pair/f'seed_{seed}'/f'{objective}_best.pt')
    conditions={}
    for task in tasks:
        c=['S','S_think','B/newturn','B/native','C']
        c += [label for label in paths if label.count('/')==1 or label.endswith('/'+str(first)) or task.task_id in extra_ids]
        c += ['H/selected']
        conditions[task.task_id]=c
    stage='confirmation_'+args.pair
    dest,ident=stage_manifest(cfg,source,target,tasks,conditions,paths,stage,
        dict(selection=artifact(root/'training'/args.pair/'selection.json'),policy=artifact(root/'tasks/adaptation_reservation.json'),
            script=snapshot(['scripts/phase2_confirmation.py']),
            final_access='No adaptation or selection based on these outputs. First-seed full balanced confirmation; additional seeds use the same prespecified subset.'))
    adapters={label:adapter_from(path,source,target) for label,path in paths.items()};chosen=f'M/{selected["selected_objective"]}/{first}'
    for i,task in enumerate(tasks):
        path=dest/'questions'/f'{task.task_id}.json';expected=conditions[task.task_id]
        if path.exists():validate_transaction(read(path),ident,task.task_id,expected);continue
        tr=None;rows=[]
        try:
            tr,sp=source_history(source,target,task)
            for c in np.random.default_rng(cfg['seed']+i).permutation(expected).tolist():
                adapter=adapters.get(c,adapters[chosen])
                run_condition='H' if c=='H/selected' else c
                row=measure_condition(source,target,adapter,task,tr,sp,run_condition);row['condition']=c;rows.append(row)
        except BaseException as exc:
            attempt=len(list((dest/'failures').glob(task.task_id+'_*.json')))
            write(dest/'failures'/f'{task.task_id}_{attempt}.json',dict(task_id=task.task_id,trajectory=tr,completed_rows=rows,error=repr(exc)))
            raise
        commit_task(dest,ident,task,tr,rows,expected);complete_stage(dest,ident,tasks,conditions)
        print(f'{stage} {i+1}/{len(tasks)} complete',flush=True)
    immutable(dest/'tasks.json',[dict(t.visible(),grader=t.grader,hidden=t.hidden,metadata=t.metadata) for t in tasks])


if __name__=='__main__':main()
