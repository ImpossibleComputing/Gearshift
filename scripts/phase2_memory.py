#!/usr/bin/env python3
"""Closed-world paired counterfactual diagnostic; never treated as fresh confirmation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import numpy as np
import torch
from gearshift.phase2_io import read,write,digest,artifact,immutable,snapshot,validate_transaction
from gearshift.phase2_tasks import from_dict
from gearshift.phase2_inference import backends,stage_manifest,source_history,adapter_from,measure_condition,commit_task,complete_stage
from gearshift.phase2_grading import grade


@torch.inference_mode()
def main():
    cfg=read('configs/phase2_v1.json');source,target=backends(cfg)
    tasks=[replace(from_dict(t),answer_budget=128,reasoning_budget=768,
        contract='For this diagnostic give only three lines: Decision: <vendor>; Monthly total: <number>; Difference: <number>. Put each field on its own line. Use the facts and eligibility correction in the preceding packet. No prose.')
        for t in read('results/phase2_v1/tasks/memory.json')]
    conditions=['B/newturn','C','M','H','N_absent'];mapper='results/followup_v1/selected_mapper.pt'
    root,ident=stage_manifest(cfg,source,target,tasks,conditions,{'frozen':mapper},'memory',
        dict(script=snapshot(['scripts/phase2_memory.py','gearshift/phase2_grading.py']),
          wrong_history='Cross-use the paired variant output: the generic bridge and decoding are identical. This is exactly the deterministic wrong-history intervention, with no additional inference and no economic claim.',
          absent='Small receiver given only a generic statement that no evidence was supplied and the generic field contract.',
          clusters='12 parent packets with two decision-changing variants each; variants stay together.'))
    adapter=adapter_from(mapper,source,target)
    for i,task in enumerate(tasks):
        path=root/'questions'/f'{task.task_id}.json'
        if path.exists():validate_transaction(read(path),ident,task.task_id,conditions);continue
        tr,sp=source_history(source,target,task);rows=[]
        for c in np.random.default_rng(cfg['seed']+i).permutation(conditions).tolist():
            if c=='N_absent':
                empty=replace(task,prompt='No task-specific evidence packet is supplied.')
                row=measure_condition(source,target,adapter,empty,tr,sp,'S');row['condition']=c
                row['absent_visible_prompt']=empty.generation_text()
            else:row=measure_condition(source,target,adapter,task,tr,sp,c)
            rows.append(row)
        commit_task(root,ident,task,tr,rows,conditions)
        complete_stage(root,ident,tasks,{t.task_id:conditions for t in tasks})
        print(f'memory {i+1}/{len(tasks)} complete',flush=True)
    objects={t.task_id:read(root/'questions'/f'{t.task_id}.json') for t in tasks};scored=[]
    for task in tasks:
        own=objects[task.task_id];other=objects[task.task_id[:-1]+str(1-int(task.task_id[-1]))]
        for label,obj in [('correct_history',own),('wrong_history',other)]:
            for row in obj['rows']:
                if row['condition']=='N_absent' and label=='wrong_history':continue
                scored.append(dict(task_id=task.task_id,cluster_id=task.cluster_id,history_assignment=label,
                    condition=row['condition'],actual_history_task_id=obj['task_id'],
                    actual_history_sha256=row['shared_source_trajectory_sha256'],actual_history_tokens=len(obj['trajectory']['prompt_ids'])+len(obj['trajectory']['source_reasoning_ids']),
                    evaluated_packet_sha256=digest(task.prompt),score=grade(task,row['answer'])))
    write(root/'counterfactual_scores.json',scored)
    immutable(root/'tasks.json',[dict(t.visible(),grader=t.grader,hidden=t.hidden,metadata=t.metadata) for t in tasks])


if __name__=='__main__':main()
