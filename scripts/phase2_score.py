#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from gearshift.phase2_io import read,write,bind,digest,artifact,snapshot,validate_transaction,runtime_identity
from gearshift.phase2_tasks import from_dict
from gearshift.phase2_grading import grade


def validate_grading_task(task,identity):
    expected=next(t for t in identity['tasks'] if t['task_id']==task.task_id)
    visible=task.visible()
    visible['reasoning_budget']=identity['config'].get('source_reasoning_caps',{}).get(task.family,task.reasoning_budget)
    if visible!=expected:raise ValueError('Grading task differs from the generated task')
    if digest(dict(hidden=task.hidden,grader=task.grader,metadata=task.metadata))!=identity['task_grading_hashes'][task.task_id]:
        raise ValueError('Hidden grading material differs from the pre-generation commitment')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--tasks');p.add_argument('--attempt',default='scores');args=p.parse_args()
    root=Path('results/phase2_v1')/args.stage
    mf=read(root/'manifest.json');identity=mf['identity']
    if digest(identity)!=mf['identity_sha256']:raise ValueError('Manifest changed')
    if args.stage=='ablation':
        from gearshift.phase2_tasks import TaskSpec
        tasks={t['task_id']:TaskSpec(**t,grader='numeric',hidden={},metadata={}) for t in identity['tasks']}
    else:tasks={t['task_id']:from_dict(t) for t in read(args.tasks)}
    if set(tasks)!={t['task_id'] for t in identity['tasks']}:raise ValueError('Scoring task membership differs from inference')
    probe=read('evidence/phase2/sandbox_probe.json');sandbox_ok=probe['passed']
    dest=root/args.attempt;dest.mkdir(exist_ok=True)
    bind(dest/'manifest.json',dict(input_manifest=artifact(root/'manifest.json'),runtime=runtime_identity(),
        task_file=artifact(args.tasks) if args.tasks else None,sandbox_probe=artifact('evidence/phase2/sandbox_probe.json'),
        source_code=snapshot(['gearshift/phase2_grading.py','gearshift/phase2_sandbox.py','gearshift/phase2_io.py','gearshift/phase2_tasks.py','scripts/phase2_score.py'])))
    rows=[]
    for q in sorted((root/'questions').glob('*.json')):
        obj=read(q);task=tasks[obj['task_id']]
        conditions=identity['conditions'] if isinstance(identity['conditions'],list) else identity['conditions'][task.task_id]
        validate_transaction(obj,mf['identity_sha256'],task.task_id,conditions)
        if args.stage=='ablation':
            object.__setattr__(task,'hidden',obj['hidden'])
            object.__setattr__(task,'metadata',dict(dataset_index=obj['trajectory']['dataset_index']))
        validate_grading_task(task,identity)
        target=dest/q.name
        if target.exists():
            record=read(target)
            if record['input_sha256']!=artifact(q)['sha256']:raise ValueError('Scored input changed')
        else:
            record=dict(task_id=task.task_id,input_sha256=artifact(q)['sha256'],rows=[dict(task_id=task.task_id,cluster_id=task.cluster_id,
                condition=r['condition'],family=task.family,score=grade(task,r['answer'],sandbox_ok)) for r in obj['rows']])
            write(target,record)
        rows.extend(record['rows'])
    write(root/'objective_scores.json',dict(state='complete' if (root/'complete.json').exists() else 'partial',rows=rows,
        scoring_manifest=artifact(dest/'manifest.json'),scoring_attempt=args.attempt))
    print(f'Scored {len(rows)} outputs from {len(list((root/"questions").glob("*.json")))} tasks')


if __name__=='__main__':main()
