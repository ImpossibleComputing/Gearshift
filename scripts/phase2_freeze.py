#!/usr/bin/env python3
"""Lock final conditions using completed development only; never inspect final outputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gearshift.phase2_io import read,immutable,artifact,digest
from gearshift.phase2_tasks import from_dict,apply_budgets


def main():
    root=Path('results/phase2_v1');cfg=read('configs/phase2_final.json')
    if not (root/'development/complete.json').exists():raise ValueError('Development must complete before final lock')
    if not read('evidence/phase2/code_grader_validation_v2.json')['passed']:raise ValueError('Executable grader validation required')
    reuse=read('evidence/phase2/code_grader_validation_reuse.json')
    if not reuse['passed'] or reuse['current_task_manifest']!=artifact(root/'tasks/development.json'):raise ValueError('Code grader validation identity drift')
    if not read('evidence/phase2/sandbox_probe.json')['passed']:raise ValueError('Secure code sandbox required')
    if not (root/'tasks/extended_reservation.json').exists():raise ValueError('Reserve extended sample first')
    review=read('evidence/phase2/development_review.json')
    if not review['final_protocol_approved_from_development'] or review['development_complete']!=artifact(root/'development/complete.json'):
        raise ValueError('Completed development review required')
    tasks=read(root/'tasks/characterization.json');rng=np.random.default_rng(cfg['seed']);secondary=set()
    for family in ['evidence','writing']:
        ids=[t['task_id'] for t in tasks if t['family']==family]
        secondary.update(ids[i] for i in rng.permutation(len(ids))[:32])
    conditions={}
    for task in tasks:
        # Preserve both small modes and both competent source protocols, avoiding selection on weak prose field checks.
        c=['S','S_think','B/newturn','B/native','C','M','H']
        if task['family']=='arithmetic' or task['task_id'] in secondary:c+=['T']
        if task['task_id'] in secondary:c+=['P']
        conditions[task['task_id']]=c
    immutable(root/'tasks/characterization_conditions.json',conditions)
    confirm=read(root/'tasks/confirmation.json');replication=[];seed_variability=[];confirmation_ids=[]
    for family in ['arithmetic','code','evidence','writing']:
        ids=[t['task_id'] for t in confirm if t['family']==family];order=rng.permutation(len(ids))
        confirmation_ids.extend(ids[i] for i in order[:32])
        replication.extend(ids[i] for i in order[:16]);seed_variability.extend(ids[i] for i in order[:8])
    immutable(root/'tasks/confirmation_evaluation.json',[t for t in confirm if t['task_id'] in confirmation_ids])
    immutable(root/'tasks/adaptation_reservation.json',dict(tasks=artifact(root/'tasks/confirmation.json'),
        evaluated_tasks=artifact(root/'tasks/confirmation_evaluation.json'),confirmation_ids=confirmation_ids,
        scope='312 tasks reserved; a balanced 128-task subset (32 per family) is selected before final evaluation for bounded adaptation confirmation. The remaining 184 stay untouched.',
        primary_seed=cfg['training']['seeds'][0],all_seed_task_ids=seed_variability,replication_task_ids=replication,
        primary_methods=['S','S_think','B/newturn','B/native','C','M/frozen','M/ordinary/first_seed','M/boundary/first_seed','H/selected'],
        other_seeds='Both objectives on the same 32 prespecified confirmation tasks; first seed on all 128 selected tasks.',
        replication='Selected objective on 4B→0.6B, one prespecified seed and 64 balanced confirmation tasks, using fresh source-specific affine fitting. Exploratory replication, not a power claim.',
        optional_larger_receiver='Consider only from native-replay/source capability gaps; no automatic expansion into a second receiver training search.'))
    immutable(root/'tasks/final_task_contracts.json',{name:[t.visible() for t in apply_budgets(cfg,[from_dict(t) for t in read(root/'tasks'/f'{name}.json')])]
        for name in ['characterization','confirmation_evaluation','train','validation']})
    immutable(root/'final_protocol_lock.json',dict(config=artifact('configs/phase2_final.json'),
        original_development_config=artifact('configs/phase2_v1.json'),source_reasoning_caps=cfg['source_reasoning_caps'],
        effective_task_contracts=artifact(root/'tasks/final_task_contracts.json'),
        task_reservation=artifact(root/'tasks/reservation.json'),characterization_conditions=artifact(root/'tasks/characterization_conditions.json'),
        extended_reservation=artifact(root/'tasks/extended_reservation.json'),adaptation_reservation=artifact(root/'tasks/adaptation_reservation.json'),
        development=artifact(root/'development/complete.json'),rubric=artifact('configs/phase2_judging.json'),
        development_review=artifact('evidence/phase2/development_review.json'),judging_execution_policy=artifact(root/'judging_execution_policy.json'),
        human_reservation=artifact(root/'tasks/human_reservation.json'),
        dataset_audit=artifact('evidence/phase2/exact_dataset_revision_audit.json'),
        resource_decision='Retain all 640 requested frozen-characterization tasks and the full planned paired conditions. Secondary compact-plan/tail methods use 32 shared cases in each longer-output family. Replication and extra-seed task evaluation are bounded subsets declared before confirmation.',
        primary_contrasts=['M−C transfer','M−B/newturn switch utility','M−B/native source reset sensitivity','M−S and M−S_think added reasoning value','M−T/P text alternative'],
        claims='Exploratory paired intervals; no equivalence or noninferiority claim; separate families and quality/format metrics; prose field checks do not establish semantic acceptability.',
        test_outputs_opened=False))
    print('Locked full 640-task characterization and disjoint adaptation/replication policies')


if __name__=='__main__':main()
