#!/usr/bin/env python3
"""Evaluate verified last-common weights after external controller interruption."""
import gc,json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha
from gearshift.coding_coverage import ARMS,validate_recovery
from gearshift.coding_training import read
from coding_coverage_experiment import exposure_report,evaluate_and_score
ROOT=Path(__file__).resolve().parents[1]


def evaluate_recovered(c,state):
    import torch
    d,train,val,schedules,panels,source,b,mappers,runtimes,optimizers=state
    path=ROOT/c['spec']['recovery_path']
    if sha(path)!=c['spec']['recovery_sha256']:raise ValueError('Recovery receipt changed')
    receipt=read(path);pair=validate_recovery(ROOT,receipt);step=pair['step'];root=c['root']
    original=ROOT/receipt['source_result_root'];started=time.time()
    if optimizers:raise ValueError('Evaluation recovery must not create optimizers')
    # Preserve original training attribution; the new identity describes only
    # fresh matched generation/scoring and its native numerical controls.
    for name in ['training_steps.json','validation_curve.json','answer_span_audit.json','planned_exposure.json',
                 'frozen_schedule_prefix.json','frozen_endpoint.json','optimizer_reset.json','training_identity.json','training_failure.json']:
        rel=str((original/name).relative_to(ROOT))
        if rel not in receipt['inputs']:raise ValueError('Unbound original training record: '+name)
        shutil.copy2(original/name,root/name)
    for p in (original/'paired_checkpoints').glob('step_*.json'):
        (root/'paired_checkpoints').mkdir(exist_ok=True);shutil.copy2(p,root/'paired_checkpoints'/p.name)
    write(root/'recovery_provenance.json',receipt)
    audits=read(original/'answer_span_audit.json')
    audits=[a for a in audits if a['task_id'] in d['training_task_ids']]
    count=receipt['completed_logged_updates']
    for arm in ARMS:
        write(root/(arm+'_actual_exposure.json'),exposure_report(train,schedules[arm],audits,count))
        write(root/(arm+'_primary_checkpoint_exposure.json'),exposure_report(train,schedules[arm],audits,step))
    stop='engineering_incomplete_last_common_checkpoint'
    write(root/'training_complete.json',{'target_additional_updates_per_arm':receipt['target_updates'],
        'completed_paired_updates':count,'completed_updates_are_last_backup_lower_bound':True,
        'primary_common_checkpoint':step,'primary_predictions_per_arm':step*32,'actual_predictions_per_arm':count*32,
        'stop_reason':stop,'checkpoint_selection':'Latest verified durable common checkpoint after external network/controller failure; never selected by loss or program correctness.',
        'hidden_tests_loaded':False,'quality_scores_generated':False,'training_resumed':False,
        'original_training_result_root':receipt['source_result_root'],'evaluation_recovery_identity_sha256':c['spec']['stage_identity']})
    curve=read(root/'validation_curve.json')
    for arm in ARMS:
        matches=[r for r in curve if r['step']==step and r['arm']==arm]
        if len(matches)!=1 or matches[0]['checkpoint']['sha256']!=pair['arms'][arm]['sha256']:
            raise ValueError('Recovered primary checkpoint lacks its matching validation panels')
    runtimes.clear();gc.collect();torch.cuda.empty_cache()
    cp={'START':{'path':d['selected_checkpoint'],'sha256':d['selected_checkpoint_sha256']},**pair['arms']}
    evaluate_and_score(c,d,train,val,source,b,mappers['FIXED'],cp,step,receipt['target_updates'],stop,started)
