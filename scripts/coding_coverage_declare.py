#!/usr/bin/env python3
"""Freeze exact corpus, schedules, panels, seeds and timing-only preflight rule."""
import json,sys,subprocess,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import bind,sha,digest
from gearshift.coding_coverage import *
from coding_partial_corpus import load_corpus
ROOT=Path(__file__).resolve().parents[1]


def freeze():
    old=json.loads((ROOT/'configs/coding_pilot_v1/post_progress01/declaration.json').read_text())
    histories,_,manifest=load_corpus(ROOT,ROOT/old['corpus_manifest'],features=False)
    train=[x for x in histories if x['split']=='training'];val=[x for x in histories if x['split']=='validation']
    schedules=paired_schedules(train);panels=validation_panels(val)
    e=ROOT/EVIDENCE;e.mkdir(parents=True,exist_ok=True)
    bind(ROOT/CONFIG/'schedules.json',schedules);bind(ROOT/CONFIG/'validation_panels.json',panels)
    seen=[x['task_id'] for x in old['four_training_histories']]
    seeds={'validation':seed_manifest([x['task_id'] for x in val],3),'seen_training':seed_manifest(seen,5)}
    bind(ROOT/CONFIG/'answer_seeds.json',seeds)
    tid_lookup={h['task_id']:h for h in train}
    rep=[round(i*1023/11) for i in range(12)]
    stress=[]
    for name,key in [('longest_prefix',lambda h:len(h['source_history']['prefix_ids'])),
                     ('longest_answer',lambda h:len(h['teacher_answer']['answer_ids'])),
                     ('largest_prefix_answer_product',lambda h:len(h['source_history']['prefix_ids'])*len(h['teacher_answer']['answer_ids']))]:
        obj=max(train,key=lambda h:(key(h),h['task_id']));tid=obj['task_id']
        index=next(i for i,row in enumerate(schedules['FIXED']) if tid in grouped(row))
        stress.append({'name':name,'task_id':tid,'schedule_index':index,'rotation_override':'last N distinct valid positions for this task; N equals its fixed contribution'})
    cp='results/coding_pilot_v1/post_progress01_memorization_20260917_01/memorization/mapper_step_0320.pt'
    if sha(ROOT/old['selected_checkpoint'])!=START_SHA:raise ValueError('Selected96 bytes differ')
    if sha(ROOT/cp)!='ddeb54e3b039c8c1d1bff786230aea7c97879e39e57a732e1f31187556aa9d30':raise ValueError('Seen fit bytes differ')
    tag=subprocess.check_output(['git','rev-parse','gearshift-progress-01^{}'],cwd=ROOT,text=True).strip()
    if tag!=PUBLICATION:raise ValueError('Publication tag changed')
    bind(e/'publication_tag_verification.json',{'tag':'gearshift-progress-01','commit':tag,
        'research_parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),'epoch':time.time()})
    original_inputs={p:sha(ROOT/p) for p in ['gearshift/coding_gradients.py','gearshift/coding_training.py','gearshift/coding_inference.py','gearshift/coding_sandbox.py','gearshift/core.py','gearshift/coding_recovery_answer.py']}
    declaration={'schema':1,'experiment_id':'coverage_generalization_v1','publication_commit':PUBLICATION,
        'owner_instruction_sha256':sha(ROOT/OWNER),'research_parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'corpus_manifest':old['corpus_manifest'],'corpus_manifest_sha256':old['corpus_manifest_sha256'],
        'training_task_ids':[x['task_id'] for x in train],'validation_task_ids':[x['task_id'] for x in val],
        'corpus_limitation':'Exact104/21 completion-selected histories. Validation excludes gradients but was used for prior KL selection, not untouched confirmation.',
        'selected_checkpoint':old['selected_checkpoint'],'selected_checkpoint_sha256':START_SHA,
        'seen_checkpoint':cp,'seen_checkpoint_sha256':sha(ROOT/cp),'seen_training_task_ids':seen,
        'model_config_sha256':sha(ROOT/'configs/coding_pilot_v1/pilot.json'),'unchanged_source_files':original_inputs,
        'optimizer':{'name':'AdamW','lr':1e-5,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0,'clip_norm':1,'lr_schedule':'constant','foreach':False,'fused':False},
        'optimizer_state':'Original selected96 stores mapper only. Both arms reset fresh identical optimizers; preflight optimizers and weights discarded. No silent optimizer resume.',
        'target_additional_updates':1024,'checkpoint_updates':[0,128,256,512,1024],
        'schedules_path':CONFIG+'/schedules.json','schedules_sha256':sha(ROOT/CONFIG/'schedules.json'),
        'panels_path':CONFIG+'/validation_panels.json','panels_sha256':sha(ROOT/CONFIG/'validation_panels.json'),
        'answer_seeds_path':CONFIG+'/answer_seeds.json','answer_seeds_sha256':sha(ROOT/CONFIG/'answer_seeds.json'),
        'task_order':'Exact original make_schedule seed20260915 continuing after original update96; four independent8-position buckets. Both arms preserve each ordered task contribution and total32 positions/update.',
        'fixed':'Original offsets0,32,128,512. Unavailable positions skipped; next seeded history fills bucket to8. No moved anchors, duplicates or padding.',
        'rotating':'Retain original early bucket. Replace later bucket contributions with seeded shuffled8-token windows from position8 through actual last answer token, exhausting cycles before repeats. Partial windows accumulate without padding. For9..32-token answers, the sole early bucket instead cycles ALL windows, including early tokens, so EOS is eligible. No repeated task/position within an update.',
        'validation_panels':'Legacy existing positions at four offsets; broad union of first8 and15 evenly spaced8-token windows ending at the last valid token. Fixed before results; equal-task means reported separately.',
        'preflight':{'representative_schedule_indices':rep,'stress_updates':stress,'maximum_seconds':7200,
            'endpoint_rule':'Largest of1024,512,256,128 fitting1.5x representative mean paired-update time plus1.5x measured full21-task validation time for every required checkpoint, plus6h free-generation/scoring and90min setup/export, within24h allocation. Losses and quality scores do not enter choice.',
            'test_shape':'Three longest-prefix/answer/product cases: native and mapped same-shape manual/deployed/repeat/checkpoint controls, full answer through EOS; exact repeat envelope, no relaxed tolerances.'},
        'evaluation':{'validation_conditions':['START_M','START_H','FIXED_M','FIXED_H','ROTATING_M','ROTATING_H','D','P'],'seeds_per_task':3,
            'seen_conditions':['START_M','SEEN_FIT_M','D'],'seen_seeds_per_task':5,'answer_cap':4096,
            'start':'Saved question + reasoning + closing-think bridge, no teacher answer prefix. P pinned enable_thinking=False template.',
            'hidden_tests':'Loaded only for isolated scoring after all optimization and free generation; never optimizer/model inputs.',
            'primary':'Final common-budget ROTATING_M minus FIXED_M; no best checkpoint/seed selection.',
            'failure':'Last paired checkpoint committed for both arms, disclose later incomplete work; never independently select checkpoints.'},
        'analysis':{'unit':'21 task clusters; mean across3 predeclared seeds','resamples':10000,'bootstrap_seed':20260918,
            'method':'Identical paired task resamples across all contrasts; percentile95%, exploratory, no equivalence inference. Seen-case resampling uses4 clusters separately.'},
        'cache_execution':'Frozen models loaded together; each full512-token historical cache pair reconstructed once per task contribution, retained immutable on GPU across both arms, then released. No disk tensor caches. Alternate arm execution order by update; independent gradients/AdamW; no context cropping.',
        'max_additional_usd':200,'max_additional_gpu_hours':32,'cumulative_cap_usd':1000,'cumulative_cap_gpu_hours':500,
        'budget_start':json.loads((e/'budget_start.json').read_text()),'confirmation_used':False,'publication_modified':False}
    bind(ROOT/DECLARATION,declaration)
    visible={}
    for split in ['training','validation']:
        visible.update({r['task_id']:r for r in json.loads((ROOT/f'data/coding_pilot_v1/visible/{split}.json').read_text())})
    bind(ROOT/'data/coding_pilot_v1/visible/coverage_generalization.json',[visible[tid] for tid in declaration['training_task_ids']+declaration['validation_task_ids']])
    print(json.dumps({'declaration_sha256':sha(ROOT/DECLARATION),'schedules_sha256':declaration['schedules_sha256'],'preflight_cases':stress,'training':104,'validation':21}))

if __name__=='__main__':freeze()
