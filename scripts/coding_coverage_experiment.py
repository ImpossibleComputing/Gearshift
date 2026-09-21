#!/usr/bin/env python3
"""Paired training first, frozen-seed generations second, private scoring last."""
import collections,gc,math,re,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.coding_control import write,sha,digest
from gearshift.coding_coverage import ARMS,grouped
from gearshift.coding_coverage_runtime import paired_update,validation
from gearshift.coding_training import read,finish_worker
from coding_coverage_worker import save_pair
from coding_coverage_evaluate import generate_suite
ROOT=Path(__file__).resolve().parents[1]


def span_audit(tokenizer,obj):
    """Descriptive token/line labels across every fenced code region, not a grader."""
    ids=obj['teacher_answer']['answer_ids'];text=tokenizer.decode(ids,skip_special_tokens=False)
    spans=[]
    for m in re.finditer(r'```[^\n]*\n.*?```',text,re.S):spans.append((m.start(),m.end()))
    # No first-fence-only assumption. Plain answers without fences stay explicitly
    # ambiguous instead of automatically declaring all their text to be prose.
    rows=[];start=0
    for i,token in enumerate(ids):
        end=len(tokenizer.decode(ids[:i+1],skip_special_tokens=False))
        lo=text.rfind('\n',0,start)+1;hi=text.find('\n',start);line=text[lo:hi if hi>=0 else len(text)].strip()
        in_code=any(a<=start<b for a,b in spans)
        if token in [151643,151645]:category='EOS'
        elif line.startswith('```'):category='code_fence'
        elif in_code and re.match(r'(async\s+def|def|class)\s',line):category='signature'
        elif in_code and re.search(r'\breturn\b',line):category='return_statement'
        elif in_code:category='code_body'
        elif spans:category='prose'
        else:category='unfenced_ambiguous'
        rows.append({'position':i,'token_id':token,'char_start':start,'char_end':end,'text':text[start:end],'category':category});start=end
    return {'task_id':obj['task_id'],'answer_text_with_special_tokens':text,'positions':rows,
        'method':'Descriptive prefix-decode character offsets and visible lines across all complete fenced regions; token spans may cross category boundaries. Unfenced text remains ambiguous. These labels do not select training positions or assess correctness.'}


def exposure_report(histories,schedule,audits,steps):
    categories={r['task_id']:{p['position']:p['category'] for p in r['positions']} for r in audits}
    counters={o['task_id']:collections.Counter() for o in histories}
    for item in schedule[:steps]:
        for tid,positions in grouped(item).items():counters[tid].update(positions)
    rows={}
    for obj in histories:
        tid=obj['task_id'];counts=counters[tid];cats=categories[tid]
        rows[tid]={'positions':dict(sorted(counts.items())),'scored_positions':sum(counts.values()),'unique_positions':len(counts),
            'available_positions':len(obj['teacher_answer']['answer_ids']),
            'by_category':{name:{'scored':sum(v for p,v in counts.items() if cats[p]==name),
                'unique':sum(cats[p]==name for p in counts),'available':sum(v==name for v in cats.values())}
                for name in sorted(set(cats.values()))}}
    return {'steps':steps,'scored_positions':sum(x['scored_positions'] for x in rows.values()),
            'unique_task_positions':sum(x['unique_positions'] for x in rows.values()),'tasks':rows}


def experiment(c,state):
    d,train,val,schedules,panels,source,b,mappers,runtimes,optimizers=state
    root=c['root'];endpoint_path=ROOT/c['spec']['endpoint_path']
    if sha(endpoint_path)!=c['spec']['endpoint_sha256']:raise ValueError('Endpoint changed')
    endpoint=read(endpoint_path);target=endpoint['endpoint'];write(root/'frozen_endpoint.json',endpoint)
    if endpoint['quality_scores_consulted'] is not False:raise ValueError('Resource-only endpoint required')
    started=time.time();lookup={h['task_id']:h for h in train};curve=[];steps=[];pairs={};last_checkpoint=0
    audits=[span_audit(b.tokenizer,h) for h in train+val];write(root/'answer_span_audit.json',audits)
    training_audits=audits[:len(train)]
    write(root/'planned_exposure.json',{a:exposure_report(train,schedules[a],training_audits,target) for a in ARMS})
    write(root/'frozen_schedule_prefix.json',{a:schedules[a][:target] for a in ARMS})
    # Six hours remain for all504+60 generations/scoring, plus15min final export.
    training_deadline=c['spec']['deadline_epoch']-6*3600-900
    def training_guard():
        c['guard']()
        if time.time()>=training_deadline:raise TimeoutError('Reserved evaluation time reached; use last common checkpoint')
    def checkpoint(step):
        nonlocal last_checkpoint
        pairs[step]=save_pair(c,mappers,step);last_checkpoint=step
        if step==0:
            result=validation(runtimes['FIXED'],val,panels,c['publish'])
            for arm in ARMS:curve.append({'step':step,'arm':arm,'checkpoint':pairs[step][arm],**result,'shared_identical_start':True})
        else:
            for arm in ARMS:
                result=validation(runtimes[arm],val,panels,c['publish'])
                curve.append({'step':step,'arm':arm,'checkpoint':pairs[step][arm],**result})
        if any(not math.isfinite(r[k]) for r in curve for k in ['legacy_mean_task_kl','broad_mean_task_kl']):raise FloatingPointError('Nonfinite validation KL')
        write(root/'validation_curve.json',curve)
    checkpoint(0)
    for runtime in runtimes.values():runtime.guard=training_guard
    stop='target_completed';failure=None
    try:
        for step in range(1,target+1):
            training_guard();c['publish'](stage='paired_coverage_training',step=step,target_updates=target)
            items={a:schedules[a][step-1] for a in ARMS}
            result=paired_update(items,lookup,runtimes,optimizers,ARMS if step%2 else ARMS[::-1])
            steps.append({'step':step,**result});write(root/'training_steps.json',steps)
            c['telemetry'].sample(stage='paired_update_committed',update=step)
            if step in d['checkpoint_updates']:checkpoint(step)
    except (RuntimeError,TimeoutError,FloatingPointError) as exc:
        stop='engineering_incomplete_last_common_checkpoint';failure={'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
            'completed_paired_updates':len(steps),'last_common_checkpoint':last_checkpoint,'target':target}
        c['telemetry'].failure(exc);write(root/'training_failure.json',failure)
        if 'out of memory' in str(exc).lower():gc.collect();torch.cuda.empty_cache()
    finally:
        for runtime in runtimes.values():runtime.guard=c['guard']
        for optimizer in optimizers.values():optimizer.zero_grad(set_to_none=True)
    if failure and last_checkpoint==0:raise RuntimeError('No trained common checkpoint; preserve partial preflight/training evidence')
    # A deadline can interrupt validation after the paired weights are saved.
    # Complete missing panels for the primary common checkpoint before answers.
    for arm in ARMS:
        if not any(r['step']==last_checkpoint and r['arm']==arm for r in curve):
            state_dict=torch.load(ROOT/pairs[last_checkpoint][arm]['path'],map_location='cpu',weights_only=True)['state_dict']
            mappers[arm].load_state_dict(state_dict);del state_dict
            result=validation(runtimes[arm],val,panels,c['publish'])
            curve.append({'step':last_checkpoint,'arm':arm,'checkpoint':pairs[last_checkpoint][arm],**result})
            write(root/'validation_curve.json',curve)
    for arm in ARMS:
        write(root/(arm+'_actual_exposure.json'),exposure_report(train,schedules[arm],training_audits,len(steps)))
        write(root/(arm+'_primary_checkpoint_exposure.json'),exposure_report(train,schedules[arm],training_audits,last_checkpoint))
    write(root/'training_complete.json',{'target_additional_updates_per_arm':target,'completed_paired_updates':len(steps),
        'primary_common_checkpoint':last_checkpoint,'primary_predictions_per_arm':last_checkpoint*32,'actual_predictions_per_arm':len(steps)*32,
        'stop_reason':stop,'checkpoint_selection':'Final common-budget paired checkpoint, or last durable common checkpoint on engineering failure; never quality or KL selection.',
        'hidden_tests_loaded':False,'quality_scores_generated':False,'training_and_validation_wall_seconds':time.time()-started})
    # Release gradients/optimizer storage before inference, not the frozen models.
    optimizers.clear();runtimes.clear();mappers['ROTATING']=None
    del optimizer,runtime
    gc.collect();torch.cuda.empty_cache()
    cp={'START':{'path':d['selected_checkpoint'],'sha256':d['selected_checkpoint_sha256']},
        'FIXED':pairs[last_checkpoint]['FIXED'],'ROTATING':pairs[last_checkpoint]['ROTATING']}
    evaluate_and_score(c,d,train,val,source,b,mappers['FIXED'],cp,last_checkpoint,target,stop,started)


def evaluate_and_score(c,d,train,val,source,b,mapper,cp,last_checkpoint,target,stop,started):
    root=c['root'];lookup={h['task_id']:h for h in train}
    visible={r['task_id']:r for r in read(ROOT/'data/coding_pilot_v1/visible/coverage_generalization.json')}
    seeds=read(ROOT/d['answer_seeds_path'])
    generation_started=time.time()
    generate_suite(c,source,b,mapper,val,visible,seeds['validation'],cp,'validation')
    seen=[lookup[tid] for tid in d['seen_training_task_ids']]
    generate_suite(c,source,b,mapper,seen,visible,seeds['seen_training'],
        {'START':cp['START'],'SEEN_FIT':{'path':d['seen_checkpoint'],'sha256':d['seen_checkpoint_sha256']}},'seen_training')
    # Tests are loaded only here, AFTER all generation and optimization. Sandbox
    # failures remain engineering failures, never silently counted as model misses.
    from gearshift.coding_sandbox import extract,score
    if not read(ROOT/'evidence/coding_pilot_v1/sandbox_gate.json')['passed']:raise RuntimeError('Sandbox not ready')
    private=read(ROOT/'data/coding_pilot_v1/private/coverage_generalization.json')
    if set(private)!=set(d['validation_task_ids']+d['seen_training_task_ids']):raise ValueError('Scorer population changed')
    scored=[];score_started=time.time()
    for scope in ['validation','seen_training']:
        for path in sorted((root/scope).glob('*/*/seed_*/answer.json')):
            c['guard']();row=read(path);c['publish'](stage='isolated_coverage_scoring',scope=scope,task_id=row['task_id'],condition=row['condition'],scored=len(scored),expected=564)
            row['code']=extract(row['answer_text']);row['score']=score(row['code'],private[row['task_id']],c['guard'])
            if row['score']['category'] in ['sandbox_unavailable','missing_tests']:raise RuntimeError('Scorer unavailable')
            write(path,row);scored.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
            write(root/'scoring_progress.json',{'scored':len(scored),'expected':564})
    if len(scored)!=564:raise ValueError('Expected504 validation and60 seen-case fresh draws')
    write(root/'scored_answer_manifest.json',{'files':scored,'hidden_tests_loaded_after_optimization_and_generation':True})
    write(root/'complete.json',{'identity_sha256':digest(c['identity']),'primary_common_checkpoint':last_checkpoint,'target_updates_per_arm':target,
        'primary_predictions_per_arm':last_checkpoint*32,'validation_task_count':21,'seen_task_count':4,'fresh_scored_answers':len(scored),
        'validation_seeds_per_task':3,'seen_seeds_per_task':5,'stop_reason':stop,'confirmation_used':False,
        'generation_and_scoring_seconds':time.time()-generation_started,'scoring_seconds':time.time()-score_started,'total_worker_seconds':time.time()-started})
    finish_worker(c,'complete',stage='coverage_comparison_complete',common_updates=last_checkpoint,scored_answers=len(scored))
