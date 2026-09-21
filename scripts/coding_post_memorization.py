#!/usr/bin/env python3
"""Bounded four-SEEN-TRAINING-CASE coverage sanity check; no hidden optimizer inputs."""
import argparse,collections,gc,math,os,random,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.coding_control import write,sha,digest
from gearshift.coding_training import read,finish_worker,TrainingRuntime,DiskLRU,atomic_tensor,safe_id
from gearshift.coding_recovery import worker_context,initialize_backends
from gearshift.coding_recovery_answer import answer_instrumented
from gearshift.coding_gradients import AffineMapper
from gearshift.coding_post_progress import DECLARATION,selected_histories
from gearshift.core import CacheInjector
ROOT=Path(__file__).resolve().parents[1]

def coverage_schedule(histories,seed,updates,width):
    """Seeded cycling without replacement plus early and EOS anchors, no padding."""
    rng=random.Random(seed);queues={};schedule=[]
    for step in range(1,updates+1):
        obj=histories[(step-1)%len(histories)];tid=obj['task_id'];n=len(obj['teacher_answer']['answer_ids'])
        positions={0,n-1};wanted=min(width,n)
        while len(positions)<wanted:
            if not queues.get(tid):queues[tid]=list(range(n));rng.shuffle(queues[tid])
            positions.add(queues[tid].pop())
        positions=sorted(positions)
        schedule.append({'step':step,'segments':[{'task_id':tid,'positions':positions,'anchor':'broad_answer_coverage'}],'predictions':len(positions)})
    return schedule

def span_audit(tokenizer,obj):
    ids=obj['teacher_answer']['answer_ids'];text=tokenizer.decode(ids,skip_special_tokens=False);rows=[];start=0
    fence=text.find('```');second=text.find('```',fence+3) if fence>=0 else -1
    for i,token in enumerate(ids):
        end=len(tokenizer.decode(ids[:i+1],skip_special_tokens=False));piece=text[start:end];line_start=text.rfind('\n',0,start)+1;line_end=text.find('\n',start);line=text[line_start:line_end if line_end>=0 else len(text)].strip()
        if token in (151643,151645):kind='stopping'
        elif fence<0 or start<fence or (second>=0 and start>=second+3):kind='explanation_or_prose'
        elif line.startswith(('def ','async def ','class ')):kind='interface_or_signature'
        elif line.startswith('return') or ' return ' in line:kind='return_behavior'
        elif '```' in line:kind='code_fence'
        else:kind='code_body'
        rows.append({'position':i,'token_id':token,'char_start':start,'char_end':end,'text':piece,'span_category':kind,'early_first16':i<16});start=end
    return {'task_id':obj['task_id'],'source_written_answer':text,'rule':'Visible token/line-based descriptive categories; not a correctness oracle. Introductory prose remains supervised.','positions':rows}

def run(c):
    d=read(ROOT/DECLARATION);recipe=d['memorization'];root=c['root'];guard=c['guard'];publish=c['publish'];t=c['telemetry']
    if sha(ROOT/DECLARATION)!=c['spec']['declaration_sha256']:raise ValueError('Declaration differs')
    histories=selected_histories(ROOT,d);lookup={o['task_id']:o for o in histories}
    source,b=initialize_backends(c);mapper=AffineMapper(source,b)
    if sha(ROOT/d['selected_checkpoint'])!=d['selected_checkpoint_sha256']:raise ValueError('Starting mapper differs')
    mapper.load_state_dict(torch.load(ROOT/d['selected_checkpoint'],map_location='cpu',weights_only=True)['state_dict'])
    runtime=TrainingRuntime(source,b,mapper,DiskLRU(root/'cache_lru'),guard)
    optimizer=torch.optim.AdamW(mapper.parameters(),lr=recipe['optimizer']['lr'],weight_decay=0)
    schedule=coverage_schedule(histories,recipe['seed'],recipe['maximum_updates'],recipe['positions_per_update']);write(root/'schedule.json',schedule)
    audits=[span_audit(b.tokenizer,o) for o in histories];write(root/'answer_span_audit.json',audits)
    identity={**c['identity'],'scope':'SEEN-TRAINING-CASE only','starting_checkpoint_sha256':d['selected_checkpoint_sha256'],'training_task_ids':list(lookup),'reference':'saved source-written continuation; native receiver distributions','schedule_sha256':digest(schedule),'language_models_frozen':True,'family':'unchanged affine','hidden_tests_are_optimizer_inputs':False}
    write(root/'training_identity.json',identity);steps=[];curve=[];checkpoints={};seen=collections.defaultdict(set);exposure=collections.Counter();streak=0;last_step=0;started=time.time();deadline=min(started+recipe['maximum_training_seconds'],c['spec']['deadline_epoch']-900)
    def save_state(step):
        p=root/f'mapper_step_{step:04d}.pt'
        if not p.exists():atomic_tensor(p,{'state_dict':{k:v.detach().cpu() for k,v in mapper.state_dict().items()},'step':step,'identity_sha256':digest(identity),'optimizer_resume_supported':False})
        checkpoints[str(p.relative_to(ROOT))]=sha(p);write(root/'checkpoint_manifest.json',{'files':checkpoints,'policy':'Immutable mapper-only checkpoints; selected96 starting point; no optimizer resume.'});return sha(p)
    def generated(obj,arm,step,cache,seconds):
        folder=root/'evaluations'/f'step_{step:04d}'/safe_id(obj['task_id']);folder.mkdir(parents=True,exist_ok=True)
        row=answer_instrumented(b,obj['source_history'],cache,obj['task_id'],'answer_small',4096,folder/arm,t,guard,publish)
        row.update(condition=arm,step=step,scope='SEEN-TRAINING-CASE',historical_cache_preparation_seconds=seconds,source_history_sha256=d['four_training_histories'][list(lookup).index(obj['task_id'])]['source_history_sha256'])
        write(folder/(arm+'.json'),row)
    def evaluate(step):
        nonlocal streak
        cp_hash=save_state(step);rows=[];publish(stage='seen_case_checkpoint',step=step)
        for obj in histories:
            guard();tid=obj['task_id'];t.sample(task_id=tid,stage='seen_case_teacher_forcing',update=step)
            before=time.time();sp,tp=runtime.pair(obj);prep=time.time()-before
            with torch.no_grad():
                positions=list(range(len(obj['teacher_answer']['answer_ids'])))
                losses=runtime.kl(obj,positions,'natural_handoff_boundary',sp,tp,require_grad=False)
                rows.append({'task_id':tid,'positions':positions,'per_position_kl':losses.cpu().flatten().tolist(),'mean_kl':float(losses.mean()),'dense_predictions':len(positions)});del losses
                mapped=mapper(sp);del sp
                cache=CacheInjector.create(mapped,clone=False);del mapped
                generated(obj,'M_seen',step,cache,prep);del cache
                if step==0:
                    generated(obj,'D_seen',step,CacheInjector.create(tp,clone=True),prep)
            del tp;gc.collect();torch.cuda.empty_cache()
        result={'step':step,'scope':'SEEN-TRAINING-CASE','rows':rows,'mean_task_kl':sum(r['mean_kl'] for r in rows)/4,'checkpoint_sha256':cp_hash,'gradient_predictions_so_far':sum(r['gradient_predictions'] for r in steps),'unique_task_positions_so_far':sum(map(len,seen.values()))}
        if not math.isfinite(result['mean_task_kl']):raise FloatingPointError('Nonfinite seen loss')
        curve.append(result);write(root/'seen_training_curve.json',curve)
        streak=streak+1 if result['mean_task_kl']<=.02 else 0
    evaluate(0)
    stop='maximum_updates'
    def training_guard():
        guard()
        if time.time()>=deadline:raise TimeoutError('Declared four-history training allowance')
    runtime.guard=training_guard
    try:
        for item in schedule:
            training_guard();publish(stage='four_history_optimization',step=item['step']);before=time.time()
            result=runtime.train_update(item,lookup,'natural_handoff_boundary',optimizer);last_step=item['step']
            tid=item['segments'][0]['task_id'];positions=item['segments'][0]['positions'];seen[tid].update(positions);exposure[tid]+=len(positions)
            steps.append({'step':last_step,'task_id':tid,'positions':positions,**result,'wall_seconds':time.time()-before});write(root/'training_steps.json',steps)
            write(root/'coverage.json',{'scored_positions':sum(exposure.values()),'unique_task_positions':sum(map(len,seen.values())),'per_task':{o['task_id']:{'scored_positions':exposure[o['task_id']],'unique_positions':sorted(seen[o['task_id']]),'available_positions':len(o['teacher_answer']['answer_ids'])} for o in histories},'span_audit':'answer_span_audit.json'})
            # Keep only a rolling durable mapper in addition to declared evaluations.
            latest=root/'mapper_latest.pt';tmp=root/'mapper_latest.tmp'
            with tmp.open('wb') as f:
                torch.save({'state_dict':{k:v.detach().cpu() for k,v in mapper.state_dict().items()},'step':last_step},f);f.flush();os.fsync(f.fileno())
            os.replace(tmp,latest);write(root/'latest_checkpoint.json',{'step':last_step,'sha256':sha(latest)})
            t.sample(stage='four_history_update_committed',update=last_step)
            if last_step in recipe['checkpoint_updates']:
                runtime.guard=guard;evaluate(last_step);runtime.guard=training_guard
                if streak>=2:stop='two_successive_dense_seen_kl_at_most_0.02';break
    except TimeoutError as exc:stop='declared_time_or_controlled_stop';write(root/'training_stop.json',{'error':str(exc),'last_committed_step':last_step})
    finally:runtime.guard=guard;optimizer.zero_grad(set_to_none=True)
    if curve[-1]['step']!=last_step and time.time()<c['spec']['deadline_epoch']-600:evaluate(last_step)
    else:save_state(last_step)
    # Only now load tests. Hidden scores do not affect gradients, schedule or stopping.
    from gearshift.coding_sandbox import extract,score
    private=read(ROOT/'data/coding_pilot_v1/private/post_progress01_four_training.json')
    if set(private)!=set(lookup):raise ValueError('Scorer cohort differs')
    for p in sorted((root/'evaluations').glob('step_*/*/*_seen.json')):
        row=read(p);row['code']=extract(row['answer_text']);row['score']=score(row['code'],private[row['task_id']],guard)
        if row['score']['category'] in ('sandbox_unavailable','missing_tests'):raise RuntimeError('Scorer unavailable')
        write(p,row)
    write(root/'cache_working_set.json',runtime.lru.events)
    inventory={}
    for cache_file in sorted((root/'cache_lru').glob('*.pt')):
        guard();inventory[str(cache_file.relative_to(ROOT))]={'bytes':cache_file.stat().st_size,'sha256':sha(cache_file),'retained':False,'regenerate':'TrainingRuntime.pair on exact declared histories with full-prefix512-token schedule; disposable cache.'}
    write(root/'cache_inventory.json',{'files':inventory})
    write(root/'complete.json',{'identity_sha256':digest(c['identity']),'training_identity_sha256':digest(identity),'steps':last_step,'stop_reason':stop,'wall_seconds':time.time()-started,'scored_positions':sum(exposure.values()),'unique_task_positions':sum(map(len,seen.values())),'task_ids':list(lookup),'scope':'SEEN-TRAINING-CASE; not generalization','hidden_tests_loaded_only_after_optimization':True,'confirmation_used':False});finish_worker(c,'complete',stage='four_history_memorization_complete',steps=last_step)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-spec');a=p.parse_args();c=None
    try:c=worker_context(ROOT,a.worker_spec);run(c)
    except BaseException as exc:
        if c:
            from scripts.coding_recovery_train import preserve_latest
            preserve_latest(c['root'])
            c['telemetry'].failure(exc);write(c['root']/'failure.json',{'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()});finish_worker(c,'failed',error=str(exc))
        raise
