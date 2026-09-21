#!/usr/bin/env python3
"""One controlled worker; preflight produces no sampled program-quality scores."""
import argparse,copy,gc,json,math,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.coding_control import write,sha,digest
from gearshift.coding_coverage import DECLARATION,ARMS,grouped
from gearshift.coding_coverage_runtime import CoverageRuntime,paired_update,validation
from gearshift.coding_recovery import worker_context,initialize_backends
from gearshift.coding_training import read,finish_worker,atomic_tensor,boundary_case
from gearshift.coding_gradients import AffineMapper,continuation
from gearshift.coding_inference import sync,logit_metrics
from coding_partial_corpus import load_corpus
ROOT=Path(__file__).resolve().parents[1]


def setup(c,training=True):
    d=read(ROOT/DECLARATION)
    if sha(ROOT/DECLARATION)!=c['spec']['declaration_sha256']:raise ValueError('Declaration changed')
    for rel,expected in d['unchanged_source_files'].items():
        if sha(ROOT/rel)!=expected:raise ValueError('Frozen numerical/scoring implementation changed: '+rel)
    if sha(ROOT/d['corpus_manifest'])!=d['corpus_manifest_sha256']:raise ValueError('Corpus changed')
    histories,_,manifest=load_corpus(ROOT,ROOT/d['corpus_manifest'],features=False)
    train=[h for h in histories if h['split']=='training'];val=[h for h in histories if h['split']=='validation']
    if [h['task_id'] for h in train]!=d['training_task_ids'] or [h['task_id'] for h in val]!=d['validation_task_ids']:raise ValueError('Split differs')
    for name in ['schedules','panels','answer_seeds']:
        if sha(ROOT/d[name+'_path'])!=d[name+'_sha256']:raise ValueError(name+' changed')
    schedules=read(ROOT/d['schedules_path']);panels=read(ROOT/d['panels_path'])
    source,b=initialize_backends(c)
    if sha(ROOT/d['selected_checkpoint'])!=d['selected_checkpoint_sha256']:raise ValueError('Starting weights differ')
    saved=torch.load(ROOT/d['selected_checkpoint'],map_location='cpu',weights_only=True)
    if any('optim' in k and k!='optimizer_resume_supported' for k in saved):raise ValueError('Unexpected optimizer state; document before changing reset')
    mappers={};runtimes={};optimizers={}
    for arm in (ARMS if training else ('FIXED',)):
        mapper=AffineMapper(source,b);mapper.load_state_dict(saved['state_dict'],strict=True)
        if any(not torch.equal(v.detach().cpu(),saved['state_dict'][k]) for k,v in mapper.state_dict().items()):raise ValueError('Starting tensor equality failed')
        mappers[arm]=mapper
        if training:
            recipe=d['optimizer'];optimizers[arm]=torch.optim.AdamW(mapper.parameters(),lr=recipe['lr'],betas=tuple(recipe['betas']),eps=recipe['eps'],weight_decay=recipe['weight_decay'],foreach=recipe['foreach'],fused=recipe['fused'])
        runtimes[arm]=CoverageRuntime(source,b,mapper,c['guard'],c['telemetry'])
    del saved;gc.collect()
    if training:
        write(c['root']/'optimizer_reset.json',{'checkpoint_sha256':d['selected_checkpoint_sha256'],'both_starting_tensor_sets_exact':True,
            'optimizer_states_available':False,'fresh_identical_optimizers':True,'configuration':d['optimizer'],'states_initially_empty':all(not o.state for o in optimizers.values())})
        write(c['root']/'training_identity.json',{'runtime_identity_sha256':digest(c['identity']),'declaration_sha256':sha(ROOT/DECLARATION),
            'schedules_sha256':d['schedules_sha256'],'corpus_sha256':d['corpus_manifest_sha256'],'selected_checkpoint_sha256':d['selected_checkpoint_sha256'],
            'teacher_reference':'native receiver distributions on unchanged saved source-written answer continuations','no_cropping':True,'hidden_tests_are_optimizer_inputs':False})
    else:
        write(c['root']/'evaluation_initialization.json',{'evaluation_only':True,'optimizer_created':False,'gradient_updates':0,'initial_mapper_tensor_equality_verified':True})
    return d,train,val,schedules,panels,source,b,mappers,runtimes,optimizers


def save_pair(c,mappers,step):
    files={};root=c['root']
    if (root/'checkpoint_manifest.json').exists():files=read(root/'checkpoint_manifest.json')['files']
    pair={}
    for arm in ARMS:
        path=root/arm/f'mapper_step_{step:04d}.pt'
        if not path.exists():atomic_tensor(path,{'state_dict':{k:v.detach().cpu() for k,v in mappers[arm].state_dict().items()},
            'step':step,'arm':arm,'identity_sha256':digest(c['identity']),'optimizer_resume_supported':False})
        rel=str(path.relative_to(ROOT));files[rel]=sha(path);pair[arm]={'path':rel,'sha256':files[rel],'bytes':path.stat().st_size}
    # Manifest and pair receipt appear only after both durable states exist.
    write(root/'checkpoint_manifest.json',{'files':files,'policy':'Immutable paired mapper states; optimizer resume unsupported.'})
    write(root/'paired_checkpoints'/f'step_{step:04d}.json',{'step':step,'arms':pair,'paired_commit':True})
    return pair


def numerical_shapes(c,runtime,histories):
    from coding_post_numerical import direct
    rows=[]
    for obj in histories:
        c['guard']();c['publish'](stage='coverage_late_position_numerical',task_id=obj['task_id'])
        sp,tp=runtime.pair(obj);n=len(obj['teacher_answer']['answer_ids'])
        positions=sorted(set([0,min(512,n-1),n-1]));ids,_=boundary_case(obj,positions)
        with torch.no_grad():
            mapped=runtime.mapper(sp)
            for kind,pairs in [('native',tp),('mapped',mapped)]:
                versions=[x._version for p in pairs for x in p]
                a=direct(runtime.receiver,pairs,ids,positions);r=direct(runtime.receiver,pairs,ids,positions)
                m=continuation(runtime.receiver.model,pairs,ids,positions,checkpoint_layers=False)
                z=continuation(runtime.receiver.model,pairs,ids,positions,checkpoint_layers=True)
                repeat=logit_metrics(a,r);cross=logit_metrics(a,m);toggle=logit_metrics(m,z)
                passed=cross['max_abs']<=repeat['max_abs'] and toggle['max_abs']==0 and versions==[x._version for p in pairs for x in p]
                rows.append({'task_id':obj['task_id'],'cache_kind':kind,'full_prefix_tokens':len(obj['source_history']['prefix_ids']),
                    'continuation_tokens':len(ids),'positions':positions,'repeat':repeat,'manual_vs_deployed':cross,'checkpoint_toggle':toggle,
                    'passed':passed,'execution_shape':'One full intervening continuation call on both paths; original absolute positions.'})
                write(c['root']/'late_numerical_controls.json',rows)
                del a,r,m,z
                if not passed:raise RuntimeError('New late-position numerical control failed; no tolerance relaxation')
        del sp,tp,mapped,pairs;gc.collect();torch.cuda.empty_cache()
    return rows


def preflight(c,state):
    d,train,val,schedules,panels,source,b,mappers,runtimes,optimizers=state
    lookup={h['task_id']:h for h in train};root=c['root'];started=time.monotonic()
    stress=d['preflight']['stress_updates'];unique=list(dict.fromkeys(x['task_id'] for x in stress))
    numerical=numerical_shapes(c,runtimes['FIXED'],[lookup[tid] for tid in unique])
    representative=[];stress_results=[]
    specifications=[{'kind':'representative','index':i} for i in d['preflight']['representative_schedule_indices']]
    specifications += [{'kind':'stress','index':s['schedule_index'],**s} for s in stress]
    for i,part in enumerate(specifications):
        index=part['index'];items={a:copy.deepcopy(schedules[a][index]) for a in ARMS}
        if part['kind']=='stress':
            tid=part['task_id'];counts=grouped(items['FIXED']);n=len(lookup[tid]['teacher_answer']['answer_ids']);remaining=list(range(n-len(counts[tid]),n))
            for segment in items['ROTATING']['segments']:
                if segment['task_id']==tid:
                    k=len(segment['positions']);segment['positions']=remaining[:k];remaining=remaining[k:]
        c['publish'](stage='coverage_preflight_training',completed_updates=i,total_updates=len(specifications))
        row=paired_update(items,lookup,runtimes,optimizers,ARMS if i%2==0 else ARMS[::-1])
        row.update(specification=part,items=items)
        (representative if part['kind']=='representative' else stress_results).append(row)
        write(root/'preflight_updates.json',{'representative':representative,'stress':stress_results})
        c['telemetry'].sample(stage='preflight_paired_update_committed',update=i+1)
    # Restore original state before timing the fixed validation panels; preflight
    # updates never become a primary starting point and do not select weights.
    saved=torch.load(ROOT/d['selected_checkpoint'],map_location='cpu',weights_only=True)
    mappers['FIXED'].load_state_dict(saved['state_dict']);del saved
    val_result=validation(runtimes['FIXED'],val,panels,c['publish']);write(root/'validation_start.json',val_result)
    result={'representative_updates':representative,'stress_updates':stress_results,'validation_seconds':val_result['wall_seconds'],
        'memory_and_gradient_checks_passed':True,'numerical_controls_passed':all(r['passed'] for r in numerical),
        'wall_seconds':time.monotonic()-started,'sampled_quality_scores_generated':False,'hidden_tests_available':False,
        'preflight_weights_discarded':True,'primary_starting_checkpoint_sha256':d['selected_checkpoint_sha256']}
    write(root/'preflight_result.json',result)
    write(root/'complete.json',{'identity_sha256':digest(c['identity']),'preflight_result_sha256':sha(root/'preflight_result.json'),
        'confirmation_used':False,'program_quality_generation':False})
    finish_worker(c,'complete',stage='coverage_preflight_complete')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker-spec');a=p.parse_args();c=None
    try:
        c=worker_context(ROOT,a.worker_spec);state=setup(c,training=c['spec']['stage']!='coverage_evaluation_recovery')
        if c['spec']['stage']=='coverage_preflight':preflight(c,state)
        elif c['spec']['stage']=='coverage_experiment':
            from coding_coverage_experiment import experiment
            experiment(c,state)
        elif c['spec']['stage']=='coverage_evaluation_recovery':
            from coding_coverage_recover import evaluate_recovered
            evaluate_recovered(c,state)
        else:raise ValueError('Unknown coverage worker')
    except BaseException as exc:
        if c:
            c['telemetry'].failure(exc);write(c['root']/'failure.json',{'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
            finish_worker(c,'failed',error=str(exc))
        raise

if __name__=='__main__':main()
