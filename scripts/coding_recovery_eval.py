#!/usr/bin/env python3
"""Evaluate preselected trained and initial mappers with matched development D.

Only development hidden tests are loaded, by the trusted sandbox scorer. Source
trajectories are copied exactly; source cache is fully reconstructed in512-token
chunks, the same call schedule used by functional training. No source redraws.
"""
import argparse,gc,shutil,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.coding_control import write,digest,sha
from gearshift.coding_training import read,finish_worker,safe_id
from gearshift.coding_recovery import worker_context,initialize_backends
from gearshift.coding_recovery_answer import answer_instrumented
from gearshift.coding_gradients import AffineMapper
from gearshift.coding_inference import sync,logit_metrics
from gearshift.coding_sandbox import extract,score
from gearshift.coding_reuse import verify_completed
from gearshift.core import CacheExtractor,CacheInjector
ROOT=Path(__file__).resolve().parents[1]

def run(c):
    spec=c['spec'];root=c['root'];guard=c['guard'];t=c['telemetry'];publish=c['publish']
    if spec['stage']!='exploratory_development':raise ValueError('Development only')
    tasks={x['task_id']:x for x in read(ROOT/'data/coding_pilot_v1/visible/development.json')}
    if not set(spec['task_ids'])<=set(tasks):raise ValueError('Non-development tasks forbidden')
    selection=read(ROOT/spec['selection'])
    if sha(ROOT/spec['selection'])!=spec['selection_sha256'] or selection['development_used_for_selection'] or selection['confirmation_used_for_selection']:raise ValueError('Validation selection changed')
    cp=ROOT/selection['checkpoint']['path'];init=ROOT/spec['initialization']
    if sha(cp)!=selection['checkpoint']['sha256'] or sha(init)!=selection['initialization_sha256']:raise ValueError('Mapper bytes differ')
    states={'C':torch.load(cp,map_location='cpu',weights_only=True)['state_dict'],'C_initial':torch.load(init,map_location='cpu',weights_only=True)['state_dict']}
    sandbox=read(ROOT/'evidence/coding_pilot_v1/sandbox_gate.json')
    if not sandbox['passed']:raise RuntimeError('Sandbox unavailable')
    write(root/'sandbox_input.json',{'sha256':sha(ROOT/'evidence/coding_pilot_v1/sandbox_gate.json')})
    source,receiver=initialize_backends(c);mapper=AffineMapper(source,receiver);mapper.requires_grad_(False)
    base=ROOT/spec['baseline_root'];baseid=read(base/'identity.json');private=read(ROOT/'data/coding_pilot_v1/private/development.json')
    completed={};replays=[]
    def grade(folder,name,row):
        row['code']=extract(row['answer_text']);write(folder/(name+'.json'),row)
        row['score']=score(row['code'],private[row['task_id']],guard);write(folder/(name+'.json'),row)
        if row['score']['category'] in ['sandbox_unavailable','missing_tests']:raise RuntimeError('Trusted evaluation unavailable')
        return row
    for i,tid in enumerate(spec['task_ids']):
        guard();folder=root/'tasks'/safe_id(tid);folder.mkdir(parents=True,exist_ok=False);parent=base/'tasks'/safe_id(tid)
        verify_completed(parent,baseid,tid);h=read(parent/'source_history.json')
        if h['prompt_ids']!=tasks[tid]['prompt_ids']:raise ValueError('Saved prompt changed')
        shutil.copyfile(parent/'source_history.json',folder/'source_history.json')
        original={arm:read(parent/(arm+'.json')) for arm in ['A','B','D']}
        for arm in ['A','B','D']:shutil.copyfile(parent/(arm+'.json'),folder/(arm+'_original.json'))
        publish(stage='development',task_id=tid,completed_tasks=i,total_tasks=len(spec['task_ids']),segment='source_cache_reconstruction')
        t.reset('development_task_'+tid);t.sample(task_id=tid,sequence_length=len(h['prefix_ids']),stage='source_cache_reconstruction')
        sync();start=time.monotonic();out=source.prefill_chunked(h['prefix_ids']);sync();reconstruction=time.monotonic()-start
        pairs=CacheExtractor.tensors(out.past_key_values);del out;rows={}
        for arm in ['C','C_initial']:
            guard();publish(segment=arm);mapper.load_state_dict(states[arm],strict=True)
            before_tokens=receiver.input_token_count;sync();start=time.monotonic()
            with torch.no_grad():mapped=mapper(pairs)
            sync();mapping=time.monotonic()-start
            if receiver.input_token_count!=before_tokens:raise AssertionError('Mapped arm performed historical target prefill')
            if any(not torch.isfinite(x).all() for pair in mapped for x in pair):raise FloatingPointError('Nonfinite mapped cache')
            if i==0:
                with torch.no_grad():
                    a=receiver.forward(h['bridge_ids'],CacheInjector.create(mapped,clone=True));alogits=a.logits.detach().clone();del a
                    b=receiver.forward(h['bridge_ids'],CacheInjector.create(mapped,clone=True));metric=logit_metrics(alogits,b.logits);del alogits,b
                if metric['max_abs']!=0 or not metric['top1_equal']:raise RuntimeError('Mapped cache reinjection changed logits')
                write(root/(arm+'_injection_control.json'),{'task_id':tid,'metric':metric,'passed':True,'full_prefix_tokens':len(h['prefix_ids'])})
            cache=CacheInjector.create(mapped,clone=False);del mapped
            row=answer_instrumented(receiver,h,cache,tid,'answer_small',4096,folder/arm,t,guard,publish);del cache
            row.update(condition=arm,source_history_sha256=sha(folder/'source_history.json'),checkpoint_sha256=sha(cp if arm=='C' else init),
                mapping_seconds=mapping,native_prefill_seconds=0.,historical_receiver_prefill_tokens=0,
                source_cache_reconstruction_seconds=reconstruction,source_reasoning_seconds=h['reasoning_seconds'],
                source_inclusive_seconds=h['reasoning_seconds']+mapping+row['answer_seconds'],
                actual_replay_execution_seconds=reconstruction+mapping+row['answer_seconds'],
                source_cache_schedule='complete saved prefix in512-token chunks; no sampling',
                online_estimate_limitation='Uses archived source reasoning cost; cache reconstructed offline, not measured live online source continuation.')
            rows[arm]=grade(folder,arm,row);gc.collect();torch.cuda.empty_cache()
        del pairs;gc.collect();torch.cuda.empty_cache();guard();publish(segment='D_matched')
        sync();start=time.monotonic();out=receiver.prefill_chunked(h['prefix_ids']);sync();prefill=time.monotonic()-start
        cache=out.past_key_values;del out
        row=answer_instrumented(receiver,h,cache,tid,'answer_small',4096,folder/'D_matched',t,guard,publish);del cache
        same=row['answer_ids']==original['D']['answer_ids']
        row.update(condition='D_matched',source_history_sha256=sha(folder/'source_history.json'),mapping_seconds=0.,native_prefill_seconds=prefill,
            historical_receiver_prefill_tokens=len(h['prefix_ids']),source_reasoning_seconds=h['reasoning_seconds'],
            source_inclusive_seconds=h['reasoning_seconds']+prefill+row['answer_seconds'],matches_original_D_tokens=same,
            original_D_sha256=sha(parent/'D.json'))
        rows['D_matched']=grade(folder,'D_matched',row)
        replay={'task_id':tid,'D_tokens_match_original':same,'original_pass':original['D']['score']['passed'],'matched_pass':row['score']['passed']};replays.append(replay);write(root/'D_replay_checks.json',replays)
        files={p.name:sha(p) for p in folder.glob('*.json')}
        receipt={'task_id':tid,'identity_sha256':digest(c['identity']),'files':files,
            'pass':{'A':original['A']['score']['passed'],'B':original['B']['score']['passed'],'D_original':original['D']['score']['passed'],**{a:r['score']['passed'] for a,r in rows.items()}},
            'source_history_sha256':sha(folder/'source_history.json'),'D_tokens_match_original':same,'coverage':'all requested arms complete; no infrastructure failures scored as quality failures'}
        write(folder/'complete.json',receipt);completed[tid]=sha(folder/'complete.json');write(root/'task_progress.json',{'tasks':completed,'complete':len(completed),'expected':len(spec['task_ids'])})
        gc.collect();torch.cuda.empty_cache();t.sample(stage='task_released')
    write(root/'complete.json',{'identity_sha256':digest(c['identity']),'stage':'exploratory_development','tasks':completed,'task_count':len(completed),'selection_sha256':spec['selection_sha256'],
        'confirmation_run':False,'original_D_exact_matches':sum(r['D_tokens_match_original'] for r in replays)})
    finish_worker(c,'complete',stage='development_complete',completed_tasks=len(completed))

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker-spec');a=p.parse_args();c=None
    try:c=worker_context(ROOT,a.worker_spec);run(c)
    except BaseException as exc:
        if c:
            c['telemetry'].failure(exc);write(c['root']/'failure.json',{'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()});finish_worker(c,'failed',exception=type(exc).__name__,error=str(exc))
        raise
if __name__=='__main__':main()
