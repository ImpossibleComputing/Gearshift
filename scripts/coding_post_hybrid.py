#!/usr/bin/env python3
"""Frozen prompt-preservation H, prompt-only P and matched D; reuse valid M."""
import argparse,gc,shutil,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.coding_control import write,sha,digest
from gearshift.coding_training import read,finish_worker,safe_id
from gearshift.coding_recovery import worker_context,initialize_backends
from gearshift.coding_recovery_answer import answer_instrumented
from gearshift.coding_gradients import AffineMapper
from gearshift.coding_post_progress import DECLARATION,splice
from gearshift.coding_inference import sync,logit_metrics
from gearshift.coding_sandbox import extract,score
from gearshift.core import CacheExtractor,CacheInjector
ROOT=Path(__file__).resolve().parents[1]

def run(c):
    spec=c['spec'];root=c['root'];guard=c['guard'];publish=c['publish'];t=c['telemetry'];d=read(ROOT/DECLARATION)
    if sha(ROOT/DECLARATION)!=spec['declaration_sha256']:raise ValueError('Declaration differs')
    tasks={r['task_id']:r for r in read(ROOT/'data/coding_pilot_v1/visible/development.json')}
    if len(spec['task_ids'])!=40 or set(spec['task_ids'])!=set(tasks):raise ValueError('Exactly40 inspected tasks')
    originals={}
    for p in ROOT.glob('results/coding_pilot_v1/recovery_dev_20260917_*/dev*/tasks/*/complete.json'):
        r=read(p);tid=r['task_id']
        if tid in originals:raise ValueError('Duplicate original outcome')
        if any(sha(p.parent/name)!=h for name,h in r['files'].items()):raise ValueError('Original transaction changed')
        originals[tid]=p.parent
    if set(originals)!=set(tasks):raise ValueError('Missing old M/D outcomes')
    if not read(ROOT/'evidence/coding_pilot_v1/sandbox_gate.json')['passed']:raise ValueError('Sandbox unavailable')
    source,b=initialize_backends(c);mapper=AffineMapper(source,b);mapper.load_state_dict(torch.load(ROOT/d['selected_checkpoint'],map_location='cpu',weights_only=True)['state_dict']);mapper.requires_grad_(False)
    if sha(ROOT/d['selected_checkpoint'])!=d['selected_checkpoint_sha256']:raise ValueError('Mapper changed')
    # Trusted scoring data never appear in model prompts or cache construction.
    private=read(ROOT/'data/coding_pilot_v1/private/development.json');completed={};controls=[]
    def grade(folder,arm,row):
        row['code']=extract(row['answer_text']);write(folder/(arm+'.json'),row)
        row['score']=score(row['code'],private[row['task_id']],guard)
        if row['score']['category'] in ('sandbox_unavailable','missing_tests'):raise RuntimeError('Scorer unavailable')
        write(folder/(arm+'.json'),row);return row
    for i,tid in enumerate(spec['task_ids']):
        guard();publish(stage='frozen_prompt_comparison',task_id=tid,completed_tasks=i,total_tasks=40)
        folder=root/'tasks'/safe_id(tid);folder.mkdir(parents=True,exist_ok=False);old=originals[tid]
        h=read(old/'source_history.json');np=len(h['prompt_ids']);n=len(h['prefix_ids'])
        if h['prompt_ids']!=tasks[tid]['prompt_ids'] or h['prefix_ids'][:np]!=h['prompt_ids']:raise ValueError('Prompt/history boundary differs')
        shutil.copyfile(old/'source_history.json',folder/'source_history.json')
        for src,dst in [('C','M_original'),('A_original','A_original'),('B_original','B_original'),('D_matched','D_original')]:shutil.copyfile(old/(src+'.json'),folder/(dst+'.json'))
        if read(folder/'M_original.json')['checkpoint_sha256']!=d['selected_checkpoint_sha256']:raise ValueError('M checkpoint differs')
        t.reset(tid);t.sample(task_id=tid,sequence_length=n,stage='hybrid_cache_reconstruction')
        with torch.no_grad():
            sync();start=time.monotonic();out=source.prefill_chunked(h['prefix_ids']);sp=CacheExtractor.tensors(out.past_key_values);del out;sync();reconstruction=time.monotonic()-start
            start=time.monotonic();mapped=mapper(sp);sync();mapping=time.monotonic()-start;del sp
            # Replay the exact original M draw for matched timing; reuse its score only
            # after token-for-token identity, never choose a favorable redraw.
            mcache=CacheInjector.create(mapped,clone=True)
            mrow=answer_instrumented(b,h,mcache,tid,'answer_small',4096,folder/'M',t,guard,publish);del mcache
            previous=read(folder/'M_original.json')
            mrow.update(condition='M',checkpoint_sha256=d['selected_checkpoint_sha256'],source_history_sha256=sha(folder/'source_history.json'),mapping_seconds=mapping,native_prefill_seconds=0.,historical_receiver_prefill_tokens=0,source_cache_reconstruction_seconds=reconstruction,source_reasoning_seconds=h['reasoning_seconds'],source_inclusive_seconds=h['reasoning_seconds']+mapping+mrow['answer_seconds'],matches_original_tokens=mrow['answer_ids']==previous['answer_ids'])
            write(folder/'M.json',mrow)
            if not mrow['matches_original_tokens']:raise RuntimeError('Timed M replay changed tokens; preserve and inspect before baseline reuse')
            mrow.update(code=previous['code'],score=previous['score'],score_reused_from_sha256=sha(folder/'M_original.json'));write(folder/'M.json',mrow)
            sync();start=time.monotonic();out=b.prefill_chunked(h['prompt_ids']);native_prompt=CacheExtractor.tensors(out.past_key_values);del out;sync();prompt_seconds=time.monotonic()-start
            prompt_cpu=tuple(tuple(x.cpu().clone() for x in pair) for pair in native_prompt)
            start=time.monotonic();hybrid=splice(native_prompt,mapped,np);sync();splice_seconds=time.monotonic()-start
            boundary={'task_id':tid,'prompt_tokens':np,'reasoning_suffix_tokens':n-np,'full_prefix_tokens':n,'prompt_ids':h['prompt_ids'],'reasoning_suffix_ids':h['prefix_ids'][np:],'suffix_absolute_positions':[np,n-1],'bridge_ids':h['bridge_ids'],'no_token_duplication_or_skip':h['prompt_ids']+h['prefix_ids'][np:]==h['prefix_ids'],'native_prompt_seconds':prompt_seconds,'mapped_suffix_keeps_full_history_RoPE':True,'new_answer_states_depend_on_history':True}
            assert hybrid[0][0].shape[-2]==n
            write(folder/'boundary.json',boundary)
            cache=CacheInjector.create(hybrid,clone=False);del hybrid,mapped,native_prompt
        row=answer_instrumented(b,h,cache,tid,'answer_small',4096,folder/'H',t,guard,publish);del cache
        row.update(condition='H',checkpoint_sha256=d['selected_checkpoint_sha256'],source_history_sha256=sha(folder/'source_history.json'),mapping_seconds=mapping,splice_seconds=splice_seconds,native_prefill_seconds=prompt_seconds,historical_receiver_prefill_tokens=np,source_cache_reconstruction_seconds=reconstruction,source_reasoning_seconds=h['reasoning_seconds'],source_inclusive_seconds=h['reasoning_seconds']+mapping+splice_seconds+prompt_seconds+row['answer_seconds'])
        grade(folder,'H',row);gc.collect();torch.cuda.empty_cache();guard()
        with torch.no_grad():
            sync();start=time.monotonic();out=b.prefill_chunked(h['prefix_ids']);sync();prefill=time.monotonic()-start;dp=CacheExtractor.tensors(out.past_key_values);del out
            prefix=tuple(tuple(x[...,:np,:] for x in pair) for pair in dp);nn=splice(prefix,dp,np)
            # Native/native splicing has identical tensors and absolute positions.
            exact=all(torch.equal(a,z) for p,q in zip(dp,nn) for a,z in zip(p,q))
            a=b.forward(h['bridge_ids'],CacheInjector.create(dp,clone=True)).logits.detach().clone()
            z=b.forward(h['bridge_ids'],CacheInjector.create(nn,clone=True)).logits.detach().clone();metric=logit_metrics(a,z)
            if not exact or metric['max_abs']!=0:raise RuntimeError('Native/native splice changed continuation')
            actual_prompt=tuple(tuple(x.to(b.device) for x in pair) for pair in prompt_cpu)
            actual_nn=splice(actual_prompt,dp,np)
            actual_logits=b.forward(h['bridge_ids'],CacheInjector.create(actual_nn,clone=True)).logits
            actual_prompt_metric=logit_metrics(a,actual_logits)
            prefix_error=max(float((x-y).abs().max()) for pair,q in zip(actual_prompt,prefix) for x,y in zip(pair,q))
            del actual_prompt,actual_nn,actual_logits,prompt_cpu
            controls.append({'task_id':tid,'native_native_tensor_exact':exact,'bridge_logits':metric,'actual_partial_prefill_prefix_max_abs':prefix_error,'actual_partial_prefill_splice_bridge_logits':actual_prompt_metric,'passed':True,'scope':'Whole native history split and rejoined at exact prompt boundary; tests cache surgery without changing prefill segmentation.'});write(root/'native_native_splice_controls.json',controls)
            cache=CacheInjector.create(dp,clone=False);del dp,prefix,nn,a,z
        row=answer_instrumented(b,h,cache,tid,'answer_small',4096,folder/'D',t,guard,publish);del cache
        same=row['answer_ids']==read(folder/'D_original.json')['answer_ids'];row.update(condition='D',source_history_sha256=sha(folder/'source_history.json'),mapping_seconds=0.,native_prefill_seconds=prefill,historical_receiver_prefill_tokens=n,source_reasoning_seconds=h['reasoning_seconds'],matches_original_tokens=same,source_inclusive_seconds=h['reasoning_seconds']+prefill+row['answer_seconds']);grade(folder,'D',row)
        if not same:
            write(folder/'reuse_invalid.json',{'reason':'Matched native replay changed token sequence; historical M must not be silently reused.'})
            raise RuntimeError('Matched D changed; inspect runtime and rerun M under new identity')
        gc.collect();torch.cuda.empty_cache();guard()
        # Valid pinned tokenizer non-thinking template, original visible task unchanged.
        ids=b.tokenizer.apply_chat_template([{'role':'user','content':tasks[tid]['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        text=b.tokenizer.decode(ids,skip_special_tokens=False)
        if '<think>\n\n</think>' not in text:raise ValueError('Unexpected non-thinking template')
        ph={'prefix_ids':ids[:-1],'bridge_ids':ids[-1:]}
        write(folder/'prompt_only_template.json',{'task_id':tid,'prompt_ids':ids,'rendered_template':text,'enable_thinking':False,'separate_thinking_stage_requested':False,'prefill_prefix_ids':ph['prefix_ids'],'final_template_bridge_ids':ph['bridge_ids']})
        with torch.no_grad():
            sync();start=time.monotonic();out=b.prefill_chunked(ph['prefix_ids']);sync();prefill=time.monotonic()-start;cache=out.past_key_values;del out
        row=answer_instrumented(b,ph,cache,tid,'answer_small',4096,folder/'P',t,guard,publish);del cache
        row.update(condition='P',mapping_seconds=0.,native_prefill_seconds=prefill,historical_receiver_prefill_tokens=len(ph['prefix_ids']),source_reasoning_seconds=0.,separate_thinking_stage_requested=False,thinking_open_token_in_answer=(b.tokenizer.convert_tokens_to_ids('<think>') in row['answer_ids']),source_inclusive_seconds=prefill+row['answer_seconds']);grade(folder,'P',row)
        files={p.name:sha(p) for p in folder.glob('*.json')};receipt={'task_id':tid,'identity_sha256':digest(c['identity']),'files':files,'source_history_sha256':sha(folder/'source_history.json'),'M_reuse_validated_by_D_token_match':same};write(folder/'complete.json',receipt);completed[tid]=sha(folder/'complete.json');write(root/'task_progress.json',{'completed':completed,'total':40});gc.collect();torch.cuda.empty_cache()
    write(root/'complete.json',{'identity_sha256':digest(c['identity']),'tasks':completed,'checkpoint_sha256':d['selected_checkpoint_sha256'],'confirmation_used':False});finish_worker(c,'complete',stage='frozen_prompt_comparison_complete',completed_tasks=40)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-spec');a=p.parse_args();c=None
    try:c=worker_context(ROOT,a.worker_spec);run(c)
    except BaseException as exc:
        if c:c['telemetry'].failure(exc);write(c['root']/'failure.json',{'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()});finish_worker(c,'failed',error=str(exc))
        raise
