#!/usr/bin/env python3
"""Matched fresh free-running generations; private tests are not imported here."""
import gc,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.coding_control import write,sha,digest
from gearshift.coding_training import safe_id
from gearshift.coding_recovery_answer import answer_instrumented
from gearshift.coding_post_progress import splice
from gearshift.coding_inference import sync,logit_metrics
from gearshift.core import CacheExtractor,CacheInjector
ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def generate_suite(c,source,b,mapper,histories,visible,seeds,checkpoints,scope):
    root=c['root']/scope;root.mkdir(parents=True,exist_ok=True);controls=[];finished=[]
    mapper.requires_grad_(False)
    for index,obj in enumerate(histories):
        tid=obj['task_id'];h=obj['source_history'];folder=root/safe_id(tid);folder.mkdir(parents=True,exist_ok=False)
        if h['prompt_ids']!=visible[tid]['prompt_ids']:raise ValueError('Visible prompt/history mismatch')
        np=len(h['prompt_ids']);n=len(h['prefix_ids'])
        if h['prefix_ids'][:np]!=h['prompt_ids']:raise ValueError('History does not start with original prompt')
        write(folder/'source_history.json',h)
        c['guard']();c['publish'](stage='coverage_free_generation',scope=scope,task_id=tid,completed_tasks=index,total_tasks=len(histories))
        c['telemetry'].reset(scope+'/'+tid)
        sync();start=time.monotonic();out=source.prefill_chunked(h['prefix_ids']);sp=CacheExtractor.tensors(out.past_key_values);del out;sync();source_prep=time.monotonic()-start
        start=time.monotonic();out=b.prefill_chunked(h['prefix_ids']);tp=CacheExtractor.tensors(out.past_key_values);del out;sync();native_prep=time.monotonic()-start
        start=time.monotonic();out=b.prefill_chunked(h['prompt_ids']);prompt=CacheExtractor.tensors(out.past_key_values);del out;sync();prompt_prep=time.monotonic()-start
        sliced=tuple(tuple(x[...,:np,:] for x in pair) for pair in tp);nn=splice(sliced,tp,np)
        exact=all(torch.equal(x,y) for p,q in zip(nn,tp) for x,y in zip(p,q))
        a=b.forward(h['bridge_ids'],CacheInjector.create(tp,clone=True)).logits.detach().clone()
        z=b.forward(h['bridge_ids'],CacheInjector.create(nn,clone=True)).logits.detach().clone();metric=logit_metrics(a,z)
        partial=splice(prompt,tp,np);p=b.forward(h['bridge_ids'],CacheInjector.create(partial,clone=True)).logits.detach().clone()
        partial_metric=logit_metrics(a,p);prefix_diff=max(float((x-y).abs().max()) for pp,qq in zip(prompt,sliced) for x,y in zip(pp,qq))
        if not exact or metric['max_abs']!=0:raise RuntimeError('Native/native splice changed whole-history execution')
        controls.append({'task_id':tid,'whole_native_split_tensor_exact':exact,'whole_native_splice_bridge':metric,
            'actual_partial_prompt_cache_max_abs':prefix_diff,'actual_partial_prompt_splice_bridge':partial_metric,
            'passed':True,'scope':'Native/native exact whole-prefill split/rejoin. Actual partial-prompt prefill rounding reported separately; no claim of later sampled equality.'})
        write(root/'native_splice_controls.json',controls)
        del sliced,nn,a,z,p,partial;gc.collect()
        write(folder/'boundary.json',{'task_id':tid,'prompt_tokens':np,'full_prefix_tokens':n,'reasoning_suffix_tokens':n-np,
            'absolute_suffix_positions':[np,n-1],'bridge_ids':h['bridge_ids'],'teacher_answer_prefix_supplied':False,
            'source_history_sha256':sha(folder/'source_history.json'),'no_cropping':True,'source_reconstruction_seconds':source_prep,
            'native_full_prefill_seconds':native_prep,'native_prompt_prefill_seconds':prompt_prep})
        conditions=[]
        for name,checkpoint in checkpoints.items():
            conditions.append((name+'_M',checkpoint,'M'))
            if scope=='validation':conditions.append((name+'_H',checkpoint,'H'))
        conditions.append(('D',None,'D'))
        if scope=='validation':conditions.append(('P',None,'P'))
        # Counterbalance wall-clock order across tasks using a fixed rotation.
        shift=index%len(conditions);conditions=conditions[shift:]+conditions[:shift]
        write(folder/'condition_order.json',[x[0] for x in conditions])
        original_versions=[x._version for pair in (*sp,*tp,*prompt) for x in pair]
        for condition,checkpoint,form in conditions:
            c['guard']();c['publish'](stage='coverage_free_generation',scope=scope,task_id=tid,condition=condition)
            mapping=0.;splicing=0.;ph=h;prefill=native_prep if form=='D' else 0.;prefill_tokens=n if form=='D' else 0
            if form in ('M','H'):
                cp=ROOT/checkpoint['path']
                if sha(cp)!=checkpoint['sha256']:raise ValueError('Evaluation checkpoint differs')
                state=torch.load(cp,map_location='cpu',weights_only=True);mapper.load_state_dict(state['state_dict']);del state
                sync();start=time.monotonic();mapped=mapper(sp);sync();mapping=time.monotonic()-start
                if form=='H':
                    start=time.monotonic();base=splice(prompt,mapped,np);sync();splicing=time.monotonic()-start
                    del mapped;prefill=prompt_prep;prefill_tokens=np
                else:base=mapped;del mapped
            elif form=='D':base=tp
            else:
                ids=b.tokenizer.apply_chat_template([{'role':'user','content':visible[tid]['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
                text=b.tokenizer.decode(ids,skip_special_tokens=False)
                if '<think>\n\n</think>' not in text:raise ValueError('Non-thinking template differs')
                ph={'prefix_ids':ids[:-1],'bridge_ids':ids[-1:]}
                write(folder/'prompt_only_template.json',{'task_id':tid,'prompt_ids':ids,'rendered_template':text,
                    'enable_thinking':False,'separate_thinking_stage_requested':False,'prefill_prefix_ids':ph['prefix_ids'],'bridge_ids':ph['bridge_ids']})
                sync();start=time.monotonic();out=b.prefill_chunked(ph['prefix_ids']);base=CacheExtractor.tensors(out.past_key_values);del out;sync()
                prefill=time.monotonic()-start;prefill_tokens=len(ph['prefix_ids'])
            for seed in seeds[tid]:
                c['guard']();dest=folder/condition/f"seed_{seed['seed_index']}"
                versions=[x._version for pair in base for x in pair]
                sync();start=time.monotonic();cache=CacheInjector.create(base,clone=True);sync();clone_seconds=time.monotonic()-start
                if any(x.data_ptr()==y.data_ptr() for pp,qq in zip(base,CacheExtractor.tensors(cache)) for x,y in zip(pp,qq)):
                    raise RuntimeError('Generation cache aliases immutable base')
                row=answer_instrumented(b,ph,cache,tid,seed['stream'],4096,dest,c['telemetry'],c['guard'],c['publish']);del cache
                if row['answer_seed']!=seed['answer_seed']:raise ValueError('Answer seed drift')
                if versions!=[x._version for pair in base for x in pair]:raise RuntimeError('Generation mutated base cache')
                source_time=0. if form=='P' else h['reasoning_seconds']
                row.update(condition=condition,form=form,scope=scope,seed_index=seed['seed_index'],stream=seed['stream'],
                    checkpoint_sha256=checkpoint['sha256'] if checkpoint else None,
                    source_history_sha256=sha(folder/'source_history.json') if form!='P' else None,
                    runtime_identity_sha256=digest(c['identity']),teacher_answer_prefix_supplied=False,
                    mapping_seconds=mapping,splice_seconds=splicing,native_prefill_seconds=prefill,
                    historical_receiver_prefill_tokens=prefill_tokens,cache_clone_seconds=clone_seconds,
                    source_cache_reconstruction_seconds=source_prep if form!='P' else 0.,source_reasoning_seconds=source_time,
                    source_inclusive_estimate_seconds=source_time+mapping+splicing+prefill+row['answer_seconds'],
                    separate_thinking_stage_requested=False,thinking_open_token_in_answer=b.tokenizer.convert_tokens_to_ids('<think>') in row['answer_ids'],
                    immutable_base_cache_unchanged=True,clone_no_alias=True)
                write(dest/'answer.json',row);gc.collect()
            del base;gc.collect();torch.cuda.empty_cache()
            if original_versions!=[x._version for pair in (*sp,*tp,*prompt) for x in pair]:raise RuntimeError('Saved historical pairs mutated')
        finished.append(tid);write(root/'generation_progress.json',{'tasks':finished,'expected_tasks':len(histories),'quality_scores_loaded':False})
        del sp,tp,prompt;gc.collect();torch.cuda.empty_cache()
    write(root/'generation_complete.json',{'tasks':finished,'seed_count':len(next(iter(seeds.values()))),'condition_count':len(conditions),
        'fresh_generations':True,'hidden_tests_loaded':False,'teacher_answer_prefix_supplied':False,'confirmation_used':False})
