"""Natural output-length diagnostics, fair prefix reuse, and later-position fidelity."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from .core import CacheExtractor,CacheInjector,timed,sync,memory
from .phase2_io import read,write,immutable,digest,artifact,snapshot,validate_transaction
from .phase2_tasks import from_dict,apply_budgets
from .phase2_inference import stage_manifest,source_history,bridge_ids,adapter_from,make_summary,commit_task,complete_stage,prompt_ids,measure_condition,render_measured


LENGTHS=[('short',128,'Provide a concise 50–70 word version focused on the essential result and the most consequential qualification. Preserve the central task constraints. No padding.'),
         ('medium',512,'Provide a useful 180–250 word version, including the key supporting details and relevant qualifications. Do not pad or repeat.'),
         ('long',1024,'Provide a developed 400–550 word version with additional useful explanation, concrete details, and implications consistent with the original brief. Preserve all facts and constraints. Do not pad or repeat.')]
BRANCHES=[('decision','Write a concise decision summary with the controlling facts and calculations. Attribute claims to the supplied source IDs.'),
          ('risks','Write a risk assessment explaining uncertainties, exceptions, what could change the decision, and the supporting source IDs.'),
          ('checklist','Write an actionable implementation checklist with evidence-based prerequisites, verification steps, and unresolved information. Attribute facts to their source IDs.')]


def reserve(cfg):
    root=Path(cfg['output'])/'tasks';tasks=[from_dict(t) for t in read(root/'characterization.json')]
    rng=np.random.default_rng(cfg['seed']);long=[];branch=[];timing=[]
    for family in ['evidence','writing']:
        group=[t for t in tasks if t.family==family];order=rng.permutation(len(group))
        long.extend(group[i].task_id for i in order[:12])
        if family=='evidence':branch=[group[i].task_id for i in order[:16]]
    for family in ['arithmetic','code','evidence','writing']:
        group=[t for t in tasks if t.family==family];timing.extend(group[i].task_id for i in rng.permutation(len(group))[:2])
    obj=dict(long_ids=long,branch_ids=branch,timing_ids=timing,lengths=LENGTHS,branches=BRANCHES,
        methods=['B','C','M','H','T','P','S_think'],timing_methods=['B/newturn','C','M','T'],
        timing_warmups=1,timing_repeats=3,teacher_forced_windows=[[1,32],[33,128],[129,256],[257,512]],
        definitions='Natural requested length overrides the earlier length request, but not material facts/constraints. Branches and lengths are clustered by original task. Source cache is generated once live for each parent in each stage.',
        source=artifact(root/'characterization.json'))
    immutable(root/'extended_reservation.json',obj);return obj


@torch.inference_mode()
def prepare_prefix(source,target,adapter,task,tr,sp,mode,summary=None):
    sync(target.device);start=time.perf_counter();map_ms=prefill_ms=splice_ms=0.;tokens=0;small_tokens=0;small_reason_ms=0.
    shared=tr['prompt_ids']+tr['source_reasoning_ids'];prefix_completed=tr['source_completed'];input_ids=None
    if mode=='B':pairs=sp
    elif mode=='C':
        out,prefill_ms=timed(lambda:target.prefill(shared),target.device);pairs=CacheExtractor.tensors(out.past_key_values);tokens=len(shared)
    elif mode in ['M','H']:
        pairs,map_ms=timed(lambda:adapter.transform(sp,'functional'),target.device)
        if mode=='H':
            out,prefill_ms=timed(lambda:target.prefill(tr['prompt_ids']),target.device);tokens=len(tr['prompt_ids'])
            pairs,splice_ms=timed(lambda:CacheInjector.splice_prefix(CacheExtractor.tensors(out.past_key_values),pairs),target.device)
    elif mode=='S_think':
        from .reasoning import think
        p=prompt_ids(target.tokenizer,task);small,wall=timed(lambda:think(target,p,task.reasoning_budget),target.device)
        pairs=small['pairs'];tokens=len(p);small_tokens=len(small['tokens']);small_reason_ms=small['reasoning_ms'];prefill_ms=small['prefill_ms']
        prefix_completed=small['completed'];input_ids=p+small['tokens']
    elif mode in ['T','P']:
        text=task.generation_text()+'\n\nAnalysis handoff:\n'+(target.tokenizer.decode(tr['source_reasoning_ids'][-256:],skip_special_tokens=True) if mode=='T' else summary['text'])
        input_ids=target.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        out,prefill_ms=timed(lambda:target.prefill(input_ids),target.device);pairs=CacheExtractor.tensors(out.past_key_values);tokens=len(input_ids);prefix_completed=True
    else:raise ValueError(mode)
    sync(target.device)
    return pairs,dict(mode=mode,wall_ms=(time.perf_counter()-start)*1000,mapping_ms=map_ms,prefill_ms=prefill_ms,splice_ms=splice_ms,
        prefill_tokens=tokens,small_reasoning_tokens=small_tokens,small_reasoning_ms=small_reason_ms,prefix_completed=prefix_completed,input_ids=input_ids)


@torch.inference_mode()
def fidelity(target,native,mapped,bridge,answer_ids):
    n=min(512,len(answer_ids));inputs=bridge[-1:]+answer_ids[:n-1]
    caches=[]
    for pairs in [native,mapped]:
        cache=CacheInjector.create(pairs)
        if bridge[:-1]:cache=target.forward(bridge[:-1],cache).past_key_values
        caches.append(cache)
    tokens=[]
    for start in range(0,n,32):
        chunk=inputs[start:start+32];pred=[]
        for i,cache in enumerate(caches):
            out=target.forward(chunk,cache,all_logits=True);caches[i]=out.past_key_values;pred.append(out.logits.float().log_softmax(-1))
        a,b=pred;gold=torch.tensor(answer_ids[start:start+len(chunk)],device=target.device).view(1,-1,1)
        kl=(a.exp()*(a-b)).sum(-1)[0];nll=-b.gather(-1,gold)[0,:,0];native_nll=-a.gather(-1,gold)[0,:,0]
        for j in range(len(chunk)):tokens.append(dict(position=start+j+1,kl=float(kl[j]),mapped_nll=float(nll[j]),native_nll=float(native_nll[j]),top1=bool(a[0,j].argmax()==b[0,j].argmax())))
    windows=[]
    for lo,hi in [[1,32],[33,128],[129,256],[257,512]]:
        rows=[r for r in tokens if lo<=r['position']<=hi]
        if rows:windows.append(dict(start=lo,end=hi,observed_tokens=len(rows),**{m:float(np.mean([r[m] for r in rows])) for m in ['kl','mapped_nll','native_nll','top1']}))
    return dict(reference='native target free-running continuation; fidelity reference, not task truth',tokens=tokens,windows=windows)


@torch.inference_mode()
def run_extended(cfg,source,target,stage):
    spec=read(Path(cfg['output'])/'tasks/extended_reservation.json');all_tasks={t['task_id']:from_dict(t) for t in read(Path(cfg['output'])/'tasks/characterization.json')}
    ids=spec['long_ids'] if stage=='long_outputs' else spec['branch_ids'];bases=apply_budgets(cfg,[all_tasks[i] for i in ids])
    variants={}
    for base in bases:
        if stage=='long_outputs':
            variants[base.task_id]=[replace(base,task_id=base.task_id+'_'+name,answer_budget=cap,contract=base.contract+'\n'+instruction,
                hidden={**base.hidden,**({'word_range':{'short':[50,70],'medium':[180,250],'long':[400,550]}[name]} if base.family=='writing' else {})},
                split='length_diagnostic') for name,cap,instruction in LENGTHS]
        else:
            variants[base.task_id]=[replace(base,task_id=base.task_id+'_'+name,answer_budget=512,contract=instruction,
                grader='evidence_prose',split='branch_diagnostic') for name,instruction in BRANCHES]
    tasks=[t for vs in variants.values() for t in vs];methods=spec['methods'];mapper='results/followup_v1/selected_mapper.pt'
    root,ident=stage_manifest(cfg,source,target,tasks,methods,{'frozen':mapper},stage,dict(reservation=artifact(Path(cfg['output'])/'tasks/extended_reservation.json'),
        script=snapshot(['gearshift/phase2_extended.py','scripts/phase2_extended.py']),source_tasks=[t.visible() for t in bases],
        prefix_reuse='Each method prepares its own prefix once, then clones it for all three outputs. Every method receives fair cache reuse.',
        timing='one-off charges source and preparation per output; amortized charges once per parent. S_think has only its own reasoning charge.'))
    adapter=adapter_from(mapper,source,target)
    for index,base in enumerate(bases):
        parent=root/'parents'/f'{base.task_id}.json'
        if parent.exists():
            p=read(parent)
            if p['identity_sha256']!=ident or p['payload_sha256']!=digest({k:v for k,v in p.items() if k!='payload_sha256'}):raise ValueError('Parent transaction changed')
            for obj in p['transactions']:
                task=next(t for t in variants[base.task_id] if t.task_id==obj['task_id'])
                validate_transaction(obj,ident,task.task_id,methods);immutable(root/'questions'/f'{task.task_id}.json',obj)
            continue
        tr,sp=source_history(source,target,base);summary=make_summary(source,tr,sp)
        prefixes={};setups={}
        for mode in np.random.default_rng(cfg['seed']+index).permutation(methods).tolist():
            prefixes[mode],setups[mode]=prepare_prefix(source,target,adapter,base,tr,sp,mode,summary)
        transactions=[];teacher_rows=[];costs={m:[] for m in methods}
        for vi,task in enumerate(variants[base.task_id]):
            rows=[]
            for mode in np.random.default_rng(cfg['seed']+100*index+vi).permutation(methods).tolist():
                backend=source if mode=='B' else target;setup=setups[mode]
                bridge=bridge_ids(backend.tokenizer,task.contract,setup['prefix_completed'])
                before=backend.input_token_count;result,wall=timed(lambda:render_measured(backend,prefixes[mode],bridge,task.answer_budget),backend.device)
                consumed=backend.input_token_count-before
                assert consumed==len(bridge)+len(result['tokens'])
                source_ms=0. if mode=='S_think' else tr['source_wall_ms'];summary_ms=summary['wall_ms'] if mode=='P' else 0.
                rows.append(dict(task_id=task.task_id,family=task.family,split=task.split,cluster_id=base.cluster_id,condition=mode,
                    answer=result['text'],answer_token_ids=result['tokens'],answer_tokens=len(result['tokens']),answer_stop_reason=result['stop_reason'],
                    cap_length=len(result['tokens'])==task.answer_budget,shared_source_trajectory_sha256=tr['token_sha256'],
                    prefill_tokens=0,bridge_tokens=len(bridge),handoff_input_ids=bridge,actual_backend_input_tokens=consumed,
                    historical_prefill_tokens_paid_once=setup['prefill_tokens'],setup=setup,source_wall_ms=source_ms,summary_wall_ms=summary_ms,
                    mapping_ms=setup['mapping_ms'],prefill_ms=setup['prefill_ms'],clone_ms=result['clone_ms'],bridge_ms=result['delimiter_ms'],answer_generation_ms=result['generation_ms'],
                    handoff_wall_ms=setup['wall_ms']+wall,total_wall_ms=source_ms+summary_ms+setup['wall_ms']+wall,
                    amortized_output_wall_ms=(source_ms+summary_ms+setup['wall_ms'])/3+wall,
                    first_answer_logit_ms=setup['wall_ms']+result['first_logit_wall_ms'],
                    first_answer_token_ms=setup['wall_ms']+result['first_answer_token_ms'] if result['first_answer_token_ms'] is not None else None,memory=memory()))
                costs[mode].append(wall)
            obj=dict(identity_sha256=ident,task_id=task.task_id,trajectory=tr,rows=rows)
            obj['payload_sha256']=digest(obj);validate_transaction(obj,ident,task.task_id,methods);transactions.append(obj)
            if stage=='long_outputs' and task.answer_budget==1024:
                reference=next(r for r in rows if r['condition']=='C')
                for mode in ['M','H']:
                    teacher_rows.append(dict(task_id=task.task_id,cluster_id=base.cluster_id,condition=mode,
                        **fidelity(target,prefixes['C'],prefixes[mode],reference['handoff_input_ids'],reference['answer_token_ids'])))
        parent_obj=dict(identity_sha256=ident,task_id=base.task_id,transactions=transactions,summary=summary,teacher_forced=teacher_rows,
            fanout_costs={mode:dict(outputs=3,source_ms=0. if mode=='S_think' else tr['source_wall_ms'],
                prefix_setup_ms=setups[mode]['wall_ms'],summary_ms=summary['wall_ms'] if mode=='P' else 0.,
                output_wall_ms=costs[mode],total_ms=(0 if mode=='S_think' else tr['source_wall_ms'])+setups[mode]['wall_ms']+(summary['wall_ms'] if mode=='P' else 0)+sum(costs[mode])) for mode in methods})
        parent_obj['payload_sha256']=digest(parent_obj);immutable(parent,parent_obj)
        for obj in transactions:immutable(root/'questions'/f'{obj["task_id"]}.json',obj)
        complete_stage(root,ident,tasks,{t.task_id:methods for t in tasks})
        print(f'{stage} parent {index+1}/{len(bases)} complete',flush=True)
    complete_stage(root,ident,tasks,{t.task_id:methods for t in tasks})
    immutable(root/'tasks.json',[dict(t.visible(),grader=t.grader,hidden=t.hidden,metadata=t.metadata) for t in tasks])


@torch.inference_mode()
def run_timing(cfg,source,target):
    spec=read(Path(cfg['output'])/'tasks/extended_reservation.json');all_tasks={t['task_id']:from_dict(t) for t in read(Path(cfg['output'])/'tasks/characterization.json')}
    tasks=apply_budgets(cfg,[all_tasks[i] for i in spec['timing_ids']]);methods=spec['timing_methods'];conditions=[f'{m}/repeat_{r}' for m in methods for r in range(4)]
    mapper='results/followup_v1/selected_mapper.pt';root,ident=stage_manifest(cfg,source,target,tasks,conditions,{'frozen':mapper},'timing_repeats',
        dict(reservation=artifact(Path(cfg['output'])/'tasks/extended_reservation.json'),warmup_repeat=0,reported_repeats=[1,2,3],script=snapshot(['gearshift/phase2_extended.py','scripts/phase2_extended.py'])))
    adapter=adapter_from(mapper,source,target)
    for i,task in enumerate(tasks):
        path=root/'questions'/f'{task.task_id}.json'
        if path.exists():validate_transaction(read(path),ident,task.task_id,conditions);continue
        tr,sp=source_history(source,target,task);rows=[]
        for repeat in range(4):
            for method in np.random.default_rng(cfg['seed']+10*i+repeat).permutation(methods).tolist():
                row=measure_condition(source,target,adapter,task,tr,sp,method);row['condition']=f'{method}/repeat_{repeat}';row['warmup']=repeat==0;rows.append(row)
        commit_task(root,ident,task,tr,rows,conditions);complete_stage(root,ident,tasks,{t.task_id:conditions for t in tasks})
        print(f'timing {i+1}/{len(tasks)} complete',flush=True)
