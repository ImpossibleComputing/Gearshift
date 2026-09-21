"""Paired phase-two inference using the validated project cache primitives."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
from .core import ModelBackend, CacheExtractor, CacheInjector, seed_all, timed, sync, memory, environment
from .identity import backend_identity
from .mapping import CacheAdapter
from .reasoning import think
from .followup import render_measured as legacy_render_measured, stop_reason
from .phase2_io import read, write, immutable, bind, artifact, digest, inference_source, validate_transaction,runtime_identity
from .phase2_tasks import from_dict,apply_budgets


def backends(cfg):
    seed_all(cfg['seed'])
    return tuple(ModelBackend(cfg[r],cfg['device'],cfg['dtype'],cfg['attention'],cfg[r+'_revision']) for r in ['source','target'])


def render_measured(backend,pairs,delimiter,budget):
    """Observe the first selected answer token while retaining the validated decoder."""
    original=backend.forward;seen=0;first_token_ms=None;start=time.perf_counter()
    def observed(ids,cache=None,all_logits=False):
        nonlocal seen,first_token_ms
        if seen==1:
            # generate_from has converted the greedy argmax to a Python int at this point.
            first_token_ms=(time.perf_counter()-start)*1000
            backend.forward=original
        seen+=1
        return original(ids,cache,all_logits)
    backend.forward=observed
    try:result=legacy_render_measured(backend,pairs,delimiter,budget)
    finally:backend.forward=original
    result['first_answer_token_ms']=first_token_ms
    return result


def adapter_from(path, source, target):
    maps = torch.load(path, weights_only=True)
    if 'functional_k' in maps:
        maps = {k:maps[k] for k in ['functional_k','functional_v']}
    else:
        maps = {f'functional_{k}':maps[n] for k,n in [('k','normalized_k_content'),('v','normalized_v')]}
    return CacheAdapter(source,target,maps,'functional')


def prompt_ids(tokenizer, task, thinking=True):
    return tokenizer.apply_chat_template([dict(role='user',content=task.generation_text())],
        tokenize=True,add_generation_prompt=True,enable_thinking=thinking)


def bridge_ids(tokenizer, contract, completed=True, protocol='newturn'):
    close = '' if completed else '\n</think>'
    if protocol == 'old': text = close + '\nAnswer:'
    elif protocol == 'native': text = close + '\n\n'
    elif protocol == 'newturn':
        # Exact successful historical scaffold; valid role boundaries checked against actual template.
        text = close + '<|im_end|>\n<|im_start|>user\n' + contract + '\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n'
    else: raise ValueError(protocol)
    return tokenizer.encode(text, add_special_tokens=False)


def protocol_manifest(tokenizer,tasks):
    # Length variants can share a family while using different contracts.
    return {t.task_id:{f'{p}/{done}':bridge_ids(tokenizer,t.contract,done,p)
        for p in ['old','newturn','native'] for done in [False,True]} for t in tasks}


@torch.inference_mode()
def source_history(source, target, task):
    prompt = prompt_ids(source.tokenizer,task)
    if prompt != prompt_ids(target.tokenizer,task): raise ValueError('Tokenizer/prompt mismatch')
    result, wall = timed(lambda:think(source,prompt,task.reasoning_budget),source.device)
    tokens = result['tokens']
    tr = dict(task_id=task.task_id,prompt_ids=prompt,source_reasoning_ids=tokens,
        source_reasoning_text=source.tokenizer.decode(tokens),source_completed=result['completed'],
        stop_reason=stop_reason(tokens,source,task.reasoning_budget,True),token_sha256=digest(prompt+tokens),
        source_wall_ms=wall,source_prefill_ms=result['prefill_ms'],source_reasoning_ms=result['reasoning_ms'],
        source_cache_origin='live incremental; every emitted token included')
    return tr,result['pairs']


@torch.inference_mode()
def measure_condition(source,target,adapter,task,tr,sp,condition, summary=None):
    parts=condition.split('/'); mode=parts[0];protocol=parts[-1] if parts[-1] in ['old','newturn','native'] else 'newturn'
    backend = source if mode == 'B' else target
    before=backend.input_token_count; sync(backend.device); start=time.perf_counter()
    mapping_ms=prefill_ms=splice_ms=small_reasoning_ms=0.; prefill_tokens=small_reasoning_tokens=0
    shared=tr['prompt_ids']+tr['source_reasoning_ids']
    bridge=bridge_ids(backend.tokenizer,task.contract,tr['source_completed'],protocol)
    extra={}
    if mode == 'B': pairs=sp
    elif mode == 'C':
        out,prefill_ms=timed(lambda:target.prefill(shared),target.device)
        pairs=CacheExtractor.tensors(out.past_key_values);prefill_tokens=len(shared)
    elif mode in ['M','H']:
        pairs,mapping_ms=timed(lambda:adapter.transform(sp,'functional'),target.device)
        if mode == 'H':
            out,prefill_ms=timed(lambda:target.prefill(tr['prompt_ids']),target.device)
            pairs,splice_ms=timed(lambda:CacheInjector.splice_prefix(CacheExtractor.tensors(out.past_key_values),pairs),target.device)
            prefill_tokens=len(tr['prompt_ids'])
    elif mode == 'S_think':
        p=prompt_ids(target.tokenizer,task)
        small,small_wall=timed(lambda:think(target,p,task.reasoning_budget),target.device)
        pairs=small['pairs'];prefill_tokens=len(p);small_reasoning_tokens=len(small['tokens'])
        prefill_ms=small['prefill_ms'];small_reasoning_ms=small['reasoning_ms']
        bridge=bridge_ids(target.tokenizer,task.contract,small['completed'],protocol)
        extra=dict(small_reasoning_ids=small['tokens'],small_reasoning_stop=stop_reason(small['tokens'],target,task.reasoning_budget,True))
    elif mode in ['S','T','P']:
        text=task.generation_text()
        if mode == 'T': text+='\n\nPartial analysis (last at most 256 tokens):\n'+target.tokenizer.decode(tr['source_reasoning_ids'][-256:],skip_special_tokens=True)
        if mode == 'P':
            if summary is None: raise ValueError('Summary arm requires a charged source summary')
            text+='\n\nAnalysis handoff plan:\n'+summary['text']
        p=target.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        # Last prompt token goes through render_measured so first-answer-logit accounting is uniform.
        out,prefill_ms=timed(lambda:target.prefill(p[:-1]),target.device)
        pairs=CacheExtractor.tensors(out.past_key_values);prefill_tokens=len(p)-1;bridge=p[-1:]
        extra['native_input_ids']=p
    else: raise ValueError(condition)
    setup_ms=(time.perf_counter()-start)*1000
    result=render_measured(backend,pairs,bridge,task.answer_budget)
    sync(backend.device);wall=(time.perf_counter()-start)*1000
    counted=backend.input_token_count-before
    assert counted==prefill_tokens+len(bridge)+len(result['tokens'])+small_reasoning_tokens
    source_cost=0. if mode in ['S','S_think'] else tr['source_wall_ms']
    summary_cost=summary['wall_ms'] if mode == 'P' else 0.
    return dict(task_id=task.task_id,family=task.family,split=task.split,cluster_id=task.cluster_id,
        condition=condition,answer=result['text'],answer_token_ids=result['tokens'],answer_tokens=len(result['tokens']),
        answer_stop_reason=result['stop_reason'],cap_length=len(result['tokens'])==task.answer_budget,
        shared_source_trajectory_sha256=tr['token_sha256'],prefill_tokens=prefill_tokens,bridge_tokens=len(bridge),
        handoff_input_ids=bridge,actual_backend_input_tokens=counted,small_reasoning_tokens=small_reasoning_tokens,
        source_wall_ms=source_cost,source_reasoning_completed=tr['source_completed'],source_stop_reason=tr['stop_reason'],
        mapping_ms=mapping_ms,prefill_ms=prefill_ms,splice_ms=splice_ms,small_reasoning_ms=small_reasoning_ms,
        clone_ms=result['clone_ms'],bridge_ms=result['delimiter_ms'],answer_generation_ms=result['generation_ms'],
        handoff_wall_ms=wall,summary_wall_ms=summary_cost,total_wall_ms=source_cost+summary_cost+wall,
        first_answer_logit_ms=setup_ms+result['first_logit_wall_ms'],
        first_answer_token_ms=setup_ms+result['first_answer_token_ms'] if result['first_answer_token_ms'] is not None else None,
        first_answer_logit_margin=result['first_answer_logit_margin'],memory=memory(),**extra)


@torch.inference_mode()
def make_summary(source, tr, sp):
    bridge=bridge_ids(source.tokenizer,'Write a compact handoff plan, at most 150 words, containing the task-specific facts, intended solution, qualifications and required output constraints. This plan will be used to finish the original task.',tr['source_completed'])
    result,wall=timed(lambda:render_measured(source,sp,bridge,256),source.device)
    return dict(text=result['text'],token_ids=result['tokens'],bridge_ids=bridge,wall_ms=wall,stop_reason=result['stop_reason'])


def stage_manifest(cfg,source,target,tasks,conditions,mapper_paths,stage,extra=None):
    identity=dict(schema=1,stage=stage,config=cfg,runtime=runtime_identity(),tasks=[t.visible() for t in tasks],
        task_grading_hashes={t.task_id:digest(dict(hidden=t.hidden,grader=t.grader,metadata=t.metadata)) for t in tasks},
        conditions=conditions,checkpoints={k:dict(path=str(p),**artifact(p)) for k,p in mapper_paths.items()},
        source=backend_identity(source),target=backend_identity(target),source_code=inference_source(),
        prompt_ids={t.task_id:prompt_ids(source.tokenizer,t) for t in tasks},
        protocols=protocol_manifest(source.tokenizer,tasks),
        decoding='greedy argmax; all effective EOS; include emitted final token in cache',extra=extra)
    root=Path(cfg['output'])/stage;bind(root/'manifest.json',identity)
    immutable(root/'environment.json',environment()) if not (root/'environment.json').exists() else None
    return root,digest(identity)


def commit_task(root,identity_sha,task,tr,rows,conditions,**extra):
    obj=dict(identity_sha256=identity_sha,task_id=task.task_id,trajectory=tr,rows=rows,**extra)
    obj['payload_sha256']=digest(obj);validate_transaction(obj,identity_sha,task.task_id,conditions)
    immutable(root/'questions'/f'{task.task_id.replace("/","_")}.json',obj)


def complete_stage(root,identity_sha,tasks,conditions_by_id):
    paths=[]
    for task in tasks:
        p=root/'questions'/f'{task.task_id.replace("/","_")}.json'
        if p.exists(): validate_transaction(read(p),identity_sha,task.task_id,conditions_by_id[task.task_id]);paths.append(p)
    write(root/'progress.json',dict(state='complete' if len(paths)==len(tasks) else 'partial',planned=len(tasks),completed=len(paths)))
    if len(paths)==len(tasks):
        immutable(root/'complete.json',dict(identity_sha256=identity_sha,questions=len(paths),files={str(p.relative_to(root)):artifact(p) for p in paths}))


@torch.inference_mode()
def run_ablation(cfg,source,target):
    from .phase2_tasks import TaskSpec, CONTRACTS
    if not read(Path(cfg['output'])/'controls/complete.json')['passed']:raise ValueError('Controls required')
    old=Path('results/followup_v1'); records=read(old/'task_records.json');mf=read(old/'task_manifest.json')
    from .identity import validate_records
    assert digest(mf['identity'])==mf['identity_sha256']
    validate_records(records,mf['identity']['ids'],mf['identity']['conditions'],complete=True)
    for name,desc in read(old/'task_complete.json')['artifacts'].items():
        if artifact(old/name)!=desc: raise ValueError('Historical artifact changed')
    trs=[__import__('json').loads(l) for l in (old/'trajectories/test.jsonl').read_text().splitlines()]
    tasks=[TaskSpec('diagnostic_gsm_'+str(tr['dataset_index']),'arithmetic','diagnostic',str(tr['dataset_index']),
        tr['question'],CONTRACTS['arithmetic'],2048,64,'numeric',dict(gold=tr['gold']),dict(dataset_index=tr['dataset_index'])) for tr in trs]
    paths={k:old/v for k,v in [('initial','affine_initialization.pt'),('plaintext','plaintext_best.pt'),('chat','selected_mapper.pt')]}
    conditions=[f'M/{a}/{s}' for a in paths for s in ['old','newturn']]+[f'{m}/{s}' for m in ['B','C'] for s in ['old','newturn']]+['S','S_think']
    root,identity_sha=stage_manifest(cfg,source,target,tasks,conditions,paths,'ablation',
        dict(history=artifact(old/'trajectories/test.jsonl'),source_cache_construction='full prefill of exact saved tokens; replay, not live-source timing',previously_inspected=True,
             actual_historical_prompt_ids={t.task_id:tr['prompt_ids'] for t,tr in zip(tasks,trs)},
             prompt_note='Top-level prompt_ids apply to fresh small-only; B/C/M use actual_historical_prompt_ids and saved reasoning.'))
    adapters={k:adapter_from(p,source,target) for k,p in paths.items()}
    for i,(task,original) in enumerate(zip(tasks,trs)):
        path=root/'questions'/f'{task.task_id}.json'
        if path.exists():validate_transaction(read(path),identity_sha,task.task_id,conditions);continue
        tr=dict(original);shared=tr['prompt_ids']+tr['source_reasoning_ids']
        if digest(shared)!=tr['token_sha256']:raise ValueError('Trajectory changed')
        out,rebuild_ms=timed(lambda:source.prefill(shared),source.device);sp=CacheExtractor.tensors(out.past_key_values)
        # Preserve original source wall as historical metadata only. No live timing inference for this diagnostic.
        tr.update(original_source_wall_ms=tr['source_wall_ms'],source_wall_ms=0.,reconstruction_wall_ms=rebuild_ms,
                  source_cache_origin='full saved-token prefill replay',task_id=task.task_id)
        rows=[]
        for c in np.random.default_rng(cfg['seed']+i).permutation(conditions).tolist():
            adapter=adapters[c.split('/')[1]] if c.startswith('M/') else adapters['chat']
            rows.append(measure_condition(source,target,adapter,task,tr,sp,c))
        commit_task(root,identity_sha,task,tr,rows,conditions,hidden=task.hidden,
            historical_rows=[r for r in records if r['dataset_index']==original['dataset_index']])
        complete_stage(root,identity_sha,tasks,{t.task_id:conditions for t in tasks})
        print(f'ablation {i+1}/{len(tasks)} complete',flush=True)


@torch.inference_mode()
def run_suite(cfg,source,target,stage,task_file,mapper_path,conditions_by_id):
    if not read(Path(cfg['output'])/'controls/complete.json')['passed']:raise ValueError('Controls required')
    tasks=apply_budgets(cfg,[from_dict(t) for t in read(task_file)])
    root,identity_sha=stage_manifest(cfg,source,target,tasks,conditions_by_id,{'selected':mapper_path},stage,
        dict(task_manifest=artifact(task_file),source_cache_construction='live incremental',grade_after_generation=True))
    adapter=adapter_from(mapper_path,source,target)
    for i,task in enumerate(tasks):
        path=root/'questions'/f'{task.task_id.replace("/","_")}.json';conditions=conditions_by_id[task.task_id]
        if path.exists():validate_transaction(read(path),identity_sha,task.task_id,conditions);continue
        tick=time.perf_counter();tr=None;rows=[];summary=None
        try:
            tr,sp=source_history(source,target,task)
            summary=make_summary(source,tr,sp) if 'P' in conditions else None
            for c in np.random.default_rng(cfg['seed']+i).permutation(conditions).tolist():
                rows.append(measure_condition(source,target,adapter,task,tr,sp,c,summary))
        except BaseException as exc:
            name=task.task_id.replace('/','_');attempt=len(list((root/'failures').glob(name+'_*.json')))
            write(root/'failures'/f'{name}_{attempt}.json',dict(identity_sha256=identity_sha,task_id=task.task_id,
                trajectory=tr,completed_condition_rows=rows,summary=summary,error=repr(exc),state='infrastructure_failure_not_scored'))
            raise
        commit_task(root,identity_sha,task,tr,rows,conditions,summary=summary,experiment_wall_ms=(time.perf_counter()-tick)*1000)
        complete_stage(root,identity_sha,tasks,conditions_by_id)
        print(f'{stage} {i+1}/{len(tasks)} {task.task_id}: source={len(tr["source_reasoning_ids"])} {tr["stop_reason"]}; elapsed={(time.perf_counter()-tick):.1f}s',flush=True)
