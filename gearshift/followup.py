"""Bounded follow-up protocol. Separate outputs, fixed settings, shared source histories."""
from __future__ import annotations
import copy
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from .core import (ModelBackend, CacheExtractor, CacheInjector, save_json, save_checkpoint,
                   file_sha256, seed_all, timed, sync, memory, environment)
from .identity import bind, digest, backend_identity, artifact, validate_records, protect_pilot, validate_array
from .data import prepare_tokens
from .mapping import CacheAdapter
from .reasoning import think, numeric_answer


def source_files():
    import shutil
    files={str(p): artifact(p) for pattern in ['gearshift/*.py','scripts/*.py','tests/*.py']
           for p in Path('.').glob(pattern)}
    files['requirements.lock.txt']=artifact('requirements.lock.txt')
    snapshot=Path('evidence/followup/source_snapshots')/digest(files)
    for name,desc in files.items():
        dest=snapshot/name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
        if artifact(dest)!=desc:raise ValueError('Source snapshot hash mismatch')
    return files


def backends(cfg):
    seed_all(cfg['seed'])
    return tuple(ModelBackend(cfg[r],cfg['device'],cfg['dtype'],cfg['attention'],cfg[r+'_revision'])
                 for r in ['source','target'])


def prompt_tokens(tokenizer, question):
    return tokenizer.apply_chat_template([dict(role='user',content=question+
        '\nSolve carefully. In your final response, give only the numerical answer.')],
        tokenize=True,add_generation_prompt=True,enable_thinking=True)


def save_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));tmp.replace(path)


def stop_reason(tokens, backend, budget, thinking=False):
    if tokens and tokens[-1] in backend.eos: return 'eos'
    if thinking and tokens and tokens[-1]==backend.tokenizer.convert_tokens_to_ids('</think>'): return 'end_think'
    if len(tokens)==budget: return 'cap'
    return 'other'


def validate_followup_config(cfg,source,target):
    exp=json.loads((Path(cfg['output'])/'experiment_manifest.json').read_text())['identity']
    if cfg!=exp['config'] or backend_identity(source)!=exp['source'] or backend_identity(target)!=exp['target']:
        raise ValueError('Follow-up configuration/backend identity changed; use a new run ID')
    return exp


def prepare(cfg, source, target):
    root=Path(cfg['output']);protect_pilot(root)
    if not (root/'controls_v2/complete.json').exists(): raise ValueError('Passing actual-model controls required')
    started=time.perf_counter()
    corpus=load_dataset('openai/gsm8k','main',revision=cfg['gsm8k_revision'])
    train_ids=np.random.default_rng(cfg['seed']).permutation(len(corpus['train']))
    splits={'train':train_ids[:cfg['chat_train_examples']].tolist(),
            'validation':train_ids[cfg['chat_train_examples']:cfg['chat_train_examples']+cfg['chat_validation_examples']].tolist()}
    excluded=sorted({i for name in ['qwen3_1.7b_to_0.6b','qwen3_4b_to_0.6b']
                     for i in json.loads((Path('results')/name/'reasoning_manifest.json').read_text())['selected_indices']})
    test_ids=[int(i) for i in np.random.default_rng(cfg['seed']).permutation(len(corpus['test'])) if i not in excluded][:cfg['gsm_examples']]
    identity=dict(schema=1,config=cfg,source=backend_identity(source),target=backend_identity(target),
        dataset=dict(name='openai/gsm8k',revision=cfg['gsm8k_revision'],fingerprints={s:corpus[s]._fingerprint for s in corpus}),
        train_ids=splits['train'],validation_ids=splits['validation'],test_ids=test_ids,pilot_excluded_ids=excluded,
        initialization=artifact(Path(cfg['initialization'].split(':')[0])),code=source_files(),
        protocol='No official answers used in training histories; final test questions/answers untouched until selection; source histories greedy with all EOS')
    bind(root/'experiment_manifest.json',identity)
    if not (root/'environment.json').exists():save_json(root/'environment.json',environment())
    token_root=prepare_tokens(cfg,target.tokenizer)
    bind(root/'plaintext_manifest.json',dict(tokens=artifact(token_root/'tokens.json')),root=str(token_root))
    initial=root/'affine_initialization.pt'
    if not initial.exists():
        maps=torch.load(Path(cfg['initialization'].split(':')[0]),weights_only=True)
        selected={key:maps[key] for key in ['normalized_k_content','normalized_v']}
        save_checkpoint(initial,selected)
        del maps,selected
    bind(root/'affine_initialization.json',dict(source=identity['initialization'],export=artifact(initial),
         variants=['normalized_k_content','normalized_v']))
    for split,ids in splits.items():
        path=root/'trajectories'/f'{split}.jsonl'
        records=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        keys=[r['dataset_index'] for r in records]
        if len(set(keys))!=len(keys) or not set(keys)<=set(ids):raise ValueError('Invalid trajectory progress')
        prompts={str(i):prompt_tokens(source.tokenizer,corpus['train'][i]['question']) for i in ids}
        bind(root/f'{split}_trajectory_manifest.json',dict(parent=digest(identity),split='official_train:'+split,
             selected_indices=ids,prompts=prompts,thinking_budget=cfg['reasoning_tokens']))
        for i,index in enumerate(ids):
            if index in keys:continue
            question=corpus['train'][index]['question']; prompt=prompts[str(index)]
            assert prompt==prompt_tokens(target.tokenizer,question)
            result,wall=timed(lambda:think(source,prompt,cfg['reasoning_tokens']),source.device)
            tokens=result['tokens']; stop=stop_reason(tokens,source,cfg['reasoning_tokens'],True)
            row=dict(dataset_index=index,dataset_split='train',role=split,question=question,
                prompt_ids=prompt,source_reasoning_ids=tokens,source_reasoning_text=source.tokenizer.decode(tokens),
                source_completed=result['completed'],stop_reason=stop,
                token_sha256=digest(prompt+tokens),source_wall_ms=wall,
                source_prefill_ms=result['prefill_ms'],source_reasoning_ms=result['reasoning_ms'])
            records.append(row);save_jsonl(path,records)
            del result
            print(f'{split} source trajectories {i+1}/{len(ids)}: {len(tokens)} tokens; {stop}; {wall/1000:.1f}s',flush=True)
        bind(root/f'{split}_trajectory_complete.json',dict(manifest=artifact(root/f'{split}_trajectory_manifest.json'),
             trajectories=artifact(path),count=len(records)))
    save_json(root/'preparation_complete.json',dict(state='complete',full_wall_seconds=time.perf_counter()-started,
         source_generation_wall_seconds=sum(json.loads(l)['source_wall_ms']/1000 for p in (root/'trajectories').glob('*.jsonl') for l in p.read_text().splitlines())))


def packed_chat(path, tokenizer):
    """Pack intact conversations. Windows start at conversation starts; longer ones can cross boundaries."""
    records=[json.loads(line) for line in Path(path).read_text().splitlines()]
    tokens=[]; starts=[]; boundaries=[]
    for r in records:
        starts.append(len(tokens)); tokens.extend(r['prompt_ids']+r['source_reasoning_ids'])
        if not r['source_completed']:tokens.extend(tokenizer.encode('\n</think>',add_special_tokens=False))
        tokens.extend(tokenizer.encode('<|im_end|>\n',add_special_tokens=False));boundaries.append(len(tokens))
    return np.asarray(tokens,dtype=np.int32),starts,boundaries


def make_cases(cfg, tokenizer):
    root=Path(cfg['output']); f=cfg['functional']; n=f['prediction_tokens']
    token_root=Path(json.loads((root/'plaintext_manifest.json').read_text())['root'])
    token_manifest=json.loads((root/'plaintext_manifest.json').read_text())
    if artifact(token_root/'tokens.json')!=token_manifest['identity']['tokens']:
        raise ValueError('Plaintext token manifest changed')
    descriptors=json.loads((token_root/'tokens.json').read_text())['splits']
    for split in ['train','validation']:validate_array(token_root/f'{split}_tokens.npy',descriptors[split]['array'])
    train=np.load(token_root/'train_tokens.npy');val=np.load(token_root/'validation_tokens.npy')
    chat={s:packed_chat(root/'trajectories'/f'{s}.jsonl',tokenizer) for s in ['train','validation']}
    rng=np.random.default_rng(cfg['seed']); schedule=[]
    while len(schedule)<f['steps']: schedule.extend(rng.permutation(f['context_lengths']).tolist())
    cases={d:[] for d in ['plaintext','chat']}; validation=[]
    for step,length in enumerate(schedule[:f['steps']]):
        row=int(rng.integers(len(train))); start=int(rng.integers(len(train[row])-length-n+1))
        tokens=train[row,start:start+length+n].tolist()
        cases['plaintext'].append(dict(step=step+1,length=length,ids=tokens,block=row,start=start,packed_boundaries=0))
        stream,starts,bounds=chat['train']; eligible=[s for s in starts if s+length+n<=len(stream)]
        start=int(rng.choice(eligible));tokens=stream[start:start+length+n].tolist()
        cases['chat'].append(dict(step=step+1,length=length,ids=tokens,start=start,
                                 packed_boundaries=sum(start<b<start+length+n for b in bounds)))
    for length in f['context_lengths']:
        for block in range(f['validation_blocks_per_domain']):
            tokens=val[block,:length+n].tolist()
            validation.append(dict(domain='plaintext',length=length,block=block,ids=tokens,packed_boundaries=0))
            stream,starts,bounds=chat['validation'];eligible=[s for s in starts if s+length+n<=len(stream)]
            start=eligible[block%len(eligible)];tokens=stream[start:start+length+n].tolist()
            validation.append(dict(domain='chat',length=length,block=block,ids=tokens,
                                  packed_boundaries=sum(start<b<start+length+n for b in bounds)))
    return cases,validation


def training_pair(case,source,target):
    n=case['length'];context=case['ids'][:n];inputs=case['ids'][n:]
    so=source.prefill(context);to=target.prefill(context)
    # Outside inference mode: autograd may save these frozen historical features.
    sp=CacheExtractor.tensors(so.past_key_values,clone=True)
    tp=CacheExtractor.tensors(to.past_key_values,clone=True)
    with torch.no_grad():
        reference=target.forward(inputs,CacheInjector.create(tp),all_logits=True).logits.float().log_softmax(-1)
    return sp,tp,inputs,reference


@torch.no_grad()
def score_validation(adapter,pairs,cases,target):
    rows=[]
    for (sp,tp,ids,reference),case in zip(pairs,cases):
        logits=target.forward(ids,adapter.inject(sp,'functional'),all_logits=True).logits.float()
        lp=logits.log_softmax(-1)
        rows.append(dict(domain=case['domain'],length=case['length'],block=case['block'],
            kl=float((reference.exp()*(reference-lp)).sum(-1).mean()),
            first_probe_top1=float((reference[:,:1].argmax(-1)==lp[:,:1].argmax(-1)).float().mean()),
            all_prediction_top1=float((reference.argmax(-1)==lp.argmax(-1)).float().mean())))
    return rows


def train_comparison(cfg,source,target):
    root=Path(cfg['output']);f=cfg['functional'];seed_all(cfg['seed'])
    validate_followup_config(cfg,source,target)
    if not (root/'preparation_complete.json').exists():raise ValueError('Complete preparation required')
    cases,validation=make_cases(cfg,target.tokenizer)
    identity=dict(schema=1,config=f,initialization=artifact(root/'affine_initialization.pt'),
        source=backend_identity(source),target=backend_identity(target),code=source_files(),
        train_cases_sha256=digest(cases),validation_sha256=digest(validation),
        trajectories={s:artifact(root/'trajectories'/f'{s}.jsonl') for s in ['train','validation']})
    bind(root/'training_manifest.json',identity)
    if (root/'selection.json').exists():
        selection=json.loads((root/'selection.json').read_text())
        if artifact(root/'selected_mapper.pt')!=selection['selected_artifact']:raise ValueError('Selected checkpoint changed')
        return
    if (root/'training_progress.json').exists():raise ValueError('Partial training exists; preserve it and use an explicit new attempt ID')
    save_json(root/'training_cases.json',cases);save_json(root/'validation_cases.json',validation)
    started=time.perf_counter()
    assert not any(p.requires_grad for b in [source,target] for p in b.model.parameters())
    initial=torch.load(root/'affine_initialization.pt',weights_only=True)
    arms={}
    for domain in ['plaintext','chat']:
        maps={f'functional_{kind}':copy.deepcopy(initial[key]) for kind,key in [('k','normalized_k_content'),('v','normalized_v')]}
        adapter=CacheAdapter(source,target,maps,'functional');parameters=[]
        for key in ['functional_k','functional_v']:
            for m in adapter.weights[key]:
                for name in ['weight','bias']:
                    m[name]=torch.nn.Parameter(m[name].detach().clone());parameters.append(m[name])
        arms[domain]=dict(adapter=adapter,parameters=parameters,
            optimizer=torch.optim.Adam(parameters,lr=f['learning_rate']),best=float('inf'),best_step=0,
            history=[],no_improvement=0)
    validation_pairs=[training_pair(case,source,target) for case in validation]
    def checkpoint(domain,step):
        arm=arms[domain];rows=score_validation(arm['adapter'],validation_pairs,validation,target)
        score=float(np.mean([r['kl'] for r in rows]));entry=dict(step=step,validation_kl=score,
             by_domain={d:float(np.mean([r['kl'] for r in rows if r['domain']==d])) for d in arms},cases=rows)
        arm['history'].append(entry)
        if score < arm['best']:
            if arm['best']-score>=f['min_delta']:arm['no_improvement']=0
            else:arm['no_improvement']+=1
            arm['best']=score;arm['best_step']=step
            weights={key:[{k:(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v) for k,v in m.items()}
                         for m in arm['adapter'].weights[key]] for key in ['functional_k','functional_v']}
            save_checkpoint(root/f'{domain}_best.pt',weights)
        else:arm['no_improvement']+=1
        save_json(root/f'{domain}_learning_curve.json',arm['history'])
        print(f'{domain} step {step}: mixed validation KL {score:.4f}; {entry["by_domain"]}',flush=True)
    for domain in arms:checkpoint(domain,0)
    rows=[];stop='fixed 128-step bound'
    for step in range(1,f['steps']+1):
        for domain,arm in arms.items():
            case=cases[domain][step-1];tick=time.perf_counter()
            sp,tp,ids,reference=training_pair(case,source,target)
            mapped=arm['adapter'].transform(sp,'functional')
            lp=target.forward(ids,CacheInjector.create(mapped,clone=False),all_logits=True).logits.float().log_softmax(-1)
            kl=(reference.exp()*(reference-lp)).sum(-1).mean()
            reconstruction=sum((p.float()-t.float()).square().mean()/t.float().square().mean().clamp_min(1e-6)
                               for pred,actual in zip(mapped,tp) for p,t in zip(pred,actual))/(2*len(tp))
            loss=kl+f['reconstruction_weight']*reconstruction
            arm['optimizer'].zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(arm['parameters'],f['gradient_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient: stop both arms; no test authorized')
            arm['optimizer'].step();sync(target.device)
            rows.append(dict(domain=domain,step=step,length=case['length'],prediction_tokens=len(ids),
                kl=float(kl.detach()),reconstruction=float(reconstruction.detach()),gradient_norm=float(norm),
                wall_ms=(time.perf_counter()-tick)*1000,packed_boundaries=case['packed_boundaries'],**memory()))
            del sp,tp,reference,mapped,lp,kl,reconstruction,loss
        save_json(root/'training_progress.json',dict(state='partial',steps_per_arm=step,planned=f['steps']))
        save_json(root/'training_steps.json',rows)
        if step%f['validation_every']==0 or step==f['steps']:
            for domain in arms:checkpoint(domain,step)
            if f['early_stopping_patience'] and all(a['no_improvement']>=f['early_stopping_patience'] for a in arms.values()):
                stop='both arms failed declared min_delta/patience';break
    selected=min(arms,key=lambda d:(arms[d]['best'],arms[d]['best_step'],d))
    selected_maps=torch.load(root/f'{selected}_best.pt',weights_only=True)
    save_checkpoint(root/'selected_mapper.pt',selected_maps)
    count=sum(v.numel() for entries in selected_maps.values() for m in entries for k,v in m.items() if k in ['weight','bias'])
    assert all(p.grad is None for b in [source,target] for p in b.model.parameters())
    selection=dict(selected_domain=selected,selected_step=arms[selected]['best_step'],objective=f['selection'],
        candidates={d:dict(best_step=a['best_step'],best_validation_kl=a['best'],checkpoint=artifact(root/f'{d}_best.pt')) for d,a in arms.items()},
        selected_artifact=artifact(root/'selected_mapper.pt'),parameter_count=count,precision='float32',
        parameter_bytes=count*4,both_models_resident=True,full_wall_seconds=time.perf_counter()-started,
        steps_per_arm=step,prediction_tokens_per_arm=step*f['prediction_tokens'],stop_reason=stop,
        final_test_opened=False,memory=memory())
    save_json(root/'selection.json',selection)
    save_json(root/'training_progress.json',dict(state='complete',steps_per_arm=step,planned=f['steps'],stop_reason=stop))
    pd.DataFrame(rows).to_csv(root/'training_steps.csv',index=False)


def valid_numeric(text):
    return bool(re.fullmatch(r'\s*[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?\s*',text))


@torch.inference_mode()
def render_measured(backend,pairs,delimiter,budget):
    started=time.perf_counter();sync(backend.device)
    cache,clone_ms=timed(lambda:CacheInjector.create(pairs),backend.device)
    out,delimiter_ms=timed(lambda:backend.forward(delimiter,cache),backend.device)
    first_logit_wall_ms=(time.perf_counter()-started)*1000
    top=out.logits[0,-1].float().topk(2)
    margin=float(top.values[0]-top.values[1])
    (tokens,_,_),generation_ms=timed(lambda:backend.generate_from(out.past_key_values,out.logits,budget),backend.device)
    sync(backend.device)
    return dict(tokens=tokens,text=backend.tokenizer.decode(tokens,skip_special_tokens=True),clone_ms=clone_ms,
        delimiter_ms=delimiter_ms,generation_ms=generation_ms,render_wall_ms=(time.perf_counter()-started)*1000,
        first_logit_wall_ms=first_logit_wall_ms,
        first_answer_logit_margin=margin,first_answer_top2_ids=top.indices.tolist(),
        stop_reason=stop_reason(tokens,backend,budget))


@torch.inference_mode()
def task_comparison(cfg,source,target):
    root=Path(cfg['output']);selection=json.loads((root/'selection.json').read_text())
    validate_followup_config(cfg,source,target)
    if artifact(root/'selected_mapper.pt')!=selection['selected_artifact']:raise ValueError('Selected mapper changed')
    exp=json.loads((root/'experiment_manifest.json').read_text())['identity']
    corpus=load_dataset('openai/gsm8k','main',split='test',revision=cfg['gsm8k_revision'])
    if corpus._fingerprint!=exp['dataset']['fingerprints']['test']:raise ValueError('Test dataset fingerprint changed')
    ids=exp['test_ids'];scaffolds=cfg['scaffolds']
    conditions=['B_large_only']+[f'{mode}/{s}' for s in scaffolds for mode in ['C_text','M_mapped','H_hybrid']]
    prompts={str(i):prompt_tokens(source.tokenizer,corpus[i]['question']) for i in ids}
    scaffold_ids={s:target.tokenizer.encode(text,add_special_tokens=False) for s,text in scaffolds.items()}
    identity=dict(schema=1,experiment=digest(exp),selection=artifact(root/'selection.json'),source=backend_identity(source),
        target=backend_identity(target),ids=ids,conditions=conditions,prompts=prompts,scaffold_ids=scaffold_ids,
        forced_closure_ids=target.tokenizer.encode('\n</think>',add_special_tokens=False),code=source_files(),
        grading=cfg['grading'],reasoning_budget=cfg['reasoning_tokens'],answer_budget=cfg['answer_tokens'],
        source_cache_protocol='live incremental generation; all emitted tokens included; same history for all C/M/H',
        hybrid='target original prompt prefill + full-history mapping, then slice suffix at original absolute position')
    rows=[];trajectories=[]
    question_dir=root/'questions'
    for p in question_dir.glob('*.json'):
        obj=json.loads(p.read_text())
        if obj['task_identity']!=digest(identity):raise ValueError('Question belongs to another task identity')
        rows.extend(obj['rows']);trajectories.append(obj['trajectory'])
    progress=validate_records(rows,ids,conditions)
    done={r['dataset_index'] for r in trajectories}
    if any(sum(r['dataset_index']==i for r in rows)!=len(conditions) for i in done):raise ValueError('Partial question transaction')
    if (root/'task_records.json').exists():
        previous=json.loads((root/'task_records.json').read_text())
        validate_records(previous,ids,conditions)
        canonical={(r['dataset_index'],r['condition']):r for r in rows}
        if any(canonical.get((r['dataset_index'],r['condition']))!=r for r in previous):
            raise ValueError('Task aggregate and canonical question records disagree; nothing will be overwritten')
    bind(root/'task_manifest.json',identity)
    if (root/'task_complete.json').exists():
        validate_records(rows,ids,conditions,complete=True)
        complete=json.loads((root/'task_complete.json').read_text())
        for name,desc in complete['artifacts'].items():
            if artifact(root/name)!=desc:raise ValueError('Completed task artifact changed')
        return
    question_dir.mkdir(exist_ok=True)
    before_mapper=memory()
    adapter=CacheAdapter(source,target,torch.load(root/'selected_mapper.pt',weights_only=True),'functional')
    sync(target.device)
    after_mapper=memory()
    if not (root/'selected_mapper_memory.json').exists():
        save_json(root/'selected_mapper_memory.json',dict(before=before_mapper,after=after_mapper,
            delta={k:after_mapper[k]-before_mapper[k] for k in before_mapper},
            mapper_artifact=artifact(root/'selected_mapper.pt'),parameter_count=selection['parameter_count'],
            precision='float32',both_models_resident=True,note='Adapter keeps a CPU state and MPS weights; allocator/RSS deltas are observed process samples, not peak usage.'))
    session_started=time.perf_counter()
    for number,index in enumerate(ids):
        if index in done:continue
        tick=time.perf_counter();prompt=prompts[str(index)];question=corpus[index]['question']
        assert prompt==prompt_tokens(target.tokenizer,question)
        gold=numeric_answer(corpus[index]['answer'].split('####')[-1])
        large,source_wall=timed(lambda:think(source,prompt,cfg['reasoning_tokens']),source.device)
        shared=prompt+large['tokens'];close=[] if large['completed'] else identity['forced_closure_ids']
        trajectory=dict(dataset_index=index,dataset_split='test',question=question,gold=gold,prompt_ids=prompt,
            source_reasoning_ids=large['tokens'],source_reasoning_text=source.tokenizer.decode(large['tokens']),
            source_completed=large['completed'],stop_reason=stop_reason(large['tokens'],source,cfg['reasoning_tokens'],True),
            token_sha256=digest(shared),source_wall_ms=source_wall,source_prefill_ms=large['prefill_ms'],
            source_reasoning_ms=large['reasoning_ms'],source_cache_origin='live incremental',forced_closure_ids=close)
        order=np.random.default_rng(cfg['seed']+index).permutation(conditions).tolist();current=[]
        for condition in order:
            backend=source if condition=='B_large_only' else target
            before=backend.input_token_count;sync(backend.device);start=time.perf_counter()
            mapper_ms=prefill_ms=splice_ms=0;prefill_tokens=0
            if condition=='B_large_only':
                pairs=large['pairs'];delimiter=close+scaffold_ids['answer'];mode='B_large_only';scaffold='answer'
            else:
                mode,scaffold=condition.split('/');delimiter=close+scaffold_ids[scaffold]
                if mode=='C_text':
                    native,prefill_ms=timed(lambda:target.prefill(shared),target.device)
                    pairs=CacheExtractor.tensors(native.past_key_values);prefill_tokens=len(shared)
                else:
                    pairs,mapper_ms=timed(lambda:adapter.transform(large['pairs'],'functional'),target.device)
                    if mode=='H_hybrid':
                        native,prefill_ms=timed(lambda:target.prefill(prompt),target.device)
                        pairs,splice_ms=timed(lambda:CacheInjector.splice_prefix(CacheExtractor.tensors(native.past_key_values),pairs),target.device)
                        prefill_tokens=len(prompt)
            setup_wall_ms=(time.perf_counter()-start)*1000
            result=render_measured(backend,pairs,delimiter,cfg['answer_tokens']);sync(backend.device)
            wall=(time.perf_counter()-start)*1000
            counted=backend.input_token_count-before
            assert counted==prefill_tokens+len(delimiter)+len(result['tokens'])
            parsed=numeric_answer(result['text']);valid=valid_numeric(result['text']);correct=parsed==gold
            words=result['text'].split();ngrams=[tuple(words[j:j+4]) for j in range(max(0,len(words)-3))]
            current.append(dict(dataset_index=index,condition=condition,mode=mode,scaffold=scaffold,gold=gold,
                answer=result['text'],answer_token_ids=result['tokens'],parsed_answer=parsed,correct=correct,
                format_valid=valid,correct_and_valid=correct and valid,answer_stop_reason=result['stop_reason'],
                answer_hit_token_limit=len(result['tokens'])==cfg['answer_tokens'],repeated_4gram_count=len(ngrams)-len(set(ngrams)),
                reasoning_completed=large['completed'],source_stop_reason=trajectory['stop_reason'],
                shared_source_trajectory_sha256=trajectory['token_sha256'],source_reasoning_tokens=len(large['tokens']),
                source_prompt_tokens=len(prompt),target_prefill_tokens=prefill_tokens,scaffold_tokens=len(scaffold_ids[scaffold]),
                forced_closure_tokens=len(close),handoff_input_token_ids=delimiter,answer_tokens=len(result['tokens']),
                actual_backend_input_tokens=counted,full_source_history_tokens=len(shared),
                mapper_processed_tokens=len(shared) if mode in ['M_mapped','H_hybrid'] else 0,
                source_wall_ms=source_wall,handoff_wall_ms=wall,end_to_end_wall_ms=source_wall+wall,
                mapper_ms=mapper_ms,target_prefill_ms=prefill_ms,splice_ms=splice_ms,
                clone_ms=result['clone_ms'],delimiter_ms=result['delimiter_ms'],answer_generation_ms=result['generation_ms'],
                first_answer_logit_margin=result['first_answer_logit_margin'],first_answer_top2_ids=result['first_answer_top2_ids'],
                first_logit_wall_ms=setup_wall_ms+result['first_logit_wall_ms'],**memory()))
            del pairs,result
            if mode in ['C_text','H_hybrid']:del native
        trajectory['experiment_question_wall_ms']=(time.perf_counter()-tick)*1000
        validate_records(current,[index],conditions,complete=True)
        save_json(question_dir/f'{index}.json',dict(task_identity=digest(identity),trajectory=trajectory,rows=current))
        rows.extend(current);trajectories.append(trajectory)
        save_json(root/'task_progress.json',validate_records(rows,ids,conditions))
        save_json(root/'task_records.json',rows);save_jsonl(root/'trajectories/test.jsonl',trajectories)
        print(f'Test {number+1}/{len(ids)}: source {len(large["tokens"])} tokens ({trajectory["stop_reason"]}); '+
              ', '.join(f'{r["condition"]}={int(r["correct"])}/{int(r["correct_and_valid"])}' for r in current),flush=True)
        del large
    validate_records(rows,ids,conditions,complete=True)
    save_json(root/'task_records.json',rows);save_jsonl(root/'trajectories/test.jsonl',trajectories)
    save_json(root/'task_progress.json',validate_records(rows,ids,conditions,complete=True))
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]).to_csv(root/'task_records.csv',index=False)
    save_json(root/'task_complete.json',dict(state='complete',questions=len(ids),observations=len(rows),
        artifacts={name:artifact(root/name) for name in ['task_records.json','task_records.csv','trajectories/test.jsonl']},
        full_experiment_question_wall_ms=sum(r['experiment_question_wall_ms'] for r in trajectories),
        current_session_wall_seconds=time.perf_counter()-session_started,
        timing_note='Each online condition = shared measured source generation wall + complete condition wall including defensive clone. Entire study costs all conditions. Model loading recorded in command log separately.'))
