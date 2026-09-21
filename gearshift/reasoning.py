from __future__ import annotations
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from .core import CacheExtractor, CacheInjector, memory, save_json, timed, file_sha256
from .mapping import CacheAdapter
from .identity import backend_identity, bind, validate_records, protect_pilot, dataset_identity


def numbers(text):
    return re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',text)


def numeric_answer(text):
    boxed=re.findall(r'\\boxed\{([^{}]+)\}',text)
    nums=numbers(boxed[-1] if boxed else text)
    if not nums: return None
    try: return str(Decimal(nums[-1].replace(',','')).normalize())
    except InvalidOperation: return None


@torch.inference_mode()
def think(backend,prompt,budget):
    out,prefill_ms=timed(lambda:backend.prefill(prompt),backend.device)
    stop=backend.tokenizer.convert_tokens_to_ids('</think>')
    (tokens,cache,logits),reason_ms=timed(lambda:backend.generate_from(out.past_key_values,out.logits,budget,[stop]),backend.device)
    return dict(tokens=tokens,pairs=CacheExtractor.tensors(cache),logits=logits,prefill_ms=prefill_ms,
                reasoning_ms=reason_ms,completed=bool(tokens and tokens[-1]==stop))


@torch.inference_mode()
def render(backend,pairs,delimiter,budget):
    cache=CacheInjector.create(pairs)
    out,delimiter_ms=timed(lambda:backend.forward(delimiter,cache),backend.device)
    (tokens,_,_),generation_ms=timed(lambda:backend.generate_from(out.past_key_values,out.logits,budget),backend.device)
    return dict(tokens=tokens,text=backend.tokenizer.decode(tokens,skip_special_tokens=True),
                delimiter_ms=delimiter_ms,generation_ms=generation_ms)


@torch.inference_mode()
def benchmark(cfg,source,target):
    out=Path(cfg['output'])
    protect_pilot(out)
    root=out/'trajectories'
    adapter=CacheAdapter.load(cfg,source,target)
    handoff_variant=cfg.get('handoff_variant','normalized_content')
    if handoff_variant=='auto':
        scores={variant:np.mean([m['metrics']['r2'] for key in keys for m in adapter.maps[key]])
                for variant,keys in [('normalized_content',['normalized_k_content','normalized_v']),
                                     ('content',['k_content','v'])]}
        handoff_variant=max(scores,key=scores.get)
    corpus=load_dataset('openai/gsm8k','main',split='test',revision=cfg.get('gsm8k_revision'))
    chosen=np.random.default_rng(cfg['seed']).permutation(len(corpus))[:cfg['gsm_examples']]
    chosen = np.asarray(cfg.get('gsm_indices', chosen.tolist()), dtype=int)
    if len(chosen) != cfg['gsm_examples'] or len(set(chosen.tolist())) != len(chosen):
        raise ValueError('Question count or uniqueness disagrees with explicit sample IDs')
    mapper_sha=file_sha256(out/'mappers.pt')
    expected={'A_small_only','B_large_only','C_text_handoff','D_kv_handoff'}
    if 'functional_k' in adapter.maps: expected.add('E_functional_kv')
    prompts = {}
    for index in chosen.tolist():
        message=corpus[index]['question']+'\nSolve carefully. In your final response, give only the numerical answer.'
        prompt=target.tokenizer.apply_chat_template([dict(role='user',content=message)],tokenize=True,
                                                    add_generation_prompt=True,enable_thinking=True)
        assert prompt == source.tokenizer.apply_chat_template([dict(role='user',content=message)],tokenize=True,
                                                    add_generation_prompt=True,enable_thinking=True)
        prompts[index] = prompt
    identity = dict(schema=2, dataset=dataset_identity(corpus,'openai/gsm8k:main',cfg.get('gsm8k_revision'),'test'),
        selected_indices=chosen.tolist(), source=backend_identity(source), target=backend_identity(target),
        prompt_token_ids={str(k): v for k,v in prompts.items()}, conditions=sorted(expected),
        delimiter_tokens={str(done): target.tokenizer.encode(('' if done else '\n</think>')+cfg.get('answer_delimiter','\nAnswer:'),add_special_tokens=False)
                          for done in [False,True]}, decoding='greedy',
        thinking_stop_id=source.tokenizer.convert_tokens_to_ids('</think>'),
        reasoning_token_budget=cfg['reasoning_tokens'],answer_token_budget=cfg['answer_tokens'],
        mapper_sha256=mapper_sha,handoff_variant=handoff_variant,grading='pilot numeric_answer v1')
    rows=json.loads((out/'reasoning.json').read_text()) if (out/'reasoning.json').exists() else []
    progress=validate_records(rows,chosen,expected)
    manifest_path=out/'reasoning_manifest.json'
    if rows and not manifest_path.exists():
        raise ValueError('Existing reasoning records have no experiment manifest')
    bind(manifest_path,identity,origin='original live run; replay belongs to a new experiment')
    done={i for i in chosen.tolist() if {r['condition'] for r in rows if r['dataset_index']==i}==expected}
    if any(r['dataset_index'] not in done for r in rows):
        raise ValueError('Partial question transaction: preserve records and resume in an explicit replay run; no rows will be replaced')
    trajectory_path=root/'reasoning_trajectories.jsonl'
    if trajectory_path.exists():
        trajectory_ids=[json.loads(line)['dataset_index'] for line in trajectory_path.read_text().splitlines()]
        if len(trajectory_ids)!=len(set(trajectory_ids)) or set(trajectory_ids)!=done:
            raise ValueError('Trajectory/answer transactions disagree; preserve partial evidence and use an explicit recovery run')
    if progress['state']=='complete': return
    root.mkdir(parents=True, exist_ok=True)
    save_json(out/'reasoning_progress.json',progress)
    for example,index in enumerate(chosen.tolist()):
        if index in done: continue
        item=corpus[index]
        gold=numeric_answer(item['answer'].split('####')[-1])
        message=item['question']+'\nSolve carefully. In your final response, give only the numerical answer.'
        prompt=target.tokenizer.apply_chat_template([dict(role='user',content=message)],tokenize=True,
                                                    add_generation_prompt=True,enable_thinking=True)
        assert prompt==source.tokenizer.apply_chat_template([dict(role='user',content=message)],tokenize=True,
                                                    add_generation_prompt=True,enable_thinking=True)
        small=think(target,prompt,cfg['reasoning_tokens'])
        large=think(source,prompt,cfg['reasoning_tokens'])
        results={}
        mapping_times={}
        small_delimiter=target.tokenizer.encode(('' if small['completed'] else '\n</think>')+cfg.get('answer_delimiter','\nAnswer:'),add_special_tokens=False)
        delimiter=target.tokenizer.encode(('' if large['completed'] else '\n</think>')+cfg.get('answer_delimiter','\nAnswer:'),add_special_tokens=False)
        results['A_small_only']=render(target,small['pairs'],small_delimiter,cfg['answer_tokens'])
        results['B_large_only']=render(source,large['pairs'],delimiter,cfg['answer_tokens'])
        shared=prompt+large['tokens']
        native,native_ms=timed(lambda:target.prefill(shared),target.device)
        results['C_text_handoff']=render(target,CacheExtractor.tensors(native.past_key_values),delimiter,cfg['answer_tokens'])
        before=target.input_token_count
        mapped,mapper_ms=timed(lambda:adapter.transform(large['pairs'],handoff_variant),target.device)
        results['D_kv_handoff']=render(target,mapped,delimiter,cfg['answer_tokens'])
        mapping_times['D_kv_handoff']=mapper_ms
        assert target.input_token_count-before==len(delimiter)+len(results['D_kv_handoff']['tokens'])
        if 'functional_k' in adapter.maps:
            before=target.input_token_count
            functional,functional_ms=timed(lambda:adapter.transform(large['pairs'],'functional'),target.device)
            results['E_functional_kv']=render(target,functional,delimiter,cfg['answer_tokens'])
            mapping_times['E_functional_kv']=functional_ms
            assert target.input_token_count-before==len(delimiter)+len(results['E_functional_kv']['tokens'])
        reasoning_text=source.tokenizer.decode(large['tokens'])
        numeric_gold_seen=gold in {numeric_answer(n) for n in numbers(reasoning_text)}
        trajectory=dict(dataset_index=index,question=item['question'],gold=gold,prompt_ids=prompt,
                        source_reasoning_ids=large['tokens'],small_reasoning_ids=small['tokens'],
                        source_reasoning_text=reasoning_text,delimiter_ids=delimiter)
        with (root/'reasoning_trajectories.jsonl').open('a') as f: f.write(json.dumps(trajectory)+'\n')
        trajectory_sha=hashlib.sha256(np.asarray(shared,dtype=np.int32).tobytes()).hexdigest()
        for condition,result in results.items():
            small_only=condition=='A_small_only'; large_only=condition=='B_large_only'
            is_c=condition=='C_text_handoff'; is_d=condition in ['D_kv_handoff','E_functional_kv']
            row=dict(example=example,dataset_index=index,condition=condition,gold=gold,
                answer=result['text'],parsed_answer=numeric_answer(result['text']),
                answer_token_ids=result['tokens'],
                correct=numeric_answer(result['text'])==gold,
                source_reasoning_tokens=0 if small_only else len(large['tokens']),
                small_reasoning_tokens=len(small['tokens']) if small_only else 0,
                source_prefill_tokens=0 if small_only else len(prompt),
                target_prefill_tokens=len(prompt) if small_only else (len(shared) if is_c else 0),
                target_handoff_tokens=0 if large_only else len(small_delimiter if small_only else delimiter),
                target_generation_tokens=0 if large_only else len(result['tokens'])+(len(small['tokens']) if small_only else 0),
                source_generation_tokens=0 if small_only else len(large['tokens'])+(len(result['tokens']) if large_only else 0),
                source_prefill_ms=0 if small_only else large['prefill_ms'],
                source_reasoning_ms=0 if small_only else large['reasoning_ms'],
                source_answer_ms=result['generation_ms']+result['delimiter_ms'] if large_only else 0,
                target_prefill_ms=small['prefill_ms'] if small_only else (native_ms if is_c else 0),
                mapper_ms=mapping_times[condition] if is_d else 0,
                target_delimiter_ms=0 if large_only else result['delimiter_ms'],
                target_generation_ms=0 if large_only else result['generation_ms']+(small['reasoning_ms'] if small_only else 0),
                reasoning_completed=small['completed'] if small_only else large['completed'],
                source_gold_number_in_reasoning=numeric_gold_seen if not small_only else None,
                shared_source_trajectory_sha256=trajectory_sha if not small_only else None,
                handoff_variant=('functional' if condition=='E_functional_kv' else handoff_variant) if is_d else None,
                answer_hit_token_limit=len(result['tokens'])==cfg['answer_tokens'],**memory())
            rows.append(row)
        save_json(out/'reasoning_progress.json',validate_records(rows,chosen,expected))
        save_json(out/'reasoning.json',rows)
        pd.DataFrame(rows).to_csv(out/'reasoning.csv',index=False)
        scores={k:numeric_answer(v['text'])==gold for k,v in results.items()}
        print(f'GSM8K {example+1}/{len(chosen)}: {scores}; source tokens={len(large["tokens"])}, completed={large["completed"]}',flush=True)
    save_json(out/'reasoning_progress.json',validate_records(rows,chosen,expected,complete=True))
