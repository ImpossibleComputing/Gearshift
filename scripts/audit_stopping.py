#!/usr/bin/env python3
"""Audit both generation EOS tokens using the exact saved reasoning token trajectories.

Source histories are teacher-forced once to reconstruct KV for this audit. Original
live-handoff results and timing records remain untouched. Replay overhead is recorded.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from gearshift.core import CacheExtractor, save_json, timed
from gearshift.runner import ExperimentRunner
from gearshift.mapping import CacheAdapter
from gearshift.reasoning import numeric_answer, render


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/qwen3_1.7b_to_0.6b.json'); args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text()); runner=ExperimentRunner(cfg)
    source,target=runner.backend('source'),runner.backend('target')
    adapter=CacheAdapter.load(cfg,source,target)
    original=json.loads((runner.out/'reasoning.json').read_text())
    index={(r['dataset_index'],r['condition']):r for r in original}
    paths=Path('data')/runner.out.name/'reasoning_trajectories.jsonl'
    trajectories={r['dataset_index']:r for r in map(json.loads,paths.read_text().splitlines())}
    rows=[]
    for i,tr in enumerate(trajectories.values()):
        assert 151643 not in tr['source_reasoning_ids'] and 151643 not in tr['small_reasoning_ids']
        source_ids=tr['prompt_ids']+tr['source_reasoning_ids']
        so,source_replay_ms=timed(lambda:source.prefill(source_ids),source.device)
        sp=CacheExtractor.tensors(so.past_key_values)
        small=target.prefill(tr['prompt_ids']+tr['small_reasoning_ids'])
        end_think=target.tokenizer.convert_tokens_to_ids('</think>')
        small_delimiter=target.tokenizer.encode(('' if tr['small_reasoning_ids'][-1]==end_think else '\n</think>')+'\nAnswer:',add_special_tokens=False)
        native=target.prefill(source_ids)
        variants=[('A_small_only',target,CacheExtractor.tensors(small.past_key_values),small_delimiter),
            ('B_large_only',source,sp,tr['delimiter_ids']),
            ('C_text_handoff',target,CacheExtractor.tensors(native.past_key_values),tr['delimiter_ids'])]
        for condition,name in [('D_kv_handoff',index[tr['dataset_index'],'D_kv_handoff']['handoff_variant']),('E_functional_kv','functional')]:
            if (tr['dataset_index'],condition) in index:
                variants.append((condition,target,adapter.transform(sp,name),tr['delimiter_ids']))
        for condition,backend,pairs,delimiter in variants:
            result=render(backend,pairs,delimiter,cfg['answer_tokens'])
            old=index[tr['dataset_index'],condition]
            row=dict(dataset_index=tr['dataset_index'],condition=condition,gold=tr['gold'],
                answer=result['text'],answer_token_ids=result['tokens'],correct=numeric_answer(result['text'])==tr['gold'],
                contains_previously_omitted_eos=151643 in result['tokens'],
                final_token=result['tokens'][-1],original_correct=old['correct'],
                same_answer_text_as_live_run=result['text']==old['answer'],
                new_text_is_prefix_of_old=old['answer'].startswith(result['text']),
                answer_hit_token_limit=len(result['tokens'])==cfg['answer_tokens'],
                source_replay_prefill_ms=source_replay_ms,answer_generation_ms=result['generation_ms'])
            rows.append(row)
        save_json(runner.out/'stopping_audit.json',rows)
        print(f'EOS audit {i+1}/{len(trajectories)}',flush=True)
    summary=pd.DataFrame(rows).groupby('condition').agg(n=('correct','count'),correct=('correct','sum'),
        original_correct=('original_correct','sum'),omitted_eos_emitted=('contains_previously_omitted_eos','sum'),
        exact_text_matches=('same_answer_text_as_live_run','sum'),capped=('answer_hit_token_limit','sum'))
    save_json(runner.out/'stopping_audit_summary.json',summary.reset_index().to_dict('records'))
    print(summary.to_string(),flush=True)


if __name__=='__main__': main()
