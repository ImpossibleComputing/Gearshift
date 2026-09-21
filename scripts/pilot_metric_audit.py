#!/usr/bin/env python3
"""Additional arithmetic from frozen pilot records; never regrades or changes pilot scores."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gearshift.core import save_json


def main():
    all_results={}
    for name in ['qwen3_1.7b_to_0.6b','qwen3_4b_to_0.6b']:
        root=Path('evidence/pilot/snapshot/results')/name
        functional=json.loads((root/'functional.json').read_text())
        original=json.loads((root/'reasoning.json').read_text())
        replay=json.loads((root/'stopping_audit.json').read_text())
        ix={(r['dataset_index'],r['condition']):r for r in original}
        metrics=[]
        for condition in ['native','normalized_content','functional','no_context']:
            group=[r for r in functional if r['condition']==condition]
            if not group:continue
            blocks=sorted({r['example'] for r in group})
            values=np.asarray([[np.mean([r['top1_agreement'] for r in group if r['example']==b]),
                np.mean([np.mean(r['teacher_forced_top1']) for r in group if r['example']==b]),
                np.mean([np.mean(r['token_nll']) for r in group if r['example']==b])] for b in blocks])
            sampled=values[np.random.default_rng(42).integers(0,len(blocks),(5000,len(blocks)))].mean(1)
            metrics.append(dict(condition=condition,context_probe_cases=len(group),independent_blocks=len(blocks),
                continuation_predictions_per_case=64,first_prediction_after_shared_bridge=float(values[:,0].mean()),
                all_64_teacher_forced_top1=float(values[:,1].mean()),perplexity=float(np.exp(values[:,2].mean())),
                cluster_bootstrap_first_probe_95=np.quantile(sampled[:,0],[.025,.975]).tolist(),
                cluster_bootstrap_all64_95=np.quantile(sampled[:,1],[.025,.975]).tolist()))
        comparisons=[]
        for row in replay:
            old=ix[row['dataset_index'],row['condition']]
            count_key='source_generation_tokens' if row['condition']=='B_large_only' else 'target_generation_tokens'
            original_count=old[count_key]-(old['source_reasoning_tokens'] if row['condition']=='B_large_only' else
                 old['small_reasoning_tokens'] if row['condition']=='A_small_only' else 0)
            comparisons.append(dict(dataset_index=row['dataset_index'],condition=row['condition'],
                original_correct=old['correct'],replay_correct=row['correct'],correctness_agrees=old['correct']==row['correct'],
                exact_text_agrees=old['answer']==row['answer'],original_answer_token_count=original_count,
                replay_answer_token_count=len(row['answer_token_ids']),token_count_agrees=original_count==len(row['answer_token_ids']),
                exact_answer_token_agreement=None,token_agreement_limitation='Original live answer token IDs were not saved; skipped-special-token text cannot establish token identity.',
                original_cap=old['answer_hit_token_limit'],replay_cap=row['answer_hit_token_limit'],
                replay_omitted_eos_emitted=row['contains_previously_omitted_eos'],
                **({'original_text':old['answer'],'replay_text':row['answer']} if old['correct']!=row['correct'] else {})))
        summary=[]
        for condition in sorted({r['condition'] for r in comparisons}):
            g=[r for r in comparisons if r['condition']==condition]
            summary.append(dict(condition=condition,n=len(g),original_correct=sum(r['original_correct'] for r in g),
                replay_correct=sum(r['replay_correct'] for r in g),pointwise_correctness_agreement=sum(r['correctness_agrees'] for r in g),
                exact_text_agreement=sum(r['exact_text_agrees'] for r in g),token_count_agreement=sum(r['token_count_agrees'] for r in g),
                exact_token_agreement='unavailable: original answer token IDs absent',correctness_flips=[r for r in g if not r['correctness_agrees']] ))
        all_results[name]=dict(functional=metrics,stopping_summary=summary,stopping_per_item=comparisons,
            interpretation='Aggregate score equality does not imply identical output behavior. Replay changes source cache construction and EOS settings together; it does not isolate numerical sensitivity.')
    save_json('evidence/followup/pilot_metric_audit.json',all_results)
    print(json.dumps({k:dict(functional=v['functional'],stopping=v['stopping_summary']) for k,v in all_results.items()},indent=2))


if __name__=='__main__':main()
