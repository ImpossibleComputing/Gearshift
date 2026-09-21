#!/usr/bin/env python3
"""Derive report tables and confidence intervals from recorded measurements."""
import argparse
import json
import math
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.identity import protect_pilot, validate_records
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def wilson(successes,n):
    if not n: return [None,None]
    z=1.959963984540054; p=successes/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--results',default='results/qwen3_1.7b_to_0.6b')
    args=parser.parse_args(); root=Path(args.results)
    protect_pilot(root)
    result={}; tables=[]
    if (root/'functional.json').exists():
        df=pd.read_json(root/'functional.json')
        metrics=['kl','js','top1_agreement','top5_overlap','logit_cosine','nll']
        grouped=df.groupby(['condition','context_length'])[metrics].mean().reset_index()
        grouped['perplexity']=np.exp(grouped.nll)
        result['functional_by_length']=grouped.to_dict('records')
        aggregate=df.groupby('condition')[metrics].mean().reset_index(); aggregate['perplexity']=np.exp(aggregate.nll)
        result['functional_aggregate']=aggregate.to_dict('records')
        # Resample independent test blocks, carrying all lengths from a block together.
        rng=np.random.default_rng(20260913); uncertainty=[]
        for condition,g in df.groupby('condition'):
            a=g.groupby('example')[['kl','nll','top1_agreement']].mean().to_numpy()
            ix=rng.integers(0,len(a),(2000,len(a)))
            means=a[ix].mean(1)
            for j,name in enumerate(['kl','nll','top1_agreement']):
                uncertainty.append(dict(condition=condition,metric=name,independent_blocks=len(a),
                    bootstrap_95_ci=np.quantile(means[:,j],[.025,.975]).tolist()))
        result['cluster_bootstrap']=uncertainty
        tables.append('Functional aggregate (PPL = exp(mean token NLL), not mean of per-example PPL):\n'+aggregate.to_string(index=False))
        tables.append('Functional by context length:\n'+grouped.to_string(index=False))
        drift=[]
        for condition,g in df.groupby('condition'):
            for start in range(0,64,8):
                drift.append(dict(condition=condition,start=start,end=start+8,
                    nll=float(np.mean([row[start:start+8] for row in g.token_nll])),
                    kl=float(np.mean([row[start:start+8] for row in g.token_kl]))))
        result['drift']=drift
        result['observed_memory_max_gb']={c:float(df[c].max()) for c in ['rss_gb','mps_allocated_gb','mps_driver_gb'] if c in df}
    if (root/'latency.csv').exists():
        df=pd.read_csv(root/'latency.csv')
        a=df.groupby(['condition','context_length'])[['handoff_core_ms','time_to_first_token_ms']].median().reset_index()
        result['latency_medians']=a.to_dict('records'); tables.append('Latency medians:\n'+a.to_string(index=False))
    if (root/'reconstruction_validation.json').exists():
        df=pd.read_json(root/'reconstruction_validation.json')
        a=df.groupby('variant')[['mse','r2','explained_variance','cosine']].mean().reset_index()
        result['reconstruction_validation']=a.to_dict('records'); tables.append('Validation reconstruction:\n'+a.to_string(index=False))
    if (root/'refinement_validation.json').exists():
        df=pd.read_json(root/'refinement_validation.json')
        a=df.groupby('variant')[['mse','r2','explained_variance','cosine']].mean().reset_index()
        result['refinement_validation']=a.to_dict('records'); tables.append('Refinements:\n'+a.to_string(index=False))
    if (root/'reasoning.json').exists():
        manifest=json.loads((root/'reasoning_manifest.json').read_text())
        if 'identity' not in manifest:
            raise ValueError('Legacy pilot records must remain frozen; canonical identity required for new summaries')
        validate_records(json.loads((root/'reasoning.json').read_text()),
            manifest['identity']['selected_indices'],manifest['identity']['conditions'],complete=True)
        df=pd.read_json(root/'reasoning.json'); scores=[]
        for condition,g in df.groupby('condition'):
            n=len(g); successes=int(g.correct.sum())
            strict=sum(bool(re.fullmatch(r'\s*(?:\\boxed\{)?[-+]?\$?\d[\d,]*(?:\.\d+)?\}?\.?\s*',str(a))) and bool(ok)
                       for a,ok in zip(g.answer,g.correct))
            scores.append(dict(condition=condition,n=n,correct=successes,accuracy=successes/n,
                strict_numeric_only_correct=strict,answer_hit_token_limit=int(g.answer_hit_token_limit.sum()),
                wilson_95_ci=wilson(successes,n),reasoning_completed=int(g.reasoning_completed.sum()),
                mean_source_reasoning_tokens=float(g.source_reasoning_tokens.mean()),
                mean_target_prefill_tokens=float(g.target_prefill_tokens.mean()),
                mean_target_generation_tokens=float(g.target_generation_tokens.mean()),
                mean_source_inference_ms=float((g.source_prefill_ms+g.source_reasoning_ms+g.source_answer_ms).mean()),
                mean_target_prefill_ms=float(g.target_prefill_ms.mean()),mean_mapper_ms=float(g.mapper_ms.mean()),
                mean_target_generation_ms=float(g.target_generation_ms.mean()),
                mean_target_delimiter_ms=float(g.target_delimiter_ms.mean()),
                mean_total_target_ms=float((g.target_prefill_ms+g.mapper_ms+g.target_delimiter_ms+g.target_generation_ms).mean())))
        result['reasoning']=scores; tables.append('Reasoning:\n'+pd.DataFrame(scores).to_string(index=False))
        pivot=df.pivot(index='dataset_index',columns='condition',values='correct')
        c=pivot['C_text_handoff']; d=pivot['D_kv_handoff']
        c_only=int((c & ~d).sum()); d_only=int((~c & d).sum())
        result['reasoning_paired']=dict(c_correct=int(c.sum()),d_correct=int(d.sum()),both_correct=int((c&d).sum()),
            c_only=c_only,d_only=d_only,accuracy_ratio=float(d.sum()/c.sum()) if c.sum() else None,
            fraction_of_c_correct_retained=float((c&d).sum()/c.sum()) if c.sum() else None,
            mcnemar_exact_p=float(binomtest(min(c_only,d_only),c_only+d_only,.5).pvalue) if c_only+d_only else 1)
        audit=[]
        for flag,g in df[df.condition!='A_small_only'].groupby('source_gold_number_in_reasoning'):
            audit.append(dict(gold_numeric_occurrence=bool(flag),conditions=[dict(condition=c,n=len(s),correct=int(s.correct.sum())) for c,s in g.groupby('condition')]))
        result['reasoning_numeric_leakage_audit']=audit
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (root/'tables.txt').write_text('\n\n'.join(tables)+'\n')
    print('\n\n'.join(tables))


if __name__=='__main__': main()
