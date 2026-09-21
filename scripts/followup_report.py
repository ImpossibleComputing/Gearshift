#!/usr/bin/env python3
"""Derive bounded follow-up tables/figures from complete, identified raw records."""
import argparse
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gearshift.core import save_json
from gearshift.identity import digest, validate_records, artifact


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-half,center+half]


def paired(a,b,seed=42):
    delta=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    rng=np.random.default_rng(seed);means=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
    return dict(n=len(delta),difference=float(delta.mean()),bootstrap_95_ci=np.quantile(means,[.025,.975]).tolist(),
                a_only=int((delta>0).sum()),b_only=int((delta<0).sum()),unit='question')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results/followup_v1')
    p.add_argument('--records-only',action='store_true');p.add_argument('--destination');p.add_argument('--verify-checkpoint',action='store_true')
    args=p.parse_args();root=Path(args.results)
    if args.records_only:
        if not args.destination:p.error('--records-only requires a new --destination')
        from gearshift.phase2_reporting import legacy_records_report
        print(legacy_records_report(root,args.destination,args.verify_checkpoint));return
    raise ValueError('Historical report is frozen. Use --records-only --destination <new-directory> to regenerate compact tables and plots.')
    exp=json.loads((root/'experiment_manifest.json').read_text());cfg=exp['identity']['config']
    if root.name!='followup_v1':raise ValueError('This bounded-run narrative must be adapted explicitly for another experiment')
    mf=json.loads((root/'task_manifest.json').read_text());selection=json.loads((root/'selection.json').read_text())
    rows=json.loads((root/'task_records.json').read_text());identity=mf['identity']
    assert digest(identity)==mf['identity_sha256']
    validate_records(rows,identity['ids'],identity['conditions'],complete=True)
    assert len(identity['ids'])==len(set(identity['ids'])) and not set(identity['ids'])&set(exp['identity']['pilot_excluded_ids'])
    if artifact(root/'selected_mapper.pt')!=selection['selected_artifact']:raise ValueError('Selected checkpoint changed')
    trajectories=[json.loads(l) for l in (root/'trajectories/test.jsonl').read_text().splitlines()]
    tri={r['dataset_index']:r for r in trajectories};assert len(tri)==len(identity['ids'])
    for row in rows:
        tr=tri[row['dataset_index']]
        assert row['shared_source_trajectory_sha256']==digest(tr['prompt_ids']+tr['source_reasoning_ids'])
        if row['mode']=='M_mapped':assert row['target_prefill_tokens']==0
        if row['mode']=='H_hybrid':assert row['target_prefill_tokens']==len(tr['prompt_ids'])
    frame=pd.DataFrame(rows);summary=[];groups=[]
    for condition,g in frame.groupby('condition',sort=True):
        entry=dict(condition=condition,n=len(g))
        for metric in ['correct','format_valid','correct_and_valid']:
            entry[metric]=int(g[metric].sum());entry[metric+'_rate']=float(g[metric].mean())
            entry[metric+'_wilson_95_ci']=wilson(entry[metric],len(g))
        entry.update(eos=int((g.answer_stop_reason=='eos').sum()),truncated=int((g.answer_stop_reason=='cap').sum()),
            cap_length=int(g.answer_hit_token_limit.sum()),repeated_outputs=int((g.repeated_4gram_count>0).sum()),
            median_handoff_ms=float(g.handoff_wall_ms.median()),mean_handoff_ms=float(g.handoff_wall_ms.mean()),
            mean_end_to_end_ms=float(g.end_to_end_wall_ms.mean()),median_first_logit_ms=float(g.first_logit_wall_ms.median()),
            mean_target_prefill_tokens=float(g.target_prefill_tokens.mean()),mean_scaffold_tokens=float(g.scaffold_tokens.mean()),
            mean_forced_closure_tokens=float(g.forced_closure_tokens.mean()))
        summary.append(entry)
        for (complete,source_stop),sub in g.groupby(['reasoning_completed','source_stop_reason']):
            groups.append(dict(condition=condition,source_completed=bool(complete),source_stop_reason=source_stop,n=len(sub),
                extracted_correct=int(sub.correct.sum()),format_valid=int(sub.format_valid.sum()),
                correct_and_valid=int(sub.correct_and_valid.sum()),answer_eos=int((sub.answer_stop_reason=='eos').sum()),
                answer_truncated=int((sub.answer_stop_reason=='cap').sum())))
    uncertainty=[]
    for scaffold in cfg['scaffolds']:
        comparisons=[(f'M_mapped/{scaffold}',f'C_text/{scaffold}'),(f'H_hybrid/{scaffold}',f'C_text/{scaffold}'),
                     (f'H_hybrid/{scaffold}',f'M_mapped/{scaffold}')]
        if scaffold!='answer':comparisons += [(f'{mode}/{scaffold}',f'{mode}/answer') for mode in ['C_text','M_mapped','H_hybrid']]
        for a,b in comparisons:
            for metric in ['correct','format_valid','correct_and_valid']:
                pivot=frame.pivot(index='dataset_index',columns='condition',values=metric)
                uncertainty.append(dict(a=a,b=b,metric=metric,**paired(pivot[a].tolist(),pivot[b].tolist())))
    source_stats=[]
    for split in ['train','validation','test']:
        tr=[json.loads(l) for l in (root/'trajectories'/f'{split}.jsonl').read_text().splitlines()]
        source_stats.append(dict(split=split,n=len(tr),end_think=sum(r['source_completed'] for r in tr),
            capped=sum(r['stop_reason']=='cap' for r in tr),eos=sum(r['stop_reason']=='eos' for r in tr),
            mean_tokens=float(np.mean([len(r['source_reasoning_ids']) for r in tr])),
            source_wall_seconds=sum(r['source_wall_ms'] for r in tr)/1000))
    save_json(root/'summary.json',dict(task=summary,paired_uncertainty=uncertainty,completion_groups=groups,source=source_stats))
    pd.DataFrame(summary).to_csv(root/'task_summary.csv',index=False);pd.DataFrame(groups).to_csv(root/'completion_groups.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(root/'paired_uncertainty.csv',index=False)
    samples=[]
    for condition in sorted(frame.condition.unique()):
        group=frame[frame.condition==condition].sort_values('dataset_index')
        for label,sub in [('valid_success',group[group.correct_and_valid]),('format_invalid_extracted_success',group[group.correct & ~group.format_valid]),
                          ('wrong_answer',group[~group.correct])]:
            for row in sub.head(2).to_dict('records'):
                samples.append(dict(category=label,question=tri[row['dataset_index']]['question'],**row))
    save_json(root/'sample_outputs.json',samples)
    plots=Path('plots')/root.name;plots.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for domain,color in [('plaintext','#2266aa'),('chat','#cf6b28')]:
        curve=json.loads((root/f'{domain}_learning_curve.json').read_text())
        for ax,key in zip(axes,['mixed','plaintext','chat']):
            ys=[r['validation_kl'] if key=='mixed' else r['by_domain'][key] for r in curve]
            ax.plot([r['step'] for r in curve],ys,'o-',label=domain+' training',color=color,ms=4)
            ax.set(title=key.capitalize()+' validation',xlabel='Optimizer steps',ylabel='Mean next-token KL')
    axes[0].legend(frameon=False);fig.suptitle('Same affine initialization; matched lengths and prediction-token budget')
    fig.tight_layout();fig.savefig(plots/'learning_curves.png',dpi=170);fig.savefig(plots/'learning_curves.svg');plt.close(fig)
    ordered=['B_large_only']+[f'{mode}/{s}' for s in cfg['scaffolds'] for mode in ['C_text','M_mapped','H_hybrid']]
    lookup={r['condition']:r for r in summary};ys=np.arange(len(ordered))
    fig,ax=plt.subplots(figsize=(10,5.6))
    for metric,offset,color,label in [('correct',-.17,'#559dc7','Extracted answer correct'),('correct_and_valid',.17,'#1c536f','Correct and numeric-only')]:
        rates=np.array([lookup[c][metric+'_rate'] for c in ordered]);ci=np.array([lookup[c][metric+'_wilson_95_ci'] for c in ordered])
        ax.barh(ys+offset,rates*100,height=.32,color=color,label=label)
        ax.errorbar(rates*100,ys+offset,xerr=np.maximum(0,np.array([rates-ci[:,0],ci[:,1]-rates]))*100,fmt='none',ecolor=color,capsize=2)
    ax.set(yticks=ys,yticklabels=ordered,xlim=(0,100),xlabel='Percent of all fresh test questions (Wilson 95% intervals)',title='Paired thinker → renderer study; identical source histories across handoffs')
    ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,frameon=False);fig.tight_layout();fig.savefig(plots/'task_quality.png',dpi=170);fig.savefig(plots/'task_quality.svg');plt.close(fig)
    task_table=table(['Condition','Extracted correct','Format valid','Correct + valid','EOS / cap','Mean online wall estimate (s)'],
        [[r['condition'],f"{r['correct']}/{r['n']}",f"{r['format_valid']}/{r['n']}",f"{r['correct_and_valid']}/{r['n']}",f"{r['eos']} / {r['truncated']}",f"{r['mean_end_to_end_ms']/1000:.2f}"] for r in [lookup[c] for c in ordered]])
    pair_table=table(['Paired comparison','Correct + valid difference (pp)','Paired bootstrap 95% CI (pp)'],
        [[r['a']+' − '+r['b'],f"{100*r['difference']:+.1f}",f"[{100*r['bootstrap_95_ci'][0]:+.1f}, {100*r['bootstrap_95_ci'][1]:+.1f}]"]
         for r in uncertainty if r['metric']=='correct_and_valid' and r['b'].startswith('C_text')])
    training_table=table(['Training domain','Selected step','Mixed validation KL'],[[d,v['best_step'],f"{v['best_validation_kl']:.4f}"] for d,v in selection['candidates'].items()])
    source_table=table(['Split','Questions','Natural </think>','Capped','Mean reasoning tokens'],[[r['split'],r['n'],r['end_think'],r['capped'],f"{r['mean_tokens']:.0f}"] for r in source_stats])
    timings={s:json.loads((root/f'{s}_command_timing.json').read_text()) if (root/f'{s}_command_timing.json').exists() else {} for s in ['prepare','train','task']}
    timing_table=table(['Command','Model loading (s)','Full command wall (minutes)'],
        [[s,f"{r['model_load_seconds']:.2f}",f"{r['full_command_seconds']/60:.2f}"] for s,r in timings.items()])
    interpretation=json.loads((root/'interpretation.json').read_text()) if (root/'interpretation.json').exists() else {}
    report=f'''# Gearshift bounded follow-up

This is a local exploratory Qwen3-1.7B → Qwen3-0.6B thinker-to-renderer experiment. Both language models remain frozen. Original pilot evidence and scores are unchanged in `RESULTS.md` and `evidence/pilot/snapshot/`; this report describes separate run `followup_v1`.

## Observations: correctness and identity repairs

Canonical reasoning identities cover exact question IDs, resolved dataset fingerprint/bytes, model/tokenizer revisions, prompt and delimiter token IDs, effective EOS settings, decoding, budgets and mapper hash. Configuration drift fails before scientific records are written. No-op reasoning resume preserves manifest bytes and audit annotations. Duplicate/out-of-sample records fail; partial progress is labeled explicitly. New task records use atomic per-question transactions.

Token selection and extraction have separate content identities. Output paths with equal basenames have distinct views. Evaluation-only changes can reuse validated training selections/extractions. Cache markers are checked against every required file's shape, precision, size and SHA-256, and against model/role/position/upstream-selection identity. Existing pilot directories are protected from runner writes. The old narrative generator is explicitly frozen.

Actual unit output is in `evidence/followup/pytest_final.txt`: **39 tests passed**, including the original seven-test suite. Real MPS model controls are in `results/followup_v1/controls_final/` (earlier attempts are also retained). Both source and target pass disk reinjection through 4096 tokens with zero logit difference and identical greedy continuations. Hybrid native-cache splicing passes exact-position controls. A too-short synthetic hybrid fixture failed in the first control attempt, was corrected, and both logs are retained. Initial pytest collection also encountered the archived duplicate test filename; `pytest.ini` now limits collection to the active test directory. Corrupted extraction bytes are rejected despite their completion marker.

## Prespecified comparison and selection

Both arms start from the same exported normalized affine initialization (original checkpoint SHA recorded). Adam learning rate is {cfg['functional']['learning_rate']}; each arm executed {selection['steps_per_arm']} steps and {selection['prediction_tokens_per_arm']} gradient prediction tokens. Context lengths are 128/512/1024/2048 with an identical balanced schedule; each update predicts 8 teacher-forced tokens. Reconstruction weight is 0.01 and gradient clipping is 1.0. Validation occurs every 16 steps. The fixed upper bound is 128 steps, with joint early stopping only if both arms fail the prespecified improvement/patience rule; nonfinite gradients abort both. Stop: {selection['stop_reason']}.

Chat histories are generated from 48 official GSM8K training questions; 12 separate training-split questions supply development validation. No official answers enter generation/training histories. Intact conversations are packed to make long contexts possible; sampled windows start at conversation prefixes and may cross conversation boundaries. Plaintext uses WikiText training/validation blocks. This is an unavoidable domain/packing difference, logged per case; architecture, initialization, optimizer, context schedule and prediction-token budgets are matched. Four windows per domain/length form development validation. Reused/nested windows are dependent, and the development set is small.

Checkpoint selection minimizes equal-weight plaintext/chat validation KL, never final-test task scores. Selected arm: **{selection['selected_domain']}**, step **{selection['selected_step']}**.

{training_table}

![Learning curves](plots/followup_v1/learning_curves.png)

The final deterministic sample contains {len(identity['ids'])} questions, excludes all 24 inspected pilot IDs, and is fixed in the experiment manifest. Its questions/answers are first used after selection is saved. The source reasoning cap is 2048 tokens, versus 768 in the primary pilot. C, M and H share identical saved source token trajectories and identical scaffolds within each comparison; all completed and capped source histories are retained.

{source_table}

## Observations: fresh paired task results

`C_text` re-prefills all question + source reasoning through the target. `M_mapped` maps the full source cache without target historical prefill. `H_hybrid` additionally prefills the original question/chat prefix in the target, then splices the full-history mapped suffix at its original absolute positions. Full-history mapping work is counted. It is partial prompt prefill, not zero target prefill, and target scaffold KV over approximate history is not an oracle native cache.

Scaffolds are prespecified `answer` (existing Answer: delimiter), `instruction` (explicit numeric-only request), and `assistant_turn` (a new user instruction and assistant chat prefix). Exact IDs and actual token counts are saved in `task_manifest.json` and each row. Forced closing tokens are separately counted. All conditions use greedy generation, the model's full EOS set, and a 64-token answer cap. No grammar-constrained decoding or post-hoc parser selection was run.

Extracted correctness retains the pilot parser. Format validity requires only a signed integer/decimal with valid thousands grouping and optional whitespace; boxed expressions, currency symbols, prose and trailing periods are invalid. This prespecified raw-number grammar is stricter than the pilot's format regex, which allowed some boxing/currency/punctuation. New format rates must not be compared directly to the pilot's 1/24 score. Correct boxed math can be usable to a human while failing this machine-readable string contract; format-invalid does not automatically mean mathematically wrong. Correct-and-valid requires both. EOS at exactly the token limit is an EOS stop, while `cap_length` is also retained as a separate count.

{task_table}

![Fresh task quality](plots/followup_v1/task_quality.png)

{pair_table}

Bootstrap draws resample paired questions, not condition rows or tokens (5000 draws). Individual rate intervals are Wilson intervals. These predefined comparisons are exploratory and intervals are not adjusted for multiple comparisons. Complete/truncated source groups are explicitly tabulated in `completion_groups.csv`; these are selected diagnostic subgroups, not a causal estimate of more thinking. Full per-condition differences for all three metrics are in `paired_uncertainty.csv`. Representative correct/valid, extracted-correct but format-invalid, and wrong outputs are selected by dataset ID in `sample_outputs.json`; all raw outputs and source trajectories are included.

## Timing and mapper artifact

Condition wall timers include native prefill or mapping, hybrid splicing, defensive cache cloning, scaffold inference and answer generation, with MPS synchronization. Each online end-to-end estimate adds the single measured live source-generation wall cost to that complete condition wall; source generation is shared across experimental arms. Full experiment wall costs are also recorded, separately from per-stage timers. Condition order is deterministically randomized by question; both models remain resident. These are local short-answer latencies, not dollars, energy or throughput. The large-only continuation baseline is retained.

Full command costs, including model loading:

{timing_table}

These three commands total approximately {sum(r['full_command_seconds'] for r in timings.values())/60:.1f} minutes, excluding the separately logged controls, sensitivity diagnostics, software work and artifact checks. No long-form economics comparison was added. Timers describe this prototype with defensive clones; an optimized standalone baseline need not retain those clones. No serving-system throughput claim follows from these measurements.

The separately exported `selected_mapper.pt` has {selection['parameter_count']:,} affine weights/biases in {selection['precision']}, {selection['parameter_bytes']/2**20:.1f} MiB of scalar storage and {selection['selected_artifact']['bytes']/2**20:.1f} MiB on disk. It is kept locally and excluded from the compact ZIP; its SHA-256 is `{selection['selected_artifact']['sha256']}`. `selected_mapper_memory.json` reports measured RSS/MPS deltas with both models resident, and distinguishes allocator observations from theoretical parameter storage. The research initialization, both best checkpoints, original multi-variant bundles and cache tensors are also preserved locally.

## Pilot metric clarification (scores unchanged)

The primary pilot remains 10/24 extracted-correct and 1/24 correct numeric-only for functional handoff. First-prediction agreement after a shared bridge is 60.0% for normalized affine and 77.5% for functional, over 40 context/probe cases from eight independent text blocks. All 64 teacher-forced predictions average 50.98% and 65.39%, respectively. They are different metrics; uncertainty resamples the eight blocks together with their nested context lengths. Native PPL is 18.92, affine 85.29, functional 31.74, and no-context 67.71.

The stopping replay preserves aggregate scores, but E correctness agrees on 22/24 questions and text on 19/24. IDs 1199 (false→true) and 317 (true→false) cancel. All 24 E answer token counts agree; **exact original-versus-replay answer-token agreement is unavailable because original live token IDs were not stored**. Decoding with skipped special tokens cannot recover them. Per-item audit evidence is in `pilot_metric_audit.json`. This does not prove identical behavior or isolate the cause of numerical sensitivity.

## Inferences and unresolved questions

{interpretation.get('analysis','Analysis pending: do not treat this draft as complete.')}

No Qwen3-4B functional replication, nonlinear/head-aware mapper search, cross-family transfer, shared-ABI training, quantized/distributed serving, long multi-switch run, constrained decoding or long-answer cost study was executed. This fixed-budget experiment cannot establish convergence or an affine-capacity ceiling. Source conclusions can contain the answer: this tests transfer for rendering, not independent acquisition of the thinker's reasoning ability.

## Reproduce and review

```sh
.venv/bin/python -m pytest -q
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/followup_controls.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/followup.py prepare
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/followup.py train
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/followup.py task
.venv/bin/python scripts/pilot_metric_audit.py
.venv/bin/python scripts/followup_report.py
.venv/bin/python scripts/check_artifacts.py
.venv/bin/python scripts/build_review_bundle.py
```

See `REPRODUCE_FOLLOWUP.md` for the exact stage history, source-snapshot versions, corrected sensitivity diagnostic, and instructions to regenerate the excluded affine initialization from a small ZIP.

To regenerate inference, use the pinned revisions and lockfile with locally cached model/dataset weights, or allow ordinary public downloads; no paid service is needed. Use a fresh output/cache namespace, and preserve old runs. Source snapshots and the run manifests identify the exact code used at each stage, including the preparation snapshot predating later hardening. Legacy pilot reproduction source is in `evidence/pilot/snapshot/`. The compact review ZIP contains both original `data/.../reasoning_trajectories.jsonl` files, all new small trajectories/records, source/config manifests, tests, lockfile and plots. Large tensors are excluded and listed by hash with regeneration commands.

The repository still needs an explicit **project LICENSE chosen by its owner** before it is labeled open source. No owner license was selected, no third-party license changed, and nothing was published or pushed. Dataset text and generated trajectories remain subject to upstream terms; see `THIRD_PARTY_NOTICES.md`. A small CPU-test/artifact-integrity workflow is provided but has not been run on GitHub.

## Verified related work

These primary-source records were verified on 2026-09-13. Heo et al., [Cross-Model KV Cache Transfer in LLM Families](https://arxiv.org/abs/2608.03893), study closed-form cache translation with source-layer selection and RoPE removal. Qu et al., [CacheBridge](https://arxiv.org/abs/2609.00891), study head-restricted support and attention-weighted calibration. Li et al., [A Universal Context-Reuse Layer for Cross-Model KV Sharing](https://arxiv.org/abs/2608.30963), study within- and cross-family cache sharing. This follow-up does not reproduce their evaluations or claim novelty/superiority over their methods.

{interpretation.get('recommendation','Publication recommendation pending.')}
'''
    Path('FOLLOWUP_RESULTS.md').write_text(report)
    print(task_table+'\n\n'+training_table+'\n\n'+pair_table)


if __name__=='__main__':main()
