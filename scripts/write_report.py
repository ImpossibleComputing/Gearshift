#!/usr/bin/env python3
"""Frozen pilot-specific report generator. See evidence/pilot/snapshot for historical source."""
raise SystemExit('Pilot RESULTS.md is frozen. Use scripts/followup_report.py for new experiments; the historical generator is preserved in evidence/pilot/snapshot.')
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/qwen3_1.7b_to_0.6b'
S=json.loads((R/'summary.json').read_text())
F={x['condition']:x for x in S['functional_aggregate']}
G={x['condition']:x for x in S.get('reasoning',[])}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def pct(x): return f'{100*x:.1f}%'
def fmt(x): return f'{x:.2f}'


def score(name):
    g=G[name]
    return f'{g["correct"]}/{g["n"]} ({pct(g["accuracy"])})'


def create():
    required=['A_small_only','B_large_only','C_text_handoff','D_kv_handoff','E_functional_kv']
    assert all(k in G and G[k]['n']==24 for k in required), 'Primary reasoning run is incomplete'
    control=json.loads((R/'controls.json').read_text())
    intensity=json.loads((R/'introspection.json').read_text())
    cfg=json.loads((R/'config.json').read_text())
    ft=json.loads((R/'functional_training.json').read_text())
    raw=pd.read_json(R/'reasoning.json')
    pivot=raw.pivot(index='dataset_index',columns='condition',values='correct')
    chosen=['native','normalized_content','content','post','neighbor','lowrank','mlp','weighted','functional','no_context','zero','naive','random']
    function_table=table(['Condition','KL, nats','Top-1 agreement','Top-5 overlap','Continuation PPL'],
        [[v,fmt(F[v]['kl']),pct(F[v]['top1_agreement']),pct(F[v]['top5_overlap']),fmt(F[v]['perplexity'])] for v in chosen])
    ctx=pd.DataFrame(S['functional_by_length']).set_index(['condition','context_length'])
    context_table=table(['Context','Native PPL','Linear + RoPE PPL','KL-trained PPL','Direct K KL','RoPE-corrected KL','KL-trained KL'],
        [[l,*[fmt(ctx.loc[(v,l),'perplexity']) for v in ['native','normalized_content','functional']],
          *[fmt(ctx.loc[(v,l),'kl']) for v in ['post','normalized_content','functional']]] for l in cfg['lengths']])
    reasoning_table=table(['Condition','Numeric accuracy','95% Wilson interval','Correct, numeric-only response','Answer token cap reached'],
        [[v,score(v),' – '.join(pct(x) for x in G[v]['wilson_95_ci']),f'{G[v]["strict_numeric_only_correct"]}/{G[v]["n"]}',G[v]['answer_hit_token_limit']] for v in required])
    lat=pd.DataFrame(S['latency_medians']).set_index(['condition','context_length'])
    lengths=sorted(lat.index.get_level_values(1).unique())
    latency_table=table(['Tokens','Native prefill ms','Linear mapper + inject ms','Native first logits ms','Linear first logits ms','KL-trained first logits ms'],
        [[l,fmt(lat.loc[('native',l),'handoff_core_ms']),fmt(lat.loc[('normalized_content',l),'handoff_core_ms']),
          *[fmt(lat.loc[(v,l),'time_to_first_token_ms']) for v in ['native','normalized_content','functional']]] for l in lengths])
    crossover=next((l for l in lengths if lat.loc[('normalized_content',l),'time_to_first_token_ms']<lat.loc[('native',l),'time_to_first_token_ms']),None)
    economy_table=table(['Condition','Source reasoning tokens','Target historical prefill tokens','Target generated tokens','Source time, s','Target prefill, ms','Mapper, ms','Target generation, ms','Sum of target stages, ms'],
        [[v,fmt(G[v]['mean_source_reasoning_tokens']),fmt(G[v]['mean_target_prefill_tokens']),fmt(G[v]['mean_target_generation_tokens']),
          fmt(G[v]['mean_source_inference_ms']/1000),fmt(G[v]['mean_target_prefill_ms']),fmt(G[v]['mean_mapper_ms']),
          fmt(G[v]['mean_target_generation_ms']),fmt(G[v]['mean_total_target_ms'])] for v in required])
    align={t:json.loads((R/f'alignment_{t}.json').read_text()) for t in ['k_post','k_content','v']}
    align_counts={t:sum(x==y for x,y in zip(a['selected_sources'],a['normalized_sources'])) for t,a in align.items()}
    reconstruction=table(['Full-rank map','Mean validation R²','Mean cosine'],
        [[x['variant'],fmt(x['r2']),fmt(x['cosine'])] for x in S['reconstruction_validation']])
    c_count=G['C_text_handoff']['correct']; d_count=G['D_kv_handoff']['correct']; e_count=G['E_functional_kv']['correct']
    conditional_e=int((pivot.C_text_handoff & pivot.E_functional_kv).sum())
    completed=raw[(raw.condition=='C_text_handoff') & raw.reasoning_completed]
    completed_ids=set(completed.dataset_index)
    completed_table=table(['Condition','Correct / questions with completed source reasoning'],
        [[v,f'{int(raw[(raw.condition==v)&raw.dataset_index.isin(completed_ids)].correct.sum())}/{len(completed_ids)}'] for v in required[1:]])
    missing_numeric=raw[(raw.condition=='C_text_handoff') & ~raw.source_gold_number_in_reasoning.fillna(False).astype(bool)]
    absent_ids=set(missing_numeric.dataset_index)
    absent_table=table(['Condition','Correct / no gold-number occurrence in source reasoning'],
        [[v,f'{int(raw[(raw.condition==v)&raw.dataset_index.isin(absent_ids)].correct.sum())}/{len(absent_ids)}'] for v in required[1:]])
    drift={}
    for row in S['drift']:
        if row['start'] in [0,56]: drift[(row['condition'],row['start'])]=row
    free=pd.read_json(R/'free_continuations.json')
    free_table=table(['Condition','Free-running positional token agreement','SequenceMatcher token similarity'],
        [[v,pct(free[free.condition==v].position_agreement.mean()),fmt(free[free.condition==v].sequence_similarity.mean())]
         for v in ['normalized_content','functional','no_context']])
    blocks=[]
    blocks.append(f'''# Actual results: cross-model KV transfer

Run date: 2026-09-13. All results below come from local, unquantized Hugging Face/PyTorch runs.

**Verdict: interesting but currently impractical as a post-hoc strong-thinker/weak-renderer optimization.**
The smaller model can consume translated state without rereading historical tokens, and translation is fast.
However, task accuracy is not preserved. A useful positive result is that functional distillation substantially
improves an **affine** mapper; adding a small reconstruction-trained MLP does not achieve the same improvement.

For Qwen3-1.7B → Qwen3-0.6B, linear transfer obtained {pct(F['normalized_content']['top1_agreement'])} next-token top-1
agreement and PPL {fmt(F['normalized_content']['perplexity'])}, versus native PPL {fmt(F['native']['perplexity'])}.
Sixty-four functional-training steps improved this to {pct(F['functional']['top1_agreement'])} agreement and PPL
{fmt(F['functional']['perplexity'])}. On 24 GSM8K questions, text handoff scored {score('C_text_handoff')}, linear KV
handoff {score('D_kv_handoff')}, and functionally trained KV handoff {score('E_functional_kv')} under numeric extraction.
The latter score drops to {G['E_functional_kv']['strict_numeric_only_correct']}/24 when requiring a correct numeric-only response.

## Setup and controls

Apple M3 Max, 16 CPU cores, 128 GiB unified memory; about 69 GiB available initially. There is no separate VRAM pool.
MPS is built and available; CUDA is unavailable. Python 3.12.4, PyTorch 2.14.0, Transformers 4.57.6, FP16 weights/caches,
SDPA attention, batch size one, greedy decoding, seed 20260913. The source has 1,720,574,976 parameters and the target
596,049,920. Both remain frozen throughout; only mapper parameters are optimized.

Both models have 28 layers, 8 KV heads, head dimension 128 (1,024 features per token/layer), 16 query heads,
RoPE theta 1,000,000, no RoPE scaling, and 40,960 maximum positions. Hidden widths are 2,048 versus 1,024.
Their serialized tokenizer backends, token-to-ID vocabularies and special-token maps are identical: 151,669 tokenizer
entries and 151,936 padded model-logit dimensions. Model revisions are pinned in configs. Actual caches are
`transformers.cache_utils.DynamicCache`, holding `DynamicLayer` tensors shaped `[1, 8, sequence_length, 128]`.

The reinjection control saved the target cache to disk, released the captured tensors, reloaded it into a new cache,
and fed the next token. Across 128/512/1024/2048/4096 tokens: **maximum and mean logit difference 0; KL 0; top-1 agreement
100%; all 16-token deterministic continuations identical.** Full-context prefill versus incremental inference also
agreed on the top token; worst KL was {max(x['full_prefill_vs_incremental']['kl'] for x in control):.2e}.
The per-length max/mean logit differences and remaining metrics are in `controls.json`.

Attention-source inspection and a pre-RoPE K hook confirm that stored keys are already rotated. Numerical inverse/forward
rotation round trips passed. The small hook discrepancy (maximum about 0.014) comes from comparing FP32 rotation arithmetic
with model FP16 cached values; the inverse/forward test on the stored target keys had maximum error 3.1e-5.

Data: 100,352 WikiText-2 training tokens in 49 nonoverlapping 2,048-token blocks; 8,192 validation tokens in four blocks.
Every token sequence is identical between source and target, and every K/V layer is captured. About 23 GiB of paired
caches are retained locally. Official test data supplies eight independent 4,161-token blocks. Each length uses a nested
suffix ending at the same point and predicts the same 64-token continuation. Thus there are 40 context/probe cases,
**not 40 independent documents**. The JSON summary includes bootstrap intervals resampling the eight blocks together
across lengths. No mapper training or selection uses the test continuations or GSM8K test questions.

## 1. Can a larger model's KV cache be transformed into usable smaller-model state?

**Yes, partially.** Both learned variants substantially outperform zero, random, naïvely reused, and absent history
on next-token distribution fidelity. Runtime counters verify zero historical target-prefill tokens at KV handoff.
The target receives only the new bridge/delimiter, then generates its own new cache entries. This establishes usable
information transfer, not equivalence to native target state.

## 2. How close is it to native target prefill?

{function_table}

KL is native-to-candidate divergence over all model-logit vocabulary entries at the first continuation prediction.
Top-5 overlap is the intersection size divided by five. PPL is `exp(mean token NLL)` across the 64 teacher-forced tokens,
not an arithmetic average of per-example perplexities. Raw JSON also contains JS divergence, cosine of logits,
per-position KL/NLL, and token agreement. To obtain logits from a cache, all context-bearing conditions receive the
same one-token bridge. The no-context condition sees that bridge at position zero.

Free-running comparisons use two held-out blocks per length, up to 24 tokens, and greedy native-target generation as reference:

{free_table}

## 3. How does context length affect quality?

{context_table}

Direct post-RoPE mapping deteriorates particularly at 4K, beyond the 2K training position range. Removing source rotation
and applying target rotation stabilizes the curve even though the RoPE configurations match. A general learned feature
matrix does not commute with rotary rotations. Functional training, conducted at 128/512-token contexts, also transfers
to 4K reasonably well, but retains a substantial native-prefill gap. There is no monotonic catastrophic length collapse
for the RoPE-corrected maps in this limited range.

![Context quality](plots/qwen3_1.7b_to_0.6b/context_quality.png)

## 4. Does strong thinker → weak renderer retain the strong model's reasoning benefit?

**Not reliably in this experiment.** The text handoff benefits from the source reasoning; the cache handoffs lose much
of that benefit. A, B, C, D, and E use the same question ordering. B/C/D/E share the exact same source token trajectory.
The source stops at `</think>` when it emits one; otherwise its 768-token thinking budget ends and the closing delimiter
is forced. All answer budgets are 48 tokens.

{reasoning_table}

The standard score extracts the last numerical value (preferring a boxed value) and normalizes it with Decimal, as
documented in the manifest. The numeric-only column is stricter: it rejects explanatory or malformed responses even
when their extracted number matches. The weak cache conditions often reach the answer budget, so reporting only the
lenient extraction score would overstate renderer reliability. All answers and parsed values are saved for audit.

Source reasoning completed naturally on {len(completed_ids)}/24 questions. Restricting to those questions:

{completed_table}

Stopping before the final response **does not eliminate answer leakage**: the reasoning often contains the gold number.
A conservative exact-numeric-occurrence audit finds {len(absent_ids)}/24 questions without that occurrence. On that small subset:

{absent_table}

Occurrences may be intermediate values, and absence is not proof that the reasoning does not semantically reveal the answer.
These controls limit the claim; this is not a clean answer-leakage-free benchmark.

![Reasoning accuracy](plots/qwen3_1.7b_to_0.6b/reasoning_accuracy.png)

## 5. How much text-handoff accuracy is retained?

Linear KV retains {d_count}/{c_count} = {pct(d_count/c_count)} of C's aggregate numeric accuracy.
The functionally trained affine mapper retains {e_count}/{c_count} = {pct(e_count/c_count)}.
Among questions C actually answered correctly, E answers {conditional_e}/{c_count} correctly.
These are small-sample ratios, not precise population estimates; per-condition Wilson intervals appear above and the
paired C/D disagreement counts and exact McNemar test are in `summary.json`.

## 6. When is cache transformation faster, and what are the economics?

{latency_table}

The first measured length with lower linear-handoff time to first-token logits is **{crossover} tokens**. The 32-token
margin is small (about 0.6 ms) and observed ranges overlap; the practical crossover is approximately 32–64 tokens,
not a precisely identified threshold. This uses synchronized per-shape warmups, five repetitions, randomized
condition order, and medians. The shaded plot spans observed min/max. First-token native timing uses a fused context+bridge
prefill; mapped timing includes transformation, cache injection and bridge inference. Model loading, source history
creation, network transport, and training are excluded. Both models and the mapper are resident on the same device.

![Handoff latency](plots/qwen3_1.7b_to_0.6b/handoff_latency.png)
![Short-context crossover](plots/qwen3_1.7b_to_0.6b/short_context_latency.png)

Mean GSM8K stage accounting separates the source cost, which is identical in C/D/E:

{economy_table}

Counts include the emitted token fed to keep the source cache current; target generated tokens in A include its own
thinking. Raw records separately include short-delimiter token counts/time. The reasoning-stage sums exclude the common
defensive cache clone before rendering and should not be read as complete wall-clock latency. The dedicated first-token
benchmark does include injection. Long incorrect generations can outweigh the prefill saving, so these are not
accuracy-preserving end-to-end speedups. No dollar, energy, or production-throughput claim is made.

Maximum *observed* MPS allocation during the primary functional sweep was {S['observed_memory_max_gb'].get('mps_allocated_gb',0):.2f} GiB
and driver allocation {S['observed_memory_max_gb'].get('mps_driver_gb',0):.2f} GiB, with several mapper variants resident.
These are sampled allocations, not exact peak GPU memory or total-machine RAM usage.

## 7. Which is harder to transfer: K or V?

**V in this pair, by both reconstruction and functional ablation.**

{reconstruction}

Using **native V + mapped K** gives KL {fmt(F['oracle_native_v']['kl'])} and {pct(F['oracle_native_v']['top1_agreement'])} top-1 agreement.
Using **native K + mapped V** gives KL {fmt(F['oracle_native_k']['kl'])} and {pct(F['oracle_native_k']['top1_agreement'])} agreement.
These are oracle diagnostics requiring real target prefill; they are not feasible handoff methods.

## 8. How strong is cross-layer alignment?

The all-pairs rank-128 screening selects exactly normalized-depth layers for {align_counts['k_content']}/28 content-K targets
and {align_counts['k_post']}/28 post-RoPE-K targets, versus {align_counts['v']}/28 V targets. Most V deviations are nearby,
but the last V layer prefers a source seven layers earlier. Screening uses 4,096 training and 2,048 validation positions.
The figure is explicitly a compressed screening estimate, not 784 full-rank ridge fits.

Final normalized and screened alignments were both refit with all 100,352 positions and four validation-selected penalties.
The full-rank normalized V alignment slightly outperforms the screened selection, so the primary D baseline uses normalized
depth, chosen from validation reconstruction. The test tables retain both choices, preventing a selective presentation.

![Layer screening](plots/qwen3_1.7b_to_0.6b/layer_alignment.png)
![Reconstruction](plots/qwen3_1.7b_to_0.6b/reconstruction.png)

## 9. Does linear mapping suffice, or does nonlinearity help?

Reconstruction-trained affine maps are inadequate for task-quality equivalence. The follow-up order was neighboring-layer
ridge, rank-128 factorization, a 128-hidden-unit residual MLP, then learned two-way convex mixtures of the normalized
and searched layer predictions. These exploratory
refinements use 16,384 training positions; the original affine initialization uses all 100,352. The MLP trains for 64
mini-batches per layer and keeps the best validation checkpoint, including its unchanged affine initialization. They
are bounded capacity probes, not exhaustive architecture or hyperparameter searches.

The neighboring-layer map helps somewhat; rank-128 weight truncation is severely damaging; the residual MLP barely
changes functional quality; layer mixtures do not solve the problem. In contrast, **the functionally trained map remains
affine in content-space K and V**. Adam optimizes only its weights/biases through the frozen target, using next-token KL
over eight teacher-forced positions plus 0.01 times normalized cache reconstruction error. It uses 64 sampled training
contexts, gradient clipping, and a validation-selected checkpoint. Validation KL fell from {ft['initial_validation_kl']:.3f}
to {ft['best_validation_kl']:.3f}. This is evidence that the training objective matters materially, not evidence that a large
nonlinear adapter is required or that affine transfer has reached its ceiling.

The additional test-cache audit is particularly informative: mean V R² changes only from 0.60795 to 0.60746 after
functional training (slightly worse), despite a large improvement in token-distribution fidelity. Numerical cache
reconstruction and useful behavior are demonstrably different objectives here. The per-layer measurements are in
`functional_reconstruction_test.json` and `.csv`.

## 10. What is the dominant failure mode, including after handoff?

Imperfect historical V causes wrong contextual influence; good aggregate cache MSE does not ensure equivalent attention
outputs and downstream token distributions. WikiText-trained mappers also fail to preserve GSM8K chat/answer behavior
reliably. Distribution shift and special-token/control-state errors are plausible contributors, but this run does not
causally separate them from capacity or training-budget limits.

After correct native continuation tokens accumulate, the linear mapper's mean KL falls from
{drift['normalized_content',0]['kl']:.2f} over positions 1–8 to {drift['normalized_content',56]['kl']:.2f} over positions 57–64;
functional-map KL changes from {drift['functional',0]['kl']:.2f} to {drift['functional',56]['kl']:.2f}.
This is incomplete recovery, not convergence to native equivalence. These measurements are **teacher forced**;
free-running errors can feed back and must not be assumed to wash out.

![Post-handoff drift](plots/qwen3_1.7b_to_0.6b/post_handoff_drift.png)

## 11. What does the evidence support architecturally?

**Interesting but currently impractical** for off-the-shelf post-hoc transfer. Cache injection itself works, inexpensive
translation carries useful state, and functional optimization improves it sharply. Those are encouraging ingredients
for a deliberately shared cache ABI, but they do not demonstrate a viable quality-preserving optimization, a co-trained
ABI, cross-family transfer, bidirectional switching, or 100B → 7B behavior. No such architecture was implemented.

## 12. The single next experiment that would reduce uncertainty most

Train the same affine mapper to convergence on **disjoint chat-formatted reasoning trajectories**, with multi-token
native-target KL as the main loss, then evaluate C/D on a larger, newly held-out GSM8K sample and the existing text suite.
Keep both models frozen and include 128–4096-token training contexts. This directly tests whether the remaining task gap
is mainly objective/domain mismatch or a hard limitation of post-hoc affine state transfer, before investing in co-training
a shared ABI. The existing 64-step WikiText functional result makes this a better-supported next step than immediately
building a more complex nonlinear translator.
''')
    extra=ROOT/'results/qwen3_4b_to_0.6b/summary.json'
    if extra.exists():
        s=json.loads(extra.read_text()); ins=json.loads((extra.parent/'introspection.json').read_text())
        g={x['condition']:x for x in s.get('reasoning',[])}
        assert len(g)==4 and all(x['n']==12 for x in g.values()), 'Larger-gap run is incomplete'
        f={x['condition']:x for x in s['functional_aggregate']}
        manifest=json.loads((extra.parent/'reasoning_manifest.json').read_text())
        aligned={}
        for tag in ['k_content','v']:
            a=json.loads((extra.parent/f'alignment_{tag}.json').read_text())
            aligned[tag]=sum(x==y for x,y in zip(a['selected_sources'],a['normalized_sources']))
        ftab=table(['Condition','Mean KL','Top-1 agreement','PPL'],
            [[v,fmt(f[v]['kl']),pct(f[v]['top1_agreement']),fmt(f[v]['perplexity'])]
             for v in ['native','normalized_content','content','post','no_context','zero','naive','random']])
        gtab=table(['Condition','Correct / total','Reasoning completed naturally'],
            [[v,f'{x["correct"]}/{x["n"]}',f'{x["reasoning_completed"]}/{x["n"]}'] for v,x in g.items()])
        blocks.append(f'''## Larger-gap replication: Qwen3-4B → Qwen3-0.6B

The source has {ins['source']['parameter_count']:,} parameters, {ins['source']['layers']} layers,
{ins['source']['kv_heads']} KV heads and head dimension {ins['source']['head_dim']}; the target retains 28 layers.
Tokenizer/backend/vocabulary compatibility and real returned caches were verified again. The same-model control passed
again before cross-model work. Target cache files were reused only after comparing exact selected-token hashes; the
source caches and all source-specific mappers were freshly fit. Both pairs use the same train/validation/test text tokens.
Only {aligned['k_content']}/28 content-K layer choices and {aligned['v']}/28 V choices match normalized depth exactly,
showing that the close pair's almost diagonal alignment does not generalize automatically to unequal depths.

{ftab}

The reasoning budget is 2,048 tokens and the sample is the first 12 of the primary run's seeded question ordering:

{gtab}

This is a larger-gap replication with a longer reasoning allowance, **not a controlled model-size-only accuracy comparison**.
The D handoff uses `{manifest['handoff_variant']}`, selected between normalized and searched alignments by mean validation
reconstruction R² across K/V layers, before running its reasoning benchmark.
Its numeric-only score is {g['D_kv_handoff']['strict_numeric_only_correct']}/12; all
{g['D_kv_handoff']['answer_hit_token_limit']} KV outputs reached the 48-token answer cap. It retains
{g['D_kv_handoff']['correct']}/{g['C_text_handoff']['correct']} of the text baseline's extracted-answer accuracy.
No nonlinear or functional-training refinements were run for the 4B pair. Its complete per-layer reconstruction, layer
screening, context sweep, oracle K/V ablations, repeated latency measurements, and raw reasoning records are retained.

![Larger gap quality](plots/qwen3_4b_to_0.6b/context_quality.png)
![Larger gap latency](plots/qwen3_4b_to_0.6b/handoff_latency.png)
''')
    blocks.append('''## Failures, limitations, and reproducibility

- One first training attempt failed because NumPy promoted a random projection to float64, which MPS does not support.
  The projection is now explicitly float32; the traceback is preserved in `failures.jsonl`. No failed-run metrics were used.
- Final audit found that the original custom decoder used config EOS 151645 but omitted generation-config EOS 151643.
  The decoder now honors both, with a regression test. No saved reasoning, control continuation, or free continuation
  contained the omitted token. A replay audit of all 168 answers with the corrected stop set emitted none either and
  reproduced every condition's numeric accuracy and token-cap counts. Source histories were rebuilt by teacher forcing
  the exact saved reasoning token IDs for this audit; original live-cache results and latency remain primary. All 108
  native/text-handoff answer strings matched exactly; 48/60 translated-cache strings matched, reflecting sensitivity of
  greedy mapped-cache output to FP16 prefill differences. `stopping_audit.json` retains all replay answer token IDs.
- Poor direct-RoPE, rank-truncation, random, zero, naïve, and reasoning-handoff results are retained; none is silently omitted.
- This is one seed, one small text corpus, eight independent test continuations, and small reasoning samples. Validation
  selects layer/penalty/MLP checkpoints. The all-pairs screen is approximate, refinements have modest unequal training
  budgets, and the functional stretch is short. Results are exploratory, not an optimized upper bound.
- The source often computes the answer inside its reasoning, and some trajectories are truncated. Completion-restricted
  and numeric-occurrence audits are reported. The numeric extractor does not normalize arbitrary algebraic expressions.
- No evaluation above 4K history, quantized models, Llama/cross-family models, network cache transfer, continuous batching,
  real monetary/energy costs, co-trained models, or dynamic multi-switch trajectories was run.
- Separate K→K and V→V maps were used. Joint K/V inputs, all-layer neural attention adapters, and source hidden-state
  transfer were not tested; these results do not bound their attainable quality.
- Both models and mapper weights reside on one MPS device. Timings are local synchronized measurements, not a dedicated
  production-server benchmark. Sampled memory is not a complete peak-memory profile.

`README.md` documents the stage commands and implementation boundaries. `requirements.lock.txt` pins the installed
environment; configs pin model and dataset revisions. `results/*/config.json`, `environment.json`, `introspection.json`,
`data_manifest.json`, and cache manifests record provenance. All plotted values derive from the JSON/CSV records and can
be regenerated with `scripts/plot_results.py`; summary tables/intervals use `scripts/summarize_results.py`. Seven focused
unit tests passed, in addition to the actual-model disk reinjection/RoPE controls and runtime token-accounting assertions.
Local mapper checkpoints and approximately tens of GiB of cache data remain available in the ignored `data/` and
`results/` tensor files; no external publication or remote repository was created.
''')
    (ROOT/'RESULTS.md').write_text('\n\n'.join(blocks))
    print(ROOT/'RESULTS.md')


if __name__=='__main__': create()
