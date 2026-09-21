# Actual results: cross-model KV transfer

Run date: 2026-09-13. All results below come from local, unquantized Hugging Face/PyTorch runs.

**Verdict: interesting but currently impractical as a post-hoc strong-thinker/weak-renderer optimization.**
The smaller model can consume translated state without rereading historical tokens, and translation is fast.
However, task accuracy is not preserved. A useful positive result is that functional distillation substantially
improves an **affine** mapper; adding a small reconstruction-trained MLP does not achieve the same improvement.

For Qwen3-1.7B → Qwen3-0.6B, linear transfer obtained 60.0% next-token top-1
agreement and PPL 85.29, versus native PPL 18.92.
Sixty-four functional-training steps improved this to 77.5% agreement and PPL
31.74. On 24 GSM8K questions, text handoff scored 17/24 (70.8%), linear KV
handoff 0/24 (0.0%), and functionally trained KV handoff 10/24 (41.7%) under numeric extraction.
The latter score drops to 1/24 when requiring a correct numeric-only response.

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
agreed on the top token; worst KL was 5.82e-08.
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

| Condition | KL, nats | Top-1 agreement | Top-5 overlap | Continuation PPL |
| --- | --- | --- | --- | --- |
| native | 0.00 | 100.0% | 100.0% | 18.92 |
| normalized_content | 1.22 | 60.0% | 53.0% | 85.29 |
| content | 1.56 | 45.0% | 51.0% | 88.97 |
| post | 2.72 | 37.5% | 36.0% | 104.24 |
| neighbor | 0.98 | 50.0% | 59.5% | 65.07 |
| lowrank | 7.09 | 25.0% | 19.5% | 11848.31 |
| mlp | 1.55 | 45.0% | 51.0% | 88.67 |
| weighted | 1.31 | 52.5% | 53.0% | 83.77 |
| functional | 0.43 | 77.5% | 66.0% | 31.74 |
| no_context | 10.09 | 0.0% | 0.0% | 67.71 |
| zero | 11.12 | 0.0% | 0.5% | 675.86 |
| naive | 8.07 | 7.5% | 5.0% | 2463.79 |
| random | 15.92 | 0.0% | 0.0% | 40325833.89 |

KL is native-to-candidate divergence over all model-logit vocabulary entries at the first continuation prediction.
Top-5 overlap is the intersection size divided by five. PPL is `exp(mean token NLL)` across the 64 teacher-forced tokens,
not an arithmetic average of per-example perplexities. Raw JSON also contains JS divergence, cosine of logits,
per-position KL/NLL, and token agreement. To obtain logits from a cache, all context-bearing conditions receive the
same one-token bridge. The no-context condition sees that bridge at position zero.

Free-running comparisons use two held-out blocks per length, up to 24 tokens, and greedy native-target generation as reference:

| Condition | Free-running positional token agreement | SequenceMatcher token similarity |
| --- | --- | --- |
| normalized_content | 16.7% | 0.23 |
| functional | 52.1% | 0.61 |
| no_context | 0.4% | 0.06 |

## 3. How does context length affect quality?

| Context | Native PPL | Linear + RoPE PPL | KL-trained PPL | Direct K KL | RoPE-corrected KL | KL-trained KL |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 23.21 | 89.41 | 35.53 | 2.57 | 1.66 | 0.43 |
| 512 | 18.14 | 82.87 | 28.91 | 1.73 | 1.12 | 0.47 |
| 1024 | 18.46 | 82.88 | 30.76 | 2.03 | 1.04 | 0.43 |
| 2048 | 18.30 | 85.42 | 31.32 | 2.44 | 1.04 | 0.34 |
| 4096 | 17.06 | 86.02 | 32.55 | 4.84 | 1.26 | 0.50 |

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

| Condition | Numeric accuracy | 95% Wilson interval | Correct, numeric-only response | Answer token cap reached |
| --- | --- | --- | --- | --- |
| A_small_only | 15/24 (62.5%) | 42.7% – 78.8% | 14/24 | 4 |
| B_large_only | 18/24 (75.0%) | 55.1% – 88.0% | 18/24 | 1 |
| C_text_handoff | 17/24 (70.8%) | 50.8% – 85.1% | 15/24 | 5 |
| D_kv_handoff | 0/24 (0.0%) | 0.0% – 13.8% | 0/24 | 24 |
| E_functional_kv | 10/24 (41.7%) | 24.5% – 61.2% | 1/24 | 23 |

The standard score extracts the last numerical value (preferring a boxed value) and normalizes it with Decimal, as
documented in the manifest. The numeric-only column is stricter: it rejects explanatory or malformed responses even
when their extracted number matches. The weak cache conditions often reach the answer budget, so reporting only the
lenient extraction score would overstate renderer reliability. All answers and parsed values are saved for audit.

Source reasoning completed naturally on 10/24 questions. Restricting to those questions:

| Condition | Correct / questions with completed source reasoning |
| --- | --- |
| B_large_only | 10/10 |
| C_text_handoff | 10/10 |
| D_kv_handoff | 0/10 |
| E_functional_kv | 8/10 |

Stopping before the final response **does not eliminate answer leakage**: the reasoning often contains the gold number.
A conservative exact-numeric-occurrence audit finds 6/24 questions without that occurrence. On that small subset:

| Condition | Correct / no gold-number occurrence in source reasoning |
| --- | --- |
| B_large_only | 0/6 |
| C_text_handoff | 0/6 |
| D_kv_handoff | 0/6 |
| E_functional_kv | 0/6 |

Occurrences may be intermediate values, and absence is not proof that the reasoning does not semantically reveal the answer.
These controls limit the claim; this is not a clean answer-leakage-free benchmark.

![Reasoning accuracy](plots/qwen3_1.7b_to_0.6b/reasoning_accuracy.png)

## 5. How much text-handoff accuracy is retained?

Linear KV retains 0/17 = 0.0% of C's aggregate numeric accuracy.
The functionally trained affine mapper retains 10/17 = 58.8%.
Among questions C actually answered correctly, E answers 10/17 correctly.
These are small-sample ratios, not precise population estimates; per-condition Wilson intervals appear above and the
paired C/D disagreement counts and exact McNemar test are in `summary.json`.

## 6. When is cache transformation faster, and what are the economics?

| Tokens | Native prefill ms | Linear mapper + inject ms | Native first logits ms | Linear first logits ms | KL-trained first logits ms |
| --- | --- | --- | --- | --- | --- |
| 8 | 16.49 | 8.19 | 17.48 | 24.14 | 23.16 |
| 16 | 20.43 | 8.29 | 21.29 | 22.71 | 23.52 |
| 32 | 22.12 | 8.12 | 23.42 | 22.79 | 23.55 |
| 64 | 24.70 | 7.50 | 26.29 | 22.65 | 23.72 |
| 128 | 31.81 | 8.21 | 35.31 | 23.63 | 24.08 |
| 512 | 75.49 | 16.09 | 79.01 | 31.05 | 31.33 |
| 1024 | 142.12 | 29.29 | 146.02 | 45.96 | 45.00 |
| 2048 | 286.43 | 55.05 | 298.64 | 72.19 | 74.51 |
| 4096 | 676.03 | 119.63 | 682.58 | 140.35 | 142.69 |

The first measured length with lower linear-handoff time to first-token logits is **32 tokens**. The 32-token
margin is small (about 0.6 ms) and observed ranges overlap; the practical crossover is approximately 32–64 tokens,
not a precisely identified threshold. This uses synchronized per-shape warmups, five repetitions, randomized
condition order, and medians. The shaded plot spans observed min/max. First-token native timing uses a fused context+bridge
prefill; mapped timing includes transformation, cache injection and bridge inference. Model loading, source history
creation, network transport, and training are excluded. Both models and the mapper are resident on the same device.

![Handoff latency](plots/qwen3_1.7b_to_0.6b/handoff_latency.png)
![Short-context crossover](plots/qwen3_1.7b_to_0.6b/short_context_latency.png)

Mean GSM8K stage accounting separates the source cost, which is identical in C/D/E:

| Condition | Source reasoning tokens | Target historical prefill tokens | Target generated tokens | Source time, s | Target prefill, ms | Mapper, ms | Target generation, ms | Sum of target stages, ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_small_only | 0.00 | 97.79 | 663.71 | 0.00 | 47.13 | 0.00 | 10908.65 | 11019.63 |
| B_large_only | 677.17 | 0.00 | 0.00 | 15.79 | 0.00 | 0.00 | 0.00 | 0.00 |
| C_text_handoff | 677.17 | 774.96 | 15.29 | 15.58 | 143.94 | 0.00 | 276.07 | 443.45 |
| D_kv_handoff | 677.17 | 0.00 | 48.00 | 15.58 | 0.00 | 46.22 | 853.72 | 923.94 |
| E_functional_kv | 677.17 | 0.00 | 46.29 | 15.58 | 0.00 | 30.95 | 796.50 | 850.68 |

Counts include the emitted token fed to keep the source cache current; target generated tokens in A include its own
thinking. Raw records separately include short-delimiter token counts/time. The reasoning-stage sums exclude the common
defensive cache clone before rendering and should not be read as complete wall-clock latency. The dedicated first-token
benchmark does include injection. Long incorrect generations can outweigh the prefill saving, so these are not
accuracy-preserving end-to-end speedups. No dollar, energy, or production-throughput claim is made.

Maximum *observed* MPS allocation during the primary functional sweep was 9.33 GiB
and driver allocation 11.07 GiB, with several mapper variants resident.
These are sampled allocations, not exact peak GPU memory or total-machine RAM usage.

## 7. Which is harder to transfer: K or V?

**V in this pair, by both reconstruction and functional ablation.**

| Full-rank map | Mean validation R² | Mean cosine |
| --- | --- | --- |
| k_content | 0.75 | 0.96 |
| k_post | 0.75 | 0.95 |
| normalized_k_content | 0.75 | 0.96 |
| normalized_k_post | 0.75 | 0.95 |
| normalized_v | 0.64 | 0.83 |
| v | 0.63 | 0.83 |

Using **native V + mapped K** gives KL 0.06 and 95.0% top-1 agreement.
Using **native K + mapped V** gives KL 0.95 and 65.0% agreement.
These are oracle diagnostics requiring real target prefill; they are not feasible handoff methods.

## 8. How strong is cross-layer alignment?

The all-pairs rank-128 screening selects exactly normalized-depth layers for 28/28 content-K targets
and 28/28 post-RoPE-K targets, versus 20/28 V targets. Most V deviations are nearby,
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
contexts, gradient clipping, and a validation-selected checkpoint. Validation KL fell from 1.361
to 0.628. This is evidence that the training objective matters materially, not evidence that a large
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
1.50 over positions 1–8 to 1.27 over positions 57–64;
functional-map KL changes from 0.64 to 0.59.
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


## Larger-gap replication: Qwen3-4B → Qwen3-0.6B

The source has 4,022,468,096 parameters, 36 layers,
8 KV heads and head dimension 128; the target retains 28 layers.
Tokenizer/backend/vocabulary compatibility and real returned caches were verified again. The same-model control passed
again before cross-model work. Target cache files were reused only after comparing exact selected-token hashes; the
source caches and all source-specific mappers were freshly fit. Both pairs use the same train/validation/test text tokens.
Only 7/28 content-K layer choices and 4/28 V choices match normalized depth exactly,
showing that the close pair's almost diagonal alignment does not generalize automatically to unequal depths.

| Condition | Mean KL | Top-1 agreement | PPL |
| --- | --- | --- | --- |
| native | 0.00 | 100.0% | 18.92 |
| normalized_content | 2.15 | 45.0% | 638.21 |
| content | 2.23 | 40.0% | 320.27 |
| post | 3.67 | 37.5% | 606.08 |
| no_context | 10.09 | 0.0% | 67.71 |
| zero | 11.12 | 0.0% | 675.86 |
| naive | 11.29 | 5.0% | 45999.45 |
| random | 14.89 | 0.0% | 4017746.33 |

The reasoning budget is 2,048 tokens and the sample is the first 12 of the primary run's seeded question ordering:

| Condition | Correct / total | Reasoning completed naturally |
| --- | --- | --- |
| A_small_only | 7/12 | 9/12 |
| B_large_only | 11/12 | 11/12 |
| C_text_handoff | 11/12 | 11/12 |
| D_kv_handoff | 1/12 | 11/12 |

This is a larger-gap replication with a longer reasoning allowance, **not a controlled model-size-only accuracy comparison**.
The D handoff uses `content`, selected between normalized and searched alignments by mean validation
reconstruction R² across K/V layers, before running its reasoning benchmark.
Its numeric-only score is 0/12; all
12 KV outputs reached the 48-token answer cap. It retains
1/11 of the text baseline's extracted-answer accuracy.
No nonlinear or functional-training refinements were run for the 4B pair. Its complete per-layer reconstruction, layer
screening, context sweep, oracle K/V ablations, repeated latency measurements, and raw reasoning records are retained.

![Larger gap quality](plots/qwen3_4b_to_0.6b/context_quality.png)
![Larger gap latency](plots/qwen3_4b_to_0.6b/handoff_latency.png)


## Failures, limitations, and reproducibility

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
