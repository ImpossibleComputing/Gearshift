# Gearshift source review

**Reviewed snapshot:** uploaded `gearshift.zip`, SHA-256 `54ee603d5ce9f5b92412b2aa8276702f06a0d5eb5fcdc641e1964d89b2ca1d53`.

**Recommendation:** fix experiment-identity/resumption defects, run one bounded quality-focused follow-up, and publish an explicitly exploratory v0.1 regardless of whether quality reaches native-prefill parity. Do not present the current system as an accuracy-preserving end-to-end speedup.

## Scope and confidence

I inspected the implementation modules, experiment/configuration paths, report and plotting scripts, tests, README, and machine-readable records. I independently recomputed the main metrics from 1,000 text-evaluation records and 168 reasoning-answer records across the two pairs. I inspected all 24 primary functional-handoff answer strings and compared the saved stopping audit against the original runs.

The aggregate accuracy, strict-format accuracy, next-token KL, and perplexity in the report reconcile with the raw records. Computing perplexity from individual token NLLs rather than saved FP32 per-case means produces only expected tiny floating-point differences; all checked values agree within relative tolerance 1e-6. There are no duplicate condition/example observations, saved source-trajectory hashes agree across B/C/D/E, and the manifest's selected question IDs match the uploaded answer records.

Four original pure numerical/parser test functions pass when executed in isolation from their unchanged source ASTs. All Python files compile. I also reproduced the resumption/configuration problems below using the original functions with explicit fixtures replacing only external dataset/model-loading endpoints.

**Not independently rerun:** the actual Qwen inference, real DynamicCache reinjection tests, learned checkpoint evaluation, MPS timing, or the complete seven-test suite. This environment has CPU PyTorch but lacks Transformers and Datasets; dependency installation failed because network name resolution was unavailable. The uploaded archive intentionally excludes model/cache/mapper tensors. The tests here are not a substitute for the original real-model controls.

Evidence files: `record_audit.json`, `regression_results.json`, `source_manifest.json`, `analyze_records.py`, and `review_regressions.py`. The original repository was not patched.

## 1. Confirmed defects to fix before follow-ups

### A. Reasoning resume can contradict its own manifest

**Locations:** `gearshift/reasoning.py:58–80`, `gearshift/reasoning.py:82–84`.

The resume check validates mapper hash, reasoning/answer budgets, and handoff variant, but not the exact selected question IDs, dataset identity, or all decoding/protocol metadata. It writes the new manifest before deciding whether old results can be reused. Old rows are not globally restricted to the newly selected sample.

**Reproduced example:** initialize with the uploaded 24-question/120-answer primary run and request `gsm_examples=12` with otherwise matching settings. The original benchmark returns without doing inference, writes a manifest listing 12 questions, and leaves `reasoning.json` containing 24 questions/120 answers. The summarizer will still report the larger sample.

This also risks overwriting useful original-run provenance when a no-op resume rewrites a manifest using current settings. In this snapshot, the original EOS settings and later replay-audit fields were explicitly preserved in the final manifest; unconditional regeneration can remove those annotations.

**Fix:** define a canonical experiment identity containing dataset revision/fingerprint, exact selected indices, source/target revisions, mapper identity, actual prompt/delimiter token protocol, decoding/EOS settings, and budgets. Validate it before any artifact write. Reject incompatible resumption or create a new immutable experiment. Do not silently relabel old rows. Assert exact agreement between declared sample IDs and records at summarization time.

**Effect on current results:** no evidence that this bug contaminated the uploaded run. Its saved IDs and rows match. It is a concrete risk to the next experiments and public reproducibility.

### B. Cache identity is too weak

**Locations:** `gearshift/data.py:12–16`, `gearshift/data.py:62–66`, `gearshift/runner.py:17–24`.

Cache directories use only the basename of the output path: `data/<output-directory-name>`. Thus `results/experiment-A/same-name` and `results/experiment-B/same-name` reuse the same data directory. `prepare_tokens` returns immediately if `tokens.json` exists; extraction similarly trusts completion markers without validating their full identity.

The runner rejects several important changes, but not dataset revisions, context/evaluation lengths, continuation length, or evaluation sample count. I reproduced acceptance of changed WikiText/GSM8K revisions, `eval_examples`, `continuation_tokens`, and `lengths`.

**Fix:** content-address or explicitly namespace data caches using a validated extraction identity. Include model and tokenizer revisions/hashes, dtype, exact token hashes, source/target role, positional convention, and extraction parameters. Token-selection identity should also include dataset revision/fingerprint, split and selected block indices, seed, and lengths. Check completion metadata and files rather than treating marker existence as sufficient. Changes affecting only evaluation should receive a new evaluation identity without needlessly invalidating valid training caches.

## 2. The core handoff implementation is consistent with the intended experiment

**Locations:** `gearshift/core.py:131–157`, `gearshift/core.py:99–105`, `gearshift/mapping.py:26–64`, `gearshift/reasoning.py:96–113`.

The explicit source generation loop feeds each emitted token, including its final token, into the source cache. Source-history state is therefore not left one token behind. Cache construction clones state for independent conditions where needed. Forward calls set past-relative positions and total attention-mask lengths explicitly. Mapped conditions receive the common new delimiter, not historical token IDs, and runtime counters check that invariant.

The content-space K route removes source RoPE before the affine transformation and applies target RoPE afterward. In functional training, both language models remain frozen while mapper weights and biases receive gradients through the target. Teacher-forced reference/candidate prediction positions are aligned in the inspected code.

I did not find a source-level off-by-one, obvious answer-scoring fabrication, or hidden target history-prefill that explains away the reported results. This is a bounded inspection conclusion, not proof that every possible implementation error is absent.

## 3. Metric interpretation needs to stay precise

**Location:** `gearshift/evaluation.py:74–105`.

The headline top-1 metric compares the first prediction **after a common one-token bridge**. It is 40 context/probe cases made from eight independent text blocks, not agreement over every generated token.

Independent recomputation for Qwen3-1.7B → Qwen3-0.6B:

| Condition | First-probe top-1 agreement | All 64 teacher-forced predictions | Continuation PPL |
|---|---:|---:|---:|
| Native target cache | 100.0% | 100.0% | 18.92 |
| Reconstruction-trained affine | 60.0% | 50.98% | 85.29 |
| Functionally trained affine | 77.5% | 65.39% | 31.74 |

The second agreement column is an additional descriptive calculation, not an independent large-sample estimate: endpoints and continuations repeat across context lengths. The existing 77.5% claim is correctly defined in the report, but a blog must not shorten it to “77.5% of all continuation tokens.”

The no-context condition has continuation PPL 67.71, better than the reconstruction-trained affine map's 85.29, though its first-probe agreement is worse. Thus the linear baseline does not outperform absent history on every functional metric. The functionally trained map improves materially over both on continuation PPL.

## 4. New details from the raw records

### Functional training has not demonstrated a plateau

**Locations:** `gearshift/functional_training.py:50–81`; `results/qwen3_1.7b_to_0.6b/functional_training.json`.

Validation KL at steps 0, 8, 16, 24, 32, 40, 48, 56, and 64 is approximately:

`1.361, 1.095, 0.998, 0.893, 0.800, 0.738, 0.684, 0.646, 0.628`.

The best checkpoint is the final checkpoint, and each measured validation point improves. This supports testing additional bounded training before changing mapper architecture. It does not guarantee that more training will preserve reasoning behavior or approach parity.

The functionally trained mapper's mean V reconstruction R² is 0.60746 versus 0.60795 before training, while behavior improves. K R² also changes only slightly. This supports the distinction between reconstruction quality and functional quality, not a theorem that reconstruction is useless.

### Completion and answer-control failures are entangled

Primary functional handoff answers 8/10 questions correctly when the source naturally finishes its reasoning, but only 2/14 when the source reasoning hits its budget. These are selected subsets; naturally completed problems may be easier. This does not causally establish that increasing the budget will yield 80% accuracy overall.

Raw outputs include correct boxed numbers followed by repeated `Answer:` sections, and correct numbers followed by unrelated “Review of the Question” prose. Some outputs are substantively wrong as well; this is not solely a cosmetic formatting problem.

Exploratory first-line parsing changes E from 10/24 to 11/24 and C from 17/24 to 18/24. Those are post-hoc diagnostics, not replacement benchmark scores. The principled next test is to predefine stopping/format constraints and apply the same rules to all conditions on fresh questions.

### Aggregate stopping-audit stability hides two correctness flips

**Locations:** the primary pair's `reasoning.json` and `stopping_audit.json`.

The replay audit preserves aggregate scores and cap counts. However, functional-handoff question 1199 changes from incorrect to correct, while question 317 changes from correct to incorrect. They cancel in the total. This is consistent with the report's disclosed sensitivity to FP16 full-prefill versus incremental source histories, but that explanation has not been independently isolated here.

Report aggregate accuracy, per-question correctness agreement, exact-text agreement, and token-level disagreement separately. “Same aggregate accuracy” is not “same answers.” In the replay for 1199, a stray numbered-list marker makes the last-number extractor mark an answer correct, underscoring why extracted-answer accuracy and reliable final-answer behavior need separate reporting.

## 5. How I would revise the next experiment

### First: fix identity and preserve the original baseline

Add regression tests for the issues above. Archive the existing measurements as an immutable pilot, with code/config/data/checkpoint hashes and an explicit account of the EOS fix/replay. Do not repeatedly overwrite the original tables while iterating.

### Second: improve the same affine model on controlled data/objective settings

Start from the same recorded affine initialization. Compare longer functional training on plaintext against matched-budget training on disjoint chat-formatted, source-generated reasoning trajectories. Use the same optimizer settings, context-length mixture, and prediction-token budgets wherever possible. Select checkpoints on validation only. Make steps, context lengths, continuation lengths, learning rate, and reconstruction-loss weight configurable rather than hard-coded.

Use a predeclared modest budget and validation-based stopping rather than an indefinite “train to convergence” assignment. Keep a plaintext suite to detect forgetting. Lock new test IDs excluding the 24 already inspected pilot questions and freeze settings before final evaluation. Approximately 100–200 new questions is a sensible scoped target, subject to available compute; report the actual count and paired uncertainty.

Give the thinker enough budget to finish most selected tasks, but preserve completion status and report both whole-sample and diagnostic subgroup results. Do not evaluate only the easy naturally completed subset.

### Third: separate semantic transfer from conversational control

The current implementation already feeds a target-native `Answer:` delimiter. Merely adding a delimiter is not a new experiment. Test additional instruction/scaffold variants against that baseline and apply each scaffold equally to text and KV handoffs.

A more diagnostic hybrid is:

`native target cache(original question + chat prefix) || mapped source cache(reasoning suffix)`

This preserves exact target-native prompt/control state while still avoiding replay of the generated reasoning. Retain absolute positions when selecting the mapped suffix; do not rotate a suffix as though it began at zero. This is **partial prompt prefill**, not zero target historical prefill. Count and time those tokens explicitly.

Also, newly computed suffix/scaffold KV on top of approximate historical KV is not necessarily “pristine”: its hidden states depend on the approximate history. A full native suffix built by replaying all history would instead be an oracle diagnostic, not a deployable optimization.

The stronger 4B pair is a useful subsequent replication after choosing the method on development data. A multi-source/head-aware affine baseline is worthwhile for a stronger comparative research claim, but it need not block a carefully scoped exploratory blog release.

### Correct the task framing

If a large model's own reasoning contains the answer, transferring that conclusion is legitimate for a **thinker → renderer** experiment. It does not prove that the smaller model independently acquired the large model's reasoning ability. That distinction should be explicit, but forcing an answer-free trace is not necessary for this use case. Providing benchmark ground truth to the source/mapper at test time would be a different and invalid contamination.

## 6. Economics and claims

The 4K handoff benchmark measures about 683 ms for native target first logits versus 143 ms for functional cache transfer plus first logits, with the source history already available. The primary reasoning stage sums are approximately 15.795 s for large-only, 16.023 s for text handoff, and 16.431 s for functional KV handoff. These sums are not complete wall-clock measurements: defensive cache cloning is excluded, as the report acknowledges. They do not demonstrate an end-to-end speedup.

Most inference in this short-answer task is still source reasoning. Once fidelity improves, long-form answer generation is the more appropriate next economics test. Do not turn mapper-only speedup into a claim of overall compute, energy, dollar, or throughput savings.

One active single-source affine mapper has 58,777,600 scalar weights/biases for the primary geometry, about 224.2 MiB in FP32. The saved multi-variant checkpoint bundle is much larger. Export and benchmark only the selected mapper for deployment measurements, and report its parameter count/storage separately.

## 7. Publication readiness

The repo currently contains no project LICENSE file, and the original source-reasoning JSONL files are excluded with `data/`. Keep large caches/weights excluded, but include or release the small trajectory manifests required to reproduce paired handoffs and audit the source histories. Preserve source/dataset IDs, token IDs, hashes, decoding settings, and model revisions. Add a small artifact-integrity check and CPU test workflow.

The report-generation script `scripts/write_report.py` contains experiment-specific narrative and a hard requirement for 24 primary questions. Treat it as the frozen pilot report generator or parameterize it with assertions, rather than reusing it unmodified after changing the protocol. The current numbers reconcile; the risk is stale narrative in future reports.

A limited high-confidence credential-pattern scan of textual files found no matches. This was not a comprehensive secret, confidentiality, dependency, or licensing audit.

I verified the existence and primary-source abstracts of these related works on 2026-09-13: Heo et al., *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse* (arXiv:2608.03893); Qu et al., *CacheBridge: Efficient Cross-Model KV Cache Transfer* (arXiv:2609.00891); Li et al., *A Universal Context-Reuse Layer for Cross-Model KV Sharing* (arXiv:2608.30963). This is not an exhaustive novelty review. Gearshift should be framed as an open empirical investigation of generated-reasoning handoff, not the invention of cross-model cache transfer.

**Publication gate:** reproducible artifacts, validated experiment identities, clearly defined metrics, fair controls, and a bounded claim. An arbitrary retention percentage is not a publication threshold. Publish success, partial success, or failure after the scoped follow-up; do not wait indefinitely for a benchmark victory.

## Additional files for the next review bundle

```
data/qwen3_1.7b_to_0.6b/reasoning_trajectories.jsonl
data/qwen3_4b_to_0.6b/reasoning_trajectories.jsonl
```

These small files, plus the corresponding new-run trajectories, are sufficient to close the most important missing source-history evidence. There is no need to send the tens of GiB of paired caches for the next source review.
