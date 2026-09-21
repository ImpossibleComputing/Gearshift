# Gearshift bounded follow-up

This is a local exploratory Qwen3-1.7B → Qwen3-0.6B thinker-to-renderer experiment. Both language models remain frozen. Original pilot evidence and scores are unchanged in `RESULTS.md` and `evidence/pilot/snapshot/`; this report describes separate run `followup_v1`.

## Observations: correctness and identity repairs

Canonical reasoning identities cover exact question IDs, resolved dataset fingerprint/bytes, model/tokenizer revisions, prompt and delimiter token IDs, effective EOS settings, decoding, budgets and mapper hash. Configuration drift fails before scientific records are written. No-op reasoning resume preserves manifest bytes and audit annotations. Duplicate/out-of-sample records fail; partial progress is labeled explicitly. New task records use atomic per-question transactions.

Token selection and extraction have separate content identities. Output paths with equal basenames have distinct views. Evaluation-only changes can reuse validated training selections/extractions. Cache markers are checked against every required file's shape, precision, size and SHA-256, and against model/role/position/upstream-selection identity. Existing pilot directories are protected from runner writes. The old narrative generator is explicitly frozen.

Actual unit output is in `evidence/followup/pytest_final.txt`: **39 tests passed**, including the original seven-test suite. Real MPS model controls are in `results/followup_v1/controls_final/` (earlier attempts are also retained). Both source and target pass disk reinjection through 4096 tokens with zero logit difference and identical greedy continuations. Hybrid native-cache splicing passes exact-position controls. A too-short synthetic hybrid fixture failed in the first control attempt, was corrected, and both logs are retained. Initial pytest collection also encountered the archived duplicate test filename; `pytest.ini` now limits collection to the active test directory. Corrupted extraction bytes are rejected despite their completion marker.

## Prespecified comparison and selection

Both arms start from the same exported normalized affine initialization (original checkpoint SHA recorded). Adam learning rate is 1e-05; each arm executed 128 steps and 1024 gradient prediction tokens. Context lengths are 128/512/1024/2048 with an identical balanced schedule; each update predicts 8 teacher-forced tokens. Reconstruction weight is 0.01 and gradient clipping is 1.0. Validation occurs every 16 steps. The fixed upper bound is 128 steps, with joint early stopping only if both arms fail the prespecified improvement/patience rule; nonfinite gradients abort both. Stop: fixed 128-step bound.

Chat histories are generated from 48 official GSM8K training questions; 12 separate training-split questions supply development validation. No official answers enter generation/training histories. Intact conversations are packed to make long contexts possible; sampled windows start at conversation prefixes and may cross conversation boundaries. Plaintext uses WikiText training/validation blocks. This is an unavoidable domain/packing difference, logged per case; architecture, initialization, optimizer, context schedule and prediction-token budgets are matched. Four windows per domain/length form development validation. Reused/nested windows are dependent, and the development set is small.

Checkpoint selection minimizes equal-weight plaintext/chat validation KL, never final-test task scores. Selected arm: **chat**, step **128**.

| Training domain | Selected step | Mixed validation KL |
| --- | --- | --- |
| plaintext | 128 | 0.6732 |
| chat | 128 | 0.5626 |

![Learning curves](plots/followup_v1/learning_curves.png)

The final deterministic sample contains 100 questions, excludes all 24 inspected pilot IDs, and is fixed in the experiment manifest. Its questions/answers are first used after selection is saved. The source reasoning cap is 2048 tokens, versus 768 in the primary pilot. C, M and H share identical saved source token trajectories and identical scaffolds within each comparison; all completed and capped source histories are retained.

| Split | Questions | Natural </think> | Capped | Mean reasoning tokens |
| --- | --- | --- | --- | --- |
| train | 48 | 41 | 7 | 1020 |
| validation | 12 | 8 | 4 | 1181 |
| test | 100 | 73 | 27 | 1184 |

## Observations: fresh paired task results

`C_text` re-prefills all question + source reasoning through the target. `M_mapped` maps the full source cache without target historical prefill. `H_hybrid` additionally prefills the original question/chat prefix in the target, then splices the full-history mapped suffix at its original absolute positions. Full-history mapping work is counted. It is partial prompt prefill, not zero target prefill, and target scaffold KV over approximate history is not an oracle native cache.

Scaffolds are prespecified `answer` (existing Answer: delimiter), `instruction` (explicit numeric-only request), and `assistant_turn` (a new user instruction and assistant chat prefix). Exact IDs and actual token counts are saved in `task_manifest.json` and each row. Forced closing tokens are separately counted. All conditions use greedy generation, the model's full EOS set, and a 64-token answer cap. No grammar-constrained decoding or post-hoc parser selection was run.

Extracted correctness retains the pilot parser. Format validity requires only a signed integer/decimal with valid thousands grouping and optional whitespace; boxed expressions, currency symbols, prose and trailing periods are invalid. This prespecified raw-number grammar is stricter than the pilot's format regex, which allowed some boxing/currency/punctuation. New format rates must not be compared directly to the pilot's 1/24 score. Correct boxed math can be usable to a human while failing this machine-readable string contract; format-invalid does not automatically mean mathematically wrong. Correct-and-valid requires both. EOS at exactly the token limit is an EOS stop, while `cap_length` is also retained as a separate count.

| Condition | Extracted correct | Format valid | Correct + valid | EOS / cap | Mean online wall estimate (s) |
| --- | --- | --- | --- | --- | --- |
| B_large_only | 81/100 | 49/100 | 38/100 | 96 / 4 | 25.79 |
| C_text/answer | 79/100 | 40/100 | 33/100 | 88 / 12 | 25.99 |
| M_mapped/answer | 49/100 | 0/100 | 0/100 | 0 / 100 | 26.66 |
| H_hybrid/answer | 65/100 | 19/100 | 9/100 | 77 / 23 | 26.06 |
| C_text/instruction | 81/100 | 45/100 | 38/100 | 89 / 11 | 25.99 |
| M_mapped/instruction | 19/100 | 0/100 | 0/100 | 0 / 100 | 26.67 |
| H_hybrid/instruction | 73/100 | 11/100 | 8/100 | 82 / 18 | 25.98 |
| C_text/assistant_turn | 76/100 | 51/100 | 38/100 | 90 / 10 | 26.01 |
| M_mapped/assistant_turn | 78/100 | 96/100 | 78/100 | 100 / 0 | 25.70 |
| H_hybrid/assistant_turn | 81/100 | 99/100 | 81/100 | 100 / 0 | 25.73 |

![Fresh task quality](plots/followup_v1/task_quality.png)

| Paired comparison | Correct + valid difference (pp) | Paired bootstrap 95% CI (pp) |
| --- | --- | --- |
| M_mapped/answer − C_text/answer | -33.0 | [-42.0, -24.0] |
| H_hybrid/answer − C_text/answer | -24.0 | [-34.0, -15.0] |
| M_mapped/instruction − C_text/instruction | -38.0 | [-48.0, -29.0] |
| H_hybrid/instruction − C_text/instruction | -30.0 | [-40.0, -21.0] |
| C_text/instruction − C_text/answer | +5.0 | [+1.0, +9.0] |
| M_mapped/assistant_turn − C_text/assistant_turn | +40.0 | [+29.0, +51.0] |
| H_hybrid/assistant_turn − C_text/assistant_turn | +43.0 | [+33.0, +53.0] |
| C_text/assistant_turn − C_text/answer | +5.0 | [+1.0, +10.0] |

Bootstrap draws resample paired questions, not condition rows or tokens (5000 draws). Individual rate intervals are Wilson intervals. These predefined comparisons are exploratory and intervals are not adjusted for multiple comparisons. Complete/truncated source groups are explicitly tabulated in `completion_groups.csv`; these are selected diagnostic subgroups, not a causal estimate of more thinking. Full per-condition differences for all three metrics are in `paired_uncertainty.csv`. Representative correct/valid, extracted-correct but format-invalid, and wrong outputs are selected by dataset ID in `sample_outputs.json`; all raw outputs and source trajectories are included.

## Timing and mapper artifact

Condition wall timers include native prefill or mapping, hybrid splicing, defensive cache cloning, scaffold inference and answer generation, with MPS synchronization. Each online end-to-end estimate adds the single measured live source-generation wall cost to that complete condition wall; source generation is shared across experimental arms. Full experiment wall costs are also recorded, separately from per-stage timers. Condition order is deterministically randomized by question; both models remain resident. These are local short-answer latencies, not dollars, energy or throughput. The large-only continuation baseline is retained.

Full command costs, including model loading:

| Command | Model loading (s) | Full command wall (minutes) |
| --- | --- | --- |
| prepare | 1.21 | 22.89 |
| train | 1.18 | 4.50 |
| task | 1.16 | 50.90 |

These three commands total approximately 78.3 minutes, excluding the separately logged controls, sensitivity diagnostics, software work and artifact checks. No long-form economics comparison was added. Timers describe this prototype with defensive clones; an optimized standalone baseline need not retain those clones. No serving-system throughput claim follows from these measurements.

The separately exported `selected_mapper.pt` has 58,777,600 affine weights/biases in float32, 224.2 MiB of scalar storage and 224.3 MiB on disk. It is kept locally and excluded from the compact ZIP; its SHA-256 is `b36ed31a43f95e7ce80282f5f0d44394da5adead2789aed538f5ea260c017d15`. `selected_mapper_memory.json` reports measured RSS/MPS deltas with both models resident, and distinguishes allocator observations from theoretical parameter storage. The research initialization, both best checkpoints, original multi-variant bundles and cache tensors are also preserved locally.

## Pilot metric clarification (scores unchanged)

The primary pilot remains 10/24 extracted-correct and 1/24 correct numeric-only for functional handoff. First-prediction agreement after a shared bridge is 60.0% for normalized affine and 77.5% for functional, over 40 context/probe cases from eight independent text blocks. All 64 teacher-forced predictions average 50.98% and 65.39%, respectively. They are different metrics; uncertainty resamples the eight blocks together with their nested context lengths. Native PPL is 18.92, affine 85.29, functional 31.74, and no-context 67.71.

The stopping replay preserves aggregate scores, but E correctness agrees on 22/24 questions and text on 19/24. IDs 1199 (false→true) and 317 (true→false) cancel. All 24 E answer token counts agree; **exact original-versus-replay answer-token agreement is unavailable because original live token IDs were not stored**. Decoding with skipped special tokens cannot recover them. Per-item audit evidence is in `pilot_metric_audit.json`. This does not prove identical behavior or isolate the cause of numerical sensitivity.

## Inferences and unresolved questions

The new assistant-turn scaffold is the useful finding in this bounded study. Pure mapped handoff changes from 49 extracted-correct / 0 correct-and-valid answers with the existing Answer: delimiter to 78 / 78 with the new turn. Merely adding a numeric-only instruction yields 19 / 0. Hybrid prompt prefill with the new turn yields 81 / 81. Both mapped variants stop on EOS for all 100 questions, versus 100 answer-cap hits for pure mapping under either earlier scaffold. These are paired comparisons with the same selected mapper and saved source histories; improved validation KL alone did not solve answer control.

Under the matched new-turn scaffold, native text handoff is 76/100 extracted-correct, pure mapping 78/100, and hybrid 81/100. The paired extracted-accuracy differences against native are +2 percentage points (95% bootstrap interval −5 to +9) and +5 points (−1 to +11). This sample does **not establish accuracy equivalence or superiority**. The large correct-and-valid gains (+40 and +43 points) combine semantic correctness with adherence to the prespecified raw-number grammar. Native outputs often contain correct boxed numbers or explanations; this is not evidence that the native model cannot do the math. The three scaffolds were specified before testing, and no further scaffold or parser was chosen to improve the reported scores.

Hybrid prefills an average 85.16 original-prompt tokens, versus 1268.76 historical tokens for C; pure mapping prefills none. All three still receive the same 24-token new-turn scaffold plus the identical two-token forced closure on the 27 capped source histories. Mean full handoff-and-answer wall time is 450.1 ms for C, 134.5 ms for M, and 167.2 ms for H. Median time through the first answer logits is 179.8 / 69.8 / 97.7 ms. These are local, resident-model prototype observations. Estimated total online wall time is 26.015 / 25.699 / 25.732 seconds: only about 1.2% / 1.1% lower than C, and 25.792 seconds for the large-only baseline. This is a promising handoff optimization, not a large end-to-end economics improvement.

Source reasoning ends naturally on 73 questions and hits its cap on 27. With the new-turn scaffold, correct-and-valid outcomes are C 27/73, M 68/73, H 70/73 among natural completions, and C 11/27, M 10/27, H 11/27 among capped cases. Native boxed/explanatory outputs explain much of its natural-completion format loss; harder or unfinished source reasoning limits all variants in the capped group. These groups are selected by source behavior, so the contrast is not a causal estimate of increasing the reasoning budget.

Chat training improves chat validation KL more than plaintext training (0.517 versus 0.944 at step 128), while plaintext training is better on plaintext (0.402 versus 0.608). Both improve on their shared initialization, and both best mixed-validation checkpoints are at the fixed final step. This supports the matched domain comparison but not a plateau/capacity claim. Packing and window-anchor differences accompany the domain change; the experiment does not isolate reasoning content from every formatting effect.

The corrected two-question diagnostic (`pilot_sensitivity_v2`) uses the main generator's argmax tie rule. Full source-history prefill reproduces the previous replay texts for IDs 1199 and 317; incremental replay of the saved source tokens reproduces their original live texts. Holding cache construction fixed, the two EOS sets produce identical output tokens in both cases. Changing construction flips the answers under either EOS set. First token divergence occurs at answer positions 7 and 17 (zero-based), with top-two logit gaps of 0 or 0.015625. This controlled diagnostic supports cache-construction/FP16 near-tie sensitivity in these cases. Original live answer token IDs and original live KV captures are still unavailable, so it does not retrospectively prove token-identical original states or generalize to every pilot item. The initial diagnostic's topk token selection was corrected; both versions and their source hashes are retained.

Remaining limits are the small development set, one fresh test sample and model pair, one training seed, a strict short numeric-output task, FP16 sensitivity, no convergence result, no fresh small-only baseline, and no external mapper replication. The selected mapper's measured MPS allocation increase is approximately 224.2 MiB. RSS fell slightly during loading due to allocator/process effects; that is not a negative or zero mapper-memory claim. Original and new weights/caches remain local and hash-addressed.

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

**Publish as promising optimization with caveats.** The decisive evidence is the predefined new-turn scaffold producing 78/100 pure-mapped and 81/100 hybrid correct-and-valid answers with zero answer caps, alongside lower complete handoff cost and validated zero/partial target-history prefill. The paired extracted-accuracy intervals, weak results under the other scaffolds, limited total-latency change and numerical sensitivity bound the claim. This is a recommendation for a future exploratory release after the owner resolves the project LICENSE and upstream release notices; nothing has been published or pushed.
