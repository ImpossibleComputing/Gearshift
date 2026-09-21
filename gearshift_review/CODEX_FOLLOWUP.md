# Gearshift: repair experiment identity, then run a bounded follow-up

We have completed an exploratory Gearshift experiment: Qwen3-1.7B or Qwen3-4B generates reasoning, a mapper translates its KV cache, and Qwen3-0.6B answers without replaying the generated reasoning. Both models stay frozen. A source review reconciled the key report numbers with the raw records but found experiment-identity bugs and identified a narrower next round.

Work in the existing Gearshift repository. Read the attached source review if available. Preserve the original measurements; do not silently replace them with improved runs. Build and run what the local hardware permits, using no paid services or external publishing. Do not claim results for experiments not executed.

## 0. Establish immutable pilot provenance

Record the current git commit and working-tree status. Create a source/config manifest with file hashes and record model, dataset, tokenizer, mapper, and dependency revisions. Preserve the original pilot results and their EOS correction/replay provenance as one immutable experiment. New configurations, checkpoints, or protocols need new run IDs and result locations.

Do not commit giant caches, model weights, environments, credentials, or proprietary material. The small source-reasoning trajectory JSONL files ARE required evidence and should be included in the review bundle, subject to the repository's dataset/publication policy:

- `data/qwen3_1.7b_to_0.6b/reasoning_trajectories.jsonl`
- `data/qwen3_4b_to_0.6b/reasoning_trajectories.jsonl`

Preserve large model/mapper artifacts locally and include hashes and regeneration instructions. No public GitHub push or blog publishing is authorized by this prompt.

## 1. Fix confirmed identity and resume bugs before any new scientific run

### Reasoning resume

`gearshift/reasoning.py` currently checks only mapper hash, two budgets, and handoff variant; it rewrites the manifest before loading/skipping completed rows. A reviewer reproduced this failure: start from the saved 24-question primary run, set `gsm_examples=12`, and rerun. No inference occurs, the new manifest lists 12 questions, and the old result file still contains 24 questions / 120 condition rows.

Define and validate a canonical reasoning experiment identity before any write. Include exact question IDs, dataset revision/fingerprint, model and tokenizer revisions, prompt/chat-template/delimiter token protocol, decoding and effective EOS settings, mapper/checkpoint identity, and budgets. Changing any identity component must fail clearly or create a new experiment. Do not silently filter, relabel, merge, or overwrite existing scientific records.

Assert uniqueness of `(question_id, condition)` and exact agreement between expected sample/conditions and completed records. Support explicit partial-progress states without presenting them as complete. Preserve original-versus-replay metadata; a no-op resume must not rewrite an original-run manifest with current settings.

### Token and KV caches

`gearshift/data.py` derives its root from only the output basename and trusts marker existence. Thus different output paths ending in the same directory name collide. The runner also fails to invalidate/reidentify runs after changes to dataset revisions, context/evaluation lengths, continuation length, or evaluation sample count.

Create validated identities for token selection, cache extraction, mapper training, and evaluation. Reuse only artifacts whose upstream identities match. Distinguish changes that require new training caches from those that require only new evaluation artifacts. Check shapes, token hashes, positions, tokenizer/model revisions, dtype, and source/target roles before accepting complete markers. A marker alone is not proof of valid extraction.

Add regression tests for at least:

1. 24 → 12 and 12 → 24 sample changes, and changed sample IDs with unchanged count.
2. Dataset revision/fingerprint changes.
3. Changed lengths/continuation/evaluation count.
4. Different output paths with identical basenames.
5. No-op resumption preserving original manifests and audit metadata.
6. Missing/incomplete cached files despite a completion marker.
7. Effective EOS or delimiter changes with otherwise identical configuration.

Run the real original unit suite and relevant actual-model cache controls locally. The source reviewer could run only four pure mathematical/parser tests, not the Transformers-dependent suite.

## 2. Clarify metrics without changing pilot scores

Keep the original 10/24 extracted-answer and 1/24 numeric-only functional-handoff scores. Do not retroactively choose a more favorable parser.

The headline 77.5% top-1 agreement is the first prediction after a shared bridge across 40 context/probe cases, not all 64 continuation predictions. Report both explicitly. Recomputed all-64 teacher-forced agreement is about 50.98% for normalized affine and 65.39% for the functional map. Keep block-level dependence explicit in uncertainty estimates.

The stopping replay preserved aggregate scores but changed correctness on two E questions: dataset ID 1199 false→true; ID 317 true→false. Report per-item correctness agreement, exact text agreement, token agreement, and aggregate scores separately. Inspect margins and stop behavior where practical. Do not claim the EOS replay proved identical output behavior or isolated the cause of numerical sensitivity.

For future runs, prespecify extracted-answer correctness, format validity, correct-and-valid output, cap/EOS rates, and any constrained-decoding evaluation. Apply identical grading and stopping rules to all comparable conditions. Update the existing report generator or explicitly freeze it as pilot-specific; it hard-codes 24 questions and some original narrative.

## 3. Run one bounded functional-training comparison

The existing primary mapper's validation KL improved at every checkpoint through step 64, approximately:

`1.361 → 1.095 → 0.998 → 0.893 → 0.800 → 0.738 → 0.684 → 0.646 → 0.628`.

The best checkpoint is the last one. We have not established a plateau or an affine-capacity ceiling.

Make training steps, learning rate, context-length mixture, continuation/prediction-token count, reconstruction weight, clipping, validation cadence, and early stopping configurable. Keep the language models frozen. Preserve the original affine checkpoint and hash it.

Compare, from the SAME affine initialization:

- Additional functional training on plaintext.
- Matched-budget functional training on chat-formatted source-generated reasoning trajectories drawn from training/development data disjoint from final test questions.

Match optimizer, context-length distribution, gradient/prediction-token budget, and selection rules as closely as possible, and log any unavoidable differences. Use enough length diversity to test beyond 128/512, including longer contexts when memory allows. Do not change domain, architecture, and training budget simultaneously and attribute the combined effect to one factor.

Choose modest fixed training budgets up front, with validation checkpoints and an explicit stopping rule. A progression such as 128/256/512 total steps is an acceptable starting bound; adapt to resources and record the choice before inspecting final test results. Do not turn this into an indefinite training job. Use both plaintext and chat/reasoning validation to observe tradeoffs. Select using a declared validation objective, not test GSM8K accuracy.

For task evaluation, reserve a fresh deterministic sample excluding all 24 already inspected pilot IDs. Aim for roughly 100–200 new questions if resources permit; a smaller actual sample is acceptable if clearly disclosed. Keep the final test sealed until settings are selected. Do not use official test answers to generate training histories or choose mapper settings. Source-generated conclusions may contain the answer; that is legitimate for this rendering task.

Give the source a larger declared thinking budget so most trajectories finish naturally, while retaining budget-limited cases and reporting completion flags. Do not select only naturally completed questions and claim performance on the full distribution. C and mapped conditions must use identical saved source token trajectories and matching bridge/scaffold tokens. Reuse source generations across conditions.

## 4. Small control-state recovery study

The current implementation already feeds a short target-native delimiter (`Answer:`). Simply adding that delimiter is not a new experiment.

After choosing a mapper using development data, test a small predefined set of additional instruction/scaffold options. Apply each scaffold equally to C/text handoff and the mapped handoff. Count actual tokenizer tokens, not words. Prespecify any grammar/stop rule rather than tailoring it to observed test answers.

Also implement one hybrid diagnostic:

```
native_target_KV(original question + chat prefix)
    concatenated with
mapped_source_KV(source-generated reasoning suffix)
```

Keep correct absolute positions and masks. When mapping a suffix, do not reset its RoPE positions to zero. A safe first implementation can map the full source history and then splice by absolute token position, explicitly counting the extra work; optimize only after correctness is tested.

This hybrid performs target prefill of the original prompt, but not the generated reasoning. Label and count that honestly. Measure all added target work. It may preserve conversational control more cheaply than replaying the entire reasoning sequence, but that is a hypothesis, not an assumed outcome.

New target-generated scaffold KV over approximate history is not identical to pristine full-native KV. If an oracle requires native target prefill of all history, label it as an oracle and exclude it from deployable timing claims.

Save representative successes and failures, numeric-only validity, repetition/cap/EOS behavior, and paired task scores. Do not declare success just because a parser can locate a correct number somewhere in malformed output.

## 5. Optional replication, not a gate to shipping

After selecting the method on development data, apply it to Qwen3-4B → Qwen3-0.6B with a comparable protocol and adequate reasoning budget. That pair currently has no functional-training refinement. Separate differing budgets from model-size effects.

Only pursue top-k multi-source/head-aware mappers, nonlinear adapters, or additional model families if the preceding work leaves a specific uncertainty that the new experiment resolves within the declared budget. A fair comparison to relevant published mappers is necessary before claiming superiority, but implementing all prior work is not a prerequisite for an explicitly exploratory blog release.

Do not start shared-ABI co-training, cross-family token alignment, quantized serving, distributed transport, or long multi-switch trajectories in this round.

## 6. Timing and release outputs

Record actual full wall-clock costs as well as stage timers. The pilot's stage sums omit defensive cache cloning, and mapper-only first-logit speedup is not end-to-end speedup. Maintain a large-only continuation baseline. Include longer user-facing generation only if quality becomes sufficiently stable for an informative economics comparison. Do not claim dollars/energy/throughput from local latency alone.

Export the selected mapper separately from the multi-variant research bundle. Report parameter count, precision, storage, resident memory, and whether both models are resident. Preserve the original full source-history accounting.

Deliver:

- Identity/resume patches and regression tests, with actual test output.
- Immutable original-pilot manifest and new experiment manifests.
- Updated code/configs and exact reproducible commands.
- New raw JSON/CSV records and source trajectories, plus small sample outputs.
- Learning curves, paired uncertainty estimates, and explicit complete/truncated groups.
- `FOLLOWUP_RESULTS.md`, separating observations, inferences, and untested hypotheses.
- A publication-ready review ZIP excluding giant tensors but including the two original trajectory JSONLs and all new small evidence files.

Before labeling the repo open source, flag that it needs an explicit project LICENSE; do not silently choose an owner license or alter third-party licenses. Add or propose a small CI/artifact-integrity workflow. Include verified related-work citations without novelty claims beyond the evidence. No external publishing until Keith authorizes it.

End with one of: **publish as partial/negative result**, **publish as promising optimization with caveats**, or **fix a blocking correctness issue before publication**. Explain the decisive evidence. Do not use an arbitrary retention percentage as a publication threshold, and do not keep expanding scope to avoid publishing a negative result.
