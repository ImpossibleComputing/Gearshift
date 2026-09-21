# Post-Progress 01 diagnostic results

This separate, exploratory diagnostic pass does not revise the frozen article or historical results. The editorial master remains unchanged. This file reports actual records; prepared code is not counted as an executed experiment.

Publication tag: `gearshift-progress-01` → `64725974fa55459350d1c9d09037bab64d0c5ec6`. Local only; not pushed or published. Branch: `research/post-progress-01`. Full subsequent commit identities are in `evidence/coding_pilot_v1/post_progress01/current_git_verification.json`.

## Frozen result and source audit

Progress 01: A=34/40, B=27/40, M=3/40, D=34/40, initial mapper=0/40. Selected update 96 had 3,072 supervised prediction positions. Validation kept improving; no plateau or trained-checkpoint ranking failure was established.

The all-40 visible-prompt/code audit confirms 13 of 16 runtime failures directly cite missing explicitly requested entrypoints. A fourteenth omits its interface but fails earlier on input parsing. The owner-named tree example ignores the requested vertex weights and prints the maximum unweighted distance sum, whereas the prompt asks for the minimum weighted sum. This is a selected explanatory example, not a prevalence estimate. Exact visible prompt/code and hashes are preserved. No output was repaired before scoring.

## Actual-model numerical path comparison

- `post_progress01_numerical_20260917_03`: 8 native/mapped history checks, complete=True, strict path comparison passed=True.
  atcoder/abc320_a, native, 926 history tokens: manual/deployed maximum logit difference 0; gradient difference 0; deployment batch/single-token KL 1.6600559e-09.
  atcoder/abc320_a, mapped, 926 history tokens: manual/deployed maximum logit difference 0; gradient difference 0; deployment batch/single-token KL -9.8077715e-09.
  atcoder/abc333_b, native, 4575 history tokens: manual/deployed maximum logit difference 0; gradient difference 0; deployment batch/single-token KL 1.536673e-07.
  atcoder/abc333_b, mapped, 4575 history tokens: manual/deployed maximum logit difference 0; gradient difference 0; deployment batch/single-token KL 0.00011498101.
  leetcode/3114, native, 9409 history tokens: manual/deployed maximum logit difference 0; gradient difference 0; deployment batch/single-token KL 0.00064251811.
  leetcode/3114, mapped, 9409 history tokens: manual/deployed maximum logit difference 0; gradient difference 0; deployment batch/single-token KL 0.00044338894.
  leetcode/3091, native, 20636 history tokens: manual/deployed maximum logit difference 0; gradient difference 0; deployment batch/single-token KL 0.00026709909.
  leetcode/3091, mapped, 20636 history tokens: manual/deployed maximum logit difference 0; gradient difference 0; deployment batch/single-token KL 0.00066260574.

The hand-written differentiable path exactly matches the deployed receiver for the same multi-token calls, including default/explicit causal masks and checkpoint-on/off controls. Nonzero, finite tested cache gradients match exactly; caches remain unchanged and clones are isolated. This weakens an implementation mismatch explanation on these four histories. The deployed model itself changes logits when executing the same answer as a batch versus one token at a time: execution-shape numerical sensitivity, observed under both cache kinds, with unchanged top1 at every checked position. This is not a defect unique to the handwritten path. It may alter sampled trajectories, and exact distributional equivalence or causal irrelevance is not asserted. No tolerance was loosened and no numerical code repair is justified by this check.

Maximum checked per-position batch-versus-token KL: 0.0028840557; all 32 checked top-ranked predictions match. This is measured sensitivity, not a statement that sampled trajectories must match.

Gradients cover the final two first-layer value-cache positions and four early continuation tokens, not all keys, layers or mapper parameters. Repeated-path controls set the comparison envelope. Batch versus one-token deployment rounding is recorded separately and is not used to excuse a manual-path discrepancy.

## Frozen prompt-preservation comparison

| Arm | Passes | Missing requested entrypoint | EOS / cap |
|---|---:|---:|---:|
|M|3/40|14|38/2|
|H|18/40|0|37/3|
|D|34/40|0|40/0|
|P|17/40|3|39/1|
|A_original|34/40|0|40/0|
|B_original|27/40|0|40/0|

| Contrast | Difference, pp | 95% paired interval, pp | Gains / losses |
|---|---:|---:|---:|
|H-M|+37.5|[+20.0, +55.0]|16/1|
|H-D|-40.0|[-55.0, -25.0]|0/16|
|H-P|+2.5|[-7.5, +12.5]|3/2|

10000 joint paired task bootstrap resamples; seed20260917; percentile95%; exploratory, unadjusted. Nonsignificance is not equivalence.

| Arm | Prompt/history prefill tokens | Native prefill, s | Mapping + splice, s | Answer, s | Source-inclusive estimate, s |
|---|---:|---:|---:|---:|---:|
|M|0.0|0.000|0.066|25.78|690.92|
|H|509.0|0.051|0.071|32.18|697.38|
|D|10033.5|1.142|0.000|19.32|685.54|
|P|512.0|0.068|0.000|21.58|21.65|

Times are per-task arithmetic means on the matched worker. Answer timing includes the bridge and periodic telemetry/checkpoint overhead. Source-inclusive estimates reuse each exact saved source-reasoning duration for M/H/D; P has no source generation. Reconstructing the source cache for this diagnostic is recorded separately in the CSV and excluded from the estimate of a handoff with that cache already resident. H saves native prefill work relative to D but has lower correctness and longer answers: this establishes no equal-quality speedup.


Retaining the native prompt improved the observed pass rate relative to pure mapping. This supports prompt-state preservation as a useful intervention in this fixed development sample, without establishing a unique cause of the original failures.
The hybrid exceeded prompt-only answering in observed aggregate passes; the paired interval above determines how uncertain that difference is. A positive point estimate alone is not proof of transferable reasoning or generalization. The optional wrong-suffix control was not part of the primary comparison.

H retains the complete original native prompt cache and inserts the mapped suffix at its original absolute positions. H includes native prompt prefill and mapping/splicing costs. M replays the same original draw for matched timing and reuses its old score only after exact token equality. Fresh D must also reproduce its original answer tokens exactly; changed draws block baseline reuse. P uses the pinned non-thinking template with no separate thinking stage requested; independently thinking B is separate. Native/native cache-surgery controls and partial-prefill rounding diagnostics are preserved. New answer states continue to depend on historical cache. H exceeding M alone does not establish a benefit from transferred reasoning; H versus P is essential. H failure would not exclude prompt corruption because mapped suffix states also depend on source prompt representations.

Native/native whole-history cache splicing is exact in the recorded controls: True. Separately prefilling only the native prompt changes execution shape; its native/native splice has maximum checked bridge-logit difference 0.5234375 and maximum bridge KL 8.8639851e-09. These bridge measurements do not establish equality of later sampled continuations. Full metrics and source hashes are in native_splice_summary.json.

### Selected semantic examples

Three cases declared before new outputs: owner-named tree objective error, missing XOR interface, missing alternating-cost interface. Qualitative, not a representative sample or prevalence estimate. Static inspection only: no generated code executed, repaired, renamed, re-extracted for primary scoring, or rescored.

- atcoder/abc348_e: Native prompt preservation recovers recognizable objective cues but not a correct algorithm in this example. M: Ignores C weights and prints maximum unweighted distance sum, changing both weighting and direction. H: Reads C and prints a minimum, restoring those task cues, but initializes total as subtree weight sums rather than weighted distances. Static substitution for the publicly allowed N=1, C=[1] gives 1 although all distances are zero. Primary result: test_assertion. D: Computes weighted root distances and reroots using total_C-2*subtree_C[v]; primary scoring passes. P: Starts with the correct weighted-minimum description, then repeats a code line until the 4096-token cap. No code fence closes; the unchanged extractor returns the full answer and primary scoring reports syntax.
- leetcode/3428: The hybrid restores interface compliance, but prompt-only also succeeds, so this case does not isolate a benefit from transferred reasoning. M: Computes duplicate-value XOR but names the method duplicateNumbers; primary scoring fails at the requested interface. H: Uses duplicateNumbersXOR and the frequency/XOR computation; primary scoring passes. D: Uses the requested interface and computation; primary scoring passes. P: Its first Python fence uses the requested interface and computation; primary scoring passes. A second illustrative Python fence is also present; primary extraction is unchanged.
- leetcode/3464: Restoring the interface is insufficient for correct computation; prompt-only primary failure here also depends on formatting and extraction. M: Emits standalone maximumTotalSum without Solution or the requested method. Both transitions add the current element, omitting alternating signs. H: Restores the requested interface and plus/minus transitions, but retains prior values without consuming the current element in some transitions. Static substitution into visible sample [1,-2,3,4] yields 7 rather than stated answer 10. Primary result: test_assertion. D: Uses the requested interface and transitions that consume each element; primary scoring passes. P: An unlabeled formula fence precedes a later Python fence. The unchanged first-eligible-fence extractor selects the formula, causing the recorded syntax failure. The later Python block is preserved but neither substituted nor separately scored.

Exact visible prompts, raw-output hashes and stopping/fence records are in evidence/coding_pilot_v1/post_progress01/semantic_case_audit.json. The prompt-only formula-fence failure illustrates dependence on output formatting and the fixed extractor; no post-hoc repair score replaces any primary score.

## Four seen-training-history optimization

The source-answer span audit is fixed before new outputs: atcoder/abc320_a, atcoder/abc333_b, leetcode/3114, leetcode/3091. The old fixed windows miss every function signature and return line in the three histories containing them; two longer answers have no code tokens in those windows. Actual selected96 exposure is counted separately from the saved update schedule. This documents limited direct supervision, not proof that explanation/prose is useless or that it caused task failures.

- `post_progress01_memorization_20260917_01`: 320 completed updates, 18000 scored positions, 1941 unique task/position pairs, stop=two_successive_dense_seen_kl_at_most_0.02.
  Native replay on the same seen cases: 4/4 passes. This is the receiver-quality reference, not an unseen evaluation.
  Update 0: dense seen KL 0.242536; free-running hidden-test passes 1/4.
  Update 4: dense seen KL 0.194059; free-running hidden-test passes 1/4.
  Update 16: dense seen KL 0.16262; free-running hidden-test passes 1/4.
  Update 40: dense seen KL 0.106472; free-running hidden-test passes 2/4.
  Update 80: dense seen KL 0.0489679; free-running hidden-test passes 2/4.
  Update 160: dense seen KL 0.015139; free-running hidden-test passes 3/4.
  Update 320: dense seen KL 0.0062183; free-running hidden-test passes 3/4.

| Seen training case | Native D | Initial M | Final M | Initial dense KL | Final dense KL |
|---|---|---|---|---:|---:|
|atcoder/abc320_a|pass|pass|pass|0.188333|0.000102064|
|atcoder/abc333_b|pass|runtime_error|pass|0.203166|0.00572523|
|leetcode/3114|pass|runtime_error|pass|0.230651|0.00715582|
|leetcode/3091|pass|runtime_error|test_assertion|0.347993|0.0118901|

All 320 committed updates report finite, nonzero cache gradients: True. Pre-clipping mapper gradient norm ranges from 0.0270695 to 9.48469. Per-case losses, exact positions and norms remain in training_steps.json.

This deliberate four-history fit changes both supervision coverage and concentration/repetition on a tiny seen cohort. It is not an ablation isolating coverage alone, and its loss is not directly comparable to the previous validation average. Improvement supports trainability on these histories; it cannot establish generalization or sufficient mapper capacity for arbitrary histories. Hidden scores are assigned after optimization and do not control updates or stopping.


## Diagnostic conclusions and remaining uncertainty

- The numerical check found exact agreement between the handwritten and deployed receiver paths for matched calls and the limited gradients tested. This weakens that specific implementation-mismatch hypothesis, without ruling out execution-shape sensitivity or errors outside the tested scope.
- The frozen hybrid improves development passes from 3/40 to 18/40 and removes the 14 observed requested-entrypoint omissions. Native replay remains 34/40. Prompt-only gives 17/40; H minus P is +2.5 percentage points with exploratory paired 95% interval [-7.5,+12.5]. There is no established aggregate benefit from transferred reasoning, and no equivalence claim.
- The unchanged affine mapper can fit these four seen histories much more closely: dense mean KL falls from 0.2425356209 to 0.0062182976 (97.44% reduction). Free-running passes rise from 1/4 to 3/4 while native replay passes 4/4. The remaining test assertion on leetcode/3091 shows that very low average teacher-forced loss need not yield a correct sampled program, even on a seen case.
- Optimization stops at its predeclared two-successive-checkpoint KL threshold, reached at updates 160 and 320. This is an objective-based stopping condition, not success on all four tasks or a proof of convergence to the best achievable solution. Hidden scores were assigned only afterward. There were 18,000 supervised positions and all 1,941 unique task/position pairs.
- Limited supervision remains a plausible contributor, and complete inability of this affine family to learn useful behavior on these histories is weakened. This experiment also concentrates repeated exposure on four cases, so it does not isolate coverage as the cause, establish broad capacity, or measure generalization. No unchanged-validation or reserved-confirmation test was run.
- All requested primary diagnostics are complete. The optional wrong-suffix control was omitted because the primary H/P comparison does not establish a benefit to explain. No additional optimization, architecture search, or model runs are needed to finish this bounded review.

## Memory telemetry

| Run | Maximum recorded allocated peak, GiB | Maximum recorded reserved peak, GiB | Allocator OOM counter |
|---|---:|---:|---:|
|post_progress01_hybrid_20260917_01|100.38|109.90|0|
|post_progress01_memorization_20260917_01|100.38|109.90|0|
|post_progress01_numerical_20260917_03|100.38|109.90|0|

These are maxima of recorded CUDA allocator peaks across scoped resets, not a claim of continuously sampled whole-device usage. The warning-based policy stayed in force. Raw telemetry records scopes, free/reserved memory, timing and warnings.


## Preserved failures, scope and cost

2 controller/worker failure records are indexed in `results/coding_pilot_v1/post_progress01_report/failures.json`. Capacity errors before allocation are not scored as model failures. Earlier attempts are never erased or silently rerolled.

Cumulative conservative ledger estimate at export: $293.61, 51.532 GPU-hours. This is not an invoice. Live-resource state and timestamps are retained in the cost/resource receipts; cumulative ceilings remain $1,000 / 500 GPU-hours. Diagnostic sub-budget: $120 / 20 additional GPU-hours.

Reserved 200 and second-seed 40 confirmation tasks remain untouched. No broad corpus collection, model-size change, subjective judging or architecture search. Hidden-test outcomes are scorer artifacts, not optimizer inputs. No inference or success is claimed for merely prepared code. Prompt corruption, path mismatch, sparse coverage and capacity remain hypotheses until the corresponding checks support an interpretation.

Heavy source checkpoints/features are separately inventoried and backed up outside Git; the publication snapshot backup is `/Users/qeetbastudio/Gearshift-artifacts/progress-01/`. New heavy inventories and backup receipts are included with this pass. Discarded reproducible caches and unrecoverable historical allocator omissions remain disclosed.
