# Coverage generalization results

At the evaluated common checkpoint, pure mapped-answer success was 31.7% for ROTATING, 25.4% for FIXED, and 17.5% for the starting mapper. The primary ROTATING-minus-FIXED difference was +6.3 percentage points (paired 95% interval -4.8 to +17.5). ROTATING improved over its starting checkpoint by +14.3 points (+6.3 to +23.8). These are exploratory results on previously selected validation tasks; the primary comparison does not establish that broader coverage caused a reliable improvement.

Separate controlled research. The first article and its original results remain frozen. Prepared code is not counted as executed research.

Publication tag `gearshift-progress-01` remains `64725974fa55459350d1c9d09037bab64d0c5ec6`. Research branch: `research/coverage-generalization`; current recorded commit: `aa7a20b68676e5f53e0b1ff7f85eacad436c77ba`. No push or publication.

## Population and comparison

Exact existing 104 training and 21 validation histories; completion-selected availability can underrepresent long or difficult histories. The 21 validation tasks are excluded from gradients, but previously informed KL selection. They are not untouched confirmation. Reserved 200 and second-seed confirmation inputs remain untouched.

Both arms start from the original selected update-96 mapper (`0b9700ffd9cb38c23bcfa1327181b32992b128c6b95078214328382f86b8d68f`). Identical fresh AdamW states replace unavailable original optimizer states. Models, affine architecture, pinned BF16 runtime, source-written references, sampler, extraction and scoring are unchanged. No teacher-answer tokens prefix the free-running answers.

FIXED retains offsets 0–7, 32–39, 128–135, 512–519. Missing fixed positions are filled from the next history in the original seeded bucket. ROTATING preserves ordered task contributions and early-bucket positions, rotating later 8-token windows through the full saved answer including actual EOS. The single 30-token training answer cycles all windows because it has no later fixed bucket. All paired updates contain exactly 32 actual predictions with identical normalization; no padding or duplicate task/position within an update.

## Actual execution

Target: 1024 additional updates per arm. Completed paired updates: 230. Primary common checkpoint: 128; 4,096 scored positions per arm. Stop: `engineering_incomplete_last_common_checkpoint`.

The common endpoint was frozen from timing/memory preflight before sampled program-quality scores. The primary checkpoint was not selected by loss or code correctness. Full prefixes and intervening continuation tokens were processed. Shared immutable historical cache preparation saves repeated setup; separate continuation/backward and optimizer times remain recorded. Matched supervision does not mean matched computation.

Training was interrupted by loss of provider DNS connectivity on the controller. Its automatic guard stopped the GPU. The completed-update count is the last verified backup and a lower bound on work performed before termination. Only the last durable paired mapper state is evaluated; later logged updates are not recoverable as weights. Optimizer states were not saved, so training was not resumed. Fresh matched generation and scoring ran separately under the recovery identity; the original 1,024-update schedule was not completed.

| Arm | Scored positions | Unique task/positions |
|---|---:|---:|
|FIXED|4,096|2,888|
|ROTATING|4,096|3,904|

Exposure at the evaluated checkpoint, using descriptive token/line labels across all complete code fences. These labels did not select training positions or grade answers.

| Answer region | FIXED scored / distinct | ROTATING scored / distinct |
|---|---:|---:|
|EOS|6 / 4|13 / 13|
|code body|656 / 424|1,158 / 1,146|
|code fence|72 / 59|98 / 89|
|prose|3,345 / 2,387|2,673 / 2,502|
|return statement|6 / 3|58 / 58|
|signature|11 / 11|96 / 96|

The timing table covers all 230 logged paired updates, including updates beyond the evaluated checkpoint. The raw log preserves timing for each update.

| Arm | Continuation forward/backward, seconds | Optimizer, seconds | Finite nonzero cache gradients |
|---|---:|---:|---|
|FIXED|703.3|1.4|True|
|ROTATING|750.4|1.4|True|

Shared full-history cache preparation: 4977.9 seconds. Total paired-update wall time: 6600.4 seconds. Arm times synchronize the device and include instrumentation; they are stage wall times, not hardware utilization integrals. Actual leased GPU-hours below include setup, controls, validation, generation and export.

Condition key: **M** translates the full historical cache; **H** combines native original-prompt cache with the translated reasoning suffix; **D** natively reads the complete source history; **P** answers from the original prompt using the non-thinking template. START is the original selected mapper; FIXED and ROTATING are the two common-budget training arms.

## Validation behavior

Each rate averages all three predeclared draws within each task, then the 21 tasks. The 63 draws per condition are not 63 independent task clusters. No best-of-three selection. Intervals use 10,000 identical paired task resamples across all contrasts; exploratory, unadjusted 95% percentile intervals.

| Condition | Passed draws / 63 | Mean task success | Task-cluster 95% interval | Missing requested entrypoint | EOS / cap |
|---|---:|---:|---:|---:|---:|
|START_M|11/63|17.5%|[6.3%, 30.2%]|24|57/6|
|START_H|30/63|47.6%|[28.6%, 66.7%]|4|59/4|
|FIXED_M|16/63|25.4%|[11.1%, 42.9%]|17|52/11|
|FIXED_H|30/63|47.6%|[28.6%, 66.7%]|3|59/4|
|ROTATING_M|20/63|31.7%|[17.5%, 47.6%]|16|59/4|
|ROTATING_H|35/63|55.6%|[36.5%, 74.6%]|5|58/5|
|D|60/63|95.2%|[85.7%, 100.0%]|0|63/0|
|P|28/63|44.4%|[27.0%, 61.9%]|1|63/0|

| Contrast | Difference, percentage points | Paired 95% interval, pp |
|---|---:|---:|
|ROTATING_M-FIXED_M|+6.3|[-4.8, +17.5]|
|ROTATING_H-FIXED_H|+7.9|[+0.0, +17.5]|
|FIXED_M-START_M|+7.9|[-1.6, +17.5]|
|ROTATING_M-START_M|+14.3|[+6.3, +23.8]|
|FIXED_H-START_H|+0.0|[-7.9, +6.3]|
|ROTATING_H-START_H|+7.9|[-3.2, +19.0]|
|START_H-D|-47.6|[-66.7, -28.6]|
|START_H-P|+3.2|[-4.8, +11.1]|
|FIXED_H-D|-47.6|[-66.7, -28.6]|
|FIXED_H-P|+3.2|[-4.8, +11.1]|
|ROTATING_H-D|-39.7|[-58.7, -22.2]|
|ROTATING_H-P|+11.1|[+0.0, +25.4]|

The primary comparison does not establish a reliable held-out benefit from rotating coverage. Its interval includes zero; this is not an equivalence finding.

H exceeding M alone does not establish a transferred-reasoning benefit. H versus prompt-only P and native full replay D are reported separately. These observations may inform a later decision but did not alter this round’s recipe.

| Condition | Mean answer tokens | Answer seconds | Native prefill seconds | Native prefill tokens | Repeated 4-gram fraction |
|---|---:|---:|---:|---:|---:|
|START_M|977.4|29.61|0.000|0.0|0.192|
|START_H|955.8|28.95|0.054|490.4|0.160|
|FIXED_M|1220.4|36.85|0.000|0.0|0.235|
|FIXED_H|959.3|29.35|0.054|490.4|0.159|
|ROTATING_M|906.7|27.52|0.000|0.0|0.173|
|ROTATING_H|998.4|30.33|0.054|490.4|0.155|
|D|560.6|16.84|0.642|6948.5|0.097|
|P|530.4|16.10|0.060|493.4|0.091|

H includes native original-prompt work. Timings include bridge and instrumentation overhead. Shared reconstruction and cloning are recorded separately; source-inclusive estimates reuse the saved source duration and are not a demonstrated equal-quality speedup. Parsing, runtime, assertion, timeout and other original score categories remain in the JSON/CSV records. Entrypoint compliance checks requested function names and Solution membership; it is not a full semantic interface proof. Repetition is descriptive, including legitimate repeated code.

## Validation loss

Separate fixed legacy and broad panels on all 21 validation histories. Their numerical magnitudes are not directly interchangeable. Lower teacher-forced KL need not yield a correct sampled program.

| Arm | Additional updates | Legacy KL | Broad KL |
|---|---:|---:|---:|
|FIXED|0|0.2928079|0.2344645|
|ROTATING|0|0.2928079|0.2344645|
|FIXED|128|0.2833838|0.2267853|
|ROTATING|128|0.2397437|0.1754253|

## Four seen training cases

Five predeclared draws on each of the same four training cases. No new optimization for this check. These are four task clusters, not 20 independent tasks and not generalization evidence.

| Condition | Passed draws / 20 | Mean task success |
|---|---:|---:|
|START_M|8/20|40.0%|
|SEEN_FIT_M|15/20|75.0%|
|D|20/20|100.0%|

| Seen task | Starting M | Four-case-fit M | Native D |
|---|---:|---:|---:|
|atcoder/abc320_a|100%|100%|100%|
|atcoder/abc333_b|60%|100%|100%|
|leetcode/3114|0%|100%|100%|
|leetcode/3091|0%|0%|100%|

| Run | Maximum recorded allocated peak, GiB | Reserved peak, GiB | Allocator OOM counter |
|---|---:|---:|---:|
|coverage_experiment_20260918_01|100.38|109.90|0|
|coverage_preflight_20260918_01|100.38|109.90|0|
|coverage_recovery_20260918_02|100.38|109.90|0|

Memory values are maxima of recorded CUDA allocator peaks across the logged reset scopes; they are not continuous whole-device measurements.

## Controls, limitations and resources

The retained numerical checks distinguish matched-call agreement from batch-versus-token execution sensitivity. Preflight adds late-position matched-path checks. Every evaluation history has exact native/native whole-cache splice checks; actual partial-prompt prefill rounding is recorded separately. All conditions are generated freshly under this round’s identity, so old 40-task scores are not reused. Historical caches are cloned and checked for isolation; absolute suffix positions are preserved.

Memory remains warning-based; no historical-peak/free-memory veto. Actual allocation errors, nonfinite or missing gradients, failed numerical controls, deadlines and budget ceilings still stop work. Private hidden tests are loaded only after optimization and generation and executed only in the existing sandbox. Extraction is unchanged, including the first eligible code block; function names and failed programs are never repaired for primary scoring.

Preserved failure records: 2. Cumulative conservative estimate: $335.81, 58.936 GPU-hours. This round: $41.68, 7.404 GPU-hours. The ledger is an upper estimate, not an invoice; retained storage continues to accrue. Round ceiling: +$200/32 GPU-hours; cumulative ceiling: $1000/500 GPU-hours.

Live resources and timestamp are recorded in `evidence/coding_pilot_v1/coverage_generalization/live_resources.json`. Heavy mapper backups, exact hashes and retrieval paths are in `excluded_heavy.json`; they are outside Git on the Studio disk, not an off-site disaster-recovery guarantee. Large model weights, tensor caches, private tests and environments are excluded from the review ZIP.

Reproduce compact tables and plots with `python scripts/coding_coverage_report.py --result-root results/coding_pilot_v1/coverage_recovery_20260918_02/evaluation_recovery`. No model weights or private tests are needed for this records-only command. See `REPRODUCE_COVERAGE_GENERALIZATION.md` for execution and provenance details.
