# Gearshift recovery results

This is the separately identified September 17 exploratory recovery. Original pilot, follow-up, phase2 and larger-model baseline records remain unchanged. The complete progress article is `publication/GEARSHIFT_PROGRESS_01.md` and can be reviewed independently. Nothing has been published or pushed.

Committed development coverage: **40/40**. Infrastructure-incomplete tasks are listed in `development_summary.json` and are not scored as wrong model answers. Reserved 200-task confirmation and 40-task second-seed sets remain untouched.

| Arm | Hidden-test passes | Pass rate | EOS / answer cap |
|---|---:|---:|---:|
|A|34/40|85.0%|40 / 0|
|B|27/40|67.5%|40 / 0|
|C|3/40|7.5%|38 / 2|
|D|34/40|85.0%|40 / 0|
|D_original|34/40|85.0%|40 / 0|
|C_initial|0/40|0.0%|39 / 1|

Development attempts: `recovery_dev_20260917_01`, `recovery_dev_20260917_02`. Tasks are merged only when disjoint, with identical selected mapper bytes; earlier committed outcomes are preserved.

D is the matched rerun; D_original preserves the saved comparison. A/B are the saved large-only/independent-small results. C is the validation-selected trained mapper; C_initial is the affine initializer. The initializer was evaluated alongside the selected checkpoint after training; its development outcomes did not select the training schedule or checkpoint.
Matched D reproduced the original answer tokens on 40/40 committed cases. Source token histories were copied exactly; source caches were rebuilt with the full-prefix training schedule.

| Arm | Test assertion | Runtime error | Syntax error | Code timeout | Mean excess token 4-grams |
|---|---:|---:|---:|---:|---:|
|A|4|1|0|1|7.64%|
|B|10|1|0|2|8.46%|
|C|18|16|2|1|16.50%|
|D|4|1|0|1|9.67%|
|D_original|4|1|0|1|9.67%|
|C_initial|21|18|1|0|11.28%|

Code timeouts are sandbox execution outcomes. Unavailable infrastructure is excluded from scoring and reported as missing coverage. Repetition is descriptive; legitimate code boilerplate can repeat.

| Paired contrast | Difference (pp) | Gains / losses | 95% paired bootstrap interval (pp) |
|---|---:|---:|---:|
|C-B|-60.0|1 / 25|[-77.5, -42.5]|
|C-D|-77.5|0 / 31|[-90.0, -65.0]|
|C-A|-77.5|0 / 31|[-90.0, -65.0]|

Intervals use 10,000 paired task resamples. They are exploratory, unadjusted for multiple comparisons, and cannot establish equivalence. The development set was already inspected and is not fresh confirmation.

Mean task timings in seconds for the current instrumented handoffs:

| Arm | Archived source prompt + reasoning | Mapping | Receiver prefill | Bridge | Answer after bridge | Source-inclusive estimate | Offline source-cache reconstruction |
|---|---:|---:|---:|---:|---:|---:|---:|
|C|665.077|0.085|0.000|0.034|27.759|692.955|3.443|
|C_initial|665.077|0.071|0.000|0.033|22.463|687.643|3.443|
|D|665.077|0.000|1.176|0.033|20.922|687.208|0.000|

The source-inclusive estimate substitutes archived source execution for offline cache reconstruction; it is not the wall time of a live online handoff. Reconstruction is measured separately. Unequal answer quality, lengths and termination prevent treating a time reduction alone as an equal-quality speedup.

The independent smaller model B additionally spent a mean 362.019 seconds on its own archived prompt/reasoning, for 385.786 seconds including its answer. Its reasoning is not free or omitted from the total.

Selected checkpoint: step 96, validation KL 0.292808, SHA-256 `0b9700ffd9cb38c23bcfa1327181b32992b128c6b95078214328382f86b8d68f`. Selection used all 21 fixed validation histories and no development outcomes. The corpus contains 104/128 training and 21/32 validation histories; completion selection may underrepresent long histories. Convergence is untested.

Per-task raw paths/hashes, parse/execution categories, repeated-token measurements and every timing component are in `results/coding_pilot_v1/recovery_review_20260917/development_tasks.csv` and `development_summary.json`.

Source time is archived original prompt prefill plus reasoning. B has no shared source; its own_reasoning_seconds is included in both total_model_execution_estimate_seconds and the backward-compatible source_inclusive_estimate_seconds. C reconstructs source KV offline in 512-token chunks. Reconstruction is separately reported, not silently called live-online speedup. Answer timers include checkpoints/telemetry; A/B and originalD are historical. Current C and matched D share instrumentation. Task component totals exclude model setup, numerical controls, cache-object assembly, sandbox scoring and archive overhead; allocated-resource charges include those stages.

Cumulative conservative usage at this report: $280.04, 49.147 GPU-hours, against the unchanged $1,000/500-hour limits. Active GPU count: 0; active resources: [['volume', 'jj2zyi9yrc']]. This is the task ledger estimate, not an invoice. The preserved original 250 GB failure volume remains approximately $0.04/hour.

Revised code/configs, test logs, telemetry, memory-failure/recovery evidence, exact corpus and checkpoint identities are included in the rolling review ZIP. Source reasoning trajectories are included; model weights, feature/cache/checkpoint tensors, environments and credentials are excluded with hash/location/regeneration inventories. See `REPRODUCE_RECOVERY.md`.

## Engineering recovery and actual training

![Single-trajectory memory diagnostic](results/coding_pilot_v1/recovery_review_20260917/recovery_probe_20260917_04_memory.png)

- `recovery_probe_20260917_04`: 419 memory samples; warning counts {'driver_free_below_10GiB': 32}; completed=True; allocation readings and reset scopes retained in raw telemetry.
  Saved prefix matched exactly through 21057 tokens; new source reasoning reached 24576 tokens; paired extraction completed=True. This is engineering evidence, not a code-quality result.
  During source generation, live allocation reached 82.40 GiB and reservation 138.38 GiB; minimum sampled driver free memory was 0.785 GiB. Allocator counters reached 1 allocation retries and 0 out-of-memory failures. Counters and periodic observations are not a complete per-allocation trace.

The amended lifecycle also clears released allocator blocks and resets peaks once after native controls. Exact token replay does not mean identical allocator state. The old incident still lacks its numeric readings; successful current recovery cannot retrospectively prove its cause. The recovery uses warnings for the old memory thresholds, strict numerical controls, durable token/RNG checkpoints, and bounded network setup after one observed TLS handshake stall.

Committed functional updates used 104 distinct histories from the fixed 104-history training corpus. The affine initializer used all 104. Partial interrupted updates, if any, are excluded from committed prediction counts.
Actual functional run: 115 full updates, 3680 committed gradient-bearing predictions, 5400.1 seconds, stop=time_or_controlled_stop; objective natural_handoff_boundary. Fresh affine initialization used 6,656 sampled positions from 104 training histories. All completed histories were natural source endings; training reasoning max 20,079 tokens and validation max 15,690, while the four interrupted originals already exceeded 20,000. This survivor bias is material and explicitly preserved.
The coding regression suite passed 237 tests before initial dispatch and 254 after the capacity-completion repair. Relevant exact test logs are in `evidence/coding_pilot_v1/recovery_20260917/`. Tests are not counted as experimental results.
