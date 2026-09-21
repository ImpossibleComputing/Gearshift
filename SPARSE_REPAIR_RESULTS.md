# Sparse-state / selective-repair diagnostic

**COMPLETE_EXPLORATORY_SCREEN**

Complete generated answers: **504/504**. Binary executed-code outcomes: **504/504**. Verified passing calibration tasks: **4/4**. Unrun draws are **not** quality failures.

This is the bounded exploratory screen on 12 previously inspected development tasks, three fixed draws per condition. It is not a new confirmation set. The original results and publication tag remain separate.

## Actual coverage and numerical controls

Draw states: `{"scored": 504}`. Generation seal present: True; complete scoring manifest present: True.

| Calibration task / receipt | Completed and hash-bound | Reported passed | Actual case wall seconds |
|---|---:|---:|---:|
| leetcode/3484 (`results/sparse_repair_01/calibration_runs/calibration_usco02/leetcode__3484/calibration.json`) | True | True | 160.6377108450979 |
| atcoder/abc362_d (`results/sparse_repair_01/calibration_runs/calibration_usco03/atcoder__abc362_d/calibration.json`) | True | True | 241.6879930649884 |
| atcoder/abc363_e (`results/sparse_repair_01/calibration_runs/calibration_usco04/atcoder__abc363_e/calibration.json`) | True | True | 395.93724322412163 |
| leetcode/3379 (`results/sparse_repair_01/calibration_runs/short_01/leetcode__3379/calibration.json`) | True | True | 161.37557796714827 |

Handoff status: STUDIO_READY. Calibration/GPU launch receipt status: GPU_WORK_LAUNCHED. A calibration launch is not evidence that the 504-answer screen launched.

- `screen_launch.json`: receipt epoch=1789964547.3344572; status=SCREEN_GENERATION_LAUNCHED; actual answer-generation progress observed for 2/5 named workers at the recorded observations. These are historical launch/progress receipts, not a current liveness check or a completion seal.
- `screen_all_workers_generating.json`: receipt epoch=1789965392.46828; status=no status string; actual answer-generation progress observed for 5/5 named workers at the recorded observations. These are historical launch/progress receipts, not a current liveness check or a completion seal.

Launching workers, observing generated tokens, or passing calibration does not establish task quality. No complete-screen quality conclusion is available until all 504 frozen answers have sealed generation, valid binary scores, and the required evidence-integrity checks.

Calibration completion discovery uses calibration_runs/** raw allocation records, not canonical calibration/ copies used by the screen gate. Complete raw run receipts must be collected before reporting.

Same-execution-shape controls require bit-exact logits; altered-shape rounding is separately reported and does not excuse same-path errors. No tolerance is widened here.

### Infrastructure interruption and continuation

Final generation/scoring coverage does not erase an earlier deadline interruption. The following are separately hash-bound historical recovery records, not new outcomes or permission to retry an answer based on quality. Staging does not imply launch; process launch does not imply successful replay or completed generation.

The saved decision records **476/504** completed answers before continuation, with 14 completed draws in the pending task(s) ['leetcode/3455']. Interrupted draw: R_recent / seed index 2, 109 committed tokens and 108 forwarded tokens.

Recorded reason: Original immutable allocation ended at stop margin after capacity-driven group merge exceeded six-hour estimate. This is infrastructure continuation, not quality/outcome retry.

- `recovery/decision.json` — recorded scalar fields (full receipt retained in `summary.json`):
  ```json
  {"additional_upper_compute_reservation_usd": 44.4, "complete_draws_in_pending_task": 14, "completed_draws_unchanged_required": true, "epoch": 1789985723.479498, "existing_diagnostic_compute_envelope_usd": 300, "expected_answers": 504, "global_complete_answers": 476, "helper_sha256": "fe12092185c914929d5eccd4581e0f9a4b3d041522e962073b98bb1342e5f516", "new_allocation_name": "screen_camtl_resume01", "new_authorization_requested": false, "new_remote_root": "/workspace/GearshiftSparseRepairResume01", "no_private_scores_seen": true, "old_absence_receipt": "results/sparse_repair_01/resources/screen_camtl05/closed.json", "old_absence_sha256": "e3e1e628031a3734e01e7814e80d3413053ea9cb02bf18da03c1ba05499cc104", "old_pod_id": "no83hxamjucm7n", "old_remote_root_preserved": "/workspace/GearshiftSparseRepair", "reason": "Original immutable allocation ended at stop margin after capacity-driven group merge exceeded six-hour estimate. This is infrastructure continuation, not quality/outcome retry.", "requested_new_lease_hours": 8}
  ```
- `recovery/resume_stage.json` — recorded scalar fields (full receipt retained in `summary.json`):
  ```json
  {"code_commit": "997f01b8c2b93955b45f917808fd9a9f6c1e3a79", "created_epoch": 1789985925.6193414, "declaration_sha256": "2fd04f4e3a20f8c38f4193582ef704ffff3d3cb28aba8f212711fa8329e13c9c", "destination_root": "/workspace/GearshiftSparseRepairResume01", "hardlinks_created": false, "implementation_sha256": "9659381ff6c381eccdfa9c7854e1eda06f26504a55c95984e01ce34f32e7efb9", "launch_authorized_by_this_receipt": false, "model_execution_performed": false, "model_weights_not_copied_or_executed": "Existing absolute /workspace/hf paths; existing screen runner must rehash all22 shards.", "new_lease_path": "results/sparse_repair_01/resources/screen_camtl_resume01/lease.json", "new_lease_sha256": "2edec6f412f7ee2604473b0c81ccc04f904345a33bd4d2f13278c48a0a6f959e", "new_pod_id": "3n63on9103q0yl", "old_control_or_locks_copied": false, "old_lease_sha256": "64e72e7a4f5ae462a5f2dc3493ebf40df05536db0ba7fabe7969cce6a1f9c370", "old_pod_id": "no83hxamjucm7n", "private_values_accessed": false, "purpose": "frozen_public_infrastructure_resume_stage", "runtime_identity_sha256": "1cbf87a33bbf56f8cafd0ffe9eddec6e07e5fb1293fd89ad1392389ab6b5092f", "schema": 1, "source_files_written": false, "source_root": "/workspace/GearshiftSparseRepair", "status": "STAGED_NO_LAUNCH"}
  ```
- `recovery/resume_launch.json` — recorded scalar fields (full receipt retained in `summary.json`):
  ```json
  {"epoch": 1789985943.8494463, "infrastructure_continuation_only": true, "plan": "results/sparse_repair_01/resources/screen_camtl_resume01/screen_job_plan.json", "plan_sha256": "9c5072be93048af4d2aaec26c50a39423ebbe6d08e99aec34096d097a004d163", "pod_id": "3n63on9103q0yl", "remote_root": "/workspace/GearshiftSparseRepairResume01", "stage_sha256": "30dcb2cdcf9a232f6740a9dff04c21ebce01b1e2120b1ed4c92af6d6a9d75566"}
  ```
- `recovery/recovery_validation.json` — recorded scalar fields (full receipt retained in `summary.json`):
  ```json
  {"all_pending_tasks_complete": true, "complete_sampler_outer_incomplete_corner": false, "epoch": 1789989408.562996, "fingerprint_evidence_basis": "Frozen answer_durable requires verify_reconstruction before new sampling; completed attempt begins from109 committed tokens, no reconstruction_failure receipt, unchanged source identity. No separate success fingerprint receipt is emitted.", "no_candidate_quality_used": true, "old_completed_draws": 14, "old_completed_files_byte_identical": 154, "original_committed_token_prefix_preserved": 109, "original_source_checkout_not_modified": true, "reconstruction_fingerprint_gate_passed": true, "reconstruction_seconds_this_call": 19.595933720935136, "resumed_draw_total_tokens": 526, "status": "RECOVERY_VALIDATED"}
  ```
- Preserved checkpoint `leetcode/3455` / `R_recent` / seed 2: state=interrupted; committed=109; forwarded=108; known active seconds at that checkpoint=20.214087300002575. Its interrupted progress union is not a complete-answer working set.

Collected per-task cache-setup attempt counts greater than one: `{"leetcode/3455": 2}`. Individual setup timings/backing sizes are retained in `summary.json`; an empty map means no duplicate setup receipt is available in this snapshot, not proof that recovery was free.

| Resumed draw | Sampler resume count | Known sampler reconstruction s | Complete-sampler trace rebuild s | Last draw-attempt wall s |
|---|---:|---:|---:|---:|
| leetcode/3455 / R_recent / 2 | 1 | 19.595933720935136 | 0.0 | 95.80766880512238 |

**Recovery accounting limits:** duplicated cache/model setup, rehashing and replay are real continuation work, not a free retry. The last task/draw-attempt wall does not include all earlier interrupted attempts; conversely, its measured replay is already inside that attempt wall and must not be added again. Known sampler completion overhead and saved original checkpoint timing remain separate. Semantic selection unions can be reconstructed along committed token history, but discarded/uncommitted crash-tail selections and timing may remain incomplete; later 504/504 completion does not retroactively measure them. Completed draw byte identity, committed-prefix preservation and reconstruction-logit fingerprint verification require the explicit validation receipt, not inference from a successful clone, stage, launch or final count.

- `leetcode/3484` / `N100_vs_dense_native`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `leetcode/3484` / `R0_vs_existing_H`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `leetcode/3484` / `R100_vs_matched_native_native_splice`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `leetcode/3484` / `unmodified_hook_vs_deployed_dense`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `atcoder/abc362_d` / `N100_vs_dense_native`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `atcoder/abc362_d` / `R0_vs_existing_H`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `atcoder/abc362_d` / `R100_vs_matched_native_native_splice`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `atcoder/abc362_d` / `unmodified_hook_vs_deployed_dense`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `atcoder/abc363_e` / `N100_vs_dense_native`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `atcoder/abc363_e` / `R0_vs_existing_H`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `atcoder/abc363_e` / `R100_vs_matched_native_native_splice`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `atcoder/abc363_e` / `unmodified_hook_vs_deployed_dense`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `leetcode/3379` / `N100_vs_dense_native`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `leetcode/3379` / `R0_vs_existing_H`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `leetcode/3379` / `R100_vs_matched_native_native_splice`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.
- `leetcode/3379` / `unmodified_hook_vs_deployed_dense`: passed=True; positions=3; max absolute logit difference=0.0; max KL=0.0.

### Independent prompt-prefill execution-shape rounding

These comparisons retain the predeclared altered-shape ceiling: max absolute logits ≤0.125, KL≤0.0001, and matching top-1. A failed comparison is reported as failed, never accepted by widening the tolerance. Full repair is checked against its **matched native/native-splice reference**, not claimed identical to full native replay D.

| Calibration task | Altered-shape check passed | Max absolute logit difference | Max KL | All sampled top-1 equal |
|---|---:|---:|---:|---:|
| leetcode/3484 | False | 0.75 | 0.0053981998935341835 | True |
| atcoder/abc362_d | True | 0.0 | 0.0 | True |
| atcoder/abc363_e | True | 0.0 | 0.0 | True |
| leetcode/3379 | True | 0.0 | 0.0 | True |

**Observed independent-prompt rounding discrepancy:** at least one actual context exceeds that declared cross-shape ceiling. The matched splice controls remain the appropriate full-repair comparison; those same-path controls still require bit-exact logits. Do not describe full repair as numerically identical to D on the affected cases.

## Matched-query local probes

Local native-reference queries diagnose one attention operation. They do not demonstrate complete-answer code quality. Distribution probes use identical prefixes but may have different later-layer queries.

| Arm / fraction | Layer-position observations | Mean relative attention-output L2 | Mean retained reasoning mass |
|---|---:|---:|---:|
| M_0.05 | 432 | 0.27130093290125606 | 0.7629460937554837 |
| M_0.1 | 432 | 0.26645855730640944 | 0.8424156816190178 |
| M_0.25 | 432 | 0.2639418952770669 | 0.9343197606536224 |
| N_0.05 | 432 | 0.08206106332147976 | 0.7629460937554837 |
| N_0.1 | 432 | 0.056526504728632666 | 0.8424156816190178 |
| N_0.25 | 432 | 0.026442159548794194 | 0.9343197606536224 |
| R_0.05 | 432 | 0.09352346307908495 | 0.7629460937554837 |
| R_0.1 | 432 | 0.07663799893580964 | 0.8424156816190178 |
| R_0.25 | 432 | 0.0517819514411881 | 0.9343197606536224 |

## Full-screen executed-code outcomes

| Condition | Generated / 36 | Binary scored / 36 | Passes | Full-population pass rate | Mean answer tokens | EOS / capped |
|---|---:|---:|---:|---:|---:|---:|
| D | 36/36 | 36/36 | 33 | 91.67% | 559.3611111111111 | 36 / 0 |
| H | 36/36 | 36/36 | 21 | 58.33% | 667.1666666666666 | 36 / 0 |
| P | 36/36 | 36/36 | 15 | 41.67% | 632.2777777777778 | 36 / 0 |
| N_5 | 36/36 | 36/36 | 33 | 91.67% | 547.6111111111111 | 36 / 0 |
| M_5 | 36/36 | 36/36 | 21 | 58.33% | 780.8888888888889 | 36 / 0 |
| R_5 | 36/36 | 36/36 | 33 | 91.67% | 580.1388888888889 | 36 / 0 |
| N_10 | 36/36 | 36/36 | 33 | 91.67% | 541.8611111111111 | 36 / 0 |
| M_10 | 36/36 | 36/36 | 21 | 58.33% | 669.8611111111111 | 36 / 0 |
| R_10 | 36/36 | 36/36 | 33 | 91.67% | 559.8055555555555 | 36 / 0 |
| N_25 | 36/36 | 36/36 | 33 | 91.67% | 550.4166666666666 | 36 / 0 |
| M_25 | 36/36 | 36/36 | 23 | 63.89% | 674.1666666666666 | 35 / 1 |
| R_25 | 36/36 | 36/36 | 33 | 91.67% | 563.0555555555555 | 36 / 0 |
| R_random | 36/36 | 36/36 | 33 | 91.67% | 570.0833333333334 | 36 / 0 |
| R_recent | 36/36 | 36/36 | 29 | 80.56% | 631.3611111111111 | 36 / 0 |

Task-paired contrasts; unadjusted exploratory 95% task-cluster bootstrap intervals (10,000 PCG64 resamples, seed 20260921):

| Contrast | Difference (pp) | 95% interval (pp) | Tasks improved / worse / tied |
|---|---:|---:|---:|
| R_10-H | +33.33 | [+11.11, +58.33] | 5 / 0 / 7 |
| R_10-R_random | +0.00 | [+0.00, +0.00] | 0 / 0 / 12 |
| R_10-R_recent | +11.11 | [+0.00, +27.78] | 2 / 0 / 10 |
| N_5-M_5 | +33.33 | [+11.11, +58.33] | 5 / 0 / 7 |
| N_10-M_10 | +33.33 | [+8.33, +58.33] | 4 / 0 / 8 |
| N_25-M_25 | +27.78 | [+5.56, +52.78] | 4 / 0 / 8 |
| R_5-H | +33.33 | [+11.11, +58.33] | 5 / 0 / 7 |
| R_25-H | +33.33 | [+11.11, +58.33] | 5 / 0 / 7 |

All 12 task clusters and paired draw changes are retained in `per_task.csv`; failure categories, output lengths, stopping and working-set summaries are in `summary.json`/`per_draw.csv`. Interface failures are not separately identified by the frozen scorer and may overlap runtime/assertion categories; their separate count is unavailable. Passing the frozen tests is the measured endpoint, not proof for every valid input.

## Working set, timing, memory and cost

The implemented oracle keeps native backing caches, scans native keys and materializes dense temporary K/V/masks. Record logical pairs separately from bytes actually read: no HBM traffic or sparse latency saving was measured by the bookkeeping.

**Denominators:** the nominal budget applies independently to each layer/KV-head's historical reasoning positions (ceil rounding). Cumulative fractions count unique (layer, KV-head, position) pairs across the entire answer, not a global token union. The complete original prompt and the current/own-answer states stay present outside this budget. D/H have no selector accounting: their zero raw selected-pair counters do not mean zero dense reads; P has no reasoning intervention.

**Pages and residency:** both page columns below are hypothetical 16-position layouts, not allocated/freed pages or transferred bytes. One layout pages each KV head independently; the other charges all KV heads when any head touches an absolute-position page. Padding/boundary pages can exceed raw reasoning-row bytes. `working_sets.csv` retains per-head cumulative fractions, per-layer logical buffer sizes, shared backing sizes and CUDA allocator snapshots. Original native/mapped/hybrid caches, the independent prompt and P cache remain resident even while P runs; the source-cache construction footprint marked `source_peak_then_freed` is not an additional simultaneously live answer cache.

**Measurement gaps:** native-key-scan timing includes current-query scoring plus probability normalization; gather/replacement/dense-buffer construction/masks share one stage and are not separately measured. KV-group mask tensor bytes omit the query-head-expanded mask and combined-mask scratch; other scoring/gather/softmax scratch is not fully itemized. Per-layer maxima are not concurrent temporary-memory peaks. Unprofiled whole-answer wall time still includes CPU union accounting and durable I/O. The repeated 16-token calibration profile is not complete-answer end-to-end cost evidence.

Working-set summaries below use available completed draws only; coverage is explicit and no quality inference is made from a partial population.

| Arm | Working-set draws | Instantaneous pair fraction mean | Cumulative pair fraction mean [min, max] | Estimated independent-head page bytes mean | Estimated shared-head page bytes mean |
|---|---:|---:|---:|---:|---:|
| D | 0 | Not applicable | Not applicable | Not applicable | Not applicable |
| H | 0 | Not applicable | Not applicable | Not applicable | Not applicable |
| P | 0 | Not applicable | Not applicable | Not applicable | Not applicable |
| N_5 | 36 | 0.0501 | 0.6409 [0.4474, 0.7451] | 1290186752 | 1428313429.3333333 |
| M_5 | 36 | 0.0501 | 0.6771 [0.5113, 0.8985] | 1297580714.6666667 | 1428074951.1111112 |
| R_5 | 36 | 0.0501 | 0.6511 [0.5096, 0.8344] | 1290534456.8888888 | 1428295224.8888888 |
| N_10 | 36 | 0.1001 | 0.7673 [0.6230, 0.9049] | 1346494464 | 1428733952 |
| M_10 | 36 | 0.1001 | 0.7900 [0.6413, 0.9242] | 1349296355.5555556 | 1428706645.3333333 |
| R_10 | 36 | 0.1001 | 0.7767 [0.6500, 0.8735] | 1346555676.4444444 | 1428726670.2222223 |
| N_25 | 36 | 0.2500 | 0.9128 [0.8334, 0.9622] | 1396781283.5555556 | 1428750336 |
| M_25 | 36 | 0.2500 | 0.9190 [0.8244, 0.9692] | 1397640760.8888888 | 1428750336 |
| R_25 | 36 | 0.2500 | 0.9151 [0.8294, 0.9599] | 1396581489.7777777 | 1428750336 |
| R_random | 36 | 0.1001 | 1.0000 [1.0000, 1.0000] | 1428750336 | 1428750336 |
| R_recent | 36 | 0.1001 | 0.1001 [0.1000, 0.1005] | 144310272 | 144310272 |

Draw-attempt wall includes cache clone, sampler call and any trace reconstruction, but is captured before the final working-set/answer/complete JSON writes. These are diagnostic mixed-residency costs, not isolated deployment latency or matched-length speedups. Completion-timing overhead (including final sampler checkpoint/materialization) is preserved separately in summary/CSV; it overlaps active/draw time and must not be added again or subtracted to claim model-only latency.

- D: mean implemented draw-attempt wall time=27.035726796363534s; cumulative reasoning-pair fraction mean=None; GPU peak allocated maximum=97245967872bytes.
- H: mean implemented draw-attempt wall time=32.74014929000987s; cumulative reasoning-pair fraction mean=None; GPU peak allocated maximum=97245967872bytes.
- P: mean implemented draw-attempt wall time=29.89620860399575s; cumulative reasoning-pair fraction mean=None; GPU peak allocated maximum=93714124288bytes.
- N_5: mean implemented draw-attempt wall time=78.60575565356218s; cumulative reasoning-pair fraction mean=0.640890556480585; GPU peak allocated maximum=97771795968bytes.
- M_5: mean implemented draw-attempt wall time=108.87650473323366s; cumulative reasoning-pair fraction mean=0.6771070892584872; GPU peak allocated maximum=97770980352bytes.
- R_5: mean implemented draw-attempt wall time=83.57703162406364s; cumulative reasoning-pair fraction mean=0.6510615425272872; GPU peak allocated maximum=97773516800bytes.
- N_10: mean implemented draw-attempt wall time=75.98341815165865s; cumulative reasoning-pair fraction mean=0.7672825057051131; GPU peak allocated maximum=97771447808bytes.
- M_10: mean implemented draw-attempt wall time=98.60444266185449s; cumulative reasoning-pair fraction mean=0.7900425513612404; GPU peak allocated maximum=97771120128bytes.
- R_10: mean implemented draw-attempt wall time=91.63041255553253s; cumulative reasoning-pair fraction mean=0.7767424894336816; GPU peak allocated maximum=97771304448bytes.
- N_25: mean implemented draw-attempt wall time=87.42059206306132s; cumulative reasoning-pair fraction mean=0.9128413456552136; GPU peak allocated maximum=97771939328bytes.
- M_25: mean implemented draw-attempt wall time=101.5694953829451s; cumulative reasoning-pair fraction mean=0.919008947040779; GPU peak allocated maximum=97771550208bytes.
- R_25: mean implemented draw-attempt wall time=118.98562935415733s; cumulative reasoning-pair fraction mean=0.9151253628125509; GPU peak allocated maximum=97867240960bytes.
- R_random: mean implemented draw-attempt wall time=91.58211678764525s; cumulative reasoning-pair fraction mean=1.0; GPU peak allocated maximum=97528360960bytes.
- R_recent: mean implemented draw-attempt wall time=83.54034946382227s; cumulative reasoning-pair fraction mean=0.10009769088611363; GPU peak allocated maximum=97528360960bytes.

Completed task-attempt timing (shared setup/check work is included in task wall, counted once rather than 42 times):

| Task | Completed draws | Recorded task-attempt wall s | Sum of recorded shared setup/check stages s |
|---|---:|---:|---:|
| leetcode/3384 | 42 | 1761.6829694416374 | 3.9022680334746838 |
| leetcode/3412 | 42 | 2024.041161855217 | 7.63183175586164 |
| leetcode/3428 | 42 | 2015.8072620178573 | 6.586507925763726 |
| leetcode/3461 | 42 | 1768.7263258006424 | 4.260489823296666 |
| atcoder/abc353_e | 42 | 5636.6062771342695 | 36.705648683942854 |
| leetcode/3464 | 42 | 4506.1497121467255 | 53.94956390792504 |
| leetcode/3413 | 42 | 4167.1326886699535 | 86.06169918878004 |
| leetcode/3482 | 42 | 4765.300628706813 | 20.58157525653951 |
| atcoder/abc364_d | 42 | 2894.0240854190197 | 15.063840402057394 |
| leetcode/3455 | 42 | 2548.291215284262 | 104.57739768363535 |
| leetcode/3454 | 42 | 3380.577416917309 | 47.171221574768424 |
| atcoder/abc360_c | 42 | 3283.818823487032 | 30.247001313604414 |

Task-attempt wall includes its final backing fingerprint check and completed draw writes, but not worker model loading/weight verification, earlier attempts, provider billing time or the task receipt's own persistence. Source reasoning was reused, not regenerated; its saved historical time is not new screen compute. Whole-fleet billing and continuing storage belong in the separate ledger.

### Isolated CPU scoring time (separate accounting scopes)

Manifest-bound score records with public test accounting: **504/504**. Recorded final test receipts: 12294; receipts with more than one infrastructure attempt: 0.

| Count or timing | Recorded sum | Coverage |
|---|---:|---|
| executed_tests | 12294 | 504 score records with count; 0 without count |
| total_tests | 15483 | 495 score records with count; 9 without count |
| recorded_final_test_cpu_seconds | 1578.89103 | 12294 test receipts with duration; 0 without duration |
| recorded_final_test_wall_seconds | 1862.6200725257513 | 12294 test receipts with duration; 0 without duration |

These sums use only the top-level final per-test receipt, not the nested attempt histories. Early stopping means executed tests may be fewer than total tests; syntax/interrupted outcomes can lack a total or timing and are not zero-filled. Test wall durations overlap across parallel workers and are not elapsed scoring wall time. CPU seconds are a different measure. Do not add either sum to wrapper elapsed time or to the cost ledger, or treat these sums as complete retry/process/pod costs.

Public CPU wrapper coverage: 1 job(s), 4 launch receipt(s), 4 exit receipt(s).

| Pod / CPU job | Stage | Observed elapsed wall s | Exit code | Receipt state |
|---|---|---:|---:|---|
| cdyft0ba33zlyz / private_score_final04 | preflight | 8.033095598220825 | 0 | EXIT_RECEIPT_PRESENT |
| cdyft0ba33zlyz / private_score_final04 | prepare | 50.18057584762573 | 0 | EXIT_RECEIPT_PRESENT |
| cdyft0ba33zlyz / private_score_final04 | run | 408.1842839717865 | 0 | EXIT_RECEIPT_PRESENT |
| cdyft0ba33zlyz / private_score_final04 | finalize | 66.25525403022766 | 0 | EXIT_RECEIPT_PRESENT |

Wrapper intervals are launch-to-exit timestamps bound to the public job plan, immutable CPU lease and pod identity. A launch without an exit has no inferred duration, completion, or current-liveness claim. Preflight/prepare/run/finalize are separate stages; stage elapsed time includes work and waiting inside that stage, overlaps contained candidate execution, and excludes other direct preflights, transfer/setup, inter-stage gaps and release/billing time. Neither wrapper receipt presence nor timing changes any computed quality outcome.

Cost ledger: `results/sparse_repair_01/cost_ledger.json`. The supplied ledger is retained verbatim in summary.json; quote×elapsed estimates and continuing storage allowances are not invoices. The 1360 USD reservation baseline must not be presented as incurred cost.

Records-only economic reference: perfect/free faithful prefill replacement saves 1.467935s (0.201929%) against recorded D, leaving 725.487221s per answer. See ECONOMIC_CEILING.md. This is hypothetical, not a result of the oracle implementation.

## Decision: five bounded questions

1. **Does this selector permit useful sparse native access?** Observed budget curve: N_5=91.7%, M_5=58.3%; N_10=91.7%, M_10=58.3%; N_25=91.7%, M_25=63.9%; D=91.7%. This small exploratory screen cannot establish equivalence or general preservation.
2. **Does sparse mapped state preserve the same benefit?** Use the three paired N_p−M_p contrasts above; free-running histories diverge, so this is not an identical-query causal contrast.
3. **Does small targeted repair rescue H versus cheap controls?** R_10−H=+33.33pp; versus random=+0.00pp; versus recent=+11.11pp. Read task-cluster intervals and all tasks, not just point estimates.
4. **Is the required working set small over a complete answer?** R_10 cumulative reasoning fraction mean=0.7767424894336816, min=0.6499780725549769, max=0.873517444635357; nominal 10% per step does not imply 10% total transfer/capacity or allocated pages.
5. **What is the next investment?** A cheaper targeted-selector/repair experiment earns consideration only if the paired repair gain survives the cheap controls and full-answer cumulative cost remains favorable. Otherwise these data favor investigating fidelity/dependencies or a separately bounded compact-text comparison, not scaling this oracle into a serving engine. No automatic follow-on is authorized.

## Analyst interpretation — not additional measurement

This separately authored review is bound to the exact numeric input snapshot and declaration. It does not replace the computed findings, change scores or intervals, establish generalization, or authorize another experiment.

Analyst: Codex Studio primary analyst; independent score-binding, timing and economics reviews retained. Reviewed at: 2026-09-21T12:00:16.968747+00:00. Review file SHA-256: `0179913c2c921a63d481c3af51881be79e8e7e7a8a778162321bd6a510ba902a`.

1. **sparse_native_access — supported**: Yes, locally: N\_5, N\_10 and N\_25 each passed 33/36 draws (91.7%), the same observed pass count as D, while retaining only the declared per-layer/KV-head reasoning fraction at each step. Native 5% is therefore a useful quality intervention on this screen, not a measured sparse speedup.
   - Evidence (computed-summary JSON pointers): /statistics/conditions/D; /statistics/conditions/N\_5; /statistics/conditions/N\_10; /statistics/conditions/N\_25; /coverage
   - Limitations: Only 12 previously inspected development task clusters, with 3 draws each; matching counts and observed paired outcomes do not establish equivalence or future quality preservation.; Native state was built in full, selected using current-query native-key scans, and consumed by masked dense attention. Native entries encode earlier-token dependencies; these results do not show cheap sparse construction.
2. **sparse_mapped_state — not_supported**: No: M\_5 and M\_10 passed 21/36 (58.3%), and M\_25 passed 23/36 (63.9%), versus 33/36 for their native counterparts. Paired native-minus-mapped gains were 33.3, 33.3 and 27.8 percentage points. Matched-query local probes independently show substantially larger sparse-mapped attention-output error. State fidelity, rather than sparsity alone, is the leading local research issue.
   - Evidence (computed-summary JSON pointers): /statistics/conditions/M\_5; /statistics/conditions/M\_10; /statistics/conditions/M\_25; /statistics/contrasts/N\_5-M\_5; /statistics/contrasts/N\_10-M\_10; /statistics/contrasts/N\_25-M\_25; /local\_probe\_summary
   - Limitations: Free-running N/M queries and masks can diverge: they share a selector rule, not necessarily identical selected indices. The full-answer contrasts do not isolate one error source as cleanly as the matched-query probes.; This does not establish that a more accurate mapper would be economical or that all translators fail.
3. **targeted_repair — mixed**: Accurate small repairs rescued the observed mapped-background deficit, but special attention targeting was not validated. R\_5, R\_10 and R\_25 each passed 33/36 versus H 21/36. The primary R\_10−H gain was +33.3 pp (exploratory task-cluster 95% interval+11.1 to+58.3 pp), with 5 tasks improved and none worse. R\_random also passed 33/36 and matched R\_10 on every paired outcome. R\_recent passed 29/36; R\_10−recent was +11.1 pp, with interval 0 to+27.8 pp.
   - Evidence (computed-summary JSON pointers): /statistics/conditions/H; /statistics/conditions/R\_5; /statistics/conditions/R\_10; /statistics/conditions/R\_25; /statistics/conditions/R\_random; /statistics/conditions/R\_recent; /statistics/contrasts/R\_10-H; /statistics/contrasts/R\_10-R\_random; /statistics/contrasts/R\_10-R\_recent; /calibration\_cases
   - Limitations: Random repair was tested at 10%, not 5%; the 5% result must not be called better than matched-budget random 5% repair, which was not measured.; The degenerate \[0, 0\] bootstrap interval for R\_10−random reflects identical outcomes in these 12 sampled clusters, not proof of population equivalence. All intervals are exploratory and unadjusted.; Full repair is bit-exact against matched native/native splice, not guaranteed identical to independently prefilling D: one calibration context had a real 0.75 maximum-logit cross-shape discrepancy.
4. **complete_answer_working_set — not_supported**: No small complete-answer working set was demonstrated for this dynamic oracle. R\_10 selected about 10.01% per step but touched an equal-draw average of 77.67% of native reasoning (layer,KV-head,position) pairs over a complete answer (range 65.00–87.35%). R\_5 still touched 65.11% on average. Random 10% reached 100%; recency stayed near 10.01% but had lower observed pass count. Per-step sparsity is real bookkeeping; it is not an equivalent reduction in total state construction, transfer, residency or page traffic.
   - Evidence (computed-summary JSON pointers): /condition\_diagnostics/R\_10/instantaneous\_reasoning\_fraction; /condition\_diagnostics/R\_10/cumulative\_reasoning\_fraction; /condition\_diagnostics/R\_5/cumulative\_reasoning\_fraction; /condition\_diagnostics/R\_random/cumulative\_reasoning\_fraction; /condition\_diagnostics/R\_recent/cumulative\_reasoning\_fraction; /condition\_diagnostics/R\_10/cumulative\_per\_head\_page\_bytes\_estimated; /condition\_diagnostics/R\_10/cumulative\_all\_heads\_shared\_page\_bytes\_estimated
   - Limitations: The cumulative union is what this selector actually touched, not a proven minimum required set. Discarded crash-tail work in the single resumed draw may be incompletely measured.; Page footprints assume hypothetical 16-position layouts, not measured allocation or HBM traffic; full native/mapped backing, original prompt and each run's own answer history remain additional costs.
5. **next_investment — no_further_work**: Do not scale this full-native-oracle repair/faithful-prefill-replacement regime into a serving system on these results. The quality rescue is real within the sampled screen, but attention targeting did not beat random, the complete-answer union is large, and implemented R\_10 draw time averaged 91.63 s versus H 32.74 s and D 27.04 s. Independently, a perfect/free faithful converter saves only 1.468 s, or 0.202%, of recorded source-inclusive D time on the original 200-task population. If a separately authorized scientific follow-up is pursued, local translation fidelity is the component implicated by N−M; this screen does not yet justify a production selector, selective reconstruction engine or training sweep.
   - Evidence (computed-summary JSON pointers): /statistics/contrasts/R\_10-R\_random; /statistics/contrasts/N\_10-M\_10; /condition\_diagnostics/R\_10/cumulative\_reasoning\_fraction; /condition\_diagnostics/R\_10/actual\_attempt\_whole\_draw\_seconds; /condition\_diagnostics/H/actual\_attempt\_whole\_draw\_seconds; /condition\_diagnostics/D/actual\_attempt\_whole\_draw\_seconds; /supplements/economics~1summary.json/comparisons/D; /supplements/economics~1summary.json/perfect
   - Limitations: This is a local deployment-investment recommendation, not a universal impossibility claim or a rejection of all sparse attention research.; The 0.202% reference removes only receiver prefill; it does not bound architectures that change source reasoning or answer decoding. Such changes, including compact text or genuinely cheaper sparse decode, would require separate measured quality/cost evidence.; New-screen draw walls are diagnostic timings with different output lengths and scopes, not matched-length serving latency; they must not be mixed with the old source-inclusive economic population. No follow-on experiment has been launched.; Operational evidence gap: all 504 score records, the scored manifest and the stopped-worker proof were collected and verified, but the final CPU release/archive-hash control receipts were missed after SSH collection ended. Later authenticated provider inventory independently established zero running compute. See MAC\_STUDIO\_HANDOFF\_STATUS.md and preflight/diagnostic\_completion.json; no candidate was restarted for cleanup metadata.

## Limitations and reproduction

- Already-inspected development data, 12 task clusters, not 36 independent tasks or fresh confirmation.
- Unrun/unscored draws are not failed candidates; missing outcomes retain frozen denominators.
- No incomplete-data bootstrap or favorable native-success subset is used.
- Exact native oracle caches plus dense scoring were paid for; masked dense SDPA does not establish sparse speed.
- Sparse training-loss positions, sparse attention reads, and sparse KV construction are different interventions; evidence for one does not validate the others.
- A retained native KV entry already encodes earlier-token dependencies through preceding model layers. Selecting it for reading does not show it can be constructed from the corresponding raw token alone.
- Free-running arms share a selector rule, not necessarily selected indices: once their queries diverge, their layer/head/step selections may differ. Only the explicitly matched-query probes hold the query fixed.
- Full repair is controlled against matched native/native splice, not guaranteed identical to D: independently prefilling the prompt can change execution shape and numerical rounding.
- No jacq kernel or serving speedup is transferred to this experiment; its different workload, kernel, batching and integration regime is not measured here.
- Logical tensor/page bookkeeping is not measured HBM traffic or allocator-page reclamation.
- Selected fractions count historical reasoning (layer, KV-head, position) pairs, not global token unions. Original prompt and current/own-answer state remain outside the budget; full-history backing bytes include prompt state.
- Dense D/H have no selector-pair accounting: a zero raw selector counter is not zero attention reads or zero history memory. P has no reasoning intervention, but the shared diagnostic task still retains other arms' backing caches.
- Page estimates assume 16 absolute positions, either independent per-KV-head pages or pages shared across all KV heads; padded and boundary pages can exceed raw selected-row bytes. Neither layout is measured allocation or traffic.
- Per-layer KV-group mask bytes omit the expanded query-head mask and combined attention-mask scratch; score/softmax and gather scratch are not fully itemized. Tensor-shape bytes are not observed HBM traffic.
- Shared task backing contains native, mapped, hybrid, prompt-only and independent prompt caches. source_peak_then_freed is an earlier construction footprint, not simultaneously resident answer-state bytes; model weights and allocator overhead remain separate.
- Shared cache setup is charged once per task attempt, not 42 times from its repeated draw metadata. Draw-attempt wall stops before final working-set/answer/complete JSON writes; task-attempt wall includes those writes but excludes worker startup, old attempts and task-receipt persistence.
- Instrument stage native_key_scan combines current-query key scoring and probability normalization; replacement_and_mask combines gathers, replacements, dense-buffer construction and masks. Those components are not separately timed.
- Per-layer temporary-buffer maxima are not concurrent whole-model temporary peak; CUDA allocator peak includes different live allocations and is recorded separately.
- Unprofiled whole-answer wall time still includes CPU cumulative-union accounting and durable sampler I/O; it is diagnostic implementation cost, not bare model latency.
- Repeated16-token calibration timings distinguish unsynchronized whole-run measurements from synchronized stage profiling; neither is a complete-answer end-to-end cost proof.
- Calibration teacher-forced local probes are not free-running program quality.
- Sampler resumes reconstruct committed masks by replay; uncommitted crash-tail selections and timing may be missing.

Portable input SHA-256: `7cea22cd891b2604ee106750d0d1a6f51b2d95e82838726e28a50d1d5e00b920`. `input_manifest.json` records every collected source hash; regeneration reads compact saved metadata, not weights, hidden tests or candidate programs.

```sh
python3 scripts/sparse_repair_report.py --input results/sparse_repair_01/report/input_records.json --output results/sparse_repair_01/report --report SPARSE_REPAIR_RESULTS.md --decision-review results/sparse_repair_01/report/decision_review.json
```

Full-screen cluster intervals additionally require NumPy; incomplete-report regeneration uses only Python standard library. Raw receipts remain authoritative and original confirmation/publication artifacts are not rewritten.
