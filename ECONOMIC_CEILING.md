# Economic ceiling: faithful receiver-prefill replacement

**Records-only hypothetical, not a measured system or a universal bound.** No model generation, candidate execution, or private-test access occurred.

Removing all D receiver prefill saves **1.467935 seconds per answer (0.201929%)**, changing its recorded stage-sum mean from **726.955155 s to 725.487221 s**. That is little leverage in this regime.

## Population and definition

All 200 frozen primary confirmation task clusters × three declared draws = 600 records per arm, with no missing scores or timings and no outcome-based filtering. The repaired `gearshift_scorer_v2_20260919_01` outcomes are preserved. A = large-only; B = independent small-model reasoning; D = full native receiver replay; ROTATING_H = native prompt plus primary step-1,024 mapped reasoning.

For each task and draw: `perfect = D.single_output_inference_seconds − D.native_prefill_seconds`. Keep D's exact answer generation, answer length, outcome, full source reasoning charge, and cache-clone cost. Conversion and transfer are assumed free. Do not divide the source charge by three even though the experiment reused histories.

## Comparisons on the same recorded population

| Arm | Passed / 600 | Pass rate | Mean stage-sum seconds | Perfect saving vs arm (seconds) | Perfect saving vs arm (%) |
|---|---:|---:|---:|---:|---:|
| A (observed) | 386 / 600 | 64.33% | 742.830276 | +17.343056 | +2.334727% |
| B (observed) | 330 / 600 | 55.00% | 497.560189 | -227.927032 | -45.808937% |
| D (observed) | 390 / 600 | 65.00% | 726.955155 | +1.467935 | +0.201929% |
| ROTATING_H (observed) | 229 / 600 | 38.17% | 745.753790 | +20.266569 | +2.717595% |
| Perfect/free faithful converter (hypothetical) | 390 / 600, inherited | 65.00%, inherited | 725.487221 | — | — |

Savings are differences of population means divided by the observed comparison mean, not a mean of per-task percentages. Negative savings mean the hypothetical is more expensive. Quality is not equal across the observed arms; the perfect arm's quality is assumed to equal D, not newly demonstrated. There is no inferred retry policy, quality-adjusted price, or deployable router.

## Stage decomposition (mean seconds per answer)

| Arm | Full reasoning | Answer | Native prefill | Mapping | Splice | Clone |
|---|---:|---:|---:|---:|---:|---:|
| A | 696.644479 | 46.163754 | 0.000000 | 0.000000 | 0.000000 | 0.022043 |
| B | 470.557334 | 26.991243 | 0.000000 | 0.000000 | 0.000000 | 0.011612 |
| D | 696.644479 | 28.833123 | 1.467935 | 0.000000 | 0.000000 | 0.009618 |
| ROTATING_H | 696.644479 | 48.944106 | 0.064154 | 0.083780 | 0.009023 | 0.008247 |

D's source reasoning alone is 696.644479 s (95.830% of its stage sum). Removing receiver prefill cannot remove this work. Across the 600 fully charged D records, removed prefill totals 880.760926 s; this is NOT actual fleet time or a billing saving, because experimental setup/history reuse differ from this single-answer accounting.

## Timing limitations

- These are instrumented stage-sum estimates, not measured end-to-end serving latency or rental invoices. They preserve sampler checkpoint I/O and instrumentation already inside active timings; no model-only latency is invented by subtracting those costs.
- Bridge/first-token time is already inside answer time. Source-cache reconstruction, interrupted/resume reconstruction, model/checkpoint loading, fleet waiting, transfers, unrecorded crash tails, and CPU scoring are not added to these established single-output estimates. This arithmetic does not claim they are free.
- No confidence interval or generalization claim is made for these descriptive fixed-record differences. The 600 draws are clustered within 200 tasks, not 600 independent tasks; worker/hardware/load differences may affect timings.
- A true converter must pay conversion, transfer, memory residency, validity checks and fallbacks. Sparse decoding or source-stage acceleration changes answer/source computation and is a separate intervention; this reference does not bound fusion or those architectures.
- Oracle repair itself constructs the full receiver-native cache and pays dense scoring; it does not realize the hypothetical prefill saving. Fleet parallelism reduces study turnaround, not per-request work.

## Provenance and reproduction

Portable input SHA-256: `9825e94334cb9ddf5d78dc2d4c899bae569349cbd45b69b28cd1813942b7d158`. The input manifest records source paths, byte sizes and hashes; each selected raw answer and repaired score receipt is hash-checked (2,400 each), plus 400 closure-bound histories. The snapshot contains timing/outcome fields and identities only, not candidate code, hidden tests or model tensors.

From a checkout or the review archive root, using Python 3.10+ (standard library only):

```sh
python3 scripts/sparse_repair_economics.py --input results/sparse_repair_01/economics/input_records.json --output results/sparse_repair_01/economics --report ECONOMIC_CEILING.md
python3 -m unittest discover -s tests -p 'test_sparse_repair_economics.py'
```

To re-extract and independently verify the immutable original source records in a full checkout (not required for portable regeneration):

```sh
python3 scripts/sparse_repair_economics.py --extract-repo . --output results/sparse_repair_01/economics --report ECONOMIC_CEILING.md
```

`summary.json` and `per_task_draw_comparisons.csv` regenerate deterministically. `input_manifest.json` binds the portable snapshot; the review ZIP's outer SHA-256/manifest should be checked when transporting it.
