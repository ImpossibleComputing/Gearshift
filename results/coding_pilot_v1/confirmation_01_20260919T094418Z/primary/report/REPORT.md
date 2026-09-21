# Fresh-task confirmation

ROTATING_M − FIXED_M: +10.17 percentage points (paired 95% interval +6.33 to +14.33).

Supports a positive difference under the frozen primary rule.

Experiment `confirmation_01_20260919T094418Z`; primary training seed `20260915`; declaration SHA-256 `16387d5361c795121929fd9244821f12ee005f9b36b2febddca5d9ef89f7536f`.

Scientific source commit `60adc572d62da3243bac4396fbb9163a6eb403ac`; preserved publication commit `64725974fa55459350d1c9d09037bab64d0c5ec6`.

Population: 200 frozen task clusters; three fixed answer draws per task and condition. 0 explicit missing outcomes across all conditions.

| Condition | Passed / declared draws | Missing | Mean pass rate |
|---|---:|---:|---:|
| A | 386 / 600 | 0 | 64.33% |
| B | 330 / 600 | 0 | 55.00% |
| D | 390 / 600 | 0 | 65.00% |
| P | 154 / 600 | 0 | 25.67% |
| FIXED_M | 114 / 600 | 0 | 19.00% |
| ROTATING_M | 175 / 600 | 0 | 29.17% |
| FIXED_H | 191 / 600 | 0 | 31.83% |
| ROTATING_H | 229 / 600 | 0 | 38.17% |

The six prespecified hybrid comparisons use Bonferroni 99.1667% intervals, with ordinary 95% intervals retained as descriptive. All comparisons share the exact saved 10,000 task-cluster resamples.

| Contrast | Difference (pp) | Descriptive 95% interval (pp) | Bonferroni 99.1667% interval (pp) |
|---|---:|---:|---:|
| ROTATING_H − FIXED_H | +6.33 | [+2.83, +10.00] | [+1.67, +11.33] |
| ROTATING_H − D | -26.83 | [-32.67, -21.33] | [-34.33, -19.50] |
| ROTATING_H − P | +12.50 | [+7.67, +17.33] | [+5.83, +19.17] |
| ROTATING_H − B | -16.83 | [-22.50, -11.33] | [-24.67, -9.33] |
| ROTATING_H − A | -26.17 | [-32.00, -20.50] | [-33.83, -18.67] |
| ROTATING_H − ROTATING_M | +9.00 | [+4.33, +13.83] | [+2.67, +15.50] |

Frozen mapper identities:

- FIXED: step `1024`, mapper SHA-256 `da2a374bcd4af01b796ec1f9a7f98577968184e744b92f27bdd0e1ce1c707739`.
- ROTATING: step `1024`, mapper SHA-256 `0e3caa7111ad8861f8e68e361ade6732e4dca5e587ca4f1ed8ba072a7ca9e386`.

Recorded single-output stage-sum estimates are descriptive seconds, with the full history charged to each answer:

| Condition | All outcomes mean | Passing answers mean | Failing answers mean | Unmeasured draws |
|---|---:|---:|---:|---:|
| A | 742.83 | 529.96 | 1126.80 | 0 |
| B | 497.56 | 311.27 | 725.25 | 0 |
| D | 726.96 | 517.22 | 1116.47 | 0 |
| P | 28.98 | 15.57 | 33.61 | 0 |
| FIXED_M | 752.38 | 299.15 | 858.69 | 0 |
| ROTATING_M | 734.67 | 355.61 | 890.75 | 0 |
| FIXED_H | 756.31 | 321.66 | 959.28 | 0 |
| ROTATING_H | 745.75 | 336.11 | 998.61 | 0 |

Separate measured setup: 19 unique worker model-loading receipts and 712 unique mapper-loading receipts; repeated references from three draws are counted once. Details are in setup_measurements.csv when present.

Missingness sensitivities retain every task and all three draw denominators. Their intervals concern extreme assigned outcomes, not observed complete-data inference.

Timing charges the full reasoning history to each single-answer estimate; B uses its independent small-model history, and H includes receiver-native prompt prefill. Bridge time is already inside answer time. History-only three-answer amortization is a separately labelled deployment estimate. Cache reconstruction, worker model setup and checkpoint loading remain separate. Closure-bound sampler sidecars report measured checkpoint persistence, preparation, telemetry, progress publication and materialization overhead; none is added again to the unchanged active sampler time or subtracted to invent model-only latency. Sidecar/attempt-receipt writes and unknown crash tails remain excluded or incomplete as recorded. Unmeasured fleet, rental and interrupted setup costs are not invented.

Timing is split by pass/failure/missing outcomes. Short failed answers are not evidence of useful speedup, and fleet parallelism is not a per-answer latency gain. Static interface checks inspect syntax and required public names only; correctness comes from the frozen scorer.

H beating P alone does not establish the causal contribution of the particular transferred reasoning. A second training seed, when available, must remain a separate report; this primary report does not wait for it.
