# Confirmation: separate second training seed

Training seed `20260919`; 200 shared frozen tasks × 4 mapped/hybrid conditions × 3 answer draws = 2400 secondary answers; 0 missing.

Primary declaration `16387d5361c795121929fd9244821f12ee005f9b36b2febddca5d9ef89f7536f`; secondary addendum `02b3c131340b27d7c2eed93378d1fff5d648558de7f9354415f9009366e79eec`. The primary report and original training seed remain independent. This report does not pool the two training seeds or select the better seed/checkpoint.

| Secondary condition | Passed / draws | Missing | Mean success |
|---|---:|---:|---:|
|FIXED_M|95/600|0|15.83%|
|ROTATING_M|197/600|0|32.83%|
|FIXED_H|196/600|0|32.67%|
|ROTATING_H|250/600|0|41.67%|

All intervals below are exploratory, unadjusted 95% paired task-cluster intervals. The same 10,000 PCG64 task-index resamples and task order as the primary analysis are used; each task averages all three fixed draws. Intervals touching/crossing zero are inconclusive, not equivalence.

| Contrast | Difference, percentage points | 95% interval | Source of negative condition |
|---|---:|---|---|
|ROTATING_M-FIXED_M|+17.00|[+12.67, +21.50]|Same secondary training seed|
|ROTATING_H-FIXED_H|+9.00|[+5.50, +12.67]|Same secondary training seed|
|ROTATING_H-ROTATING_M|+8.83|[+4.50, +13.33]|Same secondary training seed|
|ROTATING_H-D|-23.33|[-29.00, -17.83]|Reused primary native reference|
|ROTATING_H-P|+16.00|[+10.83, +21.17]|Reused primary native reference|
|ROTATING_H-B|-13.33|[-19.17, -7.67]|Reused primary native reference|
|ROTATING_H-A|-22.67|[-28.34, -17.17]|Reused primary native reference|

All four declared primary native controls reused. Native A/B/D/P records, when shown, are reused primary measurements, not new secondary generation or additional replication draws. Primary mapped outcomes are never pooled into this analysis.

Missing infrastructure outcomes remain explicit with full-denominator extreme 0/1 sensitivities. No complete-case or best-of-three substitution is made. Two paired training seeds provide limited replication conditional on one initializer and corpus; this is not exhaustive run-to-run uncertainty.

Full reused primary source history is charged to each single-answer stage-sum estimate. H pays native prompt prefill. Cache reconstruction/model loading/checkpoint I/O remain separate; failed-answer shortening is not useful speedup. Fleet accounting does not charge shared historical generation twice.

A result against P does not isolate dependence on the particular transferred reasoning. No mechanism or architecture claim follows from this secondary comparison.

Frozen secondary mapper endpoints:

- FIXED: step 1024; mapper `483370a79aee714c5c985291c84727e43f3bacceae6ddfcc9828b814ded81efb`.
- ROTATING: step 1024; mapper `8f2d41480ef98ee3d713357bc400506694c7b14b6a2f9f69ffaef5d4af210822`.

Joint task-resample matrix identity: `ad9c0731863d2a325b97ff6f04b3c25091374e852aa54cb8a269186b74e79955`.

Regeneration uses the compact generation/scoring manifests, raw answers, histories and receipts only. It does not load model weights, read private tests, execute candidates or modify the primary report.
