# Scorer repair: saved coverage-v2 validation

This uniformly rescored the saved programs; no new answers were generated or repaired. Original scores and receipts remain unchanged. These 21 previously examined tasks provide exploratory validation, not fresh confirmation.

Coverage: 1512/1512 score transactions; 0 unresolved infrastructure outcomes. Three draws per task/condition; 21 task clusters.

| Condition (update 1024) | Original passes | Corrected passes | Corrected rate / 95% interval |
|---|---:|---:|---|
|FIXED_1024_M|17/63|17/63 (0 missing)|27.0% [12.7,42.9]|
|ROTATING_1024_M|30/63|30/63 (0 missing)|47.6% [30.2,66.7]|
|FIXED_1024_H|38/63|38/63 (0 missing)|60.3% [41.3,79.4]|
|ROTATING_1024_H|44/63|45/63 (0 missing)|71.4% [54.0,88.9]|
|D|58/63|63/63 (0 missing)|100.0% [100.0,100.0]|
|P|29/63|31/63 (0 missing)|49.2% [31.7,65.1]|

Original primary ROTATING−FIXED pure-mapping contrast: +20.6 percentage points [+11.1,+31.7].

Corrected primary ROTATING−FIXED pure-mapping contrast: +20.6 percentage points [+11.1,+31.7].

All intervals use 10000 identical paired task-cluster resamples and are exploratory, unadjusted. They do not treat an interval crossing zero as equivalence. Missing draws are neither dropped nor treated as wrong; JSON includes conservative sensitivity bounds.

The repaired scorer captures stdout completely, including valid multi-megabyte output, with bounded output, CPU, wall, memory, file descriptors and process count. Function-result serialization and discarded function stdout are bounded too. Extraction and comparison import the unchanged original functions. Network, host files, subprocesses and writes remain blocked.

Frozen per-test CPU=12s, wall=46s. Allowance derives from correct public reference stress runs before candidate rescoring. Only verified infrastructure failures receive one retry; algorithmic timeouts receive none. A crashed score transaction is missing coverage.

Limits derive from the [ABC344E constraints](https://atcoder.jp/contests/abc344/tasks/abc344_e) and [linked-list editorial](https://atcoder.jp/contests/abc344/editorial/9503), the [ABC335C constraints](https://atcoder.jp/contests/abc335/tasks/abc335_c) and [history editorial](https://atcoder.jp/contests/abc335/editorial/9293), and the [ABC339D constraints](https://atcoder.jp/contests/abc339/tasks/abc339_d) and [paired-state BFS editorial](https://atcoder.jp/contests/abc339/editorial/9273). Reference source hashes, construction rules, allocation and every repeated measurement are retained in calibration receipts.

Changed outcomes by recorded cause:
- old_output_ceiling_failure_now_pass: 9
- old_timeout_now_pass_under_calibrated_policy: 4
- unchanged_failure: 788
- unchanged_pass: 711

The fixed diagnostic subset has 15/15 complete three-repeat records; 0 varied between pass and fail. Every repeat is retained; none substitutes for the uniform rescore. A timeout-to-pass change reflects the calibrated policy/environment and is not automatically labeled a spurious original timeout.

Reproduction: run `python scripts/coding_scorer_repair_report.py --root <scorer_repair_root>`. Tables and plots require only the frozen plan and compact score receipts; no private tests, candidate execution, model weights or GPU are used.
