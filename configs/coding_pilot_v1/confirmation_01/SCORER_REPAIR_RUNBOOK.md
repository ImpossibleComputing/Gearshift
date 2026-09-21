# Versioned scorer repair

The original scorer and all original coverage-v2 receipts are immutable. The new CPU scorer imports their exact extraction and comparison functions. Its resource policy was frozen from public constraint-derived correct reference programs, before rescoring any saved candidate. No mapper or generated program is changed.

Scoring machine: dedicated CPU pod `m6jcx23rppe7j6`, 16 allocated logical CPUs / eight distinct physical cores, 64,000,000,000-byte cgroup memory limit. Use `/opt/gearshift-scorer-venv/bin/python` as the controller; candidates use isolated `/usr/bin/python3` with chroot, unprivileged UID, seccomp, memory/process/file/output/time limits and no secrets. The eight fixed scorer cores are recorded in `frozen_policy.json`.

From `/workspace/GearshiftConfirmation`, the durable execution command is:

```sh
/opt/gearshift-scorer-venv/bin/python -u scripts/coding_scorer_repair_execute.py \
  --evaluation-root results/coding_pilot_v1/coverage_generalization_v2_20260918T233720Z/evaluation \
  --private-tests data/coding_pilot_v1/private/coverage_generalization_v2_exact.json \
  --policy evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/calibration/frozen_policy.json \
  --output results/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair
```

Start it detached with a persisted launcher log. It verifies all original answer/score hashes, unchanged extracted programs, original test counts and every recorded original test-input/expected-output hash before freezing the rescore plan. It executes the fixed audit subset three times, then uniformly scores all 1,512 saved answers on eight bounded workers, closes the manifest, and renders the report. Diagnostic repeats never substitute for the uniform scoring transaction. No algorithmic timeout is retried. Exactly one retry is allowed for verified launch/confinement/capture failures; a worker crash after a transaction starts leaves explicit missing coverage on recovery.

The exact original filtered 21-task test file SHA-256 is `be1bb62fd5574011a6f7278d46f727abf71440d1e1387d95b30ae28aabd1e7bc`. A broader Studio file has a different hash and is not used. Private files stay on the CPU-only volume and outside every candidate chroot; they are never packaged with public source or sent to GPU generation workers.

Frozen scorer identity and policy: `evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/calibration/scorer_identity.json`. CPU 12 s, wall 46 s, address space 4 GiB, complete stdout capture with 16 MiB minimum and four-times-reference headroom plus 1 MiB, bounded at 64 MiB. Candidate-dependent allowances or favorable reruns are forbidden. All original files remain unchanged.

For an interrupted supervisor, use the same command and immutable plan. Completed score transactions are skipped only when their bindings match. A started-but-uncommitted candidate transaction is marked missing, never silently rerolled. Inspect the versioned missing receipt before any separately declared recovery.

Report-only reproduction requires compact receipts, NumPy and Matplotlib, with neither private tests nor candidate execution:

```sh
python scripts/coding_scorer_repair_report.py \
  --root results/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair
```

The resulting `report/SCORER_REPAIR_RESULTS.md`, JSON/CSV, joint bootstrap indices and plots are safe compact review inputs. The report includes all original-versus-corrected changes, actual coverage, missing-outcome sensitivity and fixed diagnostic repeatability. These are previously inspected validation tasks, not fresh confirmation.
