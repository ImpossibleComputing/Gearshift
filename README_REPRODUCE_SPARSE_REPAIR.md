# Reproducing the sparse-state/repair review

This is a compact research review, not a model distribution or deployment. The
frozen declaration is `configs/coding_pilot_v1/sparse_repair_01/declaration.json`.
It specifies four calibration cases, twelve disjoint development screen cases,
fourteen conditions, and three draws per screen case: **504 planned answers**.
Planned coverage is not actual coverage. Read `SPARSE_REPAIR_RESULTS.md` and the
manifest-bound raw records for executed coverage, numerical failures, and gaps.

## Verify transport first

Compare the ZIP SHA-256 and size with the separately supplied receipt. From the
full checkout or extracted archive (Python 3.10+, standard library only):

```sh
python3 scripts/sparse_repair_review.py --verify /absolute/path/gearshift_sparse_repair_review.zip
```

The verifier hashes every member, rejects unsafe paths/symlinks, and verifies a
fresh temporary extraction. `SPARSE_REPAIR_FILE_MANIFEST.json` binds every other
member's size and SHA-256. Its own hash is in the external delivery receipt.
`GIT_SOURCE_IDENTITIES.json` distinguishes the archived working-tree bytes from
the Git HEAD and records the preserved publication tag.

## Regenerate reports without models or candidate execution

Run from the extracted archive root. Economics and incomplete-diagnostic reports
use Python standard library only. Complete-screen task-cluster bootstrap intervals
also require NumPy (the recorded analysis environment/version is provenance).
Neither command loads weights, accesses the network/private tests, or executes
candidate programs; both consume compact saved input snapshots.

```sh
python3 scripts/sparse_repair_economics.py \
  --input results/sparse_repair_01/economics/input_records.json \
  --output results/sparse_repair_01/economics \
  --report ECONOMIC_CEILING.md

# Optional analyst interpretation is a separate, hash-bound artifact. Keep this
# path relative and run from the extraction root for byte-identical Markdown.
set --
if test -f results/sparse_repair_01/report/decision_review.json; then
  set -- --decision-review results/sparse_repair_01/report/decision_review.json
fi
python3 scripts/sparse_repair_report.py \
  --input results/sparse_repair_01/report/input_records.json \
  --output results/sparse_repair_01/report \
  --report SPARSE_REPAIR_RESULTS.md "$@"
```

`REPRODUCE_REPORT_COMMANDS.json` contains the same canonical argv lists, adding
`--decision-review` only when that artifact is included. The optional review
contains five explicitly labelled analyst interpretations, not measurements. It
must bind the exact numerical input snapshot and frozen declaration; the report
validates its verdicts and evidence references. It is neither inserted into nor
used to rewrite `input_records.json`, avoiding a circular evidence dependency.
Absence of this review is not permission to invent a verdict or create one during
reproduction. Incomplete diagnostics permit only unavailable/blocked decisions.

If a report snapshot is absent, that is a packaging/incomplete-coverage issue,
not permission to regenerate answers or synthesize missing observations. Do not
use `--collect` merely to reproduce an existing report: that option re-collects
the current compact evidence instead of reading the archived frozen snapshot.

## Evidence boundaries

- `results/sparse_repair_01/artifact_audit.json` records exact primary ROTATING
  step-1,024 mapper, model, scorer, and original saved-history identities, plus
  verified retrieval paths and remaining limitations at the time of its audit.
- The ZIP contains exactly the sixteen selected saved development histories,
  plus the original public forty-task input file to preserve its byte identity.
  It does not contain all old result trees or the 200-task confirmation corpus.
  The economics snapshot contains the relevant same-population recorded fields.
- No private test values, credentials, language-model/mapper weights, KV tensors,
  environments, or nested backup archives belong in this ZIP. Public private-test
  **identity receipts** contain hashes only and are not test inputs.
- Oracle repairs pay full native receiver prefill and native-key scans. Masked
  dense attention does not establish sparse latency savings. Small instantaneous
  selections do not establish a small complete-answer cumulative working set.
- Sparse training-loss positions, sparse attention reads, and sparse KV
  construction are different interventions. A native KV entry already encodes
  earlier-token dependencies through preceding model layers; retaining that entry
  does not show it can be built from its corresponding raw token alone.
- Free-running arms use the same selector rule, not necessarily the same selected
  indices. Divergent queries can produce different layer/head/step selections.
  Only the explicitly matched-query local probes keep the query fixed.
- Full repair is checked against matched native/native splice, not guaranteed
  identical to D. Independently prefilling the prompt can change execution shape
  and numerical rounding; reported cross-shape discrepancies are not waived.
- No jacq kernel or live-serving speedup transfers to this diagnostic. Its
  workload, kernels, batching, and integration regime are not measured here.
- Optional `preflight/gpu_launch.json` records calibration/GPU launch; it is not
  evidence of screen generation. `preflight/screen_launch.json` and
  `preflight/screen_all_workers_generating.json` separately preserve observed
  screen-generation progress. Their timestamps describe historical observations,
  not current worker liveness, completion, or quality. These receipts are
  hash-bound supplements in the report snapshot when available.
- No complete-screen quality conclusion is drawn before all 504 frozen answers
  have sealed generation, valid binary scores, and verified evidence integrity.
- The independent corrected scorer executes only in the isolated Linux CPU
  environment after generation closes. Report regeneration does not invoke it.
- Publication remains separate: no website edit, push, deployment, or tag movement
  is performed by any report or review-packaging command.

### Reading working-set, memory, and timing records

The budget applies independently to each historical reasoning layer/KV-head pair;
cumulative fractions are unions of `(layer, KV-head, position)` pairs over the
whole answer, not global token unions. The original prompt and current/own-answer
states remain outside the budget. Dense D/H zero selector counters mean that
selection accounting is inapplicable, not that dense attention reads zero state.

The two page estimates assume 16 absolute positions per page: independent pages
for each KV head versus pages containing all heads. Padding and boundary pages
can exceed raw reasoning-row bytes. Neither is a physical allocation or HBM
traffic measurement. The compact working-set output preserves both layouts and
per-head cumulative fractions. Logical KV-group mask sizes omit the expanded
query-head mask and combined-mask scratch; per-layer buffer maxima must not be
summed into a claimed concurrent temporary peak.

Shared diagnostic residency includes native, mapped, hybrid, independent-prompt,
and P caches, including while P runs. `source_peak_then_freed` is the earlier
source-cache construction footprint, not another simultaneously live answer
cache. Model weights, current live cache, allocator reserve, and scratch still
matter; selection fractions do not describe total GPU memory.

Shared setup/check stages are charged once per task attempt, not once for each
of the 42 draw records that repeat them. Draw-attempt wall time stops before the
final working-set/answer/complete-file writes. The hash-bound task receipt retains
the broader task-attempt wall and final fingerprint check, but excludes worker
startup, prior attempts, and its own persistence. Sampler `completion_timing.json`
adds final checkpoint/materialization accounting beyond the answer's earlier
snapshot. Its components overlap active/draw wall time: do not add them again or
subtract them to claim model-only latency. None of these scopes is a serving
invoice or a matched-length speedup. Saved source reasoning was reused, not run
again for this screen.

Optional public `recovery/decision.json`, `resume_stage.json`, `resume_launch.json`,
and `recovery_validation.json` preserve infrastructure continuation separately
from final answer coverage. A staged checkout is not launched work; process launch
is not verified replay. Even a final 504/504 report retains the earlier deadline
interruption, original committed-token checkpoint metadata, repeated cache-setup
attempt receipts, and available reconstruction timing. The original interrupted
progress union is not a completed answer's working set. Replay of committed
history cannot recover discarded/uncommitted crash-tail selections or missing
timing. Do not describe the continuation as a free retry, count its duplicated
setup as though performed only once, or add replay twice to an attempt wall that
already includes it. Byte-identical completed draws, unchanged committed prefixes,
and reconstruction-logit fingerprints require explicit validation evidence; final
coverage alone does not establish them. These optional public receipts and compact
metadata are hash-bound in the numerical input snapshot when collected.

## Re-running measurements is a separate operation

Measurement source and tests are included for inspection. It requires the exact
pinned H200/CUDA runtime and separately recovered weights/mapper, the four actual
CUDA calibration gates, isolated scoring inputs, current provider authorization,
exclusive orchestration ownership, and a fresh bounded cost reservation. An
extracted review ZIP is not, by itself, a complete runnable experiment backup.
Use only the recorded declaration/lease/job commands and preserve immutable
completed draws; do not substitute checkpoints or regenerate source reasoning.

CPU-only tests for contracts and small synthetic tensor cases are not substitutes
for the actual-model CUDA calibration receipts. Historical environment lockfiles
are provenance and should not be interpreted as instructions to execute model or
candidate code during review.

### Detached isolated CPU scoring (authorized operators only)

After all 504 public generation transactions are sealed, the bounded wrapper is
`scripts/sparse_repair_cpu_job.py --plan <relative-plan.json> --plan-sha256 <sha>`.
Launch it once with the existing isolated scoring Python and
`subprocess.Popen(..., start_new_session=True)`; it must be Linux root, with an
already armed independent guard and a fresh dedicated CPU lease on the existing
private volume. It does not allocate resources or arm/restart guards.

The JSON plan contains `experiment_id` (`sparse_repair_01`), `job_id`,
`lease_path`, `lease_sha256`, `declaration_path`, `declaration_sha256`,
`result_root`, exactly eight distinct physical `cpu_ids`, `automatic_retries: 0`,
and `implementation_hashes` from the wrapper's `implementation_hashes()` helper.
Paths are checkout-relative. The separately retained private input is fixed at
`private/development.json`, owner-only, outside the public result root; it is
neither supplied by the ZIP nor copied into release backups.

The wrapper re-verifies public closure, checks capacity/sandbox setup, then runs
the unchanged trusted-reference preflight followed by `prepare`, `run`, and
public-only `finalize`, once each. After all own workers stop, it verifies two
compact **public-only** backups on the same retained volume and requests release
through the existing guard. These copies are not geographically independent.
Failure preserves evidence; it does not retry candidate outcomes. Local wrapper
tests are mocked and do not establish actual CPU preflight or scoring success.

## Packaging implementation

`python3 scripts/sparse_repair_review.py` defaults to inventory-only. `--build`
is an explicit packaging action and refuses to overwrite an existing archive.
It includes allowlisted reports, new diagnostic source/tests/configuration,
transitive local code dependencies, compact public evidence, and selected saved
histories. It records exclusions and scans for known credential formats without
mistaking ordinary words such as “key” in attention code for credentials.

The uncompressed archive guard is **1,024 MiB total, 128 MiB per member**. The
earlier 768 MiB engineering limit left little room once all 504 raw draws and
their durable state/traces were collected, before final scores and reporting.
The total bound was raised instead of deleting required evidence; the user did
not specify a numeric archive-size cap. This is not a promised compressed ZIP
size. All private-value, credential, tensor, symlink and nested-archive exclusions
are unchanged, and the manifest records these limits.

The frozen `private_scoring_storage_01.json` and its hash-bound public provider
volume receipt are included when present. They document an explicitly isolated
scoring-storage destination, not its private contents or permission to reuse it.

## Completed delivery scope

The delivered screen has 504 generated answers and 504 binary scores, zero missing
scores, and four passing required calibration cases. All original data and the
publication tag are preserved. The five conclusions are separately hash-bound in
`results/sparse_repair_01/report/decision_review.json`; they are analyst
interpretations, not additional experiments or authorization to launch one.

The final CPU's scientific records, score manifest and stopped-worker proof were
collected and verified. Its later guard-release/archive-hash receipts were not
retrieved after SSH collection ended. A subsequent authenticated provider
inventory separately confirmed no running compute. See
`results/sparse_repair_01/preflight/diagnostic_completion.json` for this limited
cleanup-metadata gap and the verified scientific-evidence references.

All seven original network-volume identities remain. The private volume was
expanded from 20 to 25 GB after a public-staging quota failure, without deleting
original data. The unused new 20 GB fallback volume was removed; its frozen receipt
is historical provenance, not an available deployment destination. Continuing and
unposted storage costs are not zero and are excluded from the quote-times-elapsed
compute estimate. Historical operator scripts under
`results/sparse_repair_01/operations/cpu_final/` must not be replayed as launch
instructions.
