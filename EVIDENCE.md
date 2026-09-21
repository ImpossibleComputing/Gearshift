# Public evidence scope

The release is a reviewed, one-commit research artifact, not a mirror of the private
working repository. The original record, branches, tags and experimental provenance
remain unchanged in a private archived repository and a verified immutable local
Git bundle. The local bundle is not an independently off-site/WORM backup.

## Included

- Mapper, cache, generation, scoring and analysis source; all original test bodies.
- Frozen metadata/configs, task IDs, schedules, versions, hashes and outcome records.
- Primary 4,800-draw and secondary 2,400-draw CSVs and original statistical summaries.
- Sparse 504-draw numerical snapshot, calibration diagnostics and working-set traces
  (selected **positions**, not token IDs); economic 2,400-record snapshot.
- Original reports, approved upstream HumanEval+ fixture and license notices.
- Source-file SHA-256 provenance in `release/source_manifest.json` and verification
  receipts. Historical SHA references identify private records, not public commits.

## Intentionally omitted

LiveCodeBench visible problems, grading tests, source/receiver prompt-token arrays,
raw reasoning, generated answers, raw scorer diagnostics, operational packet archives,
model/mapper weights, tokenizers, and WikiText/GSM8K sample payloads are not bundled.
Generated answers are conservatively omitted rather than presumed independent.
Safe result records/hashes remain; no result has been rescored or selectively removed.
Copied LiveCodeBench dataset-loader source is unnecessary and omitted; the public
fetcher uses HTTPS and standard JSON only. Upstream retrieval is separate from licensing.

Historical reports and code retain some original paths and progress-01 identifiers
as provenance. A mentioned raw artifact is not a promise it exists in this public
checkout. Original report/manifest hashes describe original/private source artifacts;
`release/PUBLIC_FILE_MANIFEST.json` describes the final public files.

## What verification establishes

The public adapter recomputes primary/secondary statistics from complete retained
per-draw CSVs with the frozen 10,000 paired task-cluster resamples, separately by
training seed. Sparse and economic full summaries and report bytes reproduce exactly.
The underlying result input files remain byte-identical to the private research state.
This is not independent regrading or a new model experiment. A public reviewer can
check statistics and code but cannot inspect omitted original programs/tests here.

The release scanner checks all public Git blobs (including unreachable objects),
known content-bearing fields, private prompt-token windows, and normalized 24-word
overlaps with the pinned upstream statements/interfaces. Numerical overlap alone is
not benchmark text. The exact approved HumanEval+ gzip is separately hash-verified.
These checks are supported by a narrow source/metadata allowlist and complete omission
of raw coding evidence, not a claim to detect every conceivable transformed short text.
