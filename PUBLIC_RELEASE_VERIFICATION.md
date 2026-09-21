# Public release verification — 2026-09-21

This is a newly constructed, allowlisted release tree, not rewritten research
history. The private archive preserves the original evidence and progress-01 tag.
No website changes, GPU jobs, benchmark model runs or new experiments were made.

## Checks on the release tree

- **Scientific identity:** retained result files are byte-identical to their source
  SHA-256 identities. Primary 4,800-draw and secondary 2,400-draw statistics reproduce
  with the frozen paired task bootstrap; seeds are not pooled. Sparse/economic full
  summaries and report bytes reproduce exactly. See [receipt](release/scientific_verification.json).
- **Tests:** 1,483 passed, 23 skipped, 48 subtests passed on macOS/Python 3.12.4;
  full collection, including eight new release-adapter tests. The 23 skips comprise
  14 Linux-only sandbox cases and 9 explicitly named historical-input/tokenizer
  integration cases. Test bodies are retained, not deleted. Warnings and all skips
  are disclosed in the [test receipt](release/test_verification.json).
- **Official retrieval:** all six pinned files, 4,485,994,821 bytes / 1,055 upstream
  rows, verified; all 400 selected IDs and source-row hashes matched. Downloads are
  private local artifacts, not part of the release. See [receipt](release/upstream_retrieval_verification.json).
- **Licensing:** standard Apache-2.0 for first-party source; exact approved upstream
  HumanEval+ gzip (164 records) and Apache/MIT notices preserved. Copied Qwen metadata
  has pinned upstream license files. LiveCodeBench inputs, raw generated outputs and
  WikiText/GSM8K sample payloads are omitted. Dependencies/models retain their terms.
  See [notices](THIRD_PARTY_NOTICES.md) and [license receipt](release/license_verification.json).
- **Secrets:** Gitleaks 8.30.1 scanned the complete tree. All 119 generic-key alerts
  were reviewed: 116 tokenizer-file SHA-256 identities, two judge-script SHA-256
  identities, and one literal contrast label. No credential was identified. The
  final Git database is separately scanned before release; hashes are not redacted.
- **Benchmark content:** explicit path allowlist; no raw answer/history/test archives.
  All public JSON is structurally inspected; long integer lists are position schedules
  or bootstrap indices, not token sequences. Text scan compares normalized 24-word
  windows against all 1,055 pinned upstream statements/interfaces; token scan compares
  7,039 interior windows from 380 locally available prompt copies. No matches remain.
  The initial numeric-table overlap was pure numbers, not problem text; the final text
  matcher requires at least eight alphabetic words. Approved HumanEval+ is a single
  exact-hash exception, not a directory-wide exemption. See [scan receipt](release/content_scan.json).

## Final visibility gate

The final release is initialized into a new Git database only after tree review.
Before public visibility, every object (including unreachable blobs) is inventoried
and scanned. It must contain exactly one commit, one branch and only the new
`gearshift-progress-02` tag; no progress-01 tag or historical refs are pushed. A
separate final receipt records the actual commit/object counts and anonymous checks,
so it does not create a self-referential commit hash or a second public commit.

## Limits

These are checks of retained evidence and software, not independent execution of
omitted historical candidates against private tests. Linux-only tests were not
executed on this Mac. No scan proves absence of every conceivable transformed or
very short fragment; conservative exclusion of raw coding payloads is the main
redistribution boundary. The original archive and immutable local bundle preserve
the full record. The local bundle is not independently off-site/WORM storage.

**Article:** no numerical correction is required by this sanitization. Link the
public progress-02 snapshot (not progress-01). Describe the public evidence as
numeric records/configs/code and reports, not publicly available raw benchmark
prompts/tests/generated programs. Keep the oracle, small reused population,
no-targeting-advantage and hypothetical economic-ceiling qualifications.
