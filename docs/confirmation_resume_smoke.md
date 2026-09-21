# Bounded sampler resume proof

This separate engineering check uses one fixed public synthetic prompt. It never
opens confirmation prompts, histories, answers, private tests, or scores. It
does not change a frozen sampler, model policy, or experiment output. Passing
does not establish universal, long-context, or arbitrary-interruption equivalence.
Training checkpoint restoration is a different property and supplies no evidence
for this check.

Required inputs are the unchanged frozen primary code/declaration, the locally
downloaded pinned 32B and 8B models, and the original declared ROTATING step-1024
`mapper.pt` (302,327,647 bytes; SHA-256
`0e3caa7111ad8861f8e68e361ade6732e4dca5e587ca4f1ed8ba072a7ca9e386`).
No model download or provisioning is performed by this script. Use an otherwise
idle reserved H200; do not start it alongside a live generation worker.
The numerical runtime must match the original: PyTorch 2.8.0+cu128, CUDA 12.8,
Transformers 4.57.6, BF16, SDPA, deterministic algorithms, TF32 disabled.

The four cases are source reasoning, receiver native-cache answer, receiver
mapped-cache answer, and receiver hybrid-cache answer. All answer cases consume
the **same immutable synthetic baseline source history**. The original mapper,
full-history mapping, native prompt splice, single closing-think bridge, and
512-token prefill / one-token continuation schedule are retained.

Each draw has a cap of 128 generated tokens. The default deliberate interruption
is after the first sampled-and-forwarded nonterminal token at the frozen sampler's
first durable progress checkpoint. A stronger cut at 65, 129, or 193 may be fixed
**before any run**, provided the cap is larger and at most 256. If a draw terminates
before the selected cut, the check is inconclusive/fails; it is not rerun with a
different seed, prompt, cut, or policy until it passes.

## One-host fresh-process check

Set `REPO`, `SMOKE`, `LEASE`, and `LEASE_SHA` to the staged repository, a new
subdirectory of the lease's allowed result root, and the actual allocation lease
and its file SHA-256. Preparation needs no GPU and prints `plan_sha256`.

```bash
python scripts/coding_confirmation_resume_smoke.py prepare \
  --repo "$REPO" --output "$SMOKE" --cap 128 --interrupt-after 1
python scripts/coding_confirmation_resume_smoke.py run \
  --repo "$REPO" --folder "$SMOKE" --plan-sha256 "$PLAN_SHA" \
  --lease "$LEASE" --lease-sha256 "$LEASE_SHA" --timeout-seconds 3600
```

`run` loads both models in a baseline process, exits that process, starts four
separate interrupted workers (one per case), then starts a fresh resume process.
The intended workers terminate using `os._exit(86)` immediately after a fsynced
checkpoint and observation receipt. Exception cleanup cannot retain or rewrite
the checkpoint; the interrupted state remains `running`. All model and cache
objects disappear with their process. The resumed process reloads models, mapper,
durable tokens, and generator state, then reconstructs every cache from public
conditioning and one-token replay. No KV tensors are written or loaded.

The launch receipts check actual exit code 86. Any unexpected failure or timeout
stops the check without retries; completed or previously attempted phases cannot
be overwritten. The allocation guard's stop marker and deadline remain binding.

## Cross-host check

The individual phase interface allows the actual cross-region transfer to be
tested. On the first otherwise idle H200 run `phase --phase baseline --case all`,
then four separate commands `phase --phase interrupt --case CASE`, checking that
each returns exactly 86. Each command also requires the common `--repo`, `--folder`,
`--plan-sha256`, and its host's `--lease`/`--lease-sha256` arguments.

Copy the **whole smoke folder**, including all baseline records, raw interrupted
states, `.sampler.lock` files, and proof receipts, to the allowed result root on
the second H200. Do not copy caches or alter paths inside records. Stage exactly
the same source files, pinned models, and mapper separately. On the second host:

```bash
python scripts/coding_confirmation_resume_smoke.py phase \
  --repo "$REPO" --folder "$SMOKE" --plan-sha256 "$PLAN_SHA" \
  --lease "$LEASE" --lease-sha256 "$LEASE_SHA" --phase resume --case all
python scripts/coding_confirmation_resume_smoke.py verify \
  --repo "$REPO" --folder "$SMOKE" --plan-sha256 "$PLAN_SHA"
```

The plan is portable; pod identities live in attempt receipts and do not seed
sampling. The frozen sampler rejects a different runtime identity or different
checkpoint-reconstruction logits before taking another sample. A one-host run
reports `cross_pod_resume_for_all_cases: false`, even if every numerical check
passes. Only actual different pod identities can report cross-pod evidence.

## Required proof

`SMOKE_PROOF.json` is written only when all four cases have complete evidence.
It records separate exact comparisons for full token sequences, initial/final
generator states, sampler identities, final next-token logits, all final KV bytes,
all reconstructed checkpoint KV bytes and logits, base-cache bytes, preserved
interrupted prefixes, new processes, pinned runtime, caps, and memory headroom.
Cache equality covers every layer's K and V shapes, dtypes, and byte hashes;
no large tensors are retained. Raw synthetic records and durable states remain
hash-bound. A failed comparison returns a failing result and nonzero exit status.
Missing evidence, early termination, reconstruction drift, or an incomplete
process is never counted as a successful smoke check.

Local regression tests exercise the actual frozen sampler with tiny synthetic
CPU backends and real abruptly exiting child processes. These tests validate
the harness and failure handling; they are explicitly **not** H200 evidence.
