# Secondary paired training launch contract

The two arms use the same frozen new training seed **20260919** and original
selected96 initializer. They each perform1024updates at32scoredpositions/update.
The existing v2 implementation, clipping, optimizer, prefix reconstruction,
resume comparison and checkpoint format remain unchanged. The primary
confirmation is independent and does not wait for this pair.

## Inputs and staging

Use `scripts/coding_confirmation_replication_stage.py inventory --manifest PATH`
after the final source commit. Its compact inventory contains the complete
Python source import closure, pinned configuration, exact104/21saved histories,
paired schedules and old numerical-control provenance. It excludes private
tests, confirmation answers, paired tensors and model weights. Rebuild the
manifest if code changes before upload. Verify it on the destination with
`scripts/coding_confirmation_replication_stage.py verify --manifest PATH`.
The launch check must not use `--compact-only`.

The separately staged original initializer must occupy
`results/coding_pilot_v1/recovery_train_20260917_04/train/mapper_step_0096.pt`:

- Bytes:302326175.
- SHA256:`0b9700ffd9cb38c23bcfa1327181b32992b128c6b95078214328382f86b8d68f`.
- Existing verified off-pod retrieval path is in
  `replication_checkpoint_inventory.json` and the prior backup manifest.

Reuse model weights only at the exact revisions in `pilot.json`, with the prior
shard hashes independently verified. `HF_HOME=/workspace/hf`. No prior full KV
cache is admitted. No runtime candidate sandbox is needed for training; this
runner executes no generated code. Runtime must match torch2.8.0,
transformers4.57.6, accelerate1.15.0, numpy2.2.6, psutil7.2.2,
safetensors0.7.0 and huggingface-hub0.36.2. Existing pinned environment may be
reused if its actual installed versions and CUDA image match; do not mutate a
shared environment while scientific workers run.

## Parent dispatch plan

The command-line `--plan` argument names a **per-pod allocation dispatch plan**,
not the scientific `replication_plan.json`. It requires:

```json
{
  "experiment_id": "confirmation_01_20260919T094418Z",
  "result_root": "results/coding_pilot_v1/confirmation_01_20260919T094418Z",
  "code_commit": "EXACT_40_CHARACTER_SOURCE_COMMIT",
  "lease_path": "evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/resources/ALLOCATION/lease.json",
  "lease_sha256": "EXACT_SHA256_OF_NEW_CONFIRMATION_LEASE"
}
```

The parent runtime appends `replication/arms/FIXED` or
`replication/arms/ROTATING`. The plan's lease fixes the actual pod identity,
absolute allowed result root and allocation deadline. It uses the new cumulative
spending policy; the old v2 dollar/GPU-hour guards are not imported or launched.

## Detached paired process contract

Launch only after the independent new lease guard is armed, its identity and
deadline are checked, volume durability is checked, and staged inputs/model
weights are verified. Two equivalent H200 GPUs are required, one per process.
Run from `/workspace/GearshiftConfirmation`. These are the exact worker argument
vectors, substituting the committed runtime Python, allocation plan and pod ID:

```text
PYTHON scripts/coding_confirmation_replication_worker.py --plan DISPATCH_PLAN --arm FIXED --worker-id POD_ID_replication_FIXED
PYTHON scripts/coding_confirmation_replication_worker.py --plan DISPATCH_PLAN --arm ROTATING --worker-id POD_ID_replication_ROTATING
```

Set `CUDA_VISIBLE_DEVICES=0` for FIXED and `CUDA_VISIBLE_DEVICES=1` for ROTATING.
Both children need `RUNPOD_POD_ID` equal to the lease pod, `HF_HOME=/workspace/hf`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`,
`TOKENIZERS_PARALLELISM=false`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`PYTHONUNBUFFERED=1`. Pass a credential-free environment. Use the parent's
detached local supervisor and guard registration protocol; children inherit its
process group. Persist stdout/stderr separately per arm. No SSH process or
laptop heartbeat governs progress. Both arms may proceed independently while
immutable checkpoint readers evaluate already committed endpoints.

Do not prebind `workers/.../identity.json`: unchanged v2 setup binds that file
after recording actual numerical runtime. Inspect both resume-preflight reports
for exact next-update equality and clean pristine restoration. Then verify both
step0 mapper tensor hashes equal the original v2 tensor hash
`6dd537e3d02372f088c2a54e5121f90fd9ab87ac3c6fc2311c87c2239a97bf77`.

On interruption, pass `--resume-checkpoint` naming a fully committed directory
under that same new arm root; the unchanged reader verifies manifest, all saved
states, schedule position and scientific identity. Never choose a partial
directory or a primary-v2 checkpoint. The fixed source commit is part of
checkpoint identity and must remain available for continuation.

Completed1024step checkpoints are the only secondary model endpoints. Retain all
six mapper/full-state checkpoints per arm and separate actual-model resume
evidence. Reserve at least40GB additional storage for the paired complete
checkpoints plus verified full-state backup copies. Prior equivalent run took about7.3hours per arm; setup and seed
variation add uncertainty. No intermediate quality result changes the endpoint.

After both arms finish, the parent may run the predeclared2400secondary answers
against the exact200primary confirmation histories, task seeds and controls.
The inherited21task validation panels are setup integrity inputs only; this
training runner generates no validation answers or scores.
