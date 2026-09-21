# Original generation quiescence and immutable snapshot

This helper is limited to the two original generation allocations:

- `x6jy63vp7jvx6n` (`primary_pair01`)
- `v7ud2o1wx86y9e` (`primary_pair02`)

The training allocation `rekbqruqox2bk9` is explicitly excluded from stop actions.
It is the required read-only host for snapshotting the surviving AP volume.
The helper makes no provider calls, deletes no pods or volumes, loads no models,
and changes no frozen numerical source or training files.

**Do not invoke `stop` until the controller has verified the actual cross-pod
H200 sampler resume proof and all eight secured replacement GPUs are ready (twice the original generation capacity).** Preparation
and inspection do not authorize stopping. Only an explicit `stop --execute`
signals a process. The controller also keeps the old unsharded launcher/monitor
from restarting original workers after quiescence.

## Freeze the control plan

Stage the committed new helper alongside the unchanged original code and public
allocation lease receipts. On a machine with the repository:

```bash
python scripts/coding_confirmation_quiesce_original.py prepare \
  --repo "$REPO" --output "$CONTROL_PLAN"
```

`CONTROL_PLAN` must be a new repository-relative path under `evidence/`, such as
`evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/migration/control_plan.json`.
The command prints the plan path and SHA-256. Preserve those exact plan bytes
on both originals and the snapshot host. It pins the helper source, frozen
primary declaration, exact original leases, shared result store, and excluded
training pod. It reads only public configuration and code.

## Inspect, then explicitly stop each original

Run locally on each named original pod, using its own `POD` value:

```bash
python scripts/coding_confirmation_quiesce_original.py inspect \
  --repo "$REPO" --plan "$CONTROL_PLAN" --plan-sha256 "$CONTROL_PLAN_SHA" \
  --pod-id "$POD"

python scripts/coding_confirmation_quiesce_original.py stop \
  --repo "$REPO" --plan "$CONTROL_PLAN" --plan-sha256 "$CONTROL_PLAN_SHA" \
  --pod-id "$POD" --output "$NEW_STOP_RECEIPT_DIRECTORY" --execute
```

The helper checks the actual local pod ID, registered supervisor PID/process
group, Linux boot ID and process start ticks, working directory, exact original
supervisor entrypoint, unchanged source, and dispatch-to-lease identity. It opens
a Linux process handle (`pidfd`), checks the identity again, and sends **SIGTERM
only to that supervisor**. Its existing handler stops its workers, preserves
partial jobs, verifies both backup archives, and writes its release receipt.
The already-armed independent lease guard handles provider release.

There is no PID-only fallback, broad process kill, direct worker kill, provider
delete, or automatic signal retry. Linux `pidfd_open` and `pidfd_send_signal` must
be available. A reused receipt directory is refused. If a request receipt exists
without `signal_sent.json`, inspect the actual state before taking another action;
do not infer whether the signal was sent from a lost console connection.

The stop command returns without waiting for provider deletion. It does not claim
that billing stopped. Allow the original supervisors to complete their own backup
and release path; a failed or incomplete release blocks the later snapshot.

## Record provider absence

After successful provider queries, save sanitized public receipts for **both**
originals. Each receipt must have:

```json
{
  "pod_id": "THE_ORIGINAL_POD_ID",
  "source": "runpod_provider_inspection",
  "state": "provider_confirmed_absent",
  "observed_epoch": 0,
  "verification_method": "get_pod_404",
  "http_status": 404
}
```

Use the actual observation timestamp and actual provider response. An API error,
failed authentication, timeout, or incomplete listing is not absence evidence.
Alternatively `verification_method: "complete_pod_list"` requires status 200,
`complete_inventory: true`, and the actual complete unique `listed_pod_ids`, with
the referenced original absent. Credentials and provider authorization headers
must never enter these receipts.

Create an index containing exactly two entries:

```json
{
  "receipts": [
    {"pod_id": "x6jy63vp7jvx6n", "path": "evidence/.../pair01_absent.json", "sha256": "ACTUAL_FILE_SHA256"},
    {"pod_id": "v7ud2o1wx86y9e", "path": "evidence/.../pair02_absent.json", "sha256": "ACTUAL_FILE_SHA256"}
  ]
}
```

## Snapshot from the surviving training pod

Run on `rekbqruqox2bk9`, where the protected AP network volume remains mounted.
The original result root is derived from the pinned original leases; it is not
inferred from the current repository or selected task list.

```bash
python scripts/coding_confirmation_quiesce_original.py snapshot \
  --repo "$REPO" --plan "$CONTROL_PLAN" --plan-sha256 "$CONTROL_PLAN_SHA" \
  --absence-index "$ABSENCE_INDEX" --absence-index-sha256 "$ABSENCE_INDEX_SHA" \
  --output "$NEW_SNAPSHOT_DIRECTORY"
```

The helper verifies both original release receipts, all files and backup hashes
in their release manifests, and each allocation's own stop proof. It then copies:

- Every exact primary task file, job file, and claim owner record, excluding lock
  files. Their inventory becomes `quiescence.json.files`.
- Original worker metadata, all referenced completed-job runtime/model-setup
  dependencies, and prior claim-owner history.
- Byte-identical original leases, stop/release/manifest evidence, and the pinned
  provider-absence evidence into a new repository-relative provenance directory.

The resulting `original_primary_snapshot.tar.gz` has a complete SHA-256/size
manifest. Every archived member is reread and verified; every original file is
rehash-checked and the source inventory is re-enumerated to detect additions or
removals. No private tests, weights, checkpoint tensors, virtual environments,
Git data, or training trees are traversed or copied. Symlinks and unexpected
tensor artifacts in the selected public tree are refused.

`SNAPSHOT_COMPLETE.json` is the final success marker. Do not use an incomplete
output directory. `partition_inventory_fragment.json` supplies the exact original
lease inventory, excluded training pod, quiescence path/hash, and touched task IDs
for the sharding partition. It does not select or freeze a new task assignment.

Transfer the **entire snapshot evidence directory** at the same repository-relative
location. Transfer/extract the verified snapshot archive into the **origin shard's
result root only**; never copy old candidate state into fresh non-origin shards.
Do not restore lock files. The archive's `SNAPSHOT_MANIFEST.json` is audit metadata,
not a task file. Preserve the referenced `workers/` records as well as task files;
copying only `quiescence.json.files` loses completed-job dependencies. The reviewed
sharding preflight must verify the exact initial dataset before starting workers.

No existing scientific artifact or prior archive is overwritten by this helper.
