# Early text handoff: local development protocol

This branch is a 40-task **development screen**, not confirmation. The frozen
protocol is `configs/early_handoff_01/declaration.json`. Original release evidence
and `gearshift-progress-02` remain unchanged. No private archive is required.

## Hardware and dependencies

Primary local execution: Apple M4 Max, 128 GB unified memory, macOS 27.0,
Python 3.12.4, PyTorch 2.14.0, Transformers 4.57.6, BF16, MPS/SDPA.
Use an environment providing these pinned versions plus numpy, pytest and
matplotlib. Models are **not quantized**. This is not an H200 timing replication.
Local ARM Linux/Colima runs the original isolated scorer with unchanged numerical
limits. CPU architecture can affect timing-sensitive scoring; disclose this.

Set `PY` to that environment's Python executable, then run from the repository:

```sh
export PYTHONDONTWRITEBYTECODE=1
$PY -m pytest -q
$PY scripts/fetch_livecodebench.py
$PY scripts/early_handoff_dependencies.py --workers 4
$PY scripts/early_handoff_prepare.py
```

Retrieval verifies the pinned official dataset revision, files, selected IDs and
row hashes; model retrieval verifies exact revisions and every shard's LFS SHA256.
`data/` is ignored. Never add raw inputs, tests, generations, token arrays, weights
or provider metadata to Git or the review ZIP. HumanEval+ notices are unchanged.

Before inference, create `data/early_handoff_01/control/budget_ledger.json` with
`soft_usd: 1000`, `hard_usd: 1500`, `new_compute_usd: 0`, `reserved_usd: 0`,
`cleanup_reserve_usd: 50`, and no new remote resources. Record current provider
inventory read-only; do not stop/restart pre-existing resources. Local work has
zero new RunPod charges, **not zero energy or hardware cost**.

## Local numerical and capacity gate

```sh
$PY scripts/early_handoff_worker.py preflight --role receiver
$PY scripts/early_handoff_worker.py preflight --role source
```

These use only synthetic inputs. Each must demonstrate bitwise exact original-
shape cache replay, matching RNG continuation, and a 36,865-token native prefill.
Any backend/numerical/capacity failure stops the local path before benchmark
inference. Never silently change precision, quantization, model or decoding.

## Generation and isolated scoring

```sh
$PY scripts/early_handoff_worker.py all
colima start --cpu 4 --memory 6 --disk 20 --arch aarch64 --vm-type vz
export DOCKER_HOST="unix://$HOME/.colima/default/docker.sock"
docker build -t gearshift-early-scorer:01 - < configs/early_handoff_01/Scorer.Dockerfile
for STAGE in preflight run; do
  docker run --rm --network none \
    --mount "type=bind,src=$PWD,dst=/repo,readonly" \
    --mount "type=bind,src=$PWD/data/early_handoff_01,dst=/experiment" \
    gearshift-early-scorer:01 python3 scripts/early_handoff_score.py "$STAGE"
done
$PY scripts/early_handoff_report.py
```

One model is resident at a time. Generation is sealed before hidden-test scoring.
Three fixed final-answer draws share one reasoning history per task/condition.
Answers and histories are immutable; no selective regeneration. Recovery restores
committed token/RNG state and replays identical forward shapes, recording overhead.
A STOP file under `data/early_handoff_01/control/` interrupts generation. Keep the
Mac awake while executing; local execution cannot survive an actual shutdown,
but committed checkpoints permit audited recovery. It does not require RunPod.

## Analysis and interpretation

```sh
$PY scripts/early_handoff_report.py \
  --input results/early_handoff_01/analysis_input.json \
  --output results/early_handoff_01_regenerated
```

Fixed fractions require the eventual source trajectory length. Prefix timing is
retrospective measured-token accounting, not deployable online stopping. Use the
full receiver work, not source-token savings alone. The oracle is retrospective,
selected on the same draws, and optimistic. Bootstrap clusters are tasks, not
individual answers. No router or confirmation is part of this screen.

Before **each commit**, stage only safe files, review `git status` and run:

```sh
$PY scripts/early_handoff_hygiene.py
```

The scanner compares staged text against freshly fetched upstream text/tests,
checks raw-data fields and literal token arrays, and invokes gitleaks. It supplements,
not replaces, a whitelist-only artifact review. Do not merge main or create tags.
