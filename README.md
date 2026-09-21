# Gearshift: cross-model KV-cache handoff

Gearshift investigates whether a frozen larger language model can perform reasoning
and transfer its attention key/value state to a frozen smaller model through a
learned mapper, rather than having the receiver replay the entire reasoning text.
The repository preserves the experiments, failed approaches, numerical controls,
numerical outcomes, scorer corrections, and analysis code. **The current implementation
has not demonstrated a useful end-to-end inference optimization.**

A research project from [Impossible Computing](https://github.com/ImpossibleComputing).
The research-note URL will be supplied by the separate website publication workflow.

This is the first Impossible Computing OSS release. Cite the immutable
[`gearshift-progress-02`](https://github.com/ImpossibleComputing/Gearshift/tree/gearshift-progress-02)
snapshot. Earlier experimental history is preserved privately. See [provenance](PROVENANCE.md).

## Architecture and terminology

```text
original problem -> frozen source model -> saved reasoning tokens + source KV cache
                                                   |
                                            trained cache mapper
                                                   |
                                      frozen receiver -> final answer
```

The mapper learns a transformation between source and receiver cache representations,
including their layer/head layouts. The language-model weights remain frozen.
Training and validation use recorded histories; inference must respect the exact
historical tokens, positions, rotary handling, and natural reasoning/answer boundary.
Cache reconstruction error or teacher-forced agreement is not proof of free-running
answer quality.

In the larger coding experiments:

- **Source:** Qwen3-32B, revision `9216db5781bf21249d130ec9da846c4624c16137`.
- **Receiver:** Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- **Pure mapping (M):** translate the entire historical prompt and reasoning cache.
- **Hybrid (H):** retain receiver-native state for the original prompt and translate
  only the source reasoning suffix. Native prompt prefill still costs computation.
- **Native text handoff (D):** the receiver natively prefills the exact source history,
  then produces an answer. This is the faithful replay reference, not a mapper.
- **Independent baselines:** A is the larger model alone; B reasons independently
  with the smaller model. P is the fixed non-thinking prompt-only baseline.

**FIXED versus ROTATING refers to mapper-training loss positions, not sparse decoding.**
Both coverage-v2 arms start from the same selected mapper and use the same 104 training
histories, task schedule, 32 scored positions per update, and 1,024 updates. FIXED
repeatedly uses the original windows anchored at answer offsets 0, 32, 128, and 512,
with declared handling of unavailable positions. ROTATING retains the early bucket
and cycles seeded windows across the later answer; short answers have an explicit
all-window rule. Full prefixes/intervening tokens are still processed. The exact
selection rules are in the [frozen declaration](configs/coding_pilot_v1/coverage_generalization_v2/declaration.json).

## Evidence map

| Evidence | Public record |
|---|---|
| Primary confirmation: 200 tasks × 3 draws × 8 conditions | [Report](results/coding_pilot_v1/confirmation_01_20260919T094418Z/primary/report/REPORT.md), [per-draw records](results/coding_pilot_v1/confirmation_01_20260919T094418Z/primary/report/per_task_seed.csv) |
| Separate training-seed replication on the same tasks | [Report](results/coding_pilot_v1/confirmation_01_20260919T094418Z/secondary/report/CONFIRMATION_SECONDARY_RESULTS.md), [per-draw records](results/coding_pilot_v1/confirmation_01_20260919T094418Z/secondary/report/per_draw.csv) |
| Coverage-v2, reused validation population | [Report](COVERAGE_GENERALIZATION_V2_RESULTS.md) |
| Sparse-state / repair, exploratory screen | [Report](SPARSE_REPAIR_RESULTS.md), [compact input](results/sparse_repair_01/report/input_records.json) |
| Records-only economic reference | [Report](ECONOMIC_CEILING.md), [compact input](results/sparse_repair_01/economics/input_records.json) |
| Release checks and evidence boundaries | [Verification](PUBLIC_RELEASE_VERIFICATION.md), [evidence scope](EVIDENCE.md) |

## Primary confirmation and separate training replication

Mean success over three fixed answer draws per task; each condition has 600 draws
clustered within 200 tasks. These are not best-of-three scores. Results use the repaired
scorer; passing tests is the measured endpoint, not proof of correctness on every input.

| Condition | Primary | Second training seed |
|---|---:|---:|
| FIXED pure mapping | 114/600 — 19.0% | 95/600 — 15.8% |
| ROTATING pure mapping | 175/600 — 29.2% | 197/600 — 32.8% |
| FIXED hybrid | 191/600 — 31.8% | 196/600 — 32.7% |
| ROTATING hybrid | 229/600 — 38.2% | 250/600 — 41.7% |
| Smaller model reasoning independently | 330/600 — 55.0% | Not rerun |
| Larger model alone | 386/600 — 64.3% | Not rerun |
| Native text handoff | 390/600 — 65.0% | Not rerun |

Broader training-position coverage improved mapping, but the tested mapped systems
remained inferior to the practical baselines. Native references reused in the secondary
analysis are not new replication measurements. The reports provide paired task-cluster
intervals and all declared draws; upstream model exposure to benchmark tasks is unknown.

## Sparse-state / selective-repair follow-up

This **oracle-assisted exploratory screen** uses 12 previously inspected development
task clusters × 3 draws × 14 conditions = 504 scored answers. It is not a fresh
confirmation or evidence of population equivalence.

| Condition | Passes / 36 |
|---|---:|
| Dense native replay | 33 |
| Mapped hybrid | 21 |
| Native sparse 5% / 10% / 25% | 33 / 33 / 33 |
| Mapped sparse 5% / 10% / 25% | 21 / 21 / 23 |
| Native repair 5% / 10% / 25% | 33 / 33 / 33 |
| Random native repair 10% | 33 |
| Recent-position native repair 10% | 29 |

Sparse native access and small instantaneous native repair recovered the observed
hybrid gap. **Attention targeting did not outperform random 10% repair.** The selector
uses full native receiver prefill and dense native-key scans; it is not a deployable
cheap repair mechanism or a measured sparse-kernel speedup.

The nominal 10% per-step repair touched **77.7%** of historical
(layer, KV-head, reasoning-position) pairs over a complete answer on average; 5%
touched **65.1%**. These are cumulative pair unions, not global-token fractions or
measured memory traffic. Small instantaneous access did not imply a small complete-answer
working set. Same-path numerical controls passed; one independently prefilling,
altered-execution-shape comparison failed its predeclared tolerance and remains
explicitly disclosed. Full repair is compared with its matched native/native splice.

## Economic conclusion: narrow, not universal

On the same primary records, native text handoff's mean instrumented stage sum is
**726.96 seconds per answer**: approximately 696.64 seconds of source reasoning,
1.47 seconds of receiver prefill, 28.83 seconds of receiver answer generation, plus
cache cloning. A **perfect, free, faithful converter** that removes only receiver
prefill reduces this to **725.49 seconds**, saving **1.468 seconds (0.202%)**.

Naive reason-then-handoff, where the principal saving is avoiding receiver text
replay, therefore has essentially no upside on this tested coding workload.
Conversion/transfer costs would further reduce that narrow opportunity.

This is hypothetical arithmetic over recorded stage timings, **not measured serving
latency, a billing saving, or a bound on heterogeneous inference generally**. It
retains the full source-reasoning charge for each answer; it does not amortize that
charge across the three experimental draws. Earlier handoff, cheaper source reasoning,
cheaper receiver decoding, or workloads with genuinely expensive receiver prefill
would be different interventions. No such new experiment is claimed here.

## Reproduce the public analyses

Python 3.12 and NumPy are sufficient for the compact analyses; no GPU, model
weights, benchmark download, candidate execution, or provider credentials are needed.
From the repository root:

```sh
python3 -m venv .analysis-venv
.analysis-venv/bin/python -m pip install numpy==2.5.3
.analysis-venv/bin/python scripts/reproduce_public.py --output reproduced
```

Use a new, empty output directory. The command verifies retained evidence hashes,
recomputes primary/secondary paired task-cluster statistics, and regenerates sparse
and economic reports. It fails if the recomputed statistics differ from the frozen
results. This is **analysis reproduction**, not a new model run or independent
regrading. Historical loaders and operational scripts retain private-archive paths
and guards; they are not a public one-command experiment launcher.

### Upstream benchmark retrieval

Gearshift does **not redistribute LiveCodeBench problem statements, tests, or
reversible prompt-token sequences**. Task IDs, pinned versions, hashes, numeric
outcomes/timings and analysis code remain public. All raw coding outputs are omitted
conservatively because generated text can repeat a problem statement.

```sh
# Six official files (~4.5 GB), pinned file/row hashes and all 400 selected IDs checked.
python3 scripts/fetch_livecodebench.py
# Smaller integrity/retrieval smoke check (134 MB):
python3 scripts/fetch_livecodebench.py --file test6.jsonl
```

Downloads go into ignored `data/coding_pilot_v1/raw/`. Do not commit or upload them.
The source is `livecodebench/code_generation_lite`, revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, release `release_v6`.
The fetcher does not execute a dataset loader, unpickle tests, run candidates, or
start experiments. Dataset terms are separate from this repository's license.
Restoring historical model-generated states/mapper weights requires separately
retained artifacts; fetching the benchmark alone does not recreate those states.

### Tests

Install the historical software dependencies in `requirements.txt`, then run:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -ra -p no:cacheprovider
```

The full test collection is retained. Nine explicitly named optional integration
cases need omitted historical inputs or upstream tokenizer assets; fourteen Linux-only
sandbox cases skip on macOS. These are disclosed skips, not passes. See the release
verification for the actual run and limitations. Tests use synthetic candidates and
small synthetic tensors, not new benchmark experiments.

## Environment and limitations

Early work used Apple Silicon/PyTorch MPS. Larger coding and sparse measurements
used NVIDIA H200s in BF16 on [RunPod](https://www.runpod.io/); orchestration/analysis
later moved to a Mac Studio. We acknowledge RunPod for research GPU infrastructure.
Model revisions, runtime details and precision controls are recorded in the
[confirmation protocol](configs/coding_pilot_v1/confirmation_01/protocol.json).
Historical local and CUDA dependency records describe different environments.

Development/validation reuse, limited training replication and twelve sparse task
clusters restrict generalization. Three draws per task are correlated. Native cache
entries can encode earlier-token dependencies; selecting entries does not prove they
can be constructed cheaply. Sparse results are oracle-assisted, not a deployed
speedup. The sparse report also discloses a missing late CPU cleanup/archive receipt.
Private raw inputs/outputs are not publicly inspectable; retained hashes establish
identity, not public access or independent regrading. See [evidence scope](EVIDENCE.md).

## License and citation

First-party software: [Apache-2.0](LICENSE). Copyright 2026 Impossible Computing, Inc.
[NOTICE](NOTICE), [third-party notices](THIRD_PARTY_NOTICES.md), and
[file-level upstream provenance](third_party/README.md) identify separate materials.
The bundled HumanEval+ fixture is upstream third-party Apache/MIT material, not
Impossible Computing-authored code. LiveCodeBench data and model weights are not
relicensed or redistributed here. Cite `gearshift-progress-02` and the report used;
no DOI or peer-reviewed publication is claimed.
