# Coverage generalization v2 results

At the common 1024-update checkpoint, ROTATING minus FIXED in pure mapped-cache free-running code success is **+20.6 percentage points**, paired 95% interval **[+11.1, +31.7]**. The exploratory paired interval is above zero, supporting better held-out free-running behavior for ROTATING on these 21 previously examined validation tasks.

Full clean 1,024/1,024 training completed: **true**. Fresh scored validation answers: **1512**. Every sampled answer used one of the exact three declared seeds per task. Checkpoints were immutable and evaluated asynchronously; sampled validation outcomes never selected the primary endpoint.

## Scientific scope

Both arms cleanly restart from the same original selected update-96 mapper with identical fresh AdamW, constant LR scheduler, RNG seed and task schedule. The exact 104 training histories, 32 scored positions per update, fixed and rotating selection recipes, frozen source/receiver model revisions, BF16 arithmetic, natural handoff, source-written teacher references, extraction and isolated hidden-test scoring are retained. Full prefixes and intervening continuation tokens are processed; free-running generation receives no teacher-answer prefix.

The 21 validation tasks are held out from gradients but already informed earlier selection and analysis. They are exploratory validation, not untouched confirmation. The reserved 200-task and second-seed confirmation sets remain untouched. Bootstrap inference averages the three draws within task and jointly resamples 21 task IDs 10,000 times, using exactly the same resample indices for every condition and contrast. Intervals are unadjusted. 63 draws are not 63 independent tasks, and best-of-three is not pass@1.

## Primary checkpoint results

| Condition | Passed draws | Mean task success | 95% task interval |
|---|---:|---:|---:|
| START_M | 11/63 | 17.5% | [6.3%, 30.2%] |
| START_H | 30/63 | 47.6% | [28.6%, 66.7%] |
| FIXED_1024_M | 17/63 | 27.0% | [12.7%, 42.9%] |
| FIXED_1024_H | 38/63 | 60.3% | [41.3%, 79.4%] |
| ROTATING_1024_M | 30/63 | 47.6% | [28.6%, 66.7%] |
| ROTATING_1024_H | 44/63 | 69.8% | [50.8%, 87.3%] |
| D | 58/63 | 92.1% | [79.4%, 100.0%] |
| P | 29/63 | 46.0% | [28.6%, 63.5%] |

M translates the complete historical cache. H preserves the receiver-native original prompt cache and translates the saved reasoning suffix. D natively prefills the exact full source history. P uses the unchanged prompt-only non-thinking template. START is generated fresh in this experiment and shared between the two identical initial mapper states.

| Paired task contrast | Difference, pp | 95% interval, pp |
|---|---:|---:|
| START_H-START_M | +30.2 | [+14.3, +46.0] |
| START_H-D | -44.4 | [-65.1, -23.8] |
| START_H-P | +1.6 | [-9.5, +11.1] |
| ROTATING_1024_M-FIXED_1024_M | +20.6 | [+11.1, +31.7] |
| ROTATING_1024_H-FIXED_1024_H | +9.5 | [-1.6, +23.8] |
| FIXED_1024_M-START_M | +9.5 | [+0.0, +22.2] |
| ROTATING_1024_M-START_M | +30.2 | [+17.5, +44.4] |
| FIXED_1024_H-START_H | +12.7 | [+1.6, +27.0] |
| ROTATING_1024_H-START_H | +22.2 | [+6.3, +39.7] |
| FIXED_1024_H-FIXED_1024_M | +33.3 | [+14.3, +52.4] |
| ROTATING_1024_H-ROTATING_1024_M | +22.2 | [+3.2, +39.7] |
| FIXED_1024_H-D | -31.7 | [-52.4, -12.7] |
| FIXED_1024_H-P | +14.3 | [-1.6, +31.7] |
| ROTATING_1024_H-D | -22.2 | [-41.3, -6.3] |
| ROTATING_1024_H-P | +23.8 | [+4.8, +44.4] |

## Execution and controls

Per-task/seed raw answers, token IDs, source trajectories and score receipts are immutable and hash-bound. Incomplete worker attempts are retained. A finished sampler transaction can be recovered from its durable tokens/RNG without another draw; unavailable timing is recorded explicitly. All planned generation closes before private tests are opened. Native/native splice controls, absolute positions, isolated cloned caches and exact frozen seeds are checked. No program repairs, function renaming, later-block rescue or favorable-seed selection occur.

Training checkpoints contain mapper, optimizer, scheduler, update/schedule position, RNG and precision state with atomic manifests and verified reload. Resume checks and gradient logs are included with each arm. Durable RunPod jobs do not depend on a laptop/Studio heartbeat or a continuing SSH session; mirror connectivity is optional.

Both KL panels are separate teacher-forced fidelity measures; neither establishes program correctness. Learning-curve tables and figures retain every available declared common checkpoint. Coverage JSON/CSV reports total and unique task/position exposure, prose, signature, code-body, return and EOS categories plus answer-position distributions. Categories are descriptive and did not choose training positions.

The earlier four-case fit remains a side diagnostic: START_M 40%, fitted M 75%, native D 100% over five seeds per seen task. It was not retrained and is not evidence of unseen generalization.

## Artifact and publication boundary

Experiment identity: `coverage_generalization_v2_20260918T233720Z`. Frozen publication remains `gearshift-progress-01` at `64725974fa55459350d1c9d09037bab64d0c5ec6`. The first article and previous results are unchanged. No push or publication.

This report regenerates from compact evidence without model weights, mapper weights, private tests or generated-code execution. Resource receipts, current Git identity, conservative cumulative cost, failures and external heavy-artifact retrieval inventory accompany the review bundle.
