# Gearshift phase two

**All planned inference, training and objective scoring are complete. Subjective judging is paused by the owner after partial-result inspection.** The 640-case frozen comparison, 128-case adaptation confirmation, 64-case 4B replication, all three primary training seeds, long-output and branching comparisons, contextual-memory controls, and repeated timings have completed. Source trajectories and raw outputs are preserved. The original pilot and follow-up are unchanged. The incomplete 447-case laptop characterization is retained as a separate archive and excluded from primary results.

The evidence supports contextual state transfer under some conditions, but does **not** establish general replacement of native text replay or a broadly useful speedup. Frozen transfer loses 7 percentage points of fresh arithmetic accuracy and 48 points of code pass rate relative to native target replay. Functional adaptation improves arithmetic and code over the frozen mapper; the boundary objective performs better on some task checks despite worse validation KL. The 4B-source replication does not reproduce reliable multi-domain transfer. Long prose exposes repetition and premature or capped responses. Evidence-field and literal-writing checks are limited proxies; semantic conclusions use only the separately reported valid blind judgments.

**A final qualitative audit found a unit ambiguity in all 80 characterization evidence tasks and their derived extensions.** The hidden reference treats a pricing unit as one thousand requests, while the prompt never defines that unit. Under the alternative per-request interpretation, native replay matches all three fields on 59/80 cases rather than 0/80; M remains 0/80. These are separately labeled sensitivity results, with original scores preserved. Read [EVIDENCE_AUDIT.md](EVIDENCE_AUDIT.md) before interpreting evidence characterization or extension judgments. Training, validation, development, memory and confirmation use different templates and are unaffected by this defect.

The pause was not prespecified: task-validity and scoring-resolution concerns require review, and priority is shifting toward coding. Subjective evaluation remains incomplete and its observed subset is not representative of the full matrix. See PAUSE_STATUS.md.

Current judging coverage, exclusions, order disagreements and acceptability are reported in [JUDGING_RESULTS.md](JUDGING_RESULTS.md). Human labels are pending in the exported 30-pair packet and are not required for this research delivery. No outcome-dependent expansion of the experiment was made.

Compute was controlled from the Mac Studio. The owner authorized a USD1,000 Runpod ceiling and later removed the initial 24-hour cutoff while retaining that ceiling. All five task pods are now deleted and confirmed absent; allocated GPU cost is zero. Provider billing observed after cleanup totals **USD71.38**, with a deliberately conservative ledger upper bound of **USD95.64**. Provider records can lag final billing. One free judging-service reset was separately approved and used; no credits were purchased and no paid judging fallback was used.

## Questions and experimental boundaries

This phase separates three questions: whether mapped state preserves useful history relative to native target replay; which training and handoff choices cause the behavior; and whether switching improves useful output quality at an attractive total latency. A favorable answer to one does not establish the others.

All new records use `results/phase2_v1/`. Original pilot and `followup_v1` results, checkpoints, manifests, and narratives are protected by `evidence/phase2/historical_manifest.json`. Exact historical reporting source is retained so the earlier compact evidence can be checked without rewriting it. No model weights or cache tensors are required for records-only reporting; an absent checkpoint is explicitly unavailable for verification.

The primary source/receiver pair is Qwen3-1.7B to Qwen3-0.6B. Diagnostic/development runs and the archived partial characterization used an Apple M3 Max with 128 GB memory; completed long outputs and branching use the Studio's Apple M4 Max with 128 GB memory. Both use MPS, float16 and SDPA. CUDA characterization, adaptation and replication completed on an NVIDIA H100 80GB HBM3 after all 26 model/tokenizer hashes, 53 unit tests, strict cache/reinjection/position/hybrid controls, and real boundary-gradient/emitted-EOS gates passed. The deterministic math-SDPA wrapper is bound into the separate CUDA config; the earlier failed 4096-token reinjection attempt remains archived. Separate terminal output and score receipts establish completion; infrastructure passes alone establish neither quality nor completion. Timing is attributed to each stage's hardware and runtime, and the laptop archive is excluded from the primary characterization comparison. Exact pinned revisions, installed software, cache geometry, tokenizer controls, prompts, token IDs, source snapshots and checkpoint hashes are in the configs and stage manifests. Generation uses shared source histories and task-appropriate matched handoff contracts. Hidden answers and unit tests are hashed separately and excluded from inference and mapper training.

Methods are: small direct (S), small with thinking (S_think), source continuation with matched new turn (B/newturn), ordinary source-native completion (B/native), full target text replay (C), pure mapped cache (M), native prompt plus mapped reasoning (H), original prompt plus the last 256 reasoning tokens (T), and an original prompt plus a source-written plan capped at 256 tokens (P). Summary production and reading are paid in P; historical prefill is paid in C, H, T and P where used. Only M has zero historical target prefill, and its bridge still costs tokens and time.

The frozen task reservation contains 400 fresh arithmetic cases, 100 HumanEval+ functions, 80 evidence packets and 60 writing briefs. Development uses 16 separate cases per family. Multi-domain adaptation reserves 128 training and 32 validation conversations. Confirmation evaluates a separately reserved, balanced 128-case subset; other reserved cases remain unused. Extra training seeds share a fixed 32-case confirmation subset, and the 4B replication uses 64 prespecified cases. Those subsets must be compared on their common cases, not by comparing unequal cohort averages.

Evidence and writing use deterministic fictional tasks. Each split uses distinct procedural templates, with one evidence template and two writing genres per split. This avoids reusing near-duplicate packets across train/test, but it is a limited custom suite, not broad coverage of real-world synthesis or writing.

## Completed checkpoint and handoff diagnosis

The old 100 arithmetic histories were reconstructed by full prefill of their exact saved token IDs. These are previously inspected diagnostic questions, not fresh confirmation. All 1,200 requested condition outputs were completed. Replay totals deliberately exclude original source generation; they cannot establish a new source-inclusive speedup.

| Condition | Extracted correct / 100 | Correct and numeric-format valid / 100 | EOS / 100 | Mean answer tokens |
|---|---:|---:|---:|---:|
| Initial affine, old delimiter | 0 | 0 | 0 | 64.00 |
| Initial affine, new turn | 38 | 27 | 100 | 4.48 |
| Plaintext-functional, old delimiter | 71 | 0 | 3 | 62.66 |
| Plaintext-functional, new turn | 71 | 70 | 100 | 3.37 |
| Chat-functional, old delimiter | 49 | 0 | 0 | 64.00 |
| Chat-functional, new turn | 78 | 78 | 100 | 3.35 |
| Text replay, old delimiter | 79 | 33 | 88 | 13.10 |
| Text replay, new turn | 76 | 38 | 90 | 14.18 |
| Source, old delimiter | 81 | 38 | 96 | 8.05 |
| Source, new turn | 82 | 39 | 100 | 5.46 |
| Small direct | 14 | 0 | 24 | 56.53 |
| Small with thinking | 71 | 60 | 98 | 5.91 |

Paired task bootstrap intervals use 5,000 resamples. Holding the new-turn protocol fixed, chat-functional exceeds initial affine by 40 percentage points, 95% interval [30, 50], with 41 gains and one loss. Chat-functional exceeds plaintext-functional by 7 points [1, 13], with nine gains and two losses. Chat-functional minus native text replay is 2 points [−5, 9], and minus small-thinking is 7 points [−2, 17]. These latter intervals do not establish equivalence or a reliable advantage.

Changing only the scaffold increases initial-affine correctness by 38 points [29, 48], changes plaintext correctness by 0 points [−8, 8], and increases chat-functional correctness by 29 points [20, 37]. Plaintext's large formatting and termination improvement occurs without an aggregate accuracy increase. Thus the numeric result depends on both protocol and trained mapper behavior; the scaffold is not the whole explanation, and formatting gains are not accuracy gains.

The reconstructed chat/new-turn and both text-replay conditions reproduce historical answer token IDs on all 100 cases. Chat/old agrees on 89/100, although its aggregate extracted correctness remains 49/100. Full-prefill reconstruction and the original incremental live cache are therefore not interchangeable at the byte-output level. Both records remain available; no historical output was replaced.

## Completed contextual-memory diagnosis

Twelve parent packets each have two versions differing in one decision-changing certification fact. All use the same generic three-field handoff. Wrong-history scoring assigns the counterpart packet's already generated response to the other packet, equivalent to deterministic decoding from that wrong state under the identical request. It does not represent a separately timed economic alternative. There are 24 variants but only 12 independent parent clusters.

| Method/history | Decision correct / 24 | All three fields correct / 24 |
|---|---:|---:|
| Source / correct | 23 | 16 |
| Source / wrong | 1 | 1 |
| Native target replay / correct | 21 | 17 |
| Native target replay / wrong | 2 | 1 |
| Mapped / correct | 23 | 16 |
| Mapped / wrong | 1 | 0 |
| Hybrid / correct | 13 | 9 |
| Hybrid / wrong | 7 | 1 |
| Absent history | 0 | 0 |

Mapped correct-history minus wrong-history all-field accuracy is 66.7 points, parent-cluster 95% interval [45.8, 87.5]. Mapped outputs meet the three-field format in all 24 cases, whereas native target replay meets it in none because of repeated fields. Consistent duplicates are counted separately from field correctness; conflicting values remain errors. The original stricter scoring and the documented pre-final correction are both retained.

This supports use of packet-specific information in the transferred whole history. It does not isolate information acquired during reasoning from information already present in the original prompt. The hybrid wrong-history control also substitutes the whole counterpart history, including its native prompt; it is not a clean correct-prompt/wrong-reasoning isolation.

## Development decisions fixed before final generation

All 64 development tasks and nine methods completed, totaling 576 outputs. These outcomes informed protocol validation and resource choices only. Final cases were neither selected nor removed according to generated quality.

Source reasoning reached its 1,024-token development cap on 11/16 code and 14/16 evidence tasks. A single declared change raised code/evidence reasoning caps to 2,048 for final runs, including the small-thinking baseline, training/validation, extensions and confirmation. Arithmetic remains at 2,048 and writing at 768. Answer contracts, graders and sample IDs were unchanged. `final_protocol_lock.json` and `tasks/final_task_contracts.json` preserve that decision before final generation.

The source-native baseline matters: it passed 10/16 development code tasks, versus 2/16 for the source with the new-turn scaffold. Native target replay passed 9/16, mapped replay 2/16. The parser uses a fixed first Python/unlabeled code fence if present, otherwise the full raw output, with no selective syntax repair. This retains protocol-induced failures while keeping a competent ordinary source baseline in the final comparison.

For development evidence, native target replay satisfied all three field checks on 9/16 tasks and mapped replay on 0/16. For development writing, the corresponding literal constraint checks were 3/16 and 0/16. These narrow checks are not full semantic quality judgments. Mapped prose frequently repeated itself; average repeated four-gram fractions were 0.348 for evidence and 0.591 for writing. Eight technical explanation sources closed thinking after only three tokens. Source history length and reasoning/answer overlap are recorded rather than assuming every case contains substantial reasoning.

The generated-code sandbox was verified before final scoring: outside-file reads/writes, network and subprocess attempts were denied, resource limits were exercised, and all 16 canonical development functions passed their hidden tests. Early successful process exit before tests is rejected by a random completion marker.

## Long-output behavior

All seven frozen methods completed all three answer caps (128/512/1,024 tokens) for 24 parents: 12 evidence packets and 12 writing briefs. This is 72 task variants and 504 outputs. Terminal hashes, membership, condition sets, paired histories, token accounting, objective score inputs and blind packet hashes passed verification. Repeated length variants remain clustered by their original parent; they are not independent samples.

These evidence parents inherit the characterization pricing-unit ambiguity; numerical and semantic claims require EVIDENCE_AUDIT.md. The following descriptive measurements compare mapped cache (M) with native target replay (C). Repeated four-gram fraction is a repetition diagnostic, not a semantic quality score. Mean output lengths include the stored generated token accounting; shorter outputs are not automatically economic wins.

| Family | Answer cap | Mean tokens M / C | Cap stops M / C (of 12) | Mean repeated four-gram fraction M / C |
|---|---:|---:|---:|---:|
| evidence | 128 | 128.0 / 125.3 | 12 / 9 | 0.154 / 0.037 |
| evidence | 512 | 421.2 / 333.4 | 5 / 1 | 0.485 / 0.092 |
| evidence | 1024 | 737.9 / 692.2 | 5 / 1 | 0.575 / 0.127 |
| writing | 128 | 105.8 / 44.6 | 5 / 0 | 0.093 / 0.000 |
| writing | 512 | 356.7 / 115.3 | 7 / 0 | 0.419 / 0.007 |
| writing | 1024 | 957.5 / 407.2 | 11 / 0 | 0.768 / 0.046 |

Mapped outputs show substantially more repetition at the larger caps in these observed cases. At the 1,024-token writing cap, M reaches the cap on 11/12 parents versus 0/12 for C. All-field and literal-constraint checks are available in the tables, but neither those checks nor these repetition measurements establish full artifact quality. Blind rubric judgments and acceptability are reported separately with their actual coverage; a repetition statistic alone does not establish a quality-matched utility advantage. The frozen adaptation recipe and confirmation sample are unchanged by these observations.

Full seven-method objective summaries, paired intervals and plots are in [the regenerated report](results/phase2_v1/report/TABLES.md). Extended latency diagnostics were regenerated in `results/phase2_v1/report/`; timings belong to their explicitly named stage and hardware.

## Completed branching and teacher-forced fidelity

Branching completed three requests for each of 16 parents under all seven frozen methods: 48 variants and 336 outputs. Evidence parents inherit the characterization pricing-unit ambiguity. All objective scores and 576 anonymous orientation packets (288 pairs) are present. Parent payload hashes, canonical transactions and fan-out accounting were verified: each method pays its source, summary and prefix preparation once per parent, then each requested output. These 48 variants retain 16 parent sampling units. Semantic acceptability coverage is reported separately. Branching quality/latency plots use source and prefix preparation paid once across all three requested outputs for every method. Repeated MPS timing also completed after scoring and packet export, as detailed below.

The long-output stage also completed 48 teacher-forced runs, comparing M and H with native target replay for each of 24 parents at the 1,024-token answer cap. The first up to 512 tokens of the native continuation are the reference. All 192 available windows were checked against their finite per-token records and native answer lengths. Window means receive equal parent weight; later windows use only observed tokens and contain no padding.

| Family | Condition | KL, positions 1–32 (95% parent interval) | KL, positions 257–512 (95% parent interval) |
|---|---|---|---|
| Evidence | M | 1.635 [1.492, 1.774] | 0.650 [0.518, 0.752] |
| Evidence | H | 0.099 [0.074, 0.127] | 0.095 [0.071, 0.115] |
| Writing | M | 0.912 [0.783, 1.039] | 0.782 [0.733, 0.824] |
| Writing | H | 0.053 [0.031, 0.075] | 0.043 [0.024, 0.063] |

Each row covers 12 parents. Native-to-mapped KL is lower for H than M in these measurements; H includes paid native prompt prefill. M's lower later-window divergence conditions on the native answer prefix and does not demonstrate free-running recovery or task correctness. Token positions are dependent observations, and native replay is a fidelity reference rather than ground truth. Full KL, NLL, top-1 agreement and observation counts are in `results/phase2_v1/report/teacher_forced_summary.csv`; the verification receipt is `evidence/phase2/studio_transfer/extensions_completion_verified_turn07.json`.

## Completed repeated MPS timing

All eight predeclared cases completed on the Apple M4 Max Studio: two per family, four methods (B/newturn, C, M and T), one warmup and three measured repeats. This produced 128 outputs, of which 96 are measured repeats. Frozen membership, completion hashes, seeded randomized condition order, shared source histories and finite source-inclusive time accounting passed validation. All 32 case/method groups emitted identical token sequences across all four repetitions. The repetitions measure timing variability; they do not create additional independent quality samples.

The table shows each case's median handoff latency over the three measured repetitions, alongside its output length. Handoff latency excludes source preparation/reasoning; full source-inclusive totals and per-component medians, quartiles and extrema are in `results/phase2_v1/report/repeated_timings.csv`. Each case has one live source trajectory shared across methods and repetitions, so source generation was not itself repeated. CUDA characterization and the archived laptop run are separate hardware measurements.

| Case | Handoff seconds C / M | Answer tokens C / M |
|---|---:|---:|
| gsm_221 | 0.133 / 0.081 | 3 / 3 |
| gsm_388 | 0.118 / 0.077 | 3 / 3 |
| HumanEval_120 | 0.888 / 1.409 | 42 / 93 |
| HumanEval_83 | 1.352 / 2.024 | 75 / 134 |
| evidence_043 | 6.367 / 4.253 | 446 / 305 |
| evidence_076 | 3.470 / 8.022 | 232 / 580 |
| writing_012 | 5.449 / 2.719 | 419 / 210 |
| writing_045 | 0.246 / 3.417 | 8 / 256 |

The existing objective scorer checked all 128 outputs against the unchanged pre-generation hidden-grader commitments, including execution of code in the Studio sandbox. Counting each case once, all four methods answered both arithmetic cases correctly. C passed both code cases; M, B/newturn and T passed neither. All four methods failed the strict all-field check on both evidence cases. For writing, B/newturn satisfied the literal constraints on both cases, C on one, and M/T on neither. These are descriptive checks of a tiny timing subset; evidence/writing checks do not establish semantic acceptability. Differences in emitted content and length prevent interpreting these raw latency differences as quality-matched economic gains.

Timing validation is recorded in `evidence/phase2/studio_transfer/timing_completion_verified_turn08.json`; the descriptive scoring scope is recorded separately in `timing_objective_scoring_scope.json`. No inference, task selection, generation setting or adaptation recipe changed.

## Frozen transfer on 640 fresh tasks

All 5,008 planned condition outputs were generated and objectively scored. The table uses the full family sample for seven primary methods. Arithmetic also includes T on all 400 cases; T and P share a predeclared 32-case subset within each prose family. Their unequal full-set means must not be compared to 80/60-case means; matched comparisons and latency plots use the shared cases.

| Method | Arithmetic correct / 400 | Code hidden tests passed / 100 | Evidence all fields correct / 80 | Writing literal checks passed / 60 |
|---|---:|---:|---:|---:|
| Small direct | 50 | 21 | 0 | 17 |
| Small with thinking | 289 | 35 | 13 | 6 |
| Source / new turn | 351 | 18 | 0 | 34 |
| Source / native | 296 | 76 | 0 | 30 |
| Native target replay C | 344 | 57 | 0 | 20 |
| Frozen mapped M | 316 | 9 | 0 | 0 |
| Hybrid H | 344 | 18 | 0 | 6 |

On arithmetic, M minus C is **−7.0 percentage points**, paired 95% bootstrap interval **[−10.3, −3.8]**, with 9 gains, 37 losses and 354 ties. M minus small-thinking is +6.8 points [1.8, 11.8], but M minus the matched source/new-turn baseline is −8.8 points [−11.8, −6.0]. M achieves correct numeric formatting on 314/400 cases versus C's 320/400; their correctness-and-format contrast is −1.5 points [−5.5, 2.8]. Formatting and extracted accuracy answer different questions. The earlier 100-case interval crossing zero was not evidence of equivalence, and this fresh larger sample finds an accuracy deficit.

On code, M minus C is **−48 points [−58, −38]**, with zero gains and 48 losses. M also trails direct small-model generation (21 passes) and small-thinking (35). The native source passes 76 tasks; the new-turn source passes 18. This large source-protocol effect survives development into the full frozen comparison, which is why both source baselines are retained. Neither the weak new-turn source nor a weak mapper should be used as the sole baseline for an economic claim. Code statuses distinguish parse errors, sandbox/test failures, timeouts and passes in the raw score records.

The frozen evidence all-field check is zero for most methods, including the source and native target replay. Final qualitative review identified the undefined pricing unit described in EVIDENCE_AUDIT.md. Using the per-request interpretation on the same recorded answers changes all-field matches to 58/80 for native source, 49/80 for new-turn source and 59/80 for C; M remains 0/80 and has no correct parsed decision fields. This is a post-hoc sensitivity analysis, not a replacement of primary scores. The original zero scores must not be presented as evidence that these native baselines cannot perform the arithmetic or evidence task. Source-marker coverage differs (C 69.7%, M 6.9%), but citation presence does not prove claim support. M has mean repeated four-gram fractions of 0.418 on evidence and 0.406 on writing, versus C's 0.117 and 0.024. M reaches its answer cap on 37/80 evidence and 26/60 writing cases. The writing literal-check deficit relative to C is −33.3 points [−45.0, −21.7]. Blind judgments assess substantive fulfillment and consistency separately.

Source reasoning is itself limited: it hits the cap on 45/400 arithmetic, 43/100 code, 17/80 evidence and 5/60 writing cases. Half of writing histories contain at most four reasoning tokens. Raw source reasoning and overlap measurements remain in the bundle. These are tests of using the actual emitted source history, which can contain the answer, errors, or very little reasoning; they do not establish independent acquisition of the source model's reasoning ability.

## Controlled functional adaptation

Both language models remained frozen. Each affine objective used 128 disjoint training conversations, 32 validation conversations, the same initialization, matched domain/context schedules and matched counts of gradient-bearing target predictions. Ordinary-continuation and actual-handoff-boundary objectives were run for all three declared seeds. The joint validation score equally weights task families and evaluates both protocols. The selected objective is **ordinary**, by mean best validation KL across seeds (0.234875 versus boundary 0.760111); the first seed, 20260915, is the fixed representative. Confirmation outcomes did not change that selection.

| Seed | Updates per arm | Stop | Best ordinary step / KL | Best boundary step / KL | Gradient-bearing predictions per arm |
|---|---:|---|---|---|---:|
| 20260915 | 1,024 | Update cap | 896 / 0.22552 | 704 / 0.71096 | 6,938 |
| 20260916 | 1,024 | Update cap | 768 / 0.23671 | 960 / 0.69302 | 6,906 |
| 20260917 | 704 | Both-arm validation plateau | 448 / 0.24240 | 384 / 0.87635 | 4,739 |

The first two seeds reached the declared cap. These runs do not establish convergence, even though some best checkpoints precede the final update. Full training curves, family/protocol validation measurements, prediction exposure and checkpoint hashes are included. The budget was not extended after opening confirmation outcomes.

### Disjoint 128-case confirmation

All 1,280 outputs completed: nine methods on 128 cases plus four extra-seed mapper conditions on the predeclared 32-case common subset. Every family has 32 full-set cases. Evidence and writing use split-specific procedural templates, so cross-split differences are not paired estimates of adaptation.

| Method | Arithmetic / 32 | Code / 32 | Evidence all fields / 32 | Writing literal checks / 32 |
|---|---:|---:|---:|---:|
| Small direct | 3 | 12 | 4 | 12 |
| Small with thinking | 26 | 10 | 4 | 6 |
| Source / new turn | 29 | 6 | 11 | 11 |
| Source / native | 22 | 30 | 12 | 15 |
| Native target replay C | 26 | 22 | 5 | 5 |
| Frozen mapper | 25 | 2 | 0 | 0 |
| Ordinary mapper, first seed (selected) | 29 | 12 | 2 | 0 |
| Boundary mapper, first seed | 29 | 17 | 12 | 0 |
| Hybrid using selected mapper | 29 | 14 | 2 | 1 |

Both trained arms improve arithmetic over the frozen mapper by 12.5 points [3.1, 25.0], with four gains and no losses. Ordinary improves code over frozen by 31.3 points [12.5, 50.0], and boundary by 46.9 points [28.1, 65.6]. Ordinary still trails C on code by 31.3 points [9.4, 53.1]; boundary minus C is −15.6 points [−34.4, 3.1], which does not establish equivalence.

Direct boundary-minus-ordinary contrasts, with training seed and case membership fixed, are 0 points [0, 0] on arithmetic, +15.6 [−3.1, 34.4] on code (7 gains, 2 losses), +31.3 [15.6, 46.9] on evidence field checks (10 gains, no losses), and 0 [0, 0] on writing literal checks. Boundary minus C on evidence fields is +21.9 [3.1, 40.6], but this narrow check is not a semantic acceptability estimate. Both mapper arms fail every writing all-literal-constraints check. Lower validation KL did not reliably rank final task performance, and we retain the predetermined ordinary selection instead of relabeling boundary as the selected recipe.

The three seeds were all evaluated on the same eight cases per family. Ordinary code passes are 4/8, 1/8 and 1/8; boundary passes are 3/8, 4/8 and 4/8. Boundary evidence field passes are 3/8, 0/8 and 0/8; ordinary has 0/8 for every seed. All six mapper conditions pass 7/8 arithmetic cases and 0/8 writing literal checks. These paired small cohorts reveal material seed sensitivity in code and evidence. Seeds and repeated task IDs are not independent sample-size multipliers, and no best-seed selection is made.

Source reasoning in confirmation reaches the cap on 26/32 evidence and 16/32 code cases. This is a limitation of the fixed source allowance. We retain those cases and do not tune caps after seeing their outcomes.

## Source-size replication: 4B to 0.6B

The preselected ordinary objective was replicated with a fresh 4B source, 128 training and 32 validation conversations, and a source-specific normalized-depth affine fit. The initialization used 64 positions per training case, standardized ridge fitting at 0.01, inverse-RoPE processing and the recorded 4B cache geometry. It did not reuse a reconstruction-only historical 4B checkpoint.

One declared seed completed 1,024 updates, with best step 896 and validation KL 0.81854 over 7,266 gradient-bearing predictions. This was an update-cap stop, with convergence untested. All 512 outputs on the 64-case reserved subset completed. Each family has 16 cases.

| Method | Arithmetic / 16 | Code / 16 | Evidence all fields / 16 | Writing literal checks / 16 |
|---|---:|---:|---:|---:|
| Source / new turn | 14 | 15 | 2 | 3 |
| Source / native | 13 | 15 | 13 | 4 |
| Native target replay C | 15 | 9 | 12 | 0 |
| Initial mapper | 0 | 0 | 0 | 0 |
| Trained ordinary mapper | 9 | 0 | 1 | 0 |
| Hybrid using trained mapper | 11 | 4 | 3 | 4 |
| Small direct | 0 | 5 | 1 | 6 |
| Small with thinking | 12 | 5 | 0 | 2 |

Training improves arithmetic over initialization by 56.3 points [31.3, 81.3], but trained M minus C is −37.5 [−62.5, −12.5]. Its code deficit is −56.3 points [−81.3, −31.3] and evidence-field deficit is −68.8 [−87.7, −43.8]. Native replay remains capable on those code/evidence cases, so their mapper failures cannot be attributed solely to receiver incapacity. On writing both M and C fail all literal checks, limiting that comparison. Extremely short failed mapped outputs must not be counted as useful speedups. This one-seed replication does not support reliable scaling of the selected recipe to a larger source. The optional 4B-to-1.7B receiver experiment was not run.

## Latency and useful output

The CUDA arithmetic comparison illustrates why receiver-stage ratios can mislead. M averages 0.133 seconds after handoff versus C's 0.204 seconds, a reduction of about 35%. Source-inclusive means are 20.144 and 20.216 seconds: about **0.35%** difference, accompanied by a 7-point accuracy loss. M and the matched source/new-turn baseline have essentially the same total time, while the source is more accurate. The tail baseline scores 85% on arithmetic, versus M's 79%; their paired difference is −6.0 points [−9.3, −2.5].

On frozen code, M averages 38.033 seconds total versus C's 38.422 seconds, while passing 9% versus 57% of tasks. On frozen evidence and writing, M is slower than C in the full-set means and fails more narrow checks. Adapted first-seed boundary code generation averages 41.736 seconds versus C's 43.029 seconds, but its pass-rate uncertainty includes a material deficit. These measurements do not establish equal-quality replacement. Timing plots always keep hardware, task family, comparison population and source-inclusive cost explicit. Pure M pays zero historical receiver prefill, but pays mapping, cloning/splicing, bridge and generation costs.

Branching charges each method its reusable source/prefix work once, then three distinct requested outputs. The per-output amortized costs are in `fanout_costs.csv` and branching quality/latency plots. Compact-plan generation and reading are included. Acceptability-based plots include only valid complete A/B and B/A pairs, retain parent clustering, and show their actual coverage. They cannot establish a broad Pareto advantage from an incomplete or resource-limited judging sample. MPS repeated timing and synthetic teacher-forced fidelity provide separate implementation diagnostics.

## Interpretation, uncertainty and review limits

**Transfer:** mapped state can carry task-specific contextual information and support short arithmetic rendering. The fresh frozen sample finds a measurable arithmetic loss and a large code loss relative to native replay. Whole-history memory controls support context dependence, not isolation of reasoning-only knowledge. Native-replay fidelity and ground-truth task correctness remain distinct.

**Behavior:** both the conversational scaffold and functional mapper training matter. Functional adaptation improves over the frozen mapper on some disjoint tasks, while seed sensitivity, long-output repetition, validation/task-ranking mismatch, and 4B failures bound the claim. The selected training recipe is not a universal renderer.

**Utility:** avoiding historical receiver prefill is an implementation property, not proof of application value. Source generation dominates short-answer cost, while longer outputs may repeat, terminate too early or lose task quality. The completed objective comparisons do not establish a broadly attractive quality/total-latency frontier. Any narrower semantic claim must be limited to the valid blinded cases documented separately.

Intervals use 5,000 paired parent-cluster bootstrap resamples with a fixed seed. Repeated lengths, branches, orientations and variants stay within their parent. Intervals are exploratory and are not adjusted for the many reported comparisons; no noninferiority margin or equivalence test was declared. Code is a small public benchmark with possible upstream training exposure; prose is a narrow custom procedural suite, with a discovered pricing-unit ambiguity in characterization and its derived evidence extensions. The confirmation subset and 4B replication are smaller than characterization. Failed outputs, source truncations and unsuccessful infrastructure attempts remain visible. Human labels, a stronger receiver, cross-family model transfer, convergence beyond the budget and production serving remain untested.

`results/phase2_v1/review_examples.json` selects illustrative cases by a written deterministic objective-outcome rule. Examples are not a representative quality estimate. Judge order disagreements have a separate blind export. The compact bundle includes original prompts, public source reasoning text/token trajectories, candidates and token IDs, hidden grading material, score traces, timing records, manifests, frozen source snapshots, raw valid/invalid judge attempts and model settings, and all reporting code. Weights, mapper tensors, large paired caches, environments, credentials and Git internals are excluded; their inventories and regeneration instructions are retained. This is a local review deliverable, with no publication, push or project-license choice.
