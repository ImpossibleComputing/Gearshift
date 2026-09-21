# Codex task: Gearshift phase 2

## Objective and authorization

The owner wants additional research, including different task families and outputs that require subjective scoring. Earlier instructions to stop at release preparation are superseded. Keep the original pilot and `followup_v1` intact; do not overwrite any historical manifest, checkpoint, result, or narrative.

Build and run experiments, not just a design document. Use the existing local environment and publicly downloadable resources. Do not incur paid cloud/API costs, choose a project license, push commits, or publish. A credential being present is not authorization to spend money. Do not assume a product name such as Codex or Astra identifies a usable inference endpoint.

Proceed autonomously within this brief. Make implementation decisions, record changes, and produce completed partial results when a resource or service is unavailable. Do not silently substitute a weak judge for an unavailable independent judge. Export unscored blind packets and continue objective experiments.

We want answers to three distinct questions:

1. **Transfer:** Does translated state let the receiver use the source history nearly as well as native target replay?
2. **Behavior:** What combination of mapper training and conversational handoff causes the result, and does it survive long generation and different tasks?
3. **Utility:** Does switching models offer an attractive quality/latency tradeoff relative to letting the source finish, a small model alone, or a compact textual handoff?

Do not merge these into a single success label.

## Existing evidence, not a new claim

Read the current `FOLLOWUP_RESULTS.md`, previous review, source, and raw records. On one Qwen3-1.7B -> Qwen3-0.6B pair and 100 questions, the selected affine mapper with a new-turn scaffold scored 78 extracted-correct answers versus 76 for matched text replay; the paired difference interval was -5 to +9 percentage points. Hybrid scored 81. These results did not establish equivalence. Raw-number formatting improved much more than mathematical accuracy. The mapped answers averaged about 3.35 generated tokens. Overall time was dominated by source reasoning; the receiver-stage advantage was not a large end-to-end advantage.

Only the chat-selected mapper received the final scaffold matrix. Initial-affine and plaintext-trained checkpoints were not evaluated under the winning scaffold; no fresh small-only baseline was present. Both training arms' best checkpoints occurred at their 128-step limit. The recipe is promising, but its necessary ingredients, generality, and useful operating range are unresolved.

This phase may reuse old evaluation cases for explicitly labeled diagnosis. Once used to change a method, they are development evidence, not an untouched confirmation set.

## 0. Protect the evidence and harden the research harness

Create a new namespace such as `results/phase2_v1/`, new configs, and a new immutable experiment identity. Retain the existing fail-closed identity, cache verification, EOS, position, final-token, and reinjection tests. Do not rewrite archived results to conform to a new schema.

Fix the known records-only reporting issue: tables and plots must regenerate without `selected_mapper.pt`, model weights, or cache tensors. Keep explicit, separate checkpoint-verification status; missing weights must never be reported as verified.

Before scientific runs, save:
- exact model/tokenizer revisions, cache geometry, precision, software versions, and available hardware;
- git SHA when available, dirty patch/source snapshot, configs and content hashes;
- task manifests, split membership, conditions, scoring definitions, and seeds;
- exact handoff text and token IDs per task family;
- implementation differences from `followup_v1`.

Extend the existing harness instead of making unrelated benchmark scripts with different cache semantics. A shared `TaskSpec` or equivalent should define the input packet, inference-visible prompt, hidden grading material, requested output contract, generation budget, and grader. Hidden reference answers, hidden unit tests, and correctness labels must not enter generation histories, mapper training, compact summaries, or handoff selection.

## 1. Diagnose the current result without retraining

Run the checkpoint/scaffold matrix in `EXPERIMENT_MATRIX.md` on the saved 100-question histories. Reuse recorded conditions only after full identity validation; distinguish identical-history replay from original live-cache inference. Restore the source cache from the exact saved token IDs using the validated construction method. Do not describe replay as a new live-source timing measurement.

Primary matrix: initial reconstruction-trained affine, plaintext-functional, chat-functional; each with old `Answer:` and the winning new-turn scaffold. Add matching text-replay controls. Include matched large-only and fresh small-only controls where not already available. Retain all six mapper cells, not just the best one.

The two contrasts of interest are:
- scaffold effect within each checkpoint;
- training effect with the new-turn scaffold fixed.

A separate small development diagnostic may decompose the new-turn protocol into valid role-boundary and thinking-mode variations. Read the actual tokenizer chat template before constructing them; malformed chat syntax is not a fair baseline. Compare the same alternatives in native replay and source continuation. Avoid an exhaustive prompt search. Freeze any chosen per-family protocol before confirmation.

Add correct-history versus absent/wrong-history controls on a declared subset. Never mix different positional layouts without recording the change. Wrong-history comparisons are deliberately destructive diagnostics, not fair economic alternatives. For pure mapping, task-specific information should come only from the transferred historical state, not be repeated in a supposed generic bridge.

## 2. Test new task families with the existing mapper frozen

Do not train on the new domains before measuring cross-task transfer. Use the existing selected chat mapper unchanged as the primary zero-adaptation condition. Include both native replay and source-only continuations so receiver limitations can be separated from transfer loss.

Implement the four task families in `EXPERIMENT_MATRIX.md`:
- fresh arithmetic as a continuity/precision anchor;
- executable code generation;
- source-grounded synthesis from self-contained evidence packets;
- open-ended explanation/creative writing with explicit briefs.

For new task families use a development pilot (default 16 independently selected cases per family) to check prompt validity, token limits, grading, native-baseline capability, and sandboxing. Lock final manifests afterward. Development prompts never enter the final sample. Do not filter final cases by whether the large model or mapped model gets them right.

Keep the same original task and source reasoning trajectory across B/C/M/H for a paired case. Source-generated reasoning is allowed to contain a conclusion or code; that is legitimate for the rendering hypothesis. Record overlap and avoid claiming independent acquisition of source reasoning ability. Do not inject benchmark answers or ground-truth plans into inference.

Use task-appropriate output instructions in the new-turn bridge, not the old universal numeric-only wording. The bridge may state the requested artifact type and output contract. It must not repeat question-specific facts, source reasoning, or computed answers in the zero-history-prefill condition. Count all bridge tokens.

For the primary protocol, hold the ordinary rendering instructions and decoding budget equal across B/C/M/H. In addition, retain an ordinary source-native final-answer baseline when resetting its conversation could itself hurt it. Do not force a deliberately poor source-only protocol to make switching look useful. Validate baseline choices on development, not final outcomes.

Start with the existing 1.7B -> 0.6B pair. A receiver that performs poorly even with exact native replay may be too weak for a task: report that region rather than calling it cache failure. Choosing easier tasks or a different receiver for later experiments must use separate development data and be reported as a new regime, not a deletion of failed results.

## 3. Evaluation and economics

Run the shared comparisons and primary/secondary scopes in `EXPERIMENT_MATRIX.md`.

Objective evaluation:
- math: semantic numeric correctness, contract validity, EOS/cap separately;
- code: one completion per task, actual hidden-test pass rate, parsing/timeout/test failures separately; retain raw code;
- factual synthesis: claims supported by the packet, required-fact coverage, contradictions, numerical correctness and source-ID fidelity; stylistic judgments separate;
- open-ended work: verifiable constraints plus blinded rubric judgments, with an explicit unscored state when judging has not occurred.

Do not treat native replay's answer as ground truth. For source-grounded work the supplied evidence packet is the truth source; for code it is the independent executable specification/tests. A source plan may be wrong. Evaluate final quality independently of plan fidelity.

Use `JUDGING.md` for the subjective tracks. Grading code must be tested before final scoring. Invalid or failed outputs count as outcomes, not missing data. Infrastructure failures are recorded separately and rerun only under a documented, condition-independent retry policy.

Measure complete per-condition wall time with device synchronization and randomized condition order. Separate source prefill/reasoning, map, cache cloning/splice, target historical prefill, bridge, and free-running generation. Record actual output tokens, stop reason, first-logit time, and resident/sampled memory. Repeat synchronized timing on a prespecified subset with warmups and medians/dispersion; do not confuse warm caches with cold starts.

Compare against:
- the source continuing to answer from its own cache;
- the small model doing the task itself (with and without thinking where relevant);
- full textual replay of the exact source history;
- original prompt plus a short tail of the source reasoning;
- a compact source-written textual plan/summary on the specified subset, paying for summary generation and receiver prefill.

The text-summary arm is a different message-content policy, not a cache-fidelity control. The tail arm uses a declared token budget and counts the original prompt prefill. Neither is zero target prefill. Keep them separate from the same-history C-versus-M comparison.

Longer answers are economically interesting only if they remain useful. Report quality-versus-total-latency plots, not only tokens/second or prefill ratios. Do not reward short incomplete outputs or padded long ones. A fixed-token decode/teacher-forcing timing test can help isolate compute effects, but label it synthetic and never substitute it for application-quality results.

## 4. Probe sustained state use and branching

On a prespecified subset of the evidence/creative tasks, ask for natural short, medium, and longer outputs from the same source history, using the same per-length instructions across conditions. Default caps: 128/512/1024 generated tokens; mark truncation and ensure requested content is feasible for each cap. Requested length is a task variable, not permission to pad. Report actual lengths.

Compare free-running quality against native replay. Separately compute teacher-forced KL/NLL at early and later positions of a native-target reference continuation (for example positions 1-32, 33-128, 129-256, and 257-512 where available). Do not count each token as an independent sample or claim that a teacher-forced recovery proves free-running recovery.

Create a small deterministic, closed-world memory diagnostic with paired evidence packets differing in one consequential detail. Use identical post-handoff requests and score whether each response reflects its own packet. This helps distinguish useful contextual transfer from fluent defaults. Log and cluster paired variants by their parent case.

One useful extension is branching from one source analysis into three distinct requested outputs (e.g. a decision summary, risks with evidence, and implementation checklist). Clone each baseline's prefix cache fairly. A native replay baseline may prefill once and reuse that prefix too; a source-only baseline may reuse its source cache. Do not manufacture a fan-out advantage by charging competitors repeated work they could cache. Branching cases are one statistical cluster, not three independent examples.

## 5. Improve the mapper only after measuring frozen transfer

Preserve the frozen-mapper cross-task result even if it fails. Then run a controlled improvement experiment, rather than immediately jumping to larger models or nonlinear adapters.

Use disjoint multi-domain training and validation cases, with a reasonable expansion beyond the original 48 training conversations. Keep all final task prompts and their near-duplicate packet seeds out. Use intact individual conversations where possible; avoid concatenating unrelated dialogues merely to reach a token count. If packing is used, log masks/anchors and keep it matched between training-objective arms.

The most useful two arms are:
A. multi-domain functional distillation on ordinary continuation positions;
B. the same training-domain mixture with loss at the actual task-appropriate handoff boundary and across multiple answer positions.

Both models remain frozen; optimize the affine mapper. Use the same initialization and match optimizer updates, domain/context schedule, and number of gradient-bearing target predictions as closely as feasible. Report unavoidable differences; do not attribute a change simultaneously in data, budget, and objective to one factor.

For B, teacher targets come from the native small receiver on the SAME history and SAME bridge. Include later answer positions, using chunked computation or sparse teacher-forced windows to manage memory. This is distinct from learning to copy the source's final answer and must not use hidden test gold labels.

The old 128-step stopping limit is not inherited. Use checkpointed learning curves (e.g. totals 256/512/1024 updates, validation every 64) with a declared validation-based plateau rule and explicit resource cap. Never extend training because final test scores were disappointing. If still improving at the cap, say that convergence is untested. More steps alone are not a claim of progress; retain validation tradeoffs across domains.

Prefer three mapper-training seeds on a reduced, declared experiment before claiming robustness. If only one is feasible, publish that limit and retain all completed seeds rather than selecting the best. Repeated seeds/questions are correlated observations; preserve cluster structure.

Choose the evaluation-access policy in advance. Either (1) score a frozen-mapper characterization set, then reserve entirely different cases for adaptation confirmation; or (2) generate frozen-mapper final outputs but keep their scores and qualitative outcomes sealed until the adaptation procedure and checkpoints are frozen from separate development data. In option 2 both frozen and adapted methods can receive one common final evaluation. Do not tune after inspecting that final evaluation. Small public benchmarks may not contain two full-size final samples; declare smaller disjoint confirmation sets or a separately identified new benchmark rather than reuse inspected cases and call them fresh. Never relabel zero-adaptation results as adapted results.

## 6. Model-size replication, after the task changes are informative

Replicate the selected recipe on 4B -> 0.6B with fresh source-specific fitting and functional training. Use the same task definitions and output contracts, comparable reasoning allowances, and the actual 4B cache geometry/layer alignment. Do not treat the earlier reconstruction-only 4B run as a test of this improved recipe.

If native 0.6B replay is the limiting factor on longer outputs, a declared 4B -> 1.7B pair is also useful: it changes receiver capacity, not just thinker size. Do not assume its cache geometry, tokenizer, or layer mappings match. Rerun controls. Keep pair comparisons conditional on their own native replay and source baselines.

Cross-family transfer, joint model pretraining, and production serving are separate research directions, not prerequisites for answering these phase-2 questions.

## 7. Statistics and reporting rules

Before confirmation record primary contrasts, metrics, sample sizes, and any noninferiority margin. A suggested initial confirmation size is 400 fresh arithmetic items; 100 code tasks; 80 evidence packets; 60 open-ended briefs. These are exploratory design defaults, not power guarantees. Adjust using development/resource information BEFORE opening confirmation results. Preserve the intended set, exact completed set, and reasons for incompletion.

Paired confidence intervals resample tasks/packets with all their conditions together. Source trajectories, multiple lengths, branches, variants, judge orientations, and generation seeds for one item are dependent. Use hierarchical or cluster bootstrap as appropriate. Report raw gains/losses/ties and absolute quality, not accuracy-retention ratios alone.

A failure to detect a difference is not equivalence. The old 100-item pilot can inform approximate sample planning but cannot certify a required final N. Do not stop when p becomes favorable. If a formal noninferiority claim is attempted, prespecify a practically acceptable margin and sample plan; otherwise report the paired interval and unresolved tradeoff.

Keep task families separate. Do not hide a failed code or creative track inside a favorable aggregate. Distinguish task-general frozen transfer, adapted transfer, model-size replication, and implementation-specific timing.

## 8. Deliverables

Create:
- `PHASE2_RESULTS.md`: actual experiments, uncertainty, failure analysis, and answers to the three main questions;
- `REPRODUCE_PHASE2.md`: runnable stage commands and exact identities;
- configs, dataset/prompt manifests, hidden-grader separation checks, selected checkpoint hashes, source/mapper revisions;
- original prompts, source token trajectories, per-condition raw output texts AND token IDs, timings, graders, and raw judgments;
- primary/secondary result tables and quality/latency plots, all regenerable from compact records;
- an explicitly labeled blinded judge/human-review packet with a separate condition key, plus pending-judgment status;
- example selection by a declared rule, including clear successes, failures, contradictions, and judge disagreements;
- a compact `gearshift_phase2_review.zip` with source, tests, docs, small JSON/CSV/JSONL records, plots, and SHA-256 manifest.

Exclude weights, mapper tensors, multi-GiB KV caches, credentials, environments and .git from the compact ZIP. List excluded heavy artifacts by hash/size and provide regeneration paths. Keep them locally for replication. Report what actually ran and what remains untested. Do not convert pending subjective judgments into invented scores.

Do not publish during this task. The next decision is evidence-led: further research, an engineering post, a stronger report, or a clear boundary/negative result. No favorable verdict is required.
