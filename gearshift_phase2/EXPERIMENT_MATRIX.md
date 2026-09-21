# Phase-2 experiment matrix

## First: what produced the numeric result?

Use the old 100 histories as diagnosis, without claiming fresh confirmation:

| Mapper | Existing Answer: | New-turn scaffold |
|---|---|---|
| Reconstruction-trained affine initialization | Run | Run |
| Plaintext-functional checkpoint | Run | Run |
| Chat-functional selected checkpoint | Already recorded, verify reuse | Already recorded, verify reuse |

Text replay must have both columns. Record math accuracy, raw-output validity, EOS/cap, and actual generated lengths separately. Further valid role/thinking-mode controls are a small development experiment, not an unbounded prompt competition.

## Core methods for the new task suite

- **S:** small alone, original task, no source history. Include small-thinking and small-direct modes in the diagnostic/development comparison; preserve a declared competent baseline for confirmation.
- **B:** large thinker continues from its own state. Same requested artifact, with matched scaffold plus a normal source-native final-answer control when needed.
- **C:** small prefills the exact source prompt/reasoning token sequence, then the task-appropriate handoff scaffold.
- **M:** translated complete source cache, zero historical target prefill, then the same scaffold.
- **H:** native target question/prompt cache plus mapped reasoning at the original positions; prompt prefill is paid and disclosed. Secondary rescue condition, not a replacement for M.
- **T:** small sees original prompt plus the final 256 source-reasoning tokens (use a shorter declared tail for very short histories). This cheap baseline does no additional source generation, but target prompt/tail prefill is paid.
- **P:** source creates a compact plan/summary, followed by small rendering from the original prompt plus that summary. Default summary cap 256 tokens. Pay for producing it and reading it. Save raw summaries and source timing. This changes message content and cannot serve as a native-cache equivalence control.

Run B/C/M/S as the primary suite. Run H/T/P on a prespecified shared subset of at least 32 cases in each longer-output family when feasible; promote comparisons to the full set only by a rule fixed before confirmation. For arithmetic include T because a final reasoning tail is a particularly plausible cheap alternative. An intentionally tiny/poor summary should not be treated as a strong optimized baseline; all policies are development-selected and reported honestly.

## Task families and default scope

| Family | Default new confirmation set | Artifact | Main question | Scoring |
|---|---:|---|---|---|
| Arithmetic anchor | 400 untouched GSM8K test IDs, excluding all previously inspected IDs | Final number, with a separate explanatory subset | Is the earlier observation stable with a narrower paired interval? | Numeric accuracy; contract separately |
| Executable code | 100 HumanEval+ tasks, with disjoint development IDs | Complete function, not just an answer label | Can the receiver execute a source-derived plan into working code? | Hidden-test pass rate, syntax, timeout, termination |
| Evidence-grounded synthesis | 80 self-contained packet cases | 200–450-word analysis or report, plus structured fields on a subset | Does the cache retain several facts, exceptions, constraints and relationships? | Fact support, required coverage, contradictions, calculations, clarity |
| Open-ended explanation / creative work | 60 briefs: balanced technical explanations and constrained short narratives | Natural prose of several hundred tokens | Does source intent survive, and is the final artifact actually useful/good? | Verifiable constraints plus blind rubric preferences |

Public datasets must be pinned and their official grading contracts preserved. Label subsets and protocol modifications; do not present scores as official leaderboards. Generated evidence packets are a diagnostic suite, not a claimed established benchmark.

Default 16 development cases per new family. Freeze final sample membership and keep confirmation outcomes out of task/prompt selection. For HumanEval+ reserve sufficient distinct task IDs before selecting a 100-task confirmation sample. A local secure code sandbox is a prerequisite: no network, no secrets or host mounts, unprivileged execution, resource/time limits, and disposable filesystem. A Python subprocess timeout alone is not a security sandbox. If unavailable, save code and mark execution scoring pending; do not run arbitrary generated code with normal user privileges.

## Evidence-packet construction

Use neutral, fictional, self-contained inputs rather than private company information or open-web fact retrieval. Examples: a small migration decision with latency/cost/compatibility tables; an incident timeline; a set of project status notes containing a later correction; a scheduling/allocation decision with explicit constraints.

Generate packet facts and reference checks from deterministic code before model outputs. Include source IDs, controlled numerical fields, required qualifications, distractors, and at least one important exception. Separate the inference-visible packet from evaluation-only required facts and reference calculations. Use independent problem templates/seeds for development, training, and confirmation; keep near-duplicates and paired counterfactual variants in the same split.

Avoid cases that can be answered generically without reading the packet. Some packet pairs should differ in exactly one decision-changing fact. The source can reason about each packet; the receiver should follow the packet it actually receives, not a memorized default. Report wrong-history and no-history diagnostics separately.

For factuality, an exact-string/entity check cannot establish arbitrary semantic correctness. Mark automated field checks separately from claim-level model/human judgments. Score coverage as well as precision so saying very little is not rewarded.

## Open-ended briefs

Technical examples: explain a novel toy protocol from a supplied specification to a specified audience; compare two designs using supplied constraints; turn a plan into a useful implementation narrative.

Creative examples: a short story with a causal outline, perspective, tone, and required plot facts. Some briefs can be more open, but include enough specificity to test source intent rather than generic fluency.

Record the distinction between hard constraints (such as facts that must stay true), soft qualities (such as voice and coherence), and subjective preference. Do not claim that a weak small model's prose is frontier-quality merely because mapped and native-small prose are similarly weak.

IFEval-style checks can be applied to the custom writing tasks, but those scores are not official IFEval results. An optional unchanged official IFEval subset is a separate track, not silently conflated with custom prompts.

## Long-generation / branching subset

Declare 24 evidence or creative cases for a short/medium/long requested-output comparison, with 128/512/1024 generation-token caps. Select valid per-length requirements on development. Use real requested elaboration, never filler. Compare early and late failures, contradiction rate, looping, EOS/caps, and quality versus total latency.

For 16 packets, branch from one source history into three outputs. Clone caches for ALL methods; amortize source work for all, target prefill once for C, and mapping once for M where the implementation actually reuses it. Report both one-off and amortized costs. Cluster the three outputs by packet in uncertainty calculations.

## Interpretive outcomes

- M matches C but both trail B badly: cache transfer may work, receiver capability is insufficient for the desired application.
- M trails C on new tasks: cross-domain or long-generation transfer gap; test adaptation before blaming receiver size.
- M beats S but not compact T/P: a useful transfer primitive may lack an economic advantage over ordinary text handoff.
- M approximately matches C/B and improves total latency at similar content quality: stronger evidence of practical value in that defined regime.
- A checkpoint-independent scaffold improvement: protocol accounts for more of the original result than training did.
- Functional training improves long-answer behavior under the fixed scaffold: better support for behavioral cache distillation as a useful ingredient.

These are hypotheses to distinguish, not desired outcomes.
