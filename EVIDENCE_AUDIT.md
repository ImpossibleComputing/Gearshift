# Evidence-task unit audit

**The characterization evidence template leaves the pricing unit ambiguous. Its frozen numeric field scores cannot support an unqualified factual-accuracy claim.** Monthly volume is stated in thousand requests, but vendor prices say credits per unit without identifying a unit. The hidden formula multiplies by the displayed number of thousands. Treating a unit as one request instead multiplies by 1,000. This issue was discovered in final qualitative review, after generation and objective scoring.

For `characterization_evidence_000`, the packet gives 28 thousand requests and the eligible vendor charges 230 plus 4 per unit. The hidden monthly total is 342 and difference 148. Under a per-request interpretation, the corresponding values are 112,230 and 28,120. The source-native and native-replay answers use the second interpretation. Their disagreement with the hidden answer is not by itself evidence of arithmetic failure.

All 80 characterization evidence parents share this template. Their derived long-output and branching evidence variants, two evidence timing cases, and the archived partial laptop evidence records inherit the same ambiguity. Training, validation, development, confirmation and the separate contextual-memory diagnostic use different templates and are unaffected by this particular issue. The 4B confirmation uses the unambiguous confirmation template. Raw stage memberships are listed in `audit.json`.

| Characterization condition | N | Decision field correct | Frozen per-thousand all fields | Alternative per-request all fields | Either interpretation |
| --- | --- | --- | --- | --- | --- |
| B/native | 80 | 64 | 0 | 58 | 58 |
| B/newturn | 80 | 66 | 0 | 49 | 49 |
| C | 80 | 70 | 0 | 59 | 59 |
| H | 80 | 61 | 0 | 34 | 34 |
| M | 80 | 0 | 0 | 0 | 0 |
| P | 32 | 25 | 0 | 19 | 19 |
| S | 80 | 42 | 0 | 0 | 0 |
| S_think | 80 | 55 | 13 | 18 | 31 |
| T | 32 | 28 | 0 | 22 | 22 |

This sensitivity analysis reuses the same recorded parsed fields and decisions. It changes only the two numeric reference values in a separate analysis; it does not rewrite `objective_scores.json`, score manifests, task files or any candidate output. T and P have 32 cases each, so their aggregate counts are not comparable to the 80-case conditions without matching membership. Neither the alternative nor the either-interpretation column is a replacement primary score.

Blinded judgments remain tied to the exact original task/evidence packet and never receive hidden grading answers. The ambiguous numerical instruction can still affect their factual assessments. Treat characterization and derived-extension evidence judgments as exploratory observations on an imperfect task, rather than a clean semantic benchmark. Preference for a complete, consistent answer may remain informative, but numerical disagreements need this qualification. Writing judgments and the separate confirmation evidence tasks are not affected by this unit defect.

For any future experiment, prices must explicitly say per request or per thousand requests and graders must check that visible-unit conversion. That would require new task identities and fresh outputs. No further inference or retrospective task repair was performed in this bounded run.
