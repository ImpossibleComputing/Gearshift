Decision: use all 200 unique confirmation tasks. Start the primary
confirmation work now; do not wait for the new training pair.

RESOLVE THE RESERVE OVERLAP

This explicitly supersedes the earlier instruction to preserve the
40-task reserve separately.

If those 40 tasks are a subset of the same unused 200-task reserve,
include them in the 200 and retire their separate holdback status.
Count each task once. Do not describe this as 240 tasks or as two
independent test sets.

Verify that the overlap is only between unused reserve manifests,
not with mapper training, prior evaluated development/validation
tasks, or task outcomes already used to guide decisions.

Record the exact IDs, overlap, and this owner-authorized amendment
in the confirmation declaration before inspecting confirmation
outputs.

If any tasks are genuinely exposed, exclude those from claims of
fresh confirmation, proceed with the verified clean tasks, and
report the exception. Do not silently replace them or stop all
independent work for another reserve-choice question.

PRIMARY CONFIRMATION DOES NOT NEED NEW TRAINING

Use the EXISTING completed v2 step-1,024 FIXED and ROTATING
checkpoints for the primary comparison. Verify their recorded
hashes. Do not retrain them or wait for replacement checkpoints.

The currently running FIXED/ROTATING pair should be the SECOND
training-seed replication. Verify that its declared seed and
provenance match that role. Do not restart valid healthy training.

Run primary confirmation and secondary training concurrently.
Use additional appropriate RunPod workers rather than making
confirmation wait for the training GPUs.

Start source reasoning and independent smaller-model reasoning
across task shards. As each source history is durably committed,
fan out its dependent answer jobs. Do not wait for all source
histories to finish before beginning answer generation.

Keep the previously specified eight primary conditions, three
answer seeds, exact shared source histories, frozen generation
recipe, and corrected scorer. Keep scoring isolated from model
generation and optimization.

Evaluate the second training pair on these same confirmation tasks
when its declared checkpoints are ready. Report it separately;
never choose whichever training seed performs best.

SPEED AND BUDGET

Parallel completion matters more than GPU thrift. The $2,000
cumulative soft target and existing authorization remain unchanged.
Do not serialize independent work merely to save money.

Keep numerical controls, task-level statistical pairing, durable
checkpoints, bounded recovery, verified backups, and idle-resource
cleanup. No new architecture or training sweep.

RETURN THE AVAILABLE RESULTS NOW

Since scorer repair and rescoring are complete, export
SCORER_REPAIR_RESULTS.md now, including:

- Original versus corrected v2 condition scores.
- Which outcomes changed and why.
- Updated primary and hybrid contrasts.
- The scope and results of the 45 repeatability checks.

Those checks establish repeatability for what was tested, not
universal evaluator correctness.

Continue the main work without waiting for me to review that export.
Report primary confirmation coverage separately from secondary
training progress. Deliver the complete primary confirmation
report when ready, even if secondary replication is still running.

The first article and gearshift-progress-01 tag remain unchanged.
No pushing, publishing, or deployment.
