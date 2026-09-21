#!/usr/bin/env python3
"""Bounded, public-input-only screen runner; no provider creation or CPU grading.

Uses the existing allocation lease, stable-inode task claims, atomic receipts and
identity-bound answer_durable sampler. All four real CUDA calibration cases must
pass with the same frozen declaration and numerical implementation first.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, digest, seed_for, sha, write
from scripts.sparse_repair_calibrate import (read, safe_public_path, runtime_gate,
    verify_weights, timed, tensor_bytes, fingerprints, memory_record, replay, MAPPER_SHA256)

NUMERICAL_FILES = (
    "gearshift/sparse_repair.py", "gearshift/coding_inference.py", "gearshift/core.py",
    "gearshift/coding_gradients.py", "gearshift/coding_post_progress.py",
    "gearshift/coding_confirmation_sampling.py", "gearshift/coding_control.py")
RUNNER_FILES = ("scripts/sparse_repair_screen.py", "scripts/sparse_repair_calibrate.py",
               "scripts/coding_coverage_v2_worker.py", "scripts/coding_confirmation_runtime.py",
               "gearshift/coding_confirmation_lease.py", "gearshift/coding_recovery.py")
CONDITIONS = ("D", "H", "P", "N_5", "M_5", "R_5", "N_10", "M_10", "R_10",
              "N_25", "M_25", "R_25", "R_random", "R_recent")
STREAMS = ("answer_small", "coverage_answer_1", "coverage_answer_2")


def implementation_hashes(root):
    return {path: sha(Path(root) / path) for path in NUMERICAL_FILES + RUNNER_FILES}


def validate_plan_implementation(plan, root, declaration_sha):
    actual = implementation_hashes(root)
    if plan.get("declaration_sha256") != declaration_sha or plan.get("implementation_hashes") != actual:
        raise ValueError("Execution plan does not bind the frozen screen implementation/declaration")
    if not isinstance(plan.get("code_commit"), str) or len(plan["code_commit"]) != 40:
        raise ValueError("Execution plan must identify the committed screen source")
    return actual


def validate_calibration(root, declaration, declaration_sha, calibration_root):
    """A missing/failed/stale case blocks the screen; never accept a local smoke."""
    root, calibration_root = Path(root), Path(calibration_root)
    required = {p: sha(root / p) for p in NUMERICAL_FILES + ("scripts/sparse_repair_calibrate.py",)}
    receipts = []
    for task in declaration["calibration_tasks"]:
        folder = calibration_root / task["task_id"].replace("/", "__")
        done, result = read(folder / "complete.json"), read(folder / "calibration.json")
        if (done.get("declaration_sha256") != declaration_sha or done.get("passed") is not True or
                done.get("calibration_sha256") != sha(folder / "calibration.json") or
                result.get("declaration_sha256") != declaration_sha or result.get("task_id") != task["task_id"] or
                result.get("history_sha256") != task["history_sha256"] or result.get("passed") is not True or
                result.get("all_same_path_controls_passed") is not True or result.get("backing_caches_unchanged") is not True):
            raise ValueError("Missing, failed or mismatched CUDA calibration: " + task["task_id"])
        implementations = result.get("implementation_hashes", {})
        if any(implementations.get(path) != expected for path, expected in required.items()):
            raise ValueError("Calibration did not validate this numerical implementation")
        runtime = result.get("runtime", {})
        if (runtime.get("torch") != "2.8.0+cu128" or runtime.get("cuda") != "12.8" or
                runtime.get("transformers") != "4.57.6" or "H200" not in runtime.get("gpu", "") or
                runtime.get("BF16") is not True or runtime.get("TF32") is not False or
                runtime.get("deterministic_algorithms") is not True):
            raise ValueError("Calibration was not the pinned real H200 CUDA execution")
        receipts.append({"task_id": task["task_id"], "calibration_path": str(folder / "calibration.json"),
                         "calibration_sha256": sha(folder / "calibration.json"),
                         "complete_sha256": sha(folder / "complete.json")})
    if len(receipts) != 4 or len({r["task_id"] for r in receipts}) != 4:
        raise ValueError("All four CUDA calibration cases are required")
    return {"all_four_passed": True, "declaration_sha256": declaration_sha,
            "numerical_implementation_hashes": required, "cases": receipts}


def validate_inputs(root, declaration_path, declaration_sha, mapper, visible_path, calibration_root, requested=None):
    root = Path(root)
    path = safe_public_path(root, declaration_path)
    if sha(path) != declaration_sha:
        raise ValueError("Frozen screen declaration changed")
    d = read(path)
    if ([c["name"] for c in d["conditions"]] != list(CONDITIONS) or d["screen_answer_records"] != 504 or
            len(d["screen_tasks"]) != 12 or len({r["task_id"] for r in d["screen_tasks"]}) != 12):
        raise ValueError("Screen matrix must remain exactly12tasks ×14conditions ×3draws")
    if d["mapper"]["sha256"] != MAPPER_SHA256 or sha(mapper) != MAPPER_SHA256:
        raise ValueError("Wrong primary ROTATING mapper")
    gate = validate_calibration(root, d, declaration_sha, calibration_root)
    public = safe_public_path(root, visible_path)
    if sha(public) != d["inputs"]["public_development_inputs"]["sha256"]:
        raise ValueError("Original public development inputs changed")
    visible = {r["task_id"]: r for r in read(public)}
    tasks = d["screen_tasks"]
    tids = {r["task_id"] for r in tasks}
    if tids & {r["task_id"] for r in d["calibration_tasks"]}:
        raise ValueError("Screen/calibration overlap")
    if requested is not None and (not requested or not set(requested) <= tids):
        raise ValueError("Only frozen screen tasks may run")
    selected = []
    for row in tasks:
        tid = row["task_id"]
        history_path = safe_public_path(root, row["history_path"])
        if sha(history_path) != row["history_sha256"]:
            raise ValueError("Saved source history changed")
        h = read(history_path)
        prefix = h["prompt_ids"] + (h["reasoning_ids"][:-1] if h["natural_boundary"] else h["reasoning_ids"])
        if (h["task_id"] != tid or prefix != h["prefix_ids"] or len(prefix) != h["prefix_cache_length"] or
                h["bridge_ids"] != [151668] or len(prefix)-len(h["prompt_ids"]) != row["historical_reasoning_positions"]):
            raise ValueError("Exact natural handoff history differs")
        seeds = [{"seed_index": i, "stream": s, "answer_seed": seed_for(tid, 0, s)} for i, s in enumerate(STREAMS)]
        if d["seeds"][tid] != seeds or tid not in visible:
            raise ValueError("Frozen seed/public-task mismatch")
        if requested is None or tid in requested:
            selected.append((row, h))
    if (root / "data/coding_pilot_v1/private").exists():
        raise ValueError("Private tests may not be mounted on generation workers")
    return d, selected, visible, gate


def verify_draw(folder, expected):
    folder = Path(folder)
    if not (folder / "complete.json").exists():
        return None
    done = read(folder / "complete.json")
    if done.get("identity") != expected:
        raise ValueError("Completed sparse draw identity differs")
    for name, expected_sha in done["files"].items():
        if sha(folder / name) != expected_sha:
            raise ValueError("Completed sparse draw artifact changed: " + name)
    result = read(folder / "answer.json")
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("Completed answer binding differs")
    return result


def condition_parameters(name):
    if name in ("D", "H", "P"):
        return "disabled", 1., "native_mass"
    if name in ("R_random", "R_recent"):
        return "R", .1, name.split("_")[1]
    mode, percent = name.split("_")
    return mode, int(percent) / 100, "native_mass"


def run_draw(c, task, h, condition, seed, bases, shared_timing, shared_memory):
    import torch
    from gearshift.core import CacheInjector, CacheExtractor
    from gearshift.coding_confirmation_sampling import answer_durable
    from gearshift.sparse_repair import HistoryLayout, SparseRepairController
    tid = task["task_id"]
    dest = c["top"] / "screen/tasks" / tid.replace("/", "__") / condition / f"seed_{seed['seed_index']}"
    identity = {"experiment_id": "sparse_repair_01", "cohort": "development_screen", "task_id": tid,
        "condition": condition, **seed, "declaration_sha256": c["declaration_sha"],
        "history_sha256": sha(dest.parent.parent / "prompt_only_template.json") if condition == "P" else task["history_sha256"],
        "original_source_history_sha256": task["history_sha256"], "mapper_sha256": MAPPER_SHA256,
        "implementation_sha256": c["implementation_sha"], "teacher_answer_prefix_supplied": False}
    existing = verify_draw(dest, identity)
    if existing is not None:
        return existing
    bind(dest / "draw_identity.json", identity)
    backend = c["receiver"]
    history = bases["P_history"] if condition == "P" else h
    base = bases["P"] if condition == "P" else bases["native"] if condition == "D" or condition.startswith("N_") else bases["hybrid"]
    mode, fraction, selector = condition_parameters(condition)
    backend.sampling_identity = {"model_id": backend.name,
        "revision": c["declaration"]["protocol"]["models"]["receiver"]["revision"],
        "runtime_identity_sha256": c["runtime_sha"], "declaration_sha256": c["declaration_sha"],
        "cache_identity_sha256": digest({**identity, "conditioning": digest(history)})}
    protected = [t for name in ("native", "mapped", "hybrid", "P") for pair in bases[name] for t in pair]
    versions = [t._version for t in protected]
    c["guard"](); c["telemetry"].reset("screen_draw_" + tid + "_" + condition + "_" + str(seed["seed_index"]))
    start = time.monotonic()
    cache, clone_seconds = timed(lambda: CacheInjector.create(base, clone=True))
    if any(x.data_ptr() == y.data_ptr() for pp, qq in zip(base, CacheExtractor.tensors(cache)) for x, y in zip(pp, qq)):
        raise ValueError("Private screen cache aliases immutable backing")
    controller = None
    if condition != "P":
        controller = SparseRepairController(backend.model, bases["native"], bases["mapped"],
            HistoryLayout(len(h["prompt_ids"]), len(h["prefix_ids"])), mode=mode, fraction=fraction,
            selector=selector, task_id=tid, draw_seed=seed["answer_seed"], trace_enabled=True, profile=False)
    resume_path = dest / "sampler/resume.json"
    prior = read(resume_path) if resume_path.exists() else None
    resumed_completed = prior is not None and prior.get("state") == "complete"
    def publish(**values):
        c["publish"](condition=condition, seed_index=seed["seed_index"], **values)
        if controller is not None:
            write(dest / "progress_working_set.json", {"summary": controller.summary(), "trace_records": controller.trace_records()})
    import contextlib
    trace_rebuild_seconds = 0.
    try:
        with controller if controller is not None else contextlib.nullcontext():
            record, attempt_seconds = timed(lambda: answer_durable(backend, history, cache, tid, seed["stream"],
                4096, dest / "sampler", c["telemetry"], c["guard"], publish))
            # A crash after sampler completion but before draw receipt must not
            # fabricate an empty union. Replay exactly the recorded forward count.
            if resumed_completed and controller is not None:
                controller.set_condition(mode, fraction=fraction, selector=selector, reset=True)
                ids = history["bridge_ids"] + record["answer_ids"][:prior["forwarded_tokens"]]
                _, trace_rebuild_seconds = timed(lambda: replay(backend, base, ids, (), c["guard"]))
        if record["answer_seed"] != seed["answer_seed"] or versions != [t._version for t in protected]:
            raise ValueError("Screen seed or immutable backing cache changed")
        result = {**record, **identity, "clone_no_alias": True, "immutable_backing_versions_unchanged": True,
            "cache_clone_seconds": clone_seconds, "this_attempt_sampler_call_wall_seconds": attempt_seconds,
            "this_attempt_whole_draw_wall_seconds": time.monotonic()-start,
            "trace_rebuild_of_completed_sampler_seconds": trace_rebuild_seconds,
            "sampler_completed_before_this_attempt": resumed_completed,
            "shared_task_cache_setup_seconds": shared_timing, "shared_task_backing_bytes": shared_memory,
            "shared_setup_is_counted_once_per_task_not_independently_per_draw": True,
            "original_saved_source_reasoning_seconds": 0. if condition == "P" else h.get("reasoning_seconds"),
            "new_source_reasoning_performed": False, "memory": memory_record(torch),
            "timing_limitations": "Sampler active time includes durable I/O; interrupted attempts may have incomplete timing. Trace operations add CPU synchronization. Stage-level oracle timings come from calibration, not invented from this unprofiled draw.",
            "trace_limitations": "Incomplete attempts rebuild selected sets by exact replay of committed token history; discarded/uncommitted crash-tail selections are not preserved. Complete-sampler recovery replays forwarded tokens solely to rebuild traces, charged separately."}
        trace = {"summary": controller.summary() if controller else {"condition": "P", "no_reasoning_state": True},
                 "selected_index_records": controller.trace_records() if controller else {},
                 "semantic_trajectory_union_complete": True,
                 "unknown_uncommitted_crash_tail_work_excluded": prior is not None,
                 "profile_enabled": False}
        write(dest / "working_set.json", trace)
        write(dest / "answer.json", result)
        bind(dest / "complete.json", {"identity": identity, "files": {
            name: sha(dest / name) for name in ("answer.json", "working_set.json", "sampler/identity.json",
                "sampler/answer_record.json", "sampler/resume.json", "sampler/complete.json", "sampler/completion_timing.json")}})
        return result
    except BaseException as exc:
        write(dest / "failure.json", {"identity": identity, "exception": type(exc).__name__, "error": str(exc),
            "this_attempt_wall_seconds": time.monotonic()-start,
            "partial_working_set": controller.summary() if controller else None})
        raise
    finally:
        del cache, controller
        gc.collect(); torch.cuda.empty_cache()


def run_task(c, task, h):
    import torch
    from gearshift.core import CacheExtractor
    from gearshift.coding_post_progress import splice
    tid = task["task_id"]
    folder = c["top"] / "screen/tasks" / tid.replace("/", "__")
    start = time.monotonic(); timing = {}
    c["guard"]()
    out, timing["source_cache_reconstruction_seconds"] = timed(lambda: c["source"].prefill_chunked(h["prefix_ids"]))
    source = CacheExtractor.tensors(out.past_key_values); del out
    source_bytes = tensor_bytes(source)
    mapped, timing["mapper_seconds"] = timed(lambda: c["mapper"](source))
    del source; gc.collect(); torch.cuda.empty_cache()
    c["guard"]()
    out, timing["native_cache_construction_seconds"] = timed(lambda: c["receiver"].prefill_chunked(h["prefix_ids"]))
    native = CacheExtractor.tensors(out.past_key_values); del out
    c["guard"]()
    out, timing["native_prompt_prefill_seconds"] = timed(lambda: c["receiver"].prefill_chunked(h["prompt_ids"]))
    prompt = CacheExtractor.tensors(out.past_key_values); del out
    hybrid, timing["hybrid_splice_seconds"] = timed(lambda: splice(prompt, mapped, len(h["prompt_ids"])))
    pids = c["receiver"].tokenizer.apply_chat_template([{"role": "user", "content": c["visible"][tid]["prompt"]}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    rendered = c["receiver"].tokenizer.decode(pids, skip_special_tokens=False)
    if "<think>\n\n</think>" not in rendered:
        raise ValueError("Pinned non-thinking template changed")
    ph = {"task_id": tid, "prompt_ids": pids, "prefix_ids": pids[:-1], "bridge_ids": pids[-1:]}
    bind(folder / "prompt_only_template.json", {**ph, "declaration_sha256": c["declaration_sha"],
        "public_prompt_sha256": digest(c["visible"][tid]["prompt"]), "enable_thinking": False, "rendered_template": rendered})
    out, timing["P_native_prefill_seconds"] = timed(lambda: c["receiver"].prefill_chunked(pids[:-1]))
    pbase = CacheExtractor.tensors(out.past_key_values); del out
    bases = {"native": native, "mapped": mapped, "hybrid": hybrid, "P": pbase, "P_history": ph}
    backing_bytes = {k: tensor_bytes(v) for k, v in bases.items() if k != "P_history"}
    backing_bytes.update(source_peak_then_freed=source_bytes, independent_native_prompt=tensor_bytes(prompt))
    before, timing["backing_fingerprint_before_seconds"] = timed(lambda: [fingerprints(bases[k]) for k in ("native", "mapped", "hybrid", "P")])
    write(folder / "cache_setup_attempts" / (c["attempt_id"] + ".json"), {
        "task_id": tid, "declaration_sha256": c["declaration_sha"], "timing": timing, "backing_bytes": backing_bytes})
    # Rotate scheduling without changing condition identities, seeds or membership.
    shift = [r["task_id"] for r in c["declaration"]["screen_tasks"]].index(tid) % len(CONDITIONS)
    completed = []
    for condition in CONDITIONS[shift:] + CONDITIONS[:shift]:
        for seed in c["declaration"]["seeds"][tid]:
            c["guard"]()
            completed.append(run_draw(c, task, h, condition, seed, bases, timing, backing_bytes))
    after, timing["backing_fingerprint_after_seconds"] = timed(lambda: [fingerprints(bases[k]) for k in ("native", "mapped", "hybrid", "P")])
    if before != after:
        raise ValueError("Task backing cache content mutated")
    receipt = {"task_id": tid, "declaration_sha256": c["declaration_sha"], "implementation_sha256": c["implementation_sha"],
        "prompt_only_template_sha256": sha(folder / "prompt_only_template.json"),
        "completed_records": len(completed), "expected_records": 42, "backing_cache_fingerprints_unchanged": True,
        "cache_setup_timing": timing, "this_attempt_whole_task_wall_seconds": time.monotonic()-start,
        "files": [{"path": f"{r['condition']}/seed_{r['seed_index']}/complete.json",
                   "sha256": sha(folder / r["condition"] / f"seed_{r['seed_index']}" / "complete.json")} for r in completed]}
    write(folder / "task_complete.json", receipt)
    del bases, native, mapped, hybrid, pbase, prompt
    gc.collect(); torch.cuda.empty_cache()
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--worker-id", required=True)
    p.add_argument("--declaration", required=True)
    p.add_argument("--declaration-sha256", required=True)
    p.add_argument("--mapper", required=True)
    p.add_argument("--calibration-root", required=True)
    p.add_argument("--visible", required=True)
    p.add_argument("--weights-receipt", required=True)
    p.add_argument("--task-id", action="append")
    args = p.parse_args(argv)
    # No model or sampler import/loading before all evidence gates pass.
    d, tasks, visible, gate = validate_inputs(ROOT, args.declaration, args.declaration_sha256,
        args.mapper, args.visible, args.calibration_root, args.task_id)
    from scripts.coding_confirmation_runtime import build_context
    from scripts.coding_coverage_v2_worker import claim_job
    c = build_context(args.plan, "sparse_repair_screen", args.worker_id)
    if c["plan"]["experiment_id"] != "sparse_repair_01":
        raise ValueError("Wrong allocation lease experiment")
    try:
        impl = validate_plan_implementation(c["plan"], ROOT, args.declaration_sha256)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        runtime = runtime_gate()
        bind(c["status_root"] / "execution_identity.json", {"declaration_sha256": args.declaration_sha256,
            "implementation_hashes": impl, "runtime": runtime, "calibration_gate": gate})
        weights, verification_seconds = timed(lambda: verify_weights(ROOT / args.weights_receipt))
        write(c["status_root"] / "weight_verification.json", {**weights, "seconds": verification_seconds})
        import torch
        from gearshift.coding_inference import Backend
        from gearshift.coding_gradients import AffineMapper
        c["guard"]()
        source, source_seconds = timed(lambda: Backend(d["protocol"]["models"]["source"]))
        c["guard"]()
        receiver, receiver_seconds = timed(lambda: Backend(d["protocol"]["models"]["receiver"]))
        c["guard"]()
        if source.tokenizer.backend_tokenizer.to_str() != receiver.tokenizer.backend_tokenizer.to_str():
            raise ValueError("Pinned source/receiver tokenizer mismatch")
        mapper = AffineMapper(source, receiver)
        payload = torch.load(args.mapper, map_location="cpu", weights_only=True)
        mapper.load_state_dict(payload["state_dict"], strict=True); mapper.requires_grad_(False); del payload
        c.update(source=source, receiver=receiver, mapper=mapper, declaration=d, declaration_sha=args.declaration_sha256,
            implementation_sha=digest(impl), runtime_sha=digest(runtime), visible=visible)
        write(c["status_root"] / "model_setup.json", {"source_seconds": source_seconds, "receiver_seconds": receiver_seconds})
        receipts = []
        with torch.no_grad():
            for task, h in tasks:
                c["guard"]()
                with claim_job(c, "sparse_screen_" + task["task_id"].replace("/", "__")) as claimed:
                    if not claimed:
                        continue
                    folder = c["top"] / "screen/tasks" / task["task_id"].replace("/", "__")
                    if (folder / "task_complete.json").exists():
                        receipt = read(folder / "task_complete.json")
                        if receipt["declaration_sha256"] != c["declaration_sha"] or receipt["implementation_sha256"] != c["implementation_sha"]:
                            raise ValueError("Existing task receipt identity differs")
                        if receipt["prompt_only_template_sha256"] != sha(folder / "prompt_only_template.json"):
                            raise ValueError("Completed prompt-only template changed")
                        for item in receipt["files"]:
                            if sha(folder / item["path"]) != item["sha256"]:
                                raise ValueError("Existing task draw receipt differs")
                            draw_folder = (folder / item["path"]).parent
                            verify_draw(draw_folder, read(draw_folder / "complete.json")["identity"])
                    else:
                        receipt = run_task(c, task, h)
                    receipts.append(receipt)
                    write(c["status_root"] / "completed_tasks.json", receipts)
        c["publish"](state="complete", stage="bounded_screen_worker_finished", completed_tasks=len(receipts),
            global_expected_tasks=12, global_expected_records=504)
    except BaseException as exc:
        write(c["status_root"] / "failure.json", {"error": str(exc), "exception": type(exc).__name__, "traceback": traceback.format_exc()})
        c["publish"](state="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
