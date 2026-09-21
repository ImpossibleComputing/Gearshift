#!/usr/bin/env python3
"""Regenerate honest sparse-repair reports from compact receipts only.

No torch/model imports, candidate execution, private test access or provider calls.
Collection verifies available committed draw/score hashes and snapshots only
whitelisted numerical metadata. Portable regeneration uses that snapshot alone.
NumPy is required only for the predeclared complete-screen cluster bootstrap.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics

CONDITIONS = ("D", "H", "P", "N_5", "M_5", "R_5", "N_10", "M_10", "R_10",
              "N_25", "M_25", "R_25", "R_random", "R_recent")
DEFAULT_ROOT = Path("results/sparse_repair_01")
NUMERICAL_FILES = ("gearshift/sparse_repair.py", "scripts/sparse_repair_calibrate.py",
    "gearshift/coding_inference.py", "gearshift/core.py", "gearshift/coding_gradients.py",
    "gearshift/coding_post_progress.py", "gearshift/coding_control.py", "gearshift/coding_confirmation_sampling.py")
REQUIRED_CONTROLS = ("unmodified_hook_vs_deployed_dense", "N100_vs_dense_native",
    "R0_vs_existing_H", "R100_vs_matched_native_native_splice")
DECISION_QUESTION_IDS = ("sparse_native_access", "sparse_mapped_state", "targeted_repair",
                       "complete_answer_working_set", "next_investment")
DECISION_VERDICTS = {key: {"supported", "not_supported", "mixed", "unavailable", "blocked"}
                    for key in DECISION_QUESTION_IDS[:-1]}
DECISION_VERDICTS["next_investment"] = {"cheaper_selector", "local_translation",
    "selective_receiver_computation", "compact_text", "no_further_work", "unavailable", "blocked"}


def summary_pointer(summary, pointer):
    """Resolve a non-root RFC 6901 JSON pointer, without evaluating any code."""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Decision evidence_refs must be non-root summary JSON pointers")
    value = summary
    for raw in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw):
            raise ValueError("Invalid JSON-pointer escape in decision evidence_refs")
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", token) and int(token) < len(value):
            value = value[int(token)]
        else:
            raise ValueError("Decision evidence_ref does not resolve in computed summary: " + pointer)
    return value


def validate_decision_review(review, summary, input_sha):
    """Validate optional analyst interpretation; never change computed evidence."""
    top = {"schema", "evidence_input_sha256", "declaration_sha256", "analyst", "reviewed_at_utc", "questions"}
    if not isinstance(review, dict) or set(review) != top or review.get("schema") != "gearshift.sparse_repair.decision_review.v1":
        raise ValueError("Invalid decision_review schema or fields")
    if (not re.fullmatch(r"[0-9a-f]{64}", review["evidence_input_sha256"] if isinstance(review["evidence_input_sha256"], str) else "") or
        review["evidence_input_sha256"] != input_sha or review["declaration_sha256"] != summary["declaration_sha256"]):
        raise ValueError("Decision review input/declaration hash binding mismatch")
    if not isinstance(review["analyst"], str) or not review["analyst"].strip():
        raise ValueError("Decision review requires analyst attribution")
    try:
        timestamp = datetime.datetime.fromisoformat(review["reviewed_at_utc"].replace("Z", "+00:00"))
        if timestamp.utcoffset() != datetime.timedelta(0):
            raise ValueError("not UTC")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Decision review requires an ISO-8601 UTC reviewed_at_utc") from exc
    questions = review["questions"]
    if (not isinstance(questions, list) or len(questions) != 5 or
        any(not isinstance(q, dict) for q in questions) or
        [q.get("id") for q in questions] != list(DECISION_QUESTION_IDS)):
        raise ValueError("Decision review requires exactly five fixed question IDs in order")
    coverage = summary["coverage"]
    complete = (summary["status"] == "COMPLETE_EXPLORATORY_SCREEN" and
        isinstance(summary.get("statistics"), dict) and not summary.get("evidence_integrity_issues") and
        coverage.get("planned_tasks") == 12 and coverage.get("planned_draws") == 504 and
        coverage.get("generated_complete_draws") == 504 and coverage.get("scored_binary_draws") == 504 and
        coverage.get("all_four_calibration_tasks_have_verified_pass") is True and
        len(coverage.get("calibration_passed_task_ids", [])) == 4 and
        set(coverage.get("calibration_passed_task_ids", [])) == set(coverage.get("calibration_required_task_ids", [])) and
        summary.get("generation_sealed") is True and summary.get("scoring_manifest_present") is True and
        all(summary.get("condition_diagnostics", {}).get(c, {}).get("scored_binary") == 36 for c in CONDITIONS))
    for question in questions:
        if set(question) != {"id", "verdict", "rationale", "limitations", "evidence_refs"}:
            raise ValueError("Invalid decision question fields")
        verdict = question["verdict"]
        if not isinstance(verdict, str) or verdict not in DECISION_VERDICTS[question["id"]]:
            raise ValueError("Invalid decision verdict")
        if not complete and verdict not in {"unavailable", "blocked"}:
            raise ValueError("Substantive decision verdict requires all 504 binary outcomes, seal, four calibration passes and clean integrity")
        if not isinstance(question["rationale"], str) or not question["rationale"].strip():
            raise ValueError("Decision question requires rationale")
        limitations, refs = question["limitations"], question["evidence_refs"]
        if (not isinstance(limitations, list) or not limitations or
            any(not isinstance(s, str) or not s.strip() for s in limitations)):
            raise ValueError("Decision question requires explicit limitations")
        if (not isinstance(refs, list) or not refs or any(not isinstance(s, str) for s in refs) or len(set(refs)) != len(refs)):
            raise ValueError("Decision question requires distinct evidence_refs")
        for pointer in refs:
            summary_pointer(summary, pointer)


def attach_decision_review(summary, review_path, input_sha):
    """Add a separately hashed annotation after validation; numeric snapshot stays separate."""
    review = read(review_path)
    validate_decision_review(review, summary, input_sha)
    return {**summary, "analyst_interpretation": {"review_sha256": sha(review_path), "review": review}}


def calibration_integrity(case, contract, expected_implementation):
    """Strict report gate, independent of truthy case.passed. Never alters data."""
    problems=[]
    if contract is None or contract.get("declaration_sha256") != case.get("declaration_sha256"):
        problems.append("missing_or_mismatched_execution_contract")
    elif (case.get("task_id") not in contract.get("calibration_task_ids",[]) or
          contract.get("implementation_hashes") != case.get("implementation_hashes")):
        problems.append("task_or_implementation_differs_from_execution_contract")
    if any(case.get("implementation_hashes",{}).get(p)!=h for p,h in expected_implementation.items()):
        problems.append("calibration_does_not_bind_current_numerical_implementation")
    runtime=case.get("runtime",{})
    if (runtime.get("torch")!="2.8.0+cu128" or runtime.get("cuda")!="12.8" or
        runtime.get("transformers")!="4.57.6" or "H200" not in runtime.get("gpu","") or
        runtime.get("BF16") is not True or runtime.get("TF32") is not False or
        runtime.get("deterministic_algorithms") is not True):
        problems.append("not_pinned_actual_H200_runtime")
    controls=case.get("controls",{})
    if set(controls)!=set(REQUIRED_CONTROLS):
        problems.append("missing_or_changed_required_controls")
    computed=[]
    for name in REQUIRED_CONTROLS:
        c=controls.get(name,{})
        rows=c.get("positions",[])
        if ([r.get("answer_position") for r in rows]!=case.get("probe_positions") or not rows or
            c.get("altered_shape") is not False):
            problems.append("control_probe_schedule_or_shape_mismatch:"+name)
        rowpass=[r.get("bit_exact") is True and r.get("max_abs")==0.0 and r.get("top1_equal") is True for r in rows]
        measured=bool(rows) and all(rowpass)
        if c.get("passed") is not measured or any(r.get("passed") is not v for r,v in zip(rows,rowpass)):
            problems.append("control_pass_flag_disagrees_with_metrics:"+name)
        computed.append(measured)
    if case.get("all_same_path_controls_passed") is not all(computed):
        problems.append("aggregate_control_pass_flag_disagrees")
    if case.get("passed") is not (all(computed) and case.get("backing_caches_unchanged") is True):
        problems.append("case_pass_flag_disagrees_with_controls_or_cache_integrity")
    return problems


def probe_integrity(records, kind, task, declaration_sha, contract, trace, expected_implementation):
    problems=[]
    if not task: return ["probe_task_not_in_frozen_calibration_sample"]
    if (contract is None or contract.get("declaration_sha256")!=declaration_sha or
        task["task_id"] not in contract.get("calibration_task_ids",[]) or
        any(contract.get("implementation_hashes",{}).get(p)!=h for p,h in expected_implementation.items())):
        problems.append("probe_execution_contract_not_bound_to_frozen_current_calibration")
    if (trace is None or trace.get("task_id")!=task["task_id"] or trace.get("declaration_sha256")!=declaration_sha or
        trace.get("kind")!="calibration_only_not_screen" or trace.get("complete") is not True or
        len(trace.get("answer_ids",[]))>129 or trace.get("seed")!=task.get("calibration_seed")):
        problems.append("probe_reference_not_bound_to_own_completed_calibration_D_trace")
    available=[] if trace is None else [p for p in (0,32,128) if p<len(trace.get("answer_ids",[]))]
    if kind=="same_query_attention":
        expected={(layer,pos,mode,p) for layer in range(36) for pos in available for mode in ("N","M","R") for p in (.05,.1,.25)}
        keys={(r.get("layer"),r.get("answer_position"),r.get("mode"),r.get("fraction")) for r in records}
        if len(keys)!=len(records) or keys!=expected:
            problems.append("same_query_probe_coverage_or_identity_mismatch")
        for r in records:
            numeric=(r.get("attention_output_relative_l2"),r.get("attention_output_max_abs"))
            masses=(r.get("selected_mass_sum_per_kv_group",[]),r.get("all_reasoning_mass_sum_per_kv_group",[]))
            if (r.get("identical_prefix_and_current_query") is not True or
                any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in numeric) or
                any(len(v)!=8 for v in masses) or
                any(not math.isfinite(a) or not math.isfinite(b) or a<0 or a>b+1e-5 for a,b in zip(*masses))):
                problems.append("invalid_same_query_probe_metrics_or_claim");break
    else:
        keys=[(r.get("mode"),r.get("fraction")) for r in records]
        if len(set(keys))!=len(keys) or not set(keys)<={(m,p) for m in ("N","M","R") for p in (.05,.1,.25)}:
            problems.append("distribution_probe_arm_identity_mismatch")
        if any([p.get("answer_position") for p in r.get("positions",[])]!=available or
               r.get("matched_prefix_not_identical_later_layer_query") is not True for r in records):
            problems.append("distribution_probe_prefix_schedule_or_query_claim_mismatch")
    return problems


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def scoped(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or "private" in relative.parts:
        raise ValueError("Unscoped/private input is forbidden")
    path = Path(root) / relative
    path.resolve().relative_to(Path(root).resolve())
    return path


def score_test_accounting(score):
    """Final public test-receipt counters only; never inspect private values."""
    receipts=score.get("receipts",[])
    if not isinstance(receipts,list) or any(not isinstance(r,dict) for r in receipts):
        raise ValueError("Malformed public score test receipts")
    if any(not isinstance(r.get("attempts",[]),list) for r in receipts):
        raise ValueError("Malformed public score attempt metadata")
    result={"test_receipt_count":len(receipts),
        "test_receipts_with_multiple_attempts":sum(len(r.get("attempts",[]))>1 for r in receipts),
        "scope":"Top-level final test-receipt durations only, not all infrastructure retries, scorer process time or pod billing time."}
    for key in ("executed_tests","total_tests"):
        value=score.get(key)
        if value is not None and (type(value) is not int or value<0):
            raise ValueError("Invalid public score test count")
        result[key]=value
    for key in ("cpu_seconds","wall_seconds"):
        values=[r[key] for r in receipts if r.get(key) is not None]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
            raise ValueError("Invalid public final-test duration")
        result["recorded_final_test_"+key]={"n":len(values),"sum":sum(values) if values else None,
            "test_receipts_without_duration":len(receipts)-len(values)}
    return result


def collect_cpu_stage_intervals(root, declaration_sha, load):
    """Hash-bound public wrapper events; an absent exit is never extrapolated."""
    records=[]
    for folder in sorted((root/"allocations").glob("*/sparse_cpu_job/*")):
        stages=("preflight","prepare","run","finalize")
        if not any((folder/(stage+suffix+".json")).exists() for stage in stages for suffix in ("_launch","_exit")):
            continue
        plan_path=folder/"plan.json";plan=load(plan_path);pod_id=folder.parent.parent.name
        if (plan.get("experiment_id")!="sparse_repair_01" or plan.get("job_id")!=folder.name or
            plan.get("declaration_sha256")!=declaration_sha or plan.get("result_root")!=str(DEFAULT_ROOT) or
            plan.get("automatic_retries")!=0):
            raise ValueError("CPU timing plan identity differs")
        try:lease_relative=Path(plan["lease_path"]).relative_to(DEFAULT_ROOT)
        except (KeyError,TypeError,ValueError) as exc:raise ValueError("CPU timing lease is outside public result root") from exc
        lease_path=scoped(root,lease_relative)
        if sha(lease_path)!=plan.get("lease_sha256"):raise ValueError("CPU timing lease hash differs")
        lease=load(lease_path);pod_path=lease_path.parent/"pod.json";pod=load(pod_path)
        if (lease.get("experiment_id")!="sparse_repair_01" or lease.get("pod_id")!=pod_id or pod.get("id")!=pod_id or
            type(lease.get("gpu_count")) is not int or lease["gpu_count"]!=0 or
            lease.get("control_relative")!="allocations/"+pod_id):
            raise ValueError("CPU timing lease/pod/allocation identity differs")
        allocation=lease.get("allocation_epoch");deadline=lease.get("deadline_epoch")
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (allocation,deadline)) or deadline<=allocation:
            raise ValueError("CPU timing lease interval invalid")
        for stage in stages:
            launch_path=folder/(stage+"_launch.json");exit_path=folder/(stage+"_exit.json")
            if not launch_path.exists():
                if exit_path.exists():raise ValueError("CPU stage exit lacks launch receipt")
                continue
            launch=load(launch_path);argv=launch.get("argv",[])
            if (not isinstance(argv,list) or any(not isinstance(v,str) for v in argv) or len(argv)<3 or
                argv[1:3]!=["scripts/sparse_repair_score.py",stage] or launch.get("automatic_retry") is not False):
                raise ValueError("CPU timing launch is not the declared fixed stage")
            def option(name):
                if argv.count(name)!=1 or argv.index(name)+1>=len(argv):raise ValueError("CPU timing launch option missing/duplicated")
                return argv[argv.index(name)+1]
            repo=Path(option("--repo"))
            if not repo.is_absolute() or str(repo/plan["result_root"])!=lease.get("allowed_result_root"):
                raise ValueError("CPU timing launch result root differs from lease")
            if stage in ("preflight","run") and (option("--lease")!=plan["lease_path"] or option("--lease-sha256")!=plan["lease_sha256"]):
                raise ValueError("CPU timing launch lease differs")
            if stage in ("preflight","prepare") and (option("--declaration")!=plan.get("declaration_path") or option("--declaration-sha256")!=declaration_sha):
                raise ValueError("CPU timing launch declaration differs")
            if stage=="prepare" and option("--result-root")!=plan["result_root"]:
                raise ValueError("CPU timing prepare result root differs")
            if stage in ("run","finalize") and option("--plan")!=str(DEFAULT_ROOT/"screen/scoring/scoring_plan.json"):
                raise ValueError("CPU timing launch score plan differs")
            start=launch.get("epoch")
            if type(start) not in (int,float) or not math.isfinite(start) or start<allocation:
                raise ValueError("CPU stage launch timestamp invalid")
            end=None;code=None
            if exit_path.exists():
                exited=load(exit_path);end=exited.get("epoch");code=exited.get("returncode")
                if type(end) not in (int,float) or not math.isfinite(end) or end<start or type(code) is not int:
                    raise ValueError("CPU stage exit timestamp/returncode invalid")
            records.append({"job_id":folder.name,"pod_id":pod_id,"stage":stage,
                "launch_epoch":start,"exit_epoch":end,"elapsed_seconds":end-start if end is not None else None,
                "returncode":code,"status":"EXIT_RECEIPT_PRESENT" if end is not None else "LAUNCH_WITHOUT_EXIT_RECEIPT",
                "launch_path":str(launch_path),"exit_path":str(exit_path) if end is not None else None,
                "launch_sha256":sha(launch_path),"exit_sha256":sha(exit_path) if end is not None else None,
                "plan_path":str(plan_path),"plan_sha256":sha(plan_path),"lease_path":str(lease_path),
                "lease_sha256":sha(lease_path),"pod_receipt_sha256":sha(pod_path),
                "lease_deadline_epoch":deadline,"launch_or_exit_after_lease_deadline":max(start,end or start)>deadline})
    return records


def collect(root, declaration_path):
    root, declaration_path = Path(root), Path(declaration_path)
    sources = []
    def load(path):
        path = Path(path)
        sources.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
        return read(path)
    d = load(declaration_path); declaration_sha = sha(declaration_path)
    if len(d["screen_tasks"]) != 12 or [c["name"] for c in d["conditions"]] != list(CONDITIONS):
        raise ValueError("Frozen screen population/matrix mismatch")
    cases, local_probes, failures, integrity_issues = [], [], [], []
    repo=Path(__file__).resolve().parents[1]
    expected_implementation={p:sha(repo/p) for p in NUMERICAL_FILES}
    contracts={}
    def contract_for(case_folder):
        path=case_folder.parent/"execution_contract.json"
        if str(path) not in contracts:
            contracts[str(path)]=load(path) if path.exists() else None
        return contracts[str(path)]
    calibration_root = root / "calibration_runs"
    for path in sorted(calibration_root.glob("**/calibration.json")):
        case = load(path)
        complete_path = path.parent / "complete.json"
        complete = load(complete_path) if complete_path.exists() else None
        declared_case = next((r for r in d["calibration_tasks"] if r["task_id"] == case.get("task_id")), None)
        verified = bool(declared_case and complete and complete.get("calibration_sha256") == sha(path) and
                        complete.get("declaration_sha256") == declaration_sha and
                        case.get("declaration_sha256") == declaration_sha and
                        case.get("history_sha256") == declared_case["history_sha256"] and
                        complete.get("passed") is case.get("passed"))
        issues=calibration_integrity(case,contract_for(path.parent),expected_implementation)
        if not verified:issues.append("calibration_completion_or_frozen_history_identity_mismatch")
        if issues:integrity_issues.append({"path":str(path),"issues":issues})
        cases.append({**case,"path":str(path),"sha256":sha(path),
            "completion_receipt_verified":verified,"evidence_integrity_verified":not issues,
            "evidence_integrity_issues":issues})
    for name, kind in (("same_query_attention_probes.json", "same_query_attention"),
                       ("local_distribution_probes.json", "matched_prefix_next_token")):
        for path in sorted(calibration_root.glob("**/" + name)):
            value = load(path)
            task=next((dict(r,calibration_seed=d["calibration_seeds"][r["task_id"]]) for r in d["calibration_tasks"]
                       if r["task_id"].replace("/","__")==path.parent.name),None)
            trace_path=path.parent/"native_D_trace.json"
            trace=load(trace_path) if trace_path.exists() else None
            issues=probe_integrity(value,kind,task,declaration_sha,contract_for(path.parent),trace,expected_implementation)
            if issues:integrity_issues.append({"path":str(path),"issues":issues})
            local_probes.append({"path":str(path),"kind":kind,"task_folder":path.parent.name,
                "task_id":task["task_id"] if task else None,"records":value,
                "evidence_integrity_verified":not issues,"evidence_integrity_issues":issues,
                "not_a_numerical_control_gate":True})
    for path in sorted(calibration_root.glob("**/failure.json")):
        failures.append({"path": str(path), **load(path)})
    # Include explicitly partial launch/runtime/status evidence, never invent a
    # completed calibration from a running sampler progress file.
    calibration_status = []
    for name in ("runtime.json", "execution_contract.json", "model_setup.json", "status.json"):
        for path in sorted(calibration_root.glob("**/" + name)):
            calibration_status.append({"path": str(path), "kind": name, "record": load(path)})
    rows = []
    for task in d["screen_tasks"]:
        tid = task["task_id"]
        for condition in CONDITIONS:
            for seed in d["seeds"][tid]:
                folder = root / "screen/tasks" / tid.replace("/", "__") / condition / f"seed_{seed['seed_index']}"
                row = {"task_id": tid, "condition": condition, "seed_index": seed["seed_index"],
                    "answer_seed": seed["answer_seed"], "status": "not_started", "passed": None, "score_missing": None,
                    "category": None, "answer_tokens": None, "answer_ended_eos": None, "answer_capped": None,
                    "answer_path": None, "answer_sha256": None, "timing": {}, "working_set": None, "memory": None,
                    "historical_reasoning_positions": task["historical_reasoning_positions"]}
                if (folder / "complete.json").exists():
                    done = load(folder / "complete.json")
                    expected = {"task_id": tid, "condition": condition, **seed, "declaration_sha256": declaration_sha}
                    if any(done["identity"].get(k) != v for k, v in expected.items()):
                        raise ValueError("Committed draw identity differs from frozen task/draw")
                    for name, expected_sha in done["files"].items():
                        path = scoped(folder, name)
                        if sha(path) != expected_sha:
                            raise ValueError("Committed generation artifact hash mismatch: " + str(path))
                        sources.append({"path": str(path), "bytes": path.stat().st_size, "sha256": expected_sha})
                    answer_path = folder / "answer.json"; raw = load(answer_path)
                    if any(raw.get(k) != v for k, v in expected.items()):
                        raise ValueError("Answer does not match frozen draw")
                    work = load(folder / "working_set.json")
                    if work.get("semantic_trajectory_union_complete") is not True:
                        raise ValueError("Completed draw lacks complete semantic working-set trace")
                    row.update(status="generated_unscored", answer_tokens=len(raw["answer_ids"]),
                        answer_ended_eos=raw["answer_ended_eos"], answer_capped=raw["answer_capped"],
                        answer_path=str(answer_path.relative_to(root)), answer_sha256=sha(answer_path),
                        timing={k: v for k, v in raw.items() if k.endswith("_seconds") or k in
                            ("sampler_timing", "shared_task_cache_setup_seconds", "timing_limitations", "trace_limitations")},
                        working_set=work["summary"], memory=raw.get("memory"),
                        shared_task_backing_bytes=raw.get("shared_task_backing_bytes"),
                        trace_resume_limitations=raw.get("trace_limitations"))
                    completion_name = "sampler/completion_timing.json"
                    if completion_name in done["files"]:
                        row["timing"]["sampler_completion_timing"] = load(folder / completion_name)
                elif (folder / "sampler/resume.json").exists() or (folder / "failure.json").exists():
                    row["status"] = "generation_incomplete"
                    if (folder / "failure.json").exists():
                        row["generation_failure"] = load(folder / "failure.json")
                rows.append(row)
    task_execution, cache_setup_attempts = [], []
    for task in d["screen_tasks"]:
        folder = root / "screen/tasks" / task["task_id"].replace("/", "__")
        for attempt_path in sorted((folder / "cache_setup_attempts").glob("*.json")):
            attempt = load(attempt_path)
            if attempt.get("task_id") != task["task_id"] or attempt.get("declaration_sha256") != declaration_sha:
                raise ValueError("Cache setup attempt differs from frozen task/declaration")
            cache_setup_attempts.append({"path":str(attempt_path), "sha256":sha(attempt_path),
                "task_id":task["task_id"], "timing":attempt.get("timing",{}),
                "backing_bytes":attempt.get("backing_bytes",{})})
        receipt_path = folder / "task_complete.json"
        if not receipt_path.exists():
            continue
        receipt = load(receipt_path)
        expected_paths = {f"{condition}/seed_{seed['seed_index']}/complete.json"
            for condition in CONDITIONS for seed in d["seeds"][task["task_id"]]}
        if (receipt.get("task_id") != task["task_id"] or receipt.get("declaration_sha256") != declaration_sha or
            receipt.get("completed_records") != 42 or receipt.get("expected_records") != 42 or
            receipt.get("backing_cache_fingerprints_unchanged") is not True or
            len(receipt.get("files", [])) != 42 or {x.get("path") for x in receipt["files"]} != expected_paths):
            raise ValueError("Task timing receipt does not bind exact completed frozen task")
        for item in receipt["files"]:
            if sha(scoped(folder, item["path"])) != item["sha256"]:
                raise ValueError("Task timing receipt draw hash mismatch")
        task_execution.append({"task_id":task["task_id"], "path":str(receipt_path),
            "receipt_sha256":sha(receipt_path), "completed_records":42,
            "cache_setup_timing":receipt.get("cache_setup_timing",{}),
            "this_attempt_whole_task_wall_seconds":receipt.get("this_attempt_whole_task_wall_seconds"),
            "backing_cache_fingerprints_unchanged":True})
    lookup = {(r["task_id"], r["condition"], r["seed_index"]): r for r in rows}
    closure_path = root / "screen/generation_closure.json"
    closure = load(closure_path) if closure_path.exists() else None
    if closure is not None:
        content = {k: v for k, v in closure.items() if k != "closure_sha256"}
        if (closure.get("closure_sha256") != digest(content) or closure.get("expected_answers") != 504 or
                closure.get("declaration_sha256") != declaration_sha or closure.get("all_screen_generation_complete") is not True):
            raise ValueError("Invalid full generation closure")
        keys = []
        for item in closure["answers"]:
            key = tuple(item["contract"][k] for k in ("task_id", "condition", "seed_index"))
            keys.append(key)
            if key not in lookup or lookup[key]["answer_sha256"] != item["answer_sha256"] or lookup[key]["answer_path"] != item["path"]:
                raise ValueError("Sealed answer bytes/population mismatch")
        if len(keys) != 504 or set(keys) != set(lookup):
            raise ValueError("Full generation seal must cover all504 draws")
    manifest_path = root / "screen/scoring/scored_answer_manifest.json"
    manifest = load(manifest_path) if manifest_path.exists() else None
    if manifest is not None:
        if (closure is None or manifest.get("generation_closure_sha256") != closure["closure_sha256"] or
                manifest.get("expected_answers") != 504 or manifest.get("committed_answers") != 504 or
                manifest.get("raw_answers_unchanged") is not True or
                manifest.get("hidden_tests_loaded_after_complete_screen_generation") is not True):
            raise ValueError("Scoring manifest lacks exact complete generation barrier")
        plan_path = root / "screen/scoring/scoring_plan.json"
        if sha(plan_path) != manifest["scoring_plan_sha256"]:
            raise ValueError("Scoring plan hash differs")
        plan = load(plan_path)
        if (plan.get("declaration_sha256") != declaration_sha or
                plan.get("scorer_identity", {}).get("scorer_version") != "gearshift_scorer_v2_20260919_01"):
            raise ValueError("Scoring plan uses wrong declaration or unrepaired scorer")
        scored = set()
        for item in manifest["files"]:
            key = item["task_id"], item["condition"], item["seed_index"]
            if key in scored or key not in lookup:
                raise ValueError("Duplicate/unexpected scored draw")
            scored.add(key); row = lookup[key]
            path = scoped(root, item["path"])
            if (sha(path) != item["sha256"] or row["answer_path"] != item["answer_path"] or
                    row["answer_sha256"] != item["answer_sha256"] or row["answer_seed"] != item["answer_seed"]):
                raise ValueError("Scored answer identity/hash differs")
            receipt = load(path); score = receipt["score_v2"]
            if (score.get("passed") is not item["passed"] or score.get("missing") is not item["missing"] or
                    score.get("scorer_identity") != plan["scorer_identity"] or score.get("category") != item["category"] or
                    type(score.get("missing")) is not bool or score["missing"] is not (score["passed"] is None) or
                    (score["passed"] is not None and type(score["passed"]) is not bool)):
                raise ValueError("Invalid repaired-score outcome/missingness")
            row.update(status="scoring_missing" if score["missing"] else "scored", passed=score["passed"],
                score_missing=score["missing"], category=score.get("category"), score_path=item["path"], score_sha256=item["sha256"],
                cpu_score_accounting=score_test_accounting(score))
        if scored != set(lookup):
            raise ValueError("Scoring manifest population incomplete")
    unmanifested_scores = len(list((root / "screen/scoring/scores").glob("*/score.json"))) if manifest is None else 0
    extra = {}
    for name in ("cost_ledger.json", "economics/summary.json", "preflight/studio_ready.json", "preflight/gpu_launch.json",
                 "preflight/screen_launch.json", "preflight/screen_all_workers_generating.json",
                 "recovery/decision.json", "recovery/resume_stage.json", "recovery/resume_launch.json",
                 "recovery/recovery_validation.json"):
        if (root / name).exists():
            extra[name] = load(root / name)
    interrupted = []
    for path in sorted((root / "recovery/original_interrupted_state").glob("**/sampler/resume.json")):
        original = load(path); folder = path.parent.parent
        identity_path = folder / "draw_identity.json"
        identity = load(identity_path) if identity_path.exists() else {}
        record = {"path":str(path), "sha256":sha(path),
            "identity":{k:identity.get(k) for k in ("task_id","condition","seed_index","answer_seed","declaration_sha256")},
            "state":original.get("state"), "committed_tokens":len(original.get("tokens",[])),
            "forwarded_tokens":original.get("forwarded_tokens"), "resume_sha256":original.get("resume_sha256"),
            "logits_sha256":original.get("logits_sha256"), "timing":original.get("timing",{})}
        failure_path = folder / "failure.json"
        if failure_path.exists():
            failure = load(failure_path)
            record["failure"] = {k:failure.get(k) for k in ("exception","error","this_attempt_wall_seconds")}
        progress_path = folder / "progress_working_set.json"
        if progress_path.exists():
            progress = load(progress_path)
            record["interrupted_progress_summary_not_complete_answer_union"] = progress.get("summary",{})
        interrupted.append(record)
    launches = [{"path": str(p), "record": load(p)} for p in sorted((root / "resources").glob("*/first_calibration_launch.json"))]
    cpu_stage_intervals=collect_cpu_stage_intervals(root,declaration_sha,load)
    unique_sources = {r["path"]: r for r in sources}
    return {"schema": "gearshift.sparse_repair.report_inputs.v1", "declaration_sha256": declaration_sha,
        "declaration": d, "calibration_cases": cases, "calibration_failures": failures,
        "calibration_status": calibration_status, "local_probes": local_probes, "evidence_integrity_issues":integrity_issues,
        "collected_numerical_implementation_hashes":expected_implementation,
        "draws": rows, "generation_sealed": closure is not None, "scoring_manifest_present": manifest is not None,
        "task_execution_records": task_execution,
        "task_cache_setup_attempts": cache_setup_attempts, "original_interrupted_sampler_metadata": interrupted,
        "cpu_wrapper_stage_intervals":cpu_stage_intervals,
        "unmanifested_score_receipts_not_used_for_quality": unmanifested_scores, "supplements": extra,
        "launches": launches, "source_inputs": [unique_sources[k] for k in sorted(unique_sources)],
        "candidate_execution_performed": False, "model_execution_performed": False, "private_tests_read": False}


def metric(values):
    values = [v for v in values if type(v) in (float, int) and math.isfinite(v)]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def cpu_accounting(data):
    scores=[r["cpu_score_accounting"] for r in data["draws"] if r.get("cpu_score_accounting") is not None]
    result={"manifest_bound_score_records_with_accounting":len(scores),
        "total_test_receipts":sum(r["test_receipt_count"] for r in scores),
        "test_receipts_with_multiple_attempts":sum(r["test_receipts_with_multiple_attempts"] for r in scores),
        "wrapper_stage_intervals":data.get("cpu_wrapper_stage_intervals",[]),
        "timing_scope":"Recorded final test-receipt CPU/wall durations; omitted earlier retry durations and missing timing are not zero. Stage intervals are separate elapsed wall time, not additive to test durations or pod billing."}
    for name in ("executed_tests","total_tests"):
        values=[r[name] for r in scores if r[name] is not None]
        result[name]={"score_records_with_count":len(values),"score_records_without_count":len(scores)-len(values),
            "sum":sum(values) if values else None}
    for name in ("recorded_final_test_cpu_seconds","recorded_final_test_wall_seconds"):
        measured=[r[name] for r in scores if r[name]["n"]]
        result[name]={"test_receipts_with_duration":sum(r["n"] for r in measured),
            "test_receipts_without_duration":sum(r[name]["test_receipts_without_duration"] for r in scores),
            "sum":sum(r["sum"] for r in measured) if measured else None}
    stages=result["wrapper_stage_intervals"]
    result["wrapper_stage_coverage"]={"launch_receipts":len(stages),
        "exit_receipts":sum(r["exit_epoch"] is not None for r in stages),
        "jobs":len({(r["pod_id"],r["job_id"]) for r in stages})}
    return result


def cluster_statistics(rows, task_ids):
    """Full frozen population only; exact task resampling keeps all3 draws."""
    import numpy as np
    lookup = {(r["task_id"], r["condition"], r["seed_index"]): r for r in rows}
    expected = {(tid, c, s) for tid in task_ids for c in CONDITIONS for s in range(3)}
    if len(rows) != len(expected) or set(lookup) != expected or any(type(r["passed"]) is not bool for r in rows):
        raise ValueError("Cluster statistics require all12tasks ×14conditions ×3 binary outcomes; no subset")
    counts = np.array([[sum(lookup[tid, c, s]["passed"] for s in range(3)) for c in CONDITIONS] for tid in task_ids], dtype=np.int64)
    indices = np.random.Generator(np.random.PCG64(20260921)).integers(0, len(task_ids), size=(10000, len(task_ids)))
    boot = counts[indices].sum(axis=1)
    denominator = len(task_ids)*3
    conditions = {c: {"passed": int(counts[:, j].sum()), "draws": denominator,
        "pass_rate": float(counts[:, j].sum()/denominator),
        "task_cluster_ci95": np.quantile(boot[:, j]/denominator, [.025, .975], method="linear").tolist()} for j, c in enumerate(CONDITIONS)}
    pairs = [("R_10", "H"), ("R_10", "R_random"), ("R_10", "R_recent")]
    pairs += [(f"N_{p}", f"M_{p}") for p in (5, 10, 25)]
    pairs += [(f"R_{p}", "H") for p in (5, 25)]
    contrasts = {}
    per_task = []
    for a, b in pairs:
        i, j = CONDITIONS.index(a), CONDITIONS.index(b)
        change = counts[:, i]-counts[:, j]
        contrasts[a+"-"+b] = {"difference": float(change.sum()/denominator),
            "task_cluster_ci95": np.quantile((boot[:, i]-boot[:, j])/denominator, [.025, .975], method="linear").tolist(),
            "tasks_improved": int((change>0).sum()), "tasks_worse": int((change<0).sum()), "tasks_tied": int((change==0).sum())}
        for n, tid in enumerate(task_ids):
            per_task.append({"task_id": tid, "contrast": a+"-"+b, "draws_per_condition": 3,
                "left_passes": int(counts[n, i]), "right_passes": int(counts[n, j]), "paired_difference": float(change[n]/3),
                "paired_draw_changes": [int(lookup[tid,a,s]["passed"])-int(lookup[tid,b,s]["passed"]) for s in range(3)]})
    return {"conditions": conditions, "contrasts": contrasts, "per_task": per_task,
            "bootstrap": {"rng": "numpy.random.Generator(numpy.random.PCG64(20260921))", "resamples": 10000,
                "seed": 20260921, "unit": "task cluster; all3 draws retained", "quantile_method": "linear",
                "resample_indices_sha256_int64_C_order": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                "interval_interpretation": "exploratory, unadjusted descriptive95%; not equivalence/generalization"}}


def working_set(row):
    w = row.get("working_set") or {}; layers = w.get("layers", {})
    capacities = [len(v["cumulative_pairs_by_kv_head"])*row["historical_reasoning_positions"] for v in layers.values()]
    capacity = sum(capacities)
    pairs = w.get("cumulative_unique_pairs_all_layers")
    return {"task_id": row["task_id"], "condition": row["condition"], "seed_index": row["seed_index"],
        "selected_pair_accounting_applicable": bool(layers),
        "controller_selected_pair_counter_raw": pairs,
        "nominal_per_head_fraction": w.get("fraction") if layers else None,
        "cumulative_unique_pairs": pairs if layers else None, "reasoning_pair_capacity": capacity if capacities else None,
        "cumulative_reasoning_fraction": pairs/capacity if pairs is not None and capacity else None,
        "peak_instantaneous_pairs_all_layers": sum(v["peak_instantaneous_selected_pairs"] for v in layers.values()) if layers else None,
        "peak_instantaneous_reasoning_pair_fraction": sum(v["peak_instantaneous_selected_pairs"] for v in layers.values())/capacity if capacity else None,
        "cumulative_pair_fraction_by_layer_and_kv_head": {k:[n/row["historical_reasoning_positions"] for n in v["cumulative_pairs_by_kv_head"]]
            for k,v in layers.items()} if row["historical_reasoning_positions"] else {},
        "cumulative_logical_selected_bytes": w.get("cumulative_logical_selected_bytes_all_layers") if layers else None,
        "native_history_bytes": w.get("frozen_native_history_bytes_exact"),
        "mapped_history_bytes": w.get("frozen_mapped_history_bytes_exact"),
        "cumulative_per_head_page_bytes_estimated": sum(v["cumulative_per_head_page_bytes_estimated"] for v in layers.values()) if layers else None,
        "cumulative_all_heads_shared_page_bytes_estimated": sum(v["cumulative_all_heads_shared_page_bytes_estimated"] for v in layers.values())
            if layers and all("cumulative_all_heads_shared_page_bytes_estimated" in v for v in layers.values()) else None,
        "cumulative_union_metadata_bytes": sum(v["cumulative_union_metadata_bytes"] for v in layers.values())
            if layers and all("cumulative_union_metadata_bytes" in v for v in layers.values()) else None,
        "per_layer_peak_buffer_sizes_logical_tensor_bytes":w.get("per_layer_peak_buffer_sizes",{}),
        "shared_task_backing_bytes":row.get("shared_task_backing_bytes"),
        "draw_allocator_snapshot":row.get("memory"),
        "physical_memory_traffic_measured": False}


def summarize(data):
    d = data["declaration"]; tids = [r["task_id"] for r in d["screen_tasks"]]
    rows = data["draws"]
    expected = {(tid,c,s) for tid in tids for c in CONDITIONS for s in range(3)}
    if len(tids) != 12 or len(set(tids)) != 12 or len(rows) != 504 or {(r["task_id"],r["condition"],r["seed_index"]) for r in rows} != expected:
        raise ValueError("Snapshot changed frozen population")
    counts = dict(collections.Counter(r["status"] for r in rows))
    generated = sum(r["status"] in ("generated_unscored", "scored", "scoring_missing") for r in rows)
    scored = sum(r["status"] == "scored" for r in rows)
    cal = data["calibration_cases"]
    passed_ids = sorted({r["task_id"] for r in cal if r.get("completion_receipt_verified") and r.get("evidence_integrity_verified",True) and r.get("passed") is True})
    failed_ids = sorted({r["task_id"] for r in cal if r.get("completion_receipt_verified") and r.get("passed") is False})
    all_four_passed = set(passed_ids) == {r["task_id"] for r in d["calibration_tasks"]}
    integrity_issues=data.get("evidence_integrity_issues",[])
    complete = not integrity_issues and generated == 504 and scored == 504 and data["generation_sealed"] and data["scoring_manifest_present"] and all_four_passed
    stats = cluster_statistics(rows, tids) if complete else None
    ws = [working_set(r) for r in rows if r["working_set"] is not None]
    diagnostics = {}
    for condition in CONDITIONS:
        rs = [r for r in rows if r["condition"] == condition]
        known = [r for r in rs if type(r["passed"]) is bool]
        successes = sum(r["passed"] for r in known)
        diagnostics[condition] = {"expected": 36, "generated": sum(r["answer_tokens"] is not None for r in rs),
            "scored_binary": len(known), "scoring_infrastructure_missing": sum(r["score_missing"] is True for r in rs),
            "passed": successes if known else None, "pass_rate_full_population": successes/36 if len(known)==36 else None,
            "possible_success_bounds_including_all_unmeasured": [successes/36,(successes+36-len(known))/36],
            "categories": dict(collections.Counter(r["category"] for r in known if r["category"] is not None)),
            "failure_taxonomy": {"syntax": sum(r["category"] == "syntax" for r in known) if known else None,
                "runtime": sum(r["category"] == "runtime_error" for r in known) if known else None,
                "assertion": sum(r["category"] == "test_assertion" for r in known) if known else None,
                "timeout": sum("timeout" in str(r["category"]) for r in known) if known else None,
                "interface": None, "interface_note": "Frozen scorer does not separately classify interface failures; they may appear among runtime/assertion failures. No post-hoc diagnosis is invented."},
            "answer_tokens": metric(r["answer_tokens"] for r in rs),
            "EOS": sum(r["answer_ended_eos"] is True for r in rs), "capped": sum(r["answer_capped"] is True for r in rs),
            "answer_active_seconds": metric(r["timing"].get("answer_seconds") for r in rs),
            "actual_attempt_whole_draw_seconds": metric(r["timing"].get("this_attempt_whole_draw_wall_seconds") for r in rs),
            "sampler_completion_overhead_seconds": {key:metric(
                r["timing"].get("sampler_completion_timing",{}).get("overhead",{}).get(key) for r in rs)
                for key in ("checkpoint_persistence_seconds","checkpoint_preparation_seconds","telemetry_seconds",
                            "progress_publish_seconds","result_materialization_seconds","identity_binding_seconds")},
            "sampler_completion_overhead_scope_complete":sum(r["timing"].get("sampler_completion_timing",{}).get("overhead",{}).get("scope_complete") is True for r in rs),
            "cumulative_reasoning_fraction": metric(w["cumulative_reasoning_fraction"] for w in ws if w["condition"]==condition),
            "instantaneous_reasoning_fraction": metric(w["peak_instantaneous_reasoning_pair_fraction"] for w in ws if w["condition"]==condition),
            "cumulative_per_head_page_bytes_estimated":metric(w["cumulative_per_head_page_bytes_estimated"] for w in ws if w["condition"]==condition),
            "cumulative_all_heads_shared_page_bytes_estimated":metric(w["cumulative_all_heads_shared_page_bytes_estimated"] for w in ws if w["condition"]==condition),
            "peak_GPU_allocated_bytes": metric((r["memory"] or {}).get("peak_allocated_bytes") for r in rs)}
    probes = collections.defaultdict(list)
    for source in data["local_probes"]:
        if source["kind"] != "same_query_attention" or not source.get("evidence_integrity_verified",True):
            continue
        for row in source["records"]:
            probes[f"{row['mode']}_{row['fraction']}"] .append(row)
    probe_summary = {key: {"observations":len(items),
        "attention_output_relative_L2":metric(r.get("attention_output_relative_l2") for r in items),
        "attention_output_max_abs":metric(r.get("attention_output_max_abs") for r in items),
        "retained_reasoning_mass_ratio":metric(sum(r["selected_mass_sum_per_kv_group"])/sum(r["all_reasoning_mass_sum_per_kv_group"])
            if sum(r["all_reasoning_mass_sum_per_kv_group"])>0 else None for r in items)} for key,items in sorted(probes.items())}
    return {"schema":"gearshift.sparse_repair.summary.v1", "declaration_sha256":data["declaration_sha256"],
        "status":"EVIDENCE_INTEGRITY_BLOCKER" if integrity_issues else ("COMPLETE_EXPLORATORY_SCREEN" if complete else "INCOMPLETE_DIAGNOSTIC_NO_COMPLETE_SCREEN_QUALITY"),
        "evidence_integrity_issues":integrity_issues,
        "coverage":{"planned_tasks":12,"planned_draws":504,"generated_complete_draws":generated,"scored_binary_draws":scored,
            "not_generated_complete_draws":504-generated,"draw_status_counts":counts,
            "calibration_completed_receipts":sum(r["completion_receipt_verified"] for r in cal),"calibration_result_records":len(cal),"calibration_passed_task_ids":passed_ids,"calibration_failed_task_ids":failed_ids,
            "calibration_required_task_ids":[r["task_id"] for r in d["calibration_tasks"]],
            "all_four_calibration_tasks_have_verified_pass":all_four_passed,
            "not_run_is_not_quality_failure":True},
        "calibration_cases":cal,"calibration_failures":data["calibration_failures"],
        "calibration_status":data["calibration_status"],"local_probe_summary":probe_summary,
        "local_distribution_probes":[x for x in data["local_probes"] if x["kind"]=="matched_prefix_next_token"],
        "statistics":stats,"condition_diagnostics":diagnostics,"working_sets":ws,"supplements":data["supplements"],
        "cpu_scoring_accounting":cpu_accounting(data),
        "task_execution_records":data.get("task_execution_records",[]),
        "task_cache_setup_attempts":data.get("task_cache_setup_attempts",[]),
        "original_interrupted_sampler_metadata":data.get("original_interrupted_sampler_metadata",[]),
        "resumed_draw_timing":[{"task_id":r["task_id"],"condition":r["condition"],"seed_index":r["seed_index"],
            "sampler_resume_count":r["timing"].get("sampler_timing",{}).get("resume_count"),
            "known_sampler_reconstruction_seconds":r["timing"].get("sampler_timing",{}).get("reconstruction_seconds"),
            "trace_rebuild_of_completed_sampler_seconds":r["timing"].get("trace_rebuild_of_completed_sampler_seconds"),
            "last_attempt_whole_draw_wall_seconds":r["timing"].get("this_attempt_whole_draw_wall_seconds"),
            "trace_limitations":r.get("trace_resume_limitations",r["timing"].get("trace_limitations"))}
            for r in rows if (r["timing"].get("sampler_timing",{}).get("resume_count") or 0)>0 or
            (r["timing"].get("trace_rebuild_of_completed_sampler_seconds") or 0)>0],
        "limitations":["Already-inspected development data, 12 task clusters, not 36 independent tasks or fresh confirmation.",
            "Unrun/unscored draws are not failed candidates; missing outcomes retain frozen denominators.",
            "No incomplete-data bootstrap or favorable native-success subset is used.",
            "Exact native oracle caches plus dense scoring were paid for; masked dense SDPA does not establish sparse speed.",
            "Sparse training-loss positions, sparse attention reads, and sparse KV construction are different interventions; evidence for one does not validate the others.",
            "A retained native KV entry already encodes earlier-token dependencies through preceding model layers. Selecting it for reading does not show it can be constructed from the corresponding raw token alone.",
            "Free-running arms share a selector rule, not necessarily selected indices: once their queries diverge, their layer/head/step selections may differ. Only the explicitly matched-query probes hold the query fixed.",
            "Full repair is controlled against matched native/native splice, not guaranteed identical to D: independently prefilling the prompt can change execution shape and numerical rounding.",
            "No jacq kernel or serving speedup is transferred to this experiment; its different workload, kernel, batching and integration regime is not measured here.",
            "Logical tensor/page bookkeeping is not measured HBM traffic or allocator-page reclamation.",
            "Selected fractions count historical reasoning (layer, KV-head, position) pairs, not global token unions. Original prompt and current/own-answer state remain outside the budget; full-history backing bytes include prompt state.",
            "Dense D/H have no selector-pair accounting: a zero raw selector counter is not zero attention reads or zero history memory. P has no reasoning intervention, but the shared diagnostic task still retains other arms' backing caches.",
            "Page estimates assume 16 absolute positions, either independent per-KV-head pages or pages shared across all KV heads; padded and boundary pages can exceed raw selected-row bytes. Neither layout is measured allocation or traffic.",
            "Per-layer KV-group mask bytes omit the expanded query-head mask and combined attention-mask scratch; score/softmax and gather scratch are not fully itemized. Tensor-shape bytes are not observed HBM traffic.",
            "Shared task backing contains native, mapped, hybrid, prompt-only and independent prompt caches. source_peak_then_freed is an earlier construction footprint, not simultaneously resident answer-state bytes; model weights and allocator overhead remain separate.",
            "Shared cache setup is charged once per task attempt, not 42 times from its repeated draw metadata. Draw-attempt wall stops before final working-set/answer/complete JSON writes; task-attempt wall includes those writes but excludes worker startup, old attempts and task-receipt persistence.",
            "Instrument stage native_key_scan combines current-query key scoring and probability normalization; replacement_and_mask combines gathers, replacements, dense-buffer construction and masks. Those components are not separately timed.",
            "Per-layer temporary-buffer maxima are not concurrent whole-model temporary peak; CUDA allocator peak includes different live allocations and is recorded separately.",
            "Unprofiled whole-answer wall time still includes CPU cumulative-union accounting and durable sampler I/O; it is diagnostic implementation cost, not bare model latency.",
            "Repeated16-token calibration timings distinguish unsynchronized whole-run measurements from synchronized stage profiling; neither is a complete-answer end-to-end cost proof.",
            "Calibration teacher-forced local probes are not free-running program quality.",
            "Sampler resumes reconstruct committed masks by replay; uncommitted crash-tail selections and timing may be missing."],
        "generation_sealed":data["generation_sealed"],"scoring_manifest_present":data["scoring_manifest_present"],
        "unmanifested_score_receipts_not_used_for_quality":data["unmanifested_score_receipts_not_used_for_quality"],
        "model_execution_or_candidate_execution_performed_by_reporter":False}


def render(s, input_sha, decision_review_path=None):
    c=s["coverage"]; full=s["statistics"] is not None
    lines=["# Sparse-state / selective-repair diagnostic", "", "**"+s["status"]+"**", "",
        f"Complete generated answers: **{c['generated_complete_draws']}/504**. Binary executed-code outcomes: **{c['scored_binary_draws']}/504**. "
        f"Verified passing calibration tasks: **{len(c['calibration_passed_task_ids'])}/4**. Unrun draws are **not** quality failures.", "",
        "This is the bounded exploratory screen on 12 previously inspected development tasks, three fixed draws per condition. It is not a new confirmation set. The original results and publication tag remain separate.", "",
        "## Actual coverage and numerical controls", "",
        "Draw states: `"+json.dumps(c["draw_status_counts"],sort_keys=True)+"`. "
        f"Generation seal present: {s['generation_sealed']}; complete scoring manifest present: {s['scoring_manifest_present']}.", "",
        "| Calibration task / receipt | Completed and hash-bound | Reported passed | Actual case wall seconds |", "|---|---:|---:|---:|"]
    if not s["calibration_cases"]:
        lines.append("| No completed calibration receipt available | No | Unavailable | Unavailable |")
    for case in s["calibration_cases"]:
        lines.append(f"| {case.get('task_id','unknown')} (`{case['path']}`) | {case['completion_receipt_verified']} | {case.get('passed')} | {case.get('actual_calibration_wall_seconds','unmeasured')} |")
    readiness=s["supplements"].get("preflight/studio_ready.json",{})
    launch=s["supplements"].get("preflight/gpu_launch.json",{})
    lines += ["", f"Handoff status: {readiness.get('status','unverified in collected inputs')}. Calibration/GPU launch receipt status: {launch.get('status','unverified in collected inputs')}. A calibration launch is not evidence that the 504-answer screen launched.", ""]
    screen_launch=s["supplements"].get("preflight/screen_launch.json")
    all_generating=s["supplements"].get("preflight/screen_all_workers_generating.json")
    for name,receipt,field in (("screen_launch.json",screen_launch,"workers"),
                               ("screen_all_workers_generating.json",all_generating,"observations")):
        if receipt is None:
            lines.append(f"- `{name}`: unavailable in this snapshot; no screen-generation claim is inferred from calibration or allocation alone.")
            continue
        observations=receipt.get(field,[])
        workers={r.get("name") for r in observations if r.get("name")}
        observed={r.get("name") for r in observations if r.get("name") and
            r.get("worker_status",{}).get("role")=="sparse_repair_screen" and
            r.get("worker_status",{}).get("stage")=="answer_generation" and
            type(r.get("worker_status",{}).get("generated_tokens")) is int and
            r["worker_status"]["generated_tokens"]>0}
        lines.append(f"- `{name}`: receipt epoch={receipt.get('epoch','unrecorded')}; "
            f"status={receipt.get('status','no status string')}; actual answer-generation progress observed for "
            f"{len(observed)}/{len(workers)} named workers at the recorded observations. These are historical launch/progress receipts, not a current liveness check or a completion seal.")
    lines += ["", "Launching workers, observing generated tokens, or passing calibration does not establish task quality. No complete-screen quality conclusion is available until all 504 frozen answers have sealed generation, valid binary scores, and the required evidence-integrity checks.", "",
        "Calibration completion discovery uses calibration_runs/** raw allocation records, not canonical calibration/ copies used by the screen gate. Complete raw run receipts must be collected before reporting.", "",
        "Same-execution-shape controls require bit-exact logits; altered-shape rounding is separately reported and does not excuse same-path errors. No tolerance is widened here.", ""]
    recovery_names=("recovery/decision.json","recovery/resume_stage.json","recovery/resume_launch.json","recovery/recovery_validation.json")
    if any(name in s["supplements"] for name in recovery_names) or s["original_interrupted_sampler_metadata"]:
        lines += ["### Infrastructure interruption and continuation", "",
            "Final generation/scoring coverage does not erase an earlier deadline interruption. The following are separately hash-bound historical recovery records, not new outcomes or permission to retry an answer based on quality. Staging does not imply launch; process launch does not imply successful replay or completed generation.", ""]
        decision=s["supplements"].get("recovery/decision.json")
        if decision:
            draw=decision.get("interrupted_draw",{})
            lines += [f"The saved decision records **{decision.get('global_complete_answers','unrecorded')}/{decision.get('expected_answers','unrecorded')}** completed answers before continuation, "
                f"with {decision.get('complete_draws_in_pending_task','unrecorded')} completed draws in the pending task(s) {decision.get('pending_task_ids',[])}. "
                f"Interrupted draw: {draw.get('condition','unrecorded')} / seed index {draw.get('seed_index','unrecorded')}, "
                f"{draw.get('committed_tokens','unrecorded')} committed tokens and {draw.get('forwarded_tokens','unrecorded')} forwarded tokens.", "",
                "Recorded reason: " + str(decision.get("reason","not recorded")), ""]
        for name in recovery_names:
            receipt=s["supplements"].get(name)
            if receipt is None:
                lines.append(f"- `{name}`: unavailable at collection; the corresponding recovery event is not inferred.")
            else:
                fields={key:value for key,value in receipt.items() if value is None or type(value) in (str,int,float,bool)}
                lines += [f"- `{name}` — recorded scalar fields (full receipt retained in `summary.json`):",
                    "  ```json", "  " + json.dumps(fields,sort_keys=True), "  ```"]
        for record in s["original_interrupted_sampler_metadata"]:
            ident=record["identity"]
            lines.append(f"- Preserved checkpoint `{ident.get('task_id')}` / `{ident.get('condition')}` / seed {ident.get('seed_index')}: "
                f"state={record.get('state')}; committed={record.get('committed_tokens')}; forwarded={record.get('forwarded_tokens')}; "
                f"known active seconds at that checkpoint={record.get('timing',{}).get('active_seconds','unmeasured')}. "
                "Its interrupted progress union is not a complete-answer working set.")
        attempts=collections.Counter(r["task_id"] for r in s["task_cache_setup_attempts"])
        repeated={tid:n for tid,n in attempts.items() if n>1}
        lines += ["", "Collected per-task cache-setup attempt counts greater than one: `"+json.dumps(repeated,sort_keys=True)+"`. "
            "Individual setup timings/backing sizes are retained in `summary.json`; an empty map means no duplicate setup receipt is available in this snapshot, not proof that recovery was free.", ""]
        if s["resumed_draw_timing"]:
            lines += ["| Resumed draw | Sampler resume count | Known sampler reconstruction s | Complete-sampler trace rebuild s | Last draw-attempt wall s |",
                "|---|---:|---:|---:|---:|"]
            for record in s["resumed_draw_timing"]:
                lines.append(f"| {record['task_id']} / {record['condition']} / {record['seed_index']} | {record['sampler_resume_count']} | "
                    f"{record['known_sampler_reconstruction_seconds']} | {record['trace_rebuild_of_completed_sampler_seconds']} | {record['last_attempt_whole_draw_wall_seconds']} |")
        else:
            lines.append("No completed resumed-draw timing is available yet; staging/launch receipts are not substituted for measured replay cost.")
        lines += ["", "**Recovery accounting limits:** duplicated cache/model setup, rehashing and replay are real continuation work, not a free retry. The last task/draw-attempt wall does not include all earlier interrupted attempts; conversely, its measured replay is already inside that attempt wall and must not be added again. Known sampler completion overhead and saved original checkpoint timing remain separate. Semantic selection unions can be reconstructed along committed token history, but discarded/uncommitted crash-tail selections and timing may remain incomplete; later 504/504 completion does not retroactively measure them. Completed draw byte identity, committed-prefix preservation and reconstruction-logit fingerprint verification require the explicit validation receipt, not inference from a successful clone, stage, launch or final count.", ""]
    if s.get("evidence_integrity_issues"):
        lines += ["**Evidence-integrity blocker:** mismatched records remain in the snapshot but do not support a passed gate or complete-screen scientific claim. Invalid local probes are explicitly excluded from aggregate averages.", ""]
        for issue in s["evidence_integrity_issues"]:
            lines.append(f"- `{issue['path']}`: " + "; ".join(issue["issues"]))
    for case in s["calibration_cases"]:
        for name, control in case.get("controls",{}).items():
            positions=control.get("positions",[])
            lines.append(f"- `{case.get('task_id')}` / `{name}`: passed={control.get('passed')}; positions={len(positions)}; max absolute logit difference={metric(r.get('max_abs') for r in positions)['max']}; max KL={metric(r.get('kl') for r in positions)['max']}.")
    altered=[case for case in s["calibration_cases"] if "independent_prompt_shape_rounding" in case]
    if altered:
        lines += ["", "### Independent prompt-prefill execution-shape rounding", "",
            "These comparisons retain the predeclared altered-shape ceiling: max absolute logits ≤0.125, KL≤0.0001, and matching top-1. A failed comparison is reported as failed, never accepted by widening the tolerance. Full repair is checked against its **matched native/native-splice reference**, not claimed identical to full native replay D.", "",
            "| Calibration task | Altered-shape check passed | Max absolute logit difference | Max KL | All sampled top-1 equal |",
            "|---|---:|---:|---:|---:|"]
        for case in altered:
            result=case["independent_prompt_shape_rounding"];rs=result.get("positions",[])
            lines.append(f"| {case['task_id']} | {result.get('passed')} | {metric(r.get('max_abs') for r in rs)['max']} | {metric(r.get('kl') for r in rs)['max']} | {all(r.get('top1_equal') is True for r in rs)} |")
        if any(case["independent_prompt_shape_rounding"].get("passed") is False for case in altered):
            lines += ["", "**Observed independent-prompt rounding discrepancy:** at least one actual context exceeds that declared cross-shape ceiling. The matched splice controls remain the appropriate full-repair comparison; those same-path controls still require bit-exact logits. Do not describe full repair as numerically identical to D on the affected cases."]
    if s["calibration_failures"]:
        lines += ["", "Execution failures (not candidate correctness failures):"]
        for failure in s["calibration_failures"]:
            lines.append(f"- `{failure['path']}`: {failure.get('exception','failure')}: {failure.get('error','see raw receipt')}")
    lines += ["", "## Matched-query local probes", "",
        "Local native-reference queries diagnose one attention operation. They do not demonstrate complete-answer code quality. Distribution probes use identical prefixes but may have different later-layer queries.", "",
        "| Arm / fraction | Layer-position observations | Mean relative attention-output L2 | Mean retained reasoning mass |", "|---|---:|---:|---:|"]
    if not s["local_probe_summary"]:
        lines.append("| No completed local probe records available | 0 | Unavailable | Unavailable |")
    for key, value in s["local_probe_summary"].items():
        lines.append(f"| {key} | {value['observations']} | {value['attention_output_relative_L2']['mean']} | {value['retained_reasoning_mass_ratio']['mean']} |")
    lines += ["", "## Full-screen executed-code outcomes", "",
        "| Condition | Generated / 36 | Binary scored / 36 | Passes | Full-population pass rate | Mean answer tokens | EOS / capped |", "|---|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        r=s["condition_diagnostics"][condition]
        rate="Unavailable" if r["pass_rate_full_population"] is None else f"{r['pass_rate_full_population']:.2%}"
        lines.append(f"| {condition} | {r['generated']}/36 | {r['scored_binary']}/36 | {r['passed'] if r['passed'] is not None else 'Unavailable'} | {rate} | {r['answer_tokens']['mean'] if r['answer_tokens']['n'] else 'Unavailable'} | {r['EOS']} / {r['capped']} |")
    if full:
        lines += ["", "Task-paired contrasts; unadjusted exploratory 95% task-cluster bootstrap intervals (10,000 PCG64 resamples, seed 20260921):", "",
            "| Contrast | Difference (pp) | 95% interval (pp) | Tasks improved / worse / tied |", "|---|---:|---:|---:|"]
        for key,r in s["statistics"]["contrasts"].items():
            lo,hi=r["task_cluster_ci95"]
            lines.append(f"| {key} | {100*r['difference']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {r['tasks_improved']} / {r['tasks_worse']} / {r['tasks_tied']} |")
        lines += ["", "All 12 task clusters and paired draw changes are retained in `per_task.csv`; failure categories, output lengths, stopping and working-set summaries are in `summary.json`/`per_draw.csv`. Interface failures are not separately identified by the frozen scorer and may overlap runtime/assertion categories; their separate count is unavailable. Passing the frozen tests is the measured endpoint, not proof for every valid input."]
    else:
        lines += ["", "**No complete-screen quality contrast or confidence interval is available.** Missing/unrun rows are retained explicitly; no native-success-only subset or partial favorable cohort is substituted."]
    lines += ["", "## Working set, timing, memory and cost", "",
        "The implemented oracle keeps native backing caches, scans native keys and materializes dense temporary K/V/masks. Record logical pairs separately from bytes actually read: no HBM traffic or sparse latency saving was measured by the bookkeeping.", "",
        "**Denominators:** the nominal budget applies independently to each layer/KV-head's historical reasoning positions (ceil rounding). Cumulative fractions count unique (layer, KV-head, position) pairs across the entire answer, not a global token union. The complete original prompt and the current/own-answer states stay present outside this budget. D/H have no selector accounting: their zero raw selected-pair counters do not mean zero dense reads; P has no reasoning intervention.", "",
        "**Pages and residency:** both page columns below are hypothetical 16-position layouts, not allocated/freed pages or transferred bytes. One layout pages each KV head independently; the other charges all KV heads when any head touches an absolute-position page. Padding/boundary pages can exceed raw reasoning-row bytes. `working_sets.csv` retains per-head cumulative fractions, per-layer logical buffer sizes, shared backing sizes and CUDA allocator snapshots. Original native/mapped/hybrid caches, the independent prompt and P cache remain resident even while P runs; the source-cache construction footprint marked `source_peak_then_freed` is not an additional simultaneously live answer cache.", "",
        "**Measurement gaps:** native-key-scan timing includes current-query scoring plus probability normalization; gather/replacement/dense-buffer construction/masks share one stage and are not separately measured. KV-group mask tensor bytes omit the query-head-expanded mask and combined-mask scratch; other scoring/gather/softmax scratch is not fully itemized. Per-layer maxima are not concurrent temporary-memory peaks. Unprofiled whole-answer wall time still includes CPU union accounting and durable I/O. The repeated 16-token calibration profile is not complete-answer end-to-end cost evidence.", ""]
    measured=[(name,r) for name,r in s["condition_diagnostics"].items() if r["generated"]]
    if measured:
        lines += ["Working-set summaries below use available completed draws only; coverage is explicit and no quality inference is made from a partial population.", "",
            "| Arm | Working-set draws | Instantaneous pair fraction mean | Cumulative pair fraction mean [min, max] | Estimated independent-head page bytes mean | Estimated shared-head page bytes mean |",
            "|---|---:|---:|---:|---:|---:|"]
        for name,r in measured:
            instant=r["instantaneous_reasoning_fraction"];cumulative=r["cumulative_reasoning_fraction"]
            interval=f"{cumulative['mean']:.4f} [{cumulative['min']:.4f}, {cumulative['max']:.4f}]" if cumulative["n"] else "Not applicable"
            inst=f"{instant['mean']:.4f}" if instant["n"] else "Not applicable"
            pages=r["cumulative_per_head_page_bytes_estimated"];shared=r["cumulative_all_heads_shared_page_bytes_estimated"]
            lines.append(f"| {name} | {cumulative['n']} | {inst} | {interval} | {pages['mean'] if pages['n'] else 'Not applicable'} | {shared['mean'] if shared['n'] else 'Not applicable'} |")
        lines += ["", "Draw-attempt wall includes cache clone, sampler call and any trace reconstruction, but is captured before the final working-set/answer/complete JSON writes. These are diagnostic mixed-residency costs, not isolated deployment latency or matched-length speedups. Completion-timing overhead (including final sampler checkpoint/materialization) is preserved separately in summary/CSV; it overlaps active/draw time and must not be added again or subtracted to claim model-only latency.", ""]
    for name,r in measured:
        lines.append(f"- {name}: mean implemented draw-attempt wall time={r['actual_attempt_whole_draw_seconds']['mean']}s; cumulative reasoning-pair fraction mean={r['cumulative_reasoning_fraction']['mean']}; GPU peak allocated maximum={r['peak_GPU_allocated_bytes']['max']}bytes.")
    if not measured:
        lines.append("No complete free-running screen timing, memory or cumulative-working-set measurement is available. Calibration setup/profile timings, if present, are retained in the compact summary and are not extrapolated into a speedup.")
    if s["task_execution_records"]:
        lines += ["", "Completed task-attempt timing (shared setup/check work is included in task wall, counted once rather than 42 times):", "",
            "| Task | Completed draws | Recorded task-attempt wall s | Sum of recorded shared setup/check stages s |", "|---|---:|---:|---:|"]
        for task in s["task_execution_records"]:
            components=[v for v in task["cache_setup_timing"].values() if type(v) in (int,float)]
            lines.append(f"| {task['task_id']} | {task['completed_records']} | {task['this_attempt_whole_task_wall_seconds']} | {sum(components) if components else 'Unavailable'} |")
        lines += ["", "Task-attempt wall includes its final backing fingerprint check and completed draw writes, but not worker model loading/weight verification, earlier attempts, provider billing time or the task receipt's own persistence. Source reasoning was reused, not regenerated; its saved historical time is not new screen compute. Whole-fleet billing and continuing storage belong in the separate ledger."]
    cpu=s["cpu_scoring_accounting"]
    lines += ["", "### Isolated CPU scoring time (separate accounting scopes)", "",
        f"Manifest-bound score records with public test accounting: **{cpu['manifest_bound_score_records_with_accounting']}/504**. "
        f"Recorded final test receipts: {cpu['total_test_receipts']}; receipts with more than one infrastructure attempt: {cpu['test_receipts_with_multiple_attempts']}.", "",
        "| Count or timing | Recorded sum | Coverage |", "|---|---:|---|"]
    for name in ("executed_tests","total_tests"):
        value=cpu[name]
        lines.append(f"| {name} | {value['sum'] if value['sum'] is not None else 'Unavailable'} | "
            f"{value['score_records_with_count']} score records with count; {value['score_records_without_count']} without count |")
    for name in ("recorded_final_test_cpu_seconds","recorded_final_test_wall_seconds"):
        value=cpu[name]
        lines.append(f"| {name} | {value['sum'] if value['sum'] is not None else 'Unavailable'} | "
            f"{value['test_receipts_with_duration']} test receipts with duration; {value['test_receipts_without_duration']} without duration |")
    lines += ["", "These sums use only the top-level final per-test receipt, not the nested attempt histories. Early stopping means executed tests may be fewer than total tests; syntax/interrupted outcomes can lack a total or timing and are not zero-filled. Test wall durations overlap across parallel workers and are not elapsed scoring wall time. CPU seconds are a different measure. Do not add either sum to wrapper elapsed time or to the cost ledger, or treat these sums as complete retry/process/pod costs.", ""]
    coverage=cpu["wrapper_stage_coverage"]
    lines.append(f"Public CPU wrapper coverage: {coverage['jobs']} job(s), {coverage['launch_receipts']} launch receipt(s), {coverage['exit_receipts']} exit receipt(s).")
    if cpu["wrapper_stage_intervals"]:
        lines += ["", "| Pod / CPU job | Stage | Observed elapsed wall s | Exit code | Receipt state |", "|---|---|---:|---:|---|"]
        for row in cpu["wrapper_stage_intervals"]:
            lines.append(f"| {row['pod_id']} / {row['job_id']} | {row['stage']} | "
                f"{row['elapsed_seconds'] if row['elapsed_seconds'] is not None else 'Unavailable'} | "
                f"{row['returncode'] if row['returncode'] is not None else 'Unavailable'} | {row['status']} |")
    else:
        lines.append("No validated CPU wrapper stage interval was available in this snapshot.")
    lines += ["", "Wrapper intervals are launch-to-exit timestamps bound to the public job plan, immutable CPU lease and pod identity. A launch without an exit has no inferred duration, completion, or current-liveness claim. Preflight/prepare/run/finalize are separate stages; stage elapsed time includes work and waiting inside that stage, overlaps contained candidate execution, and excludes other direct preflights, transfer/setup, inter-stage gaps and release/billing time. Neither wrapper receipt presence nor timing changes any computed quality outcome."]
    ledger=s["supplements"].get("cost_ledger.json")
    lines += ["", "Cost ledger: `results/sparse_repair_01/cost_ledger.json`. " +
        ("The supplied ledger is retained verbatim in summary.json; quote×elapsed estimates and continuing storage allowances are not invoices. The 1360 USD reservation baseline must not be presented as incurred cost." if ledger else "No final reconciled cost ledger was available at collection; no spend total is invented.")]
    economics=s["supplements"].get("economics/summary.json")
    if economics:
        e=economics["comparisons"]["D"]
        lines += ["", f"Records-only economic reference: perfect/free faithful prefill replacement saves {e['saving_seconds_mean']:.6f}s ({e['saving_percent_of_baseline']:.6f}%) against recorded D, leaving {economics['perfect']['mean_seconds']:.6f}s per answer. See ECONOMIC_CEILING.md. This is hypothetical, not a result of the oracle implementation."]
    lines += ["", "## Decision: five bounded questions", ""]
    if not full:
        decisions=[("Does this selector permit useful sparse native access at 5/10/25%?", "Unavailable: no complete frozen screen with code outcomes. Local probes cannot establish task quality."),
            ("Does sparse mapped state preserve the same benefit?", "Unavailable: no valid full-screen N_p versus M_p outcome comparison."),
            ("Does small targeted repair rescue H, especially versus random/recent?", "Unavailable: R_10−H and matched-budget controls have not all been measured/scored."),
            ("Is the useful working set small over a complete answer?", "Unavailable: per-step sparsity/calibration prefixes do not answer full-answer cumulative fetch cost."),
            ("What should the next investment be?", "No cheaper selector, translator, selective receiver computation or compact-text investment is validated by incomplete evidence. Resolve only the concrete numerical/runtime/cost blocker shown above before deciding whether this bounded screen can run; do not expand the research portfolio.")]
    else:
        stats=s["statistics"]; contrasts=stats["contrasts"]
        native="; ".join(f"N_{p}={stats['conditions'][f'N_{p}']['pass_rate']:.1%}, M_{p}={stats['conditions'][f'M_{p}']['pass_rate']:.1%}" for p in (5,10,25))
        repair=contrasts["R_10-H"]; rand=contrasts["R_10-R_random"]; recent=contrasts["R_10-R_recent"]
        fraction=s["condition_diagnostics"]["R_10"]["cumulative_reasoning_fraction"]
        decisions=[("Does this selector permit useful sparse native access?", f"Observed budget curve: {native}; D={stats['conditions']['D']['pass_rate']:.1%}. This small exploratory screen cannot establish equivalence or general preservation."),
            ("Does sparse mapped state preserve the same benefit?", "Use the three paired N_p−M_p contrasts above; free-running histories diverge, so this is not an identical-query causal contrast."),
            ("Does small targeted repair rescue H versus cheap controls?", f"R_10−H={100*repair['difference']:+.2f}pp; versus random={100*rand['difference']:+.2f}pp; versus recent={100*recent['difference']:+.2f}pp. Read task-cluster intervals and all tasks, not just point estimates."),
            ("Is the required working set small over a complete answer?", f"R_10 cumulative reasoning fraction mean={fraction['mean']}, min={fraction['min']}, max={fraction['max']}; nominal 10% per step does not imply 10% total transfer/capacity or allocated pages."),
            ("What is the next investment?", "A cheaper targeted-selector/repair experiment earns consideration only if the paired repair gain survives the cheap controls and full-answer cumulative cost remains favorable. Otherwise these data favor investigating fidelity/dependencies or a separately bounded compact-text comparison, not scaling this oracle into a serving engine. No automatic follow-on is authorized.")]
    for i,(question,answer) in enumerate(decisions,1):
        lines.append(f"{i}. **{question}** {answer}")
    interpretation = s.get("analyst_interpretation")
    if interpretation:
        def inline(value):
            value = " ".join(value.split())
            return re.sub(r"([\\`*_{}\[\]<>#!|])", r"\\\1", value)
        review = interpretation["review"]
        lines += ["", "## Analyst interpretation — not additional measurement", "",
            "This separately authored review is bound to the exact numeric input snapshot and declaration. It does not replace the computed findings, change scores or intervals, establish generalization, or authorize another experiment.", "",
            f"Analyst: {inline(review['analyst'])}. Reviewed at: {inline(review['reviewed_at_utc'])}. "
            f"Review file SHA-256: `{interpretation['review_sha256']}`.", ""]
        for i, question in enumerate(review["questions"], 1):
            lines += [f"{i}. **{question['id']} — {question['verdict']}**: {inline(question['rationale'])}",
                "   - Evidence (computed-summary JSON pointers): " + "; ".join(inline(ref) for ref in question["evidence_refs"]),
                "   - Limitations: " + "; ".join(inline(limit) for limit in question["limitations"])]
    lines += ["", "## Limitations and reproduction", ""]+["- "+x for x in s["limitations"]]
    command="python3 scripts/sparse_repair_report.py --input results/sparse_repair_01/report/input_records.json --output results/sparse_repair_01/report --report SPARSE_REPAIR_RESULTS.md"
    if interpretation:
        command += " --decision-review " + shlex.quote(str(decision_review_path or DEFAULT_ROOT/"report/decision_review.json"))
    lines += ["", f"Portable input SHA-256: `{input_sha}`. `input_manifest.json` records every collected source hash; regeneration reads compact saved metadata, not weights, hidden tests or candidate programs.", "", "```sh",
        command, "```", "",
        "Full-screen cluster intervals additionally require NumPy; incomplete-report regeneration uses only Python standard library. Raw receipts remain authoritative and original confirmation/publication artifacts are not rewritten."]
    return "\n".join(lines)+"\n"


def csv_write(path, rows):
    path=Path(path)
    if not rows:
        path.write_text(""); return
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True,separators=(",",":")) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collect",action="store_true")
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--declaration",type=Path,default=Path("configs/coding_pilot_v1/sparse_repair_01/declaration.json"))
    p.add_argument("--input",type=Path,default=DEFAULT_ROOT/"report/input_records.json")
    p.add_argument("--output",type=Path,default=DEFAULT_ROOT/"report")
    p.add_argument("--report",type=Path,default=Path("SPARSE_REPAIR_RESULTS.md"))
    p.add_argument("--decision-review",type=Path,
        help="Optional separately saved evidence-bound analyst review; never included in numeric input_records.json")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.collect:
        data=collect(args.root,args.declaration)
        args.input=args.output/"input_records.json";write(args.input,data)
        write(args.output/"input_manifest.json",{"file":args.input.name,"sha256":sha(args.input),
            "bytes":args.input.stat().st_size,"source_inputs":data["source_inputs"]})
    manifest=read(args.input.parent/"input_manifest.json")
    if manifest["file"]!=args.input.name or manifest["sha256"]!=sha(args.input):
        raise ValueError("Portable report input hash mismatch")
    data=read(args.input);summary=summarize(data);summary["input_sha256"]=sha(args.input)
    if args.decision_review:
        protected={args.input.resolve(), (args.input.parent/"input_manifest.json").resolve(), args.report.resolve()}
        protected.update((args.output/name).resolve() for name in ("summary.json","per_draw.csv","per_task.csv","working_sets.csv"))
        if args.decision_review.resolve() in protected:
            raise ValueError("Decision review must be a separate input, not a report output or numeric snapshot")
        summary=attach_decision_review(summary,args.decision_review,sha(args.input))
    write(args.output/"summary.json",summary)
    fields=("task_id","condition","seed_index","answer_seed","status","passed","score_missing","category","answer_tokens","answer_ended_eos","answer_capped","answer_path","answer_sha256","timing","cpu_score_accounting")
    csv_write(args.output/"per_draw.csv",[{k:r.get(k) for k in fields} for r in data["draws"]])
    csv_write(args.output/"per_task.csv",summary["statistics"]["per_task"] if summary["statistics"] else [])
    csv_write(args.output/"working_sets.csv",summary["working_sets"])
    args.report.write_text(render(summary,sha(args.input),args.decision_review))
    print(json.dumps({"status":summary["status"],"coverage":summary["coverage"]},indent=2))


if __name__=="__main__":
    main()
