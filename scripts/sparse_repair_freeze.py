#!/usr/bin/env python3
"""Freeze the bounded sparse-repair screen using identities and lengths, never scores."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/coding_pilot_v1/sparse_repair_01")
SELECTION_SEED = 20260921
MAPPER = "0e3caa7111ad8861f8e68e361ade6732e4dca5e587ca4f1ed8ba072a7ca9e386"
STREAMS = ("answer_small", "coverage_answer_1", "coverage_answer_2")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def seed_for(task_id, sample_index, stream):
    value = f"coding_pilot_v1|{task_id}|{sample_index}|{stream}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63)


def rank_digest(task_id):
    return hashlib.sha256(f"sparse_repair_01|{SELECTION_SEED}|{task_id}".encode()).hexdigest()


def sample(histories):
    if len(histories) != 40 or len({x["task_id"] for x in histories}) != 40:
        raise ValueError("Exactly the original 40 unique development histories required")
    ordered = sorted(histories, key=lambda x: (x["historical_reasoning_positions"], x["task_id"]))
    calibration, screen, population = [], [], []
    for stratum in range(4):
        ranked = sorted(ordered[10*stratum:10*(stratum+1)], key=lambda x: (rank_digest(x["task_id"]), x["task_id"]))
        for rank, source in enumerate(ranked):
            role = "calibration" if rank == 0 else "screen" if rank <= 3 else "not_selected"
            row = {"task_id": source["task_id"], "history_path": source["path"],
                   "history_sha256": source["sha256"], "prompt_tokens": source["prompt_tokens"],
                   "historical_reasoning_positions": source["historical_reasoning_positions"],
                   "natural_boundary": source["natural_boundary"], "reasoning_capped": source["reasoning_capped"],
                   "stratum": stratum, "selection_rank": rank, "rank_sha256": rank_digest(source["task_id"]),
                   "role": role, "token_field_hashes": source["token_field_hashes_compact_json_utf8"]}
            population.append(row)
            if role == "calibration":
                calibration.append(row)
            elif role == "screen":
                screen.append(row)
    return calibration, screen, population


def conditions():
    result = [{"name": "D", "mode": "native", "fraction": 1.0, "selector": "all"},
              {"name": "H", "mode": "mapped", "fraction": 1.0, "selector": "all"},
              {"name": "P", "mode": "prompt_only", "fraction": 0.0, "selector": "none"}]
    for percent in (5, 10, 25):
        for label, mode in (("N", "native_sparse"), ("M", "mapped_sparse"), ("R", "repair")):
            result.append({"name": f"{label}_{percent}", "mode": mode, "fraction": percent / 100,
                           "selector": "native_attention_mass"})
    result.extend([{"name": "R_random", "mode": "repair", "fraction": .1, "selector": "random"},
                   {"name": "R_recent", "mode": "repair", "fraction": .1, "selector": "recent"}])
    return result


def build(repo, audit_path):
    audit = read(audit_path)
    public = audit["public_development_inputs"]
    if sha(public["path"]) != public["expected_sha256"]:
        raise ValueError("Public development backup identity mismatch")
    visible = read(public["path"])
    histories = audit["development_histories"]
    if set(x["task_id"] for x in visible) != set(x["task_id"] for x in histories):
        raise ValueError("Visible development membership mismatch")
    for h in histories:
        if not h["match"] or sha(repo / h["path"]) != h["expected_sha256"]:
            raise ValueError("History identity mismatch")
        value = read(repo / h["path"])
        if len(value["prefix_ids"]) - len(value["prompt_ids"]) != h["historical_reasoning_positions"]:
            raise ValueError("History length identity mismatch")
    calibration, screen, population = sample(histories)
    protocol_path = "configs/coding_pilot_v1/confirmation_01/protocol.json"
    protocol = read(repo / protocol_path)["protocol"]
    if sha(repo / protocol_path) != "b9d28fdacec13e3d1eb9d97e9923e8063c2ea94b62b3d6e5496e8c5486b250ff":
        raise ValueError("Confirmation protocol changed")
    tids = [r["task_id"] for r in calibration + screen]
    declaration = {
        "schema": "gearshift.sparse_repair.declaration.v1", "experiment_id": "sparse_repair_01",
        "status": "FROZEN_SCIENTIFIC_DECLARATION_EXECUTION_GATED",
        "scope": "Exploratory diagnosis on already-inspected development tasks; not confirmation or a deployable speedup.",
        "publication_tag": "gearshift-progress-01", "publication_commit": "64725974fa55459350d1c9d09037bab64d0c5ec6",
        "mapper": {"training_seed": 20260915, "arm": "ROTATING", "step": 1024, "sha256": MAPPER},
        "inputs": {"artifact_audit_path": str(audit_path.relative_to(repo)), "artifact_audit_sha256": sha(audit_path),
            "history_identity_source": audit["history_identity_source"], "development_membership": audit["development_membership"],
            "public_development_inputs": public, "private_development_input_sha256": audit["private_development_inputs"]["sha256"],
            "private_tests_read_by_freezer": False, "protocol_path": protocol_path, "protocol_sha256": sha(repo / protocol_path)},
        "sample_rule": {"seed": SELECTION_SEED, "population": "original40development",
            "sort": ["historical_reasoning_positions ASC", "task_id ASC"], "strata": "four consecutive groups of 10",
            "within_stratum_rank": "SHA256 UTF8(sparse_repair_01|20260921|task_id) ascending hex; tie task_id",
            "selection": "rank0 calibration; ranks1,2,3 screen; others retained in population audit",
            "length_definition": "len(prefix_ids)-len(prompt_ids), excluding natural closing-think bridge",
            "pass_fail_outcomes_or_generated_answer_content_used": False},
        "task_ids": tids, "calibration_tasks": calibration, "screen_tasks": screen,
        "population": population, "calibration_task_count": 4, "screen_task_count": 12,
        "answer_draws_per_screen_task_condition": 3, "screen_answer_records": 504,
        "conditions": conditions(), "seeds": {tid: [
            {"seed_index": i, "stream": s, "answer_seed": seed_for(tid, 0, s)} for i, s in enumerate(STREAMS)] for tid in tids},
        "calibration_seeds": {r["task_id"]: seed_for(r["task_id"], 0, "sparse_repair_01_calibration_D") for r in calibration},
        "seed_derivation": "Existing confirmation seed_for: first8 SHA256(coding_pilot_v1|task_id|0|stream), big-endian modulo2**63; distinct streams identify the3 draws; worker order excluded.",
        "protocol": {k: protocol[k] for k in ("models", "runtime", "decoding", "budgets", "handoff", "eos_token_ids", "closing_think_token_id", "P_definition", "H_definition", "cache_isolation")},
        "selector": {"label": "ORACLE-ASSISTED", "scope": "each current query, attention layer and KV-head group independently",
            "score": "native historical keys with current treatment query; sum native attention probabilities across query heads sharing KV head",
            "normalization": "native prompt/reasoning plus this treatment's own current/generated answer positions; causal mask; never future tokens",
            "ranking": "descending FP32 native mass; stable ties lower original absolute position",
            "budget": "k=ceil(p*historical_reasoning_positions), identical k per layer/KV-head group; complete K/V pairs",
            "random": "seed from first8 big-endian bytes SHA256 compact JSON(['sparse-repair-01-random-v1',task_id,draw_seed,layer,step,kv_head]) modulo2**63; CPU torch.Generator randperm under pinned runtime; exact same k, sorted ascending selected positions",
            "recent": "k highest original reasoning positions at every layer/group",
            "outside_budget": "entire original native prompt, current token and own answer states stay present",
            "N_M": "mask unselected reasoning and renormalize actual arm attention over retained entries",
            "R": "immutable mapped reasoning background plus current native replacements; recompute actual repaired attention, never reuse oracle weights or permanently accumulate repairs",
            "position_policy": "retain original absolute positions/RoPE; no token packing or shifting",
            "comparison": "N/M/R share scoring rule; only identical queries imply identical selected positions"},
        "numerical_controls": {"same_execution_shape": {"max_abs_logit_difference": 0.0, "torch_equal_logits": True, "top1_equal": True},
            "altered_execution_shape_only": {"max_abs_logit_difference": .125, "max_KL_native_to_control": .0001, "top1_equal": True},
            "required": ["unmodified hook versus deployed dense", "full-selection native versus dense native",
                "zero repair versus existing native-prompt/mapped-reasoning hybrid", "full repair versus matched native/native splice",
                "no aliases/mutation/repeated_or_skipped_tokens/position_drift"],
            "policy": "measure and report rounding with same-path reference; no post-output tolerance widening; block screen on failed controls",
            "prompt_rounding": "independent native prompt prefill may differ; retain matched native/native splice reference rather than mislabel it exact full prefill"},
        "local_probe_positions": [0, 32, 128],
        "local_probes": {"source": "calibration D's own free-running prefix, maximum129 tokens; teacher-forced only for matched-query diagnostics",
            "query": "identical native Q for local N/M/R attention-output comparisons; identical prefix for model-distribution comparisons",
            "early_stop": "if EOS precedes fixed probe, record missing position; do not extend or replace task",
            "metrics": ["retained_native_attention_mass", "attention_output_relative_L2", "attention_output_max_abs",
                "next_token_KL_native_to_treatment", "next_token_total_variation", "next_token_top1_agreement"]},
        "metrics": {"primary_engineering_contrast": "R_10 minus H passed-draw mean, paired within task",
            "secondary": ["all14 condition rates", "full5/10/25percent curve", "R_10 versus R_random/R_recent", "N_p versus M_p"],
            "unit": "12 task clusters, each3 fixed draws; retain every task and native failures",
            "cluster_intervals": {"resamples": 10000, "seed": 20260921, "quantiles": [.025, .975],
                "rng": "numpy.random.Generator(numpy.random.PCG64(20260921))", "quantile_method": "linear",
                "interpretation": "exploratory task-cluster descriptive intervals, not equivalence or generalization"},
            "outcomes": ["pass", "interface", "syntax", "runtime", "assertion", "timeout", "infrastructure_missing"],
            "also": ["per_task_paired_draw_changes", "all_answer_lengths", "EOS/cap", "instantaneous_pairs", "cumulative_unique_pairs", "per_layer_head_union", "logical_rows", "physical_pages_or_estimate", "full_backing_and_prompt_answer_bytes"]},
        "cost_accounting": {"stages": ["native_cache_construction", "source_cache_reconstruction", "mapper", "native_key_scan", "selection", "gather", "replacement", "attention", "generation", "CPU_scoring", "transfer"],
            "timing": "whole-answer implemented walltime plus component times; tracing-disabled equivalent-semantics repeats where practical",
            "memory": "include source/native/mapped backing, native prompt/answer states, masks/indices, temporaries; distinguish measured from estimated bytes",
            "dense_mask_warning": "dense masked SDPA is a quality intervention, not sparse latency or bytes-read savings",
            "page_estimate": "report16-token bookkeeping pages under independent per-head and all-head-interleaved layouts, not measured allocator or HBM transactions",
            "selected_index_trace_steps": [0, 1, 16, 64, 256, 1024, 4095],
            "selected_index_trace_layers": [0, 17, 35],
            "selected_index_trace_KV_heads": [0, 7],
            "selected_index_trace_scope": "fixed bounded subset for bulky index lists only; ALL layer/head pair counts, instantaneous budgets, page estimates and cumulative unions retained every step",
            "cumulative_union": "track exact unique layer/KVhead/reasoning-position pairs at every step, including non-traced steps"},
        "scorer": audit["scorer"],
        "execution_gates": ["STUDIO_READY prerequisites verified", "single orchestration owner verified", "hard remaining authorization and storage ledger verified",
            "this declaration plus implementation committed before screen", "four-case calibration and controls pass", "oracle overhead profile fits bounded screen"],
        "resource_estimate": {"incremental_planning_range_usd": [250, 300], "not_a_new_authorization": True,
            "cumulative_soft_target_usd": 2000, "hard_cap": "must independently verify owner amendment and remaining ledger",
            "provisional_gpu_rate_ceiling_usd_per_hour": 5.5, "conditional_total_gpu_hours_at_rate_ceiling": 50,
            "conditional_compute_usd_at_rate_ceiling": 275, "incremental_CPU_transfer_storage_backup_allowance_usd": 25,
            "useful_parallelism": "independent task/condition groups on small compatible fleet after calibration; reuse immutable caches safely",
            "profile_rule": "project504 full answers from calibrated actual oracle overhead/length; stop dispatch and report concrete blocker if disproportionate, notify if material expansion needed; never silently change sparsity",
            "existing_jobs": "do not cancel healthy unrelated/secondary work or duplicate scheduler", "telemetry": "warning-based memory, atomic progress, bounded recovery and verified backup"},
        "forbidden": ["training", "new reasoning samples", "confirmation selector tuning", "condition search after outcomes", "new architecture sweep", "website edits", "publication", "tag movement"]}
    return declaration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--artifact-audit", type=Path, default=Path("results/sparse_repair_01/artifact_audit.json"))
    p.add_argument("--output", type=Path, default=CONFIG)
    args = p.parse_args()
    repo = args.repo.resolve()
    audit_path = args.artifact_audit if args.artifact_audit.is_absolute() else repo / args.artifact_audit
    output = args.output if args.output.is_absolute() else repo / args.output
    declaration = build(repo, audit_path)
    encoded = json.dumps(declaration, indent=2, sort_keys=True) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "declaration.json"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("Refusing to overwrite a changed frozen declaration; review/version explicitly")
    path.write_text(encoded)
    print(json.dumps({"declaration_path": str(path), "sha256": sha(path),
        "calibration": [r["task_id"] for r in declaration["calibration_tasks"]],
        "screen": [r["task_id"] for r in declaration["screen_tasks"]], "screen_answer_records": 504}, indent=2))


if __name__ == "__main__":
    main()
