#!/usr/bin/env python3
"""Records-only faithful-prefill-removal reference; Python standard library only.

--extract-repo verifies saved source receipts and writes a portable, whitelisted
input snapshot. Normal use regenerates from that snapshot without weights,
private tests, candidate execution, network access, or generation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

EXPERIMENT = "confirmation_01_20260919T094418Z"
RESULT = Path("results/coding_pilot_v1") / EXPERIMENT
OUT = Path("results/sparse_repair_01/economics")
CONDITIONS = ("A", "B", "D", "ROTATING_H")
STAGES = ("reasoning_seconds", "answer_seconds", "native_prefill_seconds",
          "mapping_seconds", "splice_seconds", "cache_clone_seconds")
MAPPER = "0e3caa7111ad8861f8e68e361ade6732e4dca5e587ca4f1ed8ba072a7ca9e386"
CSV_SHA = "8a8e932a58373d249e777d94d983224d9d0493ef9c72c032603091c4a74431bb"
DECLARATION_SHA = "16387d5361c795121929fd9244821f12ee005f9b36b2febddca5d9ef89f7536f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def checked(root, relative, expected):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unscoped evidence path")
    path = root / relative
    if sha(path) != expected:
        raise ValueError("Evidence hash mismatch: " + str(relative))
    return path


def extract(repo, out):
    """Verify original records but export no program text or private test data."""
    repo, out = Path(repo), Path(out)
    root = repo / RESULT
    report = root / "primary/report"
    manifest = read_json(report / "report_manifest.json")
    bindings = manifest["source_bindings"]
    csv_path = checked(report, "per_task_seed.csv", CSV_SHA)
    listed = next(x for x in manifest["files"] if x["path"] == "per_task_seed.csv")
    if listed["sha256"] != CSV_SHA:
        raise ValueError("Report manifest disagrees with pinned CSV")
    plan_path = checked(repo, bindings["scoring_plan_path"], bindings["scoring_plan_sha256"])
    plan = read_json(plan_path)
    declaration_path = checked(repo, plan["declaration_path"], DECLARATION_SHA)
    declaration = read_json(declaration_path)
    if (plan["cohort"] != "primary" or plan["expected_answers"] != 4800 or
            plan["scorer_identity"]["scorer_version"] != "gearshift_scorer_v2_20260919_01" or
            declaration["primary_checkpoints"]["ROTATING"]["mapper_sha256"] != MAPPER):
        raise ValueError("Wrong confirmation population/scorer/mapper")
    closure_path = checked(root, plan["generation_closure_path"], bindings["generation_closure_file_sha256"])
    closure = read_json(closure_path)
    closure_files = {x["path"]: x for x in closure["files"]}
    score_manifest_path = checked(root, plan["manifest_path"], bindings["scored_manifest_sha256"])
    score_manifest = read_json(score_manifest_path)
    scored = {(x["task_id"], x["condition"], x["seed_index"]): x for x in score_manifest["files"]}
    histories = {}
    history_hashes = []
    for tid in declaration["task_ids"]:
        for role, folder in (("source", "large_history"), ("receiver", "small_history")):
            name = f"primary/tasks/{tid.replace('/', '__')}/{folder}/source_history.json"
            expected = closure_files[name]["sha256"]
            history = read_json(checked(root, name, expected))
            if history["task_id"] != tid or history["sampler_timing"]["complete"] is not True:
                raise ValueError("History identity/timing mismatch")
            histories[tid, role] = history
            history_hashes.append({"task_id": tid, "role": role, "path": name, "sha256": expected})
    rows = []
    with csv_path.open(newline="") as stream:
        for saved in csv.DictReader(stream):
            if saved["condition"] not in CONDITIONS:
                continue
            key = saved["task_id"], saved["condition"], int(saved["seed_index"])
            item = scored[key]
            if (saved["answer_sha256"] != item["answer_sha256"] or
                    saved["answer_path"] != item["answer_path"] or
                    saved["score_sha256"] != item["sha256"] or saved["score_path"] != item["path"]):
                raise ValueError("Report/score-manifest binding mismatch")
            if closure_files[saved["answer_path"]]["sha256"] != saved["answer_sha256"]:
                raise ValueError("Answer is not generation-sealed")
            raw = read_json(checked(root, saved["answer_path"], saved["answer_sha256"]))
            receipt = read_json(checked(root, saved["score_path"], saved["score_sha256"]))
            score = receipt["score_v2"]
            role = "receiver" if key[1] == "B" else "source"
            history = histories[key[0], role]
            if (raw["reasoning_seconds"] != history["reasoning_seconds"] or
                    raw["reasoning_model"] != role or raw["source_cost_divisor_for_single_output"] != 1 or
                    raw["inference_timing_complete"] is not True or
                    raw["sampler_timing"]["complete"] is not True or
                    score["scorer_identity"] != plan["scorer_identity"] or
                    str(score["passed"]) != saved["passed"] or str(score["missing"]) != saved["missing"]):
                raise ValueError("Source charge/timing/scorer identity mismatch")
            row = {"task_id": key[0], "condition": key[1], "seed_index": key[2],
                   "answer_seed": int(saved["answer_seed"]), "passed": score["passed"],
                   "missing": score["missing"], "answer_tokens": len(raw["answer_ids"]),
                   "reasoning_model": role, "source_cost_divisor_for_single_output": 1,
                   "inference_timing_complete": True}
            for field in STAGES + ("single_output_inference_seconds",):
                if float(saved[field]) != raw[field]:
                    raise ValueError("Timing record/report mismatch: " + field)
                row[field] = raw[field]
            for field in ("task_id", "condition", "seed_index", "answer_seed"):
                if raw[field] != row[field] or receipt[field] != row[field]:
                    raise ValueError("Task/draw identity mismatch")
            row.update({name: saved[name] for name in ("answer_path", "answer_sha256", "score_path", "score_sha256")})
            rows.append(row)
    source_paths = [csv_path, report / "report_manifest.json", plan_path, declaration_path,
                    closure_path, score_manifest_path]
    data = {"schema": 1, "experiment_id": EXPERIMENT, "cohort": "primary",
            "task_ids": declaration["task_ids"], "task_count": 200, "draws_per_task_condition": 3,
            "conditions": list(CONDITIONS), "primary_mapper_sha256": MAPPER,
            "source_bindings": bindings, "source_inputs": [
                {"path": str(p.relative_to(repo)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in source_paths],
            "verification": {"raw_answer_hashes_checked": len(rows), "score_receipt_hashes_checked": len(rows),
                "history_hashes_checked": len(history_hashes), "private_test_files_read": False,
                "candidate_execution_performed": False, "model_generation_performed": False},
            "history_hashes": history_hashes, "rows": rows}
    analyze(data)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "input_records.json", data)
    write_json(out / "input_manifest.json", {"file": "input_records.json", "sha256": sha(out / "input_records.json"),
               "bytes": (out / "input_records.json").stat().st_size, "source_inputs": data["source_inputs"]})


def finite_nonnegative(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected finite nonnegative timing")
    return value


def analyze(data, expected_task_count=200):
    tids = data["task_ids"]
    if (len(tids) != expected_task_count or len(set(tids)) != len(tids) or
            data["task_count"] != len(tids) or data["draws_per_task_condition"] != 3 or
            data["conditions"] != list(CONDITIONS) or data["cohort"] != "primary"):
        raise ValueError("Frozen population mismatch; no task/outcome subset is allowed")
    lookup = {}
    for row in data["rows"]:
        key = row["task_id"], row["condition"], row["seed_index"]
        if key in lookup:
            raise ValueError("Duplicate draw")
        lookup[key] = row
        if row["missing"] is not False or type(row["passed"]) is not bool:
            raise ValueError("Incomplete outcomes; do not silently drop draws")
        if row["inference_timing_complete"] is not True or row["source_cost_divisor_for_single_output"] != 1:
            raise ValueError("Incomplete timing or amortized source charge")
        stages = [finite_nonnegative(row[k]) for k in STAGES]
        total = finite_nonnegative(row["single_output_inference_seconds"])
        if not math.isclose(sum(stages), total, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("Stage sum mismatch")
    if set(lookup) != {(t, c, s) for t in tids for c in CONDITIONS for s in range(3)}:
        raise ValueError("Incomplete or extra condition/task/draw population")
    pairs = []
    for tid in tids:
        for seed in range(3):
            current = {c: lookup[tid, c, seed] for c in CONDITIONS}
            d = current["D"]
            if len({r["answer_seed"] for r in current.values()}) != 1:
                raise ValueError("Mismatched answer draws")
            if any(current[c]["reasoning_seconds"] != d["reasoning_seconds"] for c in ("A", "ROTATING_H")):
                raise ValueError("Source reasoning charge changed between arms")
            if d["mapping_seconds"] != 0 or d["splice_seconds"] != 0:
                raise ValueError("Native replay has unexpected mapping/splice cost")
            perfect = d["single_output_inference_seconds"] - d["native_prefill_seconds"]
            p = {"task_id": tid, "seed_index": seed, "answer_seed": d["answer_seed"],
                 "perfect_seconds": perfect, "removed_prefill_seconds": d["native_prefill_seconds"],
                 "perfect_passed": d["passed"]}
            p.update({c + "_minus_perfect_seconds": r["single_output_inference_seconds"] - perfect for c, r in current.items()})
            pairs.append(p)
    conditions = {}
    for c in CONDITIONS:
        rows = [lookup[t, c, s] for t in tids for s in range(3)]
        conditions[c] = {"draws": len(rows), "passed": sum(r["passed"] for r in rows),
                         "pass_rate": statistics.mean(r["passed"] for r in rows),
                         "mean_answer_tokens": statistics.mean(r["answer_tokens"] for r in rows),
                         "mean_seconds": {k: statistics.mean(r[k] for r in rows) for k in STAGES + ("single_output_inference_seconds",)}}
    perfect_mean = statistics.mean(p["perfect_seconds"] for p in pairs)
    comparisons = {}
    for c in CONDITIONS:
        baseline = conditions[c]["mean_seconds"]["single_output_inference_seconds"]
        delta = statistics.mean(p[c + "_minus_perfect_seconds"] for p in pairs)
        comparisons[c] = {"saving_seconds_mean": delta, "saving_percent_of_baseline": delta / baseline * 100,
                          "quality_difference_percentage_points": (conditions["D"]["pass_rate"] - conditions[c]["pass_rate"]) * 100}
    summary = {"experiment_id": EXPERIMENT, "scope": "HYPOTHETICAL free faithful receiver-prefill replacement only",
               "tasks": len(tids), "draws_per_condition": len(pairs), "observed_conditions": conditions,
               "perfect": {"mean_seconds": perfect_mean, "passed": conditions["D"]["passed"],
                   "pass_rate": conditions["D"]["pass_rate"],
                   "total_removed_prefill_seconds_charged_across_draws": sum(p["removed_prefill_seconds"] for p in pairs)},
               "comparisons": comparisons, "source_bindings": data.get("source_bindings", {}),
               "timing_is_stage_sum_not_measured_end_to_end_latency": True,
               "dollar_savings_or_serving_speedup_measured": False}
    return summary, pairs


def render(summary, input_sha):
    d = summary["observed_conditions"]["D"]["mean_seconds"]
    saving = summary["comparisons"]["D"]
    lines = ["# Economic ceiling: faithful receiver-prefill replacement", "",
        "**Records-only hypothetical, not a measured system or a universal bound.** No model generation, candidate execution, or private-test access occurred.", "",
        f"Removing all D receiver prefill saves **{saving['saving_seconds_mean']:.6f} seconds per answer ({saving['saving_percent_of_baseline']:.6f}%)**, changing its recorded stage-sum mean from **{d['single_output_inference_seconds']:.6f} s to {summary['perfect']['mean_seconds']:.6f} s**. That is little leverage in this regime.", "",
        "## Population and definition", "",
        f"All {summary['tasks']} frozen primary confirmation task clusters × three declared draws = {summary['draws_per_condition']} records per arm, with no missing scores or timings and no outcome-based filtering. The repaired `gearshift_scorer_v2_20260919_01` outcomes are preserved. A = large-only; B = independent small-model reasoning; D = full native receiver replay; ROTATING_H = native prompt plus primary step-1,024 mapped reasoning.", "",
        "For each task and draw: `perfect = D.single_output_inference_seconds − D.native_prefill_seconds`. Keep D's exact answer generation, answer length, outcome, full source reasoning charge, and cache-clone cost. Conversion and transfer are assumed free. Do not divide the source charge by three even though the experiment reused histories.", "",
        "## Comparisons on the same recorded population", "",
        "| Arm | Passed / 600 | Pass rate | Mean stage-sum seconds | Perfect saving vs arm (seconds) | Perfect saving vs arm (%) |",
        "|---|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        r, comparison = summary["observed_conditions"][c], summary["comparisons"][c]
        lines.append(f"| {c} (observed) | {r['passed']} / 600 | {r['pass_rate']:.2%} | {r['mean_seconds']['single_output_inference_seconds']:.6f} | {comparison['saving_seconds_mean']:+.6f} | {comparison['saving_percent_of_baseline']:+.6f}% |")
    p = summary["perfect"]
    lines += [f"| Perfect/free faithful converter (hypothetical) | {p['passed']} / 600, inherited | {p['pass_rate']:.2%}, inherited | {p['mean_seconds']:.6f} | — | — |", "",
        "Savings are differences of population means divided by the observed comparison mean, not a mean of per-task percentages. Negative savings mean the hypothetical is more expensive. Quality is not equal across the observed arms; the perfect arm's quality is assumed to equal D, not newly demonstrated. There is no inferred retry policy, quality-adjusted price, or deployable router.", "",
        "## Stage decomposition (mean seconds per answer)", "",
        "| Arm | Full reasoning | Answer | Native prefill | Mapping | Splice | Clone |", "|---|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        r = summary["observed_conditions"][c]["mean_seconds"]
        lines.append("| " + c + " | " + " | ".join(f"{r[k]:.6f}" for k in STAGES) + " |")
    lines += ["", f"D's source reasoning alone is {d['reasoning_seconds']:.6f} s ({100*d['reasoning_seconds']/d['single_output_inference_seconds']:.3f}% of its stage sum). Removing receiver prefill cannot remove this work. Across the 600 fully charged D records, removed prefill totals {p['total_removed_prefill_seconds_charged_across_draws']:.6f} s; this is NOT actual fleet time or a billing saving, because experimental setup/history reuse differ from this single-answer accounting.", "",
        "## Timing limitations", "",
        "- These are instrumented stage-sum estimates, not measured end-to-end serving latency or rental invoices. They preserve sampler checkpoint I/O and instrumentation already inside active timings; no model-only latency is invented by subtracting those costs.",
        "- Bridge/first-token time is already inside answer time. Source-cache reconstruction, interrupted/resume reconstruction, model/checkpoint loading, fleet waiting, transfers, unrecorded crash tails, and CPU scoring are not added to these established single-output estimates. This arithmetic does not claim they are free.",
        "- No confidence interval or generalization claim is made for these descriptive fixed-record differences. The 600 draws are clustered within 200 tasks, not 600 independent tasks; worker/hardware/load differences may affect timings.",
        "- A true converter must pay conversion, transfer, memory residency, validity checks and fallbacks. Sparse decoding or source-stage acceleration changes answer/source computation and is a separate intervention; this reference does not bound fusion or those architectures.",
        "- Oracle repair itself constructs the full receiver-native cache and pays dense scoring; it does not realize the hypothetical prefill saving. Fleet parallelism reduces study turnaround, not per-request work.", "",
        "## Provenance and reproduction", "",
        f"Portable input SHA-256: `{input_sha}`. The input manifest records source paths, byte sizes and hashes; each selected raw answer and repaired score receipt is hash-checked (2,400 each), plus 400 closure-bound histories. The snapshot contains timing/outcome fields and identities only, not candidate code, hidden tests or model tensors.", "",
        "From a checkout or the review archive root, using Python 3.10+ (standard library only):", "", "```sh",
        "python3 scripts/sparse_repair_economics.py --input results/sparse_repair_01/economics/input_records.json --output results/sparse_repair_01/economics --report ECONOMIC_CEILING.md",
        "python3 -m unittest discover -s tests -p 'test_sparse_repair_economics.py'", "```", "",
        "To re-extract and independently verify the immutable original source records in a full checkout (not required for portable regeneration):", "", "```sh",
        "python3 scripts/sparse_repair_economics.py --extract-repo . --output results/sparse_repair_01/economics --report ECONOMIC_CEILING.md", "```", "",
        "`summary.json` and `per_task_draw_comparisons.csv` regenerate deterministically. `input_manifest.json` binds the portable snapshot; the review ZIP's outer SHA-256/manifest should be checked when transporting it."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-repo", type=Path)
    parser.add_argument("--input", type=Path, default=OUT / "input_records.json")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=Path("ECONOMIC_CEILING.md"))
    args = parser.parse_args()
    if args.extract_repo is not None:
        extract(args.extract_repo, args.output)
        args.input = args.output / "input_records.json"
    manifest = read_json(args.input.parent / "input_manifest.json")
    checked(args.input.parent, manifest["file"], manifest["sha256"])
    if manifest["file"] != args.input.name:
        raise ValueError("Input manifest file mismatch")
    summary, pairs = analyze(read_json(args.input))
    summary["input_sha256"] = sha(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "summary.json", summary)
    with (args.output / "per_task_draw_comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(pairs)
    args.report.write_text(render(summary, sha(args.input)))
    print(json.dumps({"perfect_seconds": summary["perfect"]["mean_seconds"], "vs_D": summary["comparisons"]["D"]}, indent=2))


if __name__ == "__main__":
    main()
