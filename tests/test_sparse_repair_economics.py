import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("economics", ROOT / "scripts/sparse_repair_economics.py")
economics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(economics)


def fixture():
    rows = []
    for tid in ("t0", "t1"):
        for c in economics.CONDITIONS:
            for seed in range(3):
                row = {"task_id": tid, "condition": c, "seed_index": seed, "answer_seed": seed + 7,
                       "passed": seed != 1, "missing": False, "answer_tokens": 20,
                       "source_cost_divisor_for_single_output": 1, "inference_timing_complete": True,
                       "reasoning_seconds": 100 if c != "B" else 50, "answer_seconds": 10,
                       "native_prefill_seconds": 2 if c == "D" else 0,
                       "mapping_seconds": 0, "splice_seconds": 0, "cache_clone_seconds": .5}
                row["single_output_inference_seconds"] = sum(row[k] for k in economics.STAGES)
                rows.append(row)
    return {"task_ids": ["t0", "t1"], "task_count": 2, "conditions": list(economics.CONDITIONS),
            "draws_per_task_condition": 3, "cohort": "primary", "rows": rows}


class EconomicReferenceTests(unittest.TestCase):
    def test_only_prefill_removed_no_reasoning_amortization_or_clone_removal(self):
        summary, pairs = economics.analyze(fixture(), expected_task_count=2)
        self.assertEqual(summary["perfect"]["mean_seconds"], 110.5)
        self.assertEqual(summary["comparisons"]["D"]["saving_seconds_mean"], 2)
        self.assertEqual(summary["comparisons"]["B"]["saving_seconds_mean"], -50)
        self.assertEqual(summary["perfect"]["passed"], 4)
        self.assertEqual(len(pairs), 6)

    def test_rejects_absent_failed_draw_instead_of_changing_population(self):
        data = fixture()
        data["rows"] = [r for r in data["rows"] if not (r["condition"] == "D" and r["seed_index"] == 1)]
        with self.assertRaisesRegex(ValueError, "population"):
            economics.analyze(data, expected_task_count=2)

    def test_rejects_duplicate(self):
        data = fixture()
        data["rows"].append(copy.deepcopy(data["rows"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            economics.analyze(data, expected_task_count=2)

    def test_rejects_amortized_source(self):
        data = fixture()
        data["rows"][0]["source_cost_divisor_for_single_output"] = 3
        with self.assertRaisesRegex(ValueError, "amortized"):
            economics.analyze(data, expected_task_count=2)

    def test_rejects_nan_or_negative_timing(self):
        for value in (float("nan"), -1):
            data = fixture()
            data["rows"][0]["reasoning_seconds"] = value
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                economics.analyze(data, expected_task_count=2)

    def test_rejects_stage_sum_drift(self):
        data = fixture()
        data["rows"][0]["single_output_inference_seconds"] += 1
        with self.assertRaisesRegex(ValueError, "Stage sum"):
            economics.analyze(data, expected_task_count=2)

    def test_rejects_different_answer_draws(self):
        data = fixture()
        data["rows"][0]["answer_seed"] += 1
        with self.assertRaisesRegex(ValueError, "Mismatched answer"):
            economics.analyze(data, expected_task_count=2)

    def test_rejects_missing_outcome(self):
        data = fixture()
        data["rows"][0]["passed"] = None
        with self.assertRaisesRegex(ValueError, "Incomplete outcomes"):
            economics.analyze(data, expected_task_count=2)

    def test_checksum_rejects_mutation_and_unscoped_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "data.json"
            path.write_text("{}")
            expected = economics.sha(path)
            self.assertEqual(economics.checked(root, "data.json", expected), path)
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                economics.checked(root, "data.json", expected)
            with self.assertRaisesRegex(ValueError, "Unscoped"):
                economics.checked(root, "../data.json", expected)

    def test_real_compact_population(self):
        path = ROOT / economics.OUT / "input_records.json"
        if not path.exists():
            self.skipTest("Portable records not provided")
        summary, pairs = economics.analyze(economics.read_json(path))
        self.assertEqual(len(pairs), 600)
        self.assertEqual(summary["perfect"]["passed"], 390)
        self.assertAlmostEqual(summary["perfect"]["mean_seconds"], 725.4872205673419)
        self.assertAlmostEqual(summary["comparisons"]["D"]["saving_seconds_mean"], 1.467934876512736)


if __name__ == "__main__":
    unittest.main()
