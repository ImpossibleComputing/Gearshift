import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("freeze", ROOT / "scripts/sparse_repair_freeze.py")
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


def histories():
    return [{"task_id": f"task/{i:02}", "path": f"task{i}/source_history.json", "sha256": str(i),
             "prompt_tokens": 5, "historical_reasoning_positions": (i // 2) + 10,
             "natural_boundary": True, "reasoning_capped": False,
             "token_field_hashes_compact_json_utf8": {}} for i in range(40)]


class SampleFreezeTests(unittest.TestCase):
    def test_stratified_counts_disjoint_and_all_original_preserved(self):
        cal, screen, pop = freeze.sample(histories())
        self.assertEqual((len(cal), len(screen), len(pop)), (4, 12, 40))
        self.assertFalse({r["task_id"] for r in cal} & {r["task_id"] for r in screen})
        for s in range(4):
            self.assertEqual(sum(r["stratum"] == s for r in cal), 1)
            self.assertEqual(sum(r["stratum"] == s for r in screen), 3)
            self.assertEqual(sum(r["stratum"] == s for r in pop), 10)

    def test_input_order_independent_and_outcomes_ignored(self):
        rows = histories()
        before = freeze.sample(rows)
        for i, row in enumerate(rows):
            row["passed"] = i % 2 == 0
        self.assertEqual(before, freeze.sample(list(reversed(rows))))

    def test_rejects_missing_and_duplicate_population(self):
        with self.assertRaises(ValueError):
            freeze.sample(histories()[:-1])
        with self.assertRaises(ValueError):
            freeze.sample(histories()[:-1] + [histories()[0]])

    def test_existing_answer_stream_identity(self):
        self.assertEqual(freeze.seed_for("leetcode/3731", 0, "answer_small"), 5674809076917256414)
        self.assertEqual(freeze.seed_for("leetcode/3731", 0, "coverage_answer_1"), 4604223465193410274)
        self.assertEqual(freeze.seed_for("leetcode/3731", 0, "coverage_answer_2"), 2555872696271698909)

    def test_matrix_fixed_fourteen(self):
        conditions = freeze.conditions()
        self.assertEqual(len(conditions), 14)
        self.assertEqual(len({c["name"] for c in conditions}), 14)
        self.assertEqual({c["fraction"] for c in conditions if c["name"].startswith("R_")}, {.05, .1, .25})
        self.assertEqual(12 * 3 * len(conditions), 504)


if __name__ == "__main__":
    unittest.main()
