import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("screen", ROOT / "scripts/sparse_repair_screen.py")
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)


def make_gate(root):
    tasks = [{"task_id": f"task/{i}", "history_sha256": str(i)} for i in range(4)]
    impl = {p: screen.sha(ROOT / p) for p in screen.NUMERICAL_FILES + ("scripts/sparse_repair_calibrate.py",)}
    runtime = {"torch": "2.8.0+cu128", "cuda": "12.8", "transformers": "4.57.6", "gpu": "NVIDIA H200",
               "BF16": True, "TF32": False, "deterministic_algorithms": True}
    for t in tasks:
        folder = root / t["task_id"].replace("/", "__")
        result = {**t, "declaration_sha256": "frozen", "passed": True,
                  "all_same_path_controls_passed": True, "backing_caches_unchanged": True,
                  "implementation_hashes": impl, "runtime": runtime}
        screen.write(folder / "calibration.json", result)
        screen.write(folder / "complete.json", {"declaration_sha256": "frozen", "passed": True,
                     "calibration_sha256": screen.sha(folder / "calibration.json")})
    return {"calibration_tasks": tasks}


class ScreenGateTests(unittest.TestCase):
    def test_four_actual_cuda_receipts_required(self):
        with tempfile.TemporaryDirectory() as f:
            root = Path(f); declaration = make_gate(root)
            result = screen.validate_calibration(ROOT, declaration, "frozen", root)
            self.assertTrue(result["all_four_passed"])
            self.assertEqual(len(result["cases"]), 4)

    def test_missing_fourth_case_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            root = Path(f); declaration = make_gate(root)
            (root / "task__3/complete.json").unlink()
            with self.assertRaises(FileNotFoundError):
                screen.validate_calibration(ROOT, declaration, "frozen", root)

    def test_failed_stale_cpu_or_mutated_receipt_rejected(self):
        for alteration in ("failed", "implementation", "cpu", "declaration", "unbound_bytes"):
            with tempfile.TemporaryDirectory() as f:
                root = Path(f); declaration = make_gate(root)
                path = root / "task__0/calibration.json"
                data = screen.read(path)
                if alteration == "failed":
                    data["passed"] = False
                elif alteration == "implementation":
                    data["implementation_hashes"]["gearshift/sparse_repair.py"] = "stale"
                elif alteration == "cpu":
                    data["runtime"]["gpu"] = "CPU"
                elif alteration == "declaration":
                    data["declaration_sha256"] = "different"
                else:
                    data["unbound_mutation"] = True
                screen.write(path, data)
                if alteration != "unbound_bytes":
                    done = screen.read(path.parent / "complete.json")
                    done["calibration_sha256"] = screen.sha(path)
                    screen.write(path.parent / "complete.json", done)
                with self.assertRaises(ValueError):
                    screen.validate_calibration(ROOT, declaration, "frozen", root)

    def test_matrix_mapping(self):
        self.assertEqual(len(screen.CONDITIONS), 14)
        self.assertEqual(screen.condition_parameters("R_random"), ("R", .1, "random"))
        self.assertEqual(screen.condition_parameters("N_5"), ("N", .05, "native_mass"))
        self.assertEqual(screen.condition_parameters("M_25"), ("M", .25, "native_mass"))
        self.assertEqual(screen.condition_parameters("H")[0], "disabled")

    def test_plan_requires_frozen_runner_and_source_commit(self):
        plan = {"declaration_sha256": "frozen", "implementation_hashes": screen.implementation_hashes(ROOT), "code_commit": "a" * 40}
        self.assertEqual(screen.validate_plan_implementation(plan, ROOT, "frozen"), plan["implementation_hashes"])
        plan["implementation_hashes"]["scripts/sparse_repair_screen.py"] = "changed"
        with self.assertRaises(ValueError):
            screen.validate_plan_implementation(plan, ROOT, "frozen")

    def test_completed_draw_checks_descendant_bytes(self):
        with tempfile.TemporaryDirectory() as f:
            root = Path(f); identity = {"task_id": "task/0", "condition": "D"}
            screen.write(root / "answer.json", {**identity, "answer_ids": [1, 2]})
            screen.write(root / "complete.json", {"identity": identity, "files": {"answer.json": screen.sha(root / "answer.json")}})
            self.assertEqual(screen.verify_draw(root, identity)["answer_ids"], [1, 2])
            screen.write(root / "answer.json", {**identity, "answer_ids": [2, 3]})
            with self.assertRaises(ValueError):
                screen.verify_draw(root, identity)


if __name__ == "__main__":
    unittest.main()
