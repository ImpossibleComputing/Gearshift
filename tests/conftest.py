"""Explicit optional integration dependencies of the sanitized OSS distribution.

No blanket failure suppression: only named tests needing omitted historical
benchmark inputs / upstream tokenizer blobs are marked, with a dependency reason.
All original test bodies are retained; the private archive retains their inputs.
"""
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
OPTIONAL={
 'test_coding_confirmation_regional_generation.py::test_source_helpers_must_be_committed_before_actual_stage': ('data/coding_pilot_v1/visible/confirmation.json','historical visible confirmation inputs omitted; pinned upstream retrieval required'),
 'test_coding_confirmation_replication.py::test_frozen_second_seed_exact_recipe_and104_history_population': ('results/coding_pilot_v1/history_recovery_20260916_01/history01/tasks/atcoder__abc302_b/source_history.json','private historical 104-history token corpus intentionally not redistributed'),
 'test_coding_confirmation_replication.py::test_frozen_contract_rejects_recipe_drift': ('results/coding_pilot_v1/history_recovery_20260916_01/history01/tasks/atcoder__abc302_b/source_history.json','private historical 104-history token corpus intentionally not redistributed'),
 'test_coding_confirmation_replication_supervisor.py::test_stage_inventory_verifies_all_compact_inputs_and_requires_original_initializer': ('results/coding_pilot_v1/history_recovery_20260916_01/history01/tasks/atcoder__abc302_b/source_history.json','private historical 104-history token corpus intentionally not redistributed'),
 'test_phase2_inspection.py::test_plan_is_unarmed_and_geometry_uses_explicit_head_dimension': ('configs/coding_pilot_v1/reference/Qwen3-32B/tokenizer.json','optional pinned upstream tokenizer blob is not bundled'),
}
def pytest_collection_modifyitems(items):
    for item in items:
        key=item.nodeid.removeprefix('tests/').split('[',1)[0]
        if key in OPTIONAL:
            dependency,reason=OPTIONAL[key]
            if not (ROOT/dependency).is_file():item.add_marker(pytest.mark.skip(reason=reason))
