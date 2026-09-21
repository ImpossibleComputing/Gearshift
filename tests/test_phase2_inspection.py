import json
from pathlib import Path
import pytest
from scripts.phase2_inspection_diagnostics import code_category
from scripts.coding_pilot_plan_check import seed_for,validate
def test_code_failure_requires_saved_grader_trace():
    assert code_category({'status':'execution_failure','stderr':'AssertionError'})[0]=='assertion_failure_origin_unknown'
    assert code_category({'status':'execution_failure','stderr':'  in assertion\n    assert exact_match\nAssertionError'})[0]=='hidden_test_assertion_failure'
    assert code_category({'status':'execution_failure','stderr':''})[0]=='execution_failure_unknown'
    assert code_category({'status':'execution_failure','stderr':'TypeError: bad arguments'})[0]=='runtime_exception'
def test_seed_is_local_to_task_and_segment():
    reference=seed_for('x',0,'answer_small')
    for i in range(100):seed_for(str(i),0,'source_reasoning')
    assert seed_for('x',0,'answer_small')==reference
    assert seed_for('x',0,'small_reasoning')!=reference
    assert seed_for('x',1,'answer_small')!=reference
def test_plan_is_unarmed_and_geometry_uses_explicit_head_dimension():
    root=Path(__file__).resolve().parents[1]
    d=validate(root/'configs/coding_pilot_v1')
    assert d['execution_allowed'] is False
    assert d['geometry']['source']['head_dim']==128
    assert d['geometry']['source']['cache_bytes_per_token']==262144
    assert d['selected_tasks']==400
