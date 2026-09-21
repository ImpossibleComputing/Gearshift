import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from gearshift.coding_control import digest, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('pipeline_under_test', ROOT / 'scripts/coding_parallel_pipeline.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def setup(monkeypatch, tmp_path):
    c = tmp_path / 'evidence/coding_pilot_v1/control'; (c / 'pipeline').mkdir(parents=True)
    monkeypatch.setattr(m, 'ROOT', tmp_path); monkeypatch.setattr(m, 'C', c)
    for name in ['coding_parallel_pipeline.py', 'coding_pipeline_plan.py']:
        path = tmp_path / 'scripts' / name; path.parent.mkdir(exist_ok=True); path.write_text('# frozen\n')
    state = {'schema': 1, 'pipeline_id': 'pilot_parallel', 'approval_sha256': 'a' * 64,
        'initial_recovery_run': 'recovery_03', 'completed_runs': {}}
    path = c / 'pipeline/state.json'; write(path, state)
    return c, path, state


def complete(c, run_id, *, passed=True):
    plan = {'run_id': run_id, 'workers': [{'worker_id': 'worker_00'}]}
    folder = c / 'parallel' / run_id; write(folder / 'plan.json', plan)
    write(folder / 'complete.json', {'stage_identity': digest(plan), 'passed': passed,
        'workers': [{'worker_id': 'worker_00', 'passed': passed}], 'errors': []})


def test_stage_order_cannot_skip_memory_or_selection():
    assert m.next_stage({'completed_runs': {}}) == 'cap_recovery'
    assert m.next_stage({'completed_runs': {'cap_recovery': 'run'}}) == 'memory'
    with pytest.raises(ValueError, match='predecessor'):
        m.next_stage({'completed_runs': {'cap_recovery': 'run', 'histories': 'other'}})
    assert m.next_stage({'completed_runs': {stage: stage for stage in ('cap_recovery',) + m.STAGES}}) == 'finished'


def test_successful_process_moves_only_to_scientific_gate_check(monkeypatch, tmp_path):
    c, path, state = setup(monkeypatch, tmp_path); complete(c, 'recovery_03')
    monkeypatch.setattr(m, 'confirmed_clean', lambda run: (True, []))
    assert m.step(path) is False
    after = json.loads(path.read_text())
    assert after['completed_runs'] == {'cap_recovery': 'recovery_03'}
    assert not after.get('measurements_complete')
    assert json.loads((c / 'pipeline/status.json').read_text())['state'] == 'checking_scientific_gates'


def test_scientific_stop_does_not_dispatch_or_resume_on_restart(monkeypatch, tmp_path):
    c, path, state = setup(monkeypatch, tmp_path)
    state['completed_runs'] = {'cap_recovery': 'recovery_03'}; write(path, state)
    calls = []
    monkeypatch.setattr(m, 'builder', lambda *args: {'scientific_stop': True, 'stage': 'memory', 'reason': 'cap gate failed'})
    monkeypatch.setattr(m, 'launch_plan', lambda *args: calls.append(args))
    monkeypatch.setattr(m, 'publish_report', lambda *args: True)
    assert m.step(path) is True
    assert calls == []
    monkeypatch.setattr(m, 'builder', lambda *args: pytest.fail('Scientific stop was retried'))
    assert m.step(path) is True


def test_failed_stage_is_not_reinterpreted_as_success(monkeypatch, tmp_path):
    c, path, state = setup(monkeypatch, tmp_path); complete(c, 'recovery_03', passed=False)
    with pytest.raises(RuntimeError, match='did not complete'): m.step(path)
    assert json.loads(path.read_text())['completed_runs'] == {}


def test_cleanup_must_complete_before_next_stage(monkeypatch, tmp_path):
    c, path, state = setup(monkeypatch, tmp_path); complete(c, 'recovery_03')
    monkeypatch.setattr(m, 'confirmed_clean', lambda run: (False, [('pod', 'still-present')]))
    assert not m.step(path)
    assert json.loads(path.read_text())['completed_runs'] == {}
    assert json.loads((c / 'pipeline/status.json').read_text())['state'] == 'waiting_for_verified_cleanup'


def test_existing_dispatch_is_observed_not_relaunched(monkeypatch, tmp_path):
    c, path, state = setup(monkeypatch, tmp_path)
    plan = {'stage': 'memory', 'run_id': 'memory_01'}
    p = tmp_path / 'configs/memory.json'; write(p, plan)
    state.update(completed_runs={'cap_recovery': 'recovery_03'}, current_stage='memory',
        current_plan='configs/memory.json', current_plan_sha256=m.sha(p), dispatch_requested_epoch=m.time.time())
    write(path, state)
    monkeypatch.setattr(m, 'launch_plan', lambda *args: pytest.fail('Duplicate dispatch'))
    assert not m.step(path)
    assert json.loads((c / 'pipeline/status.json').read_text())['state'] == 'waiting_for_stage'


def test_final_validation_is_required_after_every_process_completed(monkeypatch, tmp_path):
    c, path, state = setup(monkeypatch, tmp_path)
    state['completed_runs'] = {stage: stage for stage in ('cap_recovery',) + m.STAGES}; write(path, state)
    def builder(stage, state_path, output, saved):
        assert stage == 'finished'; write(output, {'passed': False, 'reason': 'missing confirmation task'})
    monkeypatch.setattr(m, 'builder', builder)
    with pytest.raises(ValueError, match='Final cohort'): m.step(path)
    assert not json.loads(path.read_text()).get('measurements_complete')


def test_forecast_sums_sequential_stages_but_not_parallel_worker_gpu_hours():
    state = {'completed_runs': {'cap_recovery': 'old'}, 'current_stage': 'memory',
        'dispatch_requested_epoch': 100, 'target_completion_epoch': 1000, 'report_estimate_seconds': 100,
        'stage_limits_seconds': {stage: {'estimated_seconds': 200} for stage in m.STAGES}}
    result = m.forecast(state, now=200)
    assert result['forecast_remaining_wall_seconds'] == len(m.STAGES) * 200
    assert result['forecast_completion_epoch'] == 1600
    assert result['target_at_risk']
    assert not result['current_stage_over_estimate']


def test_unknown_or_overrun_estimates_do_not_produce_false_finish_time():
    state = {'completed_runs': {'cap_recovery': 'old'}, 'current_stage': 'memory',
        'dispatch_requested_epoch': 100, 'stage_limits_seconds': {'memory': {'estimated_seconds': 10}}}
    result = m.forecast(state, now=200)
    assert result['forecast_completion_epoch'] is None
    assert result['current_stage_over_estimate']
    assert 'histories' in result['stages_without_estimates']


def test_actual_builder_exit_twenty_requires_explicit_stop_proof(monkeypatch, tmp_path):
    c, path, state = setup(monkeypatch, tmp_path)
    output = tmp_path / 'configs/next.json'
    stop = {'scientific_stop': True, 'reason': 'No eligible checkpoint at 256 updates'}
    def run(command, **kwargs):
        assert command[2:4] == ['--stage', 'development']
        write(output.with_suffix('.stop.json'), stop)
        return type('Result', (), {'returncode': 20})()
    monkeypatch.setattr(m.subprocess, 'run', run)
    assert m.builder('development', path, output, state) == stop


def export_fixture(tmp_path, pipeline_id='pilot_parallel', study_complete=False):
    output = tmp_path / 'results/coding_pilot_v1' / ('review_' + pipeline_id)
    output.mkdir(parents=True)
    report = output / 'REPORT.md'; report.write_text('Bounded measured findings\n')
    bundle = tmp_path / ('gearshift_coding_review_' + pipeline_id + '.zip'); bundle.write_bytes(b'fixture')
    receipt = {'schema': 1, 'export_complete': True, 'study_complete': study_complete,
        'report_directory': str(output), 'report_files': {'REPORT.md': m.sha(report)},
        'bundle': {'path': str(bundle), 'bytes': bundle.stat().st_size, 'sha256': m.sha(bundle),
            'source_trajectories_included': True}}
    write(output / 'EXPORT.json', receipt)
    return output, bundle, receipt


def test_finished_export_is_verified_on_restart_without_republishing(monkeypatch, tmp_path):
    c, path, state = setup(monkeypatch, tmp_path); state['scientific_stop'] = {'reason': 'cap gate'}; write(path, state)
    output, bundle, receipt = export_fixture(tmp_path)
    monkeypatch.setattr(m.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Complete export was rerun'))
    assert m.step(path)
    assert json.loads(path.read_text())['review_export']['bundle_sha256'] == m.sha(bundle)
    assert json.loads((c / 'pipeline/status.json').read_text())['state'] == 'bounded_stop_report_ready'


def test_corrupt_export_or_incomplete_study_cannot_finish_pipeline(monkeypatch, tmp_path):
    c, path, state = setup(monkeypatch, tmp_path); state['measurements_complete'] = True; write(path, state)
    output, bundle, receipt = export_fixture(tmp_path, study_complete=False)
    with pytest.raises(ValueError, match='study incomplete'): m.step(path)
    bundle.write_bytes(b'changed')
    with pytest.raises(ValueError, match='bundle'): m.step(path)


def test_interrupted_partial_export_is_not_silently_overwritten(monkeypatch, tmp_path):
    c, path, state = setup(monkeypatch, tmp_path)
    state.update(scientific_stop={'reason': 'gate'}, report_requested_epoch=123); write(path, state)
    (tmp_path / 'scripts/coding_parallel_report.py').write_text('# report')
    with pytest.raises(RuntimeError, match='interrupted'): m.step(path)


@pytest.mark.parametrize('scientific_stop', [False, True])
def test_prepared_state_runs_finite_pipeline_and_verified_export_without_resource_calls(monkeypatch, tmp_path, scientific_stop):
    """Exercise the real supervisor transitions, using mocked stage computation."""
    c, path, state = setup(monkeypatch, tmp_path)
    prepared_path = os.environ.get('GEARSHIFT_PREPARED_PIPELINE_STATE')
    original_sha = None
    if prepared_path:
        original_sha = m.sha(prepared_path)
        state = json.loads(Path(prepared_path).read_text())
        assert not state.get('completed_runs') and not state.get('supervisor_sources')
    state['completed_runs'] = {}
    approval = tmp_path / 'evidence/coding_pilot_v1/control/parallel_500h_approved.json'
    write(approval, {'approved': True, 'test_only': True})
    state['approval_sha256'] = m.sha(approval)
    state['approval_path'] = str(approval.relative_to(tmp_path))
    write(path, state)
    complete(c, state['initial_recovery_run'])
    (tmp_path / 'scripts/coding_parallel_report.py').write_text('# simulated export computation\n')
    monkeypatch.setattr(m, 'cli', lambda *args: pytest.fail('Integration attempted a provider action'))
    monkeypatch.setattr(m, 'confirmed_clean', lambda run_id: (True, []))
    builds, launches, exports = [], [], []
    def build(stage, state_path, output, current):
        builds.append(stage)
        if scientific_stop:
            assert stage == 'memory'
            return {'scientific_stop': True, 'stage': stage, 'reason': 'Mocked actual baseline gate failed'}
        if stage == 'finished':
            write(output, {'passed': True, 'headline_tasks': 200, 'seed_sensitivity_tasks': 40}); return
        workers = 8 if stage in ('histories', 'development', 'confirmation_with_second_seed', 'seed_sensitivity') else 2 if stage == 'training' else 1
        write(output, {'stage': stage, 'run_id': current['pipeline_id'] + '-' + stage,
            'workers': [{'worker_id': 'w' + str(i)} for i in range(workers)]})
        latest = json.loads(state_path.read_text()); latest['derived_evidence_preserved'] = stage; write(state_path, latest)
    monkeypatch.setattr(m, 'builder', build)
    monkeypatch.setattr(m, 'validate_plan', lambda plan, root, approval_sha: digest(plan))
    def popen(command, **kwargs):
        assert command[1] == 'scripts/coding_parallel_session.py'
        plan = json.loads(Path(command[-1]).read_text()); launches.append(plan['stage'])
        folder = c / 'parallel' / plan['run_id']; write(folder / 'plan.json', plan)
        write(folder / 'complete.json', {'stage_identity': digest(plan), 'passed': True,
            'workers': [{'worker_id': w['worker_id'], 'passed': True} for w in plan['workers']], 'errors': []})
        return type('Child', (), {'pid': 777})()
    monkeypatch.setattr(m.subprocess, 'Popen', popen)
    def run(command, **kwargs):
        assert command[1].endswith('scripts/coding_parallel_report.py')
        exports.append(command)
        export_fixture(tmp_path, pipeline_id=state['pipeline_id'], study_complete=not scientific_stop)
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(m.subprocess, 'run', run)
    for _ in range(25):
        if m.step(path): break
    else: pytest.fail('Finite stage chain did not terminate')
    saved = json.loads(path.read_text())
    assert saved['review_export']['bundle_sha256']
    assert launches == ([] if scientific_stop else list(m.STAGES))
    assert builds == (['memory'] if scientific_stop else list(m.STAGES) + ['finished'])
    assert len(exports) == 1
    assert m.step(path) is True
    assert len(exports) == 1  # Restart observes the verified export, never repeats computation.
    if not scientific_stop: assert saved['derived_evidence_preserved'] == 'seed_sensitivity'
    if prepared_path: assert m.sha(prepared_path) == original_sha
