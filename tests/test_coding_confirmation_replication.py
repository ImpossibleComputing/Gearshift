"""Second-seed isolation and identity contracts; no model loading/GPU work."""
import copy
import json
from pathlib import Path
import shutil
import sys
import types
from types import SimpleNamespace
import pytest
from gearshift.coding_control import sha, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import coding_confirmation_replication as replication


@pytest.fixture(scope='module')
def corpus():
    return replication.load_training_inputs(ROOT)


def test_frozen_second_seed_exact_recipe_and104_history_population(corpus):
    d, plan = replication.verify(ROOT)
    base, training, validation = corpus
    schedules = replication.read(ROOT / replication.SCHEDULES)
    original = replication.read(ROOT / base['schedules_path'])
    assert d['training_seed'] == 20260919 != 20260915
    assert len(training) == 104 and len(validation) == 21
    assert d['optimizer'] == base['optimizer']
    assert d['selected_checkpoint_sha256'] == replication.START_SHA
    assert d['checkpoint_updates'] == [0, 128, 256, 512, 768, 1024]
    assert schedules != original
    assert schedules == replication.paired_schedules(training, updates=1024, seed=20260919, skip=96)
    assert all(row['original_schedule_step'] == i + 97 for i, row in enumerate(schedules['FIXED']))
    proof = replication.schedule_proof(training, schedules)
    assert proof['gradient_positions_per_arm'] == 32768
    assert proof['paired_task_order_and_contributions_verified']
    assert not d['cache_reuse']['tensor_cache_reuse_enabled']
    assert not d['reserved_40_tasks_used']
    assert not plan['hidden_tests_or_confirmation_answers_permitted']


def copied_contract(tmp_path, monkeypatch, corpus):
    files = [replication.BASE, replication.DECLARATION, replication.SCHEDULES, replication.PLAN,
             replication.INVENTORY, replication.CONFIG + '/OWNER_INSTRUCTION.txt']
    files.extend(replication.UNCHANGED)
    for relative in files:
        destination = tmp_path / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    monkeypatch.setattr(replication, 'load_training_inputs', lambda repo: corpus)
    return tmp_path


@pytest.mark.parametrize('change', ['seed', 'initializer', 'schedules', 'implementation', 'endpoint'])
def test_frozen_contract_rejects_recipe_drift(tmp_path, monkeypatch, corpus, change):
    repo = copied_contract(tmp_path, monkeypatch, corpus)
    path = repo / replication.DECLARATION; d = replication.read(path)
    if change == 'seed': d['training_seed'] = 20260920; write(path, d)
    elif change == 'initializer': d['selected_checkpoint_sha256'] = 'f' * 64; write(path, d)
    elif change == 'endpoint': d['target_additional_updates'] = 512; write(path, d)
    elif change == 'schedules':
        p = repo / replication.SCHEDULES; s = replication.read(p)
        s['ROTATING'][0]['segments'][0]['positions'][0] += 1; write(p, s)
    else: (repo / replication.UNCHANGED[0]).write_text('changed numerical implementation')
    with pytest.raises(ValueError): replication.verify(repo)


def test_replication_identity_starts_original96_and_never_changes_answer_seeds():
    d = replication.read(ROOT / replication.DECLARATION)
    identities = [replication.make_checkpoint_identity(d, 'a' * 40, arm) for arm in replication.ARMS]
    assert identities[0]['selected_checkpoint_sha256'] == identities[1]['selected_checkpoint_sha256'] == replication.START_SHA
    assert identities[0]['configuration'] == identities[1]['configuration']
    assert identities[0]['configuration']['training_seed'] == 20260919
    assert identities[0]['experiment_id'].endswith('/replication')
    assert identities[0]['models'] == identities[1]['models']
    assert 'answer_seed' not in identities[0]['configuration']
    with pytest.raises(ValueError): replication.make_checkpoint_identity(d, 'z' * 40, 'FIXED')


@pytest.mark.parametrize('arm', ['FIXED', 'ROTATING'])
def test_adapter_passes_parent_guard_and_new_seed_to_exact_v2_setup_and_train(tmp_path, monkeypatch, arm):
    import coding_coverage_v2_worker as original_worker
    import coding_coverage_v2_train as original_train
    d = replication.read(ROOT / replication.DECLARATION); plan = replication.read(ROOT / replication.PLAN)
    monkeypatch.setattr(replication, 'verify', lambda repo: (copy.deepcopy(d), copy.deepcopy(plan)))
    calls = []; marker = object()
    guard = lambda: calls.append('local_parent_guard')
    c = {'guard': guard, 'publish': lambda **kw: None, 'telemetry': object(), 'identity': {'run': 'new'},
         'status_root': tmp_path, 'plan': {'code_commit': 'a' * 40}}
    def forbidden(*args, **kwargs): raise AssertionError('Obsolete v2 allocation/budget context called')
    monkeypatch.setattr(original_worker, 'context', forbidden)
    def setup(context, selected_arm):
        assert selected_arm == arm and context['guard'] is guard
        assert context['spec']['training_seed'] == 20260919
        assert context['spec']['checkpoint_identity']['selected_checkpoint_sha256'] == replication.START_SHA
        assert context['root'] == ROOT / replication.RESULT / 'arms' / arm
        assert context['declaration']['schedules_path'] == replication.SCHEDULES
        assert context['spec']['target_updates'] == 1024
        assert context['config'] == replication.read(ROOT / 'configs/coding_pilot_v1/pilot.json')
        context['guard'](); calls.append('unchanged_setup'); return marker
    def train(context, state, selected_arm):
        assert state is marker and selected_arm == arm
        calls.append('unchanged_train'); return {'completed_updates': 1024}
    monkeypatch.setattr(original_worker, 'setup', setup)
    monkeypatch.setattr(original_train, 'train', train)
    assert replication.run_replication(c, arm)['completed_updates'] == 1024
    assert calls == ['local_parent_guard', 'unchanged_setup', 'unchanged_train']
    assert 'spec' not in c and c['identity'] == {'run': 'new'}


def test_adapter_rejects_historical_result_root(tmp_path, monkeypatch):
    d = replication.read(ROOT / replication.DECLARATION); plan = replication.read(ROOT / replication.PLAN)
    monkeypatch.setattr(replication, 'verify', lambda repo: (d, plan))
    c = {'guard': lambda: None, 'publish': lambda **kw: None, 'telemetry': object(), 'identity': {},
         'status_root': tmp_path, 'code_commit': 'a' * 40,
         'root': ROOT / 'results/coding_pilot_v1/coverage_generalization_v2_20260918T233720Z/arms/FIXED'}
    with pytest.raises(ValueError, match='historical'): replication.run_replication(c, 'FIXED')


def test_worker_entrypoint_uses_new_parent_runtime_and_preserves_failure(tmp_path, monkeypatch):
    from scripts import coding_confirmation_replication_worker as worker
    events = []
    c = {'status_root': tmp_path, 'publish': lambda **kw: events.append(kw),
         'telemetry': SimpleNamespace(failure=lambda error: events.append(type(error).__name__))}
    def build_context(plan, role, worker_id, arm):
        assert (plan, role, worker_id, arm) == ('dispatch.json', 'replication_train', 'pod_FIXED', 'FIXED')
        return c
    module = types.ModuleType('scripts.coding_confirmation_runtime'); module.build_context = build_context
    monkeypatch.setitem(sys.modules, module.__name__, module)
    def fail(context, arm):
        assert context is c and arm == 'FIXED'
        raise RuntimeError('preserve actual failure')
    monkeypatch.setattr(worker, 'run_replication', fail)
    with pytest.raises(RuntimeError, match='actual failure'):
        worker.main(['--plan', 'dispatch.json', '--arm', 'FIXED', '--worker-id', 'pod_FIXED'])
    failure = json.loads((tmp_path / 'replication_failure.json').read_text())
    assert failure['error'] == 'preserve actual failure' and failure['original_v2_artifacts_modified'] is False
    assert events[-1]['state'] == 'failed'
