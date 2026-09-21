import copy

import pytest

from gearshift.coding_control import sha, write
from gearshift.coding_parallel import task_shards, validate_plan, worker_spec
from test_coding_parallel import plan_fixture


def mixed(tmp_path):
    plan = plan_fixture(tmp_path, 'confirmation', ['t' + str(i) for i in range(200)], 7)
    plan['stage'] = 'confirmation_with_second_seed'
    selection = 'results/development/selection.json'
    write(tmp_path / selection, {'objective': 'natural_handoff_boundary', 'seed': 20260915,
        'confirmation_used_for_selection': False})
    plan['files'][selection] = sha(tmp_path / selection)
    proof = 'evidence/selection.json'
    write(tmp_path / proof, {'passed': True, 'gate': 'selection', 'inputs': {selection: sha(tmp_path / selection)}})
    plan['gates']['selection'] = {'path': proof, 'sha256': sha(tmp_path / proof)}
    plan['files'][proof] = sha(tmp_path / proof)
    proof = 'evidence/initialization.json'
    write(tmp_path / proof, {'passed': True, 'gate': 'initialization', 'inputs': {'results/evidence.json': plan['files']['results/evidence.json']}})
    plan['gates']['initialization'] = {'path': proof, 'sha256': sha(tmp_path / proof)}
    plan['files'][proof] = sha(tmp_path / proof)
    old = plan['cohort_files'][0]; new = 'data/coding_pilot_v1/visible/confirmation.json'
    (tmp_path / old).rename(tmp_path / new); del plan['files'][old]
    plan['files'][new] = sha(tmp_path / new); plan['cohort_files'] = [new]
    for worker in plan['workers']:
        worker['role'] = 'confirmation'
    plan['workers'].append({'worker_id': 'seed2', 'role': 'second_seed_training', 'region': 'US-CA-2',
        'task_ids': [], 'command': ['scripts/coding_test_worker.py'], 'estimated_seconds': 7200,
        'maximum_seconds': 10800, 'input_files': [rel for rel in plan['files'] if rel != new],
        'worker_fields': {'seed': 20260916, 'objective': 'natural_handoff_boundary',
            'selection_path': selection, 'selection_sha256': sha(tmp_path / selection)}})
    return plan


def test_seven_headline_shards_run_alongside_independent_second_seed(tmp_path):
    plan = mixed(tmp_path)
    identity = validate_plan(plan, tmp_path, 'a' * 64)
    assert [w['task_ids'] for w in plan['workers'][:7]] == task_shards(plan['task_ids'], 7)
    seed = worker_spec(plan, plan['workers'][-1], identity, 123, 'allocation.json')
    assert seed['stage'] == 'training'
    assert seed['dispatch_stage'] == 'confirmation_with_second_seed'
    assert seed['seed'] == 20260916
    assert worker_spec(plan, plan['workers'][0], identity, 123, 'allocation.json')['stage'] == 'confirmation'


@pytest.mark.parametrize('mutation', ['extra_seed', 'wrong_seed', 'long_training', 'confirmation_inputs', 'confirmation_tasks', 'other_objective', 'changed_selection', 'duplicate_headline'])
def test_mixed_schedule_does_not_relax_scientific_independence(tmp_path, mutation):
    plan = mixed(tmp_path); seed = plan['workers'][-1]
    if mutation == 'extra_seed': plan['workers'][0]['role'] = 'second_seed_training'
    elif mutation == 'wrong_seed': seed['worker_fields']['seed'] = 20260917
    elif mutation == 'long_training': seed['maximum_seconds'] = 10801
    elif mutation == 'confirmation_inputs': seed['input_files'].append(plan['cohort_files'][0])
    elif mutation == 'confirmation_tasks': seed['task_ids'] = ['t0']
    elif mutation == 'other_objective': seed['worker_fields']['objective'] = 'ordinary_continuation'
    elif mutation == 'changed_selection': seed['worker_fields']['selection_sha256'] = 'b' * 64
    elif mutation == 'duplicate_headline': plan['workers'][1]['task_ids'][0] = plan['workers'][0]['task_ids'][0]
    with pytest.raises(ValueError): validate_plan(plan, tmp_path, 'a' * 64)
