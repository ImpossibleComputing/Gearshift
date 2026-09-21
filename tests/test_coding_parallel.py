import copy
import json
from pathlib import Path

import pytest

from gearshift.coding_control import sha, write
from gearshift.coding_parallel import (check_dispatch_budget, checkpoint_files,
    task_shards, validate_plan, verify_worker_status, worker_spec)


def plan_fixture(tmp_path, stage='cap_recovery', tasks=None, n=1):
    tasks = tasks or ['platform/task_' + str(i) for i in range(5)]
    gates = {'cap_recovery': ['interruption_recovery', 'original_cap_parent'],
        'confirmation': ['baseline', 'memory', 'selection']}[stage]
    inputs = {}
    for rel, text in [('scripts/coding_test_worker.py', 'print(1)'), ('results/evidence.json', '{"measured":true}')]:
        path = tmp_path / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
        inputs[rel] = sha(path)
    proofs = {}
    for name in gates:
        rel = 'evidence/' + name + '.json'
        write(tmp_path / rel, {'passed': True, 'gate': name, 'inputs': {'results/evidence.json': inputs['results/evidence.json']}})
        proofs[name] = {'path': rel, 'sha256': sha(tmp_path / rel)}
        inputs[rel] = sha(tmp_path / rel)
    cohort_rel = 'data/coding_pilot_v1/visible/development.json'
    write(tmp_path / cohort_rel, [{'task_id': tid} for tid in tasks])
    inputs[cohort_rel] = sha(tmp_path / cohort_rel)
    return {'schema': 1, 'approval_sha256': 'a' * 64, 'run_id': 'bounded_run_1',
        'stage': stage, 'image_digest': 'runpod/pytorch@sha256:' + 'b' * 64,
        'gates': proofs, 'files': inputs, 'task_ids': tasks, 'cohort_files': [cohort_rel],
        'preserve_original_measurements': True, 'scientific_scope_unchanged': True,
        'workers': [{'worker_id': 'shard_' + str(i), 'task_ids': part, 'region': 'CA-MTL-3',
            'command': ['scripts/coding_test_worker.py'], 'estimated_seconds': 1800,
            'maximum_seconds': 3600, 'worker_fields': {}}
            for i, part in enumerate(task_shards(tasks, n))]}


def budget():
    return {'stop': False, 'parallel_execution_approved': True, 'max_concurrent_gpus': 8,
        'gpu_hours': 20, 'hard_total_gpu_hours': 500, 'upper_usd': 110,
        'cleanup_dispatch_ceiling_usd': 980}


def test_complete_200_task_shards_are_disjoint_ordered_and_fixed(tmp_path):
    tasks = ['platform/task_' + str(i) for i in range(200)]
    plan = plan_fixture(tmp_path, 'confirmation', tasks, 8)
    identity = validate_plan(plan, tmp_path, 'a' * 64)
    assert len(identity) == 64
    assert all(len(w['task_ids']) == 25 for w in plan['workers'])
    assert set().union(*(set(w['task_ids']) for w in plan['workers'])) == set(tasks)
    plan['workers'][1]['task_ids'][0] = plan['workers'][0]['task_ids'][0]
    with pytest.raises(ValueError, match='fixed disjoint'):
        validate_plan(plan, tmp_path, 'a' * 64)


@pytest.mark.parametrize('mutation', ['gate_fail', 'evidence_change', 'gate_omitted', 'wrong_approval', 'moving_image', 'worker_override', 'cohort_short'])
def test_dispatch_rejects_invalid_scientific_identity_before_spending(tmp_path, mutation):
    plan = plan_fixture(tmp_path, 'confirmation', ['t' + str(i) for i in range(200)], 8)
    if mutation == 'gate_fail':
        path = tmp_path / plan['gates']['memory']['path']
        gate = json.loads(path.read_text()); gate['passed'] = False; write(path, gate)
        plan['gates']['memory']['sha256'] = sha(path); plan['files'][str(path.relative_to(tmp_path))] = sha(path)
    elif mutation == 'evidence_change': (tmp_path / 'results/evidence.json').write_text('{}')
    elif mutation == 'gate_omitted': del plan['gates']['memory']
    elif mutation == 'wrong_approval': plan['approval_sha256'] = 'c' * 64
    elif mutation == 'moving_image': plan['image_digest'] = 'runpod/pytorch:latest'
    elif mutation == 'worker_override': plan['workers'][0]['worker_fields']['task_ids'] = ['replacement']
    elif mutation == 'cohort_short':
        plan['task_ids'] = plan['task_ids'][:199]
        for w, part in zip(plan['workers'], task_shards(plan['task_ids'], 8)): w['task_ids'] = part
    with pytest.raises(ValueError): validate_plan(plan, tmp_path, 'a' * 64)


def test_stage_reservation_counts_all_simultaneous_gpu_hours_and_dollars(tmp_path):
    plan = plan_fixture(tmp_path, n=5)
    assert check_dispatch_budget(plan, budget())['reserved_gpu_hours'] == 5
    b = budget(); b['upper_usd'] = 970
    with pytest.raises(ValueError, match='dollar'): check_dispatch_budget(plan, b)
    b = budget(); b['gpu_hours'] = 498
    with pytest.raises(ValueError, match='GPU-hours'): check_dispatch_budget(plan, b)
    with pytest.raises(ValueError, match='Concurrent'): check_dispatch_budget(plan, budget(), active_pods=4)
    b = budget(); b['dispatch_blocked'] = True
    with pytest.raises(ValueError, match='Dispatch blocked'): check_dispatch_budget(plan, b)


def test_shards_do_not_change_task_seed_streams(tmp_path):
    from gearshift.coding_control import seed_for
    plan = plan_fixture(tmp_path, n=3)
    expected = {tid: seed_for(tid, 0, 'answer_small') for tid in plan['task_ids']}
    observed = {tid: seed_for(tid, 0, 'answer_small') for w in plan['workers'] for tid in w['task_ids']}
    assert observed == expected


def test_worker_paths_and_status_are_isolated(tmp_path):
    plan = plan_fixture(tmp_path, n=2)
    specs = [worker_spec(plan, w, 'identity', 12345, 'allocation.json') for w in plan['workers']]
    assert specs[0]['result_root'] != specs[1]['result_root']
    assert specs[0]['worker_status_path'] != specs[1]['worker_status_path']
    status = {'stage_identity': 'identity', 'worker_id': specs[0]['worker_id'], 'state': 'complete'}
    assert verify_worker_status(status, specs[0]) == 'complete'
    with pytest.raises(ValueError, match='identity'): verify_worker_status(status, specs[1])


@pytest.mark.parametrize('path', ['../model.pt', '/weights/model.pt', 'results/other/mapper.pt', 'results/run/worker/model.bin'])
def test_checkpoint_backup_excludes_unowned_or_full_weight_paths(path):
    with pytest.raises(ValueError): checkpoint_files({'files': {path: 'a' * 64}}, 'results/run/worker')


def test_preserved_checkpoint_manifest_requires_frozen_hashes():
    rel = 'results/run/worker/mapper_step_128.safetensors'
    assert checkpoint_files({'files': {rel: 'a' * 64}}, 'results/run/worker') == {rel: 'a' * 64}
    with pytest.raises(ValueError): checkpoint_files({'files': {rel: 'mutable'}}, 'results/run/worker')


def test_gate_evidence_must_be_transferred_to_each_worker(tmp_path):
    plan = plan_fixture(tmp_path)
    del plan['files']['results/evidence.json']
    with pytest.raises(ValueError, match='absent'): validate_plan(plan, tmp_path, 'a' * 64)


def test_symlink_input_is_rejected(tmp_path):
    plan = plan_fixture(tmp_path)
    path = tmp_path / 'results/evidence.json'; saved = path.with_name('original.json')
    path.rename(saved); path.symlink_to(saved)
    with pytest.raises(ValueError, match='unsafe'): validate_plan(plan, tmp_path, 'a' * 64)
