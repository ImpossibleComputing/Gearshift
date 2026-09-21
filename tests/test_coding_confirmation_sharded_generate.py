"""Static operational admission tests; no model or private-test execution."""
import copy
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path

import pytest

from gearshift.coding_control import digest, sha, write
from scripts import coding_confirmation_sharded_generate as s

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sharding_primary_fixture', ROOT / 'tests/test_coding_confirmation_generate.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'MOUNT_PATH', str(tmp_path))
    _, d = f.declaration_fixture(tmp_path)
    tids = ['atcoder/task_' + str(i) for i in range(200)]
    write(tmp_path / 'seeds.json', f.seeds(tids))
    write(tmp_path / 'visible.json', [{'task_id': t, 'prompt': 'public', 'prompt_ids': [1, 2]} for t in tids])
    d.update(task_ids=tids, task_count=200, primary_answer_count=4800, source_commit='b' * 40)
    d['inputs'].update({p: sha(tmp_path / p) for p in ['seeds.json', 'visible.json']})
    write(tmp_path / 'declaration.json', d); declaration_sha = sha(tmp_path / 'declaration.json')
    for name in s.IMPLEMENTATION:
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# frozen operational source fixture\n')
    write(tmp_path / 'source.json', {'schema': 1, 'operational_commit': 'c' * 40,
        'files': {p: sha(tmp_path / p) for p in s.IMPLEMENTATION}, 'numerical_primary_code_changed': False,
        'interposition': 'claim_job task admission only', 'analysis_population_or_seed_changed': False})
    write(tmp_path / 'stopped.json', {'all_registered_workers_stopped': True, 'process_group_ownership_verified': True})
    old_lease = {'experiment_id': d['experiment_id'], 'pod_id': 'old_pod', 'allocation_epoch': 1,
        'deadline_epoch': 3600, 'upper_hourly_usd': 5, 'baseline_usd': 100, 'total_cap_usd': 2500,
        'cleanup_reserve_usd': 40, 'gpu_count': 1, 'allowed_result_root': str(tmp_path / 'original'),
        'control_relative': 'allocations/old_pod'}
    write(tmp_path / 'old_lease.json', old_lease)
    write(tmp_path / 'release_manifest.json', {'experiment_id': d['experiment_id'], 'pod_id': 'old_pod',
        'files': [{'path': 'allocations/old_pod/generation_supervisor/workers_stopped.json',
                   'sha256': sha(tmp_path / 'stopped.json')}]})
    write(tmp_path / 'released.json', {'experiment_id': d['experiment_id'], 'pod_id': 'old_pod',
        'gpu_release_verified': True, 'all_gpu_workers_stopped': True, 'durable_backup_verified': True,
        'manifest_sha256': sha(tmp_path / 'release_manifest.json')})
    q = {'schema': 1, 'experiment_id': d['experiment_id'], 'declaration_sha256': declaration_sha,
        'all_original_primary_workers_stopped': True, 'original_supervisors_released': True,
        'healthy_training_untouched': True, 'files': [], 'touched_task_ids': [],
        'allocation_receipts': [{'pod_id': 'old_pod', 'files': {
            'release_manifest': {'path': 'release_manifest.json', 'sha256': sha(tmp_path / 'release_manifest.json')},
            'workers_stopped': {'path': 'stopped.json', 'sha256': sha(tmp_path / 'stopped.json')},
            'gpu_release_verified': {'path': 'released.json', 'sha256': sha(tmp_path / 'released.json')}}}]}
    write(tmp_path / 'quiescence.json', q)
    p = {'schema': 1, 'experiment_id': d['experiment_id'], 'declaration_sha256': declaration_sha,
        'source_commit': d['source_commit'], 'task_ids': tids, 'task_count': 200, 'whole_task_assignment': True,
        'static_after_freeze': True, 'shards': {'origin': tids[:100], 'new_region': tids[100:]},
        'shard_storage': {'origin': {'network_volume_id': 'volume-origin', 'allowed_result_root': str(tmp_path / 'results')},
            'new_region': {'network_volume_id': 'volume-other', 'allowed_result_root': str(tmp_path / 'other-results')}},
        'original_generation_allocations': [{'pod_id': 'old_pod', 'lease_path': 'old_lease.json', 'lease_sha256': sha(tmp_path / 'old_lease.json')}],
        'excluded_training_pod_ids': ['training_pod'],
        'origin_shard_id': 'origin', 'touched_task_ids': [], 'quiescence_path': 'quiescence.json',
        'quiescence_sha256': sha(tmp_path / 'quiescence.json')}
    write(tmp_path / 'partition.json', p)
    lease = {'experiment_id': d['experiment_id'], 'pod_id': 'new_pod', 'allocation_epoch': 100,
        'deadline_epoch': 3600, 'upper_hourly_usd': 5, 'baseline_usd': 100, 'total_cap_usd': 2500,
        'cleanup_reserve_usd': 40, 'gpu_count': 1, **p['shard_storage']['origin']}
    write(tmp_path / 'lease.json', lease)
    dev = tmp_path.stat().st_dev
    mount = {'mount_point': str(tmp_path), 'device': str(os.major(dev)) + ':' + str(os.minor(dev)),
        'filesystem_type': 'nfs4', 'source': 'network-volume:/volume-origin'}
    receipt = {'schema': 1, 'kind': 'verified_generation_store', 'experiment_id': d['experiment_id'],
        'pod_id': lease['pod_id'], **p['shard_storage']['origin'], 'mount_path': str(tmp_path),
        'provider_inspected_epoch': 200, 'staged_epoch': 201, 'provider_mount_verified': True, 'local_mount': mount}
    observation = {'pod_id': lease['pod_id'], 'network_volume_id': lease['network_volume_id'],
        'volume_mount_path': str(tmp_path), 'observed_epoch': 200, 'source': 'runpod_provider_inspection'}
    write(tmp_path / 'observation.json', observation)
    receipt.update(provider_inspection_path='observation.json', provider_inspection_sha256=sha(tmp_path / 'observation.json'))
    write(tmp_path / 'mount.json', receipt)
    monkeypatch.setattr(s, 'workspace_mount', lambda: mount)
    monkeypatch.setenv('RUNPOD_POD_ID', lease['pod_id'])
    plan = {'execution_mode': 'generation', 'pod_id': lease['pod_id'],
        'lease_path': 'lease.json', 'lease_sha256': sha(tmp_path / 'lease.json'),
        'provider_mount_receipt_path': 'mount.json', 'provider_mount_receipt_sha256': sha(tmp_path / 'mount.json'),
        'experiment_id': d['experiment_id'], 'declaration_path': 'declaration.json', 'declaration_sha256': declaration_sha,
        'code_commit': d['source_commit'], 'partition_path': 'partition.json', 'partition_sha256': sha(tmp_path / 'partition.json'),
        'adapter_source_manifest_path': 'source.json', 'adapter_source_manifest_sha256': sha(tmp_path / 'source.json'),
        'shard_id': 'origin', 'result_root': 'results'}
    write(tmp_path / 'plan.json', plan); monkeypatch.setattr(s, 'ROOT', tmp_path)
    return s.public_context(tmp_path, 'plan.json', sha(tmp_path / 'plan.json'))


def test_full200_partition_is_scientifically_unchanged_and_disjoint(tmp_path, monkeypatch):
    c = fixture(tmp_path, monkeypatch)
    assert len(c['declaration']['task_ids']) == 200 and len(c['allowed_task_ids']) == 100
    assert c['plan']['code_commit'] == c['declaration']['source_commit']


@pytest.mark.parametrize('mutation', ['overlap', 'missing', 'reordered', 'touched_elsewhere', 'not_quiesced', 'no_receipts', 'different_source', 'missing_storage', 'duplicate_store', 'relative_store'])
def test_invalid_partition_fails_closed(tmp_path, monkeypatch, mutation):
    c = fixture(tmp_path, monkeypatch); p = copy.deepcopy(c['partition']); q = copy.deepcopy(c['quiescence'])
    if mutation == 'overlap': p['shards']['new_region'].append(p['shards']['origin'][0])
    elif mutation == 'missing': p['shards']['origin'].pop()
    elif mutation == 'reordered': p['shards']['origin'].reverse()
    elif mutation == 'not_quiesced': q['all_original_primary_workers_stopped'] = False
    elif mutation == 'no_receipts': q['allocation_receipts'] = []
    elif mutation == 'different_source': p['source_commit'] = '0' * 40
    elif mutation == 'missing_storage': del p['shard_storage']['origin']
    elif mutation == 'duplicate_store': p['shard_storage']['new_region'] = p['shard_storage']['origin']
    elif mutation == 'relative_store': p['shard_storage']['origin']['allowed_result_root'] = 'results'
    else:
        tid = p['shards']['new_region'][0]
        q['files'] = [{'path': 'primary/tasks/' + tid.replace('/', '__') + '/large_history/resume.json', 'bytes': 2, 'sha256': 'f' * 64}]
        q['touched_task_ids'] = p['touched_task_ids'] = [tid]
    with pytest.raises(ValueError): s.validate_partition(c['declaration'], p, q)


@pytest.mark.parametrize('file', ['plan.json', 'partition.json', 'source.json', 'scripts/coding_confirmation_shard_merge.py', 'stopped.json', 'lease.json', 'mount.json', 'observation.json', 'old_lease.json', 'release_manifest.json'])
def test_mutable_operational_or_quiescence_inputs_block_before_admission(tmp_path, monkeypatch, file):
    c = fixture(tmp_path, monkeypatch); (tmp_path / file).write_text('changed')
    with pytest.raises(ValueError): s.public_context(tmp_path, 'plan.json', c['plan_sha256'])


def test_gate_only_admits_its_whole_tasks_and_rechecks_before_real_claim(tmp_path, monkeypatch):
    c = fixture(tmp_path, monkeypatch); c['attempt_id'] = 'new_generation_0'; admitted = []
    @contextmanager
    def claim(context, job):
        admitted.append(job); yield True
    monkeypatch.setattr(s, 'original_claim', claim); monkeypatch.setattr(s.primary, 'claim_job', claim)
    originals = {name: getattr(s.primary, name) for name in ['work', 'initialize', 'reasoning_job', 'receiver_job', 'sample_answers']}
    with s.selected_claims(c):
        for kind in ['source', 'small', 'receiver']:
            with s.primary.claim_job(c, kind + '_atcoder__task_0') as selected: assert selected
            with s.primary.claim_job(c, kind + '_atcoder__task_199') as selected: assert not selected
        (tmp_path / 'partition.json').write_text('changed')
        with pytest.raises(ValueError):
            with s.primary.claim_job(c, 'source_atcoder__task_1'): pytest.fail('Mutable assignment admitted')
    assert len(admitted) == 4 and s.primary.claim_job is claim
    assert all(getattr(s.primary, name) is function for name, function in originals.items())
    assert not (c['top'] / 'primary/jobs').exists()  # Denial never creates fake completion.


def test_origin_snapshot_checks_exact_bytes_once_and_allows_exact_resume_updates(tmp_path, monkeypatch):
    c = fixture(tmp_path, monkeypatch); path = c['top'] / 'primary/tasks/atcoder__task_0/large_history/resume.json'
    write(path, {'durable_prefix': [1, 2]})
    c['quiescence']['files'] = [{'path': str(path.relative_to(c['top'])), 'bytes': path.stat().st_size, 'sha256': sha(path)}]
    s.verify_initial_dataset(c)
    write(path, {'durable_prefix': [1, 2, 3]}); s.verify_initial_dataset(c)
    other = c['top'] / 'primary/tasks/atcoder__task_199/unexpected.json'; write(other, {})
    with pytest.raises(ValueError, match='Another shard'): s.verify_initial_dataset(c)


def test_new_region_refuses_any_preexisting_candidate_or_sampler_state(tmp_path, monkeypatch):
    c = fixture(tmp_path, monkeypatch); c['shard_id'] = 'new_region'; c['allowed_task_ids'] = c['partition']['shards']['new_region']
    write(c['top'] / 'primary/tasks/atcoder__task_199/history.json', {'unexpected': True})
    with pytest.raises(ValueError, match='Initial shard dataset'): s.verify_initial_dataset(c)


def repin(c, file, value, hash_key):
    write(c['repo_root'] / file, value)
    c['plan'][hash_key] = sha(c['repo_root'] / file)
    write(c['repo_root'] / 'plan.json', c['plan'])
    c['plan_sha256'] = sha(c['repo_root'] / 'plan.json')


@pytest.mark.parametrize('mutation', ['volume', 'root', 'pod', 'missing_mount', 'mount_volume', 'mount_pod', 'old_inspection', 'unverified_mount'])
def test_generation_store_binding_rejects_independent_or_unverified_store(tmp_path, monkeypatch, mutation):
    c = fixture(tmp_path, monkeypatch)
    lease = s.read(tmp_path / 'lease.json'); mount = s.read(tmp_path / 'mount.json')
    if mutation == 'volume': lease['network_volume_id'] = 'independent-clone'
    elif mutation == 'root': lease['allowed_result_root'] = str(tmp_path / 'independent-results')
    elif mutation == 'pod': lease['pod_id'] = 'other-pod'
    elif mutation == 'missing_mount': (tmp_path / 'mount.json').unlink()
    elif mutation == 'mount_volume': mount['network_volume_id'] = 'independent-clone'
    elif mutation == 'mount_pod': mount['pod_id'] = 'other-pod'
    elif mutation == 'old_inspection': mount['staged_epoch'] = 501
    else: mount['provider_mount_verified'] = False
    repin(c, 'lease.json', lease, 'lease_sha256')
    if mutation != 'missing_mount': repin(c, 'mount.json', mount, 'provider_mount_receipt_sha256')
    with pytest.raises((ValueError, FileNotFoundError)):
        s.public_context(tmp_path, 'plan.json', c['plan_sha256'])


def test_two_pods_can_share_one_exact_shard_store(tmp_path, monkeypatch):
    c = fixture(tmp_path, monkeypatch)
    s.validate_generation_store(c, verify_local=True)
    lease = s.read(tmp_path / 'lease.json'); mount = s.read(tmp_path / 'mount.json')
    lease['pod_id'] = mount['pod_id'] = c['plan']['pod_id'] = 'second-pod'
    observation = s.read(tmp_path / 'observation.json'); observation['pod_id'] = 'second-pod'
    write(tmp_path / 'observation.json', observation); mount['provider_inspection_sha256'] = sha(tmp_path / 'observation.json')
    repin(c, 'lease.json', lease, 'lease_sha256'); repin(c, 'mount.json', mount, 'provider_mount_receipt_sha256')
    monkeypatch.setenv('RUNPOD_POD_ID', 'second-pod')
    fresh = s.public_context(tmp_path, 'plan.json', c['plan_sha256'])
    s.validate_generation_store(fresh, verify_local=True)
    assert fresh['partition_sha256'] == c['partition_sha256']


@pytest.mark.parametrize('mutation', ['pod', 'mount', 'result_device'])
def test_current_mount_and_pod_are_checked_locally_before_admission(tmp_path, monkeypatch, mutation):
    c = fixture(tmp_path, monkeypatch)
    if mutation == 'pod': monkeypatch.setenv('RUNPOD_POD_ID', 'wrong-pod')
    elif mutation == 'mount': monkeypatch.setattr(s, 'workspace_mount', lambda: {'mount_point': '/wrong'})
    else: monkeypatch.setattr(s.os, 'major', lambda dev: 987654)
    with pytest.raises(ValueError): s.validate_generation_store(c, verify_local=True)


def test_cpu_merge_context_has_no_local_generation_authority(tmp_path, monkeypatch):
    c = fixture(tmp_path, monkeypatch)
    c['plan'].update(execution_mode='merge', result_root='cpu-merged')
    for field in ['lease_path', 'lease_sha256', 'provider_mount_receipt_path', 'provider_mount_receipt_sha256', 'pod_id']:
        c['plan'].pop(field)
    write(tmp_path / 'plan.json', c['plan'])
    merged = s.public_context(tmp_path, 'plan.json', sha(tmp_path / 'plan.json'))
    with pytest.raises(ValueError, match='generation dispatch'): s.validate_generation_store(merged)
    c['plan']['result_root'] = 'results'; write(tmp_path / 'plan.json', c['plan'])
    with pytest.raises(ValueError, match='separate CPU'): s.public_context(tmp_path, 'plan.json', sha(tmp_path / 'plan.json'))


@pytest.mark.parametrize('mutation', ['missing_live_pod_receipt', 'extra_receipt', 'missing_inventory', 'duplicate_inventory', 'training_included'])
def test_exhaustive_original_allocation_inventory_required(tmp_path, monkeypatch, mutation):
    c = fixture(tmp_path, monkeypatch); p = copy.deepcopy(c['partition']); q = copy.deepcopy(c['quiescence'])
    if mutation == 'missing_live_pod_receipt':
        p['original_generation_allocations'].append({'pod_id': 'still_live', 'lease_path': 'lease.json', 'lease_sha256': '0' * 64})
    elif mutation == 'extra_receipt': q['allocation_receipts'].append({**q['allocation_receipts'][0], 'pod_id': 'unlisted'})
    elif mutation == 'missing_inventory': p.pop('original_generation_allocations')
    elif mutation == 'duplicate_inventory': p['original_generation_allocations'] *= 2
    else: p['excluded_training_pod_ids'] = ['old_pod']
    with pytest.raises(ValueError): s.validate_partition(c['declaration'], p, q)


def test_relabelled_stop_receipt_cannot_substitute_for_another_allocation(tmp_path, monkeypatch):
    c = fixture(tmp_path, monkeypatch)
    manifest = s.read(tmp_path / 'release_manifest.json'); manifest['files'][0]['path'] = 'allocations/some_other_pod/generation_supervisor/workers_stopped.json'
    write(tmp_path / 'release_manifest.json', manifest)
    released = s.read(tmp_path / 'released.json'); released['manifest_sha256'] = sha(tmp_path / 'release_manifest.json')
    write(tmp_path / 'released.json', released)
    q = c['quiescence']; files = q['allocation_receipts'][0]['files']
    files['release_manifest']['sha256'] = sha(tmp_path / 'release_manifest.json'); files['gpu_release_verified']['sha256'] = sha(tmp_path / 'released.json')
    write(tmp_path / 'quiescence.json', q)
    p = c['partition']; p['quiescence_sha256'] = sha(tmp_path / 'quiescence.json')
    repin(c, 'partition.json', p, 'partition_sha256')
    with pytest.raises(ValueError, match='own released allocation'): s.public_context(tmp_path, 'plan.json', c['plan_sha256'])


@pytest.mark.parametrize('count', [0, 1, 2])
def test_workspace_mount_uses_exact_kernel_mount_record(monkeypatch, count):
    monkeypatch.setattr(s, 'MOUNT_PATH', '/workspace')
    line = r'31 29 0:45 / /workspace rw,relatime - nfs4 server:/volume\040origin rw'
    text = '29 1 0:1 / / rw - overlay overlay rw\n' + '\n'.join([line] * count)
    monkeypatch.setattr(s.Path, 'read_text', lambda path: text)
    if count == 1:
        assert s.workspace_mount() == {'mount_point': '/workspace', 'device': '0:45',
            'filesystem_type': 'nfs4', 'source': 'server:/volume origin'}
    else:
        with pytest.raises(ValueError, match='Exactly one'): s.workspace_mount()
