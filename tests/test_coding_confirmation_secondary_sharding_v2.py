"""Synthetic secondary packets must preserve pairing and authentic public inputs."""
import importlib.util
from pathlib import Path
import shutil

import pytest

from gearshift.coding_control import sha, write
from scripts import coding_confirmation_secondary_shard_merge_v2 as m
from scripts import coding_confirmation_secondary_sharded_generate_v2 as g
from scripts import coding_confirmation_secondary_sharded_supervisor_v2 as supervisor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('secondary_fixture', ROOT / 'tests/test_coding_confirmation_secondary_generate.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def shard(tmp_path, sid, index):
    c = f.secondary_fixture(tmp_path, 2); tids = c['declaration']['task_ids']
    for i, tid in enumerate(tids):
        if i != index:
            shutil.rmtree(m.secondary.task_folder(c, tid))
            shutil.rmtree(m.secondary.job_path(c, tid).parent)
    pod = 'pod_' + sid
    runtime = c['top'] / 'workers' / (pod + '_secondary_0') / 'attempts/a'
    shutil.copytree(c['top'] / 'workers/test', runtime)
    for path in (c['top'] / 'secondary/jobs').glob('*/complete.json'):
        row = m.read(path)
        row.update(runtime_path=str((runtime / 'runtime.json').relative_to(c['top'])),
                   setup_path=str((runtime / 'model_setup.json').relative_to(c['top'])))
        write(path, row)
    c.update(partition_sha256='3' * 64, shard_id=sid, allowed_task_ids=[tids[index]],
        partition={'shards': {'origin': [tids[0]], 'other': [tids[1]]}, 'shard_storage': {
            name: {'network_volume_id': 'volume-' + name, 'allowed_result_root': '/workspace/confirmation/results'}
            for name in ['origin', 'other']}},
        plan={'secondary_shard_source_manifest_sha256': '4' * 64, 'execution_mode': 'generation'})
    store = c['partition']['shard_storage'][sid]
    lease = {'experiment_id': c['declaration']['experiment_id'], 'pod_id': pod, **store,
        'allocation_epoch': 1000., 'deadline_epoch': 4600., 'upper_hourly_usd': 5., 'gpu_count': 1,
        'baseline_usd': 500., 'total_cap_usd': 2500, 'total_cap_gpu_hours': None,
        'cleanup_reserve_usd': 40, 'other_reserved_usd': 0, 'control_relative': 'allocations/' + pod}
    mount = {**lease, 'kind': 'verified_generation_store', 'provider_mount_verified': True}
    provider = {'pod_id': pod, 'network_volume_id': store['network_volume_id'],
                'allocation_role': g.ROLE, 'exclusive_gpu_allocation': True}
    write(runtime / 'operational_secondary_storage.json', {'lease': lease,
        'provider_mount_receipt': mount, 'provider_inspection': provider})
    write(runtime / 'secondary_checkpoint_verification.json', {
        'secondary_declaration_sha256': c['secondary_declaration_sha256'],
        'checkpoints': c['secondary_checkpoints'], 'both_final_mapper_and_full_bytes_verified': True})
    write(runtime / 'operational_secondary_shard.json', {
        'partition_sha256': c['partition_sha256'], 'shard_id': sid, 'shard_storage': store,
        'secondary_declaration_sha256': c['secondary_declaration_sha256'],
        'secondary_shard_source_manifest_sha256': c['plan']['secondary_shard_source_manifest_sha256'],
        'worker_role': g.ROLE, 'numerical_secondary_code_changed': False, 'primary_artifacts_modified': False,
        'storage_proof_sha256': sha(runtime / 'operational_secondary_storage.json'),
        'checkpoint_verification_sha256': sha(runtime / 'secondary_checkpoint_verification.json'),
        'idle_gpu_before_load': {'free_fraction_before_load': .99, 'other_compute_owners': False}})
    return c


def merge_context(c, root):
    return {**c, 'top': root, 'root': root / 'secondary', 'plan': {**c['plan'], 'execution_mode': 'merge'}}


def import_one(c, root):
    path = 'secondary/sharding/shards/' + c['shard_id'] + '.json'
    return m.import_shard(merge_context(c, root), c['top'], path, sha(c['top'] / path))


def test_merge_preserves_primary_and_closes_all_secondary_draws(tmp_path):
    a = shard(tmp_path / 'a', 'origin', 0); b = shard(tmp_path / 'b', 'other', 1)
    merged = tmp_path / 'merged'; merged.mkdir()
    shutil.copytree(a['top'] / 'primary', merged / 'primary')
    shutil.copytree(a['top'] / 'workers/test', merged / 'workers/test')
    original = {str(p.relative_to(merged)): sha(p) for p in (merged / 'primary').rglob('*') if p.is_file()}
    for c in [a, b]:
        receipt = m.seal_shard(c)
        assert receipt['answer_count'] == 12 and not receipt['whole_secondary_complete_claimed']
        import_one(c, merged); import_one(c, merged)
        for item in receipt['files']: assert sha(merged / item['path']) == item['sha256']
    result = m.merged_closure(merge_context(a, merged))
    assert result['answer_count'] == 24 and result['training_seed'] == 20260919
    assert all(sha(merged / p) == h for p, h in original.items())


@pytest.mark.parametrize('mutation', ['missing_draw', 'changed_control', 'missing_checkpoint_proof', 'wrong_store', 'occupied_gpu'])
def test_invalid_secondary_shard_cannot_be_sealed(tmp_path, mutation):
    c = shard(tmp_path, 'origin', 0); tid = c['allowed_task_ids'][0]
    worker = next((c['top'] / 'workers').glob('pod_*')) / 'attempts/a'
    if mutation == 'missing_draw': (m.secondary.task_folder(c, tid) / 'FIXED_M/seed_0/answer.json').unlink()
    elif mutation == 'changed_control':
        path = next((m.primary.task_folder(c, tid) / 'controls').glob('*.json'))
        value = m.read(path); value['passed'] = False; write(path, value)
    elif mutation == 'missing_checkpoint_proof': (worker / 'secondary_checkpoint_verification.json').unlink()
    else:
        path = worker / 'operational_secondary_shard.json'; value = m.read(path)
        if mutation == 'wrong_store': value['shard_storage']['network_volume_id'] = 'wrong-volume'
        else: value['idle_gpu_before_load']['other_compute_owners'] = True
        write(path, value)
    with pytest.raises((ValueError, FileNotFoundError)): m.seal_shard(c)


def test_merge_refuses_collision_and_incomplete_global_population(tmp_path):
    c = shard(tmp_path / 'source', 'origin', 0); seal = m.seal_shard(c); merged = tmp_path / 'merged'
    item = next(row for row in seal['files'] if row['path'].startswith('secondary/') and row['path'].endswith('/answer.json'))
    target = merged / item['path']; write(target, {'different': True}); before = target.read_bytes()
    with pytest.raises(ValueError, match='collision'): import_one(c, merged)
    assert target.read_bytes() == before
    target.unlink(); import_one(c, merged)
    with pytest.raises(ValueError, match='Every secondary shard'): m.merged_closure(merge_context(c, merged))


def test_secondary_claim_gate_denies_foreign_task_without_acquiring_lock(tmp_path, monkeypatch):
    c = {'top': tmp_path, 'root': tmp_path / 'secondary', 'allowed_task_ids': ['task/one']}
    with g.selected_claims(c):
        with g.secondary.claim_job(c, 'receiver_task__two') as acquired: assert acquired is False
        with pytest.raises(ValueError, match='namespace'):
            with g.secondary.claim_job({**c, 'root': tmp_path}, 'receiver_task__one'): pass
    assert g.secondary.claim_job is g.original_claim
    assert not (tmp_path / 'secondary').exists()


def test_registered_live_supervisor_is_not_replaced(tmp_path, monkeypatch):
    identity = {'pid': 42, 'pgid': 42, 'start_ticks': 10, 'boot_id': 'boot'}
    write(tmp_path / 'lease_guard/supervisor_registration.json', identity)
    monkeypatch.setattr(supervisor, 'linux_process_identity', lambda pid: identity)
    with pytest.raises(ValueError, match='live registered'): supervisor.refuse_live_registration(tmp_path)
    assert m.read(tmp_path / 'lease_guard/supervisor_registration.json') == identity
