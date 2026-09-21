"""Merge authentic synthetic primary transactions, without executing candidates."""
import importlib.util
from pathlib import Path
import shutil

import pytest

from gearshift.coding_control import sha, write
from scripts import coding_confirmation_shard_merge as m

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shard_merge_primary_fixture', ROOT / 'tests/test_coding_confirmation_generate.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def shard(tmp_path, sid, index):
    c = f.context(tmp_path, 2); f.complete_fixture(c); tids = c['declaration']['task_ids']
    for i, tid in enumerate(tids):
        if i != index:
            shutil.rmtree(m.primary.task_folder(c, tid))
            for kind in ['source', 'small', 'receiver']:
                shutil.rmtree(c['top'] / 'primary/jobs' / (kind + '_' + tid.replace('/', '__')))
    old = c['top'] / 'workers/test'; new = c['top'] / 'workers' / ('pod_' + sid) / 'attempts/a'
    new.parent.mkdir(parents=True); old.rename(new)
    for p in (c['top'] / 'primary/jobs').glob('*/complete.json'):
        row = m.read(p)
        row['runtime_path'] = str((new / 'runtime.json').relative_to(c['top']))
        row['setup_path'] = str((new / 'model_setup.json').relative_to(c['top'])); write(p, row)
    c.update(partition_sha256='3' * 64, shard_id=sid, allowed_task_ids=[tids[index]],
        partition={'shards': {'origin': [tids[0]], 'other': [tids[1]]}, 'shard_storage': {
            name: {'network_volume_id': 'volume-' + name, 'allowed_result_root': '/workspace/confirmation/results'}
            for name in ['origin', 'other']}}, quiescence={'files': []},
        plan={'adapter_source_manifest_sha256': '4' * 64, 'execution_mode': 'generation'})
    store = c['partition']['shard_storage'][sid]
    lease = {'experiment_id': c['declaration']['experiment_id'], 'pod_id': 'pod-' + sid, **store}
    mount = {**lease, 'kind': 'verified_generation_store', 'provider_mount_verified': True}
    write(new / 'operational_storage.json', {'lease': lease, 'provider_mount_receipt': mount,
        'lease_file_sha256': '5' * 64, 'provider_mount_receipt_file_sha256': '6' * 64})
    write(new / 'operational_shard.json', {'partition_sha256': c['partition_sha256'], 'shard_id': sid,
        'adapter_source_manifest_sha256': c['plan']['adapter_source_manifest_sha256'],
        'shard_storage': store, 'lease_sha256': '5' * 64, 'provider_mount_receipt_sha256': '6' * 64,
        'storage_proof_sha256': sha(new / 'operational_storage.json'),
        'numerical_primary_code_changed': False, 'analysis_population_or_seed_changed': False})
    return c


def merge_context(c, root): return {**c, 'top': root, 'root': root, 'plan': {**c['plan'], 'execution_mode': 'merge'}}


def import_one(c, merged):
    manifest = 'sharding/shards/' + c['shard_id'] + '.json'
    return m.import_shard(merge_context(c, merged), c['top'], manifest, sha(c['top'] / manifest))


def test_two_shards_merge_without_scientific_changes_and_original_closure_passes(tmp_path):
    a = shard(tmp_path / 'a', 'origin', 0); b = shard(tmp_path / 'b', 'other', 1); merged = tmp_path / 'merged'
    for c in [a, b]:
        seal = m.seal_shard(c)
        assert seal['answer_count'] == 24 and not seal['whole_primary_complete_claimed']
        assert not (c['top'] / 'primary/generation_closure.json').exists()
        import_one(c, merged)
        import_one(c, merged)  # Byte-identical replay is idempotent.
        for item in seal['files']: assert sha(merged / item['path']) == item['sha256']
    closed = m.merged_closure(merge_context(a, merged))
    assert closed['answer_count'] == 48 and closed['task_count'] == 2
    assert m.primary.generation_closure(merge_context(a, merged), seal=False) == closed


def test_shard_never_fakes_missing_job_or_whole_primary_completion(tmp_path):
    c = shard(tmp_path, 'origin', 0)
    next((c['top'] / 'primary/jobs').glob('receiver_*/complete.json')).unlink()
    assert m.seal_shard(c) is None
    assert not (c['top'] / 'sharding/shards/origin.json').exists()


@pytest.mark.parametrize('mutation', ['missing_answer', 'timing', 'control', 'new_admission', 'model_setup', 'storage_proof'])
def test_invalid_completed_task_cannot_enter_shard_manifest(tmp_path, mutation):
    c = shard(tmp_path, 'origin', 0); folder = m.primary.task_folder(c, c['allowed_task_ids'][0])
    if mutation == 'missing_answer': (folder / 'FIXED_M/seed_0/answer.json').unlink()
    elif mutation == 'timing': (folder / 'FIXED_M/seed_0/sampler/completion_timing.json').unlink()
    elif mutation == 'control':
        p = next((folder / 'controls').glob('*.json')); row = m.read(p); row['passed'] = False; write(p, row)
    elif mutation == 'new_admission': next((c['top'] / 'workers').rglob('operational_shard.json')).unlink()
    elif mutation == 'storage_proof': next((c['top'] / 'workers').rglob('operational_storage.json')).unlink()
    else:
        p = next((c['top'] / 'workers').rglob('model_setup.json')); row = m.read(p)
        row['mapper_initialization_seconds'] = -1; write(p, row)
    with pytest.raises((ValueError, FileNotFoundError)): m.seal_shard(c)


def test_changed_source_packet_or_conflicting_destination_never_overwrites(tmp_path):
    c = shard(tmp_path / 'source', 'origin', 0); receipt = m.seal_shard(c); merged = tmp_path / 'merged'
    item = next(r for r in receipt['files'] if r['path'].endswith('/answer.json'))
    target = merged / item['path']; write(target, {'existing_other_output': True}); before = target.read_bytes()
    with pytest.raises(ValueError, match='collision'): import_one(c, merged)
    assert target.read_bytes() == before and not (merged / 'sharding/imports/origin.json').exists()
    target.unlink(); (c['top'] / item['path']).write_text('{}')
    with pytest.raises(ValueError): import_one(c, merged)


def test_live_generation_tree_cannot_be_merge_destination(tmp_path):
    c = shard(tmp_path / 'source', 'origin', 0); m.seal_shard(c); merged = tmp_path / 'merged'
    (merged / 'claims').mkdir(parents=True)
    with pytest.raises(ValueError, match='CPU-only'): import_one(c, merged)


def test_missing_or_changed_shard_blocks_global_scoring_barrier(tmp_path):
    a = shard(tmp_path / 'a', 'origin', 0); b = shard(tmp_path / 'b', 'other', 1); merged = tmp_path / 'merged'
    m.seal_shard(a); m.seal_shard(b); import_one(a, merged)
    with pytest.raises(ValueError, match='Every disjoint shard'): m.merged_closure(merge_context(a, merged))
    import_one(b, merged)
    next((merged / 'primary/tasks').glob('*/A/seed_0/answer.json')).write_text('{}')
    with pytest.raises(ValueError): m.merged_closure(merge_context(a, merged))
    assert not (merged / 'primary/generation_closure.json').exists()


def test_merge_functions_refuse_generation_dispatch(tmp_path):
    c = shard(tmp_path / 'source', 'origin', 0); m.seal_shard(c)
    with pytest.raises(ValueError, match='merge dispatch'):
        m.import_shard(c, c['top'], 'sharding/shards/origin.json', sha(c['top'] / 'sharding/shards/origin.json'))
    with pytest.raises(ValueError, match='merge dispatch'): m.merged_closure(c)
