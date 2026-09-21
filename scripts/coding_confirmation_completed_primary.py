"""Read verified completed primary packets after their live readers release.

This is progress accounting only, never a generation or scoring closure.
"""
import hashlib
import json
from pathlib import Path
import tarfile

from gearshift.coding_control import digest
from gearshift.coding_confirmation_lease import sha256_file
from scripts.coding_confirmation_replication_supervisor import verify_archive

CONDITIONS = ['A', 'B', 'D', 'P', 'FIXED_M', 'ROTATING_M', 'FIXED_H', 'ROTATING_H']


def completed_primary(sid, partition, partition_sha256, evidence):
    if sid not in partition['shards']:
        raise ValueError('Unknown primary shard')
    folder = Path(evidence) / ('primary_' + sid + '_completed_backup')
    receipt = folder / 'verified.json'
    if not receipt.is_file():
        return None
    proof = json.loads(receipt.read_text())
    archive = folder / 'primary_complete.tar.gz'
    size = proof.get('archive_bytes', proof.get('bytes'))
    expected = proof.get('archive_sha256', proof.get('sha256'))
    if (proof.get('local_all_member_hashes_verified') is not True or
            archive.stat().st_size != size or sha256_file(archive) != expected):
        raise ValueError('Completed primary archive differs from verified receipt')
    with tarfile.open(archive, 'r:gz') as tar:
        rows = json.load(tar.extractfile('COMPACT_MANIFEST.json'))
        raw = tar.extractfile('sharding/shards/' + sid + '.json').read()
    verify_archive(archive, rows)
    if hashlib.sha256(raw).hexdigest() != proof.get('manifest_sha256', proof.get('shard_manifest_sha256')):
        raise ValueError('Completed primary manifest differs')
    value = json.loads(raw)
    wanted = {'kind': 'authentic_primary_task_shard', 'experiment_id': partition['experiment_id'],
              'declaration_sha256': partition['declaration_sha256'], 'partition_sha256': partition_sha256,
              'shard_id': sid, 'task_ids': partition['shards'][sid],
              'shard_storage': partition['shard_storage'][sid], 'all_assigned_tasks_complete': True,
              'whole_primary_complete_claimed': False, 'private_tests_loaded': False}
    if (any(value.get(k) != v for k, v in wanted.items()) or
            value.get('shard_sha256') != digest({k: v for k, v in value.items() if k != 'shard_sha256'})):
        raise ValueError('Completed primary packet identity differs')
    expected_draws = {(tid, c, seed) for tid in wanted['task_ids'] for c in CONDITIONS for seed in range(3)}
    draws = [a['contract'] for a in value['answers']]
    if (value['answer_count'] != len(expected_draws) or len(draws) != len(expected_draws) or
            {(a['task_id'], a['condition'], a['seed_index']) for a in draws} != expected_draws or
            any(a.get('cohort') != 'primary' or a.get('experiment_id') != wanted['experiment_id'] or
                a.get('declaration_sha256') != wanted['declaration_sha256'] for a in draws)):
        raise ValueError('Completed primary draw population differs')
    count = len(wanted['task_ids'])
    return {'readable': True, 'live_store_readable': False, 'coverage_source': 'verified_completed_primary_packet',
            'task_count': count, 'network_volume_id': wanted['shard_storage']['network_volume_id'],
            'completed_answers': len(draws), 'answers_by_condition': {c: 3 * count for c in CONDITIONS},
            'jobs_completed': {k: count for k in ['source', 'small', 'receiver']},
            'source_histories': count, 'small_histories': count,
            'current_workers': [], 'current_worker_failures': [],
            'archive_path': str(archive), 'archive_sha256': expected, 'archive_bytes': size,
            'files': len(rows), 'receipt_path': str(receipt), 'whole_primary_complete_claimed': False}
