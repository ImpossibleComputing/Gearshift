from scripts.coding_confirmation_status import summarize, CONDITIONS, SECONDARY_CONDITIONS


def snapshot(epoch, count):
    return {'epoch': epoch, 'completed_answers': count,
            'answers_by_condition': {c: count if c == 'A' else 0 for c in CONDITIONS},
            'source_histories': 1, 'small_histories': 0, 'jobs_completed': {},
            'workers': [], 'failures': []}


def test_same_volume_is_counted_once_and_stale_original_volume_excluded():
    partition = {'shard_storage': {'fr': {'network_volume_id': 'fr-vol'}}, 'shards': {'fr': ['task1']}}
    pods = [{'id': p, 'mounts': {'network': [{'volumeId': v, 'path': '/workspace'}]}}
            for p, v in [('fr4', 'fr-vol'), ('fr1', 'fr-vol'), ('old', 'old-vol')]]
    rows = [{'pod_id': p, 'snapshot': snapshot(e, n)}
            for p, e, n in [('fr4', 1, 2), ('fr1', 2, 3), ('old', 3, 100)]]
    value = summarize(partition, pods, rows)
    assert value['totals']['completed_answers'] == 3
    assert value['totals']['source_histories'] == 1
    assert value['all_shard_stores_readable']


def test_unreadable_store_cannot_be_reported_as_complete_coverage():
    value = summarize({'shard_storage': {'origin': {'network_volume_id': 'v'}},
                       'shards': {'origin': ['task']}}, [], [])
    assert not value['all_shard_stores_readable']
    assert value['coverage_is_partial_observation']
    assert not value['shards']['origin']['readable']


def test_primary_and_secondary_counts_and_workers_stay_separate_on_shared_store():
    partition = {'shard_storage': {'fr': {'network_volume_id': 'v'}}, 'shards': {'fr': ['task1']}}
    pods = [{'id': p, 'mounts': {'network': [{'volumeId': 'v', 'path': '/workspace'}]}} for p in ['primary', 'secondary']]
    value = snapshot(2, 24)
    value.update(secondary_completed_answers=12,
        secondary_answers_by_condition={c:3 for c in SECONDARY_CONDITIONS}, secondary_jobs_completed={'receiver':1},
        workers=[{'pod_id':p,'worker_id':p+suffix+'0'} for p,suffix in [('primary','_generation_'),('secondary','_secondary_')]],
        failures=[{'path':'/results/workers/secondary_secondary_0/failure.json'}])
    rows = [{'pod_id':p,'snapshot':value} for p in ['primary','secondary']]
    first=summarize(partition,pods,rows); second=summarize(partition,pods,rows,'secondary')
    assert first['totals']['completed_answers']==24 and second['totals']['completed_answers']==12
    assert len(first['shards']['fr']['current_workers'])==len(second['shards']['fr']['current_workers'])==1
    assert first['shards']['fr']['current_workers'][0]['pod_id']=='primary'
    assert second['shards']['fr']['current_workers'][0]['pod_id']=='secondary'
    assert not first['shards']['fr']['current_worker_failures']
    assert len(second['shards']['fr']['current_worker_failures'])==1
    assert 'source_histories' not in second['totals']
