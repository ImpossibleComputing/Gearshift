import pytest
from scripts import coding_confirmation_secondary_regional_stage_v2 as stage
from scripts.coding_confirmation_status import delegated_summary, SECONDARY_CONDITIONS


def test_delegated_fr_queue_rejected_before_provider_or_lease_access(tmp_path, monkeypatch):
    owner = tmp_path / 'configs/coding_pilot_v1/confirmation_01/secondary_fr_execution_owner_v1.json'
    owner.parent.mkdir(parents=True)
    owner.write_text('{}')
    monkeypatch.setattr(stage, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='ownership is delegated'):
        stage.stage('not_allocated', 'not_a_partition', 'fr', 'not_a_declaration', launch=True)


def test_other_regions_are_not_blocked_by_fr_ownership(tmp_path, monkeypatch):
    owner = tmp_path / 'configs/coding_pilot_v1/confirmation_01/secondary_fr_execution_owner_v1.json'
    owner.parent.mkdir(parents=True)
    owner.write_text('{}')
    monkeypatch.setattr(stage, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Unsafe allocation'):
        stage.stage('../invalid', 'not_a_partition', 'usco', 'not_a_declaration', launch=True)


def delegated_fixture():
    def shard(n):
        return {'readable':True,'completed_answers':4*n,
            'answers_by_condition':{c:n for c in SECONDARY_CONDITIONS},'current_workers':[]}
    summary={'shards':{'origin':shard(1),'usco':shard(2),'fr':shard(0)}}
    owner={'source_shard':'fr','destination_volume_id':'v','destination_root':'/separate','task_count':125}
    pods=[{'id':pid,'mounts':{'network':[{'volumeId':'v','path':'/workspace'}]}} for pid in ['a','b']]
    d={'epoch':1,'completed_answers':12,'answers_by_condition':{c:3 for c in SECONDARY_CONDITIONS},
       'jobs_completed':{'receiver':1},'ready_primary_controls':47,'workers':[],'failures':[]}
    snapshots=[{'pod_id':pid,'snapshot':{'delegated_secondary':d}} for pid in ['a','b']]
    return summary,pods,snapshots,owner


def test_delegated_store_counted_once_despite_multiple_reader_pods():
    value=delegated_summary(*delegated_fixture())
    assert value['totals']['completed_answers']==24
    assert value['shards']['fr']['execution_location']=='US-CO-1'


def test_delegated_and_original_activity_is_an_error():
    args=list(delegated_fixture());args[0]['shards']['fr']['completed_answers']=1
    with pytest.raises(ValueError,match='original-region activity'):
        delegated_summary(*args)


def test_unreadable_delegated_store_is_missing_coverage_not_claimed_zero():
    args=list(delegated_fixture());args[2]=[]
    value=delegated_summary(*args)
    assert value['coverage_is_partial_observation']
    assert not value['shards']['fr']['readable']
