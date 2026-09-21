import hashlib
import io
import json
import tarfile

import pytest

from gearshift.coding_control import digest
from scripts.coding_confirmation_completed_primary import completed_primary, CONDITIONS
from scripts.coding_confirmation_status import summarize


def packet(tmp_path, change=None):
    partition={'experiment_id':'test','declaration_sha256':'decl','shards':{'fr':['task1']},
               'shard_storage':{'fr':{'network_volume_id':'volume','allowed_result_root':'/results'}}}
    value={'kind':'authentic_primary_task_shard','experiment_id':'test','declaration_sha256':'decl',
           'partition_sha256':'partition','shard_id':'fr','task_ids':['task1'],
           'shard_storage':partition['shard_storage']['fr'],'all_assigned_tasks_complete':True,
           'whole_primary_complete_claimed':False,'private_tests_loaded':False,'answer_count':24,
           'answers':[{'contract':{'task_id':'task1','condition':c,'seed_index':s,'cohort':'primary',
                     'experiment_id':'test','declaration_sha256':'decl'}} for c in CONDITIONS for s in range(3)]}
    if change:change(value)
    value['shard_sha256']=digest(value)
    raw=json.dumps(value).encode();sha=hashlib.sha256(raw).hexdigest()
    rows=[{'path':'sharding/shards/fr.json','bytes':len(raw),'sha256':sha}]
    folder=tmp_path/'primary_fr_completed_backup';folder.mkdir()
    archive=folder/'primary_complete.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,data in [('sharding/shards/fr.json',raw),('COMPACT_MANIFEST.json',json.dumps(rows).encode())]:
            item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
    proof={'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
           'manifest_sha256':sha,'local_all_member_hashes_verified':True}
    (folder/'verified.json').write_text(json.dumps(proof))
    return partition,archive


def test_verified_completed_packet_survives_reader_release_without_claiming_live_store(tmp_path):
    partition,_=packet(tmp_path)
    saved=completed_primary('fr',partition,'partition',tmp_path)
    result=summarize(partition,[],[],completed={'fr':saved})
    assert result['totals']['completed_answers']==24
    assert not result['coverage_is_partial_observation']
    assert not result['all_shard_stores_readable']
    assert result['shards']['fr']['whole_primary_complete_claimed'] is False
    # It must never supply secondary coverage.
    secondary=summarize(partition,[],[],'secondary',completed={'fr':saved})
    assert secondary['coverage_is_partial_observation']


@pytest.mark.parametrize('change',[
    lambda v:v.update(experiment_id='old-run'),
    lambda v:v.update(partition_sha256='old-partition'),
    lambda v:v.update(task_ids=['other-task']),
    lambda v:v.update(all_assigned_tasks_complete=False),
    lambda v:v.update(answer_count=23),
    lambda v:v['answers'].__setitem__(0,v['answers'][1]),
])
def test_old_incomplete_or_duplicate_populations_are_rejected(tmp_path,change):
    partition,_=packet(tmp_path,change)
    with pytest.raises(ValueError):completed_primary('fr',partition,'partition',tmp_path)


def test_archive_corruption_is_rejected(tmp_path):
    partition,archive=packet(tmp_path)
    archive.write_bytes(archive.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='archive differs'):
        completed_primary('fr',partition,'partition',tmp_path)


def test_absent_verified_receipt_does_not_claim_completed_work(tmp_path):
    assert completed_primary('fr',{'shards':{'fr':['task1']}},'partition',tmp_path) is None
