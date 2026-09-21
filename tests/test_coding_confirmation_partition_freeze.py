"""CPU-only schedule freeze checks; no live partition or candidate access."""
import copy
import importlib.util
from pathlib import Path

import pytest

from gearshift.coding_control import sha,write
from scripts import coding_confirmation_partition_freeze as p
from scripts import coding_confirmation_sharded_generate as s

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('partition_shard_fixture',ROOT/'tests/test_coding_confirmation_sharded_generate.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def task_ids(): return ['atcoder/task_'+str(i) for i in range(200)]


@pytest.mark.parametrize('indices',[[],[0,1,2],[10,100,199],list(range(50)),list(range(150,200))])
def test_capacity_assignment_is_exact_ordered_and_touched_tasks_stay_origin(indices):
    tids=task_ids(); touched=[tids[i] for i in indices]; actual=p.assignments(tids,touched)
    assert {k:len(v) for k,v in actual.items()}=={'origin':50,'usco':25,'fr':125}
    flattened=[tid for tasks in actual.values() for tid in tasks]
    assert len(flattened)==len(set(flattened))==200 and set(flattened)==set(tids)
    assert set(touched)<=set(actual['origin'])
    assert all(tasks==[tid for tid in tids if tid in set(tasks)] for tasks in actual.values())
    expected_origin=set(touched)|set([tid for tid in tids if tid not in touched][:50-len(touched)])
    assert actual['origin']==[tid for tid in tids if tid in expected_origin]
    remaining=[tid for tid in tids if tid not in expected_origin]
    assert actual['usco']==remaining[:25] and actual['fr']==remaining[25:]


@pytest.mark.parametrize('kind',['too_many_touched','duplicate_touched','unknown_touched','reordered_touched','task_loss','duplicate_task'])
def test_invalid_population_or_oversubscribed_origin_is_not_rebalanced(kind):
    tids=task_ids();touched=[]
    if kind=='too_many_touched':touched=tids[:51]
    elif kind=='duplicate_touched':touched=[tids[0],tids[0]]
    elif kind=='unknown_touched':touched=['not-declared']
    elif kind=='reordered_touched':touched=[tids[2],tids[1]]
    elif kind=='task_loss':tids.pop()
    else:tids[-1]=tids[0]
    with pytest.raises(ValueError):p.assignments(tids,touched)


def fixture(tmp_path,monkeypatch):
    c=f.fixture(tmp_path,monkeypatch);d=c['declaration'];q=c['quiescence']
    # The production helper stays fixed to real storage identities. No mount or
    # GPU is needed for the separate CPU validation context.
    monkeypatch.setattr(s,'MOUNT_PATH','/workspace')
    write(tmp_path/'owner.txt',{'owner':'synthetic test authority'})
    write(tmp_path/'amendment.md',{'owner':'all200 synthetic test'})
    d.update(primary_conditions=list(s.primary.CONDITIONS),owner_instruction_path='owner.txt',owner_instruction_sha256=sha(tmp_path/'owner.txt'))
    d['task_scope_resolution']={'authority':'direct_owner_reply','decision':'all_200',
        'task_ids':d['task_ids'],'unique_task_count':200,'amendment_path':'amendment.md','amendment_sha256':sha(tmp_path/'amendment.md')}
    write(tmp_path/'declaration.json',d);declaration_sha=sha(tmp_path/'declaration.json');q['declaration_sha256']=declaration_sha
    q['touched_task_ids']=[d['task_ids'][i] for i in [3,80,199]]
    q['files']=[{'path':'primary/tasks/'+tid.replace('/','__')+'/large_history/resume.json','bytes':2,'sha256':'a'*64}
                for tid in q['touched_task_ids']]
    inventory=[];q['allocation_receipts']=[]
    for index,pod in enumerate(['x6jy63vp7jvx6n','v7ud2o1wx86y9e']):
        stem='old'+str(index);lease=s.read(tmp_path/'old_lease.json')
        lease.update(pod_id=pod,control_relative='allocations/'+pod)
        write(tmp_path/(stem+'_lease.json'),lease)
        inventory.append({'pod_id':pod,'lease_path':stem+'_lease.json','lease_sha256':sha(tmp_path/(stem+'_lease.json'))})
        write(tmp_path/(stem+'_stopped.json'),{'all_registered_workers_stopped':True,'process_group_ownership_verified':True})
        write(tmp_path/(stem+'_manifest.json'),{'experiment_id':d['experiment_id'],'pod_id':pod,
            'files':[{'path':'allocations/'+pod+'/generation_supervisor/workers_stopped.json','sha256':sha(tmp_path/(stem+'_stopped.json'))}]})
        write(tmp_path/(stem+'_released.json'),{'experiment_id':d['experiment_id'],'pod_id':pod,
            'gpu_release_verified':True,'all_gpu_workers_stopped':True,'durable_backup_verified':True,
            'manifest_sha256':sha(tmp_path/(stem+'_manifest.json'))})
        files={key:{'path':stem+suffix,'sha256':sha(tmp_path/(stem+suffix))}
            for key,suffix in [('workers_stopped','_stopped.json'),('gpu_release_verified','_released.json'),('release_manifest','_manifest.json')]}
        q['allocation_receipts'].append({'pod_id':pod,'files':files})
    monkeypatch.setattr(p,'ORIGINAL_ALLOCATIONS',inventory)
    write(tmp_path/'quiescence.json',q)
    (tmp_path/p.SCRIPT).write_bytes((ROOT/p.SCRIPT).read_bytes())
    return {'repo':tmp_path,'output_path':'frozen/partition.json','quiescence_path':'quiescence.json',
        'quiescence_sha':sha(tmp_path/'quiescence.json'),'adapter_path':'source.json','adapter_sha':sha(tmp_path/'source.json'),
        'declaration_path':'declaration.json','declaration_sha':declaration_sha}


def test_freeze_reuses_full_frozen_validator_and_binds_capacity_and_authority(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch);real=s.public_context;calls=[]
    def observe(*a,**kw):
        result=real(*a,**kw);calls.append(result['plan']['execution_mode']);return result
    monkeypatch.setattr(s,'public_context',observe)
    receipt=p.freeze(**args);partition=s.read(tmp_path/receipt['partition_path'])
    assert calls==['merge'] and receipt['task_counts']=={'origin':50,'usco':25,'fr':125}
    assert partition['shard_storage']=={sid:{'network_volume_id':volume,'allowed_result_root':p.RESULT_ROOT} for sid,volume,_,_ in p.SHARDS}
    assert partition['primary_conditions']==s.primary.CONDITIONS and partition['primary_answer_count']==4800
    assert partition['answer_draws_per_condition']==3 and partition['owner_authority']['decision']=='all_200'
    assert partition['adapter_source_manifest_sha256']==args['adapter_sha']
    assert partition['excluded_training_pod_ids']==['rekbqruqox2bk9']
    assert partition['allocation_algorithm']['outcome_guided'] is False
    # No candidate contents exist locally. Validation needed only verified public
    # inventory metadata, while all touchedIDs remain assigned to origin.
    assert not (tmp_path/'primary/tasks').exists()
    assert not list(tmp_path.glob('.partition-validation-*'))
    before=(tmp_path/receipt['partition_path']).read_bytes()
    with pytest.raises(ValueError,match='already exists'):p.freeze(**args)
    assert (tmp_path/receipt['partition_path']).read_bytes()==before


@pytest.mark.parametrize('mutation',['wrong_lease','lease_hash','missing_pod','stop_not_verified','adapter_drift','owner_drift','task_loss','overlap','misweighted'])
def test_bad_quiescence_lease_or_assignment_cannot_create_frozen_partition(tmp_path,monkeypatch,mutation):
    args=fixture(tmp_path,monkeypatch)
    if mutation=='wrong_lease':
        inventory=copy.deepcopy(p.ORIGINAL_ALLOCATIONS)
        inventory[0].update(lease_path=inventory[1]['lease_path'],lease_sha256=inventory[1]['lease_sha256'])
        monkeypatch.setattr(p,'ORIGINAL_ALLOCATIONS',inventory)
    elif mutation=='lease_hash':(tmp_path/'old0_lease.json').write_text('{}')
    elif mutation in ['missing_pod','stop_not_verified']:
        q=s.read(tmp_path/'quiescence.json')
        if mutation=='missing_pod':q['allocation_receipts'].pop()
        else:
            write(tmp_path/'old0_stopped.json',{'all_registered_workers_stopped':False,'process_group_ownership_verified':True})
            # Bind altered bytes all the way through the release chain so the
            # substantive all-workers-stopped check, not just SHA drift, rejects.
            m=s.read(tmp_path/'old0_manifest.json');m['files'][0]['sha256']=sha(tmp_path/'old0_stopped.json');write(tmp_path/'old0_manifest.json',m)
            r=s.read(tmp_path/'old0_released.json');r['manifest_sha256']=sha(tmp_path/'old0_manifest.json');write(tmp_path/'old0_released.json',r)
            for item in q['allocation_receipts'][0]['files'].values():item['sha256']=sha(tmp_path/item['path'])
        write(tmp_path/'quiescence.json',q);args['quiescence_sha']=sha(tmp_path/'quiescence.json')
    elif mutation=='adapter_drift':(tmp_path/'scripts/coding_confirmation_shard_merge.py').write_text('changed')
    elif mutation=='owner_drift':(tmp_path/'owner.txt').write_text('changed')
    else:
        real=p.assignments
        def invalid(tids,touched):
            result=real(tids,touched)
            if mutation=='task_loss':result['fr'].pop()
            elif mutation=='misweighted':
                result['origin'].append(result['usco'].pop(0));result['origin'].sort(key=tids.index)
            else:result['usco'][0]=result['origin'][0]
            return result
        monkeypatch.setattr(p,'assignments',invalid)
    with pytest.raises(ValueError):p.freeze(**args)
    assert not (tmp_path/args['output_path']).exists()
    assert not list(tmp_path.glob('.partition-validation-*'))


def test_missing_future_quiescence_does_not_create_partition(tmp_path):
    with pytest.raises(ValueError):p.freeze(tmp_path,'partition.json','not-yet-available.json','0'*64,'source.json','0'*64)
    assert not (tmp_path/'partition.json').exists()
