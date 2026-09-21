"""Mocked-only allocation tests: no provider operations or real lease edits."""
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
from scripts import sparse_repair_cloud as cloud


@pytest.fixture
def fake_cloud(tmp_path,monkeypatch):
    monkeypatch.setattr(cloud,'EVIDENCE',tmp_path/'resources')
    monkeypatch.setattr(cloud.time,'time',lambda:10000.)
    monkeypatch.setattr(cloud,'inventory',lambda:[])
    volumes=[{'id':v,'dataCenter':k} for k,v in cloud.PUBLIC.items()]
    volumes.append({'id':'lgk3howszi','dataCenter':'EU-RO-1'})
    monkeypatch.setattr(cloud,'volume_inventory',lambda:volumes)
    calls=[]
    def api(path,method,request):
        calls.append((path,method,request))
        return {'id':f'pod{len(calls)}','cost':5.55 if 'gpu' in request else 1.5,'name':request['name']}
    monkeypatch.setattr(cloud.d,'api',api)
    monkeypatch.setattr(cloud.d,'safe',lambda x:x)
    return calls


def lease(name):
    return json.loads((cloud.EVIDENCE/name/'lease.json').read_text())


@pytest.mark.parametrize('kind,hours,want',[
    ('gpu',None,2),('cpu',None,1),('gpu',7,7),('cpu',7.5,7.5),('gpu',12,12),('gpu',.5,.5),
])
def test_defaults_and_explicit_hours(fake_cloud,kind,hours,want):
    cloud.allocate('new',kind,'US-CO-1',1,hours)
    value=lease('new')
    assert (value['deadline_epoch']-value['allocation_epoch'])/3600==want
    assert value['gpu_count']==(1 if kind=='gpu' else 0)
    assert len(fake_cloud)==1 and fake_cloud[0][1]=='POST'
    record=json.loads((cloud.EVIDENCE/'new/pre_create.json').read_text())
    assert record['requested_allocation_hours']==want


@pytest.mark.parametrize('hours',[0,-1,12.01,float('nan'),float('inf'),-float('inf'),True,'7'])
def test_invalid_hours_rejected_before_provider(monkeypatch,hours):
    monkeypatch.setattr(cloud,'inventory',lambda:pytest.fail('provider inventory must not run'))
    with pytest.raises(ValueError,match='hours must be'):
        cloud.allocate('new','gpu','US-CO-1',1,hours)


@pytest.mark.parametrize('kind,count',[('cpu',2),('cpu',4),('cpu',True),('gpu',0),('gpu',3)])
def test_count_validation_before_provider(monkeypatch,kind,count):
    monkeypatch.setattr(cloud,'inventory',lambda:pytest.fail('provider inventory must not run'))
    with pytest.raises(ValueError,match='CPU allocation requires'):
        cloud.allocate('new',kind,'US-CO-1',count,7)


def test_old_lease_bytes_immutable_and_aggregate300_preserved(fake_cloud):
    cloud.allocate('old','gpu','US-CO-1',1,2)
    old=cloud.EVIDENCE/'old/lease.json';before=old.read_bytes()
    for i in range(6):cloud.allocate(f'screen{i}','gpu','US-CO-1',1,7)
    assert old.read_bytes()==before
    with pytest.raises(ValueError,match='envelope exceeded'):
        cloud.allocate('too_much','gpu','US-CO-1',1,12)
    assert len(fake_cloud)==7
    assert old.read_bytes()==before
    with pytest.raises(ValueError,match='Existing allocation'):
        cloud.allocate('old','gpu','US-CO-1',1,12)
    assert old.read_bytes()==before


def test_hard_cap_verifier_still_gates_before_post(fake_cloud,monkeypatch):
    actual=cloud.verify_lease;seen=[]
    def reject(value):
        seen.append(value.copy())
        actual(value)
        raise ValueError('hard2500 test rejection')
    monkeypatch.setattr(cloud,'verify_lease',reject)
    with pytest.raises(ValueError,match='hard2500'):
        cloud.allocate('new','gpu','US-CO-1',1,7)
    assert not fake_cloud and seen[0]['total_cap_usd']==2500
    assert seen[0]['deadline_epoch']-seen[0]['allocation_epoch']==7*3600


def test_higher_actual_quote_shortens_lease_keeps_reservation(fake_cloud,monkeypatch):
    def costly(path,method,request):
        fake_cloud.append((path,method,request))
        return {'id':'costlier','cost':8.,'name':request['name']}
    monkeypatch.setattr(cloud.d,'api',costly)
    cloud.allocate('costlier','gpu','US-CO-1',1,7)
    value=lease('costlier');duration=(value['deadline_epoch']-value['allocation_epoch'])/3600
    assert duration<7
    assert duration*value['upper_hourly_usd']==pytest.approx(7*5.55)


def test_arm_hours_option_rejected_without_provider():
    result=subprocess.run([sys.executable,str(cloud.ROOT/'scripts/sparse_repair_cloud.py'),
        'arm','--name','not_real','--hours','7'],capture_output=True,text=True)
    assert result.returncode==2 and 'existing leases cannot be extended' in result.stderr


@pytest.mark.parametrize('profile,expected',[
    (None,{'id':'cpu5c','vcpuCount':32}),
    ('cpu5c32',{'id':'cpu5c','vcpuCount':32}),
    ('cpu5g16',{'id':'cpu5g','vcpuCount':16}),
    ('cpu3g16',{'id':'cpu3g','vcpuCount':16}),
])
def test_explicit_cpu_profiles_preserve_scope_and_upper_reservation(fake_cloud,profile,expected):
    cloud.allocate('new','cpu','US-CO-1',1,2,cpu_profile=profile)
    request=fake_cloud[0][2];value=lease('new')
    assert request['cpu']==expected and 'gpu' not in request
    assert request['dataCenterIds']==['EU-RO-1']
    assert request['mounts']=={'network':[{'volumeId':'lgk3howszi','path':'/workspace'}]}
    assert value['upper_hourly_usd']==1.5 and value['gpu_count']==0
    assert value['deadline_epoch']-value['allocation_epoch']==2*3600
    assert value['total_cap_usd']==2500 and value['cleanup_reserve_usd']==40
    assert json.loads((cloud.EVIDENCE/'new/pre_create.json').read_text())['requested_cpu_profile']==(profile or 'cpu5c32')
    assert len(fake_cloud)==1


@pytest.mark.parametrize('kind,profile',[
    ('gpu','cpu5c32'),('gpu','cpu5g16'),('gpu','cpu3g16'),('cpu','unknown'),('cpu','cpu5g'),
    ('cpu',''),('cpu',True),('cpu',16),('cpu',[]),('cpu',{}),
])
def test_bad_cpu_profile_refused_before_any_provider_read(monkeypatch,kind,profile):
    monkeypatch.setattr(cloud,'inventory',lambda:pytest.fail('provider inventory must not run'))
    monkeypatch.setattr(cloud,'volume_inventory',lambda:pytest.fail('provider volume inventory must not run'))
    monkeypatch.setattr(cloud.d,'api',lambda *args:pytest.fail('provider POST must not run'))
    with pytest.raises(ValueError,match='CPU profile'):
        cloud.allocate('not_real',kind,'US-CO-1',1,cpu_profile=profile)


def test_cpu_profile_does_not_modify_existing_lease_or_lower_rate_to_historical_quote(fake_cloud,monkeypatch):
    cloud.allocate('old','cpu','US-CO-1',1,cpu_profile='cpu5c32')
    old=cloud.EVIDENCE/'old/lease.json';original=old.read_bytes()
    def historical_quote(path,method,request):
        fake_cloud.append((path,method,request))
        return {'id':'cpu_second','cost':.736,'name':request['name']}
    monkeypatch.setattr(cloud.d,'api',historical_quote)
    cloud.allocate('new','cpu','US-CO-1',1,2,cpu_profile='cpu5g16')
    assert lease('new')['upper_hourly_usd']==1.5
    assert old.read_bytes()==original
    with pytest.raises(ValueError,match='Existing allocation'):
        cloud.allocate('old','cpu','US-CO-1',1,2,cpu_profile='cpu5g16')
    assert old.read_bytes()==original and len(fake_cloud)==2


def test_cpu_profile_failure_never_automatically_falls_back(fake_cloud,monkeypatch):
    def reject(path,method,request):
        fake_cloud.append((path,method,request));raise RuntimeError('synthetic capacity rejection')
    monkeypatch.setattr(cloud.d,'api',reject)
    with pytest.raises(RuntimeError,match='reconcile instead of retrying'):
        cloud.allocate('new','cpu','US-CO-1',1,cpu_profile='cpu5g16')
    assert len(fake_cloud)==1 and fake_cloud[0][2]['cpu']=={'id':'cpu5g','vcpuCount':16}
    assert not (cloud.EVIDENCE/'new/lease.json').exists()


def test_arm_cpu_profile_option_rejected_without_provider():
    result=subprocess.run([sys.executable,str(cloud.ROOT/'scripts/sparse_repair_cloud.py'),
        'arm','--name','not_real','--cpu-profile','cpu5g16'],capture_output=True,text=True)
    assert result.returncode==2 and 'existing leases cannot be changed' in result.stderr


def test_exact_fallback_private_destination_explicit_bound_cpu_only(fake_cloud,monkeypatch):
    monkeypatch.setattr(cloud,'volume_inventory',lambda:[{'id':cloud.storage.FALLBACK_VOLUME,
        'dataCenter':cloud.storage.FALLBACK_REGION,'size':20}])
    cloud.allocate('fallback','cpu','US-CO-1',1,2,cpu_profile='cpu5g16',private_destination='fallback_euris01')
    request=fake_cloud[0][2];value=lease('fallback')
    assert request['cpu']=={'id':'cpu5g','vcpuCount':16} and 'gpu' not in request
    assert request['dataCenterIds']==['EUR-IS-1']
    assert request['mounts']=={'network':[{'volumeId':'kzmo0a5erc','path':'/workspace'}]}
    assert value['private_storage_binding']==cloud.storage.binding_reference()
    assert value['storage_region']=='EUR-IS-1' and value['cpu_profile']=='cpu5g16'
    assert value['upper_hourly_usd']==1.5 and value['total_cap_usd']==2500
    assert len(fake_cloud)==1


@pytest.mark.parametrize('kind,destination,profile',[
    ('gpu','fallback_euris01',None),('gpu','original',None),('cpu','other','cpu5g16'),
    ('cpu',True,'cpu5g16'),('cpu',{},'cpu5g16'),('cpu','fallback_euris01',None),
    ('cpu','fallback_euris01','cpu5c32'),('cpu','fallback_euris01','cpu3g16'),
])
def test_private_destination_wrong_scope_rejected_before_provider(monkeypatch,kind,destination,profile):
    monkeypatch.setattr(cloud,'inventory',lambda:pytest.fail('No provider read allowed'))
    with pytest.raises(ValueError):
        cloud.allocate('bad',kind,'US-CO-1',1,cpu_profile=profile,private_destination=destination)


def test_fallback_bad_receipt_rejected_before_provider(monkeypatch,tmp_path):
    monkeypatch.setattr(cloud,'ROOT',tmp_path)
    monkeypatch.setattr(cloud,'inventory',lambda:pytest.fail('No provider read allowed'))
    with pytest.raises(ValueError,match='evidence regular file missing'):
        cloud.allocate('bad','cpu','US-CO-1',1,cpu_profile='cpu5g16',private_destination='fallback_euris01')


def test_fallback_wrong_size_blocks_post(fake_cloud,monkeypatch):
    monkeypatch.setattr(cloud,'volume_inventory',lambda:[{'id':'kzmo0a5erc','dataCenter':'EUR-IS-1','size':21}])
    with pytest.raises(ValueError,match='volume identity missing'):
        cloud.allocate('bad','cpu','US-CO-1',1,cpu_profile='cpu5g16',private_destination='fallback_euris01')
    assert not fake_cloud


def test_arm_private_destination_rejected_without_provider():
    result=subprocess.run([sys.executable,str(cloud.ROOT/'scripts/sparse_repair_cloud.py'),
        'arm','--name','not_real','--private-destination','fallback_euris01'],capture_output=True,text=True)
    assert result.returncode==2 and 'existing leases cannot be changed' in result.stderr


def test_explicit_cpu3_does_not_change_frozen_scorer_or_implicitly_fallback(fake_cloud,monkeypatch):
    original=cloud.ROOT/'scripts/sparse_repair_score.py'
    before=original.read_bytes()
    def reject(path,method,request):
        fake_cloud.append((path,method,request));raise RuntimeError('synthetic capacity rejection')
    monkeypatch.setattr(cloud.d,'api',reject)
    with pytest.raises(RuntimeError,match='reconcile instead of retrying'):
        cloud.allocate('new','cpu','US-CO-1',1,cpu_profile='cpu3g16')
    assert len(fake_cloud)==1 and fake_cloud[0][2]['cpu']=={'id':'cpu3g','vcpuCount':16}
    assert fake_cloud[0][2]['mounts']['network'][0]['volumeId']=='lgk3howszi'
    assert original.read_bytes()==before
