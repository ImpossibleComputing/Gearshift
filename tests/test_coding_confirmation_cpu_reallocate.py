"""Read-only reconciliation and synthetic CPU replacements; no real resources."""
import copy
import io
import json
from pathlib import Path
import urllib.error

import pytest

from gearshift.coding_confirmation_lease import atomic_json,sha256_file,verify_lease
from scripts import coding_confirmation_cpu_reallocate as cpu

NAME='scoring_cpu_primary01'


def old_lease(pod=cpu.ORIGINAL_CPU):
    return {'experiment_id':cpu.dispatch.EXPERIMENT,'pod_id':pod,'allocation_epoch':1000.,'deadline_epoch':87400.,
        'upper_hourly_usd':1.,'gpu_count':0,'other_reserved_usd':cpu.dispatch.ENVELOPE-24,
        'baseline_usd':cpu.dispatch.BASELINE,'baseline_gpu_hours':91.,'total_cap_usd':2500,
        'total_cap_gpu_hours':None,'cleanup_reserve_usd':40,'allowed_result_root':cpu.PRIVATE_REMOTE+'/'+cpu.dispatch.RESULT,
        'network_volume_id':cpu.PRIVATE_VOLUME,'control_relative':'allocations/'+pod}


def evidence(tmp_path):
    root=tmp_path/'resources';p=root/'scoring_cpu';p.mkdir(parents=True)
    atomic_json(p/'lease.json',old_lease())
    atomic_json(p/'volume.json',{'id':cpu.PRIVATE_VOLUME,'name':cpu.PRIVATE_VOLUME_NAME,'size':20,'dataCenterId':cpu.PRIVATE_REGION})
    return root


def absent():return urllib.error.HTTPError('https://provider/pods/old',404,'Not Found',{},io.BytesIO(b'{}'))


def current_pod():
    return {'id':'newcpu','name':'gearshift-confirmation-'+NAME,'status':'PROVISIONING','gpu':None,
        'cpu':{'id':cpu.CPU_ID,'vcpuCount':16},'dataCenterId':cpu.PRIVATE_REGION,'cost':.736,
        'mounts':{'network':[{'path':'/workspace','volumeId':cpu.PRIVATE_VOLUME}]}}


class Provider:
    def __init__(self):
        self.calls=[];self.rows=[];self.previous=absent();self.created=current_pod();self.post_error=None
        self.volume={'id':cpu.PRIVATE_VOLUME,'name':cpu.PRIVATE_VOLUME_NAME,'size':20,'dataCenter':cpu.PRIVATE_REGION,'type':'STANDARD'}
        self.quote={'id':cpu.CPU_ID,'ramGbPerVcpu':4,'vcpu':{'min':2,'max':32},'price':{'securePerVcpu':.046}}
    def __call__(self,path,method='GET',body=None):
        self.calls.append((path,method,copy.deepcopy(body)))
        if method=='POST':
            assert path=='pods'
            if self.post_error:raise self.post_error
            return copy.deepcopy(self.created)
        assert method=='GET'
        if path.startswith('pods?'):return {'pods':copy.deepcopy(self.rows),'pagination':{'hasNextPage':False,'nextCursor':None}}
        if path.startswith('pods/'):
            if isinstance(self.previous,BaseException):raise self.previous
            return self.previous
        if path=='network-volumes/'+cpu.PRIVATE_VOLUME:return copy.deepcopy(self.volume)
        if path=='catalog/cpus/'+cpu.CPU_ID:return copy.deepcopy(self.quote)
        pytest.fail('Unexpected provider request: '+path)


def test_read_only_reconciliation_keeps_historical_reservations_and_exact_volume(tmp_path):
    root=evidence(tmp_path);api=Provider();before={p:sha256_file(p) for p in root.rglob('*.json')}
    value=cpu.reconcile(NAME,evidence=root,api=api,now=100000.)
    assert all(method=='GET' for _,method,_ in api.calls)
    assert value['retained_volume']['id']==cpu.PRIVATE_VOLUME and value['create_request']['dataCenterIds']==[cpu.PRIVATE_REGION]
    assert 'gpu' not in value['create_request'] and value['create_request']['cpu']=={'id':'cpu5g','vcpuCount':16}
    assert value['reservation']['reserved_total_usd']==pytest.approx(cpu.dispatch.BASELINE+cpu.dispatch.ENVELOPE+40+24)
    assert value['lease']['gpu_count']==0 and value['lease']['total_cap_gpu_hours'] is None
    assert value['old_cpu_absence'][0]['provider_get_http_status']==404
    assert {p:sha256_file(p) for p in before}==before and not (root/NAME).exists()


@pytest.mark.parametrize('previous',[{'id':cpu.ORIGINAL_CPU,'status':'RUNNING'}, {'id':cpu.ORIGINAL_CPU,'status':'EXITED'},
    urllib.error.HTTPError('x',403,'Forbidden',{},None),urllib.error.HTTPError('x',503,'Unavailable',{},None),TimeoutError('outage')])
def test_live_stopped_or_ambiguous_previous_cpu_blocks(tmp_path,previous):
    api=Provider();api.previous=previous
    with pytest.raises((ValueError,RuntimeError)):cpu.reconcile(NAME,evidence=evidence(tmp_path),api=api)
    assert all(method=='GET' for _,method,_ in api.calls)


def test_inventory_and_get_must_both_establish_absence(tmp_path):
    api=Provider();api.rows=[{'id':cpu.ORIGINAL_CPU,'name':'old','status':'EXITED','mounts':{}}]
    with pytest.raises(ValueError,match='still exists'):cpu.reconcile(NAME,evidence=evidence(tmp_path),api=api)


@pytest.mark.parametrize('kind',['ambiguous','reservation_only','unrecorded_namespace_cpu','attached_elsewhere'])
def test_previous_unknown_allocations_cannot_be_bypassed_by_a_new_name(tmp_path,kind):
    root=evidence(tmp_path);api=Provider()
    if kind=='ambiguous':
        atomic_json(root/'previous_try/create_request.json',{'cpu':{'id':'cpu5g'}})
        atomic_json(root/'previous_try/ambiguous_creation.json',{'reconciliation_required_before_retry':True})
    elif kind=='reservation_only':atomic_json(root/'previous_try/reservation.json',{'lease':{'gpu_count':0}})
    elif kind=='unrecorded_namespace_cpu':
        api.rows=[{**current_pod(),'id':'unknowncpu','mounts':{}}]
    else:api.rows=[{**current_pod(),'id':'unrelated','name':'unrelated-pod','status':'EXITED'}]
    with pytest.raises(ValueError):cpu.reconcile(NAME,evidence=root,api=api)


@pytest.mark.parametrize('field,value',[('id','other'),('name','renamed'),('size',21),('dataCenter','US-GA-2'),('type','HIGH_PERFORMANCE')])
def test_volume_identity_mismatch_refuses_reuse(tmp_path,field,value):
    api=Provider();api.volume[field]=value
    with pytest.raises(ValueError,match='volume identity'):cpu.reconcile(NAME,evidence=evidence(tmp_path),api=api)


@pytest.mark.parametrize('change',['gpu_lease','scope','volume','authority','price','ram','over_budget'])
def test_scope_quote_and_budget_invariants(tmp_path,change):
    root=evidence(tmp_path);api=Provider();p=root/'scoring_cpu/lease.json';lease=cpu.read(p)
    if change=='gpu_lease':lease['gpu_count']=1
    elif change=='scope':lease['allowed_result_root']='/workspace/Public/results/confirmation'
    elif change=='volume':lease['network_volume_id']='public_volume'
    elif change=='authority':lease['total_cap_usd']=3000
    elif change=='price':api.quote['price']['securePerVcpu']=.1
    elif change=='ram':api.quote['ramGbPerVcpu']=2
    else:lease['other_reserved_usd']=2490-lease['baseline_usd']-40-24
    atomic_json(p,lease)
    with pytest.raises(ValueError):cpu.reconcile(NAME,evidence=root,api=api)


def test_successive_replacement_retains_prior_full_cpu_reservations(tmp_path):
    root=evidence(tmp_path);first=cpu.reconcile(NAME,evidence=root,api=Provider(),now=100000.)
    second=dict(first['lease'],pod_id='released_replacement',control_relative='allocations/released_replacement')
    atomic_json(root/'scoring_cpu_first_replacement/lease.json',second)
    value=cpu.reconcile(NAME,evidence=root,api=Provider(),now=200000.)
    assert value['prior_reserved_total_usd']==first['reservation']['reserved_total_usd']
    assert value['reservation']['reserved_total_usd']==pytest.approx(first['reservation']['reserved_total_usd']+24)
    assert len(value['old_cpu_absence'])==2


def test_missing_generation_seal_blocks_before_any_provider_call_or_reservation(tmp_path,monkeypatch):
    root=evidence(tmp_path);api=Provider()
    def blocked(*args):raise ValueError('Primary generation seal incomplete')
    monkeypatch.setattr(cpu,'public_generation_ready',blocked)
    with pytest.raises(ValueError,match='incomplete'):cpu.allocate(NAME,'plan.json','a'*64,repo=tmp_path,evidence=root,api=api)
    assert not api.calls and not (root/NAME).exists()


def test_allocate_uses_one_cpu_post_reuses_volume_and_preserves_all_old_bytes(tmp_path,monkeypatch):
    root=evidence(tmp_path);api=Provider();old=(root/'scoring_cpu/lease.json').read_bytes()
    monkeypatch.setattr(cpu,'public_generation_ready',lambda *a:{'whole_public_generation_recomputed':True})
    value=cpu.allocate(NAME,'public_plan.json','a'*64,repo=tmp_path,evidence=root,api=api,now=100000.)
    posts=[x for x in api.calls if x[1]!='GET'];assert len(posts)==1 and posts[0][0]=='pods' and posts[0][1]=='POST'
    assert posts[0][2]['mounts']=={'network':[{'volumeId':cpu.PRIVATE_VOLUME,'path':'/workspace'}]}
    assert (root/'scoring_cpu/lease.json').read_bytes()==old
    lease=cpu.read(root/NAME/'lease.json');assert lease['pod_id']=='newcpu' and lease['gpu_count']==0
    assert value['guard_armed'] is False and 'Immediately' in value['next_action']
    assert not (root/NAME/'volume.json').exists() and (root/NAME/'volume_reuse.json').exists()


def test_ambiguous_post_is_durable_and_never_retried_under_another_name(tmp_path,monkeypatch):
    root=evidence(tmp_path);api=Provider();api.post_error=TimeoutError('secret provider detail must not be logged')
    monkeypatch.setattr(cpu,'public_generation_ready',lambda *a:{'whole_public_generation_recomputed':True})
    with pytest.raises(TimeoutError):cpu.allocate(NAME,'public_plan.json','a'*64,repo=tmp_path,evidence=root,api=api)
    assert len([x for x in api.calls if x[1]=='POST'])==1
    failure=cpu.read(root/NAME/'ambiguous_creation.json');assert failure['error_type']=='TimeoutError' and 'secret' not in json.dumps(failure)
    with pytest.raises(ValueError,match='Unresolved previous'):cpu.reconcile('scoring_cpu_next',evidence=root,api=api)


def test_create_price_race_shortens_lease_without_increasing_reserved_dollars(tmp_path,monkeypatch):
    root=evidence(tmp_path);api=Provider();api.created['cost']=1.2
    monkeypatch.setattr(cpu,'public_generation_ready',lambda *a:{'whole_public_generation_recomputed':True})
    value=cpu.allocate(NAME,'public_plan.json','a'*64,repo=tmp_path,evidence=root,api=api,now=100000.)
    lease=cpu.read(root/NAME/'lease.json')
    assert lease['upper_hourly_usd']==pytest.approx(1.49)
    assert lease['deadline_epoch']-lease['allocation_epoch']<86400
    assert value['reservation']['reserved_total_usd']==pytest.approx(cpu.dispatch.BASELINE+cpu.dispatch.ENVELOPE+40+24)


def test_failed_post_scope_retains_owned_identity_and_blocks_future_allocation(tmp_path,monkeypatch):
    root=evidence(tmp_path);api=Provider();api.created['cpu']['vcpuCount']=8
    monkeypatch.setattr(cpu,'public_generation_ready',lambda *a:{'whole_public_generation_recomputed':True})
    with pytest.raises(ValueError,match='Created pod scope'):cpu.allocate(NAME,'p.json','a'*64,repo=tmp_path,evidence=root,api=api)
    assert cpu.read(root/NAME/'lease.json')['pod_id']=='newcpu'
    assert cpu.read(root/NAME/'scope_failure.json')['do_not_stage_private_inputs_or_candidates'] is True


def test_complete_paginated_inventory_required():
    calls=[]
    def api(path,method='GET'):
        calls.append(path)
        if len(calls)==1:return {'pods':[{'id':'one'}],'pagination':{'hasNextPage':True,'nextCursor':'token/next'}}
        return {'pods':[{'id':'two'}],'pagination':{'hasNextPage':False,'nextCursor':None}}
    assert [x['id'] for x in cpu.list_pods(api)]==['one','two']
    assert calls[-1].endswith('cursor=token%2Fnext')
    with pytest.raises(ValueError,match='pagination'):cpu.list_pods(lambda *a,**kw:{'pods':[],'pagination':{'hasNextPage':True,'nextCursor':None}})


@pytest.mark.parametrize('name',['../scoring_cpu_test','primary_pair03','scoring_cpu','scoring_cpu_UPPER'])
def test_namespace_is_scoring_only(name,tmp_path):
    with pytest.raises(ValueError,match='dedicated'):cpu.reconcile(name,evidence=evidence(tmp_path),api=Provider())


def test_real_public_validator_rejects_missing_seal_without_private_access(tmp_path,monkeypatch):
    import importlib.util
    path=Path(__file__).resolve().parent/'test_coding_confirmation_score.py'
    spec=importlib.util.spec_from_file_location('cpu_public_fixture',path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    original=fixture.gfixture.context
    def context(path,count=1):
        c=original(path,count);c['declaration']['experiment_id']=cpu.dispatch.EXPERIMENT;return c
    monkeypatch.setattr(fixture.gfixture,'context',context)
    c,private=fixture.fixture(tmp_path);private.unlink()
    plan='generation_plan.json';expected=sha256_file(tmp_path/plan)
    closed=cpu.public_generation_ready(tmp_path,plan,expected)
    assert closed['whole_public_generation_recomputed'] and closed['answer_count']==24 and closed['private_bytes_read'] is False
    (tmp_path/'results/primary/generation_closure.json').unlink()
    api=Provider();root=evidence(tmp_path)
    with pytest.raises((ValueError,FileNotFoundError)):
        cpu.allocate(NAME,plan,expected,repo=tmp_path,evidence=root,api=api)
    assert not api.calls and not (root/NAME).exists()
