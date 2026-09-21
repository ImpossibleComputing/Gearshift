"""Synthetic placement/idempotency/budget tests. No real provider mutations."""
import copy
import io
import json
from pathlib import Path
import subprocess
import urllib.error

import pytest

from gearshift.coding_confirmation_lease import atomic_json,sha256_file,verify_lease
from scripts import coding_confirmation_regional_dispatch as r


class Provider:
    def __init__(self):
        self.calls=[];self.cli=[];self.rows=[];self.post_error=None;self.volume_error=None;self.pod_change={};self.volume_change={};self.volumes={}
    def api(self,path,method='GET',body=None):
        self.calls.append((path,method,copy.deepcopy(body)))
        if method=='GET':
            if path=='network-volumes':
                return {'networkVolumes':[{**{k:v for k,v in x.items() if k!='dataCenterId'},'dataCenter':x['dataCenterId'],'type':'STANDARD'} for x in self.volumes.values()]}
            assert path.startswith('pods?')
            return {'pods':copy.deepcopy(self.rows),'pagination':{'hasNextPage':False,'nextCursor':None}}
        assert method=='POST' and path=='pods'
        if self.post_error:raise self.post_error
        return {**body,'id':'pod_'+body['name'].removeprefix('gearshift-confirmation-'),'gpu':{'id':'NVIDIA H200','count':body['gpu']['count']},
            'dataCenterId':body['dataCenterIds'][0],'cost':4.59*body['gpu']['count'],'status':'PROVISIONING',**self.pod_change}
    def command(self,args,**kwargs):
        self.cli.append(args)
        if self.volume_error:raise self.volume_error
        if args[2]=='create':
            name=args[args.index('--name')+1];region=args[args.index('--data-center-id')+1]
            v={'id':'volume_'+str(len(self.volumes)),'name':name,'size':200,'dataCenterId':region,**self.volume_change}
            self.volumes[v['id']]=v
        else:
            assert args[2]=='get';v={**self.volumes[args[3]],**self.volume_change}
        return json.dumps(v).encode()


def setup(tmp_path,monkeypatch):
    evidence=tmp_path/'resources';evidence.mkdir()
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r.base,'EVIDENCE',evidence)
    monkeypatch.setattr(r.time,'time',lambda:100000.)
    api=Provider();monkeypatch.setattr(r.base,'api',api.api);monkeypatch.setattr(r.subprocess,'check_output',api.command)
    return evidence,api


def capacity_error():return urllib.error.HTTPError('x',400,'Bad Request',{},io.BytesIO(json.dumps({'detail':r.CAPACITY_DETAIL,'secret':'do not record'}).encode()))


def past_lease(pod='past',count=2,extra=0):
    return {'experiment_id':r.base.EXPERIMENT,'pod_id':pod,'allocation_epoch':1000.,'deadline_epoch':44200.,
        'upper_hourly_usd':5.55*count if count else 1.,'gpu_count':count,
        'other_reserved_usd':r.base.ENVELOPE-12*(5.55*count if count else 1.)+extra,
        'baseline_usd':r.base.BASELINE,'baseline_gpu_hours':91.,'total_cap_usd':2500,'total_cap_gpu_hours':None,
        'cleanup_reserve_usd':40,'allowed_result_root':r.REMOTE+'/'+r.base.RESULT,'control_relative':'allocations/'+pod}


def test_new_region_conserves_existing_fleet_budget_and_records_public_provenance(tmp_path,monkeypatch,capsys):
    e,api=setup(tmp_path,monkeypatch);r.allocate('primary_usco01','US-CO-1',4)
    receipt=json.loads(capsys.readouterr().out);lease=r.read(e/'primary_usco01/lease.json')
    assert receipt['reservation']['reserved_total_usd']==pytest.approx(r.base.BASELINE+r.base.ENVELOPE+40)
    assert lease['gpu_count']==4 and lease['upper_hourly_usd']==22.2 and lease['deadline_epoch']==143200.
    assert lease['network_volume_id']=='volume_0' and lease['allowed_result_root'].startswith('/workspace/GearshiftConfirmationPrimary/')
    assert r.read(e/'primary_usco01/volume_request.json')['action']=='create'
    assert len([x for x in api.calls if x[1]=='POST'])==1
    assert r.read(e/'primary_usco01/create_request.json')['gpu']['id']=='NVIDIA H200'


def test_explicit_capacity_rejection_allows_only_reconciled_reuse_not_new_volume(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch);api.post_error=capacity_error()
    with pytest.raises(RuntimeError):r.allocate('primary_usco01','US-CO-1',4)
    failure=r.read(e/'primary_usco01/creation_failure.json');assert failure['explicit_capacity_rejection']
    assert 'secret' not in json.dumps(failure)
    old=(e/'primary_usco01/volume.json').read_bytes();api.post_error=None
    with pytest.raises(ValueError,match='Reuse the existing'):r.allocate('primary_usco02','US-CO-1',2)
    r.allocate('primary_usco02','US-CO-1',2,volume_id='volume_0')
    assert [x[2] for x in api.cli]==['create','get']
    assert (e/'primary_usco01/volume.json').read_bytes()==old
    assert r.read(e/'primary_usco02/volume_request.json')['origin']['sha256']==sha256_file(e/'primary_usco01/volume.json')


@pytest.mark.parametrize('error',[TimeoutError('secret transport text'),urllib.error.HTTPError('x',503,'unavailable',{},io.BytesIO(b'{}'))])
def test_ambiguous_post_blocks_any_next_name_and_retains_volume(tmp_path,monkeypatch,error):
    e,api=setup(tmp_path,monkeypatch);api.post_error=error
    with pytest.raises(RuntimeError):r.allocate('first','US-CO-1',2)
    assert r.read(e/'first/volume.json')['id']=='volume_0'
    with pytest.raises(ValueError,match='Unresolved prior pod'):r.allocate('second','US-GA-2',2)
    assert len(api.cli)==1 and len([c for c in api.calls if c[1]=='POST'])==1
    assert 'secret' not in (e/'first/creation_failure.json').read_text()


def test_capacity_rejection_does_not_override_a_live_same_named_pod(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch);api.post_error=capacity_error()
    with pytest.raises(RuntimeError):r.allocate('first','US-CO-1',2)
    api.rows=[{'id':'unexpected','name':'gearshift-confirmation-first','status':'RUNNING'}]
    with pytest.raises(ValueError,match='now exists'):r.allocate('second','US-GA-2',2)


def test_volume_create_failure_is_durable_and_blocks_duplicate_storage(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch);api.volume_error=subprocess.CalledProcessError(1,['provider'],stderr=b'secret')
    with pytest.raises(RuntimeError):r.allocate('first','US-CO-1',2)
    assert r.read(e/'first/volume_creation_failure.json')['reconciliation_required']
    assert not any(c[1]=='POST' for c in api.calls)
    with pytest.raises(ValueError,match='volume creation'):r.allocate('second','US-GA-2',2)
    assert len(api.cli)==1 and 'secret' not in (e/'first/volume_creation_failure.json').read_text()


@pytest.mark.parametrize('change',[{'size':300},{'dataCenterId':'EU-FR-1'},{'name':'unrelated'},{'id':'lgk3howszi'}])
def test_successful_but_unexpected_volume_is_saved_before_rejection(tmp_path,monkeypatch,change):
    e,api=setup(tmp_path,monkeypatch);api.volume_change=change
    with pytest.raises(ValueError,match='provenance'):r.allocate('first','US-CO-1',2)
    assert (e/'first/volume.json').is_file() and r.read(e/'first/volume_scope_failure.json')['volume_preserved']
    assert not any(c[1]=='POST' for c in api.calls)


@pytest.mark.parametrize('volume',['lgk3howszi','o0ndo35b3f','jj2zyi9yrc','r4q5tbmdb9','unknown_public'])
def test_protected_or_unproven_volume_cannot_be_attached(tmp_path,monkeypatch,volume):
    e,api=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError):r.allocate('first','US-CO-1',2,volume_id=volume)
    assert not api.cli and not any(c[1]=='POST' for c in api.calls)


def test_named_lookalike_volume_receipt_is_not_creation_provenance(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch)
    atomic_json(e/'copied/volume.json',{'id':'volume','name':'gearshift-confirmation-public-other','size':200,'dataCenterId':'US-CO-1'})
    with pytest.raises(ValueError,match='provenance'):r.allocate('new','US-CO-1',2,volume_id='volume')


def test_changed_reused_volume_fails_without_another_volume_creation(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch);api.post_error=capacity_error()
    with pytest.raises(RuntimeError):r.allocate('first','US-CO-1',2)
    api.volume_change={'size':400};api.post_error=None
    with pytest.raises(ValueError,match='provenance'):r.allocate('second','US-CO-1',2,volume_id='volume_0')
    assert [x[2] for x in api.cli]==['create','get'] and len([x for x in api.calls if x[1]=='POST'])==1


def test_twenty_issued_slots_include_historical_released_leases(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch)
    for i,n in enumerate([8,8,4]):atomic_json(e/f'past{i}/lease.json',past_lease('past'+str(i),n))
    with pytest.raises(ValueError,match='envelope exhausted'):r.allocate('new','US-CO-1',1)
    assert not api.cli


def test_later_cpu_replacement_budget_high_water_is_not_lost(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch);atomic_json(e/'cpu/lease.json',past_lease('cpu',0,extra=24))
    r.allocate('new','US-CO-1',2)
    lease=r.read(e/'new/lease.json')
    assert verify_lease(lease)['reserved_total_usd']==pytest.approx(r.base.BASELINE+r.base.ENVELOPE+40+24)


def test_post_create_price_rise_shortens_lease_without_increasing_reserved_dollars(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch);api.pod_change['cost']=20.
    r.allocate('new','US-CO-1',2);lease=r.read(e/'new/lease.json')
    assert lease['upper_hourly_usd']==pytest.approx(24.06)
    assert lease['deadline_epoch']-lease['allocation_epoch']<12*3600
    assert (lease['deadline_epoch']-lease['allocation_epoch'])/3600*lease['upper_hourly_usd']==pytest.approx(12*11.1)
    assert verify_lease(lease)['reserved_total_usd']==pytest.approx(r.base.BASELINE+r.base.ENVELOPE+40)


@pytest.mark.parametrize('change',[{'gpu':{'id':'other','count':2}},{'gpu':{'id':'NVIDIA H200','count':4}},
    {'dataCenterId':'EU-FR-1'},{'cost':None},{'cost':float('nan')},{'mounts':None},{'mounts':{'network':[{'volumeId':'lgk3howszi','path':'/workspace'}]}}])
def test_post_create_scope_failure_retains_pod_lease_and_volume(tmp_path,monkeypatch,change):
    e,api=setup(tmp_path,monkeypatch);api.pod_change=change
    with pytest.raises(ValueError,match='Created pod differs'):r.allocate('new','US-CO-1',2)
    assert (e/'new/pod.json').is_file() and (e/'new/lease.json').is_file() and (e/'new/volume.json').is_file()
    assert r.read(e/'new/pod_scope_failure.json')['do_not_stage_scientific_work']
    if change.get('gpu',{}).get('count')==4:assert r.read(e/'new/lease.json')['gpu_count']==4


def test_missing_pod_identity_is_ambiguous_not_safe_to_retry(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch);api.pod_change={'id':None}
    with pytest.raises(RuntimeError):r.allocate('first','US-CO-1',2)
    with pytest.raises(ValueError,match='Unresolved'):r.allocate('second','US-GA-2',2)
    assert r.read(e/'first/creation_failure.json')['explicit_capacity_rejection'] is False


def test_incomplete_pagination_is_rejected_before_mutation(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(r.base,'api',lambda *a,**kw:{'pods':[],'pagination':{'hasNextPage':True,'nextCursor':None}})
    with pytest.raises(ValueError,match='pagination'):r.allocate('new','US-CO-1',2)
    assert not api.cli


def test_pages_are_all_read_and_duplicate_id_is_rejected(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch);calls=[]
    def pages(path):
        calls.append(path)
        if len(calls)==1:return {'pods':[{'id':'one'}],'pagination':{'hasNextPage':True,'nextCursor':'x/y'}}
        return {'pods':[{'id':'two'}],'pagination':{'hasNextPage':False}}
    monkeypatch.setattr(r.base,'api',pages)
    assert [x['id'] for x in r.inventory()]==['one','two'] and calls[-1].endswith('cursor=x%2Fy')
    monkeypatch.setattr(r.base,'api',lambda *a:{'pods':[{'id':'same'},{'id':'same'}],'pagination':{'hasNextPage':False}})
    with pytest.raises(ValueError,match='duplicate'):r.inventory()


def test_legacy_capacity_reconciliation_still_requires_explicit_400_and_live_absence(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch);f=e/'legacy'
    atomic_json(f/'reservation.json',{'lease':past_lease()})
    atomic_json(f/'create_request.json',{'name':'gearshift-confirmation-legacy'})
    atomic_json(f/'ambiguous_creation.json',{'http_status':400})
    atomic_json(f/'reconciliation.json',{'no_pod_created':True,'matching_pods':[],'observed_at_utc':'2026-09-19T00:00:00Z'})
    r.allocate('new','US-CO-1',2)
    assert (e/'new/lease.json').is_file()


def failed_volume(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch)
    api.volume_error=subprocess.CalledProcessError(1,['provider'],output=b'',stderr=b'')
    with pytest.raises(RuntimeError):r.allocate('failed_usga','US-GA-2',4)
    monkeypatch.setattr(r.time,'time',lambda:100100.)
    monkeypatch.setattr(r.time,'sleep',lambda n:None)
    return e,api


def test_volume_absence_reconciliation_is_readonly_and_preserves_uncertainty(tmp_path,monkeypatch):
    e,api=failed_volume(tmp_path,monkeypatch);p=e/'failed_usga/volume_creation_failure.json';old=p.read_bytes()
    before=len(api.calls);receipt=r.reconcile_volume_create('failed_usga')
    assert p.read_bytes()==old and len(receipt['observations'])==2
    assert api.calls[before:]==[('network-volumes','GET',None),('network-volumes','GET',None)]
    assert receipt['creation_outcome_still_unknown'] and receipt['potential_storage_gb_reserved']==200
    assert receipt['other_region_attempt_permitted'] and not receipt['retry_original_request_permitted']
    assert receipt['inputs']['volume_creation_failure.json']==sha256_file(p)
    api.volume_error=None;r.allocate('new_ca','CA-MTL-3',2)
    assert (e/'new_ca/lease.json').is_file()
    with pytest.raises(ValueError,match='already exists'):r.reconcile_volume_create('failed_usga')


def test_reconciliation_never_retries_failed_creation_in_same_region(tmp_path,monkeypatch):
    e,api=failed_volume(tmp_path,monkeypatch);r.reconcile_volume_create('failed_usga');api.volume_error=None
    with pytest.raises(ValueError,match='unresolved possible volume'):r.allocate('usga_retry','US-GA-2',2)
    assert len(api.cli)==1


def matching_volume():
    return {'id':'latevolume','name':'gearshift-confirmation-public-failed_usga','size':200,'dataCenterId':'US-GA-2'}


def test_matching_volume_blocks_receipt_and_keeps_failure(tmp_path,monkeypatch):
    e,api=failed_volume(tmp_path,monkeypatch);api.volumes['latevolume']=matching_volume()
    with pytest.raises(ValueError,match='Matching volume exists'):r.reconcile_volume_create('failed_usga')
    assert not (e/'failed_usga/volume_creation_reconciliation.json').exists()
    assert (e/'failed_usga/volume_creation_failure.json').exists()


def test_late_appearing_volume_invalidates_saved_absence_before_new_allocation(tmp_path,monkeypatch):
    e,api=failed_volume(tmp_path,monkeypatch);r.reconcile_volume_create('failed_usga')
    api.volumes['latevolume']=matching_volume();api.volume_error=None
    with pytest.raises(ValueError,match='now visible'):r.allocate('new_ca','CA-MTL-3',2)
    assert len(api.cli)==1


@pytest.mark.parametrize('change',['failure','intent','inventory_hash'])
def test_receipt_must_bind_original_failure_intent_and_full_inventory(tmp_path,monkeypatch,change):
    e,api=failed_volume(tmp_path,monkeypatch);r.reconcile_volume_create('failed_usga')
    if change=='failure':p=e/'failed_usga/volume_creation_failure.json';v=r.read(p);v['returncode']=2
    elif change=='intent':p=e/'failed_usga/volume_request.json';v=r.read(p);v['region']='EU-FR-1'
    else:p=e/'failed_usga/volume_creation_reconciliation.json';v=r.read(p);v['observations'][0]['inventory_sha256']='0'*64
    atomic_json(p,v)
    with pytest.raises(ValueError):r.allocate('new_ca','CA-MTL-3',2)
    assert len(api.cli)==1


def test_reconcile_requires_settling_age_and_full_consistent_provider_reads(tmp_path,monkeypatch):
    e,api=failed_volume(tmp_path,monkeypatch);monkeypatch.setattr(r.time,'time',lambda:100010.)
    with pytest.raises(ValueError,match='sixty seconds'):r.reconcile_volume_create('failed_usga')
    monkeypatch.setattr(r.time,'time',lambda:100100.)
    monkeypatch.setattr(r.base,'api',lambda *a,**kw:{'networkVolumes':[],'pagination':{'hasNextPage':True}})
    with pytest.raises(ValueError,match='not complete'):r.reconcile_volume_create('failed_usga')
    assert not (e/'failed_usga/volume_creation_reconciliation.json').exists()


def test_inconsistent_volume_reads_do_not_make_a_receipt(tmp_path,monkeypatch):
    e,api=failed_volume(tmp_path,monkeypatch);calls=[]
    def changing(path):
        calls.append(path)
        return {'networkVolumes':[] if len(calls)==1 else [{'id':'new','name':'unrelated','size':20,'dataCenter':'CA-MTL-3'}]}
    monkeypatch.setattr(r.base,'api',changing)
    with pytest.raises(ValueError,match='inventories changed'):r.reconcile_volume_create('failed_usga')
    assert not (e/'failed_usga/volume_creation_reconciliation.json').exists()


def test_future_cli_failure_keeps_sanitized_structured_error(tmp_path,monkeypatch):
    e,api=setup(tmp_path,monkeypatch);token='rpa_'+'a'*40
    failure={'code':'bad_request','status':400,'error':'Unsupported data center; api_key='+token,'api_key':token}
    api.volume_error=subprocess.CalledProcessError(1,['provider'],output=json.dumps(failure).encode(),stderr=b'not JSON: omitted')
    with pytest.raises(RuntimeError):r.allocate('failed','US-GA-2',4)
    saved=r.read(e/'failed/volume_creation_failure.json')['sanitized_cli_output']
    assert saved['stdout']['code']=='bad_request' and saved['stdout']['status']==400
    assert 'Unsupported data center' in saved['stdout']['error'] and token not in json.dumps(saved)
    assert saved['stderr']['unstructured_text_omitted'] and 'not JSON' not in json.dumps(saved)
