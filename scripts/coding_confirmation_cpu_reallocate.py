#!/usr/bin/env python3
"""Reconcile and replace the released dedicated CPU, reusing its private volume.

`prepare --name scoring_cpu_primary01` makes only live read-only provider calls.
`allocate` additionally requires a caller-pinned, fully closed PUBLIC generation
plan. It creates one CPU pod, never a volume, and never retries a POST. Existing
leases are immutable. This helper does not read/stage private tests or candidates.
After creation, arm the independent guard using the existing dispatch arm action
before staging/running the dedicated scorer. Any failed/ambiguous creation needs
explicit reconciliation; changing the name does not bypass that check.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import coding_confirmation_dispatch as dispatch
from gearshift.coding_confirmation_lease import atomic_json,sha256_file,verify_lease

PRIVATE_VOLUME='lgk3howszi'
PRIVATE_REGION='EU-RO-1'
PRIVATE_VOLUME_NAME='gearshift-confirmation-private-scoring'
ORIGINAL_CPU='m6jcx23rppe7j6'
PRIVATE_REMOTE='/workspace/GearshiftConfirmation'
CPU_ID='cpu5g'
VCPUS=16
MAX_HOURS=24
UPPER_HOURLY_USD=1.0
SAFE_ID=re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')


def read(path):return json.loads(Path(path).read_text())
def number(value):return type(value) in (int,float) and math.isfinite(value) and value>=0

def allocation_name(name):
    if not re.fullmatch(r'scoring_cpu_[a-z0-9][a-z0-9_-]{0,63}',name):
        raise ValueError('Use a new dedicated scoring_cpu_* allocation name')
    return name


def get(api,path):
    # All reconciliation traffic is an explicit GET. Only allocate() can POST.
    return api(path,method='GET')


def list_pods(api):
    result=[];seen=set();cursors=set();cursor=None
    for _ in range(100):
        path='pods?includeClusterPods=true&limit=100'
        if cursor:path+='&cursor='+urllib.parse.quote(cursor,safe='')
        value=get(api,path)
        if not isinstance(value,dict) or not isinstance(value.get('pods'),list) or not isinstance(value.get('pagination'),dict):
            raise ValueError('Ambiguous provider pod inventory shape')
        for row in value['pods']:
            if not isinstance(row,dict) or not SAFE_ID.fullmatch(str(row.get('id',''))) or row['id'] in seen:
                raise ValueError('Invalid or duplicate provider pod identity')
            seen.add(row['id']);result.append(row)
        pagination=value['pagination']
        if pagination.get('hasNextPage') is False:return result
        cursor=pagination.get('nextCursor')
        if pagination.get('hasNextPage') is not True or not isinstance(cursor,str) or not cursor or cursor in cursors:
            raise ValueError('Incomplete provider pagination')
        cursors.add(cursor)
    raise ValueError('Provider pagination exceeded bounded inventory limit')


def mounts(pod):
    value=pod.get('mounts')
    if not isinstance(value,dict):raise ValueError('Ambiguous provider mount inventory')
    networks=value.get('network',[])
    if not isinstance(networks,list) or any(not isinstance(x,dict) or not isinstance(x.get('volumeId'),str) for x in networks):
        raise ValueError('Ambiguous provider network mounts')
    return networks


def historical_leases(evidence):
    leases=[]
    for path in sorted(Path(evidence).glob('*/lease.json')):
        if path.is_symlink() or path.parent.is_symlink():raise ValueError('Linked allocation evidence')
        value=read(path);reservation=verify_lease(value)
        if value['experiment_id']!=dispatch.EXPERIMENT:raise ValueError('Allocation evidence belongs to another experiment')
        leases.append((path,value,reservation))
    original=[x for x in leases if x[1]['pod_id']==ORIGINAL_CPU]
    if len(original)!=1 or original[0][1]['gpu_count']!=0:raise ValueError('Original dedicated CPU lease is missing or duplicated')
    cpu_ids=[v['pod_id'] for _,v,_ in leases if v['gpu_count']==0]
    if len(cpu_ids)!=len(set(cpu_ids)):raise ValueError('Duplicate historical CPU lease')
    expected_root=PRIVATE_REMOTE+'/'+dispatch.RESULT
    for _,value,_ in leases:
        if value['gpu_count']==0 and (value.get('network_volume_id')!=PRIVATE_VOLUME or value['allowed_result_root']!=expected_root):
            raise ValueError('Historical dedicated CPU volume/scope differs')
    # A lost create response may leave no lease at all. Do not silently retry it.
    for folder in sorted(Path(evidence).iterdir()):
        if not folder.is_dir() or (folder/'lease.json').exists():continue
        request=read(folder/'create_request.json') if (folder/'create_request.json').is_file() else {}
        reserve=read(folder/'reservation.json') if (folder/'reservation.json').is_file() else {}
        if request.get('cpu') is not None or reserve.get('lease',{}).get('gpu_count')==0:
            raise ValueError('Unresolved previous CPU creation/reservation: '+folder.name)
    return leases


def reconcile(name,*,evidence=None,api=None,now=None):
    name=allocation_name(name);evidence=Path(evidence or dispatch.EVIDENCE);api=api or dispatch.api
    epoch=time.time() if now is None else now
    if not number(epoch):raise ValueError('Invalid reconciliation time')
    if (evidence/name).exists():raise ValueError('Allocation folder already exists; reconcile it without a new POST')
    leases=historical_leases(evidence)
    original=next(path for path,v,_ in leases if v['pod_id']==ORIGINAL_CPU)
    saved_volume=read(original.parent/'volume.json')
    if {k:saved_volume.get(k) for k in ('id','name','size','dataCenterId')}!={
        'id':PRIVATE_VOLUME,'name':PRIVATE_VOLUME_NAME,'size':20,'dataCenterId':PRIVATE_REGION}:
        raise ValueError('Original retained-volume identity differs')
    volume=get(api,'network-volumes/'+PRIVATE_VOLUME)
    if not isinstance(volume,dict) or {k:volume.get(k) for k in ('id','name','size','dataCenter','type')}!={
        'id':PRIVATE_VOLUME,'name':PRIVATE_VOLUME_NAME,'size':20,'dataCenter':PRIVATE_REGION,'type':'STANDARD'}:
        raise ValueError('Live retained private volume identity differs')
    inventory=list_pods(api);present={p['id'] for p in inventory};absence=[]
    for path,value,_ in leases:
        if value['gpu_count']!=0:continue
        pod_id=value['pod_id']
        if pod_id in present:raise ValueError('Previous dedicated CPU still exists, including stopped/exited pods')
        try:get(api,'pods/'+pod_id)
        except urllib.error.HTTPError as exc:
            if exc.code!=404:raise RuntimeError('Previous CPU absence is ambiguous: HTTP '+str(exc.code)) from None
        except Exception as exc:raise RuntimeError('Previous CPU absence is ambiguous: '+type(exc).__name__) from None
        else:raise ValueError('Previous dedicated CPU still exists')
        absence.append({'pod_id':pod_id,'lease_path':str(path.relative_to(evidence)),
            'lease_sha256':sha256_file(path),'provider_get_http_status':404,'absent_from_full_inventory':True})
    for row in inventory:
        if PRIVATE_VOLUME in {x['volumeId'] for x in mounts(row)}:
            raise ValueError('Protected private volume is attached to another retained/live pod')
        if row.get('name')=='gearshift-confirmation-'+name:raise ValueError('Requested allocation name already exists at provider')
        if str(row.get('name','')).startswith('gearshift-confirmation-'):
            gpu=row.get('gpu')
            if row.get('cpu') is not None or not isinstance(gpu,dict) or type(gpu.get('count')) is not int or gpu['count']<1:
                raise ValueError('Unreconciled CPU or unknown compute allocation in experiment namespace')
    quote=get(api,'catalog/cpus/'+CPU_ID)
    if not isinstance(quote,dict) or quote.get('id')!=CPU_ID:raise ValueError('CPU quote identity differs')
    price=quote.get('price',{}).get('securePerVcpu');ram=quote.get('ramGbPerVcpu');vcpu=quote.get('vcpu',{})
    if (not number(price) or price<=0 or not number(ram) or ram<4 or
        not number(vcpu.get('min')) or not number(vcpu.get('max')) or not vcpu['min']<=VCPUS<=vcpu['max']):
        raise ValueError('CPU price or memory/vCPU contract is unverified')
    # Twenty percent price headroom plus container/private-volume storage margin.
    if VCPUS*price*1.2+.05>UPPER_HOURLY_USD:raise ValueError('Live CPU quote exceeds the conservative hourly reservation')
    old_high_water=max([dispatch.BASELINE+dispatch.ENVELOPE+40,*[r['reserved_total_usd'] for _,_,r in leases]])
    baseline=max([dispatch.BASELINE,*[v['baseline_usd'] for _,v,_ in leases]])
    other=old_high_water-baseline-40
    if other<0:raise ValueError('Inconsistent historical baseline/reservation envelope')
    lease={'experiment_id':dispatch.EXPERIMENT,'pod_id':'reservation_pending','allocation_epoch':epoch,
        'deadline_epoch':epoch+MAX_HOURS*3600,'upper_hourly_usd':UPPER_HOURLY_USD,'gpu_count':0,
        'other_reserved_usd':other,'baseline_usd':baseline,
        'baseline_gpu_hours':max([0,*[v.get('baseline_gpu_hours',0) for _,v,_ in leases]]),
        'total_cap_usd':2500,'total_cap_gpu_hours':None,'cleanup_reserve_usd':40,
        'allowed_result_root':PRIVATE_REMOTE+'/'+dispatch.RESULT,'network_volume_id':PRIVATE_VOLUME}
    reservation=verify_lease(lease)
    request={'name':'gearshift-confirmation-'+name,'image':dispatch.IMAGE,'disk':40,'cloud':'SECURE',
        'dataCenterIds':[PRIVATE_REGION],'ports':['22/tcp'],'startSsh':True,
        'mounts':{'network':[{'volumeId':PRIVATE_VOLUME,'path':'/workspace'}]},
        'cpu':{'id':CPU_ID,'vcpuCount':VCPUS}}
    return {'schema':1,'experiment_id':dispatch.EXPERIMENT,'name':name,'observed_epoch':epoch,
        'provider_reads_only':True,'private_bytes_read':False,'volume_created_or_deleted':False,
        'old_cpu_absence':absence,'retained_volume':{k:volume[k] for k in ('id','name','size','dataCenter','type')},
        'volume_receipt_sha256':sha256_file(original.parent/'volume.json'),
        'inventory':[dispatch.safe(x) for x in inventory if str(x.get('name','')).startswith('gearshift-confirmation-')],
        'cpu_quote':quote,'lease':lease,'reservation':reservation,'create_request':request,
        'budget_policy':'Keep prior cumulative reservation high-water mark, including all historical CPU reservations; add the entire new CPU lease. Never reclaim historical reservation credits.',
        'prior_reserved_total_usd':old_high_water,
        'generation_closure_revalidated':False,'allocation_authorized_by_prepare_receipt':False}


def public_generation_ready(repo,plan_path,plan_sha256):
    # Lazy import, only for explicit allocation; no GPU loading or private reads.
    from scripts.coding_confirmation_score import public_context
    c=public_context(repo,plan_path,plan_sha256)
    if c['declaration']['experiment_id']!=dispatch.EXPERIMENT:raise ValueError('Closed generation belongs to another experiment')
    return {'generation_plan_path':plan_path,'generation_plan_sha256':plan_sha256,
        'declaration_sha256':c['declaration_sha256'],'closure_file_sha256':c['closure_file_sha256'],
        'closure_identity_sha256':c['closure']['closure_sha256'],'answer_count':c['closure']['answer_count'],
        'task_count':c['declaration']['task_count'],'whole_public_generation_recomputed':True,'private_bytes_read':False}


def allocate(name,generation_plan,generation_plan_sha256,*,repo=ROOT,evidence=None,api=None,now=None):
    """Explicit future action: one CPU POST only after full generation closure."""
    evidence=Path(evidence or dispatch.EVIDENCE);api=api or dispatch.api
    generation=public_generation_ready(repo,generation_plan,generation_plan_sha256)
    evidence.mkdir(parents=True,exist_ok=True)
    with (evidence/'allocation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preparation=reconcile(name,evidence=evidence,api=api,now=now)
        folder=evidence/name;folder.mkdir()
        atomic_json(folder/'reconciliation.json',preparation)
        atomic_json(folder/'generation_ready.json',generation)
        atomic_json(folder/'reservation.json',{'lease':preparation['lease'],'reservation':preparation['reservation'],
            'retained_historical_reservation_usd':preparation['prior_reserved_total_usd']})
        atomic_json(folder/'create_request.json',preparation['create_request'])
        atomic_json(folder/'volume_reuse.json',preparation['retained_volume'])
        try:
            pod=api('pods',method='POST',body=preparation['create_request'])
            if not isinstance(pod,dict) or not SAFE_ID.fullmatch(str(pod.get('id',''))):raise ValueError('Create response has no safe pod identity')
        except Exception as exc:
            atomic_json(folder/'ambiguous_creation.json',{'error_type':type(exc).__name__,
                'http_status':getattr(exc,'code',None),'reconciliation_required_before_retry':True,'epoch':time.time()})
            raise
        # Preserve the successful allocation identity before any post-create checks.
        atomic_json(folder/'pod.json',dispatch.safe(pod))
        lease=dict(preparation['lease']);rate=pod.get('cost')
        if number(rate) and rate>0 and rate*1.2+.05>lease['upper_hourly_usd']:
            # A price race may shorten this immutable lease, never enlarge its budget.
            lease['upper_hourly_usd']=rate*1.2+.05
            lease['deadline_epoch']=lease['allocation_epoch']+MAX_HOURS*UPPER_HOURLY_USD/lease['upper_hourly_usd']*3600
        lease.update(pod_id=pod['id'],control_relative='allocations/'+pod['id'])
        verify_lease(lease);atomic_json(folder/'lease.json',lease)
        cpu=pod.get('cpu',{});valid=(pod.get('gpu') is None and isinstance(cpu,dict) and cpu.get('id')==CPU_ID and
            cpu.get('vcpuCount')==VCPUS and pod.get('dataCenterId')==PRIVATE_REGION and
            mounts(pod)==[{'path':'/workspace','volumeId':PRIVATE_VOLUME}] and number(rate) and rate>0)
        if not valid:
            atomic_json(folder/'scope_failure.json',{'pod_id':pod['id'],'reconciliation_required':True,
                'do_not_stage_private_inputs_or_candidates':True,'arm_guard_before_remediation':True})
            raise ValueError('Created pod scope/price differs; recorded own pod needs immediate guarded reconciliation')
        return {'name':name,'pod':dispatch.safe(pod),'reservation':verify_lease(lease),
            'lease_path':str(folder/'lease.json'),'lease_sha256':sha256_file(folder/'lease.json'),
            'guard_armed':False,'next_action':'Immediately run coding_confirmation_dispatch.py arm --name '+name+
            ' with REMOTE=/workspace/GearshiftConfirmation, then dedicated scorer preflight before any private inputs.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','allocate'])
    p.add_argument('--name',required=True);p.add_argument('--generation-plan');p.add_argument('--generation-plan-sha256')
    a=p.parse_args()
    if a.action=='allocate':
        if not a.generation_plan or not a.generation_plan_sha256:p.error('allocate requires a caller-pinned generation plan')
        value=allocate(a.name,a.generation_plan,a.generation_plan_sha256)
    else:value=reconcile(a.name)
    print(json.dumps(value,indent=2,allow_nan=False))
