#!/usr/bin/env python3
"""Bounded sparse-diagnostic allocations, reusing existing provider/lease code.

Never creates/deletes a volume, retries a POST, or changes an old experiment.
"""
import argparse
import fcntl
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import coding_confirmation_dispatch as d
from scripts.coding_confirmation_regional_dispatch import inventory, volume_inventory, CAPACITY_DETAIL
from gearshift.coding_confirmation_lease import atomic_json, verify_lease
from gearshift import sparse_repair_storage as storage

EVIDENCE = ROOT / 'results/sparse_repair_01/resources'
REMOTE = '/workspace/GearshiftSparseRepair'
EXPERIMENT = 'sparse_repair_01'
PUBLIC = {'US-CO-1': 'zeicr9elbn', 'EU-FR-1': 'em4bdfudg4', 'CA-MTL-3': 'u4rtme34ja', 'AP-JP-1': 'o0ndo35b3f'}
CPU_PROFILES = {'cpu5c32': {'id': 'cpu5c', 'vcpuCount': 32},
                'cpu5g16': {'id': 'cpu5g', 'vcpuCount': 16},
                'cpu3g16': {'id': 'cpu3g', 'vcpuCount': 16}}


def allocate(name, kind, region, count, hours=None, cpu_profile=None, private_destination=None):
    # Validate before inventory reads or any provider request. This option only
    # defines a NEW immutable lease; it cannot extend an existing allocation.
    if kind not in ('gpu', 'cpu'):
        raise ValueError('Unknown allocation kind')
    if cpu_profile is not None and (kind != 'cpu' or type(cpu_profile) is not str or cpu_profile not in CPU_PROFILES):
        raise ValueError('CPU profile is valid only for a new CPU allocation: cpu5c32, cpu5g16 or cpu3g16')
    selected_cpu_profile = (cpu_profile or 'cpu5c32') if kind == 'cpu' else None
    if private_destination is not None and (kind != 'cpu' or type(private_destination) is not str
            or private_destination not in ('original', 'fallback_euris01')):
        raise ValueError('Private destination is valid only for a new CPU allocation')
    fallback = private_destination == 'fallback_euris01'
    if fallback and selected_cpu_profile != storage.FALLBACK_PROFILE:
        raise ValueError('Fallback private destination requires explicit cpu5g16 profile')
    binding = storage.load_binding(ROOT) if fallback else None
    if type(count) is not int or count not in (1, 2, 4) or (kind == 'cpu' and count != 1):
        raise ValueError('GPU count must be 1, 2 or 4; CPU allocation requires count exactly 1')
    hours = (2 if kind == 'gpu' else 1) if hours is None else hours
    if type(hours) not in (int, float) or not math.isfinite(hours) or not 0 < hours <= 12:
        raise ValueError('Allocation hours must be finite, positive and at most 12')
    if not name.replace('_', '').isalnum():
        raise ValueError('Unsafe name')
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE / 'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        folder = EVIDENCE / name
        if folder.exists():
            raise ValueError('Existing allocation must be reconciled, not repeated')
        rows = inventory()
        volumes = volume_inventory()
        own = [r for r in rows if r.get('name', '').startswith('gearshift-sparse-repair-')]
        # Retain every reservation until an explicit closed receipt, even if absent.
        reserved = 0.
        for p in EVIDENCE.glob('*/lease.json'):
            lease = json.loads(p.read_text())
            closed = p.parent / 'closed.json'
            reserved += (json.loads(closed.read_text())['upper_compute_usd'] if closed.exists()
                         else (lease['deadline_epoch']-lease['allocation_epoch'])/3600*lease['upper_hourly_usd'])
        known = {json.loads(p.read_text())['pod_id'] for p in EVIDENCE.glob('*/lease.json')}
        if any(r['id'] not in known for r in own):
            raise ValueError('Unreconciled diagnostic pod')
        if kind == 'gpu':
            volume = PUBLIC[region]; rate = 5.55*count
        else:
            volume = storage.FALLBACK_VOLUME if fallback else storage.ORIGINAL_VOLUME
            region = storage.FALLBACK_REGION if fallback else 'EU-RO-1'; rate = 1.5
        if reserved + hours*rate > 300:
            raise ValueError('Initial diagnostic reservation envelope exceeded; profile/reconcile first')
        if not any(v['id']==volume and v['dataCenter']==region
                   and (not fallback or v.get('size') == 20) for v in volumes):
            raise ValueError('Protected volume identity missing')
        now=time.time()
        # Historical conservative1342.894 + up to17.106 storage/rounding reserve.
        # Additional storage reserve is deliberately not credited back here.
        lease={'experiment_id':EXPERIMENT,'pod_id':'reservation_pending',
            'allocation_epoch':now,'deadline_epoch':now+hours*3600,
            'upper_hourly_usd':rate,'gpu_count':count if kind=='gpu' else 0,
            'baseline_usd':1360.,'other_reserved_usd':reserved+20.,
            'total_cap_usd':2500,'total_cap_gpu_hours':None,'cleanup_reserve_usd':40,
            'allowed_result_root':REMOTE+'/results/sparse_repair_01','network_volume_id':volume}
        if fallback:
            lease.update(private_storage_binding=storage.binding_reference(),
                         storage_region=region,cpu_profile=selected_cpu_profile)
            storage.validate_cpu_lease(ROOT,lease)
        request={'name':'gearshift-sparse-repair-'+name,'image':d.IMAGE,'disk':40,
            'cloud':'SECURE','dataCenterIds':[region],'ports':['22/tcp'],'startSsh':True,
            'mounts':{'network':[{'volumeId':volume,'path':'/workspace'}]}}
        if kind=='gpu': request['gpu']={'id':'NVIDIA H200','count':count,'minRamPerGpu':96,'minVcpuCountPerGpu':8}
        else: request['cpu']=dict(CPU_PROFILES[selected_cpu_profile])
        folder.mkdir()
        atomic_json(folder/'pre_create.json',{'epoch':now,'pods':[d.safe(x) for x in rows],'volumes':volumes,
            'historical_conservative_cumulative_usd':1342.894120825932,
            'baseline_with_storage_reserve_usd':1360,'additional_storage_reserve_usd':20,
            'diagnostic_reserved_before_usd':reserved,'requested_allocation_hours':hours,
            'requested_cpu_profile':selected_cpu_profile,
            'requested_private_destination':private_destination or ('original' if kind=='cpu' else None),
            'private_storage_binding':storage.binding_reference() if binding else None,
            'no_protected_volume_mutation':True})
        atomic_json(folder/'reservation.json',{'lease':lease,'reservation':verify_lease(lease)})
        atomic_json(folder/'create_request.json',request)
        try: pod=d.api('pods','POST',request)
        except Exception as exc:
            detail=None
            if hasattr(exc,'read'):
                try: detail=json.loads(exc.read(4096)).get('detail')
                except Exception: pass
            atomic_json(folder/'failure.json',{'http_status':getattr(exc,'code',None),
                'explicit_capacity_rejection':detail==CAPACITY_DETAIL,'error_type':type(exc).__name__,
                'reconcile_before_new_request':True,'epoch':time.time()})
            raise RuntimeError('Allocation failed; receipt saved; reconcile instead of retrying') from None
        atomic_json(folder/'pod.json',d.safe(pod))
        actual=pod.get('cost',0)
        if not isinstance(actual,(int,float)) or actual<=0: raise ValueError('Missing actual quote; reconcile allocated pod')
        if actual>rate:
            lease['upper_hourly_usd']=actual*1.1
            lease['deadline_epoch']=now+hours*rate/lease['upper_hourly_usd']*3600
        lease.update(pod_id=pod['id'],control_relative='allocations/'+pod['id'])
        verify_lease(lease);atomic_json(folder/'lease.json',lease)
        print(json.dumps({'name':name,'pod':d.safe(pod),'guard_required':True}))


def arm(name):
    d.EVIDENCE=EVIDENCE;d.REMOTE=REMOTE;d.EXPERIMENT=EXPERIMENT
    d.arm(name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['allocate','arm'])
    p.add_argument('--name',required=True);p.add_argument('--kind',choices=['gpu','cpu'],default='gpu')
    p.add_argument('--region',choices=list(PUBLIC),default='US-CO-1');p.add_argument('--count',type=int,choices=[1,2,4],default=1)
    p.add_argument('--hours',type=float,default=None,help='New immutable allocation duration: >0 to 12 hours; default GPU 2, CPU 1')
    p.add_argument('--cpu-profile',choices=list(CPU_PROFILES),default=None,
                   help='Explicit new CPU allocation profile; default cpu5c32. No automatic profile fallback.')
    p.add_argument('--private-destination',choices=['original','fallback_euris01'],default=None,
                   help='Explicit CPU-only destination; fallback is bound to one frozen private volume receipt.')
    a=p.parse_args()
    if a.action=='arm' and a.hours is not None:
        p.error('--hours applies only to a new allocation; existing leases cannot be extended')
    if a.action=='arm' and a.cpu_profile is not None:
        p.error('--cpu-profile applies only to a new CPU allocation; existing leases cannot be changed')
    if a.action=='arm' and a.private_destination is not None:
        p.error('--private-destination applies only to a new CPU allocation; existing leases cannot be changed')
    if a.action=='allocate': allocate(a.name,a.kind,a.region,a.count,a.hours,a.cpu_profile,a.private_destination)
    else: arm(a.name)
