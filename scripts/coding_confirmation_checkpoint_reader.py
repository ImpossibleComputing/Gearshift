#!/usr/bin/env python3
"""One-hour public checkpoint reader on the retained training volume."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
import urllib.error

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import coding_confirmation_dispatch as b
from scripts import coding_confirmation_regional_dispatch as r
from scripts import coding_confirmation_checkpoint_transfer as transfer


def allocate(name):
    if not r.re.fullmatch('checkpoint_reader_ap[0-9]{2}',name):raise ValueError('Expected unique checkpoint_reader_apNN name')
    with (b.EVIDENCE/'allocation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        folder=b.EVIDENCE/name
        if folder.exists():raise ValueError('Existing request must be reconciled, never replayed')
        # Public endpoint provenance must already exist; this is file recovery only.
        transfer.expected_weights()
        rows=r.inventory();leases,_=r.history(rows)
        if any(p['id']=='rekbqruqox2bk9' for p in rows):raise ValueError('Original training pod remains available')
        volume=b.api('network-volumes/o0ndo35b3f')
        if any(volume.get(k)!=v for k,v in {'id':'o0ndo35b3f','size':250,'dataCenter':'AP-JP-1','type':'STANDARD'}.items()):
            raise ValueError('Retained training volume identity differs')
        rows=[b.safe(b.api('pods/'+p['id'])) for p in rows];present={p['id'] for p in rows};absent={}
        for old in leases:
            if old['pod_id'] in present:continue
            try:b.api('pods/'+old['pod_id'])
            except urllib.error.HTTPError as exc:
                if exc.code!=404:raise
            else:raise ValueError('Previously absent pod reappeared')
            absent[old['pod_id']]={'pod_id':old['pod_id'],'http_status':404}
        now=time.time()
        for proof in absent.values():proof['observed_epoch']=now
        closed={v['pod_id']:v for v in [r.read(p) for p in b.EVIDENCE.glob('*/final_compute_estimate.json')]}
        budget=r.reconciled_budget(leases,rows,absent,closed,now,5.55,1)
        lease={'experiment_id':b.EXPERIMENT,'pod_id':'reservation_pending','allocation_epoch':now,
            'deadline_epoch':now+3600,'upper_hourly_usd':5.55,'gpu_count':1,
            'baseline_usd':budget['baseline_usd'],'baseline_gpu_hours':91.24181750045884,
            'other_reserved_usd':budget['other_reserved_usd'],'total_cap_usd':2500,'total_cap_gpu_hours':None,
            'cleanup_reserve_usd':40,'allowed_result_root':transfer.SOURCE_ROOT+'/'+b.RESULT,
            'purpose':'checkpoint_transfer_reader'}
        request={'name':'gearshift-confirmation-'+name,'image':b.IMAGE,'disk':40,'cloud':'SECURE',
            'dataCenterIds':['AP-JP-1'],'ports':['22/tcp'],'startSsh':True,
            'mounts':{'network':[{'volumeId':'o0ndo35b3f','path':'/workspace'}]},
            'gpu':{'id':'NVIDIA H200','count':1,'minRamPerGpu':96,'minVcpuCountPerGpu':8}}
        folder.mkdir();b.atomic_json(folder/'reservation.json',{'lease':lease,'reservation':b.verify_lease(lease)})
        b.atomic_json(folder/'budget_reconciliation.json',budget);b.atomic_json(folder/'pre_create_inventory.json',rows)
        b.atomic_json(folder/'protected_volume_read_scope.json',{'volume':volume,'purpose':lease['purpose'],
            'models_or_training_run':False,'writes_limited_to_allocation_guard_records':True})
        b.atomic_json(folder/'create_request.json',request)
        try:pod=b.api('pods','POST',request)
        except Exception as exc:
            status=getattr(exc,'code',None);detail=None
            if hasattr(exc,'read'):
                try:detail=json.loads(exc.read(4096)).get('detail')
                except (ValueError,TypeError):pass
            b.atomic_json(folder/'creation_failure.json',{'error_type':type(exc).__name__,'http_status':status,
                'explicit_capacity_rejection':status==400 and detail==r.CAPACITY_DETAIL,
                'provider_message':r.CAPACITY_DETAIL if detail==r.CAPACITY_DETAIL else 'Response omitted; reconcile inventory.',
                'epoch':time.time(),'reconciliation_required':True})
            raise RuntimeError('Reader creation rejected or uncertain; reconcile before new requests') from None
        b.atomic_json(folder/'allocated_pod_identity.json',{'pod_id':pod['id'],'network_volume_id':'o0ndo35b3f'})
        b.atomic_json(folder/'pod.json',b.safe(pod))
        if r.finite(pod.get('cost')) and pod['cost']>5.55:
            lease['upper_hourly_usd']=pod['cost']*1.2
            lease['deadline_epoch']=now+5.55/lease['upper_hourly_usd']*3600
        lease.update(pod_id=pod['id'],network_volume_id='o0ndo35b3f',control_relative='allocations/'+pod['id'])
        b.verify_lease(lease);b.atomic_json(folder/'lease.json',lease)
        print(json.dumps({'pod':b.safe(pod),'guard_arming_required':True,'lease_hours':(lease['deadline_epoch']-now)/3600}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();allocate(a.name)
