#!/usr/bin/env python3
"""Offline, invoice-nonfinal sparse-repair ledger from local provider receipts.

No provider API, credential, private-test, model or candidate access. By default
print the ledger; --collect atomically writes it. Observation times come from
receipts, never the current wall clock, so identical inputs regenerate exactly.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO/'results/sparse_repair_01'
DEFAULT_HISTORY = REPO/'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/resources/final_cleanup_and_cost.json'
HISTORICAL_BASELINE = 1342.894120825932
RESERVATION_BASELINE = 1360.0
EXTRA_STORAGE_RESERVE = 20.0
CLEANUP_RESERVE = 40.0
SOFT_TARGET = 2000.0
HARD_AUTHORIZATION = 2500.0
DIAGNOSTIC_ENVELOPE = 300.0


def number(value, name, *, positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('Invalid finite nonnegative '+name)
    return float(value)


def snapshot(path, sources):
    path=Path(path)
    if path.is_symlink():raise ValueError('Refuse symlinked ledger input')
    raw=path.read_bytes()
    sources.append({'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)


def indexed(rows,name):
    if not isinstance(rows,list) or any(not isinstance(r,dict) or not isinstance(r.get('id'),str) for r in rows):
        raise ValueError('Malformed '+name+' inventory')
    result={r['id']:r for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate '+name+' identity')
    return result


def compact_pod(pod):
    return {k:pod.get(k) for k in ('id','name','status','cost','dataCenterId','createdAt','startedAt')}


def make_ledger(root=DEFAULT_ROOT,historical=DEFAULT_HISTORY):
    root=Path(root).resolve();resources=root/'resources';sources=[];warnings=[]
    history=snapshot(historical,sources)
    baseline=number(history['conservative_cumulative_estimate_usd'],'historical baseline')
    if not math.isclose(baseline,HISTORICAL_BASELINE,rel_tol=0,abs_tol=1e-9):
        raise ValueError('Historical source baseline differs from this bounded diagnostic')
    if history.get('hard_ceiling_usd')!=HARD_AUTHORIZATION or history.get('soft_target_usd')!=SOFT_TARGET:
        raise ValueError('Historical budget identity differs')
    old_pods=indexed(history['complete_provider_pod_inventory'],'historical pod')
    protected=indexed(history['protected_network_volumes'],'protected volume')
    protected_gb=sum(number(p['size'],'protected volume size') for p in protected.values())
    if len(protected)!=7 or protected_gb!=1370:
        raise ValueError('Historical protected seven-volume/1370 GB inventory changed')
    latest=snapshot(resources/'inventory_latest.json',sources)
    epoch=number(latest['epoch'],'inventory epoch',positive=True)
    pods=indexed(latest['pods'],'current pod');volumes=indexed(latest['volumes'],'current volume')
    rows=[];owned=set();actual_sum=0.;upper_elapsed_sum=0.;reservation_sum=0.;future_sum=0.
    endpoints=[epoch]
    for path in sorted(resources.glob('*/lease.json')):
        lease=snapshot(path,sources);pod=snapshot(path.parent/'pod.json',sources)
        pid=lease['pod_id']
        if lease.get('experiment_id')!='sparse_repair_01' or pid!=pod.get('id') or pid in owned:
            raise ValueError('Allocation/pod identity changed or duplicated')
        owned.add(pid)
        start=number(lease['allocation_epoch'],'allocation epoch',positive=True)
        deadline=number(lease['deadline_epoch'],'lease deadline',positive=True)
        upper_rate=number(lease['upper_hourly_usd'],'reserved hourly quote',positive=True)
        rate=number(pod.get('cost'),'actual allocation hourly quote',positive=True)
        if deadline<=start or lease.get('total_cap_usd')!=HARD_AUTHORIZATION or lease.get('cleanup_reserve_usd')!=CLEANUP_RESERVE:
            raise ValueError('Immutable allocation reservation/authorization differs')
        if start>epoch:
            raise ValueError('Inventory predates an allocation; refresh inventory before collection')
        original_reservation=(deadline-start)/3600*upper_rate
        present=pods.get(pid);closed_path=path.parent/'closed.json';closed=None;row_warnings=[]
        if present and present.get('cost') is not None and present['cost']!=pod['cost']:
            row_warnings.append('Inventory quote changed; elapsed estimate uses original allocation quote, not a reconstructed rate history.')
        if closed_path.exists():
            closed=snapshot(closed_path,sources)
            if closed.get('pod_id')!=pid:raise ValueError('Closed receipt names another allocation')
            end=number(closed['observed_absent_epoch'],'absence epoch',positive=True)
            if end<start:raise ValueError('Absence precedes allocation')
            if present is not None and epoch>=end:
                raise ValueError('Closed allocation still present in later inventory; reconcile contradictory receipts')
            basis='allocation_epoch through first closed-receipt provider absence observation'
            status='ABSENT_CONFIRMED'
        else:
            end=epoch
            if present is None:
                basis='allocation_epoch through latest complete provider inventory showing absence; no closed receipt'
                status='ABSENT_IN_INVENTORY_UNCLOSED_RESERVATION'
                row_warnings.append('No closed receipt: full immutable reservation retained, not automatically credited back.')
            else:
                basis='allocation_epoch through latest complete provider inventory timestamp'
                status=present.get('status','UNKNOWN')
                if status!='RUNNING':
                    row_warnings.append('Owned non-running pod: elapsed quote estimate deliberately includes the full interval until current observation; no exact billing stop is inferred.')
        endpoints.append(end)
        hours=(end-start)/3600
        quote_elapsed=hours*rate;upper_elapsed=hours*upper_rate
        if closed:
            for field,expected in [('hours_upper_to_absence',hours),('quoted_compute_usd_upper',quote_elapsed),('upper_compute_usd',upper_elapsed)]:
                if field in closed and not math.isclose(number(closed[field],field),expected,rel_tol=1e-9,abs_tol=1e-7):
                    raise ValueError('Closed receipt arithmetic differs: '+field)
            counted_reservation=upper_elapsed;remaining=0.
        else:
            # Do not pretend the deadline capped a resource still present later.
            counted_reservation=max(original_reservation,upper_elapsed)
            remaining=max(0.,counted_reservation-upper_elapsed)
            if end>deadline:
                row_warnings.append('Observed interval exceeds immutable deadline; upper scenario expanded to observed elapsed rather than clamped.')
        row={'allocation_name':path.parent.name,'pod_id':pid,'gpu_count':lease.get('gpu_count'),
             'region':pod.get('dataCenterId'),'resource_status':status,'allocation_epoch':start,
             'lease_deadline_epoch':deadline,'duration_endpoint_epoch':end,'duration_basis':basis,
             'duration_hours_conservative_to_observation':hours,'allocation_hourly_quote_usd':rate,
             'lease_upper_hourly_usd':upper_rate,'quote_times_elapsed_compute_estimate_usd':quote_elapsed,
             'upper_rate_times_elapsed_usd':upper_elapsed,'original_upper_compute_reservation_usd':original_reservation,
             'compute_reservation_scenario_usd':counted_reservation,'remaining_upper_compute_reservation_usd':remaining,
             'closed_receipt_present':closed is not None,'invoice_final':False,'warnings':row_warnings}
        rows.append(row);actual_sum+=quote_elapsed;upper_elapsed_sum+=upper_elapsed
        reservation_sum+=counted_reservation;future_sum+=remaining
    unowned=[compact_pod(p) for pid,p in pods.items() if pid not in owned and p.get('status')=='RUNNING']
    old_exited=[compact_pod(p) for pid,p in pods.items() if pid not in owned and p.get('status')=='EXITED' and old_pods.get(pid,{}).get('status')=='EXITED']
    new_exited=[compact_pod(p) for pid,p in pods.items() if pid not in owned and p.get('status')=='EXITED' and old_pods.get(pid,{}).get('status')!='EXITED']
    unowned_other=[compact_pod(p) for pid,p in pods.items() if pid not in owned and p.get('status') not in ('RUNNING','EXITED')]
    if unowned:warnings.append('Unowned RUNNING resources exist: their elapsed charges are not assigned to this diagnostic; review ownership/cumulative scope.')
    if unowned_other:warnings.append('Unowned resources have unknown/nonstandard state; no zero-cost claim is made.')
    if new_exited:warnings.append('Newly observed unowned EXITED resources lack historical exited evidence; their prior compute charges are unknown, not assumed zero.')
    unresolved=[]
    for p in sorted(resources.glob('*/reservation.json')):
        if not (p.parent/'lease.json').exists():
            snap=snapshot(p,sources);failed=p.parent/'failure.json'
            failure=snapshot(failed,sources) if failed.exists() else {}
            rejected=failure.get('explicit_capacity_rejection') is True
            unresolved.append({'allocation_name':p.parent.name,'explicit_capacity_rejection':rejected,
                               'compute_cost_assumed_usd':0.0 if rejected else None,
                               'requires_reconciliation':not rejected})
    if any(r['requires_reconciliation'] for r in unresolved):warnings.append('Unresolved allocation attempt(s) lack immutable pod lease; possible charges are not invented.')
    # The original protected inventory stays fixed. New diagnostic storage is
    # additional liability, never folded into or substituted for those seven.
    liability_path=resources/'private_storage_fallback/storage_liability.json'
    liability=None;temporary_id=None
    if liability_path.exists():
        liability=snapshot(liability_path,sources)
        original_ids=liability.get('protected_original_ids')
        if (not isinstance(original_ids,list) or any(not isinstance(v,str) for v in original_ids) or
            len(original_ids)!=7 or set(original_ids)!=set(protected)):
            raise ValueError('Temporary-storage receipt changes original protected-volume identity')
        temporary_id=liability.get('new_volume_id')
        if not isinstance(temporary_id,str) or not temporary_id or temporary_id in protected:
            raise ValueError('Temporary storage must be separate from original protected volumes')
        number(liability.get('new_volume_size_gb'),'temporary storage receipt size',positive=True)
        liability_epoch=number(liability.get('epoch'),'storage liability epoch',positive=True)
        if liability_epoch>epoch:
            warnings.append('Inventory predates temporary-storage liability receipt; current totals describe only that earlier inventory.')
        if temporary_id not in volumes:
            warnings.append('Recorded temporary volume is absent from current inventory; no storage-charge cessation or zero unposted cost is inferred.')
        elif (volumes[temporary_id].get('size')!=liability['new_volume_size_gb'] or
              volumes[temporary_id].get('dataCenter')!=liability.get('new_volume_region')):
            warnings.append('Current temporary-volume size/region differs from its liability receipt; reconcile storage identity without altering original protected volumes.')
    current_sizes={pid:number(v.get('size'),'current volume size',positive=True) for pid,v in volumes.items()}
    expansions=[]
    for intent_path in sorted(resources.glob('*/private_volume_expansion_intent.json')):
        intent=snapshot(intent_path,sources);pid=intent.get('volume_id')
        if pid not in protected:raise ValueError('Protected-volume expansion names a non-protected identity')
        before=number(intent.get('from_size_gb'),'pre-expansion size',positive=True)
        after=number(intent.get('to_size_gb'),'post-expansion size',positive=True)
        if after<=before or number(intent.get('new_incremental_gb'),'expansion increment')!=after-before:
            raise ValueError('Protected-volume expansion size arithmetic differs')
        result_path=intent_path.with_name('private_volume_expansion_result.json')
        result=snapshot(result_path,sources) if result_path.exists() else None
        success=bool(result is not None and type(result.get('returncode')) is int and result['returncode']==0)
        result_volume=None
        if success:
            try:result_volume=json.loads(result.get('stdout',''))
            except (TypeError,ValueError) as exc:raise ValueError('Expansion success lacks provider volume JSON') from exc
            if (not isinstance(result_volume,dict) or result_volume.get('id')!=pid or result_volume.get('size')!=after or
                result_volume.get('dataCenterId')!=protected[pid].get('dataCenter')):
                raise ValueError('Expansion provider result differs from protected volume/size/region')
            result_epoch=number(result.get('epoch'),'expansion result epoch',positive=True)
            endpoints.append(result_epoch)
            if result_epoch>epoch:
                warnings.append('Inventory predates confirmed protected-volume expansion; current inventory totals are stale, not proof that expansion failed. Refresh before final collection.')
        expansions.append({'volume_id':pid,'intent_path':str(intent_path.resolve()),
            'result_path':str(result_path.resolve()) if result is not None else None,
            'from_size_gb':before,'to_size_gb':after,'incremental_gb':after-before,
            'provider_success_result_reported':success,'intent':intent,'result':result,
            'current_inventory_size_gb':current_sizes.get(pid),
            'current_inventory_matches_successful_expansion':success and current_sizes.get(pid)==after,
            'historical_protected_baseline_unchanged':True,'actual_hourly_rate_usd':None,'unposted_cost_usd':None})
    size_changes=[]
    for pid in sorted(set(protected)&set(volumes)):
        previous=number(protected[pid]['size'],'historical protected size',positive=True)
        current=current_sizes[pid]
        if current!=previous:
            matches=[r['result_path'] for r in expansions if r['volume_id']==pid and r['provider_success_result_reported'] and r['to_size_gb']==current]
            size_changes.append({'id':pid,'historical_size_gb':previous,'current_inventory_size_gb':current,
                'delta_gb':current-previous,'change':'expansion' if current>previous else 'reduction',
                'same_original_protected_identity':True,'matching_successful_expansion_result_paths':matches})
    additional=[]
    for pid in sorted(set(volumes)-set(protected)):
        v=volumes[pid]
        additional.append({'id':pid,'name':v.get('name'),'size_gb':current_sizes[pid],
            'region':v.get('dataCenter'),'type':v.get('type'),'original_protected_volume':False,
            'temporary_diagnostic_volume_identified_by_receipt':pid==temporary_id,
            'actual_hourly_rate_usd':None,'unposted_cost_usd':None,'actual_storage_charge_assumed_zero':False,
            'rate_status':'UNKNOWN; no hourly storage tariff is established by current inventory'})
    if additional:
        warnings.append('Additional unprotected network volumes are currently inventoried; their ongoing/unposted storage cost is unknown, not zero, and is excluded from the compute-only estimate.')
    current_storage={'inventory_observed_epoch':epoch,'current_total_volume_count':len(volumes),
        'current_total_allocated_gb':sum(current_sizes.values()),
        'current_protected_volume_count':len(set(protected)&set(volumes)),
        'current_protected_allocated_gb':sum(current_sizes[pid] for pid in sorted(set(protected)&set(volumes))),
        'additional_unprotected_volume_count':len(additional),
        'additional_unprotected_allocated_gb':sum(v['size_gb'] for v in additional),
        'additional_unprotected_volumes':additional,
        'original_protected_volume_size_changes':size_changes,
        'protected_expansion_gb_in_current_inventory':sum(r['delta_gb'] for r in size_changes if r['delta_gb']>0),
        'protected_volume_expansion_receipts':expansions,
        'temporary_storage_liability_receipt':liability,
        'temporary_storage_receipt_path':str(liability_path.resolve()) if liability is not None else None,
        'temporary_storage_volume_present_in_current_inventory':temporary_id in volumes if temporary_id else None,
        'actual_hourly_rate_usd':None,'unposted_cost_usd':None,'actual_storage_charge_assumed_zero':False,
        'unposted_cost_status':'UNKNOWN; current allocated GB are not a storage invoice or a zero-cost claim',
        'excluded_from_quote_elapsed_compute_estimate':True,
        'storage_reserve_is_separate_from_actual_charges':True,
        'limitations':'Current total is the latest inventory, not historical protected capacity plus assumed extras. A missing recorded volume does not establish when billing stopped. No provider mutation or cleanup is performed.'}
    missing=sorted(set(protected)-set(volumes))
    changed=[pid for pid in protected if pid in volumes and volumes[pid].get('size')!=protected[pid].get('size')]
    if missing:warnings.append('Current protected-volume identity is missing from inventory; do not infer deletion or storage cessation from missing evidence.')
    if changed:warnings.append('Current capacity of existing protected volume identity/identities differs from the historical baseline; explicit size-change details distinguish expansion from missing/deleted identity. Historical seven-volume/1370 GB baseline is not rewritten.')
    storage={'protected_volume_count':len(protected),'protected_allocated_gb':protected_gb,
        'protected_sizes_are_historical_baseline_not_current_capacity':True,
        'current_inventory_protected_volume_count':len(set(protected)&set(volumes)),
        'inventory_missing_protected_ids':missing,'inventory_changed_size_ids':changed,
        'protected_volumes':[{'id':p['id'],'name':p.get('name'),'size_gb':p['size'],
                             'region':p.get('dataCenter'),'type':p.get('type')} for p in protected.values()],
        'ongoing_after_compute_completion':True,'actual_hourly_rate_usd':None,'unposted_cost_usd':None,
        'unposted_cost_status':'UNKNOWN; no current storage rate or final invoice was supplied',
        'excluded_from_quote_elapsed_compute_estimate':True,
        'additional_storage_reserve_usd':EXTRA_STORAGE_RESERVE,'reserve_is_actual_charge':False,
        'limitations':'Retained network volumes continue accruing storage. Exited pod disks or other storage charges are also not estimated from unprovided rates.'}
    active=[compact_pod(p) for pid,p in pods.items() if p.get('status')=='RUNNING' and not any(r['pod_id']==pid and r['closed_receipt_present'] for r in rows)]
    cumulative=baseline+actual_sum
    conservative=RESERVATION_BASELINE+reservation_sum+EXTRA_STORAGE_RESERVE+CLEANUP_RESERVE
    return {'schema':1,'experiment_id':'sparse_repair_01','offline_only':True,'invoice_final':False,
        'inventory_observed_epoch':epoch,'latest_evidence_epoch':max(endpoints),
        'latest_evidence_utc':datetime.fromtimestamp(max(endpoints),timezone.utc).isoformat(),
        'historical_source':sources[0],'historical_baseline_estimate_usd':baseline,
        'historical_baseline_is_final_invoice':False,
        'reservation_baseline_usd':RESERVATION_BASELINE,
        'inherited_baseline_reserve_allowance_usd':RESERVATION_BASELINE-baseline,
        'inherited_allowance_is_actual_charge':False,
        'incremental_quote_elapsed_compute_estimate_usd':actual_sum,
        'incremental_upper_rate_elapsed_usd':upper_elapsed_sum,
        'incremental_compute_reservation_scenario_usd':reservation_sum,
        'remaining_upper_compute_reservation_usd':future_sum,
        'cumulative_estimate_excluding_unposted_storage_usd':cumulative,
        'conservative_cumulative_reservation_scenario_usd':conservative,
        'scenario_formula':'1360 conservative baseline + per-pod closed upper elapsed/open max(full lease reservation,upper observed elapsed) + 20 storage reserve + 40 cleanup reserve; do not add quote-elapsed estimates again.',
        'cleanup_reserve_usd':CLEANUP_RESERVE,'reserves_are_not_actual_charges':True,
        'soft_target_usd':SOFT_TARGET,'hard_authorization_usd':HARD_AUTHORIZATION,
        'diagnostic_compute_envelope_usd':DIAGNOSTIC_ENVELOPE,
        'scenario_exceeds_diagnostic_compute_envelope':reservation_sum>DIAGNOSTIC_ENVELOPE,
        'scenario_exceeds_soft_target':conservative>SOFT_TARGET,
        'scenario_reaches_hard_authorization':conservative>=HARD_AUTHORIZATION,
        'per_allocations':rows,'active_resources':active,'active_owned_pod_ids':[p['id'] for p in active if p['id'] in owned],
        'unowned_running_resources':unowned,'unowned_other_state_resources':unowned_other,
        'newly_observed_unowned_exited_resources':new_exited,
        'historically_exited_unrelated_pods':old_exited,'new_compute_assumed_for_historically_exited_unrelated_usd':0.0,
        'unresolved_allocation_attempts':unresolved,'protected_storage':storage,'current_storage_inventory':current_storage,
        'compute_estimate_scope_complete':not unowned and not unowned_other and not new_exited and not any(r['requires_reconciliation'] for r in unresolved),
        'warnings':warnings,'sources':sources,
        'limitations':['Quote × elapsed is a conservative compute estimate, not measured invoice spend; time starts before provider allocation completes.',
            'No duration is extrapolated beyond local receipt timestamps. Refresh inventory and close resources before final delivery.',
            'No unprovided storage rate is invented; ongoing/unposted storage is excluded from cumulative estimate.',
            'Additional temporary or otherwise unprotected volume capacity is reported separately; it neither changes the protected seven-volume baseline nor establishes its unknown storage charges.',
            'Unowned running resource charges and unresolved provider requests require separate reconciliation.',
            'The protected volumes are neither deleted nor modified by this offline collector.']}


def write_atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--historical',type=Path,default=DEFAULT_HISTORY)
    p.add_argument('--collect',action='store_true',help='Write ledger atomically; otherwise print only')
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    if args.output is not None and not args.collect:p.error('--output requires --collect')
    value=make_ledger(args.root,args.historical)
    if args.collect:
        output=args.output or args.root/'cost_ledger.json';write_atomic(output,value)
        print(json.dumps({'output':str(output.resolve()),'invoice_final':False,
            'cumulative_estimate_excluding_unposted_storage_usd':value['cumulative_estimate_excluding_unposted_storage_usd'],
            'conservative_cumulative_reservation_scenario_usd':value['conservative_cumulative_reservation_scenario_usd']}))
    else:print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
