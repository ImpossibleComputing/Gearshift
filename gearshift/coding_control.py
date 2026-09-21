"""Pilot-only immutable identities, atomic receipts and conservative budget decisions."""
import hashlib
import json
import math
import os
import re
from pathlib import Path
import time

NAMESPACE = 'coding_pilot_v1'
PREFIX = 'gearshift-coding-v1-20260915-'
CAP = 300.0
HOURS = 48.0
STAGE_HOURS = {'preflight': 4, 'preparation': 8, 'training': 12, 'confirmation': 20, 'cleanup': 4}
PARALLEL_PARENT_SHA256 = 'b20f5a9efa9ac1d3c3efd305652ee2d61a1f52263f8bfeec189ca94938e6738a'

def valid_storage_inspection(intent, parallel):
    """One short, read-only CPU mount; it cannot authorize model inference."""
    if not isinstance(intent,dict) or not isinstance(parallel,dict):return False
    numeric=lambda x:type(x) in (int,float) and math.isfinite(x)
    safe=lambda x:isinstance(x,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',x) is not None
    return (intent.get('purpose')=='storage_inspection'
        and bool(intent.get('receipt_sha256')) and intent.get('approval_sha256')==parallel.get('receipt_sha256')
        and bool(parallel.get('receipt_sha256')) and intent.get('read_only') is True
        and intent.get('model_inference_allowed') is False and intent.get('maximum_cpu_pods')==1
        and type(intent.get('gpu_count')) is int and intent['gpu_count']==0
        and intent.get('maximum_hourly_usd')==.5 and safe(intent.get('controller_id')) and safe(intent.get('volume_id'))
        and isinstance(intent.get('resource_name'),str) and intent['resource_name'].startswith(PREFIX)
        and numeric(intent.get('started_epoch')) and numeric(intent.get('deadline_epoch'))
        and 0<intent['deadline_epoch']-intent['started_epoch']<=3600)

def storage_inspection_resource(resource,intent):
    return (isinstance(intent,dict) and resource.get('kind')=='pod'
        and resource.get('storage_inspection_intent_sha256')==intent.get('receipt_sha256')
        and bool(intent.get('receipt_sha256')) and resource.get('controller_id')==intent.get('controller_id')
        and resource.get('name')==intent.get('resource_name') and resource.get('volume_id')==intent.get('volume_id'))

def valid_parallel_approval(approval, previous):
    """A separate, bound receipt widens execution only; old identities stay intact."""
    return (isinstance(approval, dict) and isinstance(previous, dict)
        and approval.get('approved') is True
        and approval.get('scope') == 'coding_pilot_v1_experiment'
        and bool(approval.get('owner_instruction')) and bool(approval.get('receipt_sha256'))
        and approval.get('previous_experiment_approval_sha256') == PARALLEL_PARENT_SHA256
        and previous.get('receipt_sha256') == PARALLEL_PARENT_SHA256
        and approval.get('limits_apply_cumulatively') is True
        and approval.get('supersedes_stage_budget_limits') is True
        and type(approval.get('gpu_hours')) in (int, float) and approval['gpu_hours'] == 500
        and type(approval.get('usd_cap')) in (int, float) and approval['usd_cap'] == 1000
        and approval.get('cleanup_reserve_usd') == 20
        and type(approval.get('max_concurrent_gpus')) is int
        and 1 <= approval['max_concurrent_gpus'] <= 8
        and approval.get('one_gpu_per_pod') is True
        and approval.get('allowed_gpu') == 'NVIDIA H200'
        and approval.get('gpu_price_ceiling_usd') == 5.5
        and approval.get('parallel_execution_approved') is True
        and approval.get('scientific_scope_unchanged') is True
        and approval.get('fixed_cohort_and_sampling_unchanged') is True)

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def bind(path, identity):
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        if old != identity: raise ValueError('Experiment identity changed; use a new attempt')
    else: write(path, identity)
    return digest(identity)

def seed_for(task_id, sample_index, stream):
    s = f'{NAMESPACE}|{task_id}|{sample_index}|{stream}'
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], 'big') % (2**63)

def decision(ledger, now=None):
    """Charge even stopped/unknown resources until absence is confirmed. No silent extensions."""
    now = time.time() if now is None else now
    usd = float(ledger.get('prior_upper_usd', 0)); gpu_hours = 0.0
    reasons = []; warnings = []
    cap, hours, dispatch_cap = CAP, HOURS, 280.0
    max_concurrent_gpus = 1
    experiment = ledger.get('approved_experiment_extension')
    experiment_valid = False
    if experiment is not None:
        experiment_valid = (isinstance(experiment,dict) and experiment.get('approved') is True
            and experiment.get('scope')=='coding_pilot_v1_experiment'
            and bool(experiment.get('owner_instruction')) and bool(experiment.get('receipt_sha256'))
            and experiment.get('limits_apply_cumulatively') is True
            and experiment.get('supersedes_stage_budget_limits') is True
            and experiment.get('one_gpu_only') is True and experiment.get('gpu_price_ceiling_usd')==5.5
            and experiment.get('scientific_scope_unchanged') is True
            and type(experiment.get('gpu_hours')) in (int,float) and math.isfinite(experiment['gpu_hours'])
            and type(experiment.get('usd_cap')) in (int,float) and math.isfinite(experiment['usd_cap'])
            and 48<=experiment['gpu_hours']<=100 and 300<=experiment['usd_cap']<=1000
            and experiment.get('cleanup_reserve_usd')==20)
        if not experiment_valid:reasons.append('invalid_experiment_approval')
        else:
            cap, hours = experiment['usd_cap'], experiment['gpu_hours']
            dispatch_cap = cap-20

    parallel = ledger.get('approved_parallel_extension')
    parallel_valid = experiment_valid and valid_parallel_approval(parallel, experiment)
    if parallel is not None:
        if not parallel_valid: reasons.append('invalid_parallel_approval')
        else:
            cap, hours, dispatch_cap = 1000.0, 500.0, 980.0
            max_concurrent_gpus = parallel['max_concurrent_gpus']

    active_gpu_count = 0
    active_cpu_count = 0
    inspection=ledger.get('approved_storage_inspection')
    inspection_valid=parallel_valid and valid_storage_inspection(inspection,parallel)
    active_upper_rate_usd = 0.0
    for r in ledger.get('resources', []):
        elapsed = max(0, r.get('absent_epoch', now) - r['started_epoch']) / 3600
        usd += elapsed * r['upper_rate_usd']
        gpu_count = r.get('gpu_count', 1) if r['kind'] == 'pod' else 0
        if r['kind'] == 'pod' and (type(gpu_count) is not int or gpu_count < 0):
            reasons.append('invalid_resource_gpu_count')
            gpu_count = 1
        gpu_hours += elapsed * gpu_count
        if 'absent_epoch' not in r:
            active_gpu_count += gpu_count
            active_upper_rate_usd += r['upper_rate_usd']
            if r['kind'] == 'pod':
                inspection_bound=storage_inspection_resource(r,inspection)
                if gpu_count==0:
                    active_cpu_count+=1
                    if not inspection_valid or not inspection_bound:reasons.append('unauthorized_cpu_storage_inspection')
                    elif not inspection['started_epoch']<=now<inspection['deadline_epoch']:
                        reasons.append('storage_inspection_deadline')
                elif gpu_count!=1 or inspection_bound:
                    reasons.append('allocation_outside_approved_hardware_or_quote')
    if active_cpu_count>1:reasons.append('cpu_storage_inspection_count')
    if active_gpu_count > max_concurrent_gpus:
        reasons.append('concurrent_gpu_limit')
    stage = ledger.get('stage', 'preflight')
    wall_hours = max(0, now - ledger.get('stage_started_epoch', now)) / 3600
    # The approved plan allocates GPU-hours by stage. Off-GPU recovery/monitor
    # intervals remain visible as wall time and still accrue any storage charge.
    stage_hours = max(0, gpu_hours - ledger.get('stage_start_gpu_hours', 0))
    stage_usd = usd - ledger.get('stage_start_upper_usd', 0)
    stage_ceiling=STAGE_HOURS[stage];stage_dollar_ceiling=25 if stage=='preflight' else None
    extension=ledger.get('approved_preflight_extension')
    if extension is not None:
        valid=(isinstance(extension,dict) and extension.get('approved') is True and extension.get('stage')=='preflight'
            and bool(extension.get('owner_instruction')) and bool(extension.get('receipt_sha256'))
            and isinstance(extension.get('gpu_hours'),(int,float)) and isinstance(extension.get('usd_cap'),(int,float))
            and 4<=extension.get('gpu_hours',0)<=HOURS and 25<=extension.get('usd_cap',0)<=CAP)
        if not valid:reasons.append('invalid_budget_approval')
        elif stage=='preflight':stage_ceiling=extension['gpu_hours'];stage_dollar_ceiling=extension['usd_cap']
    if experiment_valid:
        # The owner's replacement budget is cumulative across every stage and
        # restart. Fixed task membership and per-run scientific limits are separate.
        stage_ceiling=max(0,hours-ledger.get('stage_start_gpu_hours',0))
        stage_dollar_ceiling=max(0,cap-ledger.get('stage_start_upper_usd',0))
    checks = [('total_cost', usd, dispatch_cap), ('GPU_hours', gpu_hours, hours),
              ('stage_hours', stage_hours, stage_ceiling)]
    if stage == 'preflight': checks.append(('preflight_cost', stage_usd, stage_dollar_ceiling))
    for label, used, ceiling in checks:
        if used >= ceiling: reasons.append(label)
        elif used >= .75 * ceiling: warnings.append(label)
    if ledger.get('controller_lost'): reasons.append('controller_heartbeat_lost')
    if ledger.get('provider_state_unknown') and not ledger.get('provider_unknown_grace'):
        reasons.append('provider_state_unknown')
    dispatch_reasons = list(reasons)
    if ledger.get('provider_state_unknown') and 'provider_state_unknown' not in dispatch_reasons:
        dispatch_reasons.append('provider_state_unknown')
    if ledger.get('resource_registration_pending'):
        dispatch_reasons.append('resource_registration_pending')
    return dict(upper_usd=round(usd, 6), gpu_hours=gpu_hours, stage=stage,
                stage_hours=stage_hours,stage_gpu_hours=stage_hours,stage_wall_hours=wall_hours,stage_upper_usd=stage_usd,
                stop=bool(reasons), stop_reasons=reasons, warning=warnings,
                dispatch_blocked=bool(dispatch_reasons), dispatch_block_reasons=dispatch_reasons,
                stage_gpu_hour_ceiling=stage_ceiling,stage_dollar_ceiling=stage_dollar_ceiling,
                hard_total_cap_usd=cap, hard_total_gpu_hours=hours, cleanup_dispatch_ceiling_usd=dispatch_cap,
                max_concurrent_gpus=max_concurrent_gpus,parallel_execution_approved=parallel_valid,
                active_gpu_count=active_gpu_count,active_cpu_storage_inspection_count=active_cpu_count,
                storage_inspection_approved=inspection_valid,active_upper_rate_usd=active_upper_rate_usd)

def remaining_budget_seconds(ledger, now=None, *, dispatch_gpu_count=None, dispatch_upper_rate_usd=None):
    """Common wall-time allowance at the entire intended fleet's concurrent burn.

    Dispatchers must pass the intended total after allocation, not only new pods.
    Recorded pod rates are whole-pod dollars per hour, while GPU time is summed
    across GPU counts. One inactive future GPU conservatively costs $5.56/hour.
    """
    d = decision(ledger, now)
    if d['dispatch_blocked']: raise ValueError('Dispatch blocked: ' + str(d['dispatch_block_reasons']))
    count = max(1, d['active_gpu_count']) if dispatch_gpu_count is None else dispatch_gpu_count
    if type(count) is not int or not max(1, d['active_gpu_count']) <= count <= d['max_concurrent_gpus']:
        raise ValueError('Invalid intended concurrent GPU count')
    minimum_rate = max(count * 5.56, d['active_upper_rate_usd']
                       + max(0, count - d['active_gpu_count']) * 5.56)
    rate = minimum_rate if dispatch_upper_rate_usd is None else dispatch_upper_rate_usd
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate < minimum_rate:
        raise ValueError('Concurrent burn rate omits approved upper-rate resources')
    limits = [(d['stage_gpu_hour_ceiling'] - d['stage_gpu_hours']) / count,
              (d['hard_total_gpu_hours'] - d['gpu_hours']) / count,
              (d['cleanup_dispatch_ceiling_usd'] - d['upper_usd']) / rate]
    if d['stage_dollar_ceiling'] is not None:
        limits.append((d['stage_dollar_ceiling'] - d['stage_upper_usd']) / rate)
    return max(0, min(limits) * 3600)

def remaining_preflight_seconds(ledger,now=None):
    d=decision(ledger,now)
    if d['stage']!='preflight' or d['stop']:raise ValueError('Preflight dispatch blocked: '+str(d['stop_reasons']))
    return remaining_budget_seconds(ledger, now)
