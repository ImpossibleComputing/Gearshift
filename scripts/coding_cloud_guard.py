#!/usr/bin/env python3
"""Independent Studio watchdog; discovers only this pilot's tagged resources."""
import fcntl,json,math,os,re,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import PREFIX,decision,write,sha,valid_storage_inspection,storage_inspection_resource
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'evidence/coding_pilot_v1/control'
CLI='/opt/homebrew/bin/runpodctl'


def cli(*args):
    p=subprocess.run([CLI,*args,'--output','json'],capture_output=True,text=True,timeout=30)
    if p.returncode:raise RuntimeError('Provider '+args[0]+' '+args[1]+' failed: '+p.stderr[-500:])
    return json.loads(p.stdout) if p.stdout.strip() else {}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def safe_id(value):
    return isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',value) is not None


def backup_for(resource):
    if safe_id(resource['id']):
        scoped=read_json(C/'backups_verified'/f"{resource['id']}.json")
        if scoped.get('verified') is True and scoped.get('volume_id')==resource['id']:
            return scoped
    old=read_json(C/'backup_verified.json')
    final_proof=(old.get('worker_confirmed_absent') is True
        and re.fullmatch(r'[0-9a-f]{64}',str(old.get('sha256',''))) is not None
        and len(old.get('copies',[]))>=2)
    return old if (old.get('verified') is True or final_proof) and old.get('volume_id')==resource['id'] else {}


def fresh_backup(resource,now):
    proof=backup_for(resource)
    if safe_id(resource['id']):
        ack=read_json(C/'backups_ack'/f"{resource['id']}.json")
        if ack.get('verified') is True and ack.get('volume_id')==resource['id']:
            proof=ack
    if not resource.get('controller_id'):
        # The original single-worker collector binds its live ack through the
        # current volume receipt; do not apply it to other parallel volumes.
        ack=read_json(C/'backup_ack.json');volume=read_json(C/'volume.json')
        if volume.get('id')==resource['id'] and ack.get('verified_files',0)>0 and ack.get('archive_sha256'):
            proof=ack
    return 0<=now-proof.get('epoch',0)<=300


def fresh_heartbeat(resource,now,intent):
    owner=resource.get('controller_id')
    if owner is None:
        h=read_json(C/'controller_heartbeat.json')
    elif safe_id(owner):
        h=read_json(C/'controller_heartbeats'/f'{owner}.json')
        if h.get('controller_id')!=owner:return False
    else:return False
    epoch=h.get('epoch',resource.get('started_epoch',intent['started_epoch']))
    return type(epoch) in (int,float) and -60<=now-epoch<=300


def tick():
    # Controllers also call tick before dispatch. Serialize the complete
    # read/discover/write cycle so parallel callers cannot lose usage receipts.
    C.mkdir(parents=True,exist_ok=True)
    with (C/'ledger_tick.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _tick()


def _tick():
    if not (C/'allocation_intent.json').exists():return
    intent=read_json(C/'allocation_intent.json');now=time.time()
    ledger=read_json(C/'ledger.json') or {'stage':'preflight','stage_started_epoch':intent['started_epoch'],'resources':[]}
    for filename,key in [('preflight_extension_approved.json','approved_preflight_extension'),
                         ('experiment_extension_approved.json','approved_experiment_extension'),
                         ('parallel_500h_approved.json','approved_parallel_extension'),
                         ('storage_inspection_intent.json','approved_storage_inspection')]:
        approval=C/filename
        if approval.exists():ledger[key]={**read_json(approval),'receipt_sha256':sha(approval)}
    known={(r['kind'],r['id']):r for r in ledger['resources']}
    for path in (C/'resource_receipts').glob('*.json'):
        r=read_json(path);key=(r['kind'],r['id'])
        if key not in known:known[key]=r
        else:
            # Identity/timing never moves forward on refresh. Allow the launch
            # receipt to bind ownership and hardware before discovery catches up.
            for k in ('absent_epoch','controller_id','gpu_count','gpu_type','storage_inspection_intent_sha256','volume_id'):
                if k in r:known[key][k]=r[k]
    authorized=decision(ledger,now)
    parallel=authorized['parallel_execution_approved']
    inspection=ledger.get('approved_storage_inspection')
    inspection_valid=parallel and valid_storage_inspection(inspection,ledger.get('approved_parallel_extension'))
    planned={}
    if parallel:
        for path in (C/'allocation_intents').glob('*.json'):
            item=read_json(path)
            if not safe_id(item.get('controller_id')):continue
            if type(item.get('started_epoch')) not in (int,float):continue
            for name in item.get('resource_names',[]):
                if isinstance(name,str) and name.startswith(PREFIX):planned[name]=item
    state_unknown=False;active=[];hardware_violations=[]
    for kind,command in [('pod','pod'),('volume','network-volume')]:
        try:
            listing=cli(command,'list')
            if not isinstance(listing,list):raise ValueError('Provider resource listing is not a list')
            for row in listing:
                if not row.get('name','').startswith(PREFIX):continue
                ident=row['id'];key=(kind,ident);active.append((kind,ident))
                if key not in known:
                    plan=planned.get(row['name'],{})
                    inspecting=kind=='pod' and inspection_valid and row['name']==inspection['resource_name']
                    if inspecting:plan={**inspection,'storage_inspection_intent_sha256':inspection['receipt_sha256']}
                    known[key]={'kind':kind,'id':ident,'name':row['name'],
                        'started_epoch':plan.get('started_epoch',intent.get('attempt_started_epoch',intent['started_epoch'])),
                        'upper_rate_usd':(.50 if inspecting else 5.52) if kind=='pod' else .04}
                    for field in ('controller_id','gpu_type','gpu_count','storage_inspection_intent_sha256','volume_id'):
                        if field in plan:known[key][field]=plan[field]
                resource=known[key]
                if 'absent_epoch' in resource:raise ValueError('Previously absent resource reappeared')
                if kind=='pod':
                    inspecting=inspection_valid and storage_inspection_resource(resource,inspection)
                    price=float(row.get('costPerHr') or row.get('adjustedCostPerHr') or 5.5)
                    if not math.isfinite(price):raise ValueError('Provider quote is not finite')
                    count=row.get('gpuCount',resource.get('gpu_count',1))
                    resource['gpu_count']=count
                    resource['upper_rate_usd']=max(resource['upper_rate_usd'],price if inspecting else price+.02)
                    model=row.get('gpuTypeId') or row.get('gpuDisplayName') or (row.get('machine') or {}).get('gpuDisplayName') or resource.get('gpu_type')
                    # An allocation can appear in provider discovery before the
                    # create call returns its ID for writing the owner receipt.
                    awaiting_receipt=parallel and resource.get('controller_id') is None
                    if inspecting:
                        # CPU ownership is bound to this exact existing volume.
                        # Provider GPU count must be explicit, never inferred.
                        if (type(row.get('gpuCount')) is not int or row['gpuCount']!=0 or not 0<price<=.5
                            or row.get('networkVolumeId',resource.get('volume_id'))!=inspection['volume_id']):
                            hardware_violations.append(ident)
                    elif (price>5.5 or price<=0 or count!=1 or
                        (parallel and model not in ('NVIDIA H200','H200','H200 SXM') and not (awaiting_receipt and model is None))):
                        hardware_violations.append(ident)
            ids={x['id'] for x in listing}
            for (k,ident),r in known.items():
                if k==kind and ident not in ids and 'absent_epoch' not in r:r['absent_epoch']=now
        except Exception as exc:
            state_unknown=True;write(C/'provider_error.json',{'epoch':now,'error':str(exc)})
    ledger['resources']=list(known.values());ledger['provider_state_unknown']=state_unknown
    pending=[r for r in known.values() if 'absent_epoch' not in r]
    stale=[];registering=[]
    for r in pending:
        if parallel and r.get('controller_id') is None:
            first=r.setdefault('registration_started_epoch',now)
            if now-first<60:registering.append(r['id'])
            else:stale.append(r)
        elif not fresh_heartbeat(r,now,intent):stale.append(r)
    ledger['resource_registration_pending']=registering
    # A transient API failure blocks dispatch immediately. Existing workers get
    # at most two failed ticks / 120 seconds, and only with healthy supervision
    # and fresh verified backups. Known resources remain charged throughout.
    if state_unknown:
        ledger['provider_failure_streak']=ledger.get('provider_failure_streak',0)+1
        ledger.setdefault('provider_first_failure_epoch',now)
        volumes=[r for r in pending if r['kind']=='volume']
        backups_fresh=bool(volumes) and all(fresh_backup(r,now) for r in volumes)
        ledger['provider_unknown_grace']=(ledger['provider_failure_streak']<3
            and now-ledger['provider_first_failure_epoch']<120 and not stale and backups_fresh)
    else:
        ledger['provider_failure_streak']=0;ledger.pop('provider_first_failure_epoch',None)
        ledger['provider_last_success_epoch']=now;ledger['provider_unknown_grace']=False
    ledger['controller_lost']=bool(stale) and not parallel
    ledger['quote_violation']=bool(hardware_violations)
    if ledger['resources'] and not pending and not state_unknown:
        ledger['controller_lost']=False
        end=max(r['absent_epoch'] for r in ledger['resources'])
        write(C/'ledger.json',ledger)
        write(C/'watchdog_status.json',{'epoch':now,'pid':os.getpid(),'state':'all_task_resources_absent',
            'active_resources':[],**decision(ledger,end)})
        return
    d=decision(ledger,now)
    expired_inspections={r['id'] for r in pending if inspection_valid and storage_inspection_resource(r,inspection)
        and not inspection['started_epoch']<=now<inspection['deadline_epoch']}
    if expired_inspections:
        # The cleanup mount's deadline cannot cancel healthy research shards.
        d['stop_reasons']=[reason for reason in d['stop_reasons'] if reason!='storage_inspection_deadline']
        d['stop']=bool(d['stop_reasons'])
        d['scoped_stop_reasons']=['storage_inspection_deadline']
    if hardware_violations:
        d['stop']=True;d['stop_reasons'].append('allocation_outside_approved_hardware_or_quote')
    if (C/'STOP').exists():d['stop']=True;d['stop_reasons'].append('stop_marker')
    if d['stop']:
        d['dispatch_blocked']=True
        d['dispatch_block_reasons']=list(dict.fromkeys(d['dispatch_block_reasons']+d['stop_reasons']))
    # Scope a lost shard controller to its own resources; an unrelated healthy
    # controller cannot mask it or be killed just because that shard failed.
    scoped_ids=({r['id'] for r in stale} if parallel else set())|expired_inspections
    write(C/'ledger.json',ledger)
    write(C/'watchdog_status.json',{'epoch':now,'pid':os.getpid(),'active_resources':active,
        'controller_stop_resource_ids':sorted(scoped_ids),**d})
    if d['warning'] or d['stop'] or scoped_ids:
        limit_event=bool(d['warning'] or set(d['stop_reasons']) & {'total_cost','GPU_hours','stage_hours','preflight_cost'})
        write(C/'ALERT.json',{'epoch':now,**d,'event':'limit' if limit_event else 'worker_or_controller_stop',
            'controller_stop_resource_ids':sorted(scoped_ids),'approval_required_to_extend':limit_event})
    if d['stop']:(C/'STOP').touch()
    targets=pending if d['stop'] else [r for r in pending if r['id'] in scoped_ids]
    for r in targets:
        if r.get('controller_id') and safe_id(r['controller_id']):
            write(C/'controller_stops'/f"{r['controller_id']}.json",{'epoch':now,'controller_id':r['controller_id'],
                'resource_id':r['id'],'reason':d['stop_reasons'] if d['stop'] else
                ['storage_inspection_deadline'] if r['id'] in expired_inspections else ['controller_heartbeat_lost']})
        if r['kind']=='volume' and not backup_for(r):continue
        try:cli('pod' if r['kind']=='pod' else 'network-volume','delete',r['id'])
        except Exception as exc:write(C/('cleanup_error_'+r['id']+'.json'),{'epoch':now,'error':str(exc)})


def main():
    C.mkdir(parents=True,exist_ok=True)
    with (C/'watchdog.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:tick()
            except Exception as e:write(C/'watchdog_error.json',{'epoch':time.time(),'error':str(e)})
            time.sleep(15)
if __name__=='__main__':main()
