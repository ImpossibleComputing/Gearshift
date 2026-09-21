#!/usr/bin/env python3
"""One Studio-owned restart of the approved six-task checkpoint, attempt 09."""
import argparse,fcntl,json,os,shlex,shutil,subprocess,sys,tarfile,threading,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import PREFIX,write,sha,decision,remaining_preflight_seconds
from gearshift.coding_checkpoint import validate_checkpoint
from gearshift.coding_snapshot import verify_archive
from coding_cloud_guard import cli,tick
from coding_cloud_session import C,ROOT,REMOTE,quote,remote,ssh_args,collect
IMAGE='runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'
NAMESPACE='preflight_v5'
ATTEMPT=9
REGION='CA-MTL-4'
PARENT_SHA='462d32ae82a34e02a28eb89e1eaeb2a02717680a244025a3739217f689847485'


def verified_approval(c):
    path=c/'experiment_extension_approved.json'
    a=json.loads(path.read_text())
    bound={**a,'receipt_sha256':sha(path)}
    check=decision({'approved_experiment_extension':bound})
    if check['stop'] or (check['hard_total_gpu_hours'],check['hard_total_cap_usd'])!=(100,1000):
        raise ValueError('Matching affirmative 100-hour/$1000 experiment approval required')
    return a


def prepare_checkpoint():
    archive=ROOT/'evidence/coding_pilot_v1/pod_backup/final_extension_08.tar.gz'
    if sha(archive)!=PARENT_SHA:raise ValueError('Preserved final checkpoint archive changed')
    stage=C/f'resume_parent_{ATTEMPT:02d}'
    if stage.exists():raise ValueError('Restart staging already exists; inspect before any retry')
    verify_archive(archive,stage)
    checked=validate_checkpoint(stage/'results/coding_pilot_v1/preflight_v4',ROOT,6)
    return stage,{k:v for k,v in checked.items() if k!='parent_identity'}


def package_upload(conn,stage):
    files={str(p.relative_to(ROOT)):p for pattern in ['gearshift/*.py','scripts/coding_*.py',
        'configs/coding_pilot_v1/**/*.json','data/coding_pilot_v1/visible/*.json'] for p in ROOT.glob(pattern)}
    for name in ['data/coding_pilot_v1/private/development.json','data/coding_pilot_v1/identity.json',
                 'evidence/coding_pilot_v1/allocation_resume_09.json','evidence/coding_pilot_v1/experiment_100h_approved.json']:
        files[name]=ROOT/name
    for p in (stage/'results').rglob('*'):
        if p.is_file():files[str(p.relative_to(stage))]=p
    for p in (stage/'evidence/coding_pilot_v1').rglob('*'):
        if p.is_file():files['evidence/coding_pilot_v1/restart_parent_08/'+str(p.relative_to(stage/'evidence/coding_pilot_v1'))]=p
    files['evidence/coding_pilot_v1/parent_snapshot_manifest.json']=stage/'SNAPSHOT_MANIFEST.json'
    hashes={name:sha(p) for name,p in files.items()}
    manifest=C/'resume_upload_09_manifest.json';write(manifest,{'files':hashes,'parent_archive_sha256':PARENT_SHA})
    package=C/'resume_upload_09.tar'
    with tarfile.open(package,'w') as tar:
        for name,p in files.items():tar.add(p,arcname=name,recursive=False)
        tar.add(manifest,arcname='evidence/coding_pilot_v1/resume_upload_09_manifest.json',recursive=False)
    with package.open('rb') as f:
        result=subprocess.run(ssh_args(conn)+['mkdir -p '+REMOTE+' && tar --no-same-owner --no-same-permissions -xf - -C '+REMOTE],
            stdin=f,capture_output=True,timeout=300)
    if result.returncode:raise RuntimeError('Verified restart upload failed: '+result.stderr.decode()[-500:])
    check="import hashlib,json,pathlib; r=pathlib.Path("+repr(REMOTE)+"); m=json.loads((r/'evidence/coding_pilot_v1/resume_upload_09_manifest.json').read_text()); assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in m['files'].items()); print(len(m['files']))"
    count=int(remote(conn,'python3 -c '+shlex.quote(check),120).strip())
    if count!=len(files):raise ValueError('Uploaded manifest membership differs')
    write(C/'resume_upload_verified.json',{'epoch':time.time(),'file_count':count,'manifest_sha256':sha(manifest),'archive_sha256':sha(package)})


def record_status(saved):
    path=saved/'evidence/coding_pilot_v1/worker_status.json'
    status=json.loads(path.read_text()) if path.exists() else {'state':'bootstrapping'}
    write(C/'latest_worker_status.json',status)
    progress=saved/f'results/coding_pilot_v1/{NAMESPACE}/progress.json'
    if progress.exists():write(C/'latest_progress.json',json.loads(progress.read_text()))
    alert=saved/'evidence/coding_pilot_v1/ALERT.json'
    if alert.exists():write(C/'worker_ALERT.json',json.loads(alert.read_text()))
    return status


def alive(conn):
    return bool(remote(conn,r"ps -eo args | awk '/python.*scripts\/coding_(bootstrap_resumed|preflight_resumed|download_models)\.py/ && !/awk/ {print}'",30).strip())


def run():
    approval=verified_approval(C)
    if (C/f'resume_session_{ATTEMPT:02d}.json').exists():raise ValueError('This bounded restart has already been attempted')
    for kind in ['pod','network-volume']:
        if any(r.get('name','').startswith(PREFIX) for r in cli(kind,'list')):
            raise ValueError('Existing pilot resources require inspection before allocation')
    if (ROOT/f'results/coding_pilot_v1/{NAMESPACE}').exists():raise ValueError('Restart namespace already exists')
    auth=json.loads((C/'authorization.json').read_text())
    if (auth['approved_cap_usd'],auth['maximum_single_gpu_hours'],auth['gpu_price_ceiling_usd'])!=(300,48,5.5):
        raise ValueError('Overall authorization differs')
    for name in ['data_gate.json','local_linux_sandbox_gate.json']:
        if not json.loads((ROOT/'evidence/coding_pilot_v1'/name).read_text())['passed']:raise ValueError('Required gate failed: '+name)
    probe=subprocess.run(['launchctl','print',f'gui/{os.getuid()}/com.gearshift.coding-pilot.watchdog'],capture_output=True,text=True)
    if probe.returncode or 'state = running' not in probe.stdout:raise ValueError('Independent watchdog must be running')
    guard_state=json.loads((C/'watchdog_status.json').read_text())
    if (guard_state.get('hard_total_gpu_hours'),guard_state.get('hard_total_cap_usd'))!=(100,1000) or time.time()-guard_state['epoch']>60:
        raise ValueError('Independent watchdog has not adopted the new experiment ceiling')
    stage,checkpoint=prepare_checkpoint()
    audit=C/f'resume_control_parent_{ATTEMPT-1:02d}';audit.mkdir(exist_ok=False)
    for p in C.glob('*.json'):shutil.copy2(p,audit/p.name)
    running=True
    def heartbeat():
        while running:
            write(C/'controller_heartbeat.json',{'epoch':time.time(),'pid':os.getpid()});time.sleep(10)
    write(C/'controller_heartbeat.json',{'epoch':time.time(),'pid':os.getpid()})
    threading.Thread(target=heartbeat,daemon=True).start()
    (C/'STOP').unlink(missing_ok=True);tick()
    ledger=json.loads((C/'ledger.json').read_text());allowance=remaining_preflight_seconds(ledger)
    if allowance<checkpoint['remaining_forecast_seconds']+900:raise ValueError('Approved remaining time cannot cover the fixed-cohort forecast plus setup')
    actual_quote=quote();started=time.time();deadline=started+allowance
    old_intent=json.loads((C/'allocation_intent.json').read_text())
    intent={**old_intent,'attempt':ATTEMPT,'attempt_started_epoch':started,'quote':actual_quote,'image_digest':IMAGE,'region':REGION}
    write(C/'allocation_intent.json',intent)
    write(C/f'resume_session_{ATTEMPT:02d}.json',{'epoch':started,'namespace':NAMESPACE,'parent_namespace':'preflight_v4',
        'parent_archive_sha256':PARENT_SHA,'approval_sha256':sha(C/'experiment_extension_approved.json'),
        'remaining_seconds':allowance,'deadline_epoch':deadline,'prior_usage':decision(ledger,started),'checkpoint':checkpoint})
    for name in ['ALERT.json','worker_ALERT.json','session_error.json','backup_verified.json','latest_worker_status.json']:(C/name).unlink(missing_ok=True)
    vol=None;pod=None;conn=None;final_verified=False;launched=False
    try:
        write(C/'session_status.json',{'state':'allocating_approved_resume','epoch':time.time(),'namespace':NAMESPACE})
        vol=cli('network-volume','create','--name',PREFIX+f'storage-{ATTEMPT:02d}','--size','250','--data-center-id',REGION)
        write(C/'volume.json',{k:vol.get(k) for k in ['id','name','size','dataCenterId']})
        write(C/'resource_receipts'/f"{vol['id']}.json",{'kind':'volume','id':vol['id'],'name':vol['name'],'started_epoch':started,'upper_rate_usd':.04})
        if (C/'STOP').exists():raise RuntimeError('Watchdog stopped allocation')
        pod=cli('pod','create','--name',PREFIX+f'preflight-{ATTEMPT:02d}','--image',IMAGE,'--gpu-id','NVIDIA H200','--gpu-count','1',
            '--cloud-type','SECURE','--data-center-ids',REGION,'--container-disk-in-gb','40','--network-volume-id',vol['id'],'--ports','22/tcp','--ssh')
        write(C/'pod.json',{k:pod.get(k) for k in ['id','name','imageName','costPerHr','adjustedCostPerHr','machineId','gpuCount','networkVolumeId']})
        write(C/'resource_receipts'/f"{pod['id']}.json",{'kind':'pod','id':pod['id'],'name':pod['name'],'started_epoch':started,'upper_rate_usd':5.52})
        if pod.get('gpuCount')!=1 or not 0<float(pod.get('costPerHr') or 0)<=5.5:raise ValueError('Allocation count or actual price exceeds authorization')
        allocation={'pod_id':pod['id'],'volume_id':vol['id'],'started_epoch':started,'preflight_deadline_epoch':deadline,
            'image_digest':IMAGE,'quote':actual_quote,'cap_usd':1000,'preflight_cap_usd':1000,'preflight_gpu_hours':100,
            'remaining_authorized_seconds':allowance,'prior_usage':decision(ledger,started),'session':ATTEMPT,
            'approval_sha256':sha(C/'experiment_extension_approved.json'),'parent_archive_sha256':PARENT_SHA,'parent_checkpoint':checkpoint}
        write(ROOT/'evidence/coding_pilot_v1/allocation_resume_09.json',allocation)
        shutil.copy2(C/'experiment_extension_approved.json',ROOT/'evidence/coding_pilot_v1/experiment_100h_approved.json')
        for _ in range(60):
            if (C/'STOP').exists():raise RuntimeError('Watchdog stop during startup')
            try:
                info=cli('ssh','info',pod['id']);candidate=info.get('connection',info)
                if candidate and candidate.get('ip'):remote(candidate,'true',30);conn=candidate;break
            except Exception:pass
            time.sleep(10)
        if conn is None:raise TimeoutError('Startup exceeded ten minutes')
        write(C/'connection.json',conn);package_upload(conn,stage)
        launcher="import subprocess,pathlib; p=pathlib.Path("+repr(REMOTE)+"); f=(p/'evidence/coding_pilot_v1/bootstrap.log').open('ab'); child=subprocess.Popen(['python3','scripts/coding_bootstrap_resumed.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(child.pid)"
        pid=int(remote(conn,'python3 -c '+shlex.quote(launcher),30).strip());launched=True
        write(C/'bootstrap_process.json',{'pid':pid,'pod_id':pod['id'],'epoch':time.time(),'script':'coding_bootstrap_resumed.py'})
        write(C/'session_status.json',{'state':'approved_resume_running','pod_id':pod['id'],'namespace':NAMESPACE,'epoch':time.time(),'deadline_epoch':deadline})
        last_backup=time.time()
        while time.time()<deadline-180 and not (C/'STOP').exists():
            try:
                saved=collect(conn);last_backup=time.time();status=record_status(saved)
                if status['state'] in ['stopped','preflight_baselines_finished','bootstrap_failed']:
                    if not alive(conn):break
            except Exception as exc:
                write(C/'backup_retry.json',{'epoch':time.time(),'error':str(exc),'last_verified_epoch':last_backup})
                if time.time()-last_backup>600:raise RuntimeError('Ten minutes without a verified backup')
            time.sleep(30)
        remote(conn,'touch '+REMOTE+'/evidence/coding_pilot_v1/STOP',30)
        for _ in range(12):
            if not alive(conn):break
            time.sleep(5)
        else:raise RuntimeError('Worker did not drain; retain storage')
        for attempt in range(3):
            try:saved=collect(conn);record_status(saved);break
            except Exception:
                if attempt==2:raise
                time.sleep(5)
        latest=ROOT/'evidence/coding_pilot_v1/pod_backup/latest.tar.gz';second=latest.with_name(f'final_resume_{ATTEMPT:02d}.tar.gz');shutil.copy2(latest,second)
        if sha(latest)!=sha(second):raise RuntimeError('Final backup copies differ')
        write(C/'backup_verified.json',{'epoch':time.time(),'pod_id':pod['id'],'volume_id':vol['id'],'worker_confirmed_absent':True,
            'copies':[str(latest.relative_to(ROOT)),str(second.relative_to(ROOT))],'sha256':sha(latest)})
        final_verified=True
    except BaseException as exc:
        write(C/'session_error.json',{'epoch':time.time(),'error':str(exc)});traceback.print_exc()
        if conn:
            try:remote(conn,'touch '+REMOTE+'/evidence/coding_pilot_v1/STOP',30);collect(conn)
            except Exception:pass
    finally:
        (C/'STOP').touch()
        if pod:
            try:cli('pod','delete',pod['id'])
            except Exception as exc:write(C/'cleanup_pending.json',{'epoch':time.time(),'error':str(exc)})
        if vol and (final_verified or not launched):
            try:cli('network-volume','delete',vol['id'])
            except Exception as exc:write(C/'volume_cleanup_pending.json',{'epoch':time.time(),'error':str(exc)})
        running=False
        write(C/'session_status.json',{'state':'stopped_backed_up' if final_verified else 'stopped_inspect_backup','namespace':NAMESPACE,'epoch':time.time()})
        tick()


def capacity_retry(previous, error, ledger):
    # A provider capacity rejection before any GPU exists may try a different
    # region. Any allocated or ambiguous attempt requires explicit inspection.
    prior_attempt=int(previous['attempt'])
    if not 9<=prior_attempt<12 or 'no longer any instances available' not in error.get('error',''):
        raise ValueError('Not a bounded pre-allocation capacity failure')
    if any(r['kind']=='pod' and r['started_epoch']>=previous['attempt_started_epoch'] for r in ledger['resources']):
        raise ValueError('Prior attempt allocated compute; capacity retry forbidden')
    return prior_attempt+1


def main():
    global ATTEMPT,REGION
    os.chdir(ROOT)
    parser=argparse.ArgumentParser();parser.add_argument('--capacity-retry',action='store_true');parser.add_argument('--region',default='CA-MTL-4')
    args=parser.parse_args();REGION=args.region
    with (C/'recovery_controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.capacity_retry:
            prior=json.loads((C/'allocation_intent.json').read_text())
            ATTEMPT=capacity_retry(prior,json.loads((C/'session_error.json').read_text()),json.loads((C/'ledger.json').read_text()))
            if REGION==prior['region']:raise ValueError('Capacity retry must use another discovered location')
            locations=cli('datacenter','list')
            if not any(row['id']==REGION and any(g.get('gpuId')=='NVIDIA H200' and g.get('stockStatus') in ['Low','Medium','High'] for g in row.get('gpuAvailability',[])) for row in locations):
                raise ValueError('Requested location does not report H200 availability')
        run()

if __name__=='__main__':main()
