#!/usr/bin/env python3
"""One bounded recovery of the verified interrupted preflight on retained storage."""
import fcntl,json,os,shlex,shutil,subprocess,sys,tarfile,threading,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import PREFIX,write,sha,decision
from coding_cloud_guard import cli,tick
from coding_cloud_session import C,ROOT,REMOTE,quote,remote,ssh_args,collect
VOLUME='d57n8t2mm4'
IMAGE='runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'

def remaining_seconds(ledger,now=None):
    d=decision(ledger,now)
    if d['stop']:raise ValueError('Stop condition must be resolved before allocating: '+str(d['stop_reasons']))
    return max(0,min((4-d['stage_gpu_hours'])*3600,(48-d['gpu_hours'])*3600,
        (25-d['stage_upper_usd'])/5.56*3600,(280-d['upper_usd'])/5.56*3600))

def upload(conn,files,name):
    package=C/name
    with tarfile.open(package,'w') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(ROOT),recursive=False)
    manifest={'epoch':time.time(),'archive_sha256':sha(package),'files':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    write(C/(name+'.manifest.json'),manifest)
    with package.open('rb') as f:
        p=subprocess.run(ssh_args(conn)+['tar --no-same-owner --no-same-permissions -xf - -C '+REMOTE],stdin=f,capture_output=True,timeout=120)
    if p.returncode:raise RuntimeError('Recovery upload failed: '+p.stderr.decode()[-500:])
    return manifest

def record_status(saved):
    p=saved/'evidence/coding_pilot_v1/worker_status.json'
    status=json.loads(p.read_text()) if p.exists() else {'state':'bootstrapping'}
    write(C/'latest_worker_status.json',status)
    p=saved/'results/coding_pilot_v1/preflight_v3/progress.json'
    if p.exists():write(C/'latest_progress.json',json.loads(p.read_text()))
    p=saved/'evidence/coding_pilot_v1/ALERT.json'
    if p.exists():write(C/'worker_ALERT.json',json.loads(p.read_text()))
    return status

def main():
    os.chdir(ROOT)
    with (C/'recovery_controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()

def run():
    if (C/'recovery_attempt_07.json').exists():raise RuntimeError('This bounded recovery has already been attempted; inspect its receipt before any further allocation')
    pods=[r for r in cli('pod','list') if r.get('name','').startswith(PREFIX)]
    volumes=[r for r in cli('network-volume','list') if r.get('name','').startswith(PREFIX)]
    if pods or len(volumes)!=1 or volumes[0]['id']!=VOLUME:raise RuntimeError('Unexpected task resource membership; no mutation')
    vol=volumes[0]
    if vol['name']!=PREFIX+'storage-06' or vol['dataCenterId']!='CA-MTL-4':raise RuntimeError('Retained-volume identity mismatch')
    auth=json.loads((C/'authorization.json').read_text())
    if (auth['approved_cap_usd'],auth['preflight_cap_usd'],auth['preflight_hours'])!=(300,25,4):raise RuntimeError('Authorization mismatch')
    probe=subprocess.run(['launchctl','print',f'gui/{os.getuid()}/com.gearshift.coding-pilot.watchdog'],capture_output=True,text=True)
    if probe.returncode or 'state = running' not in probe.stdout:raise RuntimeError('Independent watchdog must be running')
    archive=C/'recovery_parent_06';archive.mkdir(exist_ok=False)
    for p in C.glob('*.json'):shutil.copy2(p,archive/p.name)
    prior_manifest=json.loads((C/'upload_manifest.json').read_text())
    numerical=['gearshift/core.py','gearshift/coding_inference.py','gearshift/coding_sandbox.py','scripts/coding_sandbox_child.py']
    # Parent v2 identity is authoritative for the pre-generation input correction.
    parent=json.loads((ROOT/'results/coding_pilot_v1/preflight_v2/identity.json').read_text())
    expected={f:(prior_manifest['files'][f] if f=='gearshift/core.py' else parent['implementation'][f]) for f in numerical}
    if any(sha(ROOT/f)!=h for f,h in expected.items()):raise RuntimeError('Recovery changed numerical generation/scoring code')
    running=True
    def heartbeat():
        while running:
            write(C/'controller_heartbeat.json',{'epoch':time.time(),'pid':os.getpid()});time.sleep(10)
    write(C/'controller_heartbeat.json',{'epoch':time.time(),'pid':os.getpid()})
    threading.Thread(target=heartbeat,daemon=True).start()
    # Fresh provider discovery and heartbeat, not elapsed downtime, resolve stale
    # operational flags. All recorded cost/GPU hours remain cumulative.
    (C/'STOP').unlink(missing_ok=True);tick()
    ledger=json.loads((C/'ledger.json').read_text());allowance=remaining_seconds(ledger)
    if allowance<900:raise RuntimeError('Less than 15 minutes of authorized preflight remains')
    actual_quote=quote();started=time.time();deadline=started+allowance
    intent=json.loads((C/'allocation_intent.json').read_text())
    intent.update(attempt=7,attempt_started_epoch=started,quote=actual_quote,image_digest=IMAGE,region='CA-MTL-4',reused_volume_id=VOLUME)
    write(C/'allocation_intent.json',intent)
    receipt={'epoch':started,'retained_volume_id':VOLUME,'remaining_authorized_seconds':allowance,
        'prior_budget':decision(ledger,started),'numerical_source_hashes':expected,'no_new_sampling_draw':True}
    write(C/'recovery_attempt_07.json',receipt)
    for name in ['ALERT.json','worker_ALERT.json','session_error.json','backup_verified.json']:(C/name).unlink(missing_ok=True)
    conn=None;pod=None;final_verified=False
    try:
        write(C/'session_status.json',{'state':'recovering_allocation','epoch':time.time()})
        pod=cli('pod','create','--name',PREFIX+'preflight-07','--image',IMAGE,'--gpu-id','NVIDIA H200','--gpu-count','1',
            '--cloud-type','SECURE','--data-center-ids','CA-MTL-4','--container-disk-in-gb','40','--network-volume-id',VOLUME,'--ports','22/tcp','--ssh')
        write(C/'pod.json',{k:pod.get(k) for k in ['id','name','imageName','costPerHr','adjustedCostPerHr','machineId','gpuCount','networkVolumeId']})
        write(C/'resource_receipts'/f"{pod['id']}.json",{'kind':'pod','id':pod['id'],'name':pod['name'],'started_epoch':started,'upper_rate_usd':5.52})
        if float(pod.get('costPerHr') or 5.5)>5.5:raise RuntimeError('Actual price exceeds approved hardware quote')
        allocation={'pod_id':pod['id'],'volume_id':VOLUME,'started_epoch':started,'preflight_deadline_epoch':deadline,
            'image_digest':IMAGE,'quote':actual_quote,'cap_usd':300,'preflight_cap_usd':25,'preflight_gpu_hours':4,
            'remaining_authorized_seconds':allowance,'prior_usage':receipt['prior_budget'],'recovery_attempt':7}
        write(ROOT/'evidence/coding_pilot_v1/allocation.json',allocation)
        for _ in range(60):
            if (C/'STOP').exists():raise RuntimeError('Watchdog stop during startup')
            try:
                info=cli('ssh','info',pod['id']);candidate=info.get('connection',info)
                if candidate and candidate.get('ip'):
                    remote(candidate,'true',30);conn=candidate;break
            except Exception:pass
            time.sleep(10)
        if conn is None:raise TimeoutError('Recovery startup exceeded ten minutes')
        write(C/'connection.json',conn)
        check='import hashlib,json,pathlib; r=pathlib.Path('+repr(REMOTE)+'); e='+repr(expected)+'; assert all(hashlib.sha256((r/f).read_bytes()).hexdigest()==h for f,h in e.items()); print("Numerical source hashes match")'
        remote(conn,'python3 -c '+shlex.quote(check))
        upload(conn,[ROOT/'gearshift/coding_snapshot.py',ROOT/'scripts/coding_snapshot.py'],'snapshot_helper.tar')
        saved=collect(conn)
        parent_archive=ROOT/'evidence/coding_pilot_v1/pod_backup/recovery_parent_06.tar.gz'
        shutil.copy2(parent_archive.parent/'latest.tar.gz',parent_archive)
        write(C/'recovery_parent_snapshot.json',{'epoch':time.time(),'sha256':sha(parent_archive),'bytes':parent_archive.stat().st_size})
        # Preserve all old generic evidence before bootstrap rewrites its own status.
        preserve='import pathlib,shutil; e=pathlib.Path('+repr(REMOTE+'/evidence/coding_pilot_v1')+'); d=e/"recovery_parent_06"; assert not d.exists(); paths=list(e.iterdir()); d.mkdir(); [shutil.copytree(p,d/p.name) if p.is_dir() else shutil.copy2(p,d/p.name) for p in paths if p.name!="control"]'
        remote(conn,'python3 -c '+shlex.quote(preserve))
        files=[p for pat in ['gearshift/*.py','scripts/coding_*.py','configs/coding_pilot_v1/**/*.json'] for p in ROOT.glob(pat)]
        files+=[ROOT/'evidence/coding_pilot_v1/allocation.json']
        upload(conn,files,'recovery_upload.tar')
        remote(conn,'rm -f '+REMOTE+'/evidence/coding_pilot_v1/STOP '+REMOTE+'/evidence/coding_pilot_v1/ALERT.json')
        launcher="import subprocess,pathlib; p=pathlib.Path("+repr(REMOTE)+"); f=(p/'evidence/coding_pilot_v1/bootstrap.log').open('ab'); child=subprocess.Popen(['python3','scripts/coding_bootstrap.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(child.pid)"
        pid=remote(conn,'python3 -c '+shlex.quote(launcher),30)
        write(C/'bootstrap_process.json',{'pid':int(pid.strip()),'pod_id':pod['id'],'epoch':time.time()})
        write(C/'session_status.json',{'state':'preflight_recovery_running','pod_id':pod['id'],'epoch':time.time()})
        last_backup=time.time();consecutive=0
        while time.time()<deadline-180 and not (C/'STOP').exists():
            try:
                saved=collect(conn);last_backup=time.time();consecutive=0;status=record_status(saved)
                if status['state'] in ['stopped','preflight_baselines_finished','bootstrap_failed']:
                    # Ignore only the preserved old status before new bootstrap starts.
                    stamp=max(status.get('finished_epoch',0),status.get('heartbeat_epoch',0),status.get('epoch',0))
                    if stamp>=started:break
            except Exception as exc:
                consecutive+=1;write(C/'backup_retry.json',{'epoch':time.time(),'error':str(exc),'consecutive':consecutive,'last_verified_epoch':last_backup})
                if time.time()-last_backup>600:raise RuntimeError('Verified backup is over ten minutes old; stopping compute')
            time.sleep(30)
        remote(conn,'touch '+REMOTE+'/evidence/coding_pilot_v1/STOP',30)
        # A terminal worker receipt and absent bootstrap/preflight processes make
        # the final snapshot stable; never certify an active worker as final.
        settled=False
        for _ in range(12):
            alive=remote(conn,r"ps -eo args | awk '/python.*scripts\/coding_(bootstrap|preflight)\.py/ && !/awk/ {print}'",30).strip()
            if not alive:settled=True;break
            time.sleep(5)
        if not settled:raise RuntimeError('Worker did not drain within one minute; preserve task storage')
        for attempt in range(3):
            try:saved=collect(conn);record_status(saved);break
            except Exception:
                if attempt==2:raise
                time.sleep(5)
        latest=ROOT/'evidence/coding_pilot_v1/pod_backup/latest.tar.gz';second=latest.with_name('final_recovery_07.tar.gz')
        shutil.copy2(latest,second)
        if sha(latest)!=sha(second):raise RuntimeError('Final backup copies differ')
        write(C/'backup_verified.json',{'epoch':time.time(),'pod_id':pod['id'],'volume_id':VOLUME,'worker_confirmed_absent':True,
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
        if final_verified:
            try:cli('network-volume','delete',VOLUME)
            except Exception as exc:write(C/'volume_cleanup_pending.json',{'epoch':time.time(),'error':str(exc)})
        running=False
        write(C/'session_status.json',{'state':'stopped_backed_up' if final_verified else 'stopped_inspect_backup','epoch':time.time()})
        tick()

if __name__=='__main__':main()
