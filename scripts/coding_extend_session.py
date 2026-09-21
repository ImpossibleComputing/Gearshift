#!/usr/bin/env python3
"""Adopt the existing task pod and continue at its five-case transaction boundary."""
import fcntl,json,os,shlex,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import PREFIX,write,sha,decision,remaining_preflight_seconds
from gearshift.coding_reuse import compatible_parent,verify_completed
from coding_cloud_guard import cli,tick
from coding_cloud_session import C,ROOT,REMOTE,remote,collect
from coding_recover_session import upload
POD='zhd2nwoyivylca';VOLUME='d57n8t2mm4'

def status_from(saved,namespace):
    status=json.loads((saved/'evidence/coding_pilot_v1/worker_status.json').read_text())
    write(C/'latest_worker_status.json',status)
    p=saved/f'results/coding_pilot_v1/{namespace}/progress.json'
    if p.exists():write(C/'latest_progress.json',json.loads(p.read_text()))
    p=saved/'evidence/coding_pilot_v1/ALERT.json'
    if p.exists():write(C/'worker_ALERT.json',json.loads(p.read_text()))
    return status

def alive(conn):
    cmd=r"ps -eo args | awk '/python.*scripts\/coding_(bootstrap|preflight|preflight_extended)\.py/ && !/awk/ {print}'"
    return bool(remote(conn,cmd,30).strip())

def main():
    os.chdir(ROOT)
    with (C/'recovery_controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()

def run():
    approval=json.loads((C/'preflight_extension_approved.json').read_text())
    if not approval['approved'] or (approval['gpu_hours'],approval['usd_cap'])!=(10,150):raise ValueError('Approved extension receipt required')
    if (C/'extension_session_08.json').exists():raise ValueError('Extension controller already attempted; inspect before retrying')
    pods=[r for r in cli('pod','list') if r.get('name','').startswith(PREFIX)]
    vols=[r for r in cli('network-volume','list') if r.get('name','').startswith(PREFIX)]
    if len(pods)!=1 or pods[0]['id']!=POD or len(vols)!=1 or vols[0]['id']!=VOLUME:raise ValueError('Adoption requires the exact existing task resources')
    if pods[0]['gpuCount']!=1 or float(pods[0]['costPerHr'])>5.5:raise ValueError('Allocation differs from approval')
    conn=json.loads((C/'connection.json').read_text());remote(conn,'true',30)
    archive=C/'extension_parent_07';archive.mkdir(exist_ok=False)
    for p in C.glob('*.json'):shutil.copy2(p,archive/p.name)
    running=True
    def heartbeat():
        while running:
            write(C/'controller_heartbeat.json',{'epoch':time.time(),'pid':os.getpid()});time.sleep(10)
    write(C/'controller_heartbeat.json',{'epoch':time.time(),'pid':os.getpid()})
    threading.Thread(target=heartbeat,daemon=True).start()
    (C/'STOP').unlink(missing_ok=True);tick()
    ledger=json.loads((C/'ledger.json').read_text());allowance=remaining_preflight_seconds(ledger);started=time.time();deadline=started+allowance
    write(C/'extension_session_08.json',{'epoch':started,'pod_id':POD,'volume_id':VOLUME,'approval_sha256':sha(C/'preflight_extension_approved.json'),
        'remaining_seconds':allowance,'deadline_epoch':deadline,'adopted_existing_pod':True,'prior_usage':decision(ledger,started)})
    final_verified=False;last_backup=time.time();namespace='preflight_v3';launched=False
    try:
        write(C/'session_status.json',{'state':'finishing_parent_transaction_boundary','pod_id':POD,'epoch':time.time(),'deadline_epoch':deadline})
        while time.time()<deadline-180 and not (C/'STOP').exists():
            try:
                saved=collect(conn);last_backup=time.time();status=status_from(saved,namespace)
            except Exception as exc:
                write(C/'backup_retry.json',{'epoch':time.time(),'error':str(exc),'last_verified_epoch':last_backup})
                if time.time()-last_backup>600:raise RuntimeError('Verified backup older than ten minutes')
                time.sleep(15);continue
            terminal=status['state'] in ['stopped','preflight_baselines_finished','bootstrap_failed']
            if terminal and not launched:
                expected='Measured preflight forecast exceeds ceiling; stop without shrinking cohort'
                if status.get('error')!=expected:raise RuntimeError('Parent stopped for a different reason: '+str(status))
                if alive(conn):time.sleep(2);continue
                parent=saved/'results/coding_pilot_v1/preflight_v3';ident=compatible_parent(parent,ROOT)
                tasks=json.loads((ROOT/'data/coding_pilot_v1/visible/development.json').read_text());completed=[]
                for p in sorted(parent.glob('tasks/*/complete.json')):
                    row=json.loads(p.read_text());verify_completed(p.parent,ident,row['task_id']);completed.append(row['task_id'])
                if set(completed)!={t['task_id'] for t in tasks[:5]}:raise ValueError('Expected exactly the first five completed parent tasks')
                old_archive=ROOT/'evidence/coding_pilot_v1/pod_backup/parent_before_extension_08.tar.gz'
                shutil.copy2(old_archive.parent/'latest.tar.gz',old_archive)
                write(C/'extension_parent_backup.json',{'epoch':time.time(),'sha256':sha(old_archive),'completed_tasks':5})
                preserve='import pathlib,shutil; e=pathlib.Path('+repr(REMOTE+'/evidence/coding_pilot_v1')+'); d=e/"extension_parent_07"; assert not d.exists(); paths=list(e.iterdir()); d.mkdir(); [shutil.copytree(p,d/p.name) if p.is_dir() else shutil.copy2(p,d/p.name) for p in paths if p.name!="control"]'
                remote(conn,'python3 -c '+shlex.quote(preserve))
                allocation=json.loads((ROOT/'evidence/coding_pilot_v1/allocation.json').read_text())
                allocation.update(preflight_deadline_epoch=deadline,preflight_cap_usd=150,preflight_gpu_hours=10,
                    approval_sha256=sha(C/'preflight_extension_approved.json'),prior_usage=decision(json.loads((C/'ledger.json').read_text())),extension_session=8)
                write(ROOT/'evidence/coding_pilot_v1/allocation_extension.json',allocation)
                shutil.copy2(C/'preflight_extension_approved.json',ROOT/'evidence/coding_pilot_v1/preflight_extension_approved.json')
                files=[ROOT/f for f in ['gearshift/coding_control.py','gearshift/coding_reuse.py','scripts/coding_preflight_extended.py',
                    'evidence/coding_pilot_v1/allocation_extension.json','evidence/coding_pilot_v1/preflight_extension_approved.json']]
                upload(conn,files,'extension_upload.tar')
                remote(conn,'rm -f '+REMOTE+'/evidence/coding_pilot_v1/STOP '+REMOTE+'/evidence/coding_pilot_v1/ALERT.json')
                (C/'worker_ALERT.json').unlink(missing_ok=True);(C/'backup_verified.json').unlink(missing_ok=True)
                # Preserved parent remains immutable. New worker reuses all five
                # completed task files verbatim and generates only remaining tasks.
                launcher="import subprocess,pathlib,os; p=pathlib.Path("+repr(REMOTE)+"); env=dict(os.environ,HF_HOME='/workspace/hf',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8'); f=(p/'evidence/coding_pilot_v1/extension.log').open('ab'); child=subprocess.Popen([str(p/'.pilot-venv/bin/python'),'scripts/coding_preflight_extended.py'],cwd=p,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(child.pid)"
                pid=int(remote(conn,'python3 -c '+shlex.quote(launcher),30).strip());launched=True;namespace='preflight_v4'
                write(C/'bootstrap_process.json',{'pid':pid,'pod_id':POD,'epoch':time.time(),'script':'coding_preflight_extended.py'})
                write(C/'session_status.json',{'state':'extended_preflight_running','pod_id':POD,'namespace':namespace,'epoch':time.time(),'deadline_epoch':deadline})
                time.sleep(5);continue
            if terminal and launched:
                # New process may not yet have replaced the previous terminal status.
                if (saved/f'results/coding_pilot_v1/{namespace}/identity.json').exists() or not alive(conn):break
            time.sleep(30)
        remote(conn,'touch '+REMOTE+'/evidence/coding_pilot_v1/STOP',30)
        for _ in range(12):
            if not alive(conn):break
            time.sleep(5)
        else:raise RuntimeError('Worker did not drain within one minute')
        for attempt in range(3):
            try:saved=collect(conn);status_from(saved,namespace);break
            except Exception:
                if attempt==2:raise
                time.sleep(5)
        latest=ROOT/'evidence/coding_pilot_v1/pod_backup/latest.tar.gz';second=latest.with_name('final_extension_08.tar.gz');shutil.copy2(latest,second)
        if sha(latest)!=sha(second):raise RuntimeError('Final backup copies differ')
        write(C/'backup_verified.json',{'epoch':time.time(),'pod_id':POD,'volume_id':VOLUME,'worker_confirmed_absent':True,
            'copies':[str(latest.relative_to(ROOT)),str(second.relative_to(ROOT))],'sha256':sha(latest)})
        final_verified=True
    except BaseException as exc:
        write(C/'session_error.json',{'epoch':time.time(),'error':str(exc)});traceback.print_exc()
        try:remote(conn,'touch '+REMOTE+'/evidence/coding_pilot_v1/STOP',30);collect(conn)
        except Exception:pass
    finally:
        (C/'STOP').touch()
        try:cli('pod','delete',POD)
        except Exception as exc:write(C/'cleanup_pending.json',{'epoch':time.time(),'error':str(exc)})
        if final_verified:
            try:cli('network-volume','delete',VOLUME)
            except Exception as exc:write(C/'volume_cleanup_pending.json',{'epoch':time.time(),'error':str(exc)})
        running=False;write(C/'session_status.json',{'state':'stopped_backed_up' if final_verified else 'stopped_inspect_backup','namespace':namespace,'epoch':time.time()});tick()

if __name__=='__main__':main()
