#!/usr/bin/env python3
"""Adopt the existing H200; wait for the complete original cohort, then amend once."""
import fcntl,json,os,shlex,shutil,sys,threading,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import PREFIX,write,sha,decision,remaining_preflight_seconds
from gearshift.coding_cap_amendment import validate_parent
from coding_cloud_guard import cli,tick
from coding_cloud_session import C,ROOT,REMOTE,remote,collect
from coding_recover_session import upload
from coding_resume_session import verified_approval
POD='fx3dcts5rez8gp';VOLUME='jezvugz1r2'
PARENT='preflight_v5';CHILD='preflight_cap_v1'


def verify_adoption(pods,volumes):
    pods=[r for r in pods if r.get('name','').startswith(PREFIX)]
    volumes=[r for r in volumes if r.get('name','').startswith(PREFIX)]
    if len(pods)!=1 or pods[0]['id']!=POD or len(volumes)!=1 or volumes[0]['id']!=VOLUME:
        raise ValueError('Adoption requires exactly the existing pilot resources')
    if pods[0].get('gpuCount')!=1 or not 0<float(pods[0].get('costPerHr') or 0)<=5.5:
        raise ValueError('Allocation count or price differs from approval')


def alive(conn):
    return bool(remote(conn,r"ps -eo args | awk '/python.*scripts\/coding_(bootstrap_resumed|preflight_resumed|preflight_cap)\.py/ && !/awk/ {print}'",30).strip())


def status_from(saved,namespace):
    p=saved/'evidence/coding_pilot_v1/worker_status.json'
    status=json.loads(p.read_text()) if p.exists() else {'state':'starting'}
    write(C/'latest_worker_status.json',status)
    p=saved/f'results/coding_pilot_v1/{namespace}/progress.json'
    if p.exists():write(C/'latest_progress.json',json.loads(p.read_text()))
    p=saved/'evidence/coding_pilot_v1/ALERT.json'
    if p.exists():write(C/'worker_ALERT.json',json.loads(p.read_text()))
    return status


def transition_ready(status,parent,root):
    if status.get('state')!='preflight_baselines_finished':
        raise ValueError('Original worker has not finished every baseline task')
    return validate_parent(parent,root)


def main():
    os.chdir(ROOT)
    with (C/'recovery_controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()


def run():
    verified_approval(C)
    if (C/'cap_session_11.json').exists():raise ValueError('This one-shot cap controller has already been attempted')
    verify_adoption(cli('pod','list'),cli('network-volume','list'))
    if json.loads((C/'session_status.json').read_text()).get('namespace')!=PARENT:
        raise ValueError('Expected the active original forty-task run')
    declaration=ROOT/'configs/coding_pilot_v1/cap_amendment_v1.json'
    prepared=json.loads((C/'cap_amendment_preparation.json').read_text())
    if sha(declaration)!=prepared['declaration_sha256']:raise ValueError('Declared amendment changed')
    conn=json.loads((C/'connection.json').read_text());remote(conn,'true',30)
    archive=C/'cap_control_parent_10';archive.mkdir(exist_ok=False)
    for p in C.glob('*.json'):shutil.copy2(p,archive/p.name)
    running=True
    def heartbeat():
        while running:
            write(C/'controller_heartbeat.json',{'epoch':time.time(),'pid':os.getpid()});time.sleep(10)
    write(C/'controller_heartbeat.json',{'epoch':time.time(),'pid':os.getpid()});threading.Thread(target=heartbeat,daemon=True).start()
    # Never clear a stop signal while adopting an existing paid allocation.
    if (C/'STOP').exists():raise ValueError('Existing stop marker requires inspection')
    tick();ledger=json.loads((C/'ledger.json').read_text());allowance=remaining_preflight_seconds(ledger)
    started=time.time();deadline=started+allowance
    write(C/'cap_session_11.json',{'epoch':started,'pod_id':POD,'volume_id':VOLUME,'parent_namespace':PARENT,'child_namespace':CHILD,
        'approval_sha256':sha(C/'experiment_extension_approved.json'),'declaration_sha256':sha(declaration),
        'deadline_epoch':deadline,'remaining_seconds':allowance,'adopted_existing_pod':True,'prior_usage':decision(ledger,started)})
    final_verified=False;last_backup=time.time();namespace=PARENT;launched=False
    try:
        write(C/'session_status.json',{'state':'waiting_for_original_full_cohort','pod_id':POD,'namespace':PARENT,'epoch':time.time(),'deadline_epoch':deadline})
        while time.time()<deadline-180 and not (C/'STOP').exists():
            try:saved=collect(conn);last_backup=time.time();status=status_from(saved,namespace)
            except Exception as exc:
                write(C/'backup_retry.json',{'epoch':time.time(),'error':str(exc),'last_verified_epoch':last_backup})
                if time.time()-last_backup>600:raise RuntimeError('Ten minutes without a verified backup')
                time.sleep(15);continue
            terminal=status['state'] in ['stopped','preflight_baselines_finished','bootstrap_failed']
            if terminal and not launched:
                if status['state']!='preflight_baselines_finished':raise RuntimeError('Original run stopped unexpectedly: '+str(status))
                if alive(conn):time.sleep(2);continue
                # Take a fresh stable snapshot after process exit; a live snapshot
                # may observe the terminal status just after enumerating task files.
                saved=collect(conn);last_backup=time.time();status=status_from(saved,namespace)
                parent=saved/f'results/coding_pilot_v1/{PARENT}'
                checkpoint=transition_ready(status,parent,ROOT)
                # Preserve a complete original archive before changing any pod files.
                latest=ROOT/'evidence/coding_pilot_v1/pod_backup/latest.tar.gz'
                original=latest.with_name('original_40_before_cap_v1.tar.gz')
                if original.exists():raise ValueError('Original forty-task archive already exists')
                shutil.copy2(latest,original)
                if sha(latest)!=sha(original):raise ValueError('Parent backup copies differ')
                write(C/'cap_parent_backup.json',{'epoch':time.time(),'sha256':sha(original),'completed_tasks':40,
                    'checkpoint':{k:v for k,v in checkpoint.items() if k!='parent_identity'}})
                tick();current=json.loads((C/'ledger.json').read_text());remaining=remaining_preflight_seconds(current)
                progress=json.loads((parent/'progress.json').read_text())
                estimated=progress['observed_baseline_seconds_including_parent']/40*1.5*len(checkpoint['repeat_task_ids'])
                if remaining<estimated+900:
                    write(C/'worker_ALERT.json',{'epoch':time.time(),'reason':'Cap amendment forecast exceeds remaining experiment allowance',
                        'forecast_seconds':estimated,'remaining_seconds':remaining,'approval_required_to_extend':True})
                    raise TimeoutError('Insufficient authorized time for complete cap amendment')
                allocation_started=time.time();deadline=min(deadline,allocation_started+remaining)
                preserve='import pathlib,shutil; e=pathlib.Path('+repr(REMOTE+'/evidence/coding_pilot_v1')+'); d=e/"before_cap_v1"; assert not d.exists(); paths=list(e.iterdir()); d.mkdir(); [shutil.copytree(p,d/p.name) if p.is_dir() else shutil.copy2(p,d/p.name) for p in paths if p.name!="control"]'
                remote(conn,'python3 -c '+shlex.quote(preserve))
                allocation=json.loads((ROOT/'evidence/coding_pilot_v1/allocation_resume_09.json').read_text())
                allocation.update(started_epoch=allocation_started,preflight_deadline_epoch=deadline,remaining_authorized_seconds=remaining,
                    prior_usage=decision(current,allocation_started),session='cap_session_11',approval_sha256=sha(C/'experiment_extension_approved.json'),
                    cap_amendment_sha256=sha(declaration),parent_checkpoint={k:v for k,v in checkpoint.items() if k!='parent_identity'},
                    parent_archive_sha256=sha(original))
                write(ROOT/'evidence/coding_pilot_v1/allocation_cap_v1.json',allocation)
                files=[ROOT/f for f in ['gearshift/coding_cap_amendment.py','scripts/coding_preflight_cap.py',
                    'configs/coding_pilot_v1/cap_amendment_v1.json','evidence/coding_pilot_v1/allocation_cap_v1.json']]
                manifest=upload(conn,files,'cap_upload_11.tar')
                check='import hashlib,pathlib; r=pathlib.Path('+repr(REMOTE)+'); e='+repr(manifest['files'])+'; assert all(hashlib.sha256((r/f).read_bytes()).hexdigest()==h for f,h in e.items())'
                remote(conn,'python3 -c '+shlex.quote(check))
                launcher="import subprocess,pathlib,os; p=pathlib.Path("+repr(REMOTE)+"); env=dict(os.environ,HF_HOME='/workspace/hf',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8'); f=(p/'evidence/coding_pilot_v1/bootstrap.log').open('ab'); child=subprocess.Popen([str(p/'.pilot-venv/bin/python'),'scripts/coding_preflight_cap.py'],cwd=p,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(child.pid)"
                pid=int(remote(conn,'python3 -c '+shlex.quote(launcher),30).strip());launched=True;namespace=CHILD
                write(C/'bootstrap_process.json',{'pid':pid,'pod_id':POD,'epoch':time.time(),'script':'coding_preflight_cap.py'})
                write(C/'session_status.json',{'state':'cap_amendment_running','pod_id':POD,'namespace':CHILD,'epoch':time.time(),'deadline_epoch':deadline})
                time.sleep(5);continue
            if terminal and launched:
                if (saved/f'results/coding_pilot_v1/{CHILD}/identity.json').exists() or not alive(conn):break
            time.sleep(30)
        remote(conn,'touch '+REMOTE+'/evidence/coding_pilot_v1/STOP',30)
        for _ in range(12):
            if not alive(conn):break
            time.sleep(5)
        else:raise RuntimeError('Worker did not drain; retain storage')
        for attempt in range(3):
            try:saved=collect(conn);status_from(saved,namespace);break
            except Exception:
                if attempt==2:raise
                time.sleep(5)
        latest=ROOT/'evidence/coding_pilot_v1/pod_backup/latest.tar.gz';second=latest.with_name('final_cap_session_11.tar.gz');shutil.copy2(latest,second)
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
        running=False
        write(C/'session_status.json',{'state':'stopped_backed_up' if final_verified else 'stopped_inspect_backup','namespace':namespace,'epoch':time.time()});tick()

if __name__=='__main__':main()
