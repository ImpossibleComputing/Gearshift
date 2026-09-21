#!/usr/bin/env python3
"""Read-only health checks, with bounded recovery after a verified host reboot."""
import datetime as dt
import fcntl
import json
import os
import plistlib
from pathlib import Path
import re
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read,write,artifact
from gearshift.phase2_completion_state import transient_usage_check_failure
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'evidence/phase2/studio_transfer'
RUNTIME=STATE/'finalization_runtime'

def recovery_action(job,boot_epoch,alive):
    if job['state']=='complete':return 'ensure_finalizer'
    if job['state']=='running' and alive:return 'healthy'
    if job['state']=='running' and job['started_epoch']<boot_epoch:return 'resume_after_reboot'
    return 'attention_required'

def matches(pid,fragment):
    if not pid:return False
    p=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    return fragment in p.stdout

def launch_job(name):
    folder=STATE/'jobs'/name;folder.mkdir()
    write(folder/'spec.json',dict(name=name,commands=[['scripts/phase2_judge_queue.py']]))
    with (folder/'wrapper.txt').open('w') as log:
        p=subprocess.Popen([sys.executable,'scripts/phase2_studio_job.py',str((folder/'spec.json').resolve())],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(20):
        if (folder/'status.json').exists():break
        time.sleep(.1)
    return p.pid

def run_once():
    from gearshift.phase2_pause import is_paused
    if is_paused(ROOT):return
    os.chdir(ROOT);RUNTIME.mkdir(parents=True,exist_ok=True)
    lock=(RUNTIME/'recovery.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    if (STATE/'CONTROL_DONE.json').exists():
        label='com.gearshift.phase2.review-recovery'
        config_path=Path.home()/'Library/LaunchAgents'/f'{label}.plist'
        if config_path.exists():
            config=plistlib.loads(config_path.read_bytes())
            if config.get('ProgramArguments') not in [[str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve())],[str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--watch']]:
                raise ValueError('Refusing to remove a different recovery agent')
            config_path.unlink()
        write(RUNTIME/'recovery_retirement.json',dict(state='retirement_requested_after_verified_completion',label=label))
        subprocess.run(['/bin/launchctl','bootout',f'gui/{os.getuid()}/'+label],capture_output=True)
        return
    boot_text=subprocess.check_output(['/usr/sbin/sysctl','kern.boottime'],text=True)
    boot=int(re.search(r'sec = (\d+)',boot_text).group(1))
    active=read(STATE/'active_judge_job.json');name=active['job'];path=STATE/'jobs'/name/'status.json';job=read(path)
    action=recovery_action(job,boot,matches(job.get('pid'),f'scripts/phase2_studio_job.py {str((path.parent/"spec.json").resolve())}'))
    status=dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),boot_epoch=boot,judge_job=name,action=action,watcher_pid=os.getpid(),watch_mode='--watch' in sys.argv)
    if action=='ensure_finalizer':
        queue=read(ROOT/'results/phase2_v1/judging_queue_status.json')
        if transient_usage_check_failure(queue):
            checked=RUNTIME/'last_service_recheck.json'
            if checked.exists() and time.time()-read(checked)['epoch']<300:
                status['action']='waiting_for_service_recheck';write(RUNTIME/'recovery_status.json',status);return
            os.environ.update(read(ROOT/'configs/phase2_studio_runtime.json')['environment'])
            sys.path.insert(0,str(ROOT/'scripts'))
            from phase2_usage_guard import included_usage
            try:usage=included_usage()
            except Exception as exc:usage=dict(allowed=False,error=repr(exc))
            write(checked,dict(epoch=time.time(),usage=usage))
            if usage.get('error'):
                status['action']='waiting_for_service_recheck';write(RUNTIME/'recovery_status.json',status);return
            record=STATE/'service_recoveries'/str(time.time_ns());record.mkdir(parents=True)
            write(record/'prior_queue_status.json',queue);write(record/'fresh_usage.json',usage)
            if usage.get('allowed'):
                previous=name;name='final_judge_queue_service_'+str(time.time_ns())
                new_pid=launch_job(name)
                write(STATE/'active_judge_job.json',dict(job=name,previous_job=previous,recovery_record=str(record.relative_to(ROOT)),
                    reason='Failed read-only allowance check recovered; fresh ordinary included permission is affirmative. Saved valid/invalid results remain reused.'))
                status.update(new_judge_pid=new_pid,new_job=name);action='healthy'
            else:
                queue['status']['usage_guard_stop']=usage
                queue['service_recheck_receipt']=str((record/'fresh_usage.json').relative_to(ROOT))
                write(ROOT/'results/phase2_v1/judging_queue_status.json',queue)
    if action=='resume_after_reboot':
        record=STATE/'reboot_recoveries'/str(boot);record.mkdir(parents=True,exist_ok=True)
        write(record/'prior_job_status.json',job)
        completed={str(p.relative_to(ROOT)):artifact(p) for p in (ROOT/'results/phase2_v1').glob('*/judging/judgments/*/result.json')}
        write(record/'completed_judgments_before_restart.json',completed)
        write(path,{**job,'state':'interrupted_by_studio_reboot','recovery_record':str(record.relative_to(ROOT))})
        previous=name;name='final_judge_queue_reboot_'+str(time.time_ns())
        new_pid=launch_job(name)
        write(STATE/'active_judge_job.json',dict(job=name,previous_job=previous,recovery_record=str(record.relative_to(ROOT))))
        status.update(new_judge_pid=new_pid,new_job=name,preserved_judgments=len(completed));action='healthy'
    if action=='attention_required':
        write(RUNTIME/'recovery_status.json',status);return
    final_path=RUNTIME/'status.json';final=read(final_path) if final_path.exists() else {}
    final_alive=matches(final.get('pid'),'scripts/phase2_finish_review.py')
    if final_alive and final.get('judge_job')!=name and final.get('state')!='waiting_for_service_recovery':
        status.update(action='attention_required',reason='A finalizer for another job is alive; do not duplicate')
        write(RUNTIME/'recovery_status.json',status);return
    if not final_alive:
        if final.get('state')=='attention_required':
            status.update(action='attention_required',reason='Finalization failed; preserve diagnostics for repair')
            write(RUNTIME/'recovery_status.json',status);return
        if final:write(RUNTIME/f'prior_finalizer_{time.time_ns()}.json',final)
        with (RUNTIME/'launcher.txt').open('a') as log:
            p=subprocess.Popen([sys.executable,'scripts/phase2_finish_review.py','--job',name],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owner=read(STATE/'active_finalization_owner.json');owner.update(finalizer_pid=p.pid,judge_job=name);write(STATE/'active_finalization_owner.json',owner)
        status['new_finalizer_pid']=p.pid
    write(RUNTIME/'recovery_status.json',status)

def main():
    from gearshift.phase2_pause import is_paused
    if is_paused(ROOT):return
    if '--watch' not in sys.argv:run_once();return
    subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    while True:
        if is_paused(ROOT):return
        try:run_once()
        except Exception as exc:
            write(RUNTIME/'recovery_status.json',dict(action='attention_required',error=repr(exc),checked_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        if (STATE/'CONTROL_DONE.json').exists():return
        time.sleep(60)

if __name__=='__main__':main()
