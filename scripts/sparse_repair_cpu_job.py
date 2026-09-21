#!/usr/bin/env python3
"""One detached, leased CPU scoring job; no provider calls or outcome retries.

Requires an already sealed 504-answer public screen and a separately mounted
private development file. Reuses frozen scoring, trusted-reference preflight,
lease guard, orphan cleanup and compact-release helpers. Never backs up private
inputs. The guard, not this process, deletes only its own CPU allocation.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gearshift.coding_confirmation_lease import (atomic_json,control_root,linux_process_identity,
    load_lease,sha256_file,verify_release_receipt)
from gearshift import sparse_repair_storage as storage
from scripts import sparse_repair_score as scorer
from scripts import sparse_repair_job as jobs

PYTHON='/workspace/GearshiftConfirmation/.scoring-venv/bin/python'
PRIVATE_RELATIVE='private/development.json'
IMPLEMENTATION=(
    'scripts/sparse_repair_cpu_job.py','scripts/sparse_repair_score.py','scripts/sparse_repair_job.py',
    'scripts/coding_confirmation_score.py','scripts/coding_scorer_repair_calibrate.py',
    'scripts/coding_confirmation_generation_supervisor.py','scripts/coding_confirmation_replication_supervisor.py',
    'gearshift/coding_confirmation_lease.py','gearshift/coding_control.py','gearshift/sparse_repair_storage.py',
    'gearshift/coding_sandbox.py','gearshift/coding_sandbox_v2.py','scripts/coding_sandbox_child_v2.py')


def implementation_hashes(repo=ROOT):
    return {p:sha256_file(jobs.scoped(repo,p)) for p in IMPLEMENTATION}


def read(path):return json.loads(Path(path).read_text())


def validate_plan(plan,repo=ROOT):
    repo=Path(repo).resolve()
    if plan.get('experiment_id')!='sparse_repair_01' or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',plan.get('job_id','')):
        raise ValueError('Wrong experiment or unsafe CPU job ID')
    if plan.get('automatic_retries',0)!=0:
        raise ValueError('Automatic retries forbidden; only explicit infrastructure recovery may make another job')
    lease=load_lease(jobs.scoped(repo,plan['lease_path']),plan['lease_sha256'])
    storage.validate_cpu_lease(repo,lease,plan)
    root=jobs.scoped(repo,plan['result_root'])
    if root.resolve()!=Path(lease['allowed_result_root']).resolve():
        raise ValueError('CPU job result root differs from immutable allocation')
    cpus=plan['cpu_ids']
    if not isinstance(cpus,list) or len(cpus)!=8 or len(set(cpus))!=8 or any(type(c) is not int or c<0 for c in cpus):
        raise ValueError('Exactly eight distinct physical CPU IDs must be declared')
    if plan.get('private_tests',PRIVATE_RELATIVE)!=PRIVATE_RELATIVE:
        raise ValueError('Only the existing isolated private development path is allowed')
    if plan.get('implementation_hashes')!=implementation_hashes(repo) or implementation_hashes(repo)!=implementation_hashes(ROOT):
        raise ValueError('CPU job loaded implementation differs from its frozen plan')
    if sha256_file(jobs.scoped(repo,plan['declaration_path']))!=plan['declaration_sha256']:
        raise ValueError('CPU job frozen declaration differs')
    return lease,root


def command(stage,plan,repo,wrapper):
    """Fixed argv only: no shell, caller-supplied code, extra scope or retry flags."""
    repo=Path(repo);preflight=str((wrapper/'preflight.json').relative_to(repo))
    common=[PYTHON,'scripts/sparse_repair_score.py',stage,'--repo',str(repo)]
    declaration=['--declaration',plan['declaration_path'],'--declaration-sha256',plan['declaration_sha256']]
    lease=['--lease',plan['lease_path'],'--lease-sha256',plan['lease_sha256']]
    cpus=['--cpu-ids',','.join(map(str,plan['cpu_ids']))]
    private=['--private-tests',str(repo/PRIVATE_RELATIVE)]
    score_plan=str(Path(plan['result_root'])/scorer.SCORING/'scoring_plan.json')
    if stage=='preflight':return common+declaration+lease+cpus+['--output',preflight]
    if stage=='prepare':return common+declaration+['--result-root',plan['result_root']]+private
    if stage=='run':
        return common+['--plan',score_plan]+private+lease+cpus+['--preflight',preflight,
            '--preflight-sha256',sha256_file(wrapper/'preflight.json')]
    if stage=='finalize':return common+['--plan',score_plan]
    raise ValueError('Undeclared CPU job stage')


def existing_owner_is_live(control):
    path=Path(control)/'lease_guard/supervisor_registration.json'
    if not path.exists():return False
    old=read(path)
    try:current=linux_process_identity(old['pid'])
    except FileNotFoundError:return False
    return current['pid']!=os.getpid() and all(current.get(k)==old.get(k) for k in ('pid','pgid','start_ticks','boot_id'))


def candidate_processes(repo=ROOT,proc_root=Path('/proc')):
    """Identify only the fixed sandbox launcher; never signal processes by name."""
    prefix=[b'/usr/bin/python3',b'-I',str(Path(repo)/'scripts/coding_sandbox_child_v2.py').encode()]
    found=[]
    for folder in Path(proc_root).iterdir():
        if not folder.name.isdigit():continue
        try:args=(folder/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if len(args)==5 and args[:3]==prefix and args[3].endswith(b'/payload.json') and args[-1]==b'':
            found.append(int(folder.name))
    return sorted(found)


def stopped_evidence(lease,repo=ROOT):
    """No release while a worker or even a confined launcher could still write."""
    removed=scorer.base.cleanup_orphans(lease)
    # Existing cleanup targets only verified PPid=1 confined children on this pod;
    # briefly wait for those asynchronous SIGKILLs to be reflected in /proc.
    deadline=time.monotonic()+5
    while True:
        group=[p for p in jobs.group_members(os.getpgrp()) if p['pid']!=os.getpid()]
        candidates=candidate_processes(repo)
        if not group and not candidates:break
        if time.monotonic()>=deadline:break
        time.sleep(.1)
    return {'all_own_cpu_workers_stopped':not group and not candidates,
            'remaining_process_group_members':group,'remaining_sandbox_launcher_pids':candidates,
            'verified_orphan_cleanup_pids':removed}


def preserve_after_stop(root,lease,scoring,wrapper,outcome,code):
    """Hold the scorer's own exclusion lock through backup and release receipt.

    A separately launched frozen scorer does not use supervisor.lock. Its
    execution.lock is the authoritative protection against writers between
    candidates, where a process-name scan alone cannot establish quiescence.
    """
    with (scoring/'execution.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            atomic_json(wrapper/'exited.json',{'outcome':outcome,'returncode':code,'epoch':time.time(),
                'automatic_retry':False,'all_own_cpu_workers_stopped':False,'execution_lock_busy':True})
            atomic_json(wrapper/'release_withheld.json',{
                'reason':'Scorer execution lock is owned elsewhere; no backup or release is safe.'})
            return False
        proof=stopped_evidence(lease)
        atomic_json(wrapper/'exited.json',{'outcome':outcome,'returncode':code,'epoch':time.time(),
            'automatic_retry':False,**proof})
        if not proof['all_own_cpu_workers_stopped']:
            atomic_json(wrapper/'release_withheld.json',{
                'reason':'Cannot prove all own workers and sandbox launchers stopped; existing immutable guard remains authoritative.'})
            return False
        # Legacy receipt field names say GPU; its verifier also supports zero
        # GPUs and still binds only this immutable allocation.
        jobs.compact_release(root,lease,[scoring],wrapper,outcome)
        return True


def run(plan_path,expected_sha):
    path=jobs.scoped(ROOT,plan_path)
    if sha256_file(path)!=expected_sha:raise ValueError('CPU job plan hash differs')
    plan=read(path);lease,root=validate_plan(plan)
    if sys.platform!='linux' or os.geteuid()!=0 or os.getpid()!=os.getpgrp():
        raise ValueError('CPU job must run as Linux root, detached in its own process group')
    control=control_root(lease);control.mkdir(parents=True,exist_ok=True)
    scoring=jobs.scoped(root,scorer.SCORING);scoring.mkdir(parents=True,exist_ok=True)
    with ExitStack() as locks:
        # Not execution.lock: the frozen child preflight/scorer owns that lock.
        for lockpath in (control/'sparse_cpu_job.lock',scoring/'supervisor.lock'):
            handle=locks.enter_context(lockpath.open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if verify_release_receipt(lease):return 0
        jobs.validate_guard(lease,control)
        if existing_owner_is_live(control):raise RuntimeError('Another live owner is registered; no takeover')
        # Refuse a direct/legacy scorer before registration/setup. Release this
        # probe immediately: our frozen stage children acquire the same lock.
        with (scoring/'execution.lock').open('a') as execution:
            fcntl.flock(execution,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['RUNPOD_POD_ID']=lease['pod_id']
        wrapper=control/'sparse_cpu_job'/plan['job_id']
        if wrapper.exists():raise ValueError('CPU job already attempted; no automatic replay of orchestration')
        wrapper.mkdir(parents=True)
        atomic_json(wrapper/'plan.json',plan)
        atomic_json(control/'lease_guard/supervisor_registration.json',{
            **linux_process_identity(os.getpid()),'experiment_id':lease['experiment_id'],'pod_id':lease['pod_id']})
        stopped=False
        def request_stop(signum,frame):
            nonlocal stopped
            stopped=True
        signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
        child=None;outcome='failed_preserved';code=None
        env=jobs.child_environment(lease,0);env['CUDA_VISIBLE_DEVICES']=''
        def guard():
            if stopped or time.time()>=lease['deadline_epoch']-120 or (control/'lease_guard/STOP').exists():
                raise TimeoutError('CPU job stop or immutable allocation deadline')
        try:
            guard()
            # Recompute the entire public seal before setup/preflight/private access.
            c=scorer.public_context(ROOT,plan['declaration_path'],plan['declaration_sha256'],plan['result_root'])
            atomic_json(wrapper/'public_closure_verified.json',{
                'closure_sha256':c['closure']['closure_sha256'],'answers':len(c['closure']['answers']),
                'private_test_values_loaded':False})
            binding=storage.validate_cpu_lease(ROOT,lease,plan)
            if binding and sha256_file('/usr/bin/python3')!=binding['candidate_python_sha256']:
                raise ValueError('Fallback candidate Python differs from the frozen interpreter identity')
            scorer.check_cpus(plan['cpu_ids'],c['policy']);guard()
            ctypes.CDLL('libseccomp.so.2')
            if not Path('/opt/gearshift-sandbox/READY').is_file():scorer.base.calibration.setup_sandbox()
            # Private bytes remain untouched here; prepare will verify their exact
            # frozen hash after recomputing the public seal again.
            private=ROOT/PRIVATE_RELATIVE
            if private.is_symlink() or private.parent.is_symlink():raise ValueError('Private input must not be linked')
            stat=private.stat()
            if stat.st_uid!=os.getuid() or stat.st_mode&0o077 or private.parent.stat().st_mode&0o077:
                raise ValueError('Private input must be owned by this root user and owner-only')
            for stage in ('preflight','prepare','run','finalize'):
                guard();argv=command(stage,plan,ROOT,wrapper)
                atomic_json(wrapper/(stage+'_launch.json'),{'argv':argv,'epoch':time.time(),'automatic_retry':False})
                with (wrapper/(stage+'.log')).open('ab') as log:
                    child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=False)
                    while child.poll() is None:
                        guard();time.sleep(1)
                    code=child.wait()
                atomic_json(wrapper/(stage+'_exit.json'),{'returncode':code,'epoch':time.time()})
                if code!=0:raise RuntimeError('CPU scoring stage failed: '+stage+'; no automatic retry')
                child=None
            outcome='complete'
        except BaseException as exc:
            outcome='stopped_preserved' if isinstance(exc,TimeoutError) else 'failed_preserved'
            # No traceback/local variables or private values enter public evidence.
            atomic_json(wrapper/'failure.json',{'exception_type':type(exc).__name__,'error':str(exc),
                'automatic_retry':False,'epoch':time.time()})
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=90)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
            preserve_after_stop(root,lease,scoring,wrapper,outcome,code)
        return 0 if outcome=='complete' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True)
    args=p.parse_args();sys.exit(run(args.plan,args.plan_sha256))
