#!/usr/bin/env python3
"""One detached, leased sparse-repair job; no scheduler and no automatic retries.

Plan: experiment_id, job_id, lease_path, lease_sha256, argv, output_folders.
output_folders are owned, non-overlapping paths relative to allowed_result_root.
Calibration argv must --output exactly that one folder. Screen argv must declare
--task-id for each owned task plus --worker-id; no unscoped full-screen dispatch.
After the one child stops, only its declared outputs and wrapper evidence are
hashed and copied twice on the retained public volume. The existing lease guard,
not this wrapper, consumes gpu_release_verified.json and deletes the leased pod.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gearshift.coding_confirmation_lease import (atomic_json, control_root, linux_process_identity,
    load_lease, sha256_file, verify_release_receipt)
from scripts.coding_confirmation_generation_supervisor import validate_guard
from scripts.coding_confirmation_replication_supervisor import child_environment, copy_verified, file_receipt, verify_archive

PYTHON='/workspace/GearshiftConfirmationPrimary/.pilot-venv/bin/python'
SCRIPTS=('scripts/sparse_repair_calibrate.py','scripts/sparse_repair_screen.py')
ALLOWED_SUFFIXES={'.json','.jsonl','.md','.txt','.log','.csv','.tsv'}


def read(path):return json.loads(Path(path).read_text())


def scoped(root,relative):
    root=Path(root).resolve(); p=Path(relative)
    if p.is_absolute() or not p.parts or '..' in p.parts or 'private' in p.parts:
        raise ValueError('Expected nonprivate scoped relative path')
    result=root/p
    for cursor in (result,*result.parents):
        if cursor==root:break
        if cursor.is_symlink():raise ValueError('Linked job evidence forbidden')
    if not result.resolve().is_relative_to(root):raise ValueError('Job path escapes root')
    return result


def option_values(argv,name):
    return [argv[i+1] for i,item in enumerate(argv[:-1]) if item==name]


def validate_plan(plan,repo=ROOT):
    if plan.get('experiment_id')!='sparse_repair_01' or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',plan.get('job_id','')):
        raise ValueError('Wrong experiment or unsafe job ID')
    lease=load_lease(scoped(repo,plan['lease_path']),plan['lease_sha256'])
    if lease['experiment_id']!=plan['experiment_id'] or lease['gpu_count']!=1:
        raise ValueError('Exactly one matching leased GPU required')
    root=Path(lease['allowed_result_root']).resolve()
    if not root.is_relative_to(Path(repo).resolve()):raise ValueError('Result root not in this checkout')
    argv=plan['argv']
    if not isinstance(argv,list) or not all(isinstance(x,str) and '\0' not in x for x in argv) or len(argv)<3 or argv[0]!=PYTHON or argv[1] not in SCRIPTS:
        raise ValueError('Only exact Python argv for bounded calibration/screen is allowed')
    folders=[scoped(root,p) for p in plan['output_folders']]
    if not folders or len(set(folders))!=len(folders) or any(p==root or p==control_root(lease) or p.is_relative_to(control_root(lease)) for p in folders):
        raise ValueError('Outputs must be nonempty strictly owned scientific folders')
    if any(a!=b and (a.is_relative_to(b) or b.is_relative_to(a)) for a in folders for b in folders):
        raise ValueError('Nested duplicate output scopes')
    if argv[1]==SCRIPTS[0]:
        output=option_values(argv,'--output')
        if len(output)!=1 or len(folders)!=1 or (Path(repo)/output[0]).resolve()!=folders[0]:
            raise ValueError('Calibration output scope differs from exact argv')
    else:
        task_ids=option_values(argv,'--task-id'); worker=option_values(argv,'--worker-id')
        if not task_ids or len(set(task_ids))!=len(task_ids) or len(worker)!=1 or not re.fullmatch(r'[A-Za-z0-9_-]+',worker[0]):
            raise ValueError('Screen wrapper requires exact owned task IDs and worker ID')
        expected={root/'screen/tasks'/t.replace('/','__') for t in task_ids}|{root/'workers'/worker[0]}
        if set(folders)!=expected:raise ValueError('Screen output scopes must match only its tasks and worker')
        child_plan=option_values(argv,'--plan')
        if len(child_plan)!=1:raise ValueError('Screen requires one runtime plan')
        runtime=read(scoped(repo,child_plan[0]))
        if runtime['lease_path']!=plan['lease_path'] or runtime['lease_sha256']!=plan['lease_sha256']:
            raise ValueError('Screen child plan lease differs')
    if plan.get('automatic_retries',0)!=0:raise ValueError('Automatic retries forbidden; use separate explicit infrastructure resume plan')
    return lease,root,folders


def public_files(root,folders):
    files=[]
    for folder in folders:
        if not folder.exists():continue
        for path in sorted(folder.rglob('*')):
            if path.is_symlink():raise ValueError('Symlink in output')
            if not path.is_file():continue
            relative=path.relative_to(root)
            if path.name.endswith(('.lock','.tmp')):continue
            if 'private' in relative.parts or any(x.startswith('.') for x in relative.parts):
                raise ValueError('Private/hidden output in compact backup')
            if path.suffix not in ALLOWED_SUFFIXES or path.stat().st_size>128*1024**2:
                raise ValueError('Noncompact/unknown output rejected: '+str(relative))
            files.append(path)
    if sum(p.stat().st_size for p in files)>2*1024**3:raise ValueError('Compact backup exceeds2GiB')
    return sorted(set(files))


def compact_release(root,lease,folders,wrapper,outcome):
    """All writers are stopped; only owned scopes may enter archive/receipt."""
    control=control_root(lease)
    rows=[file_receipt(root,p) for p in public_files(root,[*folders,wrapper])]
    if not rows:raise ValueError('No own evidence to preserve')
    backup=control/'release_backups'/('sparse_'+str(time.time_ns()));backup.mkdir(parents=True)
    first,second=backup/'compact_a.tar.gz',backup/'compact_b.tar.gz'
    with tarfile.open(first,'w:gz',compresslevel=1) as archive:
        for row in rows:archive.add(root/row['path'],arcname=row['path'],recursive=False)
        data=json.dumps(rows,sort_keys=True).encode();info=tarfile.TarInfo('COMPACT_MANIFEST.json');info.size=len(data);archive.addfile(info,io.BytesIO(data))
    with first.open('rb') as f:os.fsync(f.fileno())
    verify_archive(first,rows)
    copy_verified(first,second,file_receipt(root,first));verify_archive(second,rows)
    rows += [file_receipt(root,first),file_receipt(root,second)]
    manifest=control/'release_manifest.json'
    atomic_json(manifest,{'experiment_id':lease['experiment_id'],'pod_id':lease['pod_id'],'outcome':outcome,
        'files':rows,'network_volume_deletion_authorized':False,
        'scope':'Only this stopped job declared output folders and wrapper evidence.',
        'backup_limitation':'Two verified copies on one retained public volume, not geographically independent backup.'})
    atomic_json(control/'gpu_release_verified.json',{'experiment_id':lease['experiment_id'],'pod_id':lease['pod_id'],
        'outcome':outcome,'gpu_release_verified':True,'all_gpu_workers_stopped':True,'durable_backup_verified':True,
        'manifest_path':str(manifest.relative_to(root)),'manifest_sha256':sha256_file(manifest)})
    if not verify_release_receipt(lease):raise ValueError('Existing lease release verifier rejected backup')
    return manifest


def group_members(group):
    members=[]
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        try:
            identity=linux_process_identity(int(folder.name))
            if identity['pgid']==group:members.append(identity)
        except (FileNotFoundError,ProcessLookupError,ValueError):pass
    return members


def run(plan_path,expected_sha):
    plan_path=scoped(ROOT,plan_path)
    if sha256_file(plan_path)!=expected_sha:raise ValueError('Exact job plan hash differs')
    plan=read(plan_path);lease,root,folders=validate_plan(plan)
    if os.getpid()!=os.getpgrp():raise ValueError('Wrapper must launch detached in own process group')
    control=control_root(lease);control.mkdir(parents=True,exist_ok=True)
    with ExitStack() as locks:
        for path in [control/'sparse_job.lock',*[root/'claims'/('sparse_output_'+hashlib.sha256(str(p).encode()).hexdigest()+'.lock') for p in folders]]:
            path.parent.mkdir(parents=True,exist_ok=True);lock=locks.enter_context(path.open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if verify_release_receipt(lease):return 0
        validate_guard(lease,control)
        registration=control/'lease_guard/supervisor_registration.json'
        if registration.exists():
            old=read(registration)
            try:current=linux_process_identity(old['pid'])
            except FileNotFoundError:current=None
            if current is not None and current['pid']!=os.getpid() and all(current.get(k)==old.get(k) for k in ('pid','pgid','start_ticks','boot_id')):
                raise RuntimeError('Existing live orchestration owner; refusing takeover')
        wrapper=control/'sparse_job'/plan['job_id']
        if wrapper.exists():raise ValueError('Job already attempted; explicit new infrastructure resume plan required')
        wrapper.mkdir(parents=True)
        atomic_json(wrapper/'plan.json',plan)
        atomic_json(control/'lease_guard/supervisor_registration.json',{
            **linux_process_identity(os.getpid()),'experiment_id':lease['experiment_id'],'pod_id':lease['pod_id']})
        stopped=False
        def request_stop(signum,frame):
            nonlocal stopped
            stopped=True
        signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
        child=None;code=None;outcome='not_started';ownership=True
        try:
            with (wrapper/'worker.log').open('ab') as log:
                child=subprocess.Popen(plan['argv'],cwd=ROOT,env=child_environment(lease,0),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=log,start_new_session=False)
                try:
                    identity=linux_process_identity(child.pid)
                    if identity['pgid']!=os.getpgrp():ownership=False;raise RuntimeError('Child escaped wrapper group')
                except FileNotFoundError:
                    if child.poll() is None:ownership=False;raise
                    identity={'pid':child.pid,'already_exited':True}
                atomic_json(wrapper/'started.json',{'epoch':time.time(),'plan_sha256':expected_sha,'child':identity})
                while child.poll() is None:
                    if stopped or time.time()>=lease['deadline_epoch']-120 or (control/'lease_guard/STOP').exists():
                        outcome='stopped_preserved';child.terminate()
                        try:child.wait(timeout=90)
                        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
                        break
                    time.sleep(1)
                code=child.wait();outcome=outcome if outcome=='stopped_preserved' else ('complete' if code==0 else 'failed_preserved')
        except BaseException as exc:
            outcome='failed_preserved'
            atomic_json(wrapper/'failure.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=90)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
            remaining=[x for x in group_members(os.getpgrp()) if x['pid']!=os.getpid()]
            ownership=ownership and not remaining
            atomic_json(wrapper/'exited.json',{'epoch':time.time(),'outcome':outcome,'returncode':code,
                'automatic_retry':False,'all_registered_workers_stopped':ownership,'remaining_owned_group_members':remaining})
            if ownership:compact_release(root,lease,folders,wrapper,outcome)
            else:atomic_json(wrapper/'release_withheld.json',{'reason':'Cannot prove all own group workers stopped; immutable guard remains authoritative'})
        return 0 if outcome=='complete' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True)
    a=p.parse_args();sys.exit(run(a.plan,a.plan_sha256))
