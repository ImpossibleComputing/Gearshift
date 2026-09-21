#!/usr/bin/env python3
"""Regional secondary admission only; all frozen numerical functions are unchanged."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gearshift.coding_control import bind,digest,sha,write
from scripts import coding_confirmation_secondary_generate as secondary
from scripts import coding_confirmation_sharded_generate as regional
from scripts.coding_coverage_v2_worker import claim_job as original_claim

ROLE='confirmation_secondary_regional_generation'
PARTITION='configs/coding_pilot_v1/confirmation_01/regional_partition.json'
IMPLEMENTATION={'scripts/coding_confirmation_secondary_sharded_generate_v2.py',
    'scripts/coding_confirmation_secondary_sharded_supervisor_v2.py','scripts/coding_confirmation_secondary_shard_merge_v2.py'}


def read(path):return json.loads(Path(path).read_text())


def source_manifest(repo,output):
    repo=Path(repo).resolve();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    for name in IMPLEMENTATION:
        if subprocess.check_output(['git','show',commit+':'+name],cwd=repo)!=(repo/name).read_bytes():
            raise ValueError('Commit the tested secondary regional wrappers before freezing')
    value={'schema':1,'operational_commit':commit,'files':{name:sha(repo/name) for name in sorted(IMPLEMENTATION)},
        'interposition':'secondary claim_job admission only','numerical_secondary_code_changed':False,
        'scientific_population_or_analysis_changed':False}
    bind(regional.output(repo,output),value);return value


def public_context(repo,plan_path,plan_sha256,*,verify_weights=False):
    c=regional.public_context(repo,plan_path,plan_sha256)
    if c['plan']['partition_path']!=PARTITION:raise ValueError('Secondary must use the frozen regional primary partition')
    # This performs the existing complete final1024/original96/seed20260919 checks.
    c.update(secondary.public_context(repo,plan_path,plan_sha256,verify_weights=verify_weights))
    plan=c['plan'];source=regional.checked(repo,plan['secondary_shard_source_manifest_path'],plan['secondary_shard_source_manifest_sha256'])
    expected={'schema':1,'interposition':'secondary claim_job admission only','numerical_secondary_code_changed':False,
              'scientific_population_or_analysis_changed':False}
    if (any(source.get(k)!=v for k,v in expected.items()) or set(source.get('files',{}))!=IMPLEMENTATION or
            not re.fullmatch('[0-9a-f]{40}',source.get('operational_commit',''))):
        raise ValueError('Secondary regional source binding differs')
    for name,expected_sha in source['files'].items():
        if sha(secondary.primary.scoped_file(repo,name))!=expected_sha or sha(ROOT/name)!=expected_sha:
            raise ValueError('Secondary regional source bytes differ')
    c['secondary_shard_source']=source
    if plan['execution_mode']=='generation':verify_allocation(c)
    return c


def verify_allocation(c,*,verify_local=False):
    plan=c['plan']
    if plan.get('worker_role')!=ROLE or plan.get('exclusive_gpu_allocation') is not True:
        raise ValueError('Dedicated secondary allocation plan is required')
    lease,mount=regional.validate_generation_store(c,verify_local=verify_local)
    provider=regional.checked(c['repo_root'],mount['provider_inspection_path'],mount['provider_inspection_sha256'])
    if provider.get('allocation_role')!=ROLE or provider.get('exclusive_gpu_allocation') is not True:
        raise ValueError('Fresh provider inspection does not identify a dedicated secondary allocation')
    return lease,mount


def idle_gpu():
    import torch
    # NVML reports host PIDs, which need not equal container os.getpid(). Check
    # the empty device BEFORE creating our CUDA context, then require exactly
    # the one newly created context. Never accept a preexisting compute owner.
    if torch.cuda.is_initialized():raise ValueError('GPU admission requires a fresh uninitialized CUDA process')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    index=os.environ.get('CUDA_VISIBLE_DEVICES','')
    visible=gpu[int(index)].strip() if index.isdigit() and int(index)<len(gpu) else index
    if not visible or visible not in [x.strip() for x in gpu]:raise ValueError('Visible GPU ownership is not identifiable')
    def owners():
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
        found=[]
        for line in rows.splitlines():
            fields=[x.strip() for x in line.split(',')]
            if len(fields)!=2 or not fields[0].isdigit() or not fields[1]:raise ValueError('Unparseable compute ownership inventory')
            if fields[1]==visible:found.append(int(fields[0]))
        return found
    before=owners()
    if before:raise ValueError('Assigned secondary GPU has a preexisting compute owner')
    if torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name(0):
        raise ValueError('Exactly one assigned H200 must be visible')
    free,total=torch.cuda.mem_get_info()
    if free<.95*total:raise ValueError('Assigned secondary GPU is occupied; primary GPUs must not be shared')
    after=owners()
    if not torch.cuda.is_initialized() or len(after)!=1:
        raise ValueError('GPU ownership changed beyond the single newly initialized CUDA context')
    return {'device':visible,'free_fraction_before_load':free/total,'other_compute_owners':False,
        'local_pid':os.getpid(),'compute_owner_host_pid':after[0],'preexisting_compute_owners':before,
        'cuda_initialized_before_admission':False,'single_context_transition_verified':True}


def verify_dataset(c):
    expected={'partition_sha256':c['partition_sha256'],'shard_id':c['shard_id'],
        'quiescence_sha256':c['partition']['quiescence_sha256'],'allowed_task_ids':c['allowed_task_ids']}
    if read(c['top']/'sharding/initial_dataset_verified.json')!=expected:
        raise ValueError('Secondary does not reference the established primary shard dataset')
    folder=c['top']/'secondary/sharding';folder.mkdir(parents=True,exist_ok=True)
    identity={**expected,'secondary_declaration_sha256':c['secondary_declaration_sha256'],
              'shard_storage':c['partition']['shard_storage'][c['shard_id']]}
    with (folder/'dataset.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker=folder/'initial_dataset_verified.json'
        if marker.exists():
            if read(marker)!=identity:raise ValueError('Secondary dataset belongs to another declaration or shard')
        else:
            if any(p.is_file() and not p.name.endswith('.lock') for pattern in ['tasks/**/*','jobs/**/*','claims/*.owner.json']
                   for p in (c['top']/'secondary').glob(pattern)):
                raise ValueError('Unbound preexisting secondary output cannot be adopted')
            bind(marker,identity)
        keys={tid.replace('/','__') for tid in c['allowed_task_ids']}
        if any(p.name not in keys for p in (c['top']/'secondary/tasks').glob('*') if p.is_dir()):
            raise ValueError('Secondary task output crossed its immutable shard boundary')


def primary_inputs(c,tid):
    h,hs,folder=secondary.primary.verify_history(c,tid,'source')
    row=secondary.primary.verify_job_receipt(c,tid,'receiver')
    if row is None:raise ValueError('Primary history/native control is not committed')
    control=secondary.primary.scoped_file(c['top'],row['control_path'])
    if control.parent!=(secondary.primary.task_folder(c,tid)/'controls').resolve():
        raise ValueError('Primary native control belongs to another task')
    secondary.primary.validate_control_record(read(control),h,hs)
    paths=[folder/name for name in ['identity.json','resume.json','complete.json','completion_timing.json','source_history.json','history_ready.json']]
    paths += [control,c['top']/'primary/jobs'/('receiver_'+tid.replace('/','__'))/'complete.json']
    return {str(path.relative_to(c['top'])):sha(path) for path in paths}


@contextmanager
def selected_claims(c):
    if secondary.claim_job is not original_claim:raise ValueError('Unexpected secondary scheduler interposition')
    jobs={'receiver_'+tid.replace('/','__'):tid for tid in c['allowed_task_ids']}
    @contextmanager
    def gate(context,job):
        if context is not c or context['root']!=c['top']/'secondary':raise ValueError('Secondary claim namespace differs')
        if job not in jobs:yield False;return
        with original_claim(context,job) as acquired:
            protected=None
            if acquired:
                fresh=public_context(c['repo_root'],c['plan_path'],c['plan_sha256'],verify_weights=False)
                verify_allocation(fresh,verify_local=True)
                if fresh['allowed_task_ids']!=c['allowed_task_ids'] or fresh['shard_id']!=c['shard_id']:
                    raise ValueError('Secondary shard assignment drift')
                protected=primary_inputs(c,jobs[job])
            try:yield acquired
            finally:
                if protected and any(sha(c['top']/name)!=expected for name,expected in protected.items()):
                    raise ValueError('Secondary execution modified an immutable primary history/control input')
    secondary.claim_job=gate
    try:yield
    finally:secondary.claim_job=original_claim


def work(c):
    c.update(public_context(ROOT,c['plan_path'],c['plan_sha256'],verify_weights=True))
    lease,mount=verify_allocation(c,verify_local=True);verify_dataset(c)
    gpu=idle_gpu()
    storage={'lease':lease,'provider_mount_receipt':mount,
        'provider_inspection':regional.checked(c['repo_root'],mount['provider_inspection_path'],mount['provider_inspection_sha256']),
        'lease_file_sha256':c['plan']['lease_sha256'],'provider_mount_receipt_file_sha256':c['plan']['provider_mount_receipt_sha256']}
    bind(c['status_root']/'operational_secondary_storage.json',storage)
    checkpoint_receipt={'secondary_declaration_sha256':c['secondary_declaration_sha256'],
        'checkpoints':c['secondary_checkpoints'],'both_final_mapper_and_full_bytes_verified':True}
    bind(c['status_root']/'secondary_checkpoint_verification.json',checkpoint_receipt)
    bind(c['status_root']/'operational_secondary_shard.json',{
        'partition_sha256':c['partition_sha256'],'shard_id':c['shard_id'],
        'shard_storage':c['partition']['shard_storage'][c['shard_id']],
        'secondary_declaration_sha256':c['secondary_declaration_sha256'],
        'secondary_shard_source_manifest_sha256':c['plan']['secondary_shard_source_manifest_sha256'],
        'plan_path':c['plan_path'],'plan_sha256':c['plan_sha256'],'worker_role':ROLE,
        'storage_proof_sha256':sha(c['status_root']/'operational_secondary_storage.json'),
        'checkpoint_verification_sha256':sha(c['status_root']/'secondary_checkpoint_verification.json'),
        'idle_gpu_before_load':gpu,'numerical_secondary_code_changed':False,'primary_artifacts_modified':False})
    with selected_claims(c):secondary.work(c)
    from scripts.coding_confirmation_secondary_shard_merge_v2 import seal_shard
    seal_shard(c)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['source-manifest','run'])
    parser.add_argument('--output');parser.add_argument('--plan');parser.add_argument('--plan-sha256');parser.add_argument('--worker-id')
    a=parser.parse_args()
    if a.action=='source-manifest':
        result=source_manifest(ROOT,a.output);print(json.dumps({'path':a.output,'sha256':sha(ROOT/a.output),'commit':result['operational_commit']}));return
    from scripts.coding_confirmation_runtime import build_context
    c=None
    try:
        public=public_context(ROOT,a.plan,a.plan_sha256,verify_weights=True);verify_allocation(public,verify_local=True)
        c=build_context(a.plan,ROLE,a.worker_id);c.update(plan_path=a.plan,plan_sha256=a.plan_sha256);work(c)
    except BaseException as exc:
        if c:
            c['telemetry'].failure(exc);write(c['status_root']/'failure.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'epoch':time.time()})
            c['publish'](state='failed',stage='secondary_regional_failed',error=str(exc))
        if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
        raise SystemExit(65 if isinstance(exc,(ValueError,FloatingPointError,AssertionError)) else 1) from exc


if __name__=='__main__':main()
