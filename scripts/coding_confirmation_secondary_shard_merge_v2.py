#!/usr/bin/env python3
"""Authentic secondary shard export/merge, followed by the frozen full2400 closure.

Only public transactions are verified. No private tests, scoring, GPU work, or
selection among conflicting outputs occurs. The primary tree remains unchanged.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gearshift.coding_control import bind,digest,sha
from gearshift.coding_confirmation_lease import verify_lease
from scripts import coding_confirmation_secondary_generate as secondary
from scripts import coding_confirmation_secondary_sharded_generate_v2 as operational
from scripts import coding_confirmation_shard_merge as primary_merge
primary=secondary.primary


def read(path):return json.loads(Path(path).read_text())


def admission_files(c,row):
    top=c['top'];runtime=primary.scoped_file(top,row['runtime_path'])
    if runtime.name!='runtime.json' or not runtime.is_relative_to(top/'workers'):
        raise ValueError('Secondary runtime is outside its worker attempt')
    proof_path=runtime.parent/'operational_secondary_shard.json';proof=read(proof_path)
    storage_path=runtime.parent/'operational_secondary_storage.json';storage=read(storage_path)
    cp_path=runtime.parent/'secondary_checkpoint_verification.json';cp=read(cp_path)
    store=c['partition']['shard_storage'][c['shard_id']]
    expected={'partition_sha256':c['partition_sha256'],'shard_id':c['shard_id'],'shard_storage':store,
        'secondary_declaration_sha256':c['secondary_declaration_sha256'],
        'secondary_shard_source_manifest_sha256':c['plan']['secondary_shard_source_manifest_sha256'],
        'worker_role':operational.ROLE,'numerical_secondary_code_changed':False,'primary_artifacts_modified':False,
        'storage_proof_sha256':sha(storage_path),'checkpoint_verification_sha256':sha(cp_path)}
    if any(proof.get(k)!=v for k,v in expected.items()):raise ValueError('Secondary regional admission identity differs')
    if cp!={'secondary_declaration_sha256':c['secondary_declaration_sha256'],
            'checkpoints':c['secondary_checkpoints'],'both_final_mapper_and_full_bytes_verified':True}:
        raise ValueError('Secondary final checkpoint byte-verification proof differs')
    lease,mount,provider=(storage.get(k,{}) for k in ['lease','provider_mount_receipt','provider_inspection'])
    verify_lease(lease)
    if (any(lease.get(k)!=v or mount.get(k)!=v for k,v in store.items()) or
            lease.get('experiment_id')!=c['declaration']['experiment_id'] or
            mount.get('experiment_id')!=lease['experiment_id'] or mount.get('pod_id')!=lease['pod_id'] or
            provider.get('pod_id')!=lease['pod_id'] or provider.get('network_volume_id')!=store['network_volume_id'] or
            provider.get('allocation_role')!=operational.ROLE or provider.get('exclusive_gpu_allocation') is not True or
            mount.get('provider_mount_verified') is not True or mount.get('kind')!='verified_generation_store' or
            not runtime.relative_to(top/'workers').parts[0].startswith(lease['pod_id']+'_secondary_')):
        raise ValueError('Secondary physical allocation provenance differs')
    gpu=proof.get('idle_gpu_before_load',{})
    if (type(gpu.get('free_fraction_before_load')) not in (int,float) or
            not .95<=gpu['free_fraction_before_load']<=1 or gpu.get('other_compute_owners') is not False):
        raise ValueError('Secondary did not verify an idle exclusive GPU before loading')
    return {proof_path,storage_path,cp_path}


def task_files(c,tid):
    top=c['top'];row=secondary.verify_job(c,tid)
    if row is None:return None
    protected=operational.primary_inputs(c,tid)
    files={};answers=[];histories=[];expected_answers=set();jobs=[(tid,row)]
    def include(path):
        path=Path(path);relative=str(path.relative_to(top));path=primary.scoped_file(top,relative)
        files[relative]={'path':relative,'bytes':path.stat().st_size,'sha256':sha(path)}
    def sampler(folder,kind):
        for name in ['identity.json','resume.json','complete.json','completion_timing.json',
                     'source_history.json' if kind=='reasoning' else 'answer_record.json']:include(folder/name)
    for name in protected:include(top/name)
    for path in admission_files(c,row):include(path)
    for tid, row in jobs:
        folder = secondary.task_folder(c, tid); include(secondary.job_path(c, tid))
        runtime_path = primary.scoped_file(top, row['runtime_path']); runtime = read(runtime_path)
        setup_path = primary.scoped_file(top, row['setup_path']); setup = read(setup_path)
        if (digest(runtime) != row['runtime_identity_sha256'] or
                runtime.get('torch') != '2.8.0+cu128' or runtime.get('transformers') != '4.57.6' or
                runtime.get('attention') != 'sdpa' or runtime.get('dtype') != 'bfloat16' or
                'H200' not in runtime.get('gpu', '') or runtime.get('TF32') is not False or
                runtime.get('deterministic_algorithms') is not True or
                setup_path != runtime_path.parent / 'model_setup.json' or
                setup.get('declaration_sha256') != c['declaration_sha256'] or
                setup.get('runtime_identity_sha256') != row['runtime_identity_sha256'] or
                setup.get('charged_to_single_output_inference') is not False or
                set(setup.get('model_loading_seconds', {})) != {'source', 'receiver'} or
                any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in [*setup['model_loading_seconds'].values(), setup.get('mapper_initialization_seconds')])):
            raise ValueError('Secondary runtime or model setup differs')
        include(runtime_path); include(setup_path)
        h, hs, history_folder = primary.verify_history(c, tid, 'source')
        sampler(history_folder, 'reasoning'); include(history_folder / 'history_ready.json')
        histories.append({'task_id': tid, 'kind': 'source', 'sha256': hs, 'reused_from_primary': True})
        old = primary.verify_job_receipt(c, tid, 'receiver')
        control_path = primary.scoped_file(top, old['control_path'])
        if control_path.parent != (primary.task_folder(c, tid) / 'controls').resolve():
            raise ValueError('Primary native control belongs to another task')
        primary.validate_control_record(read(control_path), h, hs); include(control_path)
        reuse_path = primary.scoped_file(top, row['control_reuse_path'])
        expected_reuse = {'primary_control_path': old['control_path'], 'primary_control_sha256': sha(control_path),
                          'source_history_sha256': hs, 'declaration_sha256': c['declaration_sha256']}
        if reuse_path != folder / 'control_reuse.json' or read(reuse_path) != expected_reuse:
            raise ValueError('Secondary control/history reuse differs')
        include(reuse_path)
        for condition in secondary.CONDITIONS:
            cp = c['secondary_checkpoints'][condition.split('_')[0]]['mapper_sha256']
            for seed in c['seeds']['tasks'][tid]['answers']:
                expected = primary.contract(c, tid, condition, seed, hs, cp, 'secondary')
                dest = folder / condition / ('seed_' + str(seed['seed_index']))
                if not primary.verify_draw(dest, expected, h, require_receipt=True):
                    raise ValueError('Secondary job lacks a declared answer')
                raw = read(dest / 'answer.json'); seconds = raw.get('checkpoint_load_seconds')
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError('Secondary mapper setup timing invalid')
                load = raw.get('checkpoint_load_receipt_path')
                if load is not None:
                    path = primary.scoped_file(top, load); receipt = read(path)
                    required = {'task_id': tid, 'condition': condition, 'cohort': 'secondary',
                        'mapper_sha256': cp, 'declaration_sha256': c['declaration_sha256'],
                        'checkpoint_load_seconds': seconds, 'charged_to_single_output_inference': False}
                    if (path.parent != (folder / 'mapper_loads').resolve() or
                            sha(path) != raw.get('checkpoint_load_receipt_sha256') or
                            any(receipt.get(k) != v for k, v in required.items())):
                        raise ValueError('Secondary mapper load receipt differs')
                    include(path)
                elif seconds != 0 or raw.get('checkpoint_load_receipt_sha256') is not None:
                    raise ValueError('Secondary mapper load lacks receipt')
                for name in ['draw_identity.json', 'answer.json', 'draw_complete.json']: include(dest / name)
                sampler(dest / 'sampler', 'answer')
                relative = str((dest / 'answer.json').relative_to(top)); expected_answers.add(relative)
                answers.append({'contract': expected, 'contract_sha256': digest(expected), 'path': relative,
                                'answer_sha256': files[relative]['sha256']})
    folder=secondary.task_folder(c,tid)
    if {str(path.relative_to(top)) for path in folder.rglob('answer.json')}!=expected_answers or list(folder.rglob('source_history.json')):
        raise ValueError('Undeclared secondary task population')
    for path in folder.rglob('*'):
        if path.is_file() and not path.name.endswith('.lock'):include(path)
    return files,answers


def seal_shard(c,seal=True):
    files={};answers=[]
    for tid in c['allowed_task_ids']:
        actual=task_files(c,tid)
        if actual is None:return None
        files.update(actual[0]);answers.extend(actual[1])
    if len(answers)!=12*len(c['allowed_task_ids']):raise ValueError('Secondary shard answer count differs')
    value={'schema':1,'kind':'authentic_secondary_task_shard','experiment_id':c['declaration']['experiment_id'],
        'declaration_sha256':c['declaration_sha256'],'secondary_declaration_sha256':c['secondary_declaration_sha256'],
        'partition_sha256':c['partition_sha256'],'shard_id':c['shard_id'],
        'shard_storage':c['partition']['shard_storage'][c['shard_id']],
        'secondary_shard_source_manifest_sha256':c['plan']['secondary_shard_source_manifest_sha256'],
        'training_seed':secondary.SEED,'checkpoints':c['secondary_checkpoints'],'task_ids':c['allowed_task_ids'],
        'conditions':secondary.CONDITIONS,'answer_count':len(answers),'answers':answers,
        'files':[files[name] for name in sorted(files)],'all_assigned_tasks_complete':True,
        'whole_secondary_complete_claimed':False,'private_tests_loaded':False,'primary_artifacts_modified':False}
    value['shard_sha256']=digest(value)
    if seal:bind(c['top']/'secondary/sharding/shards'/(c['shard_id']+'.json'),value)
    return value


def require_merge(c):
    primary_merge.require_merge(c)
    if (c['top']/'claims').exists() or (c['top']/'sharding/initial_dataset_verified.json').exists() or (c['top']/'secondary/claims').exists() or (c['top']/'secondary/sharding/initial_dataset_verified.json').exists():
        raise ValueError('Secondary merge requires a CPU-only tree, never an active regional store')


def import_shard(c,source_root,manifest_path,manifest_sha256):
    require_merge(c);source_root=Path(source_root).resolve();destination=c['top']
    receipt=operational.regional.checked(source_root,manifest_path,manifest_sha256)
    actual=seal_shard({**c,'top':source_root,'root':source_root/'secondary'},seal=False)
    if actual is None or actual!=receipt:raise ValueError('Secondary shard packet differs from authentic declared transactions')
    folder=destination/'secondary/sharding';folder.mkdir(parents=True,exist_ok=True)
    # Same lock as primary import prevents cross-cohort check/copy races on the
    # immutable primary history/control references included in secondary packets.
    shared=destination/'sharding';shared.mkdir(parents=True,exist_ok=True)
    with (shared/'merge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for item in receipt['files']:
            target=operational.regional.output(destination,item['path'])
            if target.exists() and (target.stat().st_size!=item['bytes'] or sha(target)!=item['sha256']):
                raise ValueError('Nonidentical secondary merge collision; overwrite forbidden')
        for item in receipt['files']:
            target=operational.regional.output(destination,item['path'])
            if target.exists():continue
            source=primary.scoped_file(source_root,item['path']);target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.merge-'+uuid.uuid4().hex)
            with source.open('rb') as inp,temporary.open('xb') as out:
                shutil.copyfileobj(inp,out,1024*1024);out.flush();os.fsync(out.fileno())
            if temporary.stat().st_size!=item['bytes'] or sha(temporary)!=item['sha256']:
                raise ValueError('Immutable secondary source changed during import')
            os.replace(temporary,target);fd=os.open(target.parent,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        bind(folder/'shards'/(c['shard_id']+'.json'),receipt)
        bind(folder/'imports'/(c['shard_id']+'.json'),{'shard_id':c['shard_id'],'partition_sha256':c['partition_sha256'],
            'secondary_declaration_sha256':c['secondary_declaration_sha256'],'manifest_sha256':manifest_sha256,
            'shard_sha256':receipt['shard_sha256'],'files':receipt['files'],'nonidentical_overwrites':False,'candidate_reruns':False})
    return receipt


def merged_closure(c):
    require_merge(c);folder=c['top']/'secondary/sharding';shared=c['top']/'sharding';shared.mkdir(parents=True,exist_ok=True)
    with (shared/'merge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if {p.stem for p in (folder/'imports').glob('*.json')}!=set(c['partition']['shards']):
            raise ValueError('Every secondary shard must be imported before complete closure')
        for sid,tasks in c['partition']['shards'].items():
            row=read(folder/'imports'/(sid+'.json'))
            context={**c,'shard_id':sid,'allowed_task_ids':tasks}
            actual=seal_shard(context,seal=False)
            if (row['partition_sha256']!=c['partition_sha256'] or row['secondary_declaration_sha256']!=c['secondary_declaration_sha256'] or
                    actual is None or actual!=read(folder/'shards'/(sid+'.json')) or actual['shard_sha256']!=row['shard_sha256']):
                raise ValueError('Merged secondary shard evidence changed')
        result=secondary.generation_closure(c,seal=True)
        if result is None or result['answer_count']!=c['declaration']['secondary_answer_count']:
            raise ValueError('Frozen full secondary closure requires all2400 answers and the original complete primary seal')
        bind(folder/'merged_complete.json',{'declaration_sha256':c['declaration_sha256'],
            'secondary_declaration_sha256':c['secondary_declaration_sha256'],'partition_sha256':c['partition_sha256'],
            'secondary_closure_sha256':result['closure_sha256'],'secondary_closure_file_sha256':sha(c['top']/'secondary/generation_closure.json'),
            'actual_task_count':len(c['declaration']['task_ids']),'actual_answer_count':result['answer_count'],
            'original_numerical_generation_and_analysis_unchanged':True})
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['seal-shard','import','close'])
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--source-root');parser.add_argument('--manifest');parser.add_argument('--manifest-sha256')
    a=parser.parse_args();c=operational.public_context(ROOT,a.plan,a.plan_sha256,verify_weights=False)
    if a.action=='seal-shard':result=seal_shard(c)
    elif a.action=='import':result=import_shard(c,a.source_root,a.manifest,a.manifest_sha256)
    else:result=merged_closure(c)
    print(json.dumps({'complete':result is not None,'answers':result.get('answer_count') if result else None}))
