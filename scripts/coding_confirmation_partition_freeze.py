#!/usr/bin/env python3
"""Freeze the authorized200-task regional schedule from public quiescence metadata.

Scheduling inputs are declared task order, prior task exposure, and fixed worker
capacity only. Candidate text, outcomes and private tests are never read. This
helper cannot launch workers, stop allocations, or change the numerical recipe.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, sha, write
from scripts import coding_confirmation_sharded_generate as sharded

SCRIPT = 'scripts/coding_confirmation_partition_freeze.py'
DECLARATION = 'configs/coding_pilot_v1/confirmation_01/declaration.json'
DECLARATION_SHA = '16387d5361c795121929fd9244821f12ee005f9b36b2febddca5d9ef89f7536f'
ADAPTER = 'configs/coding_pilot_v1/confirmation_01/sharded_adapter_source.json'
RESULT_ROOT = '/workspace/GearshiftConfirmationPrimary/results/coding_pilot_v1/confirmation_01_20260919T094418Z'
RESOURCE_ROOT = 'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/resources/'
SHARDS = [('origin', 'u4rtme34ja', 2, 50), ('usco', 'zeicr9elbn', 1, 25), ('fr', 'em4bdfudg4', 5, 125)]
ORIGINAL_ALLOCATIONS = [
    {'pod_id':'x6jy63vp7jvx6n', 'lease_path':RESOURCE_ROOT+'primary_pair01/lease.json',
     'lease_sha256':'b93f6415fb0ee38fb3cd2ac6b4454f65dbef7db8809a2158622a11d04da576ae'},
    {'pod_id':'v7ud2o1wx86y9e', 'lease_path':RESOURCE_ROOT+'primary_pair02/lease.json',
     'lease_sha256':'e37936502d5e07f0786182a1b4bb21915155aca32c919e1f9b7b2170d3a54d0e'}]
EXCLUDED_TRAINING = ['rekbqruqox2bk9']
ALGORITHM = 'static_capacity_proportional_with_preserved_touched_tasks_v1'


def assignments(task_ids, touched):
    if (len(task_ids) != 200 or len(set(task_ids)) != 200 or not isinstance(touched, list) or
            len(touched) != len(set(touched)) or not set(touched) <= set(task_ids)):
        raise ValueError('Expected the unchanged200 unique tasks and unique declared touched IDs')
    if touched != [tid for tid in task_ids if tid in set(touched)]:
        raise ValueError('Touched task IDs must retain the original declared order')
    if len(touched) > 50:
        raise ValueError('Touched task count exceeds the fixed50-task origin capacity; explicit new allocation decision required')
    origin = set(touched)
    origin.update([tid for tid in task_ids if tid not in origin][:50-len(origin)])
    remaining = [tid for tid in task_ids if tid not in origin]
    return {'origin':[tid for tid in task_ids if tid in origin], 'usco':remaining[:25], 'fr':remaining[25:]}


def owner_authority(repo, declaration):
    scope = declaration['task_scope_resolution']
    if (scope.get('authority') != 'direct_owner_reply' or scope.get('decision') != 'all_200' or
            scope.get('task_ids') != declaration['task_ids'] or scope.get('unique_task_count') != 200):
        raise ValueError('Frozen owner authority does not authorize this unchanged200-task population')
    files = [{'path':declaration['owner_instruction_path'],'sha256':declaration['owner_instruction_sha256']},
             {'path':scope['amendment_path'],'sha256':scope['amendment_sha256']}]
    for item in files:
        if sha(sharded.primary.scoped_file(repo,item['path'])) != item['sha256']:
            raise ValueError('Pinned owner authority bytes differ')
    return {'authority':'direct_owner_reply','decision':'all_200','files':files,
            'authority_from_frozen_declaration':True,'capacity_assignment_does_not_amend_scientific_scope':True}


def freeze(repo, output_path, quiescence_path, quiescence_sha, adapter_path, adapter_sha,
           declaration_path=DECLARATION, declaration_sha=DECLARATION_SHA):
    repo = Path(repo).resolve(); output = sharded.output(repo,output_path)
    if output.exists(): raise ValueError('Partition already exists; overwrite or implicit repartition is forbidden')
    d = sharded.primary.validate_declaration(repo,declaration_path,declaration_sha)
    q = sharded.checked(repo,quiescence_path,quiescence_sha)
    source = sharded.checked(repo,adapter_path,adapter_sha)
    if d.get('primary_conditions') != sharded.primary.CONDITIONS or d.get('primary_answer_count') != 4800:
        raise ValueError('Primary eight-condition/three-draw population differs')
    p = {'schema':1,'experiment_id':d['experiment_id'],'declaration_sha256':declaration_sha,
        'source_commit':d['source_commit'],'task_ids':d['task_ids'],'task_count':200,
        'whole_task_assignment':True,'static_after_freeze':True,
        'shards':assignments(d['task_ids'],q['touched_task_ids']),
        'shard_storage':{sid:{'network_volume_id':volume,'allowed_result_root':RESULT_ROOT} for sid,volume,_,_ in SHARDS},
        'origin_shard_id':'origin','touched_task_ids':q['touched_task_ids'],
        'quiescence_path':quiescence_path,'quiescence_sha256':quiescence_sha,
        'original_generation_allocations':ORIGINAL_ALLOCATIONS,'excluded_training_pod_ids':EXCLUDED_TRAINING,
        'adapter_source_manifest_path':adapter_path,'adapter_source_manifest_sha256':adapter_sha,
        'operational_source_commit':source['operational_commit'],'owner_authority':owner_authority(repo,d),
        'partition_builder':{'path':SCRIPT,'sha256':sha(repo/SCRIPT)},
        'allocation_algorithm':{'name':ALGORITHM,'worker_weights':{sid:weight for sid,_,weight,_ in SHARDS},
            'target_task_counts':{sid:count for sid,_,_,count in SHARDS},
            'rule':'Keep all touched tasks in origin; fill origin to50 with earliest untouched declared IDs; assign next25 to USCO and final125 to FR. Every shard retains declared order.',
            'task_order_input':'unchanged_primary_declaration','touched_input':'verified_quiescence_file_inventory',
            'outcome_guided':False,'candidate_content_read':False,'private_tests_read':False},
        'primary_conditions':list(sharded.primary.CONDITIONS),'answer_draws_per_condition':3,
        'primary_answer_count':4800,'numerical_generation_or_analysis_changed':False,
        'training_restarted':False,'healthy_secondary_training_excluded':True}
    if {sid:len(tasks) for sid,tasks in p['shards'].items()} != {sid:count for sid,_,_,count in SHARDS}:
        raise ValueError('Partition does not match the fixed50/25/125 capacity allocation')
    sharded.validate_partition(d,p,q)
    # Reuse the entire frozen admission validator before making a durable public
    # partition. A temporary merge-only plan makes no physical ownership claim.
    with tempfile.TemporaryDirectory(prefix='.partition-validation-',dir=repo) as temporary:
        root = Path(temporary); candidate = root/'partition.json'; plan = root/'merge_validation.json'
        write(candidate,p)
        values = {'execution_mode':'merge','experiment_id':d['experiment_id'],'code_commit':d['source_commit'],
            'declaration_path':declaration_path,'declaration_sha256':declaration_sha,
            'partition_path':str(candidate.relative_to(repo)),'partition_sha256':sha(candidate),
            'adapter_source_manifest_path':adapter_path,'adapter_source_manifest_sha256':adapter_sha,
            'shard_id':'origin','result_root':str((root/'unused_cpu_destination').relative_to(repo))}
        write(plan,values)
        verified = sharded.public_context(repo,str(plan.relative_to(repo)),sha(plan))
        if verified['partition'] != p: raise ValueError('Full partition verification changed the proposed assignment')
    # This stable lock is unrelated to live generation claims; never unlink it.
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_name(output.name+'.freeze.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if output.exists(): raise ValueError('Concurrent partition freeze already exists; overwrite forbidden')
        bind(output,p)
    return {'partition_path':str(output.relative_to(repo)),'partition_sha256':sha(output),
        'task_counts':{sid:len(tasks) for sid,tasks in p['shards'].items()},
        'touched_task_count':len(p['touched_task_ids']),'algorithm':ALGORITHM,'outcome_guided':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=ROOT); parser.add_argument('--output',required=True)
    parser.add_argument('--quiescence',required=True); parser.add_argument('--quiescence-sha256',required=True)
    parser.add_argument('--adapter-source',default=ADAPTER); parser.add_argument('--adapter-source-sha256',required=True)
    parser.add_argument('--declaration',default=DECLARATION); parser.add_argument('--declaration-sha256',default=DECLARATION_SHA)
    a=parser.parse_args()
    print(json.dumps(freeze(a.repo,a.output,a.quiescence,a.quiescence_sha256,a.adapter_source,a.adapter_source_sha256,
                           a.declaration,a.declaration_sha256),indent=2))


if __name__=='__main__': main()
