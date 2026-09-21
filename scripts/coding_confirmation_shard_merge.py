#!/usr/bin/env python3
"""Public immutable shard export/merge; original whole-primary closure is authoritative.

Merge into a separate CPU dataset, never a live generation tree. This module
does not load models, private tests or candidate executors. Nonidentical file
collisions are fatal and never resolved by choosing a result or rerunning it.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, digest, sha, write
from scripts import coding_confirmation_generate as primary
from scripts import coding_confirmation_sharded_generate as operational


def read(path): return json.loads(Path(path).read_text())


def task_files(c, tid):
    """Validate each authentic task using frozen verification primitives."""
    top = c['top']; folder = primary.task_folder(c, tid); files = set(); answers = []
    jobs = {kind: primary.verify_job_receipt(c, tid, kind) for kind in ['source', 'small', 'receiver']}
    if any(row is None for row in jobs.values()): return None
    for kind, row in jobs.items():
        path = top / 'primary/jobs' / (kind + '_' + tid.replace('/', '__')) / 'complete.json'; files.add(path)
        runtime_path = primary.scoped_file(top, row['runtime_path']); runtime = read(runtime_path)
        setup_path = primary.scoped_file(top, row['setup_path']); setup = read(setup_path)
        if (digest(runtime) != row['runtime_identity_sha256'] or runtime.get('torch') != '2.8.0+cu128' or
                runtime.get('transformers') != '4.57.6' or runtime.get('attention') != 'sdpa' or
                runtime.get('dtype') != 'bfloat16' or 'H200' not in runtime.get('gpu', '') or
                runtime.get('TF32') is not False or runtime.get('deterministic_algorithms') is not True or
                runtime_path.name != 'runtime.json' or not runtime_path.is_relative_to(top / 'workers') or
                setup_path != runtime_path.parent / 'model_setup.json' or
                setup.get('declaration_sha256') != c['declaration_sha256'] or
                setup.get('runtime_identity_sha256') != row['runtime_identity_sha256'] or
                setup.get('charged_to_single_output_inference') is not False or
                set(setup.get('model_loading_seconds', {})) != {'source', 'receiver'} or
                any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in [*setup['model_loading_seconds'].values(), setup.get('mapper_initialization_seconds')])):
            raise ValueError('Shard runtime/model setup evidence differs')
        files.update([runtime_path, setup_path])
        proof_path = runtime_path.parent / 'operational_shard.json'
        if proof_path.exists():
            proof = read(proof_path)
            if (proof.get('partition_sha256') != c['partition_sha256'] or proof.get('shard_id') != c['shard_id'] or
                    proof.get('adapter_source_manifest_sha256') != c['plan']['adapter_source_manifest_sha256'] or
                    proof.get('numerical_primary_code_changed') is not False or proof.get('analysis_population_or_seed_changed') is not False):
                raise ValueError('Worker operational admission proof differs')
            storage_path = runtime_path.parent / 'operational_storage.json'
            storage = read(primary.scoped_file(top, str(storage_path.relative_to(top))))
            expected_store = c['partition']['shard_storage'][c['shard_id']]
            lease, mount = storage.get('lease', {}), storage.get('provider_mount_receipt', {})
            if (proof.get('shard_storage') != expected_store or sha(storage_path) != proof.get('storage_proof_sha256') or
                    storage.get('lease_file_sha256') != proof.get('lease_sha256') or
                    storage.get('provider_mount_receipt_file_sha256') != proof.get('provider_mount_receipt_sha256') or
                    any(lease.get(k) != v or mount.get(k) != v for k, v in expected_store.items()) or
                    lease.get('experiment_id') != c['declaration']['experiment_id'] or
                    mount.get('experiment_id') != lease.get('experiment_id') or mount.get('pod_id') != lease.get('pod_id') or
                    mount.get('kind') != 'verified_generation_store' or mount.get('provider_mount_verified') is not True):
                raise ValueError('Worker physical store admission proof differs')
            files.update([proof_path, storage_path])
        else:
            old = {r['path']: r for r in c['quiescence']['files']}
            rel = str(path.relative_to(top))
            if rel not in old or sha(path) != old[rel]['sha256']:
                raise ValueError('New job lacks its immutable operational admission proof')
    large, large_sha, _ = primary.verify_history(c, tid, 'source')
    small, small_sha, _ = primary.verify_history(c, tid, 'small')
    control = primary.scoped_file(top, jobs['receiver']['control_path'])
    if control.parent != (folder / 'controls').resolve(): raise ValueError('Control belongs to another task')
    primary.validate_control_record(read(control), large, large_sha)
    ppath = folder / 'prompt_only_template.json'; p = read(ppath)
    if (p.get('task_id') != tid or p.get('declaration_sha256') != c['declaration_sha256'] or
            p.get('public_prompt_sha256') != digest(c['visible'][tid]['prompt']) or p.get('enable_thinking') is not False or
            not p.get('prompt_ids') or p.get('prefix_ids') != p['prompt_ids'][:-1] or
            p.get('bridge_ids') != p['prompt_ids'][-1:] or '<think>\n\n</think>' not in p.get('rendered_template', '')):
        raise ValueError('Shard prompt-only template differs')
    ph = {'prefix_ids': p['prefix_ids'], 'bridge_ids': p['bridge_ids']}; psha = sha(ppath)
    expected_paths = set()
    for condition in primary.CONDITIONS:
        h, hs = (small, small_sha) if condition == 'B' else ((ph, psha) if condition == 'P' else (large, large_sha))
        cp = c['declaration']['primary_checkpoints'][condition.split('_')[0]]['mapper_sha256'] if '_' in condition else None
        for seed in c['seeds']['tasks'][tid]['answers']:
            expected = primary.contract(c, tid, condition, seed, hs, cp)
            dest = folder / condition / ('seed_' + str(seed['seed_index']))
            if not primary.verify_draw(dest, expected, h, require_receipt=True): raise ValueError('Shard task lacks an authentic answer')
            raw = read(dest / 'answer.json'); seconds = raw.get('checkpoint_load_seconds'); load = raw.get('checkpoint_load_receipt_path')
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0: raise ValueError('Mapper setup timing invalid')
            if load is not None:
                path = primary.scoped_file(top, load); value = read(path)
                wanted = {'task_id': tid, 'condition': condition, 'cohort': 'primary', 'mapper_sha256': cp,
                    'declaration_sha256': c['declaration_sha256'], 'checkpoint_load_seconds': seconds,
                    'charged_to_single_output_inference': False}
                if (path.parent != (folder / 'mapper_loads').resolve() or sha(path) != raw.get('checkpoint_load_receipt_sha256') or
                        any(value.get(k) != v for k, v in wanted.items())): raise ValueError('Mapper setup receipt binding differs')
            elif seconds != 0 or raw.get('checkpoint_load_receipt_sha256') is not None: raise ValueError('Mapper setup receipt missing')
            relative = str((dest / 'answer.json').relative_to(top)); expected_paths.add(relative)
            answers.append({'contract': expected, 'contract_sha256': digest(expected), 'path': relative,
                            'answer_sha256': sha(dest / 'answer.json')})
    if {str(p.relative_to(top)) for p in folder.rglob('answer.json')} != expected_paths:
        raise ValueError('Unexpected shard answer population')
    # All task-local attempts are immutable once all three whole-task jobs close.
    files.update(p for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.lock'))
    return files, answers


def seal_shard(c, seal=True):
    files = set(); answers = []
    for tid in c['allowed_task_ids']:
        value = task_files(c, tid)
        if value is None: return None
        files.update(value[0]); answers.extend(value[1])
    rows = []
    for path in sorted(files):
        rel = str(path.relative_to(c['top'])); path = primary.scoped_file(c['top'], rel)
        rows.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    value = {'schema': 1, 'kind': 'authentic_primary_task_shard', 'experiment_id': c['declaration']['experiment_id'],
        'declaration_sha256': c['declaration_sha256'], 'partition_sha256': c['partition_sha256'],
        'adapter_source_manifest_sha256': c['plan']['adapter_source_manifest_sha256'],
        'shard_storage': c['partition']['shard_storage'][c['shard_id']],
        'shard_id': c['shard_id'], 'task_ids': c['allowed_task_ids'], 'answer_count': len(answers),
        'answers': answers, 'files': rows, 'all_assigned_tasks_complete': True,
        'whole_primary_complete_claimed': False, 'private_tests_loaded': False}
    value['shard_sha256'] = digest(value)
    if seal: bind(c['top'] / 'sharding/shards' / (c['shard_id'] + '.json'), value)
    return value


def import_shard(c, source_root, manifest_path, manifest_sha256):
    """Byte-preserving import from a verified public packet into a CPU-only tree."""
    require_merge(c)
    source_root = Path(source_root).resolve(); destination = c['top']
    receipt = operational.checked(source_root, manifest_path, manifest_sha256)
    if (receipt.get('shard_sha256') != digest({k: v for k, v in receipt.items() if k != 'shard_sha256'}) or
            receipt.get('shard_storage') != c['partition']['shard_storage'][c['shard_id']] or
            receipt.get('shard_id') != c['shard_id'] or receipt.get('task_ids') != c['allowed_task_ids'] or
            receipt.get('experiment_id') != c['declaration']['experiment_id'] or receipt.get('kind') != 'authentic_primary_task_shard' or
            receipt.get('declaration_sha256') != c['declaration_sha256'] or receipt.get('partition_sha256') != c['partition_sha256'] or
            receipt.get('adapter_source_manifest_sha256') != c['plan']['adapter_source_manifest_sha256'] or
            receipt.get('all_assigned_tasks_complete') is not True or receipt.get('whole_primary_complete_claimed') is not False or
            receipt.get('private_tests_loaded') is not False): raise ValueError('Shard manifest identity differs')
    source_context = {**c, 'top': source_root, 'root': source_root}
    actual = seal_shard(source_context, seal=False)
    if actual != receipt: raise ValueError('Shard packet differs from its authentic completed transactions')
    if (destination / 'claims').exists() or (destination / 'sharding/initial_dataset_verified.json').exists():
        raise ValueError('Merge requires a separate CPU-only dataset, never a generation tree')
    folder = destination / 'sharding'; folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'merge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Check every collision before copying anything. No overwrite or selection.
        for item in receipt['files']:
            target = operational.output(destination, item['path'])
            if target.exists() and (target.stat().st_size != item['bytes'] or sha(target) != item['sha256']):
                raise ValueError('Nonidentical merge collision: ' + item['path'])
        for item in receipt['files']:
            target = operational.output(destination, item['path'])
            if target.exists(): continue
            source = primary.scoped_file(source_root, item['path']); target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.merge-' + uuid.uuid4().hex)
            with source.open('rb') as inp, temporary.open('xb') as out:
                shutil.copyfileobj(inp, out, 1024 * 1024); out.flush(); os.fsync(out.fileno())
            if temporary.stat().st_size != item['bytes'] or sha(temporary) != item['sha256']:
                raise ValueError('Source changed while copying immutable shard')
            os.replace(temporary, target)
            fd = os.open(target.parent, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        bind(folder / 'shards' / (c['shard_id'] + '.json'), receipt)
        imports = folder / 'imports'; imports.mkdir(exist_ok=True)
        bind(imports / (c['shard_id'] + '.json'), {'shard_id': c['shard_id'], 'partition_sha256': c['partition_sha256'],
            'manifest_sha256': manifest_sha256, 'shard_sha256': receipt['shard_sha256'], 'files': receipt['files'],
            'nonidentical_overwrites': False, 'candidate_reruns': False})
    return receipt


def require_merge(c):
    if c['plan'].get('execution_mode') != 'merge': raise ValueError('Import and closure require a merge dispatch')
    if str(c['top']) in {s['allowed_result_root'] for s in c['partition']['shard_storage'].values()}:
        raise ValueError('Merge must use a separate CPU result store')


def merged_closure(c):
    require_merge(c)
    folder = c['top'] / 'sharding'; folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'merge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        expected = set(c['partition']['shards'])
        if {p.stem for p in (folder / 'imports').glob('*.json')} != expected:
            raise ValueError('Every disjoint shard must be imported before whole-primary closure')
        for sid, tasks in c['partition']['shards'].items():
            row = read(folder / 'imports' / (sid + '.json'))
            if row['partition_sha256'] != c['partition_sha256']: raise ValueError('Mixed operational partitions')
            context = {**c, 'shard_id': sid, 'allowed_task_ids': tasks}
            actual = seal_shard(context, seal=False)
            if actual is None or actual != read(folder / 'shards' / (sid + '.json')) or actual['shard_sha256'] != row['shard_sha256']:
                raise ValueError('Merged shard evidence changed')
        result = primary.generation_closure(c, seal=True)
        if result is None or result['answer_count'] != c['declaration']['primary_answer_count']:
            raise ValueError('Original complete primary closure did not validate')
        bind(folder / 'merged_complete.json', {'declaration_sha256': c['declaration_sha256'],
            'partition_sha256': c['partition_sha256'], 'primary_closure_sha256': result['closure_sha256'],
            'primary_closure_file_sha256': sha(c['top'] / 'primary/generation_closure.json'),
            'actual_task_count': len(c['declaration']['task_ids']), 'actual_answer_count': result['answer_count'],
            'original_numerical_generation_and_analysis_unchanged': True})
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['seal-shard', 'import', 'close'])
    parser.add_argument('--plan', required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--source-root'); parser.add_argument('--manifest'); parser.add_argument('--manifest-sha256')
    args = parser.parse_args(); c = operational.public_context(ROOT, args.plan, args.plan_sha256)
    if args.action == 'seal-shard': result = seal_shard(c)
    elif args.action == 'import': result = import_shard(c, args.source_root, args.manifest, args.manifest_sha256)
    else: result = merged_closure(c)
    print(json.dumps({'complete': result is not None, 'answers': result.get('answer_count') if result else None}))
