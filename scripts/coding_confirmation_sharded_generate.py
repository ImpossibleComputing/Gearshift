#!/usr/bin/env python3
"""Explicit operational task partition; frozen numerical primary code is unchanged.

Only claim_job scheduling is interposed. Scientific declarations, source/answer
seeds, sampler calls, caches, checkpoints and authentic completion records retain
their original identities. Shards never pretend that another shard completed.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, digest, sha, write
from gearshift.coding_confirmation_lease import load_lease
from scripts import coding_confirmation_generate as primary
from scripts.coding_coverage_v2_worker import claim_job as original_claim

MOUNT_PATH = '/workspace'

IMPLEMENTATION = {'scripts/coding_confirmation_sharded_generate.py',
    'scripts/coding_confirmation_sharded_supervisor.py', 'scripts/coding_confirmation_shard_merge.py',
    'scripts/coding_confirmation_replication_supervisor.py', 'scripts/coding_confirmation_replication_stage.py',
    'scripts/coding_confirmation_replication.py', 'gearshift/coding_coverage_v2_lease.py'}


def read(path): return json.loads(Path(path).read_text())


def output(root, relative):
    root = Path(root).resolve(); relative = Path(relative)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Operational output must remain scoped')
    path = root / relative
    for item in [path, *path.parents]:
        if item == root: break
        if item.is_symlink(): raise ValueError('Linked operational output')
    return path


def source_manifest(repo, destination):
    repo = Path(repo).resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    for name in IMPLEMENTATION:
        if subprocess.check_output(['git', 'show', commit + ':' + name], cwd=repo) != (repo / name).read_bytes():
            raise ValueError('Commit the tested operational adapter before freezing: ' + name)
    value = {'schema': 1, 'operational_commit': commit, 'files': {p: sha(repo / p) for p in sorted(IMPLEMENTATION)},
        'numerical_primary_code_changed': False, 'interposition': 'claim_job task admission only',
        'analysis_population_or_seed_changed': False}
    bind(output(repo, destination), value); return value


def checked(repo, path, expected):
    file = primary.scoped_file(repo, path)
    if sha(file) != expected: raise ValueError('Pinned operational input hash differs: ' + path)
    return read(file)


def task_from_path(relative, task_ids):
    """Only authentic task/job/claim paths can establish prior task exposure."""
    parts = Path(relative).parts; lookup = {t.replace('/', '__'): t for t in task_ids}
    if len(parts) >= 4 and parts[:2] == ('primary', 'tasks'):
        key = parts[2]
    elif len(parts) >= 4 and parts[:2] == ('primary', 'jobs'):
        kind, _, key = parts[2].partition('_')
        if kind not in ('source', 'small', 'receiver'): raise ValueError('Unknown original job kind')
    elif len(parts) == 2 and parts[0] == 'claims' and parts[1].endswith('.owner.json'):
        kind, _, key = parts[1][:-len('.owner.json')].partition('_')
        if kind not in ('source', 'small', 'receiver'): raise ValueError('Unknown original claim kind')
    else: raise ValueError('Quiescence inventory is outside public task/job/claim evidence')
    if key not in lookup: raise ValueError('Quiescence contains an undeclared task')
    return lookup[key]


def validate_partition(d, partition, quiescence):
    tids = d['task_ids']; ordered = set(tids)
    if d['task_count'] != 200 or len(tids) != 200 or len(ordered) != 200:
        raise ValueError('Sharding requires the unchanged200-task primary')
    required = {'schema': 1, 'experiment_id': d['experiment_id'], 'source_commit': d['source_commit'],
                'task_ids': tids, 'task_count': 200, 'whole_task_assignment': True, 'static_after_freeze': True}
    if any(partition.get(k) != v for k, v in required.items()): raise ValueError('Partition changes primary scientific population')
    shards = partition.get('shards', {})
    if not isinstance(shards, dict) or not shards: raise ValueError('Missing whole-task shards')
    assigned = []
    for sid, tasks in shards.items():
        if (not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}', sid) or not isinstance(tasks, list) or not tasks or
                len(tasks) != len(set(tasks)) or tasks != [t for t in tids if t in set(tasks)]):
            raise ValueError('Shard tasks must be unique and retain declared order')
        assigned.extend(tasks)
    if len(assigned) != 200 or set(assigned) != ordered: raise ValueError('Task shards overlap or omit tasks')
    storage = partition.get('shard_storage', {})
    if not isinstance(storage, dict) or set(storage) != set(shards):
        raise ValueError('Every shard must bind its physical result store')
    stores = set()
    for value in storage.values():
        if not isinstance(value, dict) or set(value) != {'network_volume_id', 'allowed_result_root'}:
            raise ValueError('Invalid shard storage binding')
        volume, root = value['network_volume_id'], value['allowed_result_root']
        if not isinstance(volume, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', volume):
            raise ValueError('Invalid shard network volume identity')
        if (not isinstance(root, str) or not Path(root).is_absolute() or '..' in Path(root).parts or
                str(Path(root)) != root or not Path(root).is_relative_to(MOUNT_PATH) or root == MOUNT_PATH):
            raise ValueError('Shard result root must be an absolute path on its declared mount')
        store = (volume, root)
        if store in stores: raise ValueError('Independent shards cannot share one result store')
        stores.add(store)
    expected = {'schema': 1, 'experiment_id': d['experiment_id'],
        'declaration_sha256': partition['declaration_sha256'], 'all_original_primary_workers_stopped': True,
        'original_supervisors_released': True, 'healthy_training_untouched': True}
    if any(quiescence.get(k) != v for k, v in expected.items()): raise ValueError('Supported original-worker quiescence is unverified')
    files = quiescence.get('files', []); seen = set(); touched = set()
    for item in files:
        path = item['path']
        if path in seen or Path(path).is_absolute() or '..' in Path(path).parts: raise ValueError('Duplicate or unsafe quiescence file')
        if type(item.get('bytes')) is not int or item['bytes'] < 0 or not re.fullmatch('[0-9a-f]{64}', item.get('sha256', '')):
            raise ValueError('Invalid quiescence byte inventory')
        seen.add(path); touched.add(task_from_path(path, tids))
    touched_order = [t for t in tids if t in touched]
    if quiescence.get('touched_task_ids') != touched_order or partition.get('touched_task_ids') != touched_order:
        raise ValueError('Touched-task assignment differs from preserved evidence')
    if partition.get('origin_shard_id') not in shards or not touched <= set(shards[partition['origin_shard_id']]):
        raise ValueError('Every touched task must remain in the logical origin dataset')
    inventory = partition.get('original_generation_allocations', [])
    excluded = partition.get('excluded_training_pod_ids')
    if (not isinstance(inventory, list) or not inventory or
            any(set(item) != {'pod_id', 'lease_path', 'lease_sha256'} for item in inventory) or
            any(not isinstance(item['pod_id'], str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', item['pod_id']) or
                not re.fullmatch('[0-9a-f]{64}', item.get('lease_sha256', '')) for item in inventory)):
        raise ValueError('Exhaustive original generation lease inventory is missing')
    original_pods = [item['pod_id'] for item in inventory]
    if len(original_pods) != len(set(original_pods)):
        raise ValueError('Original generation allocation inventory contains duplicates')
    if (not isinstance(excluded, list) or not excluded or len(excluded) != len(set(excluded)) or
            any(not isinstance(pod, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', pod) for pod in excluded) or
            set(excluded) & set(original_pods)):
        raise ValueError('Healthy training allocations must be explicitly excluded from quiescence')
    receipts = quiescence.get('allocation_receipts', [])
    if not receipts or len({r.get('pod_id') for r in receipts}) != len(receipts):
        raise ValueError('Original allocation stop/release proofs missing')
    if {r['pod_id'] for r in receipts} != set(original_pods):
        raise ValueError('Quiescence receipts do not cover every original generation allocation')
    for row in receipts:
        if not row.get('pod_id') or set(row.get('files', {})) != {'workers_stopped', 'gpu_release_verified', 'release_manifest'}:
            raise ValueError('Incomplete original allocation quiescence proof')
    return shards


def public_context(repo, plan_path, plan_sha256, *, verify_loaded_source=True):
    repo = Path(repo).resolve(); plan = checked(repo, plan_path, plan_sha256)
    d = primary.validate_declaration(repo, plan['declaration_path'], plan['declaration_sha256'])
    if plan.get('code_commit') != d['source_commit'] or plan.get('experiment_id') != d['experiment_id']:
        raise ValueError('Sharded plan changes the frozen primary source identity')
    source = checked(repo, plan['adapter_source_manifest_path'], plan['adapter_source_manifest_sha256'])
    if (source.get('schema') != 1 or set(source.get('files', {})) != IMPLEMENTATION or
            not re.fullmatch('[0-9a-f]{40}', source.get('operational_commit', '')) or
            source.get('numerical_primary_code_changed') is not False or
            source.get('analysis_population_or_seed_changed') is not False or
            source.get('interposition') != 'claim_job task admission only'):
        raise ValueError('Operational source identity differs')
    for name, expected in source['files'].items():
        if sha(primary.scoped_file(repo, name)) != expected or (verify_loaded_source and sha(ROOT / name) != expected):
            raise ValueError('Loaded sharding implementation differs: ' + name)
    partition = checked(repo, plan['partition_path'], plan['partition_sha256'])
    if partition.get('declaration_sha256') != plan['declaration_sha256']: raise ValueError('Partition declaration differs')
    quiescence = checked(repo, partition['quiescence_path'], partition['quiescence_sha256'])
    shards = validate_partition(d, partition, quiescence)
    if plan.get('shard_id') not in shards: raise ValueError('Undeclared execution shard')
    # Stop/release receipts are public operational records, never private tests.
    original_leases = {}
    for item in partition['original_generation_allocations']:
        lease = load_lease(primary.scoped_file(repo, item['lease_path']), item['lease_sha256'])
        if lease['pod_id'] != item['pod_id'] or lease['experiment_id'] != d['experiment_id']:
            raise ValueError('Original generation lease identity differs from exhaustive inventory')
        original_leases[item['pod_id']] = lease
    for allocation in quiescence['allocation_receipts']:
        records = {name: checked(repo, item['path'], item['sha256']) for name, item in allocation['files'].items()}
        stop, release, manifest = records['workers_stopped'], records['gpu_release_verified'], records['release_manifest']
        old_lease = original_leases[allocation['pod_id']]
        stopped_path = str(Path(old_lease.get('control_relative', '')) / 'generation_supervisor/workers_stopped.json')
        stopped_rows = [r for r in manifest.get('files', []) if r.get('path') == stopped_path]
        if (release.get('manifest_sha256') != allocation['files']['release_manifest']['sha256'] or
                manifest.get('pod_id') != allocation['pod_id'] or manifest.get('experiment_id') != d['experiment_id'] or
                len(stopped_rows) != 1 or stopped_rows[0].get('sha256') != allocation['files']['workers_stopped']['sha256']):
            raise ValueError('Original stop proof is not bound to its own released allocation')
        if (stop.get('all_registered_workers_stopped') is not True or stop.get('process_group_ownership_verified') is not True or
                release.get('pod_id') != allocation['pod_id'] or release.get('experiment_id') != d['experiment_id'] or
                any(release.get(k) is not True for k in ['gpu_release_verified', 'all_gpu_workers_stopped', 'durable_backup_verified'])):
            raise ValueError('Original allocation did not complete supported verified quiescence')
    rows = read(primary.scoped_file(repo, d['visible_path']))
    visible = {r['task_id']: r for r in rows if r['task_id'] in d['task_ids']}
    if len({r['task_id'] for r in rows}) != len(rows) or set(visible) != set(d['task_ids']): raise ValueError('Public task population differs')
    top = output(repo, plan['result_root'])
    context = {'repo_root': repo, 'root': top, 'top': top, 'plan': plan, 'plan_path': plan_path, 'plan_sha256': plan_sha256,
        'declaration': d, 'declaration_sha256': plan['declaration_sha256'], 'partition': partition,
        'partition_sha256': plan['partition_sha256'], 'quiescence': quiescence, 'operational_source': source,
        'shard_id': plan['shard_id'], 'allowed_task_ids': shards[plan['shard_id']],
        'seeds': read(repo / d['seeds_path']), 'visible': visible}
    mode = plan.get('execution_mode')
    if mode == 'generation':
        validate_generation_store(context)
    elif mode == 'merge':
        if str(top) in {s['allowed_result_root'] for s in partition['shard_storage'].values()}:
            raise ValueError('Merge dispatch must use a separate CPU result destination')
    else: raise ValueError('Dispatch must explicitly select generation or merge mode')
    return context


def workspace_mount():
    """Read this process's real mount; no provider credential or network dependency."""
    def unescape(value):
        return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), value)
    matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before, separator, after = line.partition(' - ')
        fields, fs = before.split(), after.split()
        if separator and len(fields) >= 6 and len(fs) >= 2 and unescape(fields[4]) == MOUNT_PATH:
            matches.append({'mount_point': MOUNT_PATH, 'device': fields[2],
                'filesystem_type': unescape(fs[0]), 'source': unescape(fs[1])})
    if len(matches) != 1: raise ValueError('Exactly one real workspace mount must be visible')
    return matches[0]


def validate_generation_store(c, *, verify_local=False):
    """Pin one logical shard to one shared store, even across several worker pods.

    Staging supplies a fresh provider inspection and local mount receipt. The
    immutable receipt is checked throughout the allocation; it is not an online
    heartbeat. CPU merge verification never claims this physical ownership.
    """
    plan = c['plan']; repo = c['repo_root']
    if plan.get('execution_mode') != 'generation': raise ValueError('Worker requires a generation dispatch')
    lease = load_lease(primary.scoped_file(repo, plan['lease_path']), plan['lease_sha256'])
    store = c['partition']['shard_storage'][c['shard_id']]
    if (any(lease.get(k) != v for k, v in store.items()) or
            str(c['top']) != store['allowed_result_root'] or
            lease['experiment_id'] != c['declaration']['experiment_id'] or plan.get('pod_id') != lease['pod_id']):
        raise ValueError('Generation lease differs from immutable shard result store')
    receipt = checked(repo, plan['provider_mount_receipt_path'], plan['provider_mount_receipt_sha256'])
    expected = {'schema': 1, 'kind': 'verified_generation_store',
        'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id'], **store,
        'mount_path': MOUNT_PATH, 'provider_mount_verified': True}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Provider mount receipt differs from shard or allocation')
    observed, staged = receipt.get('provider_inspected_epoch'), receipt.get('staged_epoch')
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in [observed, staged]) or
            not lease['allocation_epoch'] <= observed <= staged <= lease['deadline_epoch'] or staged - observed > 300):
        raise ValueError('Provider mount inspection was not fresh when staged')
    observation = checked(repo, receipt['provider_inspection_path'], receipt['provider_inspection_sha256'])
    expected_observation = {'pod_id': lease['pod_id'], 'network_volume_id': store['network_volume_id'],
        'volume_mount_path': MOUNT_PATH, 'observed_epoch': observed, 'source': 'runpod_provider_inspection'}
    if any(observation.get(k) != v for k, v in expected_observation.items()):
        raise ValueError('Provider inspection payload differs from staged mount receipt')
    mount = receipt.get('local_mount', {})
    if (not isinstance(mount, dict) or set(mount) != {'mount_point', 'device', 'filesystem_type', 'source'} or
            mount.get('mount_point') != MOUNT_PATH or not re.fullmatch('[0-9]+:[0-9]+', mount.get('device', '')) or
            not all(isinstance(mount.get(k), str) and mount[k] for k in ['filesystem_type', 'source'])):
        raise ValueError('Local mount identity is missing or invalid')
    if verify_local:
        pod = os.environ.get('RUNPOD_POD_ID')
        if not pod:
            pod = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x).get(b'RUNPOD_POD_ID', b'').decode()
        if pod != lease['pod_id']: raise ValueError('Generation dispatch belongs to another pod')
        if workspace_mount() != mount: raise ValueError('Current workspace mount differs from staged provider mount')
        existing = c['top']
        while not existing.exists(): existing = existing.parent
        device = existing.stat().st_dev
        if str(os.major(device)) + ':' + str(os.minor(device)) != mount['device']:
            raise ValueError('Generation result store is not on the verified workspace mount')
    return lease, receipt


def verify_initial_dataset(c):
    """One locked initial snapshot check; later exact sampler resumes own changes."""
    folder = c['top'] / 'sharding'; folder.mkdir(parents=True, exist_ok=True)
    identity = {'partition_sha256': c['partition_sha256'], 'shard_id': c['shard_id'],
        'quiescence_sha256': c['partition']['quiescence_sha256'], 'allowed_task_ids': c['allowed_task_ids']}
    with (folder / 'dataset.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = folder / 'initial_dataset_verified.json'
        if receipt.exists():
            if read(receipt) != identity: raise ValueError('Dataset was assigned to a different immutable shard')
        else:
            expected = c['quiescence']['files'] if c['shard_id'] == c['partition']['origin_shard_id'] else []
            actual = []
            for pattern in ['primary/tasks/**/*', 'primary/jobs/**/*', 'claims/*.owner.json']:
                actual.extend(p for p in c['top'].glob(pattern) if p.is_file() and not p.name.endswith('.lock'))
            if {str(p.relative_to(c['top'])) for p in actual} != {r['path'] for r in expected}:
                raise ValueError('Initial shard dataset does not exactly match the quiesced task snapshot')
            for item in expected:
                p = primary.scoped_file(c['top'], item['path'])
                if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']: raise ValueError('Initial resume snapshot bytes differ')
            bind(receipt, identity)
        keys = {t.replace('/', '__') for t in c['allowed_task_ids']}
        if any(p.name not in keys for p in (c['top'] / 'primary/tasks').glob('*') if p.is_dir()):
            raise ValueError('Another shard wrote into this active dataset; use a separate merge destination')


@contextmanager
def selected_claims(c):
    if primary.claim_job is not original_claim: raise ValueError('Unexpected preexisting scheduler interposition')
    jobs = {kind + '_' + tid.replace('/', '__') for tid in c['allowed_task_ids'] for kind in ['source', 'small', 'receiver']}
    @contextmanager
    def gate(context, job_id):
        if context is not c: raise ValueError('Scheduler context differs')
        if job_id not in jobs:
            yield False; return
        with original_claim(context, job_id) as acquired:
            if acquired:
                # No admission after mutable plan, ownership or loaded-source drift.
                fresh = public_context(c['repo_root'], c['plan_path'], c['plan_sha256'])
                validate_generation_store(fresh, verify_local=True)
                if fresh['shard_id'] != c['shard_id'] or fresh['allowed_task_ids'] != c['allowed_task_ids']:
                    raise ValueError('Static shard assignment drift')
            yield acquired
    primary.claim_job = gate
    try: yield
    finally: primary.claim_job = original_claim


def work(c, preference):
    public = public_context(ROOT, c['plan_path'], c['plan_sha256']); c.update(public)
    lease, mount = validate_generation_store(c, verify_local=True)
    verify_initial_dataset(c)
    store_proof = {'lease': lease, 'provider_mount_receipt': mount,
        'provider_inspection': checked(c['repo_root'], mount['provider_inspection_path'], mount['provider_inspection_sha256']),
        'lease_file_sha256': c['plan']['lease_sha256'],
        'provider_mount_receipt_file_sha256': c['plan']['provider_mount_receipt_sha256']}
    bind(c['status_root'] / 'operational_storage.json', store_proof)
    bind(c['status_root'] / 'operational_shard.json', {'partition_sha256': c['partition_sha256'],
        'shard_id': c['shard_id'], 'adapter_source_manifest_sha256': c['plan']['adapter_source_manifest_sha256'],
        'plan_path': c['plan_path'], 'plan_sha256': c['plan_sha256'],
        'shard_storage': c['partition']['shard_storage'][c['shard_id']],
        'provider_mount_receipt_sha256': c['plan']['provider_mount_receipt_sha256'],
        'lease_sha256': c['plan']['lease_sha256'],
        'storage_proof_sha256': sha(c['status_root'] / 'operational_storage.json'),
        'numerical_primary_code_changed': False, 'analysis_population_or_seed_changed': False,
        'timing_note': 'Cold model loads and cache reconstruction remain setup/resume costs; fleet sharding is not a model latency gain.'})
    with selected_claims(c): primary.work(c, preference)
    from scripts.coding_confirmation_shard_merge import seal_shard
    seal_shard(c)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['source-manifest', 'run'])
    parser.add_argument('--output'); parser.add_argument('--plan'); parser.add_argument('--plan-sha256')
    parser.add_argument('--worker-id'); parser.add_argument('--preference', choices=['source', 'small', 'receiver'], default='source')
    args = parser.parse_args()
    if args.action == 'source-manifest':
        result = source_manifest(ROOT, args.output)
        print(json.dumps({'path': args.output, 'sha256': sha(ROOT / args.output), 'operational_commit': result['operational_commit']})); return
    from scripts.coding_confirmation_runtime import build_context
    c = None
    try:
        public = public_context(ROOT, args.plan, args.plan_sha256)
        validate_generation_store(public, verify_local=True)
        c = build_context(args.plan, 'confirmation_generation', args.worker_id)
        c.update(plan_path=args.plan, plan_sha256=args.plan_sha256); work(c, args.preference)
    except BaseException as exc:
        if c:
            c['telemetry'].failure(exc)
            write(c['status_root'] / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
                  'traceback': traceback.format_exc(), 'epoch': time.time()})
            c['publish'](state='failed', stage='sharded_generation_failed', error=str(exc))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)): raise
        raise SystemExit(65 if isinstance(exc, (ValueError, FloatingPointError, AssertionError)) else 1) from exc


if __name__ == '__main__': main()
