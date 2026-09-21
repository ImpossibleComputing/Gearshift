#!/usr/bin/env python3
"""Stage and launch dedicated regional secondary workers on existing public stores.

No provisioning or weight transfer occurs here. Both final checkpoints must be
independently transferred and verified before preflight. Existing primary files
are immutable inputs; this helper never starts a primary worker.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import coding_confirmation_regional_generation as common
from scripts import coding_confirmation_secondary_sharded_generate_v2 as regional
from gearshift.coding_confirmation_lease import control_root, load_lease, linux_process_identity

SELF = 'scripts/coding_confirmation_secondary_regional_stage_ca01.py'
SUPERVISOR = 'scripts/coding_confirmation_secondary_sharded_supervisor_v2.py'
SECONDARY_SOURCE = common.CONFIG + '/secondary_shard_source_manifest_v2.json'
read, sha, bind_json = common.read, common.sha, common.bind_json


def source_paths(repo, secondary_path, extra):
    repo = Path(repo)
    d = regional.secondary.primary.validate_declaration(repo, common.DECLARATION, sha(repo / common.DECLARATION))
    secondary = read(common.public_file(repo, secondary_path))
    base_plan = {'experiment_id': d['experiment_id'], 'code_commit': d['source_commit'],
        'declaration_path': common.DECLARATION, 'declaration_sha256': sha(repo / common.DECLARATION),
        'secondary_declaration_path': secondary_path, 'secondary_declaration_sha256': sha(repo / secondary_path)}
    regional.secondary.validate_secondary(repo, base_plan, verify_weights=False)
    pins = {**d['inputs'], **d['implementation']}
    manifests = [common.ADAPTER, SECONDARY_SOURCE]
    source_proof = secondary['adapter_source_proof']
    if source_proof.get('method') != 'pinned_compact_source_manifest':
        raise ValueError('Regional staging requires the pinned compact secondary source proof')
    manifests.append(source_proof['path'])
    if sha(repo / source_proof['path']) != source_proof['sha256']:
        raise ValueError('Secondary adapter source proof changed')
    for manifest in manifests:
        for name, expected in read(common.public_file(repo, manifest))['files'].items():
            if name in pins and pins[name] != expected: raise ValueError('Conflicting scientific source pins')
            pins[name] = expected
    pins.update(secondary['implementation'])
    for cp in secondary['checkpoints'].values(): pins[cp['manifest_path']] = cp['manifest_sha256']
    paths = common.dependencies(repo, set(pins) | set(manifests) |
        {common.DECLARATION, secondary_path, common.SELF, SELF, SUPERVISOR} | set(extra))
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    for name in paths:
        path = common.public_file(repo, name)
        if name in pins:
            if sha(path) != pins[name]: raise ValueError('Pinned secondary input changed: ' + name)
        elif name.endswith('.py'):
            if path.read_bytes() != subprocess.check_output(['git', 'show', head + ':' + name], cwd=repo):
                raise ValueError('Commit tested operational dependency before staging: ' + name)
    return d, paths, head


def remote_call(pod, action, args):
    from scripts import coding_confirmation_dispatch as base
    command = ['.pilot-venv/bin/python', SELF, action, *args]
    result = subprocess.run(base.ssh_args(pod) + ['cd ' + shlex.quote(common.REMOTE) + ' && ' + shlex.join(command)],
                            capture_output=True, text=True, timeout=240)
    if result.returncode: raise RuntimeError('Secondary remote action failed: ' + result.stderr[-3000:])
    return json.loads(result.stdout)


def preflight(plan_path, plan_sha):
    c = regional.public_context(ROOT, plan_path, plan_sha, verify_weights=True)
    lease, _ = regional.verify_allocation(c, verify_local=True)
    actual = common.remote_inspect(c['plan']['lease_path'], c['plan']['lease_sha256'])
    regional.verify_dataset(c)
    if c['plan'].get('stage_helper_sha256') != sha(ROOT / SELF):
        raise ValueError('Secondary staging helper bytes differ from caller-pinned plan')
    # A completed native-control transaction is needed before any secondary job.
    ready = [tid for tid in c['allowed_task_ids']
             if regional.secondary.primary.verify_job_receipt(c, tid, 'receiver') is not None
             and regional.secondary.verify_job(c, tid) is None]
    if not ready: raise ValueError('No committed unprocessed primary controls are ready on this shard')
    return c, lease, {**actual, 'ready_task_count': len(ready),
                      'final_mapper_and_full_checkpoint_bytes_verified': True}


def verify_launch(lease, plan_path, plan_sha, identity, proc_root=Path('/proc')):
    current = linux_process_identity(identity['pid'], proc_root)
    if current != identity or identity['pgid'] != identity['pid']:
        raise ValueError('Secondary supervisor process identity differs')
    command = (proc_root / str(identity['pid']) / 'cmdline').read_bytes().split(b'\0')
    if not all(value.encode() in command for value in [SUPERVISOR, plan_path, plan_sha]):
        raise ValueError('Secondary supervisor command differs')
    registration = read(control_root(lease) / 'lease_guard/supervisor_registration.json')
    if any(registration.get(k) != v for k, v in {**identity, 'pod_id': lease['pod_id'],
                                                'experiment_id': lease['experiment_id']}.items()):
        raise ValueError('Secondary guard registration differs')


def launch_remote(plan_path, plan_sha):
    # Validate identity without requiring idle GPUs or new jobs for idempotent
    # observation of an already-running launch.
    c = regional.public_context(ROOT, plan_path, plan_sha, verify_weights=True)
    lease, _ = regional.verify_allocation(c, verify_local=True); control = control_root(lease)
    with (control / 'secondary_dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = control / 'secondary_launch_verified.json'; intent = control / 'secondary_launch_intent.json'
        if receipt.exists():
            saved = read(receipt)
            if saved['plan_path'] != plan_path or saved['plan_sha256'] != plan_sha:
                raise ValueError('Allocation already launched a different secondary dispatch')
            verify_launch(lease, plan_path, plan_sha, saved['process_identity'])
            return {**saved, 'already_running': True}
        if (intent.exists() or (control / 'secondary_sharded_supervisor').exists() or
                (control / 'lease_guard/supervisor_registration.json').exists()):
            raise ValueError('Existing or ambiguous supervisor ownership requires reconciliation')
        c, lease, actual = preflight(plan_path, plan_sha)
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        if processes.strip(): raise ValueError('Dedicated secondary allocation already has GPU compute processes')
        bind_json(intent, {'plan_path': plan_path, 'plan_sha256': plan_sha, 'pod_id': lease['pod_id'], 'epoch': time.time()})
        with (control / 'secondary_launcher.log').open('ab') as log:
            proc = subprocess.Popen([str(ROOT / '.pilot-venv/bin/python'), SUPERVISOR,
                '--plan', plan_path, '--plan-sha256', plan_sha], cwd=ROOT,
                env=common.all_device_environment(lease), stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True)
        identity = linux_process_identity(proc.pid)
        bind_json(control / 'secondary_launch_started.json', {'process_identity': identity, 'plan_sha256': plan_sha})
        for _ in range(60):
            if proc.poll() is not None: raise RuntimeError('Secondary supervisor exited; inspect saved logs before recovery')
            if (control / 'lease_guard/supervisor_registration.json').exists():
                verify_launch(lease, plan_path, plan_sha, identity)
                value = {'plan_path': plan_path, 'plan_sha256': plan_sha, 'process_identity': identity,
                    'pod_id': lease['pod_id'], 'guard_registered': True, 'epoch': time.time(), 'preflight': actual}
                bind_json(receipt, value); return value
            time.sleep(.5)
        raise TimeoutError('Secondary registration response is ambiguous; reconcile instead of relaunching')


def stage(allocation, partition_path, shard_id, secondary_path, launch=False):
    owner = ROOT / 'configs/coding_pilot_v1/confirmation_01/secondary_fr_execution_owner_v1.json'
    if shard_id == 'fr' and owner.exists():
        raise ValueError('France secondary ownership is delegated; resume its declared destination, not the original regional queue')
    from scripts import coding_confirmation_dispatch as base
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,99}', allocation): raise ValueError('Unsafe allocation')
    folder = base.EVIDENCE / allocation; lease_path = str((folder / 'lease.json').relative_to(ROOT))
    lease = load_lease(common.public_file(ROOT, lease_path), sha(ROOT / lease_path))
    partition = read(common.public_file(ROOT, partition_path))
    if partition_path != regional.PARTITION: raise ValueError('Use the original frozen regional partition')
    if partition['shard_storage'].get(shard_id) != {k: lease[k] for k in ('network_volume_id', 'allowed_result_root')}:
        raise ValueError('Secondary lease belongs to a different shard store')
    q = regional.regional.checked(ROOT, partition['quiescence_path'], partition['quiescence_sha256'])
    extra = {lease_path, partition_path, partition['quiescence_path'],
        *[r['lease_path'] for r in partition['original_generation_allocations']],
        *[v['path'] for r in q['allocation_receipts'] for v in r['files'].values()]}
    d, paths, head = source_paths(ROOT, secondary_path, extra)
    regional.regional.validate_partition(d, partition, q)
    if partition['declaration_sha256'] != sha(ROOT / common.DECLARATION): raise ValueError('Partition declaration differs')
    area = folder / 'secondary_regional_generation'; area.mkdir(exist_ok=True)
    bind_json(area / 'request.json', {'allocation': allocation, 'shard_id': shard_id,
        'partition_path': partition_path, 'partition_sha256': sha(ROOT / partition_path),
        'lease_sha256': sha(ROOT / lease_path), 'secondary_declaration_path': secondary_path,
        'secondary_declaration_sha256': sha(ROOT / secondary_path),
        'secondary_shard_source_manifest_sha256': sha(ROOT / SECONDARY_SOURCE)})
    complete = area / 'stage_complete.json'
    pod = base.api('pods/' + lease['pod_id'])
    observation = {**common.inspect_provider(pod, lease, allocation, time.time()),
                   'allocation_role': regional.ROLE, 'exclusive_gpu_allocation': True}
    if complete.exists():
        saved = read(complete)
        if sha(ROOT / saved['plan_path']) != saved['plan_sha256']: raise ValueError('Staged secondary plan changed')
    else:
        attempt = area / ('attempt_' + str(time.time_ns())); attempt.mkdir()
        op = attempt / 'provider_inspection.json'; bind_json(op, observation); paths.add(str(op.relative_to(ROOT)))
        common.upload(pod, paths, attempt, 'public_inputs')
        actual = common.remote_call(pod, '_inspect', ['--lease', lease_path, '--lease-sha256', sha(ROOT / lease_path)])
        if actual['pod_id'] != lease['pod_id'] or actual['visible_gpu_count'] != lease['gpu_count']:
            raise ValueError('Actual secondary GPU allocation differs')
        staged = time.time()
        if staged - observation['observed_epoch'] > 300: raise ValueError('Fresh provider inspection expired during staging')
        mount = {'schema': 1, 'kind': 'verified_generation_store', 'experiment_id': lease['experiment_id'],
            'pod_id': lease['pod_id'], **partition['shard_storage'][shard_id], 'mount_path': '/workspace',
            'provider_mount_verified': True, 'provider_inspected_epoch': observation['observed_epoch'],
            'staged_epoch': staged, 'local_mount': actual['local_mount'],
            'provider_inspection_path': str(op.relative_to(ROOT)), 'provider_inspection_sha256': sha(op),
            'actual_visible_gpu_count': actual['visible_gpu_count']}
        mp = attempt / 'provider_mount.json'; bind_json(mp, mount)
        plan = {'execution_mode': 'generation', 'pod_id': lease['pod_id'], 'experiment_id': d['experiment_id'],
            'code_commit': d['source_commit'], 'result_root': str(Path(lease['allowed_result_root']).relative_to(common.REMOTE)),
            'declaration_path': common.DECLARATION, 'declaration_sha256': sha(ROOT / common.DECLARATION),
            'lease_path': lease_path, 'lease_sha256': sha(ROOT / lease_path), 'partition_path': partition_path,
            'partition_sha256': sha(ROOT / partition_path), 'shard_id': shard_id,
            'adapter_source_manifest_path': common.ADAPTER, 'adapter_source_manifest_sha256': sha(ROOT / common.ADAPTER),
            'secondary_declaration_path': secondary_path, 'secondary_declaration_sha256': sha(ROOT / secondary_path),
            'secondary_shard_source_manifest_path': SECONDARY_SOURCE, 'secondary_shard_source_manifest_sha256': sha(ROOT / SECONDARY_SOURCE),
            'provider_mount_receipt_path': str(mp.relative_to(ROOT)), 'provider_mount_receipt_sha256': sha(mp),
            'worker_role': regional.ROLE, 'exclusive_gpu_allocation': True,
            'stage_helper_commit': head, 'stage_helper_sha256': sha(ROOT / SELF)}
        pp = attempt / 'generation_plan.json'; bind_json(pp, plan); plan_path = str(pp.relative_to(ROOT))
        common.upload(pod, {str(mp.relative_to(ROOT)), plan_path}, attempt, 'dispatch')
        proof = remote_call(pod, '_preflight', ['--plan', plan_path, '--plan-sha256', sha(pp)])
        saved = {'plan_path': plan_path, 'plan_sha256': sha(pp), 'preflight': proof, 'generation_launched': False}
        bind_json(complete, saved)
    if launch:
        outcome = remote_call(pod, '_launch', ['--plan', saved['plan_path'], '--plan-sha256', saved['plan_sha256']])
        common.save_launch(area / 'launch_verified.json', outcome); return outcome
    return saved


def launch_staged(allocation):
    from scripts import coding_confirmation_dispatch as base
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,99}', allocation): raise ValueError('Unsafe allocation')
    area = base.EVIDENCE / allocation / 'secondary_regional_generation'; saved = read(area / 'stage_complete.json')
    plan = read(common.public_file(ROOT, saved['plan_path']))
    if sha(ROOT / saved['plan_path']) != saved['plan_sha256']: raise ValueError('Secondary staged plan changed')
    lease = load_lease(common.public_file(ROOT, plan['lease_path']), plan['lease_sha256'])
    pod = base.api('pods/' + lease['pod_id']); common.inspect_provider(pod, lease, allocation, time.time())
    outcome = remote_call(pod, '_launch', ['--plan', saved['plan_path'], '--plan-sha256', saved['plan_sha256']])
    common.save_launch(area / 'launch_verified.json', outcome); return outcome


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['stage', 'launch', '_preflight', '_launch'])
    p.add_argument('allocation', nargs='?'); p.add_argument('--partition'); p.add_argument('--shard-id')
    p.add_argument('--secondary-declaration'); p.add_argument('--launch', action='store_true')
    p.add_argument('--plan'); p.add_argument('--plan-sha256'); a = p.parse_args()
    if a.action == 'stage': value = stage(a.allocation, a.partition, a.shard_id, a.secondary_declaration, a.launch)
    elif a.action == 'launch': value = launch_staged(a.allocation)
    elif a.action == '_preflight': value = preflight(a.plan, a.plan_sha256)[2]
    else: value = launch_remote(a.plan, a.plan_sha256)
    print(json.dumps(value))
