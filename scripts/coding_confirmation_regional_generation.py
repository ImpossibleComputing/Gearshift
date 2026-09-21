#!/usr/bin/env python3
"""Byte-bound public staging and explicit detached launch for regional task shards.

This console has no lease authority, never provisions, and never transfers model
weights, private test bytes or candidate state. Initial origin-state migration is
an independent verified operation. A lost launch response is reconciled, not retried.
"""
import argparse
import ast
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REMOTE = '/workspace/GearshiftConfirmationPrimary'
SELF = 'scripts/coding_confirmation_regional_generation.py'
CONFIG = 'configs/coding_pilot_v1/confirmation_01'
DECLARATION = CONFIG + '/declaration.json'
ADAPTER = CONFIG + '/sharded_adapter_source.json'
SUPERVISOR = 'scripts/coding_confirmation_sharded_supervisor.py'


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    from gearshift.coding_confirmation_lease import sha256_file
    return sha256_file(path)


def public_file(repo, relative):
    from scripts.coding_confirmation_generate import scoped_file
    p = Path(relative)
    if (any(x.lower() in {'private', 'hidden_tests', 'reference_answers', '.git', '.ssh', '.runpod', '.venv', '.pilot-venv'} for x in p.parts)
            or p.suffix.lower() not in {'.py', '.json', '.md', '.txt', '.toml', '.yaml', '.yml', '.csv'}
            or p.name.startswith('.env')):
        raise ValueError('Nonpublic or heavyweight staging path: ' + str(relative))
    return scoped_file(repo, relative)


def bind_file(path, data):
    path = Path(path)
    for p in [path, *path.parents]:
        if p.is_symlink(): raise ValueError('Linked staging destination')
    if path.exists():
        if not path.is_file() or path.read_bytes() != data: raise ValueError('Existing staged bytes differ: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: a competing stage cannot overwrite an existing file.
    try:
        with path.open('xb') as out: out.write(data); out.flush(); os.fsync(out.fileno())
    except FileExistsError:
        if path.read_bytes() != data: raise ValueError('Concurrent staging bytes differ')


def bind_json(path, value):
    bind_file(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def dependencies(repo, paths):
    """Follow local static imports, including legacy bare script imports."""
    found = set(paths); pending = [p for p in paths if p.endswith('.py')]
    while pending:
        rel = pending.pop(); tree = ast.parse(public_file(repo, rel).read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): modules.extend(x.name for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    parent = list(Path(rel).parent.parts)
                    base = '.'.join(parent[:len(parent)-node.level+1] + ([base] if base else []))
                modules += [base, *[base + '.' + x.name for x in node.names]]
        for module in modules:
            candidates = [module.replace('.', '/') + '.py', module.replace('.', '/') + '/__init__.py']
            if '.' not in module: candidates.append('scripts/' + module + '.py')
            for candidate in candidates:
                if candidate.split('/')[0] not in ('scripts', 'gearshift') or not (repo / candidate).is_file(): continue
                if candidate not in found: found.add(candidate); pending.append(candidate)
    return found


def source_paths(repo, declaration_path, adapter_path, extra):
    from scripts.coding_confirmation_generate import validate_declaration
    from scripts.coding_confirmation_sharded_generate import IMPLEMENTATION
    d = validate_declaration(repo, declaration_path, sha(repo / declaration_path)); source = read(repo / adapter_path)
    if (set(source['files']) != IMPLEMENTATION or source.get('numerical_primary_code_changed') is not False or
            source.get('analysis_population_or_seed_changed') is not False): raise ValueError('Adapter source manifest differs')
    pins = {**d['inputs'], **d['implementation']}
    for rel, value in source['files'].items():
        if rel in pins and pins[rel] != value: raise ValueError('Adapter conflicts with frozen numerical source')
        pins[rel] = value
    paths = dependencies(repo, set(pins) | {declaration_path, adapter_path, SELF} | set(extra))
    helper_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    for rel in sorted(paths):
        p = public_file(repo, rel)
        if rel in pins:
            if sha(p) != pins[rel]: raise ValueError('Pinned source/input changed: ' + rel)
        elif rel.endswith('.py'):
            commit = helper_commit if rel == SELF else source['operational_commit']
            original = subprocess.check_output(['git', 'show', commit + ':' + rel], cwd=repo)
            if p.read_bytes() != original: raise ValueError('Commit/review operational dependency first: ' + rel)
    return d, source, paths, helper_commit


def inspect_provider(pod, lease, allocation, now):
    from gearshift.coding_confirmation_lease import verify_lease
    verify_lease(lease)
    if (pod.get('id') != lease['pod_id'] or pod.get('name') != 'gearshift-confirmation-' + allocation or
            pod.get('status') != 'RUNNING' or pod.get('gpu', {}).get('id') != 'NVIDIA H200' or
            pod.get('gpu', {}).get('count') != lease['gpu_count'] or pod.get('cpu') is not None or
            pod.get('mounts', {}).get('network') != [{'volumeId': lease['network_volume_id'], 'path': '/workspace'}] or
            lease['allowed_result_root'] != REMOTE + '/results/coding_pilot_v1/' + lease['experiment_id'] or
            not lease['allocation_epoch'] <= now < lease['deadline_epoch'] - 120):
        raise ValueError('Live provider allocation/mount/lease identity differs')
    return {'pod_id': lease['pod_id'], 'network_volume_id': lease['network_volume_id'],
        'volume_mount_path': '/workspace', 'observed_epoch': now, 'source': 'runpod_provider_inspection',
        'provider_gpu_count': lease['gpu_count'], 'provider_status': 'RUNNING'}


def all_device_environment(lease):
    from scripts.coding_confirmation_replication_supervisor import child_environment
    env = child_environment(lease, 0); env.pop('CUDA_VISIBLE_DEVICES', None)
    return env


def remote_inspect(lease_path, expected):
    from gearshift.coding_confirmation_lease import load_lease, control_root
    from scripts.coding_confirmation_generation_supervisor import validate_guard
    from scripts.coding_confirmation_sharded_generate import workspace_mount
    lease = load_lease(public_file(ROOT, lease_path), expected)
    if str(ROOT) != REMOTE: raise ValueError('Remote repository root differs')
    control = control_root(lease); guard = validate_guard(lease, control)
    if b'coding_confirmation_lease.py' not in Path('/proc/' + str(guard['pid']) + '/cmdline').read_bytes():
        raise ValueError('Guard PID no longer belongs to the lease guard')
    if time.time() >= lease['deadline_epoch'] - 120 or (control / 'lease_guard/STOP').exists() or (control / 'gpu_release_verified.json').exists():
        raise ValueError('Allocation has stopped or entered release')
    mount = workspace_mount()
    if mount['filesystem_type'] in ('overlay', 'tmpfs', 'ramfs'): raise ValueError('Workspace is not durable network storage')
    existing = Path(lease['allowed_result_root'])
    while not existing.exists(): existing = existing.parent
    if existing.is_symlink() or str(os.major(existing.stat().st_dev)) + ':' + str(os.minor(existing.stat().st_dev)) != mount['device']:
        raise ValueError('Result root device differs from workspace mount')
    command = [str(ROOT / '.pilot-venv/bin/python'), '-c',
        'import json,torch;print(json.dumps([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))']
    p = subprocess.run(command, env=all_device_environment(lease), capture_output=True, text=True, timeout=90, check=True)
    gpus = json.loads(p.stdout)
    if len(gpus) != lease['gpu_count'] or not gpus or any('H200' not in x for x in gpus):
        raise ValueError('Actual all-device H200 count differs from allocation')
    return {'pod_id': lease['pod_id'], 'local_mount': mount, 'visible_gpu_names': gpus,
        'visible_gpu_count': len(gpus), 'guard_pid': guard['pid'], 'observed_epoch': time.time()}


def unpack(repo, stream, manifest_path, manifest_sha):
    """Two-pass archive validation prevents unchecked members from being written."""
    with tarfile.open(fileobj=stream, mode='r:gz') as archive:
        members = archive.getmembers(); names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members): raise ValueError('Invalid archive member')
        if manifest_path not in names: raise ValueError('Stage manifest missing')
        data = archive.extractfile(manifest_path).read()
        if hashlib.sha256(data).hexdigest() != manifest_sha: raise ValueError('Stage manifest hash differs')
        manifest = json.loads(data)
        if set(names) != set(manifest['files']) | {manifest_path}: raise ValueError('Archive contains unlisted files')
        for m in members:
            rel = Path(m.name)
            if rel.is_absolute() or '..' in rel.parts or not rel.parts: raise ValueError('Unsafe archive path')
            # A path can be absent remotely, so validate the public name separately.
            if any(x.lower() in {'private', 'hidden_tests', 'reference_answers', '.git', '.ssh', '.runpod', '.venv', '.pilot-venv'} for x in rel.parts) or rel.suffix.lower() not in {'.py','.json','.md','.txt','.toml','.yaml','.yml','.csv'} or rel.name.startswith('.env'):
                raise ValueError('Nonpublic archive entry')
            payload = archive.extractfile(m).read()
            if m.name != manifest_path:
                want = manifest['files'][m.name]
                if len(payload) != want['bytes'] or hashlib.sha256(payload).hexdigest() != want['sha256']: raise ValueError('Archive bytes differ')
            target = repo / rel
            for p in [target, *target.parents]:
                if p.is_symlink(): raise ValueError('Linked archive destination')
            if target.exists() and target.read_bytes() != payload: raise ValueError('Remote immutable bytes differ: ' + m.name)
        for m in members: bind_file(repo / m.name, archive.extractfile(m).read())
    return {'files_verified': len(manifest['files']), 'manifest_sha256': manifest_sha}


def upload(pod, paths, folder, label):
    from scripts import coding_confirmation_dispatch as base
    manifest = {'schema': 1, 'files': {p: {'sha256': sha(public_file(ROOT,p)), 'bytes': public_file(ROOT,p).stat().st_size} for p in sorted(paths)}}
    mp = folder / (label + '_manifest.json'); bind_json(mp, manifest); packet = folder / (label + '.tar.gz')
    with tarfile.open(packet, 'w:gz') as out:
        for rel in sorted(set(paths) | {str(mp.relative_to(ROOT))}): out.add(ROOT / rel, arcname=rel, recursive=False)
    # Installer source travels over encrypted stdin as a public code string; no auth config is packaged.
    script = 'import io,json,os,pathlib,sys;process_env=dict(x.split(b"=",1) for x in pathlib.Path("/proc/1/environ").read_bytes().split(b"\\0") if b"=" in x);assert (os.environ.get("RUNPOD_POD_ID") or process_env.get(b"RUNPOD_POD_ID",b"").decode())=='+repr(pod['id'])+';ns={"__name__":"stager","__file__":'+repr(REMOTE+'/'+SELF)+'};exec('+repr(Path(__file__).read_text())+',ns);print(json.dumps(ns["unpack"](pathlib.Path('+repr(REMOTE)+'),io.BytesIO(sys.stdin.buffer.read()),'+repr(str(mp.relative_to(ROOT)))+','+repr(sha(mp))+')))'
    with packet.open('rb') as stream:
        r = subprocess.run(base.ssh_args(pod) + ['python3 -c ' + shlex.quote(script)], stdin=stream, capture_output=True, timeout=120, check=True)
    return json.loads(r.stdout)


def remote_call(pod, action, values):
    from scripts import coding_confirmation_dispatch as base
    command = ['.pilot-venv/bin/python', SELF, action, *values]
    r = subprocess.run(base.ssh_args(pod) + ['cd ' + shlex.quote(REMOTE) + ' && ' + shlex.join(command)],
        capture_output=True, text=True, timeout=180, check=True)
    return json.loads(r.stdout)


def remote_preflight(plan_path, plan_sha):
    from scripts import coding_confirmation_sharded_generate as sharded
    c = sharded.public_context(ROOT, plan_path, plan_sha); lease, _ = sharded.validate_generation_store(c, verify_local=True)
    actual = remote_inspect(c['plan']['lease_path'], c['plan']['lease_sha256'])
    if len(c['plan']['preferences']) != lease['gpu_count'] or any(p not in ('source','small','receiver') for p in c['plan']['preferences']):
        raise ValueError('GPU preferences differ from lease')
    # Existing mapper files were transferred separately; this only verifies bytes.
    for cp in c['declaration']['primary_checkpoints'].values():
        p = sharded.primary.scoped_file(ROOT, cp['mapper_path'])
        if p.stat().st_size != cp['mapper_bytes'] or sha(p) != cp['mapper_sha256']: raise ValueError('Final mapper bytes differ')
    sharded.verify_initial_dataset(c)
    return c, lease, actual


def verify_launch_process(lease, plan_path, plan_sha, identity, proc_root=Path('/proc')):
    from gearshift.coding_confirmation_lease import control_root, linux_process_identity
    current = linux_process_identity(identity['pid'], proc_root)
    if current != identity or current['pgid'] != current['pid']: raise ValueError('Supervisor process identity changed')
    command = (proc_root / str(identity['pid']) / 'cmdline').read_bytes().split(b'\0')
    if not all(x.encode() in command for x in [SUPERVISOR, plan_path, plan_sha]): raise ValueError('Supervisor command identity differs')
    registration = read(control_root(lease) / 'lease_guard/supervisor_registration.json')
    if any(registration.get(k) != v for k,v in {**identity, 'experiment_id':lease['experiment_id'], 'pod_id':lease['pod_id']}.items()):
        raise ValueError('Guard supervisor registration differs')
    return True


def remote_launch(plan_path, plan_sha):
    from gearshift.coding_confirmation_lease import control_root, linux_process_identity
    c, lease, actual = remote_preflight(plan_path, plan_sha); control = control_root(lease)
    with (control / 'sharded_dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent = control / 'sharded_launch_intent.json'; receipt = control / 'sharded_launch_verified.json'
        if receipt.exists():
            saved = read(receipt)
            if saved['plan_path'] != plan_path or saved['plan_sha256'] != plan_sha: raise ValueError('Allocation already launched a different dispatch')
            verify_launch_process(lease, plan_path, plan_sha, saved['process_identity'])
            return {**saved, 'already_running': True}
        if intent.exists() or (control / 'sharded_generation_supervisor').exists():
            raise ValueError('Prior launch is ambiguous or finished; reconcile, never duplicate launch')
        registration = control / 'lease_guard/supervisor_registration.json'
        if registration.exists(): raise ValueError('An existing supervisor registration requires explicit reconciliation')
        bind_json(intent, {'plan_path': plan_path, 'plan_sha256': plan_sha, 'epoch': time.time(), 'pod_id': lease['pod_id']})
        with (control / 'sharded_launcher.log').open('ab') as log:
            proc = subprocess.Popen([str(ROOT / '.pilot-venv/bin/python'), SUPERVISOR, '--plan', plan_path, '--plan-sha256', plan_sha],
                cwd=ROOT, env=all_device_environment(lease), stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        identity = linux_process_identity(proc.pid)
        bind_json(control / 'sharded_launch_started.json', {'process_identity': identity, 'plan_sha256': plan_sha})
        for _ in range(60):
            if proc.poll() is not None: raise RuntimeError('Supervisor exited during launch; preserve logs and reconcile')
            if registration.exists():
                verify_launch_process(lease, plan_path, plan_sha, identity)
                value = {'plan_path': plan_path, 'plan_sha256': plan_sha, 'process_identity': identity,
                    'pod_id': lease['pod_id'], 'guard_registered': True, 'epoch': time.time(), 'preflight': actual}
                bind_json(receipt, value); return value
            time.sleep(.5)
        raise TimeoutError('Launch registration not yet verified; preserve intent and reconcile without relaunch')


def save_launch(path, outcome):
    if Path(path).exists():
        old = read(path)
        if any(old.get(k) != outcome.get(k) for k in ('plan_path','plan_sha256','pod_id','process_identity')):
            raise ValueError('Existing launch receipt differs')
    else: bind_json(path,outcome)


def stage(allocation, partition_path, shard_id, preferences=None, launch=False):
    from scripts import coding_confirmation_dispatch as base
    from scripts import coding_confirmation_sharded_generate as sharded
    from gearshift.coding_confirmation_lease import load_lease
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,99}', allocation): raise ValueError('Unsafe allocation')
    folder = base.EVIDENCE / allocation; leasepath = str((folder / 'lease.json').relative_to(ROOT))
    lease = load_lease(public_file(ROOT, leasepath), sha(ROOT / leasepath))
    partition = read(public_file(ROOT, partition_path)); q = sharded.checked(ROOT, partition['quiescence_path'], partition['quiescence_sha256'])
    prefs = preferences or [('source','small','receiver')[i % 3] for i in range(lease['gpu_count'])]
    if len(prefs) != lease['gpu_count'] or any(x not in ('source','small','receiver') for x in prefs): raise ValueError('One valid preference per GPU required')
    if partition['shard_storage'].get(shard_id) != {k: lease[k] for k in ('network_volume_id','allowed_result_root')}:
        raise ValueError('Partition does not assign this exact regional store')
    extra = {leasepath, partition_path, partition['quiescence_path'], *[x['lease_path'] for x in partition['original_generation_allocations']],
        *[x['path'] for row in q['allocation_receipts'] for x in row['files'].values()]}
    d, source, paths, helper_commit = source_paths(ROOT, DECLARATION, ADAPTER, extra)
    sharded.validate_partition(d, partition, q)
    if partition['declaration_sha256'] != sha(ROOT / DECLARATION): raise ValueError('Partition declaration differs')
    for row in partition['original_generation_allocations']: sharded.checked(ROOT,row['lease_path'],row['lease_sha256'])
    for row in q['allocation_receipts']:
        for p in row['files'].values(): sharded.checked(ROOT,p['path'],p['sha256'])
    area = folder / 'regional_generation'; area.mkdir(exist_ok=True)
    request = {'allocation':allocation,'partition_path':partition_path,'partition_sha256':sha(ROOT/partition_path),
        'shard_id':shard_id,'preferences':prefs,'lease_sha256':sha(ROOT/leasepath),'adapter_source_manifest_sha256':sha(ROOT/ADAPTER)}
    bind_json(area / 'request.json', request)
    complete = area / 'stage_complete.json'
    if complete.exists():
        saved = read(complete); planpath = saved['plan_path']; plan = read(public_file(ROOT,planpath))
        if sha(ROOT/planpath) != saved['plan_sha256']: raise ValueError('Staged plan changed')
        pod = base.api('pods/'+lease['pod_id']); inspect_provider(pod,lease,allocation,time.time())
        remote_call(pod,'_preflight',['--plan',planpath,'--plan-sha256',saved['plan_sha256']])
    else:
        attempt = area / ('attempt_' + str(time.time_ns())); attempt.mkdir()
        pod = base.api('pods/'+lease['pod_id']); observation = inspect_provider(pod,lease,allocation,time.time())
        op = attempt/'provider_inspection.json'; bind_json(op,observation); paths.add(str(op.relative_to(ROOT)))
        upload(pod,paths,attempt,'public_inputs')
        actual = remote_call(pod,'_inspect',['--lease',leasepath,'--lease-sha256',sha(ROOT/leasepath)])
        if actual['pod_id'] != lease['pod_id'] or actual['visible_gpu_count'] != lease['gpu_count']: raise ValueError('Remote inspection differs')
        staged = time.time()
        if staged-observation['observed_epoch'] > 300: raise ValueError('Stage inspection expired; preserve attempt and stage fresh')
        mount = {'schema':1,'kind':'verified_generation_store','experiment_id':lease['experiment_id'],'pod_id':lease['pod_id'],
            **partition['shard_storage'][shard_id],'mount_path':'/workspace','provider_mount_verified':True,
            'provider_inspected_epoch':observation['observed_epoch'],'staged_epoch':staged,'local_mount':actual['local_mount'],
            'provider_inspection_path':str(op.relative_to(ROOT)),'provider_inspection_sha256':sha(op),
            'actual_visible_gpu_count':actual['visible_gpu_count']}
        mp=attempt/'provider_mount.json';bind_json(mp,mount)
        plan={'execution_mode':'generation','pod_id':lease['pod_id'],'experiment_id':d['experiment_id'],'code_commit':d['source_commit'],
            'result_root':str(Path(lease['allowed_result_root']).relative_to(REMOTE)),'declaration_path':DECLARATION,'declaration_sha256':sha(ROOT/DECLARATION),
            'lease_path':leasepath,'lease_sha256':sha(ROOT/leasepath),'partition_path':partition_path,'partition_sha256':sha(ROOT/partition_path),
            'adapter_source_manifest_path':ADAPTER,'adapter_source_manifest_sha256':sha(ROOT/ADAPTER),'shard_id':shard_id,'preferences':prefs,
            'provider_mount_receipt_path':str(mp.relative_to(ROOT)),'provider_mount_receipt_sha256':sha(mp),
            'stage_helper_commit':helper_commit,'stage_helper_sha256':sha(ROOT/SELF)}
        pp=attempt/'generation_plan.json';bind_json(pp,plan);planpath=str(pp.relative_to(ROOT))
        upload(pod,{str(mp.relative_to(ROOT)),planpath},attempt,'dispatch')
        proof=remote_call(pod,'_preflight',['--plan',planpath,'--plan-sha256',sha(pp)])
        saved={'plan_path':planpath,'plan_sha256':sha(pp),'preflight':proof,'generation_launched':False}
        bind_json(complete,saved)
    if launch:
        outcome=remote_call(pod,'_launch',['--plan',planpath,'--plan-sha256',saved['plan_sha256']])
        save_launch(area/'launch_verified.json',outcome);return outcome
    return saved


def launch_staged(allocation):
    from scripts import coding_confirmation_dispatch as base
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,99}',allocation): raise ValueError('Unsafe allocation')
    area=base.EVIDENCE/allocation/'regional_generation';saved=read(area/'stage_complete.json')
    plan=read(public_file(ROOT,saved['plan_path']))
    if sha(ROOT/saved['plan_path']) != saved['plan_sha256']: raise ValueError('Staged plan changed')
    lease=read(public_file(ROOT,plan['lease_path']));pod=base.api('pods/'+lease['pod_id']);inspect_provider(pod,lease,allocation,time.time())
    outcome=remote_call(pod,'_launch',['--plan',saved['plan_path'],'--plan-sha256',saved['plan_sha256']])
    save_launch(area/'launch_verified.json',outcome)
    return outcome


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','launch','_inspect','_preflight','_launch']);p.add_argument('allocation',nargs='?')
    p.add_argument('--partition');p.add_argument('--shard-id');p.add_argument('--preferences');p.add_argument('--launch',action='store_true')
    p.add_argument('--lease');p.add_argument('--lease-sha256');p.add_argument('--plan');p.add_argument('--plan-sha256');a=p.parse_args()
    if a.action=='stage': result=stage(a.allocation,a.partition,a.shard_id,a.preferences.split(',') if a.preferences else None,a.launch)
    elif a.action=='launch': result=launch_staged(a.allocation)
    elif a.action=='_inspect': result=remote_inspect(a.lease,a.lease_sha256)
    elif a.action=='_preflight': result=remote_preflight(a.plan,a.plan_sha256)[2]
    else: result=remote_launch(a.plan,a.plan_sha256)
    print(json.dumps(result))

if __name__=='__main__': main()
