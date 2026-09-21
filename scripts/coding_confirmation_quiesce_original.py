#!/usr/bin/env python3
"""Explicit original-supervisor stop and public-only immutable migration snapshot.

No provider calls, pod deletion, model execution, or training mutation. prepare
and inspect are read-only except their new public receipts. stop requires an
explicit --execute; snapshot requires both originals proved absent and both
original release/backups verified. Run snapshot from the surviving training pod.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, digest, sha, write
from gearshift.coding_confirmation_lease import (
    control_root, linux_process_identity, load_lease, verify_release_receipt)
from scripts import coding_confirmation_sharded_generate as sharded

SCRIPT = 'scripts/coding_confirmation_quiesce_original.py'
DECLARATION = 'configs/coding_pilot_v1/confirmation_01/declaration.json'
DECLARATION_SHA = '16387d5361c795121929fd9244821f12ee005f9b36b2febddca5d9ef89f7536f'
EXPERIMENT = 'confirmation_01_20260919T094418Z'
EVIDENCE = 'evidence/coding_pilot_v1/' + EXPERIMENT
ORIGINALS = {'x6jy63vp7jvx6n': EVIDENCE + '/resources/primary_pair01/lease.json',
             'v7ud2o1wx86y9e': EVIDENCE + '/resources/primary_pair02/lease.json'}
TRAINING_POD = 'rekbqruqox2bk9'
SUPERVISOR = 'scripts/coding_confirmation_generation_supervisor.py'
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.bin', '.safetensors', '.npy', '.npz', '.pkl', '.pickle'}
FORBIDDEN_PARTS = {'.git', '.venv', '.pilot-venv', 'private', 'models', 'model_cache', '.cache'}


def read(path):
    return json.loads(Path(path).read_text())


def public_file(root, relative):
    if FORBIDDEN_PARTS & set(Path(relative).parts):
        raise ValueError('Forbidden nonpublic or generated input')
    return sharded.primary.scoped_file(root, relative)


def evidence_output(repo, relative):
    if not str(relative).startswith('evidence/') or FORBIDDEN_PARTS & set(Path(relative).parts):
        raise ValueError('New receipts must be in a public evidence directory')
    return sharded.output(repo, relative)


def checked(repo, row):
    path = public_file(repo, row['path'])
    if sha(path) != row['sha256']:
        raise ValueError('Pinned public input changed')
    return path


def declaration(repo):
    path = public_file(repo, DECLARATION)
    if sha(path) != DECLARATION_SHA:
        raise ValueError('Original declaration changed')
    d = read(path)
    if d['experiment_id'] != EXPERIMENT or d['task_count'] != 200 or d['primary_answer_count'] != 4800:
        raise ValueError('Unexpected original scientific population')
    for name, expected in d['implementation'].items():
        if sha(public_file(repo, name)) != expected:
            raise ValueError('Frozen implementation changed: ' + name)
    return d


def prepare(repo, output):
    repo = Path(repo).resolve(); declaration(repo)
    rows = []; roots = set(); volumes = set()
    for pod, relative in ORIGINALS.items():
        path = public_file(repo, relative); checksum = sha(path)
        lease = load_lease(path, checksum)
        if lease['pod_id'] != pod or lease['experiment_id'] != EXPERIMENT or lease['gpu_count'] != 2:
            raise ValueError('Original allocation identity differs')
        roots.add(lease['allowed_result_root']); volumes.add(lease['network_volume_id'])
        rows.append({'pod_id': pod, 'lease_path': relative, 'lease_sha256': checksum})
    if len(roots) != 1 or len(volumes) != 1:
        raise ValueError('Originals must share exactly one source result store')
    value = {'schema': 1, 'purpose': 'supported_original_primary_quiescence',
             'experiment_id': EXPERIMENT, 'declaration_path': DECLARATION,
             'declaration_sha256': DECLARATION_SHA, 'original_generation_allocations': rows,
             'excluded_training_pod_ids': [TRAINING_POD], 'snapshot_reader_pod_id': TRAINING_POD,
             'original_result_root': next(iter(roots)), 'network_volume_id': next(iter(volumes)),
             'helper_sha256': sha(repo / SCRIPT), 'provider_calls_authorized': False,
             'training_mutation_authorized': False,
             'stop_requires_explicit_execute': True}
    dest = evidence_output(repo, output); bind(dest, value)
    return {'plan_path': str(dest.relative_to(repo)), 'plan_sha256': sha(dest)}


def load_plan(repo, path, expected):
    repo = Path(repo).resolve(); raw = checked(repo, {'path': path, 'sha256': expected})
    p = read(raw); d = declaration(repo)
    if (p.get('schema') != 1 or p.get('purpose') != 'supported_original_primary_quiescence'
            or p.get('experiment_id') != EXPERIMENT or p.get('declaration_sha256') != DECLARATION_SHA
            or p.get('declaration_path') != DECLARATION
            or p.get('excluded_training_pod_ids') != [TRAINING_POD]
            or p.get('snapshot_reader_pod_id') != TRAINING_POD
            or p.get('provider_calls_authorized') is not False
            or p.get('training_mutation_authorized') is not False
            or p.get('stop_requires_explicit_execute') is not True):
        raise ValueError('Quiescence helper authority differs')
    if sha(repo / SCRIPT) != p['helper_sha256'] or sha(Path(__file__)) != p['helper_sha256']:
        raise ValueError('Loaded helper source differs')
    inventory = p['original_generation_allocations']
    if len(inventory) != 2 or {r['pod_id'] for r in inventory} != set(ORIGINALS):
        raise ValueError('Exactly the two original generation allocations are required')
    leases = {}
    for row in inventory:
        if row['lease_path'] != ORIGINALS[row['pod_id']]:
            raise ValueError('Original lease location differs')
        lease = load_lease(public_file(repo, row['lease_path']), row['lease_sha256'])
        if (lease['pod_id'] != row['pod_id'] or lease['experiment_id'] != EXPERIMENT
                or lease['allowed_result_root'] != p['original_result_root']
                or lease['network_volume_id'] != p['network_volume_id'] or lease['gpu_count'] != 2):
            raise ValueError('Original lease and immutable quiescence plan differ')
        leases[row['pod_id']] = lease
    return p, d, leases


def local_pod(proc_root=Path('/proc')):
    value = os.environ.get('RUNPOD_POD_ID')
    if not value:
        value = dict(x.split(b'=', 1) for x in (Path(proc_root) / '1/environ').read_bytes().split(b'\0')
                     if b'=' in x).get(b'RUNPOD_POD_ID', b'').decode()
    return value


def original_repo(lease):
    root = Path(lease['allowed_result_root'])
    if root.parts[-3:] != ('results', 'coding_pilot_v1', EXPERIMENT):
        raise ValueError('Unexpected original result-root geometry')
    return root.parents[2]


def inspect_supervisor(lease, *, proc_root=Path('/proc')):
    """Check registered boot/start identity and exact original supervisor entrypoint."""
    if lease['pod_id'] not in ORIGINALS or local_pod(proc_root) != lease['pod_id']:
        raise ValueError('Stop inspection must run on the named original generation pod')
    repo = original_repo(lease); declaration(repo)
    root = Path(lease['allowed_result_root']); control = control_root(lease)
    registration_path = public_file(root, str((control / 'lease_guard/supervisor_registration.json').relative_to(root)))
    registration = read(registration_path)
    if any(registration.get(k) != lease[k] for k in ['experiment_id', 'pod_id']):
        raise ValueError('Registered supervisor belongs to another allocation')
    pid = registration.get('pid')
    if type(pid) is not int or pid <= 1 or pid in (os.getpid(), os.getpgrp()) or registration.get('pgid') != pid:
        raise ValueError('Unsafe or unowned supervisor process group')
    current = linux_process_identity(pid, proc_root)
    if any(registration.get(k) != current[k] for k in ['pid', 'pgid', 'start_ticks', 'boot_id']):
        raise ValueError('Registered supervisor process identity changed')
    proc_root = Path(proc_root)
    if Path(os.readlink(proc_root / str(pid) / 'cwd')).resolve() != repo.resolve():
        raise ValueError('Supervisor working directory differs')
    argv = [v.decode() for v in (proc_root / str(pid) / 'cmdline').read_bytes().split(b'\0') if v]
    i = 1
    while i < len(argv) and argv[i] in ('-u', '-B'):
        i += 1
    if not argv or not Path(argv[0]).name.startswith('python') or i >= len(argv):
        raise ValueError('Unexpected supervisor invocation')
    script = Path(argv[i]); script = script if script.is_absolute() else repo / script
    if script.resolve() != (repo / SUPERVISOR).resolve() or argv.count('--plan') != 1:
        raise ValueError('Registered process is not the frozen original generation supervisor')
    try:
        plan_arg = argv[argv.index('--plan') + 1]
    except IndexError as exc:
        raise ValueError('Missing original supervisor dispatch') from exc
    dispatch_path = Path(plan_arg)
    if dispatch_path.is_absolute():
        dispatch_path = dispatch_path.relative_to(repo)
    dispatch_path = public_file(repo, str(dispatch_path)); dispatch = read(dispatch_path)
    original_lease_path = public_file(repo, dispatch['lease_path'])
    if (dispatch.get('experiment_id') != EXPERIMENT or dispatch.get('declaration_sha256') != DECLARATION_SHA
            or (repo / dispatch['result_root']).resolve() != root.resolve()
            or sha(original_lease_path) != dispatch['lease_sha256'] or read(original_lease_path) != lease):
        raise ValueError('Supervisor dispatch does not bind this original allocation')
    return {'experiment_id': EXPERIMENT, 'pod_id': lease['pod_id'], **current,
            'registration_sha256': sha(registration_path), 'dispatch_sha256': sha(dispatch_path),
            'supervisor_source_sha256': sha(repo / SUPERVISOR), 'observed_epoch': time.time()}


def signal_verified_pid(lease, before, *, proc_root=Path('/proc')):
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support is required; unsafe PID-only fallback refused')
    fd = os.pidfd_open(before['pid'], 0)
    try:
        after = inspect_supervisor(lease, proc_root=proc_root)
        if any(after[k] != before[k] for k in ['pid', 'pgid', 'start_ticks', 'boot_id', 'registration_sha256', 'dispatch_sha256']):
            raise ValueError('Supervisor changed before signal')
        signal.pidfd_send_signal(fd, signal.SIGTERM, None, 0)
    finally:
        os.close(fd)


def stop(repo, plan_path, plan_sha, pod_id, output, *, execute=False):
    p, _, leases = load_plan(repo, plan_path, plan_sha)
    if pod_id not in ORIGINALS or pod_id == TRAINING_POD:
        raise ValueError('Only the two original generation supervisors may be stopped')
    if not execute:
        raise ValueError('Explicit stop --execute is required')
    before = inspect_supervisor(leases[pod_id])
    folder = evidence_output(repo, output); folder.mkdir(parents=True, exist_ok=False)
    bind(folder / 'stop_request.json', {'schema': 1, 'plan_sha256': plan_sha,
         'verified_supervisor': before, 'signal': 'SIGTERM', 'target': 'supervisor_pid_only',
         'workers_backup_and_release_owned_by_original_supervisor': True,
         'training_pod_excluded': TRAINING_POD, 'automatic_retry_authorized': False})
    signal_verified_pid(leases[pod_id], before)
    result = {'pod_id': pod_id, 'supervisor_pid': before['pid'], 'signal_sent': True,
              'epoch': time.time(), 'provider_absence_not_yet_proved': True}
    bind(folder / 'signal_sent.json', result)
    return result


def verify_absence(repo, index_path, index_sha, leases):
    index = read(checked(repo, {'path': index_path, 'sha256': index_sha}))
    rows = index.get('receipts', [])
    if len(rows) != 2 or {r['pod_id'] for r in rows} != set(ORIGINALS):
        raise ValueError('Provider absence evidence must cover exactly both originals')
    result = {}
    for row in rows:
        path = checked(repo, row); value = read(path); epoch = value.get('observed_epoch')
        if (value.get('pod_id') != row['pod_id'] or value.get('source') != 'runpod_provider_inspection'
                or value.get('state') != 'provider_confirmed_absent'
                or type(epoch) not in (int, float) or not math.isfinite(epoch)
                or not leases[row['pod_id']]['allocation_epoch'] <= epoch <= time.time() + 5):
            raise ValueError('Invalid provider absence observation')
        method = value.get('verification_method')
        if method == 'get_pod_404':
            valid = value.get('http_status') == 404
        elif method == 'complete_pod_list':
            listed = value.get('listed_pod_ids')
            valid = (value.get('http_status') == 200 and value.get('complete_inventory') is True
                     and isinstance(listed, list) and all(isinstance(p, str) for p in listed)
                     and len(set(listed)) == len(listed) and row['pod_id'] not in listed)
        else:
            valid = False
        if not valid:
            raise ValueError('Provider request did not prove absence')
        result[row['pod_id']] = path
    return result


def file_row(root, path):
    relative = str(Path(path).relative_to(root)); path = public_file(root, relative)
    if not stat.S_ISREG(path.stat().st_mode) or set(s.lower() for s in path.suffixes) & FORBIDDEN_SUFFIXES:
        raise ValueError('Noncompact artifact in public snapshot')
    return {'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)}


def collect_tree(root, directory):
    root, directory = Path(root), Path(directory)
    result = set()
    if not directory.exists():
        return result
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ValueError('Linked source artifact refused')
        if path.is_file() and not path.name.endswith('.lock'):
            file_row(root, path); result.add(path)
    return result


def snapshot_files(root, d):
    root = Path(root)
    main = collect_tree(root, root / 'primary/tasks') | collect_tree(root, root / 'primary/jobs')
    for path in (root / 'claims').glob('*.owner.json'):
        file_row(root, path); main.add(path)
    touched = {sharded.task_from_path(str(p.relative_to(root)), d['task_ids']) for p in main}
    dependencies = set()
    for pod in ORIGINALS:
        for directory in (root / 'workers').glob(pod + '_generation_*'):
            dependencies.update(collect_tree(root, directory))
    for path in (root / 'primary/jobs').glob('*/complete.json'):
        job = read(path)
        if job.get('declaration_sha256') != DECLARATION_SHA or job.get('experiment_id') != EXPERIMENT:
            raise ValueError('Completed job is outside the frozen primary experiment')
        runtime_path = public_file(root, job['runtime_path'])
        setup_path = public_file(root, job['setup_path'])
        rel = runtime_path.relative_to(root)
        if (len(rel.parts) < 4 or rel.parts[0] != 'workers'
                or not any(rel.parts[1].startswith(pod + '_generation_') for pod in ORIGINALS)
                or runtime_path.name != 'runtime.json' or setup_path != runtime_path.parent / 'model_setup.json'
                or digest(read(runtime_path)) != job['runtime_identity_sha256']):
            raise ValueError('Original completed-job runtime/setup dependencies differ')
        setup = read(setup_path)
        if setup.get('declaration_sha256') != DECLARATION_SHA or setup.get('runtime_identity_sha256') != job['runtime_identity_sha256']:
            raise ValueError('Original model setup identity differs')
        dependencies.update([runtime_path, setup_path])
    # Owner history is audit evidence, not an input to initial_dataset_verified.
    dependencies.update(collect_tree(root, root / 'claims/history'))
    return ([file_row(root, p) for p in sorted(main)],
            [file_row(root, p) for p in sorted(dependencies - main)],
            [tid for tid in d['task_ids'] if tid in touched])


def copy_exact(source, destination):
    source, destination = Path(source), Path(destination)
    expected = sha(source); size = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as inp, destination.open('xb') as out:
        while block := inp.read(1024 * 1024):
            out.write(block)
        out.flush(); os.fsync(out.fileno())
    if destination.stat().st_size != size or sha(destination) != expected or sha(source) != expected:
        raise ValueError('Public provenance changed during copy')
    directory = os.open(destination.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return {'path': str(destination), 'bytes': size, 'sha256': expected}


def make_archive(root, path, rows):
    path = Path(path)
    with path.open('xb') as handle:
        with tarfile.open(fileobj=handle, mode='w:gz', compresslevel=1, dereference=True) as archive:
            for row in rows:
                source = public_file(root, row['path'])
                if source.stat().st_size != row['bytes'] or sha(source) != row['sha256']:
                    raise ValueError('Original dataset changed before snapshot copy')
                archive.add(source, arcname=row['path'], recursive=False)
            data = json.dumps({'schema': 1, 'files': rows}, sort_keys=True).encode()
            info = tarfile.TarInfo('SNAPSHOT_MANIFEST.json'); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        handle.flush(); os.fsync(handle.fileno())
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        if {m.name for m in members} != {r['path'] for r in rows} | {'SNAPSHOT_MANIFEST.json'} or len(members) != len(rows) + 1:
            raise ValueError('Archive membership differs')
        for row in rows:
            member = archive.getmember(row['path'])
            if not member.isfile() or member.size != row['bytes']:
                raise ValueError('Archive member geometry differs')
            h = hashlib.sha256()
            with archive.extractfile(member) as source:
                while block := source.read(1024 * 1024): h.update(block)
            if h.hexdigest() != row['sha256'] or sha(public_file(root, row['path'])) != row['sha256']:
                raise ValueError('Archive or original source changed during snapshot')


def snapshot(repo, plan_path, plan_sha, absence_path, absence_sha, output):
    repo = Path(repo).resolve(); p, d, leases = load_plan(repo, plan_path, plan_sha)
    if local_pod() != TRAINING_POD:
        raise ValueError('Snapshot must run on the surviving training pod, without altering it')
    absence = verify_absence(repo, absence_path, absence_sha, leases)
    original = Path(p['original_result_root']); release_inputs = {}
    for pod, lease in leases.items():
        if not verify_release_receipt(lease):
            raise ValueError('Original verified release is missing')
        control = control_root(lease)
        stop_path = public_file(original, str((control / 'generation_supervisor/workers_stopped.json').relative_to(original)))
        stop_record = read(stop_path)
        if stop_record.get('all_registered_workers_stopped') is not True or stop_record.get('process_group_ownership_verified') is not True:
            raise ValueError('Original worker ownership/stop proof is incomplete')
        release_path = control / 'gpu_release_verified.json'; release = read(release_path)
        manifest_path = public_file(original, release['manifest_path']); manifest = read(manifest_path)
        matched = [r for r in manifest['files'] if r['path'] == str(stop_path.relative_to(original))]
        if len(matched) != 1 or matched[0]['sha256'] != sha(stop_path):
            raise ValueError('Released allocation does not bind its own worker stop proof')
        release_inputs[pod] = {'workers_stopped': stop_path, 'gpu_release_verified': release_path,
                               'release_manifest': manifest_path}
    files, dependencies, touched = snapshot_files(original, d)
    if not files:
        raise ValueError('Unexpected empty original primary dataset')
    folder = evidence_output(repo, output); folder.mkdir(parents=True, exist_ok=False)
    archive = folder / 'original_primary_snapshot.tar.gz'
    all_rows = sorted(files + dependencies, key=lambda row: row['path'])
    make_archive(original, archive, all_rows)
    # Re-enumeration detects additions/removals, not only changed existing bytes.
    if snapshot_files(original, d) != (files, dependencies, touched):
        raise ValueError('Original task population changed while snapshotting')
    allocations, receipts, absence_copies = [], [], []
    for row in p['original_generation_allocations']:
        pod = row['pod_id']; prefix = folder / 'provenance' / pod
        lease_copy = prefix / 'lease.json'
        copy_exact(public_file(repo, row['lease_path']), lease_copy)
        allocations.append({'pod_id': pod, 'lease_path': str(lease_copy.relative_to(repo)),
                            'lease_sha256': sha(lease_copy)})
        items = {}
        for key, source in release_inputs[pod].items():
            target = prefix / (key + '.json'); copy_exact(source, target)
            items[key] = {'path': str(target.relative_to(repo)), 'sha256': sha(target)}
        absence_copy = prefix / 'provider_absence.json'; copy_exact(absence[pod], absence_copy)
        absence_copies.append({'pod_id': pod, 'path': str(absence_copy.relative_to(repo)),
                              'sha256': sha(absence_copy)})
        receipts.append({'pod_id': pod, 'files': items})
    copy_exact(public_file(repo, plan_path), folder / 'control_plan.json')
    copy_exact(checked(repo, {'path': absence_path, 'sha256': absence_sha}), folder / 'input_provider_absence_index.json')
    bind(folder / 'provider_absence_index.json', {'receipts': absence_copies})
    q = {'schema': 1, 'experiment_id': EXPERIMENT, 'declaration_sha256': DECLARATION_SHA,
         'all_original_primary_workers_stopped': True, 'original_supervisors_released': True,
         'healthy_training_untouched': True, 'files': files, 'touched_task_ids': touched,
         'allocation_receipts': receipts}
    bind(folder / 'quiescence.json', q)
    fragment = {'original_generation_allocations': allocations, 'excluded_training_pod_ids': [TRAINING_POD],
                'quiescence_path': str((folder / 'quiescence.json').relative_to(repo)),
                'quiescence_sha256': sha(folder / 'quiescence.json'), 'touched_task_ids': touched}
    bind(folder / 'partition_inventory_fragment.json', fragment)
    provenance = [file_row(repo, path) for path in sorted(folder.rglob('*.json'))]
    receipt = {'schema': 1, 'experiment_id': EXPERIMENT, 'plan_sha256': plan_sha,
               'original_result_root': str(original), 'snapshot_reader_pod_id': TRAINING_POD,
               'original_source_unchanged_after_copy': True, 'no_training_files_written': True,
               'no_private_inputs_read': True, 'original_pods_provider_absent': list(ORIGINALS),
               'quiescence_path': fragment['quiescence_path'], 'quiescence_sha256': fragment['quiescence_sha256'],
               'archive': {'path': str(archive.relative_to(repo)), 'sha256': sha(archive), 'bytes': archive.stat().st_size},
               'files': all_rows, 'provenance_files': provenance,
               'task_file_count': len(files), 'dependency_file_count': len(dependencies),
               'touched_task_count': len(touched), 'completed_epoch': time.time()}
    bind(folder / 'SNAPSHOT_COMPLETE.json', receipt)
    return {k: receipt[k] for k in ['archive', 'quiescence_path', 'quiescence_sha256', 'task_file_count',
                                    'dependency_file_count', 'touched_task_count']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'inspect', 'stop', 'snapshot'])
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--plan'); parser.add_argument('--plan-sha256')
    parser.add_argument('--pod-id'); parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--absence-index'); parser.add_argument('--absence-index-sha256')
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.repo, args.output)
    elif args.action == 'inspect':
        _, _, leases = load_plan(args.repo, args.plan, args.plan_sha256)
        if args.pod_id not in ORIGINALS:
            raise ValueError('Only original generation supervisors can be inspected')
        result = inspect_supervisor(leases[args.pod_id])
    elif args.action == 'stop':
        result = stop(args.repo, args.plan, args.plan_sha256, args.pod_id, args.output, execute=args.execute)
    else:
        result = snapshot(args.repo, args.plan, args.plan_sha256, args.absence_index,
                          args.absence_index_sha256, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
