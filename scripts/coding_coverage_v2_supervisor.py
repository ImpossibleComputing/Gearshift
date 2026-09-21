#!/usr/bin/env python3
"""Detached, on-pod coverage execution with no controller or provider dependency."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gearshift.coding_coverage_v2_lease import (
    SAFE_ID, atomic_json, control_root, linux_process_identity, sha256_file, verify_lease,
    verify_release_receipt,
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def scoped(repo, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('plan paths must be repository-relative')
    result = (Path(repo) / path).resolve()
    if not result.is_relative_to(Path(repo).resolve()):
        raise ValueError('plan path leaves repository')
    return result


def child_environment(gpu):
    """Preserve numerical/runtime settings while dropping credential variables."""
    denied = ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'AUTH', 'CREDENTIAL', 'COOKIE')
    environment = {key: value for key, value in os.environ.items()
                   if not any(word in key.upper() for word in denied)}
    environment['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
    environment['HF_HUB_OFFLINE'] = '1'
    environment['TRANSFORMERS_OFFLINE'] = '1'
    environment['HF_DATASETS_OFFLINE'] = '1'
    environment['PYTHONUNBUFFERED'] = '1'
    # TOKENIZERS_PARALLELISM is numerical/runtime configuration, not a token.
    environment['TOKENIZERS_PARALLELISM'] = 'false'
    return environment


def verify_plan(repo, plan, lease):
    verify_lease(lease)
    if plan.get('experiment_id') != lease['experiment_id']:
        raise ValueError('plan/lease experiment identity differs')
    root = scoped(repo, plan['result_root'])
    if root != Path(lease['allowed_result_root']).resolve():
        raise ValueError('plan result root differs from lease scope')
    if sha256_file(scoped(repo, plan['declaration_path'])) != plan['declaration_sha256']:
        raise ValueError('frozen declaration changed')
    jobs = plan.get('expected_jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('expected jobs must be a frozen nonempty list')
    ids = [job.get('job_id') for job in jobs]
    if any(not isinstance(value, str) or not SAFE_ID.fullmatch(value) for value in ids) or len(ids) != len(set(ids)):
        raise ValueError('expected job IDs must be unique and safe')
    if plan.get('execution_role', 'primary') not in {'primary', 'evaluation_helper'}:
        raise ValueError('invalid execution_role')
    count = plan.get('local_gpu_count', 4)
    allowed_counts = {1, 2} if plan.get('execution_role') == 'evaluation_helper' else {2, 4}
    if type(count) is not int or count not in allowed_counts or count != lease['gpu_count']:
        raise ValueError('local GPU count differs from allocation')
    if plan.get('execution_role') == 'evaluation_helper' and not lease.get('control_relative'):
        raise ValueError('evaluation helper requires allocation-scoped control files')
    return root


def all_generation_complete(root, plan):
    return all((Path(root) / 'evaluation/jobs' / job['job_id'] / 'generation_complete.json').is_file()
               for job in plan['expected_jobs'])


def assemble_evaluation_plan(root, plan, *, persist=True):
    """Bind final checkpoint hashes while preserving frozen logical job order.

    The CPU scoring entrypoint separately performs the full answer, seed,
    control and file-hash closure before it opens any private tests.
    """
    base = Path(root) / 'evaluation'
    jobs = []
    expected_ids = {job['job_id'] for job in plan['expected_jobs']}
    actual_ids = {path.parent.name for path in (base / 'jobs').glob('*/job.json')}
    if actual_ids != expected_ids:
        raise ValueError('generation jobs differ from frozen expected population')
    for expected in plan['expected_jobs']:
        directory = base / 'jobs' / expected['job_id']
        job = read(directory / 'job.json')
        complete = read(directory / 'generation_complete.json')
        if any(job.get(key) != value for key, value in expected.items()):
            raise ValueError('completed job changed a frozen logical field')
        if job.get('experiment_id') != plan['experiment_id']:
            raise ValueError('completed job experiment differs')
        if complete.get('job_identity_sha256') != digest(job) or complete.get('hidden_tests_loaded') is not False:
            raise ValueError('generation completion is not bound to its job')
        jobs.append(job)
    result = {'experiment_id': plan['experiment_id'], 'jobs': jobs,
              'declaration_path': plan['declaration_path'],
              'declaration_sha256': plan['declaration_sha256'],
              'training_roots': {arm: str(Path(plan['result_root']) / 'arms' / arm)
                                 for arm in ('FIXED', 'ROTATING')}}
    output = base / 'evaluation_plan.json'
    if output.exists() and read(output) != result:
        raise ValueError('immutable evaluation plan already exists with different content')
    if persist and not output.exists():
        atomic_json(output, result)
    return result


def immutable_result_files(root, *, multipod=False):
    root = Path(root)
    excluded_names = {'gpu_release_verified.json', 'release_manifest.json'}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError('result symlink is not eligible for durable release')
        if not path.is_file():
            continue
        if relative.parts[0] in {'lease_guard', 'release_backups', 'allocations'} or path.name in excluded_names:
            continue
        if multipod and relative.parts[0] in {'workers', 'bootstrap', 'claims'}:
            continue
        if multipod and 'claims' in relative.parts:
            continue
        if str(relative) in {'supervisor/status.json', 'supervisor/events.jsonl', 'supervisor/supervisor.lock'}:
            continue
        if path.name.endswith('.tmp') or any(part in {'private', '.venv', '.pilot-venv', '.git', '__pycache__'} for part in relative.parts):
            continue
        yield path


def durable_checkpoint_files(root, *, require_complete=False):
    """Check immutable committed tensor hashes without importing the GPU stack."""
    for arm in ('FIXED', 'ROTATING'):
        folders = sorted((Path(root) / 'arms' / arm / 'checkpoints').glob('step_*'))
        if require_complete and {p.name for p in folders} != {
                f'step_{step:04d}' for step in (0, 128, 256, 512, 768, 1024)}:
            raise ValueError('all six full durable checkpoints per arm are required')
        for folder in folders:
            manifest_path = folder / 'manifest.json'
            manifest = read(manifest_path)
            if (manifest.get('complete_resumable') is not True
                    or manifest.get('verified_roundtrip') is not True
                    or manifest.get('arm') != arm
                    or folder.name != f'step_{manifest.get("step", -1):04d}'
                    or set(manifest.get('files', {})) != {'full.pt', 'mapper.pt'}):
                raise ValueError('unverified complete checkpoint during release')
            for name, record in manifest['files'].items():
                path = folder / name
                if (path.is_symlink() or path.stat().st_size != record['bytes']
                        or sha256_file(path) != record['sha256']):
                    raise ValueError('durable full checkpoint hash differs')
                yield path
            yield manifest_path


def helper_snapshot_files(root, lease, *, require_complete):
    """Only immutable generation/checkpoint data can outlive helper termination."""
    root = Path(root)
    yield from durable_checkpoint_files(root, require_complete=require_complete)
    for folder in sorted((root / 'evaluation/jobs').glob('*')):
        if not (folder / 'generation_complete.json').is_file():
            continue
        for path in sorted(folder.rglob('*')):
            if path.is_symlink():
                raise ValueError('linked generation evidence cannot authorize release')
            if (path.is_file() and path.name != 'score.json'
                    and not path.name.endswith(('.tmp', '.lock'))
                    and not any(part.startswith('.') for part in path.relative_to(folder).parts)):
                yield path
    # This pod's already-closed attempts provide failure/restart evidence even
    # when no scientific checkpoint was reached. Other pods keep writing.
    own = control_root(lease) / 'supervisor'
    for path in sorted(own.rglob('*')):
        if path.is_file() and path.name not in {'status.json', 'events.jsonl', 'supervisor.lock'}:
            yield path


def _record_file(root, path):
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def verify_compact_archive(path, expected):
    expected = {row['path']: row for row in expected}
    with tarfile.open(path, 'r:gz') as archive:
        names = archive.getnames()
        if len(names) != len(set(names)) or set(names) != set(expected) | {'COMPACT_MANIFEST.json'}:
            raise ValueError('compact archive member set differs')
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError('compact archive contains non-file member')
            content = archive.extractfile(member).read()
            if member.name == 'COMPACT_MANIFEST.json':
                if json.loads(content) != list(expected.values()):
                    raise ValueError('compact archive manifest differs')
            else:
                row = expected[member.name]
                if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
                    raise ValueError('compact archive hash mismatch')


def prepare_gpu_release(root, lease, *, outcome, helper=False):
    """Verify durable results and two compact copies; NEVER authorize volume deletion."""
    root = Path(root)
    control = control_root(lease)
    if helper or (lease.get('control_relative') and outcome != 'complete'):
        files = list(helper_snapshot_files(root, lease, require_complete=outcome == 'complete'))
    else:
        files = list(immutable_result_files(root, multipod=bool(lease.get('control_relative'))))
    records = [_record_file(root, path) for path in files]
    compact_extensions = {'.json', '.jsonl', '.csv', '.tsv', '.txt', '.log', '.md',
                          '.yaml', '.yml', '.png', '.svg', '.npz', '.npy'}
    compact = [row for row in records if Path(row['path']).suffix in compact_extensions
               and row['bytes'] <= 32 * 1024 * 1024]
    if not compact or not records:
        raise ValueError('no durable evidence to preserve before GPU release')
    backup = control / 'release_backups'
    backup.mkdir(parents=True, exist_ok=True)
    first, second = backup / 'compact_a.tar.gz', backup / 'compact_b.tar.gz'
    if first.exists() or second.exists():
        raise ValueError('release archives already exist; preserve previous attempt before rebuilding')
    temporary = backup / 'compact_a.tar.gz.tmp'
    with tarfile.open(temporary, 'w:gz', compresslevel=1) as archive:
        for row in compact:
            archive.add(root / row['path'], arcname=row['path'], recursive=False)
        content = json.dumps(compact, sort_keys=True, allow_nan=False).encode()
        member = tarfile.TarInfo('COMPACT_MANIFEST.json')
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    with temporary.open('rb') as source:
        os.fsync(source.fileno())
    verify_compact_archive(temporary, compact)
    os.replace(temporary, first)
    shutil.copyfile(first, second)
    with second.open('rb') as source:
        os.fsync(source.fileno())
    if sha256_file(first) != sha256_file(second):
        raise ValueError('independent compact archive copies differ')
    verify_compact_archive(second, compact)
    records.extend(_record_file(root, path) for path in (first, second))
    manifest = {'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id'],
                'outcome': outcome, 'files': records,
                'heavy_files_retained_on_network_volume': True,
                'off_pod_backup_verified': False,
                'network_volume_deletion_authorized': False}
    manifest['release_scope'] = 'immutable_generation_and_checkpoints' if helper else 'experiment_results'
    atomic_json(control / 'release_manifest.json', manifest)
    receipt = {'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id'],
               'gpu_release_verified': True, 'all_gpu_workers_stopped': True,
               'durable_backup_verified': True, 'manifest_path': str((control / 'release_manifest.json').relative_to(root)),
               'manifest_sha256': sha256_file(control / 'release_manifest.json'),
               'outcome': outcome, 'network_volume_deletion_authorized': False,
               'backup_scope': 'Network volume originals plus two verified compact archives; off-pod mirror follows.'}
    atomic_json(control / 'gpu_release_verified.json', receipt)
    if not verify_release_receipt(lease):
        raise ValueError('GPU release receipt failed final verification')
    return receipt


class Supervisor:
    def __init__(self, repo, plan_path, lease_path):
        self.repo = Path(repo).resolve()
        self.plan_path = scoped(self.repo, plan_path)
        self.lease_path = scoped(self.repo, lease_path)
        self.plan, self.lease = read(self.plan_path), read(self.lease_path)
        if scoped(self.repo, self.plan['allocation_lease_path']) != self.lease_path:
            raise ValueError('supervisor lease differs from frozen allocation plan')
        self.root = verify_plan(self.repo, self.plan, self.lease)
        self.allocation = control_root(self.lease)
        self.control = self.allocation / 'supervisor'
        self.role = self.plan.get('execution_role', 'primary')
        self.gpu_count = self.plan.get('local_gpu_count', 4)
        self.control.mkdir(parents=True, exist_ok=True)
        self.children, self.attempts = {}, {}
        self.completed = set()
        self.deadline = self.lease['deadline_epoch'] - 120
        self.stopping = False

    def worker_id(self, name):
        return self.lease['pod_id'] + '_' + name if self.lease.get('control_relative') else name

    def event(self, state, **details):
        value = {'epoch': time.time(), 'experiment_id': self.plan['experiment_id'],
                 'pod_id': self.lease['pod_id'], 'state': state, **details}
        atomic_json(self.control / 'status.json', value)
        with (self.control / 'events.jsonl').open('a') as output:
            output.write(json.dumps(value, allow_nan=False) + '\n')
            output.flush()
            os.fsync(output.fileno())

    def start(self, worker_id, role, gpu=None, **options):
        if worker_id in self.children:
            raise ValueError('worker is already running')
        attempt = self.attempts.get(worker_id, 0) + 1
        if attempt > 3:
            raise RuntimeError(f'bounded restart attempts exhausted for {worker_id}')
        self.attempts[worker_id] = attempt
        folder = self.control / 'attempts' / worker_id / f'attempt_{attempt:02d}'
        folder.mkdir(parents=True, exist_ok=False)
        command = [str(self.repo / '.pilot-venv/bin/python'),
                   'scripts/coding_coverage_v2_worker.py', '--plan', str(self.plan_path.relative_to(self.repo)),
                   '--role', role, '--worker-id', worker_id]
        for key, value in options.items():
            command.extend(['--' + key.replace('_', '-'), str(value)])
        metadata = {'worker_id': worker_id, 'role': role, 'gpu': gpu, 'options': options,
                    'attempt': attempt, 'command': command, 'started_epoch': time.time(),
                    'same_experiment_identity': self.plan['experiment_id']}
        output = (folder / 'worker.log').open('ab')
        try:
            process = subprocess.Popen(command, cwd=self.repo, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       env=child_environment(gpu), start_new_session=False)
        finally:
            output.close()
        metadata.update(pid=process.pid, pgid=os.getpgid(process.pid))
        if metadata['pgid'] != os.getpgrp():
            process.terminate()
            raise RuntimeError('worker escaped registered supervisor process group')
        atomic_json(folder / 'started.json', metadata)
        self.children[worker_id] = {'process': process, 'metadata': metadata, 'folder': folder}
        self.event('worker_started', worker=metadata)

    def poll(self):
        for worker_id, child in list(self.children.items()):
            code = child['process'].poll()
            if code is None:
                continue
            atomic_json(child['folder'] / 'exited.json', {'epoch': time.time(), 'returncode': code})
            del self.children[worker_id]
            meta = child['metadata']
            self.event('worker_exited', worker_id=worker_id, returncode=code, attempt=meta['attempt'])
            if code == 0:
                self.completed.add(worker_id)
            elif not self.stopping:
                if meta['attempt'] >= 3 or code in (64, 65, 66):
                    raise RuntimeError(f'worker {worker_id} failed with returncode {code}; attempt {meta["attempt"]}')
                self.start(worker_id, meta['role'], meta['gpu'], **meta['options'])

    def guard(self):
        if self.stopping or (self.allocation / 'lease_guard/STOP').exists() or time.time() >= self.deadline:
            raise TimeoutError('on-pod worker allocation deadline or local stop')

    def wait_phase(self, expected):
        while not expected <= self.completed:
            self.guard()
            self.poll()
            time.sleep(2)

    def stop_children(self):
        self.stopping = True
        atomic_json(self.allocation / 'lease_guard/STOP', {'epoch': time.time(), 'reason': 'supervisor_failure',
                    'experiment_id': self.plan['experiment_id'], 'pod_id': self.lease['pod_id']})
        end = time.monotonic() + 120
        while self.children and time.monotonic() < end:
            self.poll()
            time.sleep(1)
        for child in self.children.values():
            child['process'].terminate()
        end = time.monotonic() + 10
        while self.children and time.monotonic() < end:
            self.poll()
            time.sleep(0.2)
        for child in self.children.values():
            child['process'].kill()
            child['process'].wait(timeout=10)
        self.poll()

    def run(self):
        if os.getpid() != os.getpgrp():
            raise RuntimeError('supervisor must be launched in its own detached process group')
        registration = {**linux_process_identity(os.getpid()), 'experiment_id': self.plan['experiment_id'],
                        'pod_id': self.lease['pod_id']}
        atomic_json(self.allocation / 'lease_guard/supervisor_registration.json', registration)
        self.event('started', plan_sha256=sha256_file(self.plan_path), lease_sha256=sha256_file(self.lease_path))
        outcome = 'complete'
        try:
            self.guard()
            training_workers = {self.worker_id('train_' + arm) for arm in ('FIXED', 'ROTATING')}
            if self.role == 'primary':
                self.start(self.worker_id('train_FIXED'), 'train', 0, arm='FIXED')
                self.start(self.worker_id('train_ROTATING'), 'train', 1, arm='ROTATING')
                initial_eval_gpus = range(2, self.gpu_count)
            else:
                initial_eval_gpus = range(self.gpu_count)
            for gpu in initial_eval_gpus:
                self.start(self.worker_id(f'eval_gpu{gpu}'), 'evaluate', gpu)
            repurposed = set()
            while self.children:
                self.guard()
                self.poll()
                for arm, gpu in [('FIXED', 0), ('ROTATING', 1)]:
                    if self.role == 'primary' and self.worker_id('train_' + arm) in self.completed and gpu not in repurposed:
                        repurposed.add(gpu)
                        if not all_generation_complete(self.root, self.plan):
                            self.start(self.worker_id(f'eval_gpu{gpu}'), 'evaluate', gpu)
                time.sleep(2)
            if not all_generation_complete(self.root, self.plan):
                raise RuntimeError('evaluation workers exited before all frozen generations completed')
            if self.role == 'evaluation_helper':
                assemble_evaluation_plan(self.root, self.plan, persist=False)
                self.event('all_generation_complete_helper_idle')
            else:
                if not training_workers <= self.completed:
                    raise RuntimeError('training arms did not both finish')
                for arm in ('FIXED', 'ROTATING'):
                    training = read(self.root / 'arms' / arm / 'training_complete.json')
                    if training.get('full_target_completed') is not True or training.get('completed_updates') != 1024:
                        raise RuntimeError('a training process did not finish all 1024 updates')
                assemble_evaluation_plan(self.root, self.plan)
                self.event('all_generation_complete_before_scoring')
                scorers = {self.worker_id(f'score_cpu{i}') for i in range(4)}
                for index in range(4):
                    self.start(self.worker_id(f'score_cpu{index}'), 'score', shard_index=index, shard_count=4)
                self.wait_phase(scorers)
                self.start(self.worker_id('finalize_cpu'), 'finalize')
                self.wait_phase({self.worker_id('finalize_cpu')})
                scores = read(self.root / 'evaluation/scored_answer_manifest.json')
                if scores.get('experiment_id') != self.plan['experiment_id'] or scores.get('hidden_tests_loaded_after_all_generation') is not True:
                    raise RuntimeError('final scoring manifest is missing the full generation barrier')
                atomic_json(self.root / 'execution_complete.json', {
                    'experiment_id': self.plan['experiment_id'], 'pod_id': self.lease['pod_id'],
                    'full_1024_1024_training_completed': True, 'all_jobs_generated_and_scored': True,
                    'scored_answer_manifest_sha256': sha256_file(self.root / 'evaluation/scored_answer_manifest.json'),
                    'epoch': time.time()})
        except BaseException as error:
            outcome = 'failed_preserved'
            atomic_json(self.control / f'failure_{time.time_ns()}.json', {
                'epoch': time.time(), 'exception': type(error).__name__, 'error': str(error),
                'experiment_id': self.plan['experiment_id'], 'pod_id': self.lease['pod_id']})
            self.event('failure_draining', exception=type(error).__name__)
            self.stop_children()
        if self.children:
            raise RuntimeError('cannot release GPU while child processes remain')
        self.event('verifying_durable_results', outcome=outcome)
        receipt = prepare_gpu_release(self.root, self.lease, outcome=outcome,
                                      helper=self.role == 'evaluation_helper')
        # Only these two mutable supervisor files change after the manifest.
        self.event('gpu_release_verified', outcome=outcome, receipt_sha256=sha256_file(self.allocation / 'gpu_release_verified.json'))
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--lease', required=True)
    args = parser.parse_args()
    supervisor = Supervisor(REPO, args.plan, args.lease)
    with (supervisor.control / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor.run()


if __name__ == '__main__':
    main()
