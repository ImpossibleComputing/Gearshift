#!/usr/bin/env python3
"""Independent GPU/CPU process; only local immutable inputs and lease govern work."""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import json
import os
import re
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, digest, sha, write


def read(path):
    return json.loads(Path(path).read_text())


def context(plan_path, role, worker_id, arm=None):
    plan = read(ROOT / plan_path)
    declaration = ROOT / plan['declaration_path']
    if sha(declaration) != plan['declaration_sha256']:
        raise ValueError('Frozen declaration changed')
    for name, receipt in plan['input_manifest'].items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha(path) != receipt['sha256'] or path.stat().st_size != receipt['bytes']:
            raise ValueError('Immutable input changed: ' + name)
    d = read(declaration)
    top = ROOT / plan['result_root']
    top.resolve().relative_to(Path('/workspace').resolve())
    from gearshift.coding_coverage_v2_lease import control_root, load_lease
    lease = load_lease(ROOT / plan['allocation_lease_path'], plan['allocation_lease_sha256'])
    if lease['experiment_id'] != plan['experiment_id'] or Path(lease['allowed_result_root']).resolve() != top.resolve():
        raise ValueError('Allocation/experiment root differs')
    if sha(ROOT / plan['allocation_lease_path']) != plan['allocation_lease_sha256']:
        raise ValueError('Allocation lease changed')
    allocation_control = control_root(lease)
    attempt = worker_id + '_' + str(time.time_ns())
    status_root = top / 'workers' / worker_id / attempt
    status_root.mkdir(parents=True, exist_ok=False)
    result_root = top / 'arms' / arm if role == 'train' else top / 'evaluation'
    result_root.mkdir(parents=True, exist_ok=True)
    current = {'state': 'running', 'role': role, 'worker_id': worker_id, 'attempt_id': attempt,
               'experiment_id': plan['experiment_id'], 'pid': os.getpid()}
    def publish(**kw):
        current.update(kw)
        write(status_root / 'status.json', {**current, 'epoch': time.time()})
        write(top / 'workers' / worker_id / 'latest.json', {**current, 'epoch': time.time(), 'attempt_path': str(status_root.relative_to(ROOT))})
    def guard():
        if time.time() >= lease['deadline_epoch'] - 120 or (allocation_control / 'lease_guard/STOP').exists():
            raise TimeoutError('Immutable allocation deadline or local lease stop')
    def stop(signum, frame):
        raise TimeoutError('Controlled lease termination signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    from gearshift.coding_recovery import Telemetry
    telemetry = Telemetry(status_root)
    identity = {'experiment_id': plan['experiment_id'], 'plan_sha256': sha(ROOT / plan_path),
                'declaration_sha256': plan['declaration_sha256'], 'source_commit': plan['code_commit'],
                'role': role, 'worker_id': worker_id, 'attempt_id': attempt,
                'input_manifest_sha256': digest(plan['input_manifest']), 'pod_id': lease['pod_id']}
    spec = {'arm': arm, 'declaration_sha256': plan['declaration_sha256'], 'code_commit': plan['code_commit'],
            'training_seed': 20260915, 'target_updates': 1024, 'checkpoint_updates': [0,128,256,512,768,1024]}
    if arm:
        spec['checkpoint_identity'] = checkpoint_identity(plan, d, arm)
    publish(stage='local_input_verification_passed')
    return {'plan': plan, 'spec': spec, 'declaration': d, 'root': result_root, 'top': top, 'allocation_control': allocation_control,
            'repo_root': ROOT, 'declaration_path': plan['declaration_path'], 'declaration_sha256': plan['declaration_sha256'],
            'attempt_id': attempt, 'identity': identity, 'status_root': status_root,
            'private_tests_sha256': plan['private_tests_sha256'],
            'config': read(ROOT / 'configs/coding_pilot_v1/pilot.json'), 'guard': guard,
            'publish': publish, 'telemetry': telemetry}


def checkpoint_identity(plan, d, arm):
    return {'experiment_id': plan['experiment_id'], 'arm': arm,
            'code_commit': plan['code_commit'], 'config_sha256': plan['declaration_sha256'],
            'schedules_sha256': d['schedules_sha256'], 'corpus_sha256': d['corpus_manifest_sha256'],
            'selected_checkpoint_sha256': d['selected_checkpoint_sha256'],
            'models': read(ROOT / 'configs/coding_pilot_v1/pilot.json')['models'],
            'configuration': {'optimizer': d['optimizer'], 'dtype': 'bfloat16', 'attention': 'sdpa',
                'gradient_positions': 32, 'full_context': True, 'scheduler': 'constant', 'training_seed': 20260915}}


def setup(c, arm=None):
    import torch
    import transformers
    from gearshift.coding_inference import Backend
    from gearshift.coding_gradients import AffineMapper
    from gearshift.coding_coverage_runtime import CoverageRuntime
    from scripts.coding_partial_corpus import load_corpus
    d = c['declaration']
    if torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
        raise ValueError('Exactly one equivalent H200 must be visible per process')
    if not torch.__version__.startswith('2.8.0') or transformers.__version__ != '4.57.6':
        raise ValueError('Pinned numerical runtime differs')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for key, expected in d['unchanged_source_files'].items():
        if sha(ROOT / key) != expected:
            raise ValueError('Frozen numerical implementation differs: ' + key)
    for key in ['schedules','panels','answer_seeds']:
        if sha(ROOT / d[key + '_path']) != d[key + '_sha256']:
            raise ValueError('Frozen schedule/panel/seed differs')
    if sha(ROOT / d['corpus_manifest']) != d['corpus_manifest_sha256']:
        raise ValueError('Frozen corpus differs')
    histories, _, _ = load_corpus(ROOT, ROOT / d['corpus_manifest'], features=False)
    training = [h for h in histories if h['split'] == 'training']
    validation = [h for h in histories if h['split'] == 'validation']
    if [h['task_id'] for h in training] != d['training_task_ids'] or [h['task_id'] for h in validation] != d['validation_task_ids']:
        raise ValueError('Exact ordered 104/21 split differs')
    c['guard'](); c['publish'](stage='load_source'); source = Backend(c['config']['models']['source'])
    c['telemetry'].sample(stage='source_loaded')
    c['guard'](); c['publish'](stage='load_receiver'); receiver = Backend(c['config']['models']['receiver'])
    if source.tokenizer.backend_tokenizer.to_str() != receiver.tokenizer.backend_tokenizer.to_str():
        raise ValueError('Tokenizer alignment changed')
    if sha(ROOT / d['selected_checkpoint']) != d['selected_checkpoint_sha256']:
        raise ValueError('Original selected96 checkpoint differs')
    saved = torch.load(ROOT / d['selected_checkpoint'], map_location='cpu', weights_only=True)
    mapper = AffineMapper(source, receiver); mapper.load_state_dict(saved['state_dict'], strict=True)
    if any(not torch.equal(v.detach().cpu(), saved['state_dict'][k]) for k,v in mapper.state_dict().items()):
        raise ValueError('Fresh mapper tensors differ from original selected96')
    del saved; gc.collect()
    c['identity']['runtime'] = {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'gpu_capability': list(torch.cuda.get_device_capability(0)),
        'dtype': 'bfloat16', 'attention': 'sdpa', 'tf32': False, 'deterministic_algorithms': True,
        'device_isolation': os.environ.get('CUDA_VISIBLE_DEVICES'), 'models_frozen': True}
    bind(c['status_root'] / 'identity.json', c['identity'])
    # Established full-path controls are immutable inputs, rather than repeated
    # expensive work on every asynchronous consumer. The new training-resume
    # check and each evaluated native/native splice run on actual models here.
    bind(c['status_root'] / 'control_provenance.json', {'prior_controls': c['plan']['prior_controls'],
        'numerical_implementation_unchanged': True, 'training_exact_resume_check_required': bool(arm),
        'per_evaluation_history_native_splice_checks_required': not bool(arm)})
    c['telemetry'].reset('models_loaded_before_scientific_work')
    if not arm:
        return source, receiver, mapper
    r = d['optimizer']
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=r['lr'], betas=tuple(r['betas']), eps=r['eps'],
        weight_decay=r['weight_decay'], foreach=r['foreach'], fused=r['fused'])
    runtime = CoverageRuntime(source, receiver, mapper, c['guard'], c['telemetry'])
    return d, training, validation, read(ROOT / d['schedules_path']), read(ROOT / d['panels_path']), source, receiver, {arm:mapper}, {arm:runtime}, {arm:optimizer}


def materialize_job(c, logical):
    job = dict(logical)
    if job['arm'] == 'START':
        job['checkpoint'] = {'path': c['declaration']['selected_checkpoint'], 'sha256': c['declaration']['selected_checkpoint_sha256']}
    else:
        folder = c['top'] / 'arms' / job['arm'] / 'checkpoints' / f"step_{job['step']:04d}"
        if not (folder / 'manifest.json').exists():
            return None
        m = read(folder / 'manifest.json')
        if m['step'] != job['step'] or m['arm'] != job['arm'] or m.get('complete_resumable') is not True or m.get('verified_roundtrip') is not True:
            raise ValueError('Uncommitted or mismatched asynchronous checkpoint')
        expected = checkpoint_identity(c['plan'], c['declaration'], job['arm'])
        if m['checkpoint_identity'] != expected or m['checkpoint_identity_sha256'] != digest(expected) or m['schedule_position'] != job['step']:
            raise ValueError('Asynchronous checkpoint scientific identity differs')
        mapper_file = folder / 'mapper.pt'
        if mapper_file.is_symlink() or mapper_file.stat().st_size != m['files']['mapper.pt']['bytes'] or sha(mapper_file) != m['files']['mapper.pt']['sha256']:
            raise ValueError('Asynchronous mapper checksum differs')
        job['checkpoint'] = {'path': str((folder / 'mapper.pt').relative_to(ROOT)), 'sha256': m['files']['mapper.pt']['sha256']}
        job['full_checkpoint_manifest_sha256'] = sha(folder / 'manifest.json')
    return job



@contextmanager
def claim_job(c, job_id):
    """Hold one stable kernel lock for the complete job, including crash recovery.

    Lock files are never renamed/unlinked: every contender must lock the same
    inode. A crashed process automatically releases its lock. Owner metadata is
    audit evidence only and is never used to decide whether a lock is stale.
    """
    if not isinstance(job_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,180}', job_id):
        raise ValueError('Unsafe evaluation job identity')
    claims = c['root'] / 'claims'; claims.mkdir(parents=True, exist_ok=True)
    lock_path = claims / (job_id + '.lock')
    # O_NOFOLLOW prevents an unexpected symlink from redirecting ownership.
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        acquired = True
        owner_path = claims / (job_id + '.owner.json')
        if owner_path.exists():
            old = read(owner_path)
            write(claims / 'history' / job_id / (str(time.time_ns()) + '.json'),
                  {**old, 'superseded_epoch': time.time(),
                   'previous_owner_did_not_release_cleanly': old.get('state') == 'active'})
        try:
            from gearshift.coding_coverage_v2_lease import linux_process_identity
            process = linux_process_identity(os.getpid())
        except (FileNotFoundError, NotADirectoryError):
            # CPU test environments need not expose Linux /proc. Lock ownership
            # still comes solely from flock, never this diagnostic fallback.
            process = {'pid': os.getpid(), 'platform': sys.platform, 'linux_identity_available': False}
        owner = {**process, 'attempt_id': c['attempt_id'], 'job_id': job_id,
                 'state': 'active', 'epoch': time.time(), 'mechanism': 'stable_inode_flock'}
        write(owner_path, owner)
        try:
            yield True
        finally:
            write(owner_path, {**owner, 'state': 'released', 'released_epoch': time.time()})
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def evaluator(c):
    from coding_coverage_v2_evaluate import run_job
    source, receiver, mapper = setup(c)
    logicals = c['plan']['expected_jobs']
    # Final primary jobs take priority when available; earlier checkpoints run
    # while training is progressing, without blocking either optimizer.
    ordered = sorted(logicals, key=lambda j: (-j['step'], j['job_id']))
    while True:
        c['guard']()
        done = [j for j in logicals if (c['root'] / 'jobs' / j['job_id'] / 'generation_complete.json').exists()]
        if len(done) == len(logicals):
            c['publish'](state='complete', stage='all_generation_complete', jobs=len(done)); return
        took = False
        for logical in ordered:
            c['guard'](); job_id = logical['job_id']
            if (c['root'] / 'jobs' / job_id / 'generation_complete.json').exists(): continue
            job = materialize_job(c, logical)
            if job is None: continue
            with claim_job(c, job_id) as acquired:
                if not acquired:
                    continue
                # A contender may have completed between the first completion
                # check and lock acquisition. Never repeat its scientific work.
                if (c['root'] / 'jobs' / job_id / 'generation_complete.json').exists():
                    continue
                run_job(c, source, receiver, mapper, job)
            took = True; break
        if not took:
            c['publish'](stage='waiting_for_immutable_checkpoint_or_other_worker', completed_jobs=len(done), total_jobs=len(logicals))
            time.sleep(2)


def failure_exit_code(exc):
    """Separate scientific/identity rejection from bounded engineering retries."""
    if isinstance(exc, (ValueError, AssertionError, FloatingPointError)):
        return 65
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        scientific_failures = (
            'historical cache mutated by training',
            'missing, zero or nonfinite cache gradient',
            'missing, zero or nonfinite mapper gradient',
            'frozen language model received gradients',
            'actual-model exact checkpoint continuation control failed',
            'could not restore clean primary initialization after preflight',
            'native/native splice changed whole-history execution',
            'generation cache aliases immutable base',
            'answer seed or immutable base cache drift',
            'saved historical pairs mutated',
            'frozen isolated sandbox gate is not passed',
        )
        if any(text in message for text in scientific_failures) or re.search(r'\bnon[- ]?finite\b', message):
            return 65
    return 1


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True)
    p.add_argument('--role', choices=['train','evaluate','score','finalize'], required=True)
    p.add_argument('--worker-id', required=True); p.add_argument('--arm', choices=['FIXED','ROTATING'])
    p.add_argument('--shard-index', type=int, default=0); p.add_argument('--shard-count', type=int, default=4)
    a = p.parse_args(); c = None
    try:
        c = context(a.plan, a.role, a.worker_id, a.arm)
        if a.role == 'train':
            from coding_coverage_v2_train import train
            train(c, setup(c, a.arm), a.arm)
        elif a.role == 'evaluate': evaluator(c)
        else:
            from coding_coverage_v2_evaluate import score_jobs, finalize_scores
            plan = read(c['root'] / 'evaluation_plan.json')
            if a.role == 'score': score_jobs(c, plan, shard_index=a.shard_index, shard_count=a.shard_count)
            else:
                finalize_scores(c, plan)
                from coding_coverage_v2_report import report
                report(c['root'], plan, repo_root=ROOT, training_roots=plan['training_roots'], primary_step=1024)
        c['publish'](state='complete', stage=a.role + '_complete')
    except BaseException as exc:
        if c:
            c['telemetry'].failure(exc)
            write(c['status_root'] / 'failure.json', {'epoch': time.time(), 'error': str(exc), 'type': type(exc).__name__, 'traceback': traceback.format_exc()})
            c['publish'](state='failed', stage=a.role + '_failed', error=str(exc))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        traceback.print_exc()
        raise SystemExit(failure_exit_code(exc)) from exc


if __name__ == '__main__': main()
