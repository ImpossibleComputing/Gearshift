"""Immutable, task-disjoint dispatch plans for the bounded coding pilot.

This module does not allocate resources or choose scientific outcomes. Stage
proofs are produced by stage-specific validators; every proof input is checked
again immediately before dispatch. Scheduling never changes sampling streams.
"""
import json
import math
import re
from pathlib import Path, PurePosixPath

from gearshift.coding_control import digest, sha

MAX_WORKERS = 8
STAGE_GATES = {
    'coverage_preflight': {'baseline','amendment','coverage_declaration','numerical_paths'},
    'coverage_experiment': {'baseline','amendment','coverage_declaration','numerical_paths','coverage_endpoint'},
    'coverage_evaluation_recovery': {'baseline','amendment','coverage_declaration','numerical_paths','coverage_endpoint','coverage_recovery'},
    'post_progress_numerical': {'baseline', 'amendment', 'post_progress_declaration'},
    'post_progress_hybrid': {'baseline', 'amendment', 'post_progress_declaration', 'numerical_paths'},
    'post_progress_memorization': {'baseline', 'amendment', 'post_progress_declaration', 'numerical_paths', 'prompt_experiment'},
    'recovery_probe': {'baseline', 'amendment'},
    'exploratory_training': {'baseline', 'amendment', 'partial_corpus'},
    'exploratory_development': {'baseline', 'amendment', 'exploratory_checkpoint'},
    'cap_recovery': {'interruption_recovery', 'original_cap_parent'},
    'memory': {'baseline'},
    'histories': {'baseline', 'memory'},
    'initialization': {'baseline', 'memory', 'histories'},
    'training': {'baseline', 'memory', 'initialization'},
    'development': {'baseline', 'memory', 'training'},
    'confirmation': {'baseline', 'memory', 'selection'},
    'confirmation_with_second_seed': {'baseline', 'memory', 'selection', 'initialization'},
    'seed_sensitivity': {'baseline', 'memory', 'selection', 'second_seed'},
}
RESERVED_FIELDS = {'stage', 'run_id', 'worker_id', 'stage_identity', 'task_ids',
    'deadline_epoch', 'allocation_path', 'result_root', 'worker_root', 'worker_status_path', 'native_controls_required',
    'dispatch_stage', 'role'}


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}', value):
        raise ValueError('Unsafe run or worker identifier')
    return value


def relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Unsafe relative path')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or str(p) != value:
        raise ValueError('Unsafe relative path')
    return value


def file_in(root, value):
    relative_path(value)
    root = Path(root).resolve()
    p = root / value
    if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root):
        raise ValueError('Missing or unsafe input: ' + value)
    return p


def task_shards(task_ids, workers):
    """Preserve fixed task order within deterministic, round-robin shards."""
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError('One to eight single-GPU workers required')
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError('Fixed cohort must contain unique task IDs')
    if workers > len(task_ids):
        raise ValueError('Empty paid shards are forbidden')
    return [list(task_ids[i::workers]) for i in range(workers)]


def verify_gate(root, name, reference):
    p = file_in(root, reference['path'])
    if sha(p) != reference['sha256']:
        raise ValueError('Stage proof changed: ' + name)
    gate = json.loads(p.read_text())
    if gate.get('passed') is not True or gate.get('gate') != name:
        raise ValueError('Stage proof did not pass: ' + name)
    if not isinstance(gate.get('inputs'), dict) or not gate['inputs']:
        raise ValueError('Stage proof must bind its evidence inputs')
    for rel, expected in gate['inputs'].items():
        if sha(file_in(root, rel)) != expected:
            raise ValueError('Stage proof evidence changed: ' + rel)
    return gate


def worker_input_files(plan, worker):
    subset = worker.get('input_files')
    if subset is None:
        return plan['files']
    if not isinstance(subset, list) or len(set(subset)) != len(subset) or not subset:
        raise ValueError('Explicit worker input subset must contain unique files')
    if any(rel not in plan['files'] for rel in subset):
        raise ValueError('Worker input is not in the immutable upload manifest')
    return {rel: plan['files'][rel] for rel in subset}


def confirmation_input(rel):
    parts = PurePosixPath(rel).parts
    return ((rel.startswith('data/coding_pilot_v1/visible/') or rel.startswith('data/coding_pilot_v1/private/'))
        and 'confirmation' in PurePosixPath(rel).name) or (
        (rel.startswith('results/') or rel.startswith('evidence/'))
        and any('confirmation' in part or 'headline' in part for part in parts))


def validate_plan(plan, root, approval_sha256):
    """Fail closed before any paid allocation; return the immutable plan digest."""
    if plan.get('schema') != 1 or plan.get('approval_sha256') != approval_sha256:
        raise ValueError('Unbound dispatch authorization')
    safe_id(plan['run_id'])
    stage = plan['stage']
    if stage not in STAGE_GATES:
        raise ValueError('Unsupported bounded stage')
    if not STAGE_GATES[stage].issubset(plan.get('gates', {})):
        raise ValueError('Required scientific prerequisite missing')
    for name, reference in plan['gates'].items():
        verify_gate(root, name, reference)
    if not re.fullmatch(r'runpod/pytorch@sha256:[a-f0-9]{64}', plan.get('image_digest', '')):
        raise ValueError('Immutable approved container image required')
    files = plan.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Exact upload file hashes required')
    for rel, expected in files.items():
        if sha(file_in(root, rel)) != expected:
            raise ValueError('Upload input changed: ' + rel)
    # Proofs and their inputs must accompany each worker for independent checking.
    for ref in plan['gates'].values():
        gate = json.loads(file_in(root, ref['path']).read_text())
        for rel, expected in {ref['path']: ref['sha256'], **gate['inputs']}.items():
            if files.get(rel) != expected:
                raise ValueError('Gate evidence absent from upload manifest')
    workers = plan.get('workers', [])
    if not 1 <= len(workers) <= MAX_WORKERS:
        raise ValueError('Worker count outside bound')
    if len({safe_id(w['worker_id']) for w in workers}) != len(workers):
        raise ValueError('Duplicate worker ID')
    for w in workers:
        worker_files = worker_input_files(plan, w)
        if not isinstance(w.get('command'), list) or not w['command'] or any(not isinstance(x, str) for x in w['command']):
            raise ValueError('Worker command must be literal argv')
        script = relative_path(w['command'][0])
        if not script.startswith('scripts/coding_') or not script.endswith('.py') or script not in worker_files:
            raise ValueError('Worker script must be uploaded and hash-bound')
        if RESERVED_FIELDS.intersection(w.get('worker_fields', {})):
            raise ValueError('Worker fields cannot override dispatch identity')
        seconds = w.get('estimated_seconds')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Measured stage estimate required')
        maximum = w.get('maximum_seconds')
        if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum < seconds or maximum > 24 * 3600:
            raise ValueError('Finite worker deadline within one day required')
        if stage == 'training' and maximum > 3 * 3600:
            raise ValueError('Per-run training limit remains three GPU-hours')
        if not isinstance(w.get('region'), str) or not re.fullmatch(r'[A-Z0-9-]+', w['region']):
            raise ValueError('Explicit region required')
        if 'region_candidates' in w:
            regions = w['region_candidates']
            if (not isinstance(regions, list) or not 1 <= len(regions) <= 3
                or len(set(regions)) != len(regions) or regions[0] != w['region']
                or any(not isinstance(r, str) or not re.fullmatch(r'[A-Z0-9-]+', r) for r in regions)):
                raise ValueError('Capacity retry permits at most three frozen distinct regions')
        for ref in plan['gates'].values():
            gate = json.loads(file_in(root, ref['path']).read_text())
            for rel, expected in {ref['path']: ref['sha256'], **gate['inputs']}.items():
                if worker_files.get(rel) != expected:
                    raise ValueError('Worker input subset omits required gate evidence')
    task_workers = workers
    if stage == 'confirmation_with_second_seed':
        training = [w for w in workers if w.get('role') == 'second_seed_training']
        task_workers = [w for w in workers if w.get('role') == 'confirmation']
        if len(training) != 1 or not 1 <= len(task_workers) <= 7 or len(training) + len(task_workers) != len(workers):
            raise ValueError('Mixed stage requires one second-seed training worker and one to seven confirmation shards')
        train = training[0]; fields = train.get('worker_fields', {})
        if train.get('task_ids') != [] or fields.get('seed') != 20260916 or train['maximum_seconds'] > 10800:
            raise ValueError('Second-seed training retains its seed, no confirmation tasks, and three-hour bound')
        if train.get('input_files') is None or any(confirmation_input(rel) for rel in worker_input_files(plan, train)):
            raise ValueError('Second-seed training must not receive confirmation inputs or outcomes')
        if fields.get('objective') not in ('ordinary_continuation', 'natural_handoff_boundary'):
            raise ValueError('Second-seed objective must be the frozen selected recipe')
        selection_ref = plan['gates']['selection']
        selection_proof = json.loads(file_in(root, selection_ref['path']).read_text())
        selection_path = fields.get('selection_path')
        if (selection_proof['inputs'].get(selection_path) != fields.get('selection_sha256')
            or worker_input_files(plan, train).get(selection_path) != fields.get('selection_sha256')):
            raise ValueError('Second seed must bind the same pre-confirmation selection proof')
        selected = json.loads(file_in(root, selection_path).read_text())
        if (fields['objective'] != selected.get('objective') or selected.get('seed') != 20260915
            or selected.get('confirmation_used_for_selection') is not False):
            raise ValueError('Second seed cannot select its recipe using confirmation outcomes')
    cohort = plan.get('task_ids', [])
    if cohort:
        cohort_files = plan.get('cohort_files', [])
        if not cohort_files:
            raise ValueError('Fixed cohort must be bound to visible task manifests')
        from_files = []
        for rel in cohort_files:
            if not rel.startswith('data/coding_pilot_v1/visible/') or files.get(rel) != sha(file_in(root, rel)):
                raise ValueError('Visible cohort input not bound to upload')
            rows = json.loads(file_in(root, rel).read_text())
            from_files.extend(row['task_id'] for row in rows)
        if cohort != from_files:
            raise ValueError('Declared task cohort differs from frozen visible manifests')
        want = task_shards(cohort, len(task_workers))
        if [w.get('task_ids') for w in task_workers] != want:
            raise ValueError('Task shards differ from fixed disjoint assignment')
    elif any(w.get('task_ids') for w in workers):
        raise ValueError('Worker task IDs missing fixed parent cohort')
    elif stage in ('cap_recovery', 'histories', 'development', 'confirmation', 'seed_sensitivity'):
        raise ValueError('Task-bearing stage requires a fixed cohort')
    if stage in ('confirmation', 'confirmation_with_second_seed') and len(cohort) != 200:
        raise ValueError('Headline confirmation requires all 200 tasks')
    if stage == 'seed_sensitivity' and len(cohort) != 40:
        raise ValueError('Seed sensitivity requires the fixed forty-task subset')
    amended = stage in ('recovery_probe', 'exploratory_training', 'exploratory_development')
    if amended:
        amendment_path = 'configs/coding_pilot_v1/recovery_20260917/amendment.json'
        proof = verify_gate(root, 'amendment', plan['gates']['amendment'])
        if (proof['inputs'].get(amendment_path) != sha(file_in(root, amendment_path))
            or plan.get('scientific_scope_unchanged') is not False
            or plan.get('scope_amendment') != 'recovery_20260917_partial_v1'):
            raise ValueError('Exploratory scope must explicitly bind the owner amendment')
        if any(confirmation_input(rel) for rel in files):
            raise ValueError('Reserved confirmation inputs forbidden in recovery')
        if stage in ('recovery_probe', 'exploratory_training') and len(workers) != 1:
            raise ValueError('First recovery/training attempt uses one isolated worker')
        if stage == 'recovery_probe' and len(cohort) != 1:
            raise ValueError('Recovery starts with exactly one saved failing trajectory')
        if stage == 'exploratory_training' and workers[0]['maximum_seconds'] > 10800:
            raise ValueError('Exploratory training allocation remains bounded to three hours')
        if stage == 'exploratory_development' and len(cohort) != 40:
            if not 1 <= len(cohort) < 40 or 'recovery_coverage' not in plan['gates']:
                raise ValueError('Exploratory comparison retains all forty development tasks')
            from .coding_development_recovery import verify_coverage
            coverage=verify_gate(root,'recovery_coverage',plan['gates']['recovery_coverage'])
            verify_coverage(root,coverage,cohort,plan=plan)
    if stage.startswith('coverage_'):
        from .coding_coverage import validate_coverage_plan
        validate_coverage_plan(plan, root)
        amended = True
    if stage.startswith('post_progress_'):
        from .coding_post_progress import validate_diagnostic_plan
        validate_diagnostic_plan(plan, root)
        amended = True
    if plan.get('preserve_original_measurements') is not True or (not amended and plan.get('scientific_scope_unchanged') is not True):
        raise ValueError('Scientific invariants must remain explicit')
    return digest(plan)


def check_dispatch_budget(plan, decision, *, active_pods=0):
    if decision.get('stop') or decision.get('dispatch_blocked') or decision.get('parallel_execution_approved') is not True:
        raise ValueError('Dispatch blocked by cumulative authorization or watchdog')
    if active_pods + len(plan['workers']) > min(MAX_WORKERS, decision['max_concurrent_gpus']):
        raise ValueError('Concurrent single-GPU ceiling exceeded')
    # Charge worst permitted worker lifetimes, including concurrent bootstrap.
    reserved_hours = sum(w['maximum_seconds'] for w in plan['workers']) / 3600
    if decision['gpu_hours'] + reserved_hours > decision['hard_total_gpu_hours']:
        raise ValueError('Whole-stage reservation exceeds cumulative GPU-hours')
    if decision['upper_usd'] + reserved_hours * 5.56 > decision['cleanup_dispatch_ceiling_usd']:
        raise ValueError('Whole-stage reservation exceeds dollar dispatch ceiling')
    return {'reserved_gpu_hours': reserved_hours, 'reserved_upper_usd': reserved_hours * 5.56}


def worker_spec(plan, worker, identity, deadline, allocation_path):
    run_id, worker_id = safe_id(plan['run_id']), safe_id(worker['worker_id'])
    stage = plan['stage']
    if stage == 'confirmation_with_second_seed':
        stage = 'training' if worker['role'] == 'second_seed_training' else 'confirmation'
    stage = {'development': 'development_candidates', 'seed_sensitivity': 'second_seed'}.get(stage, stage)
    return {**worker.get('worker_fields', {}), 'worker_fields': worker.get('worker_fields', {}), 'stage': stage,
        'dispatch_stage': plan['stage'], 'role': worker.get('role', stage), 'run_id': run_id,
        'worker_id': worker_id, 'stage_identity': identity, 'task_ids': worker['task_ids'],
        'deadline_epoch': deadline, 'allocation_path': allocation_path,
        'result_root': f'results/coding_pilot_v1/{run_id}/{worker_id}',
        'worker_root': f'evidence/coding_pilot_v1/parallel/{run_id}/{worker_id}',
        'worker_status_path': f'evidence/coding_pilot_v1/parallel/{run_id}/{worker_id}/worker_status.json',
        'native_controls_required': True, 'gates': plan['gates']}


def verify_worker_status(status, spec):
    if status.get('stage_identity') != spec['stage_identity'] or status.get('worker_id') != spec['worker_id']:
        raise ValueError('Worker status identity mismatch')
    if status.get('state') not in ('running', 'complete', 'failed'):
        raise ValueError('Unrecognized worker state')
    return status['state']


def checkpoint_files(manifest, result_root, *, features=False):
    files = manifest.get('files', {})
    if not isinstance(files, dict) or len(files) > (128 if features else 32):
        raise ValueError('Checkpoint manifest outside bounded mapper set')
    for rel, expected in files.items():
        relative_path(rel)
        if not rel.startswith(result_root + '/') or Path(rel).suffix not in ('.pt', '.safetensors'):
            raise ValueError('Only task-owned mapper checkpoints may be copied')
        if features and not rel.endswith('/paired_features.pt'):
            raise ValueError('Feature manifest may contain only sampled paired features')
        if not re.fullmatch(r'[a-f0-9]{64}', expected):
            raise ValueError('Checkpoint requires immutable file hash')
    return files
