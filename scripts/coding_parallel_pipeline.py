#!/usr/bin/env python3
"""Studio supervisor for the finite coding-pilot stage dependency chain.

The stage builder imports verified backups and checks actual scientific gates.
This supervisor never turns successful process exit into scientific success,
never relaunches a one-shot dispatch, and never holds GPUs for a human decision.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gearshift.coding_control import PREFIX, digest, sha, write
from gearshift.coding_parallel import file_in, safe_id, validate_plan
from coding_cloud_guard import cli

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'evidence/coding_pilot_v1/control'
CURRENT_STATE_PATH = None
STAGES = ('memory', 'histories', 'initialization', 'training', 'development',
    'confirmation_with_second_seed', 'seed_sensitivity')


def next_stage(state):
    completed = state.get('completed_runs', {})
    seen_gap = False
    for stage in STAGES:
        if stage not in completed:
            seen_gap = True
        elif seen_gap:
            raise ValueError('Pipeline state skips a required predecessor')
    if 'cap_recovery' not in completed:
        if any(stage in completed for stage in STAGES):
            raise ValueError('Pipeline stages cannot precede baseline recovery')
        return 'cap_recovery'
    return next((stage for stage in STAGES if stage not in completed), 'finished')


def read_completion(root, run_id):
    safe_id(run_id)
    control = Path(root) / 'evidence/coding_pilot_v1/control/parallel' / run_id
    path = control / 'complete.json'
    if not path.exists(): return None
    receipt = json.loads(path.read_text())
    plan = json.loads((control / 'plan.json').read_text())
    if receipt.get('stage_identity') != digest(plan) or plan.get('run_id') != run_id:
        raise ValueError('Completed dispatch identity differs from frozen plan')
    if receipt.get('passed') is not True:
        raise RuntimeError('Stage did not complete: ' + str(receipt.get('errors')))
    expected = {worker['worker_id'] for worker in plan['workers']}
    if {worker.get('worker_id') for worker in receipt.get('workers', [])} != expected:
        raise ValueError('Stage completion does not cover every allocated worker')
    if any(worker.get('passed') is not True for worker in receipt['workers']):
        raise RuntimeError('A stage worker failed')
    return receipt


def confirmed_clean(run_id):
    """No next stage while previous owned compute/storage remains unaccounted."""
    prefix = PREFIX + run_id + '-'
    found = []
    for command, kind in [('pod', 'pod'), ('network-volume', 'volume')]:
        for row in cli(command, 'list'):
            if row.get('name', '').startswith(prefix): found.append((kind, row['id']))
    return not found, found


def forecast(state, now=None):
    """Report a schedule estimate; never turn its target into a quality override."""
    now = time.time() if now is None else now
    completed = state.get('completed_runs', {})
    stages = [stage for stage in ('cap_recovery',) + STAGES if stage not in completed]
    current = state.get('current_stage', stages[0] if stages else None)
    limits = state.get('stage_limits_seconds', {})
    remaining = float(state.get('report_estimate_seconds', 600))
    missing = []; over_estimate = False
    for stage in stages:
        estimate = limits.get(stage, {}).get('estimated_seconds')
        if not isinstance(estimate, (float, int)) or estimate <= 0:
            missing.append(stage); continue
        if stage == current:
            started = state.get('initial_recovery_started_epoch', now) if stage == 'cap_recovery' else state.get('dispatch_requested_epoch', now)
            elapsed = max(0, now - started)
            over_estimate = elapsed > estimate
            estimate = max(0, estimate - elapsed)
        remaining += estimate
    target = state.get('target_completion_epoch')
    reliable_schedule = not missing and not over_estimate
    completion = now + remaining if reliable_schedule else None
    return {'forecast_completion_epoch': completion,
        'forecast_remaining_wall_seconds': remaining if reliable_schedule else None,
        'target_completion_epoch': target,
        'target_at_risk': bool(target and ((completion is not None and completion > target) or now + remaining > target)),
        'current_stage_over_estimate': over_estimate, 'stages_without_estimates': missing,
        'forecast_is_estimate_not_quality_override': True}


def status(pipeline_id, state, **extra):
    estimate = {}
    if CURRENT_STATE_PATH is not None and CURRENT_STATE_PATH.exists():
        estimate = forecast(json.loads(CURRENT_STATE_PATH.read_text()))
    write(C / 'pipeline/status.json', {'epoch': time.time(), 'pipeline_id': pipeline_id, 'state': state, **estimate, **extra})


def store_state(path, state):
    state['updated_epoch'] = time.time()
    write(path, state)


def freeze_sources(state):
    paths = ['scripts/coding_pipeline_plan.py', 'scripts/coding_parallel_pipeline.py']
    observed = {rel: sha(file_in(ROOT, rel)) for rel in paths}
    expected = state.get('supervisor_sources')
    if expected is not None and expected != observed:
        raise ValueError('Pipeline supervisor/builder source changed; inspect before continuation')
    state['supervisor_sources'] = observed


def builder(stage, state_path, output, state):
    """Only the builder determines whether an actual scientific gate passed."""
    freeze_sources(state)
    command = [sys.executable, 'scripts/coding_pipeline_plan.py', '--stage', stage,
        '--state', str(state_path), '--output', str(output)]
    log = C / 'pipeline' / (stage + '_builder.log')
    with log.open('ab') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
    if result.returncode == 20:
        stop = output.with_suffix('.stop.json')
        record = json.loads(stop.read_text())
        if record.get('scientific_stop') is not True:
            raise ValueError('Builder scientific stop lacks an explicit proof')
        return record
    if result.returncode:
        raise RuntimeError('Stage builder failed; inspect ' + str(log))
    if not output.is_file(): raise ValueError('Builder did not produce its immutable output')
    return None


def verify_export(receipt_path, output, bundle):
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('export_complete') is not True or Path(receipt['report_directory']).resolve() != output.resolve():
        raise ValueError('Review export is incomplete or its directory changed')
    record = receipt['bundle']
    if Path(record['path']).resolve() != bundle.resolve() or sha(bundle) != record['sha256']:
        raise ValueError('Review bundle does not match its verified receipt')
    if bundle.stat().st_size != record['bytes'] or record.get('source_trajectories_included') is not True:
        raise ValueError('Review bundle size or source-trajectory receipt differs')
    for name, expected in receipt['report_files'].items():
        if Path(name).name != name or sha(output / name) != expected:
            raise ValueError('Review report file differs from export receipt')
    return receipt


def publish_report(state_path, state):
    pipeline_id = state['pipeline_id']
    output = ROOT / 'results/coding_pilot_v1' / ('review_' + pipeline_id)
    bundle = ROOT / ('gearshift_coding_review_' + pipeline_id + '.zip')
    receipt_path = output / 'EXPORT.json'
    if receipt_path.exists():
        receipt = verify_export(receipt_path, output, bundle)
    else:
        report_script = ROOT / 'scripts/coding_parallel_report.py'
        if not report_script.exists():
            status(pipeline_id, 'waiting_for_review_exporter', no_gpu_waiting=True)
            return False
        if state.get('report_requested_epoch'):
            raise RuntimeError('Review export was interrupted without a completion receipt; inspect rather than overwrite')
        state['report_requested_epoch'] = time.time()
        state['report_source_sha256'] = sha(report_script)
        store_state(state_path, state)
        status(pipeline_id, 'building_review_bundle', no_gpu_waiting=True)
        with (C / 'pipeline/report.log').open('ab') as stream:
            result = subprocess.run([sys.executable, str(report_script), '--state', str(state_path),
                '--output', str(output), '--bundle', str(bundle)], cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT, timeout=1800)
        if result.returncode or not receipt_path.exists():
            raise RuntimeError('Review exporter did not produce a verified completion receipt')
        receipt = verify_export(receipt_path, output, bundle)
    if state.get('measurements_complete') and receipt.get('study_complete') is not True:
        raise ValueError('Exporter found the study incomplete; inspect the cohort evidence')
    state['review_export'] = {'path': str(receipt_path.relative_to(ROOT)), 'sha256': sha(receipt_path),
        'bundle': str(bundle.relative_to(ROOT)), 'bundle_sha256': receipt['bundle']['sha256']}
    state['finished_epoch'] = time.time(); store_state(state_path, state)
    status(pipeline_id, 'complete' if receipt.get('study_complete') else 'bounded_stop_report_ready',
        review_export=state['review_export'], study_complete=receipt.get('study_complete', False),
        forecast_completion_epoch=state['finished_epoch'], forecast_remaining_wall_seconds=0)
    return True


def wait_for_run(state_path, state, stage, run_id):
    if stage == 'cap_recovery' and 'initial_recovery_started_epoch' not in state:
        dispatch = C / 'parallel' / run_id / 'dispatch.json'
        if dispatch.exists():
            state['initial_recovery_started_epoch'] = json.loads(dispatch.read_text())['epoch']
            store_state(state_path, state)
    complete = read_completion(ROOT, run_id)
    if complete is None:
        heartbeat = C / 'controller_heartbeats' / (run_id + '.json')
        if heartbeat.exists():
            age = time.time() - json.loads(heartbeat.read_text()).get('epoch', 0)
            if age > 360:
                raise RuntimeError('Stage controller heartbeat lost; watchdog owns cleanup; no automatic relaunch')
        elif stage != 'cap_recovery' and time.time() - state.get('dispatch_requested_epoch', time.time()) > 120:
            raise RuntimeError('Dispatch intent has no live controller; inspect before any retry')
        status(state['pipeline_id'], 'waiting_for_stage', stage=stage, run_id=run_id)
        return False
    clean, remaining = confirmed_clean(run_id)
    if not clean:
        first = state.setdefault('cleanup_wait_started_epoch', time.time())
        store_state(state_path, state)
        if time.time() - first > 180:
            raise RuntimeError('Stage cleanup remains incomplete: ' + str(remaining))
        status(state['pipeline_id'], 'waiting_for_verified_cleanup', stage=stage, resources=remaining)
        return False
    state.setdefault('completed_runs', {})[stage] = run_id
    for key in ['current_stage', 'current_plan', 'current_plan_sha256', 'dispatch_requested_epoch', 'dispatcher_pid', 'cleanup_wait_started_epoch']:
        state.pop(key, None)
    store_state(state_path, state)
    status(state['pipeline_id'], 'checking_scientific_gates', completed_stage=stage, run_id=run_id)
    return True


def launch_plan(state_path, state, plan_path):
    plan = json.loads(plan_path.read_text())
    approval_path = ROOT / state.get('approval_path', 'evidence/coding_pilot_v1/control/parallel_500h_approved.json')
    if sha(approval_path) != state['approval_sha256']:
        raise ValueError('Pipeline approval changed')
    validate_plan(plan, ROOT, state['approval_sha256'])
    run_control = C / 'parallel' / plan['run_id']
    if run_control.exists() or state.get('dispatch_requested_epoch'):
        raise ValueError('One-shot dispatch already requested; observe instead of relaunching')
    state['current_stage'] = plan['stage']
    state['current_plan'] = str(plan_path.relative_to(ROOT))
    state['current_plan_sha256'] = sha(plan_path)
    state['dispatch_requested_epoch'] = time.time()
    store_state(state_path, state)  # Persist intent before a child can allocate.
    log_path = C / 'pipeline' / (plan['stage'] + '_dispatcher.log')
    with log_path.open('ab') as stream:
        child = subprocess.Popen([sys.executable, 'scripts/coding_parallel_session.py', '--plan', str(plan_path)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    state['dispatcher_pid'] = child.pid
    store_state(state_path, state)
    status(state['pipeline_id'], 'stage_dispatched', stage=plan['stage'], run_id=plan['run_id'])


def step(state_path):
    global CURRENT_STATE_PATH
    CURRENT_STATE_PATH = state_path
    state = json.loads(state_path.read_text())
    safe_id(state['pipeline_id'])
    if state.get('schema') != 1: raise ValueError('Unsupported pipeline state')
    freeze_sources(state)
    store_state(state_path, state)
    if state.get('scientific_stop') or state.get('measurements_complete'):
        return publish_report(state_path, state)
    stage = next_stage(state)
    if stage == 'cap_recovery':
        wait_for_run(state_path, state, stage, state['initial_recovery_run'])
        return False
    if state.get('current_plan'):
        path = file_in(ROOT, state['current_plan'])
        if sha(path) != state['current_plan_sha256']:
            raise ValueError('Dispatched immutable plan changed')
        plan = json.loads(path.read_text())
        if plan['stage'] != stage or state['current_stage'] != stage:
            raise ValueError('Current dispatch is outside fixed pipeline order')
        wait_for_run(state_path, state, stage, plan['run_id'])
        return False
    directory = ROOT / 'configs/coding_pilot_v1/pipeline_plans' / state['pipeline_id']
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (stage + '.json')
    # A builder-produced file may survive a supervisor interruption. It is
    # checked and dispatched once; an existing dispatch intent is never retried.
    if not output.exists():
        status(state['pipeline_id'], 'validating_next_stage', stage=stage)
        stop = builder(stage, state_path, output, state)
        state = json.loads(state_path.read_text())
        if stop:
            state['scientific_stop'] = stop
            state['finished_epoch'] = time.time(); store_state(state_path, state)
            status(state['pipeline_id'], 'scientific_stop', **{**stop, 'stage': stage})
            return publish_report(state_path, state)
    if stage == 'finished':
        receipt = json.loads(output.read_text())
        if receipt.get('passed') is not True:
            raise ValueError('Final cohort verification did not pass')
        state['measurements_complete'] = True; state['completion_proof'] = str(output.relative_to(ROOT))
        state['finished_epoch'] = time.time(); store_state(state_path, state)
        status(state['pipeline_id'], 'measurements_complete', completion_proof=state['completion_proof'],
            report_and_review_bundle_pending=True)
        return publish_report(state_path, state)
    launch_plan(state_path, state, output)
    return False


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=15); parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 30: parser.error('Poll interval must be 1–30 seconds')
    path = args.state.resolve(); (C / 'pipeline').mkdir(parents=True, exist_ok=True)
    initial = json.loads(path.read_text())
    with (C / 'pipeline/supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                if step(path) or args.once: break
                time.sleep(args.poll_seconds)
        except BaseException as exc:
            status(initial['pipeline_id'], 'stopped_inspect', error=str(exc), automatic_retry=False)
            raise


if __name__ == '__main__': main()
