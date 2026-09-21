#!/usr/bin/env python3
"""One independent coverage arm; immutable checkpoints never await evaluation.

The worker supplies an offline, deadline-only guard and the already staged frozen
models. This file has no provider API, SSH, mirror, heartbeat or network calls.
"""
from __future__ import annotations
import copy
import gc
import json
from pathlib import Path
import random
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gearshift.coding_control import bind, digest, write
from gearshift.coding_coverage import ARMS, check_pair, grouped, START_SHA
from gearshift.coding_inference import sync
from gearshift.coding_coverage_v2_checkpoint import (
    capture_state, checkpoint_receipt, load_checkpoint, refresh_index, restore_state,
    save_checkpoint, state_difference, state_tensor_sha256, verify_checkpoint)

TARGET = 1024
CHECKPOINTS = (0, 128, 256, 512, 768, 1024)


def constant_scheduler(optimizer):
    # The prior recipe used a constant LR. The explicit scheduler preserves that
    # exact behavior while making its continuation position a saved state.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.)


def single_update(item, lookup, runtime, optimizer):
    """The unchanged per-arm arithmetic/order from coverage paired_update."""
    groups = grouped(item)
    if item['predictions'] != 32 or sum(map(len, groups.values())) != 32:
        raise ValueError('Every update requires exactly 32 scored positions')
    for tid, positions in groups.items():
        n = len(lookup[tid]['teacher_answer']['answer_ids'])
        if len(set(positions)) != len(positions) or not all(0 <= p < n for p in positions):
            raise ValueError('Duplicate, padded or unavailable prediction')
    optimizer.zero_grad(set_to_none=True); runtime.cache_gradient_checks = []
    before = time.monotonic(); preparation = 0.; loss_total = 0.; forward_backward = 0.; tasks = []
    for tid, positions in groups.items():
        obj = lookup[tid]; runtime.guard(); sync(); start = time.monotonic()
        sp, tp = runtime.pair(obj); sync(); pair_seconds = time.monotonic() - start
        preparation += pair_seconds
        versions = [x._version for p in (*sp, *tp) for x in p]
        runtime.guard(); sync(); start = time.monotonic()
        runtime.telemetry.sample(stage='coverage_gradient', task_id=tid,
            sequence_length=len(obj['source_history']['prefix_ids']) + max(positions) + 1)
        values = runtime.kl(obj, positions, 'natural_handoff_boundary', sp, tp)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite training KL')
        loss = values.sum() / 32
        loss_total += float(loss.detach()); loss.backward(); sync()
        forward_backward += time.monotonic() - start
        del loss, values
        if versions != [x._version for p in (*sp, *tp) for x in p]:
            raise RuntimeError('Historical cache mutated by training')
        tasks.append({'task_id': tid, 'pair_preparation_seconds': pair_seconds,
                      'predictions': len(positions), 'cache_versions_unchanged': True})
        del sp, tp; gc.collect()
    checks = [r for part in runtime.cache_gradient_checks for r in part]
    if not checks or not all(r['finite'] and r['absolute_sum'] > 0 for r in checks):
        raise RuntimeError('Missing, zero or nonfinite cache gradient')
    if any(p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum() > 0 for p in runtime.mapper.parameters()):
        raise RuntimeError('Missing, zero or nonfinite mapper gradient')
    if any(p.grad is not None for b in [runtime.source, runtime.receiver] for p in b.model.parameters()):
        raise RuntimeError('Frozen language model received gradients')
    norm = float(torch.nn.utils.clip_grad_norm_(runtime.mapper.parameters(), 1, error_if_nonfinite=True))
    runtime.guard(); sync(); start = time.monotonic()
    optimizer.step(); optimizer.zero_grad(set_to_none=True); sync()
    return {'kl': loss_total, 'gradient_norm_before_clipping': norm, 'gradient_predictions': 32,
            'cache_gradient_tensors_checked': len(checks), 'cache_gradients_finite_nonzero': True,
            'continuation_forward_backward_seconds': forward_backward,
            'optimizer_seconds': time.monotonic() - start, 'cache_preparation_seconds': preparation,
            'tasks': tasks, 'wall_seconds': time.monotonic() - before, 'normalization_per_arm': 32,
            'execution': 'Independent arm; complete immutable historical prefixes; unchanged continuation and loss.'}


def _numerical_state(payload):
    # Timing/progress records are intentionally excluded; every state that can
    # influence subsequent numerical updates remains in this comparison.
    return {k: v for k, v in payload.items() if k != 'training_log'}


def resume_preflight(c, mapper, optimizer, scheduler, runtime, schedule, lookup, identity):
    """Two warm-up updates, then the same third update before/after disk reload.

    All numerical state is restored to the pristine original96 initialization
    afterwards. No preflight update can enter the primary run.
    """
    root = c['root'] / 'resume_preflight'
    pristine = capture_state(mapper, optimizer, scheduler, 0, identity)
    token_counts = [(b, getattr(b, 'input_token_count', None)) for b in (runtime.source, runtime.receiver)]
    rows = []; started = time.time(); arm = identity['arm']; passed = False
    try:
        for index in range(2):
            c['guard'](); c['publish'](stage='actual_model_resume_preflight', arm=arm, step=index + 1)
            result = single_update(schedule[index], lookup, runtime, optimizer); scheduler.step()
            rows.append({'step': index + 1, **result})
        receipt = save_checkpoint(root / 'after_two_updates', mapper, optimizer, scheduler, 2, identity, rows)
        uninterrupted_result = single_update(schedule[2], lookup, runtime, optimizer); scheduler.step()
        uninterrupted_rows = rows + [{'step': 3, **uninterrupted_result}]
        uninterrupted = capture_state(mapper, optimizer, scheduler, 3, identity, uninterrupted_rows)
        manifest, loaded = load_checkpoint(root / 'after_two_updates', mapper, optimizer, scheduler, identity)
        if loaded['schedule_position'] != 2:
            raise ValueError('Resume preflight loaded a different next-update position')
        del loaded
        resumed_result = single_update(schedule[2], lookup, runtime, optimizer); scheduler.step()
        resumed = capture_state(mapper, optimizer, scheduler, 3, identity, rows + [{'step': 3, **resumed_result}])
        comparison = state_difference(_numerical_state(uninterrupted), _numerical_state(resumed))
        loss_equal = uninterrupted_result['kl'] == resumed_result['kl']
        grad_equal = uninterrupted_result['gradient_norm_before_clipping'] == resumed_result['gradient_norm_before_clipping']
        passed = comparison['exact'] and loss_equal and grad_equal
        report = {'passed': passed, 'arm': arm, 'actual_frozen_models': True,
                  'updates_before_interrupt': 2, 'compared_next_update': 3,
                  'checkpoint': receipt, 'state_comparison': comparison,
                  'uninterrupted_kl': uninterrupted_result['kl'], 'resumed_kl': resumed_result['kl'],
                  'loss_exact': loss_equal, 'gradient_norm_exact': grad_equal,
                  'numerical_envelope': 'Exact equality of mapper, optimizer, scheduler, all RNG states and next-update KL/gradient norm; no relaxed tolerance.',
                  'preflight_updates_excluded_from_primary': True, 'seconds': time.time() - started}
        write(c['root'] / 'resume_preflight.json', report)
        if not passed:
            raise RuntimeError('Actual-model exact checkpoint continuation control failed')
        return report
    finally:
        restore_state(pristine, mapper, optimizer, scheduler, identity)
        restored = capture_state(mapper, optimizer, scheduler, 0, identity)
        reset = state_difference(pristine, restored)
        for backend, value in token_counts:
            if value is not None: backend.input_token_count = value
        write(c['root'] / 'pristine_restore.json', {'exact': reset['exact'], 'comparison': reset,
            'mapper_tensor_sha256': state_tensor_sha256(pristine['state_dict']),
            'optimizer_initially_empty': not pristine['optimizer']['state'],
            'preflight_passed': passed, 'clean_primary_start_restored': reset['exact']})
        if not reset['exact']:
            raise RuntimeError('Could not restore clean primary initialization after preflight')
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def train(c, state, arm=None):
    """Run/continue one arm to update1024; immutable checkpoint readers may overlap."""
    d, training, validation, schedules, panels, source, receiver, mappers, runtimes, optimizers = state
    arm = arm or c['spec']['arm']
    if arm not in ARMS or any(set(x) != {arm} for x in (mappers, runtimes, optimizers)):
        raise ValueError('One independent arm/mapper/optimizer per training process is required')
    identity = c['spec']['checkpoint_identity']
    if identity['arm'] != arm or identity['selected_checkpoint_sha256'] != START_SHA:
        raise ValueError('Primary training must start from the original selected96 identity')
    if identity['schedules_sha256'] != d['schedules_sha256'] or identity['corpus_sha256'] != d['corpus_manifest_sha256']:
        raise ValueError('Checkpoint identity does not bind the frozen schedules and corpus')
    if c['spec'].get('target_updates', TARGET) != TARGET:
        raise ValueError('The clean primary target is exactly 1024 additional updates')
    if c['spec'].get('checkpoint_updates', list(CHECKPOINTS)) != list(CHECKPOINTS):
        raise ValueError('The common checkpoint schedule differs')
    lookup = {h['task_id']: h for h in training}
    if len(training) != 104 or len(validation) != 21:
        raise ValueError('The exact 104/21 corpus is required')
    if any(len(schedules[a]) < TARGET for a in ARMS):
        raise ValueError('Incomplete frozen paired schedules')
    for index in range(TARGET):
        check_pair(schedules['FIXED'][index], schedules['ROTATING'][index], lookup)
    root = Path(c['root']); root.mkdir(parents=True, exist_ok=True); c['root'] = root
    bind(root / 'training_identity.json', {'checkpoint_identity': identity,
         'checkpoint_identity_sha256': digest(identity), 'target_updates': TARGET,
         'checkpoint_updates': list(CHECKPOINTS), 'normalization': 32, 'hidden_tests_loaded': False,
         'scheduler': 'constant multiplier 1.0, identical to original constant LR',
         'network_required_after_initialization': False, 'evaluation_blocks_training': False})
    bind(root / 'frozen_schedule.json', {'arm': arm, 'items': schedules[arm][:TARGET],
         'paired_schedule_sha256': d['schedules_sha256']})
    mapper, runtime, optimizer = mappers[arm], runtimes[arm], optimizers[arm]
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError('The frozen optimizer must be AdamW')
    for group in optimizer.param_groups:
        for key in ('lr', 'betas', 'eps', 'weight_decay', 'foreach', 'fused'):
            expected = tuple(d['optimizer'][key]) if key == 'betas' else d['optimizer'][key]
            if group[key] != expected:
                raise ValueError('Optimizer recipe differs: ' + key)
    scheduler = constant_scheduler(optimizer)
    checkpoints_root = root / 'checkpoints'; checkpoints_root.mkdir(exist_ok=True)
    existing = sorted(checkpoints_root.glob('step_*'))
    requested = c['spec'].get('resume_checkpoint')
    if requested and Path(requested).resolve().parent != checkpoints_root.resolve():
        raise ValueError('Resume must use this arm\'s committed checkpoint directory')
    if requested and existing and Path(requested).resolve() != existing[-1].resolve():
        raise ValueError('Resume must use the latest fully committed checkpoint, not select an older state')
    start = 0; steps = []; started = time.time()
    try:
        if existing or requested:
            path = Path(requested) if requested else existing[-1]
            manifest, payload = load_checkpoint(path, mapper, optimizer, scheduler, identity)
            start = payload['step']; steps = copy.deepcopy(payload['training_log']); del payload
            if start not in CHECKPOINTS:
                raise ValueError('Resume checkpoint is not a declared common checkpoint')
            # Preserve logs from work after the last durable checkpoint rather
            # than silently claiming those updates never ran.
            old_log = root / 'training_steps.json'
            if old_log.exists():
                old = json.loads(old_log.read_text())
                if len(old) > start:
                    write(root / 'attempts' / f'undurable_training_{time.time_ns()}.json',
                          {'resumed_from_step': start, 'previous_logged_updates': old,
                           'reason': 'These later updates were not durably checkpointed and are replayed from complete saved state.'})
            write(root / 'resume_events' / f'{time.time_ns()}.json',
                  {'checkpoint': checkpoint_receipt(path, manifest), 'next_update': start + 1,
                   'optimizer_scheduler_rng_restored': True, 'attempt_identity': c.get('identity', {})})
            refresh_index(checkpoints_root, identity)
        else:
            if optimizer.state:
                raise ValueError('Clean restart requires a fresh empty AdamW state')
            seed = c['spec'].get('training_seed', 20260915)
            random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            write(root / 'optimizer_reset.json', {'selected_checkpoint_sha256': START_SHA,
                'fresh_adamw': True, 'initial_state_empty': True, 'configuration': d['optimizer'],
                'training_seed': seed, 'same_seed_and_schedule_across_arms': True})
            resume_preflight(c, mapper, optimizer, scheduler, runtime, schedules[arm], lookup, identity)
            save_checkpoint(checkpoints_root / 'step_0000', mapper, optimizer, scheduler, 0, identity)
            refresh_index(checkpoints_root, identity)
        write(root / 'training_steps.json', steps)
        # Span labels are descriptive only. They never choose the schedule or
        # change any optimizer input. No teacher-free evaluation happens here.
        from coding_coverage_experiment import span_audit, exposure_report
        audits = [span_audit(receiver.tokenizer, obj) for obj in training]
        write(root / 'answer_span_audit.json', audits)
        write(root / 'planned_exposure.json', exposure_report(training, schedules[arm], audits, TARGET))
        for step in CHECKPOINTS:
            if step <= start:
                write(root / 'coverage' / f'step_{step:04d}.json', exposure_report(training, schedules[arm], audits, step))
        for step in range(start + 1, TARGET + 1):
            c['guard'](); c['publish'](stage='independent_coverage_training', arm=arm, step=step, target_updates=TARGET)
            result = single_update(schedules[arm][step - 1], lookup, runtime, optimizer)
            scheduler.step()
            steps.append({'step': step, 'arm': arm, 'lr': [g['lr'] for g in optimizer.param_groups], **result})
            write(root / 'training_steps.json', steps)
            c['telemetry'].sample(stage='independent_update_committed', arm=arm, update=step)
            if step in CHECKPOINTS:
                c['guard']()
                save_checkpoint(checkpoints_root / f'step_{step:04d}', mapper, optimizer, scheduler, step, identity, steps)
                index = refresh_index(checkpoints_root, identity)
                write(root / 'coverage' / f'step_{step:04d}.json', exposure_report(training, schedules[arm], audits, step))
                c['publish'](stage='immutable_checkpoint_available', arm=arm, step=step,
                             manifest=str(root / 'checkpoint_manifest.json'), checkpoint=index['checkpoints'][-1])
        if start == 0:
            write(root / 'coverage' / 'step_0000.json', exposure_report(training, schedules[arm], audits, 0))
        result = {'arm': arm, 'target_updates': TARGET, 'completed_updates': len(steps), 'full_target_completed': len(steps) == TARGET,
                  'scored_positions': 32 * len(steps), 'last_durable_checkpoint': TARGET, 'resume_start_update': start,
                  'this_attempt_wall_seconds': time.time() - started, 'checkpoint_identity_sha256': digest(identity),
                  'hidden_tests_loaded': False, 'free_running_answers_generated': 0,
                  'primary_checkpoint_selection': 'Predeclared1024; no KL or answer scores consulted.'}
        write(root / 'training_complete.json', result)
        return result
    except BaseException as exc:
        failure = {'arm': arm, 'exception': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
                   'completed_logged_updates': len(steps), 'target_updates': TARGET, 'epoch': time.time(),
                   'checkpoint_identity_sha256': digest(identity), 'attempt_identity': c.get('identity', {})}
        write(root / 'failures' / f'{time.time_ns()}.json', failure)
        write(root / 'training_failure.json', failure)
        c['telemetry'].failure(exc)
        raise
