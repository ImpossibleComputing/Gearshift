"""Meaningful CPU controls for complete state continuation and arm arithmetic."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gearshift.coding_control import sha, write
from gearshift.coding_coverage import paired_schedules
from gearshift.coding_coverage_runtime import paired_update
from gearshift.coding_coverage_v2_checkpoint import (
    capture_state, load_checkpoint, refresh_index, restore_state, save_checkpoint,
    state_difference, verify_checkpoint)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from coding_coverage_v2_train import constant_scheduler, resume_preflight, single_update


def identity(arm='FIXED'):
    return {'experiment_id': 'test_clean_v2', 'arm': arm, 'code_commit': 'a' * 40,
            'config_sha256': 'b' * 64, 'schedules_sha256': 'c' * 64,
            'corpus_sha256': 'd' * 64, 'selected_checkpoint_sha256': 'e' * 64,
            'models': {'source': {'revision': 'source'}, 'receiver': {'revision': 'receiver'}}}


def components():
    model = torch.nn.Linear(3, 2, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.013, weight_decay=0., foreach=False, fused=False)
    return model, optimizer, constant_scheduler(optimizer)


def stochastic_step(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    # All three CPU RNG sources materially affect the update. Comparing only
    # mapper weights without restoring any one RNG/Adam state cannot pass.
    x = torch.randn(7, 3, dtype=torch.float64) + random.random() + float(np.random.normal())
    target = torch.randn(7, 2, dtype=torch.float64)
    loss = (model(x) - target).square().mean(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def test_resume_matches_uninterrupted_next_updates_and_all_rng_sources(tmp_path):
    torch.manual_seed(31); random.seed(32); np.random.seed(33)
    model, optimizer, scheduler = components()
    for _ in range(3): stochastic_step(model, optimizer, scheduler)
    logs = [{'step': i} for i in range(1, 4)]
    saved = save_checkpoint(tmp_path / 'step_0003', model, optimizer, scheduler, 3, identity(), logs)
    control_losses = [stochastic_step(model, optimizer, scheduler) for _ in range(3)]
    expected = capture_state(model, optimizer, scheduler, 6, identity(), logs + [{'step': i} for i in range(4, 7)])
    new_model, new_optimizer, new_scheduler = components()
    random.seed(999); np.random.seed(888); torch.manual_seed(777)
    manifest, payload = load_checkpoint(tmp_path / 'step_0003', new_model, new_optimizer, new_scheduler, identity())
    assert manifest['complete_resumable'] and payload['schedule_position'] == 3
    actual_losses = [stochastic_step(new_model, new_optimizer, new_scheduler) for _ in range(3)]
    actual = capture_state(new_model, new_optimizer, new_scheduler, 6, identity(), expected['training_log'])
    assert actual_losses == control_losses
    assert state_difference(expected, actual)['exact']
    assert saved['full']['sha256'] == sha(tmp_path / 'step_0003/full.pt')
    assert payload['optimizer']['state'] and payload['scheduler']['last_epoch'] == 3
    # Negative control: the same weights/RNG with a fresh Adam state changes the
    # next update. This is the failure mode the prior mapper-only saves allowed.
    reset_model, reset_optimizer, reset_scheduler = components()
    reset_model.load_state_dict(payload['state_dict'])
    reset_payload = copy.deepcopy(payload)
    reset_payload['optimizer'] = reset_optimizer.state_dict()
    reset_payload['scheduler'] = reset_scheduler.state_dict()
    restore_state(reset_payload, reset_model, reset_optimizer, reset_scheduler, identity())
    for _ in range(3): stochastic_step(reset_model, reset_optimizer, reset_scheduler)
    assert not state_difference(expected['state_dict'], reset_model.state_dict())['exact']


def test_checkpoint_rejects_identity_change_and_immutable_overwrite(tmp_path):
    m, o, s = components(); path = tmp_path / 'step_0000'
    save_checkpoint(path, m, o, s, 0, identity())
    with pytest.raises(ValueError, match='identity mismatch'): verify_checkpoint(path, identity('ROTATING'))
    with pytest.raises(ValueError, match='already exists'): save_checkpoint(path, m, o, s, 0, identity())


def test_checkpoint_corruption_cannot_load_or_enter_reader_index(tmp_path):
    m, o, s = components(); path = tmp_path / 'checkpoints/step_0000'
    save_checkpoint(path, m, o, s, 0, identity())
    with (path / 'full.pt').open('ab') as out: out.write(b'corruption')
    with pytest.raises(ValueError, match='verification failed'): verify_checkpoint(path, identity())
    with pytest.raises(ValueError, match='verification failed'): refresh_index(path.parent, identity())


def test_full_checkpoint_requires_optimizer_even_if_file_is_rehashed(tmp_path):
    m, o, s = components(); path = tmp_path / 'step_0000'
    save_checkpoint(path, m, o, s, 0, identity())
    payload = torch.load(path / 'full.pt', weights_only=True); del payload['optimizer']
    torch.save(payload, path / 'full.pt')
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['files']['full.pt'] = {'sha256': sha(path / 'full.pt'), 'bytes': (path / 'full.pt').stat().st_size}
    write(path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='missing resumable'): verify_checkpoint(path, identity())


def test_uncommitted_directory_never_published_and_roundtrip_failure_preserved(tmp_path, monkeypatch):
    import gearshift.coding_coverage_v2_checkpoint as module
    m, o, s = components(); root = tmp_path / 'checkpoints'; root.mkdir()
    (root / '.incomplete-step_0000-crash').mkdir()
    assert refresh_index(root, identity())['checkpoints'] == []
    def fail(*args, **kwargs): raise ValueError('simulated verification failure')
    monkeypatch.setattr(module, '_tensor_file', fail)
    with pytest.raises(ValueError, match='simulated'): save_checkpoint(root / 'step_0000', m, o, s, 0, identity())
    assert not (root / 'step_0000').exists()
    assert len(list(root.glob('.incomplete-*'))) == 2
    assert refresh_index(root, identity())['checkpoints'] == []


def test_all_committed_checkpoints_retained_and_mapper_matches_full(tmp_path):
    m, o, s = components(); root = tmp_path / 'checkpoints'; logs = []
    for step in range(3):
        if step:
            stochastic_step(m, o, s); logs.append({'step': step})
        save_checkpoint(root / f'step_{step:04d}', m, o, s, step, identity(), logs)
    result = refresh_index(root, identity())
    assert result['latest_step'] == 2
    assert [r['step'] for r in result['checkpoints']] == [0, 1, 2]
    for row in result['checkpoints']:
        full = torch.load(row['full']['path'], weights_only=True)
        mapper = torch.load(row['mapper']['path'], weights_only=True)
        assert state_difference(full['state_dict'], mapper['state_dict'])['exact']


def test_incomplete_schedule_log_is_not_checkpointable(tmp_path):
    m, o, s = components()
    with pytest.raises(ValueError, match='exactly once'):
        save_checkpoint(tmp_path / 'step_0002', m, o, s, 2, identity(), [{'step': 2}])


class ToyRuntime:
    def __init__(self, model):
        self.mapper = model
        self.source = SimpleNamespace(model=torch.nn.Linear(1, 1).requires_grad_(False), input_token_count=0)
        self.receiver = SimpleNamespace(model=torch.nn.Linear(1, 1).requires_grad_(False), input_token_count=0)
        self.guard = lambda: None
        self.telemetry = SimpleNamespace(sample=lambda **kw: None)
        self.cache_gradient_checks = []
    def pair(self, obj):
        cache = ((torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2)),)
        return cache, tuple((a.clone(), b.clone()) for a, b in cache)
    def kl(self, obj, positions, protocol, sp, tp):
        assert protocol == 'natural_handoff_boundary'
        x = torch.tensor(positions, dtype=torch.float64)[:, None] / 100 + torch.tensor([[.2, .4, .6]])
        loss = self.mapper(x).square().sum(-1) + .01
        self.cache_gradient_checks.append([{'finite': True, 'absolute_sum': 1.}])
        return loss


def sample_corpus():
    return [{'task_id': f'task/{i}', 'teacher_answer': {'answer_ids': list(range(540 + i))},
             'source_history': {'prefix_ids': [1, 2, 3]}} for i in range(4)]


@pytest.mark.parametrize('arm', ['FIXED', 'ROTATING'])
def test_independent_update_is_exactly_the_previous_paired_arm_arithmetic(arm):
    torch.manual_seed(99)
    model, optimizer, scheduler = components(); original = copy.deepcopy(model.state_dict())
    solo = ToyRuntime(model)
    histories = sample_corpus(); lookup = {h['task_id']: h for h in histories}
    schedules = paired_schedules(histories, updates=3)
    paired_runtimes = {}; paired_optimizers = {}
    for name in ('FIXED', 'ROTATING'):
        pm, po, _ = components(); pm.load_state_dict(original)
        paired_runtimes[name] = ToyRuntime(pm); paired_optimizers[name] = po
    for step in range(3):
        items = {a: schedules[a][step] for a in ('FIXED', 'ROTATING')}
        result = single_update(items[arm], lookup, solo, optimizer)
        scheduler.step()
        reference = paired_update(items, lookup, paired_runtimes, paired_optimizers, ('FIXED', 'ROTATING'))
        assert result['kl'] == reference['arms'][arm]['kl']
        assert result['gradient_norm_before_clipping'] == reference['arms'][arm]['gradient_norm_before_clipping']
        assert state_difference(model.state_dict(), paired_runtimes[arm].mapper.state_dict())['exact']
        assert state_difference(optimizer.state_dict()['state'], paired_optimizers[arm].state_dict()['state'])['exact']


def test_early_resume_control_restores_pristine_weights_optimizer_scheduler_and_rng(tmp_path):
    m, o, s = components(); runtime = ToyRuntime(m)
    histories = sample_corpus(); lookup = {h['task_id']: h for h in histories}
    schedule = paired_schedules(histories, updates=3)['ROTATING']
    ident = identity('ROTATING'); before = capture_state(m, o, s, 0, ident)
    c = {'root': tmp_path, 'guard': lambda: None, 'publish': lambda **kw: None}
    report = resume_preflight(c, m, o, s, runtime, schedule, lookup, ident)
    assert report['passed'] and report['loss_exact'] and report['gradient_norm_exact']
    assert state_difference(before, capture_state(m, o, s, 0, ident))['exact']
    assert not o.state and s.last_epoch == 0
    assert json.loads((tmp_path / 'pristine_restore.json').read_text())['clean_primary_start_restored']


def test_early_resume_control_detects_lost_optimizer_and_restores_clean_state(tmp_path, monkeypatch):
    import coding_coverage_v2_train as module
    m, o, s = components(); runtime = ToyRuntime(m)
    histories = sample_corpus(); lookup = {h['task_id']: h for h in histories}
    schedule = paired_schedules(histories, updates=3)['FIXED']; ident = identity()
    before = capture_state(m, o, s, 0, ident); original_load = module.load_checkpoint
    def lose_optimizer(*args, **kwargs):
        result = original_load(*args, **kwargs)
        args[2].state.clear()
        return result
    monkeypatch.setattr(module, 'load_checkpoint', lose_optimizer)
    c = {'root': tmp_path, 'guard': lambda: None, 'publish': lambda **kw: None}
    with pytest.raises(RuntimeError, match='continuation control failed'):
        resume_preflight(c, m, o, s, runtime, schedule, lookup, ident)
    assert not json.loads((tmp_path / 'resume_preflight.json').read_text())['passed']
    assert state_difference(before, capture_state(m, o, s, 0, ident))['exact']
