"""CPU-only synthetic protocol tests. They are NOT an actual H200 proof."""
import copy
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import pytest
import torch

from gearshift.coding_control import sha, write
from gearshift import coding_confirmation_sampling  # Preload before tiny CPU forks.
from gearshift.core import CacheExtractor
from scripts import coding_confirmation_resume_smoke as s

ROOT = Path(__file__).resolve().parents[1]


class Cache:
    def __init__(self):
        self.tokens, self.state, self.layers = [], 0, []

    def get_seq_length(self):
        return len(self.tokens)


class Backend:
    name, device, eos = 'synthetic-model', 'cpu', {99}
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids),
                                apply_chat_template=lambda *a, **kw: [1, 2])

    def __init__(self, drift=False):
        self.drift = drift
        self.sampling_identity = {'model_id': self.name, 'revision': 'a' * 40,
                                 'runtime_identity_sha256': 'b' * 64,
                                 'declaration_sha256': 'c' * 64}

    def forward(self, ids, cache=None):
        cache = cache or Cache()
        cache.tokens.extend(ids)
        # Segmentation dependence makes whole-prefix replay an invalid shortcut.
        cache.state = (cache.state * 13 + sum(ids) + len(ids) ** 2) % 997
        keys = torch.tensor(cache.tokens, dtype=torch.float32).reshape(1, 1, -1, 1)
        cache.layers = [SimpleNamespace(keys=keys, values=keys + cache.state)]
        logits = torch.arange(32).float().reshape(1, 1, 32) / 20
        logits = logits + torch.sin(torch.arange(32).float() + cache.state).reshape(1, 1, 32) / 5
        if self.drift:
            logits[0, 0, 3] += .125
        return SimpleNamespace(logits=logits, past_key_values=cache)

    def prefill_chunked(self, ids, chunk=512):
        cache = None
        for start in range(0, len(ids), chunk):
            out = self.forward(ids[start:start + chunk], cache)
            cache = out.past_key_values
        return out


def fake_base(repo, plan, case, models, history):
    backend = models['receiver'][0]
    cache = backend.prefill_chunked(history['prefix_ids']).past_key_values
    # Three distinct deterministic conditioning paths; no trained mapper used.
    cache.state += {'receiver_native': 0, 'receiver_mapped': 101, 'receiver_hybrid': 202}[case]
    keys = cache.layers[0].keys
    cache.layers[0].values = keys + cache.state
    return cache


def child(folder, plan, case, phase, drift=False, early_terminal=False):
    import gearshift.coding_inference as inference
    torch.set_num_threads(1)
    inference.memory_record = lambda: {'passed': True, 'test_fake_CPU_only': True}
    s.answer_base = fake_base
    models = {}
    for role in ('source', 'receiver'):
        backend = Backend(drift=drift)
        if early_terminal:
            backend.eos = set(range(32))
        models[role] = (backend, s.Observer(backend))
    attempt = {'pid': os.getpid(), 'pod_id': 'synthetic-cpu-test', 'host': 'cpu-only',
               'process_started_ns': time.time_ns(), 'runtime': {'test_fake_CPU_only': True}}
    s.run_case(ROOT, folder, plan, case, phase, models, attempt, lambda: None)


def spawn(folder, plan, case, phase, drift=False, early_terminal=False):
    process = multiprocessing.get_context('fork').Process(target=child,
                    args=(folder, plan, case, phase, drift, early_terminal))
    process.start(); process.join(20)
    if process.is_alive():
        process.kill(); process.join()
        pytest.fail('Synthetic process exceeded bounded deadline')
    return process.exitcode


@pytest.fixture(scope='module')
def proof_template(tmp_path_factory):
    folder = tmp_path_factory.mktemp('resume_smoke') / 'smoke'
    prepared = s.prepare(ROOT, folder, cap=8)
    plan = s.read(folder / 'plan.json')
    for case in s.CASES:
        assert spawn(folder, plan, case, 'baseline') == 0
        assert spawn(folder, plan, case, 'interrupt') == s.EXIT_INTERRUPTED
        state = s.read(folder / 'restarted' / case / 'resume.json')
        assert state['state'] == 'running'  # Abrupt process exit bypassed exception cleanup.
        assert len(state['tokens']) == 1
        assert spawn(folder, plan, case, 'resume') == 0
    return folder, plan, prepared['plan_sha256']


@pytest.fixture
def proof(tmp_path, proof_template):
    source, plan, plan_sha = proof_template
    folder = tmp_path / 'smoke'
    shutil.copytree(source, folder)
    return folder, copy.deepcopy(plan), plan_sha


def test_four_paths_real_process_exit_reconstruct_all_bytes(proof):
    folder, plan, plan_sha = proof
    result = s.verify(ROOT, folder, plan_sha)
    assert result['all_cases_passed']
    assert not result['cross_pod_resume_for_all_cases']
    for row in result['cases']:
        assert row['generated_tokens'] == 8
        assert all(row['checks'].values())
    assert not list(folder.rglob('*.pt'))
    assert not list(folder.rglob('*.safetensors'))


@pytest.mark.parametrize('field', ['cache', 'next_logits_sha256'])
def test_boundary_reconstruction_mismatch_cannot_pass(proof, field):
    folder, plan, _ = proof
    path = folder / 'proof/receiver_hybrid/resume_result.json'
    row = s.read(path)
    row['reconstructed_boundary'][field] = 'different'
    write(path, row)
    actual = s.compare_case(folder, plan, 'receiver_hybrid')
    assert not actual['passed']
    assert not actual['checks']['all_reconstructed_KV_bytes_equal']


def test_changed_raw_record_and_saved_state_rejected(proof):
    folder, plan, _ = proof
    path = folder / 'baseline/receiver_native/answer_record.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='raw synthetic record changed'):
        s.compare_case(folder, plan, 'receiver_native')
    path = folder / 'restarted/source_reasoning/resume.json'
    row = s.read(path); row['tokens'][0] += 1; write(path, row)
    with pytest.raises(ValueError, match='durable state changed'):
        s.compare_case(folder, plan, 'source_reasoning')


def test_no_reroll_or_repeated_phase(proof):
    folder, plan, _ = proof
    assert spawn(folder, plan, 'source_reasoning', 'baseline') != 0
    assert spawn(folder, plan, 'receiver_native', 'resume') != 0


def test_actual_frozen_validator_rejects_drift_before_new_samples(tmp_path):
    folder = tmp_path / 'smoke'
    s.prepare(ROOT, folder, cap=8)
    plan = s.read(folder / 'plan.json')
    assert spawn(folder, plan, 'source_reasoning', 'baseline') == 0
    assert spawn(folder, plan, 'source_reasoning', 'interrupt') == 86
    before = s.read(folder / 'restarted/source_reasoning/resume.json')['tokens']
    assert spawn(folder, plan, 'source_reasoning', 'resume', drift=True) != 0
    assert s.read(folder / 'restarted/source_reasoning/resume.json')['tokens'] == before
    assert (folder / 'restarted/source_reasoning/reconstruction_failure.json').is_file()
    assert not (folder / 'proof/source_reasoning/resume_result.json').exists()


def test_early_terminal_cannot_be_misreported_as_interrupted(tmp_path):
    folder = tmp_path / 'smoke'
    s.prepare(ROOT, folder, cap=8)
    plan = s.read(folder / 'plan.json')
    assert spawn(folder, plan, 'source_reasoning', 'interrupt', early_terminal=True) not in (0, 86)
    assert (folder / 'proof/source_reasoning/interruption_not_exercised.json').is_file()
    assert not (folder / 'proof/source_reasoning/interruption.json').exists()
    assert not (folder / 'proof/source_reasoning/resume_result.json').exists()


def test_runner_requires_exact_hard_exit_and_stops_on_unexpected_failure(tmp_path, monkeypatch):
    prepared = s.prepare(ROOT, tmp_path / 'smoke', cap=8)
    calls = []

    class Process:
        pid = 1234
        def __init__(self, command, **kw):
            calls.append(command)
        def wait(self, **kw):
            # Baseline closes normally; an interrupt that returns zero is invalid.
            return 0
    monkeypatch.setattr(s.subprocess, 'Popen', Process)
    with pytest.raises(RuntimeError, match='expected completion'):
        s.run(ROOT, tmp_path / 'smoke', prepared['plan_sha256'], 'unused', 'a' * 64)
    assert len(calls) == 2
    receipts = s.read(tmp_path / 'smoke/launch/process_receipts.json')
    assert receipts[-1]['expected_exit_code'] == 86
    assert receipts[-1]['exit_code'] == 0


@pytest.mark.parametrize('cap,cut', [(257, 1), (1, 1), (128, 2), (65, 65), (128, 129)])
def test_invalid_bounds_rejected_before_inputs(tmp_path, cap, cut):
    with pytest.raises(ValueError):
        s.prepare(tmp_path / 'no_repository', tmp_path / 'out', cap=cap, interrupt_after=cut)


def test_plan_drift_or_harness_drift_rejected(tmp_path):
    prepared = s.prepare(ROOT, tmp_path / 'out')
    assert s.validate_plan(ROOT, tmp_path / 'out', prepared['plan_sha256'])['task_id'] == s.TASK
    plan = s.read(tmp_path / 'out/plan.json'); plan['prompt'] = 'Other prompt'
    write(tmp_path / 'out/plan.json', plan)
    with pytest.raises(ValueError, match='plan hash'):
        s.validate_plan(ROOT, tmp_path / 'out', prepared['plan_sha256'])
    with pytest.raises(ValueError, match='synthetic protocol'):
        s.validate_plan(ROOT, tmp_path / 'out', sha(tmp_path / 'out/plan.json'))


def test_observer_preserves_original_call_segmentation():
    a, b = Backend(), Backend()
    observer = s.Observer(b)
    prompt = [1] * 520
    out_a, out_b = a.prefill_chunked(prompt), b.prefill_chunked(prompt)
    assert observer.forward_lengths == [512, 8]
    assert s.cache_fingerprint(out_a.past_key_values) == s.cache_fingerprint(out_b.past_key_values)
    assert torch.equal(out_a.logits, out_b.logits)
