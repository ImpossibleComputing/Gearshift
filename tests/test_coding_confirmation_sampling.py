import json
from types import SimpleNamespace

import pytest
import torch

from gearshift import coding_confirmation_sampling as m
from gearshift import coding_inference as old
from gearshift.coding_control import digest, write
from gearshift.coding_recovery_answer import answer_instrumented


class Cache:
    def __init__(self, tokens=(), state=0):
        self.tokens, self.state = list(tokens), state

    def get_seq_length(self):
        return len(self.tokens)


class Backend:
    name = 'test-model'
    device = 'cpu'
    eos = {99}
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids))

    def __init__(self, fail_call=None, drift=False):
        self.calls = []
        self.fail_call, self.drift = fail_call, drift
        self.sampling_identity = {'model_id': self.name, 'revision': 'a' * 40,
                                 'runtime_identity_sha256': 'b' * 64,
                                 'declaration_sha256': 'c' * 64,
                                 'cache_identity_sha256': 'd' * 64}

    def forward(self, ids, cache=None):
        self.calls.append(list(ids))
        if len(self.calls) == self.fail_call:
            raise RuntimeError('simulated forward failure')
        cache = cache or Cache()
        cache.tokens.extend(ids)
        # Deliberately depend on call segmentation, not only the token history.
        cache.state = (cache.state * 13 + sum(ids) + len(ids) ** 2) % 997
        logits = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32) / 20
        logits = logits + torch.sin(torch.arange(32).float() + cache.state).reshape(1, 1, 32) / 5
        logits[0, 0, 0] = -1_000_000 - len(cache.tokens)
        if self.drift:
            logits[0, 0, 3] += .125
        return SimpleNamespace(past_key_values=cache, logits=logits)

    def prefill_chunked(self, ids, chunk=512):
        cache = None
        for start in range(0, len(ids), chunk):
            out = self.forward(ids[start:start + chunk], cache)
            cache = out.past_key_values
        return out


class Telemetry:
    def __init__(self):
        self.failures = []

    def sample(self, **kw):
        pass

    def failure(self, exc):
        self.failures.append(type(exc).__name__)


def noop(*a, **kw):
    pass


def read(path):
    return json.loads(path.read_text())


def reason(backend, folder, cap=130, guard=noop, prompt=None):
    return m.reason_durable(backend, prompt or [1, 2], 'task', 'source_reasoning', cap,
                            folder, Telemetry(), guard, noop)


def answer(backend, folder, cap=130, guard=noop):
    history = {'prefix_ids': [1, 2], 'bridge_ids': [151668]}
    cache = backend.prefill_chunked(history['prefix_ids']).past_key_values
    return m.answer_durable(backend, history, cache, 'task', 'answer_small', cap,
                            folder, Telemetry(), guard, noop)


def assert_reason_equal(a, b):
    for key in ['reasoning_ids', 'prefix_ids', 'bridge_ids', 'natural_boundary',
                'reasoning_capped', 'early_eos', 'rng_initial', 'rng_after_reasoning']:
        assert a[key] == b[key], key


def assert_answer_equal(a, b):
    for key in ['answer_ids', 'answer_text', 'answer_seed', 'rng_initial', 'rng_final',
                'answer_ended_eos', 'answer_capped', 'bridge_token_count']:
        assert a[key] == b[key], key


def force_terminal(monkeypatch, terminal, at_position):
    sample = old.sample

    def forced(logits, rng, **kw):
        token = sample(logits, rng, **kw)
        position = round(-float(logits[0, 0, 0]) - 1_000_000)
        return terminal if position >= at_position else token

    monkeypatch.setattr(old, 'sample', forced)
    monkeypatch.setattr(m, 'sample', forced)
    # answer_instrumented imports its own binding.
    import gearshift.coding_recovery_answer as recovery
    monkeypatch.setattr(recovery, 'sample', forced)


@pytest.mark.parametrize('terminal', [None, 151668, 99])
def test_reason_preserves_original_calls_tokens_rng_and_boundary(tmp_path, monkeypatch, terminal):
    if terminal:
        force_terminal(monkeypatch, terminal, 1003)
    prompt = [1] * 1000
    before = Backend()
    expected, expected_cache = old.reason(before, prompt, 'task', 'source_reasoning', 8)
    after = Backend()
    actual, cache = reason(after, tmp_path / 'run', 8, prompt=prompt)
    assert_reason_equal(expected, actual)
    assert before.calls == after.calls
    assert [len(x) for x in after.calls[:2]] == [512, 488]
    assert cache.tokens == expected_cache.tokens
    assert cache.state == expected_cache.state
    if terminal == 151668:
        assert 151668 not in cache.tokens
    if terminal == 99:
        assert cache.tokens[-1] == 99


@pytest.mark.parametrize('terminal', [None, 99])
def test_answer_matches_existing_instrumented_sampler(tmp_path, monkeypatch, terminal):
    if terminal:
        force_terminal(monkeypatch, terminal, 6)
    h = {'prefix_ids': [1, 2], 'bridge_ids': [151668]}
    b = Backend()
    expected = answer_instrumented(b, h, b.prefill_chunked(h['prefix_ids']).past_key_values,
                                    'task', 'answer_small', 10, tmp_path / 'old', Telemetry(), noop, noop)
    new = Backend()
    actual = answer(new, tmp_path / 'new', 10)
    assert_answer_equal(expected, actual)
    assert b.calls == new.calls


@pytest.mark.parametrize('kind', ['reason', 'answer'])
def test_interrupted_forward_restores_exact_rng_without_resampling_prefix(tmp_path, monkeypatch, kind):
    run = reason if kind == 'reason' else answer
    expected = run(Backend(), tmp_path / 'reference', 15)
    expected = expected[0] if kind == 'reason' else expected
    interrupted = Backend(fail_call=7)
    with pytest.raises(RuntimeError, match='forward failure'):
        run(interrupted, tmp_path / 'interrupted', 15)
    saved = read(tmp_path / 'interrupted/resume.json')
    assert saved['state'] == 'interrupted'
    assert saved['forwarded_tokens'] == len(saved['tokens']) - 1
    sampled = []
    original = m.sample

    def counted(*a, **kw):
        sampled.append(True)
        return original(*a, **kw)

    monkeypatch.setattr(m, 'sample', counted)
    resumed = run(Backend(), tmp_path / 'interrupted', 15)
    actual = resumed[0] if kind == 'reason' else resumed
    assert len(sampled) == 15 - len(saved['tokens'])
    (assert_reason_equal if kind == 'reason' else assert_answer_equal)(expected, actual)
    assert actual['sampler_timing']['resume_count'] == 1
    assert actual['sampler_timing']['reconstruction_seconds'] > 0
    assert actual['sampler_timing']['complete'] is True


@pytest.mark.parametrize('kind', ['reason', 'answer'])
def test_guard_abort_has_durable_prefix_and_exact_resume(tmp_path, kind):
    count = 0

    def guard():
        nonlocal count
        count += 1
        if count == 2:
            raise TimeoutError('controlled')

    run = reason if kind == 'reason' else answer
    expected = run(Backend(), tmp_path / 'reference', 8)
    with pytest.raises(TimeoutError):
        run(Backend(), tmp_path / 'run', 8, guard)
    saved = read(tmp_path / 'run/resume.json')
    assert len(saved['tokens']) == 1
    actual = run(Backend(), tmp_path / 'run', 8)
    if kind == 'reason':
        assert_reason_equal(expected[0], actual[0])
    else:
        assert_answer_equal(expected, actual)


@pytest.mark.parametrize('kind', ['reason', 'answer'])
def test_completed_transaction_recovers_missing_materialized_result_without_samples(tmp_path, monkeypatch, kind):
    run = reason if kind == 'reason' else answer
    result = run(Backend(), tmp_path / 'run', 8)
    record = result[0] if kind == 'reason' else result
    name = 'source_history.json' if kind == 'reason' else 'answer_record.json'
    (tmp_path / 'run' / name).unlink()
    (tmp_path / 'run/complete.json').unlink()
    monkeypatch.setattr(m, 'sample', lambda *a, **kw: pytest.fail('completed draw resampled'))
    after = Backend()
    recovered = run(after, tmp_path / 'run', 8)
    assert (recovered[0] if kind == 'reason' else recovered) == record
    assert read(tmp_path / 'run' / name) == record
    if kind == 'answer':
        assert after.calls == [[1, 2]]  # Caller prepared the base; sampler did no work.


def test_answer_completion_persistence_failure_cannot_overwrite_complete_transaction(tmp_path, monkeypatch):
    original = m.bind

    def fail_materialize(path, obj):
        if path.name == 'answer_record.json':
            raise OSError('temporary record write failure')
        return original(path, obj)

    monkeypatch.setattr(m, 'bind', fail_materialize)
    with pytest.raises(OSError):
        answer(Backend(), tmp_path / 'run', 8)
    assert read(tmp_path / 'run/resume.json')['state'] == 'complete'
    monkeypatch.setattr(m, 'bind', original)
    monkeypatch.setattr(m, 'sample', lambda *a, **kw: pytest.fail('completed draw resampled'))
    assert len(answer(Backend(), tmp_path / 'run', 8)['answer_ids']) == 8


@pytest.mark.parametrize('kind', ['reason', 'answer'])
def test_numerical_reconstruction_drift_fails_before_new_sampling(tmp_path, monkeypatch, kind):
    run = reason if kind == 'reason' else answer
    with pytest.raises(RuntimeError):
        run(Backend(fail_call=7), tmp_path / 'run', 15)
    before = (tmp_path / 'run/resume.json').read_bytes()
    monkeypatch.setattr(m, 'sample', lambda *a, **kw: pytest.fail('drift proceeded to sampling'))
    with pytest.raises(ValueError, match='reconstruction logits differ'):
        run(Backend(drift=True), tmp_path / 'run', 15)
    assert (tmp_path / 'run/resume.json').read_bytes() == before


@pytest.mark.parametrize('field', ['revision', 'runtime_identity_sha256', 'declaration_sha256', 'cache_identity_sha256'])
def test_answer_cannot_reuse_changed_scientific_identity(tmp_path, field):
    answer(Backend(), tmp_path / 'run', 3)
    b = Backend()
    b.sampling_identity[field] = 'changed'
    with pytest.raises(ValueError, match='identity changed'):
        answer(b, tmp_path / 'run', 3)


def test_reason_prompt_stream_and_cap_are_bound(tmp_path):
    reason(Backend(), tmp_path / 'run', 3)
    for prompt, stream, cap in [([1, 3], 'source_reasoning', 3), ([1, 2], 'small_reasoning', 3), ([1, 2], 'source_reasoning', 4)]:
        with pytest.raises(ValueError, match='identity changed'):
            m.reason_durable(Backend(), prompt, 'task', stream, cap, tmp_path / 'run', Telemetry(), noop, noop)


def test_state_corruption_is_not_silently_restarted(tmp_path):
    reason(Backend(), tmp_path / 'run', 3)
    state = read(tmp_path / 'run/resume.json')
    state['tokens'][0] += 1
    write(tmp_path / 'run/resume.json', state)
    with pytest.raises(ValueError, match='state hash'):
        reason(Backend(), tmp_path / 'run', 3)


def test_missing_explicit_model_identity_fails_before_any_forward(tmp_path):
    b = Backend()
    del b.sampling_identity['revision']
    with pytest.raises(ValueError, match='sampling_identity'):
        reason(b, tmp_path / 'run', 3)
    assert b.calls == []


def test_hard_crash_checkpoint_does_not_invent_missing_timing(tmp_path):
    with pytest.raises(RuntimeError):
        reason(Backend(fail_call=7), tmp_path / 'run', 15)
    state = read(tmp_path / 'run/resume.json')
    state['state'] = 'running'
    state.pop('resume_sha256')
    state['resume_sha256'] = digest(state)
    write(tmp_path / 'run/resume.json', state)
    actual, _ = reason(Backend(), tmp_path / 'run', 15)
    assert actual['sampler_timing']['complete'] is False


@pytest.mark.parametrize('terminal', [151668, 99])
def test_completed_terminal_reasoning_rebuilds_exact_cache_without_sampling(tmp_path, monkeypatch, terminal):
    force_terminal(monkeypatch, terminal, 4)
    expected, cache = reason(Backend(), tmp_path / 'run', 10)
    monkeypatch.setattr(m, 'sample', lambda *a, **kw: pytest.fail('completed history resampled'))
    actual, restored = reason(Backend(), tmp_path / 'run', 10)
    assert actual == expected
    assert restored.tokens == cache.tokens
    assert restored.state == cache.state


def test_pending_early_eos_is_cached_once_after_resume(tmp_path, monkeypatch):
    force_terminal(monkeypatch, 99, 4)
    expected, cache = reason(Backend(), tmp_path / 'reference', 10)
    with pytest.raises(RuntimeError):
        reason(Backend(fail_call=4), tmp_path / 'run', 10)
    state = read(tmp_path / 'run/resume.json')
    assert state['tokens'][-1] == 99
    assert state['forwarded_tokens'] == len(state['tokens']) - 1
    monkeypatch.setattr(m, 'sample', lambda *a, **kw: pytest.fail('terminal token resampled'))
    actual, restored = reason(Backend(), tmp_path / 'run', 10)
    assert_reason_equal(expected, actual)
    assert restored.tokens == cache.tokens
    assert restored.tokens.count(99) == 1


@pytest.mark.parametrize('kind', ['reason', 'answer'])
def test_exception_inside_sampling_cannot_pair_advanced_rng_with_missing_token(tmp_path, monkeypatch, kind):
    run = reason if kind == 'reason' else answer
    expected = run(Backend(), tmp_path / 'reference', 10)
    original = m.sample
    calls = 0

    def fail_after_rng(*args, **kwargs):
        nonlocal calls
        calls += 1
        token = original(*args, **kwargs)
        if calls == 5:
            raise RuntimeError('sampler interrupted after RNG advance')
        return token

    monkeypatch.setattr(m, 'sample', fail_after_rng)
    with pytest.raises(RuntimeError):
        run(Backend(), tmp_path / 'run', 10)
    assert len(read(tmp_path / 'run/resume.json')['tokens']) == 4
    monkeypatch.setattr(m, 'sample', original)
    actual = run(Backend(), tmp_path / 'run', 10)
    if kind == 'reason':
        assert_reason_equal(expected[0], actual[0])
    else:
        assert_answer_equal(expected, actual)


def test_concurrent_sampler_owner_is_rejected(tmp_path):
    folder = tmp_path / 'run'
    with m._lock(folder):
        b = Backend()
        with pytest.raises(BlockingIOError):
            reason(b, folder, 5)
        assert b.calls == []


def timed_operations(monkeypatch):
    """A synthetic counter measures only deliberately charged host operations."""
    clock = {'seconds': 0., 'resume_writes': 0}
    monkeypatch.setattr(m.time, 'perf_counter', lambda: clock['seconds'])
    original_write, original_bind = m.write, m.bind

    def measured_write(path, value):
        if path.name == 'resume.json':
            clock['seconds'] += 2.; clock['resume_writes'] += 1
        elif path.parent.name == 'attempts':
            clock['seconds'] += 17.  # Excluded measurement receipt overhead.
        return original_write(path, value)

    def measured_bind(path, value):
        if path.name == 'identity.json': clock['seconds'] += .5
        elif path.name in ('answer_record.json', 'source_history.json'): clock['seconds'] += 7.
        elif path.name == 'complete.json': clock['seconds'] += 11.
        elif path.name == 'completion_timing.json': clock['seconds'] += 13.  # Cannot time itself.
        return original_bind(path, value)

    class TimedTelemetry(Telemetry):
        def sample(self, **kw): clock['seconds'] += 4.
        def failure(self, exc):
            clock['seconds'] += 5.; super().failure(exc)

    def publish(**kw): clock['seconds'] += 6.
    monkeypatch.setattr(m, 'write', measured_write); monkeypatch.setattr(m, 'bind', measured_bind)
    return clock, TimedTelemetry(), publish


@pytest.mark.parametrize('kind', ['reason', 'answer'])
def test_checkpoint_io_measured_separately_and_final_commit_lives_in_immutable_sidecar(tmp_path, monkeypatch, kind):
    clock, telemetry, publish = timed_operations(monkeypatch)
    folder = tmp_path / kind; b = Backend()
    if kind == 'reason':
        record, _ = m.reason_durable(b, [1, 2], 'task', 'source_reasoning', 5, folder, telemetry, noop, publish)
    else:
        history = {'prefix_ids': [1, 2], 'bridge_ids': [151668]}
        cache = b.prefill_chunked(history['prefix_ids']).past_key_values
        record = m.answer_durable(b, history, cache, 'task', 'answer_small', 5, folder, telemetry, noop, publish)
    snapshot = record['sampler_timing']['overhead']
    completion = read(folder / 'completion_timing.json'); measured = completion['overhead']
    assert snapshot['checkpoint_persistence_seconds'] == 4.
    assert snapshot['checkpoint_persistence_calls'] == 2
    assert snapshot['result_materialization_seconds'] == 0.
    assert measured['checkpoint_persistence_seconds'] == 6.
    assert measured['checkpoint_persistence_calls'] == 3 == clock['resume_writes']
    assert measured['result_materialization_seconds'] == 18.
    assert measured['identity_binding_seconds'] == .5
    assert measured['telemetry_seconds'] == 4. and measured['progress_publish_seconds'] == 6.
    assert measured['scope_complete'] and completion['this_receipt_persistence_excluded']
    assert completion['record_sha256'] == digest(record)
    assert not measured['subtract_from_active_seconds'] and not measured['model_only_latency_measured']
    before = (folder / 'completion_timing.json').read_bytes()
    # Retrieval may bind existing records, but cannot mutate the first completion receipt.
    if kind == 'reason': actual, _ = reason(Backend(), folder, 5)
    else: actual = answer(Backend(), folder, 5)
    assert actual == record and (folder / 'completion_timing.json').read_bytes() == before


@pytest.mark.parametrize('kind', ['reason', 'answer'])
def test_closed_interrupted_attempt_recovers_actual_io_without_double_counting(tmp_path, monkeypatch, kind):
    clock, _, _ = timed_operations(monkeypatch)
    run = reason if kind == 'reason' else answer
    with pytest.raises(RuntimeError): run(Backend(fail_call=7), tmp_path / 'run', 15)
    checkpoint = read(tmp_path / 'run/resume.json')
    previous = read(next((tmp_path / 'run/attempts').glob('*.json')))
    assert previous['last_resume_sha256'] == checkpoint['resume_sha256']
    assert previous['overhead']['checkpoint_persistence_seconds'] == clock['resume_writes'] * 2.
    # The transaction itself cannot include the elapsed write that persisted it.
    assert checkpoint['timing']['overhead']['checkpoint_persistence_seconds'] < previous['overhead']['checkpoint_persistence_seconds']
    result = run(Backend(), tmp_path / 'run', 15); record = result[0] if kind == 'reason' else result
    completion = read(tmp_path / 'run/completion_timing.json'); overhead = completion['overhead']
    assert overhead['scope_complete'] and record['sampler_timing']['complete']
    assert overhead['checkpoint_persistence_seconds'] == clock['resume_writes'] * 2.
    assert len(overhead['prior_attempt_receipts']) == 1


@pytest.mark.parametrize('kind', ['reason', 'answer'])
def test_unclosed_crash_preserves_only_known_io_prefix_and_flags_unknown_tail(tmp_path, monkeypatch, kind):
    clock, _, _ = timed_operations(monkeypatch)
    run = reason if kind == 'reason' else answer
    original_failed = m._Session.failed
    monkeypatch.setattr(m._Session, 'failed', lambda self, error: None)  # Simulate no shutdown hook.
    with pytest.raises(RuntimeError): run(Backend(fail_call=7), tmp_path / 'run', 15)
    assert read(tmp_path / 'run/resume.json')['state'] == 'running'
    monkeypatch.setattr(m._Session, 'failed', original_failed)
    result = run(Backend(), tmp_path / 'run', 15); record = result[0] if kind == 'reason' else result
    completion = read(tmp_path / 'run/completion_timing.json'); overhead = completion['overhead']
    assert not overhead['scope_complete'] and not record['sampler_timing']['complete']
    assert overhead['checkpoint_persistence_seconds'] < clock['resume_writes'] * 2.
    assert overhead['unmeasured_gaps']


def test_lost_final_timing_receipt_is_not_replaced_with_invented_complete_timing(tmp_path, monkeypatch):
    clock, _, _ = timed_operations(monkeypatch)
    original_materialize = m._Session.materialize
    original_failed = m._Session.failed
    monkeypatch.setattr(m._Session, 'materialize', lambda self, record: (_ for _ in ()).throw(RuntimeError('crash after final checkpoint')))
    monkeypatch.setattr(m._Session, 'failed', lambda self, error: None)
    with pytest.raises(RuntimeError): answer(Backend(), tmp_path / 'run', 5)
    durable = read(tmp_path / 'run/resume.json'); assert durable['state'] == 'complete'
    assert not (tmp_path / 'run/completion_timing.json').exists()
    monkeypatch.setattr(m._Session, 'materialize', original_materialize)
    monkeypatch.setattr(m._Session, 'failed', original_failed)
    monkeypatch.setattr(m, 'sample', lambda *a, **kw: pytest.fail('completed draw resampled'))
    recovered = answer(Backend(), tmp_path / 'run', 5)
    assert recovered == durable['record']
    completion = read(tmp_path / 'run/completion_timing.json')
    assert completion['complete_transaction_recovered'] and not completion['overhead']['scope_complete']
    assert completion['overhead']['checkpoint_persistence_seconds'] < clock['resume_writes'] * 2.
