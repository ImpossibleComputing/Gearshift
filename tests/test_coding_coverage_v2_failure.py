from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import coding_coverage_v2_worker as worker


@pytest.mark.parametrize('error', [
    ValueError('Checkpoint scientific identity mismatch'),
    AssertionError('Numerical control differed'),
    FloatingPointError('Nonfinite training KL'),
    RuntimeError('Shared historical cache mutated by training'),
    RuntimeError('Missing, zero or nonfinite cache gradient'),
    RuntimeError('Missing, zero or nonfinite mapper gradient'),
    RuntimeError('Frozen language model received gradients'),
    RuntimeError('Actual-model exact checkpoint continuation control failed'),
    RuntimeError('Could not restore clean primary initialization after preflight'),
    RuntimeError('Native/native splice changed whole-history execution'),
    RuntimeError('Generation cache aliases immutable base'),
    RuntimeError('Answer seed or immutable base cache drift'),
    RuntimeError('Saved historical pairs mutated'),
    RuntimeError('Frozen isolated sandbox gate is not passed'),
    RuntimeError('The total norm for gradients is non-finite and cannot be clipped'),
])
def test_scientific_failures_are_not_automatically_retried(error):
    assert worker.failure_exit_code(error) == 65


@pytest.mark.parametrize('error', [
    RuntimeError('CUDA out of memory'), RuntimeError('Private scorer is unavailable; this is not a model failure'),
    RuntimeError('Transient CUDA driver failure'), OSError('Temporary I/O failure'),
    TimeoutError('on-pod deadline'), FileNotFoundError('Incomplete input upload'),
])
def test_engineering_failures_keep_bounded_retry_exit(error):
    assert worker.failure_exit_code(error) == 1


@pytest.mark.parametrize(('error', 'expected'), [(ValueError('changed identity'), 65), (OSError('transient'), 1)])
def test_main_uses_classification_even_before_context_is_available(monkeypatch, error, expected):
    monkeypatch.setattr(sys, 'argv', ['worker', '--plan', 'unused.json', '--role', 'evaluate', '--worker-id', 'eval_gpu2'])
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(worker, 'context', fail)
    with pytest.raises(SystemExit) as caught:
        worker.main()
    assert caught.value.code == expected


@pytest.mark.parametrize('error', [KeyboardInterrupt(), SystemExit(9)])
def test_main_preserves_interrupt_and_explicit_exit(monkeypatch, error):
    monkeypatch.setattr(sys, 'argv', ['worker', '--plan', 'unused.json', '--role', 'evaluate', '--worker-id', 'eval_gpu2'])
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(worker, 'context', fail)
    with pytest.raises(type(error)) as caught:
        worker.main()
    assert caught.value is error
