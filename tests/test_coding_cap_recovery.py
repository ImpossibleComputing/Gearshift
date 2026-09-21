import json
import shutil
from pathlib import Path

import pytest

from gearshift.coding_cap_recovery import recovery_constraints, validate_recovery
from gearshift.coding_control import digest, sha, write
from gearshift.coding_resume import verify_segment
from test_coding_cap_amendment import parent


@pytest.fixture
def interrupted(parent):
    root, original, original_id = parent
    # Make the final fixed task eligible for the already-declared amendment.
    folder = original / 'tasks/t39'
    history = json.loads((folder / 'source_history.json').read_text())
    history.update(reasoning_ids=[7] * 16384, reasoning_capped=True, natural_boundary=False)
    write(folder / 'source_history.json', history)
    row = json.loads((folder / 'complete.json').read_text())
    row['source_capped'] = True
    row['files']['source_history.json'] = sha(folder / 'source_history.json')
    write(folder / 'complete.json', row)
    declaration = root / 'configs/coding_pilot_v1/cap_amendment_v1.json'
    write(declaration, {'amendment_number': 1})
    interrupted_root = root / 'results/interrupted'
    ident = {**original_id, 'reasoning_cap': 24576, 'cap_amendment_number': 1,
             'cap_amendment_sha256': sha(declaration),
             'parent_identity_sha256': sha(original / 'identity.json')}
    write(interrupted_root / 'identity.json', ident)
    write(interrupted_root / 'native_gate.json', {'passed': True, 'identity_sha256': digest(ident)})
    for i in range(39):
        dest = interrupted_root / 'tasks' / f't{i}'
        shutil.copytree(original / 'tasks' / f't{i}', dest)
        row = json.loads((dest / 'complete.json').read_text())
        row['identity_sha256'] = digest(ident)
        for kind in ['source', 'small']:
            history = json.loads((dest / (kind + '_history.json')).read_text())
            if history['reasoning_capped']:
                history['reasoning_ids'] += [151668]
                history.update(reasoning_capped=False, natural_boundary=True)
                write(dest / (kind + '_history.json'), history)
                row[kind + '_capped'] = False
        row['files'] = {p.name: sha(p) for p in dest.glob('*.json') if p.name != 'complete.json'}
        write(dest / 'complete.json', row)
    write(interrupted_root / 'progress.json', {'completed_tasks': 39, 'passes': {'A': 39, 'B': 39, 'D': 39},
                                              'regenerated_tasks': 8, 'amendment_generation_wall_seconds': 100})
    write(interrupted_root / 'tasks/t39/partial_source_reasoning.json', {
        'task_id': 't39', 'segment': 'source_reasoning', 'token_ids': [7] * 192,
        'identity_sha256': digest(ident),
    })
    return root, original, original_id, interrupted_root, ident


def test_recovery_preserves_longer_original_prefix_and_completed_segments(interrupted):
    root, original, original_id, saved, _ = interrupted
    proof = validate_recovery(saved, original, root)
    assert proof['completed_task_ids'] == [f't{i}' for i in range(39)]
    assert proof['remaining_task_ids'] == ['t39']
    prefixes, complete, lineage = recovery_constraints(original, original_id, saved, 't39')
    assert len(prefixes['source_reasoning']) == 16384
    assert 'small_reasoning' in complete and 'answer_B' in complete
    assert lineage['additional_cap_increase'] is False
    verify_segment([7] * 16384 + [151668], 'source_reasoning', prefixes, complete)
    with pytest.raises(ValueError, match='diverged'):
        verify_segment([7] * 192 + [151668], 'source_reasoning', prefixes, complete)


def test_recovery_refuses_completed_outcomes_changed_membership_and_identity(interrupted):
    root, original, _, saved, ident = interrupted
    write(saved / 'baseline_gate.json', {'passed': False})
    with pytest.raises(ValueError, match='completed scientific outcome'):
        validate_recovery(saved, original, root)
    (saved / 'baseline_gate.json').unlink()
    (saved / 'tasks/t38/complete.json').unlink()
    with pytest.raises(ValueError, match='39-case checkpoint'):
        validate_recovery(saved, original, root)


def test_recovery_rejects_divergent_partial_and_keeps_extended_saved_prefix(interrupted):
    root, original, original_id, saved, ident = interrupted
    partial = saved / 'tasks/t39/partial_source_reasoning.json'
    r = json.loads(partial.read_text())
    r['token_ids'] = [7] * 16384 + [8] * 20
    write(partial, r)
    prefixes, complete, _ = recovery_constraints(original, original_id, saved, 't39')
    assert len(prefixes['source_reasoning']) == 16404
    with pytest.raises(ValueError, match='diverged'):
        verify_segment([7] * 16384 + [151668], 'source_reasoning', prefixes, complete)
    r['token_ids'][0] = 9
    write(partial, r)
    with pytest.raises(ValueError, match='diverged'):
        validate_recovery(saved, original, root)


def test_recovery_worker_reuses_39_and_finishes_only_the_original_final_draw(interrupted, monkeypatch):
    import importlib.util
    import time
    import types
    import torch
    import gearshift.coding_inference as inference
    import gearshift.coding_control_calibration as calibration
    import gearshift.coding_sandbox as sandbox
    root, original, original_id, saved, parent_id = interrupted
    new_original = root / 'results/coding_pilot_v1/preflight_v5'
    new_original.parent.mkdir(parents=True)
    original.rename(new_original)
    original = new_original
    new_saved = root / 'results/coding_pilot_v1/preflight_cap_v1'
    saved.rename(new_saved)
    saved = new_saved
    script = Path(__file__).resolve().parents[1] / 'scripts/coding_preflight_cap_recovered.py'
    spec = importlib.util.spec_from_file_location('recovery_worker_test', script)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    e = root / 'evidence/coding_pilot_v1'
    out = root / 'results/coding_pilot_v1/preflight_cap_v1_recovery1'
    for name, value in [('ROOT', root), ('E', e), ('R', out)]:
        monkeypatch.setattr(m, name, value)
    monkeypatch.delenv('GEARSHIFT_WORKER_SPEC', raising=False)
    monkeypatch.chdir(root)
    approval = e / 'parallel_500h_approved.json'
    write(approval, {'test': True})
    checkpoint = validate_recovery(saved, original, root)
    write(e / 'allocation_cap_recovery1.json', {
        'preflight_deadline_epoch': time.time() + 3600, 'image_digest': 'test-image',
        'approval_sha256': sha(approval),
        'cap_amendment_sha256': sha(root / 'configs/coding_pilot_v1/cap_amendment_v1.json'),
        'parent_checkpoint': {k: v for k, v in checkpoint.items() if k not in ['parent_identity', 'original_identity']},
    })
    write(e / 'sandbox_gate.json', {'passed': True, 'canonical_development': {'a': True, 'b': True}})
    write(root / 'data/coding_pilot_v1/visible/development.json', [{'task_id': f't{i}', 'prompt_ids': [3]} for i in range(40)])
    write(root / 'data/coding_pilot_v1/private/development.json', {f't{i}': {} for i in range(40)})
    calls = []
    class Backend:
        def __init__(self, spec):
            self.tokenizer = types.SimpleNamespace(backend_tokenizer=types.SimpleNamespace(to_str=lambda: 'same'), encode=lambda _: [3])
        def prefill_chunked(self, ids):
            return types.SimpleNamespace(past_key_values=None)
    def reason(backend, prompt, tid, stream, cap, callback):
        assert tid == 't39' and cap == 24576
        calls.append(stream)
        kind = 'source' if stream == 'source_reasoning' else 'small'
        history = json.loads((original / 'tasks/t39' / (kind + '_history.json')).read_text())
        tokens = history['reasoning_ids'] + ([151668] if history['reasoning_capped'] else [])
        callback(tokens)
        return {**history, 'reasoning_ids': tokens, 'prefix_ids': prompt + tokens[:-1],
                'reasoning_capped': False, 'natural_boundary': True, 'reasoning_seconds': 1}, None
    def answer(backend, history, cache, tid, stream, cap, callback):
        callback([8, 9])
        return {'task_id': tid, 'answer_ids': [8, 9], 'answer_text': 'print(1)'}
    monkeypatch.setattr(inference, 'Backend', Backend)
    monkeypatch.setattr(inference, 'reason', reason)
    monkeypatch.setattr(inference, 'answer', answer)
    monkeypatch.setattr(inference, 'memory_record', lambda: {'passed': True})
    monkeypatch.setattr(inference, 'sync', lambda: None)
    monkeypatch.setattr(calibration, 'matched_native_controls', lambda *a, **kw: None)
    monkeypatch.setattr(sandbox, 'score', lambda *a: {'passed': True})
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *_: 'test-device')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(m.platform, 'platform', lambda: 'test-platform')
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **kw: 'test')
    before = {p.relative_to(saved): p.read_bytes() for p in saved.rglob('*') if p.is_file()}
    m.main()
    assert calls == ['source_reasoning', 'small_reasoning']
    progress = json.loads((out / 'progress.json').read_text())
    assert (progress['completed_tasks'], progress['reused_parent_tasks'], progress['regenerated_tasks']) == (40, 39, 1)
    for p, data in before.items():
        assert (saved / p).read_bytes() == data
    for i in range(39):
        row = json.loads((out / f'tasks/t{i}/complete.json').read_text())
        assert row['measurement_identity_sha256'] == digest(parent_id)
        assert row['reused_from']['no_new_generation_or_scoring']
    assert json.loads((out / 'baseline_gate.json').read_text())['protocol_cap_amendment_needed'] is False
