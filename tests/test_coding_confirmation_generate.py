import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gearshift.coding_control import bind, digest, seed_for, sha, write
from scripts import coding_confirmation_generate as g


def noop(*a, **kw): pass


def runtime():
    return {'torch': '2.8.0+cu128', 'transformers': '4.57.6', 'attention': 'sdpa',
            'dtype': 'bfloat16', 'gpu': 'NVIDIA H200', 'TF32': False, 'deterministic_algorithms': True}


def seeds(tids):
    return {'namespace': 'coding_pilot_v1', 'source_stream': 'source_reasoning',
            'small_reasoning_stream': 'small_reasoning', 'tasks': {
                t: {'source_reasoning_seed': seed_for(t, 0, 'source_reasoning'),
                    'small_reasoning_seed': seed_for(t, 0, 'small_reasoning'),
                    'answers': [{'seed_index': i, 'stream': s, 'answer_seed': seed_for(t, 0, s)}
                                for i, s in enumerate(g.ANSWER_STREAMS)]} for t in tids}}


def context(tmp_path, count=1):
    tids = ['atcoder/task_' + str(i) for i in range(count)]
    top = tmp_path / 'results'; status = top / 'workers/test'; status.mkdir(parents=True)
    write(status / 'runtime.json', runtime())
    write(status / 'model_setup.json', {'declaration_sha256': 'a' * 64,
        'runtime_identity_sha256': digest(runtime()), 'model_loading_seconds': {'source': 5., 'receiver': 2.},
        'mapper_initialization_seconds': .1, 'charged_to_single_output_inference': False})
    d = {'experiment_id': 'confirmation_test', 'task_count': count, 'task_ids': tids,
         'primary_conditions': g.CONDITIONS, 'primary_answer_count': 24 * count,
         'primary_checkpoints': {arm: {'mapper_sha256': ch * 64, 'mapper_path': arm + '.pt', 'step': 1024}
                                 for arm, ch in [('FIXED', 'd'), ('ROTATING', 'e')]}}
    events = []
    return {'repo_root': tmp_path, 'root': top, 'top': top, 'status_root': status,
            'declaration': d, 'declaration_sha256': 'a' * 64, 'seeds': seeds(tids),
            'visible': {t: {'task_id': t, 'prompt': 'public question ' + t, 'prompt_ids': [1, 2]} for t in tids},
            'runtime_sha256': digest(runtime()), 'attempt_id': 'test_attempt',
            'publish': lambda **kw: events.append(kw), 'events': events, 'guard': noop,
            'telemetry': SimpleNamespace(reset=noop, sample=noop, failure=noop)}


def durable(folder, kind, tid, stream, conditioning, role, record, c, cache=None, timing_complete=True):
    spec = g.MODELS[role]
    model = {'model_id': spec['id'], 'revision': spec['revision'],
             'declaration_sha256': c['declaration_sha256'], 'runtime_identity_sha256': c['runtime_sha256']}
    if cache is not None: model['cache_identity_sha256'] = cache
    identity = {'schema': 1, 'sampler': kind, 'task_id': tid, 'stream': stream,
                'seed': seed_for(tid, 0, stream), 'cap': 24576 if kind == 'reasoning' else 4096,
                'model': model, 'conditioning': conditioning, 'prefill_chunk': 512,
                'continuation_forward_chunk': 1, 'sampling': {'temperature': .6, 'top_p': .95, 'top_k': 20},
                'eos': [151643, 151645], 'closing_think': 151668}
    record = dict(record, sampler_identity_sha256=digest(identity), rng_initial=[1],
                  sampler_timing={'complete': timing_complete, 'reconstruction_seconds': 0.})
    ids = record['reasoning_ids' if kind == 'reasoning' else 'answer_ids']
    rng_key = 'rng_after_reasoning' if kind == 'reasoning' else 'rng_final'
    record[rng_key] = [2]
    state = {'schema': 1, 'state': 'complete', 'identity_sha256': digest(identity),
             'tokens': ids, 'forwarded_tokens': len(ids) - 1, 'rng_initial': [1], 'rng_state': [2],
             'record': record, 'record_sha256': digest(record)}
    state['resume_sha256'] = digest(state)
    name = 'source_history.json' if kind == 'reasoning' else 'answer_record.json'
    write(folder / 'identity.json', identity); write(folder / name, record); write(folder / 'resume.json', state)
    write(folder / 'complete.json', {'identity_sha256': digest(identity), 'record_sha256': digest(record),
          'record_file_sha256': sha(folder / name), 'record_file': name})
    write(folder / 'completion_timing.json', {'schema': 1, 'state': 'complete',
          'identity_sha256': digest(identity), 'record_sha256': digest(record),
          'terminal_attempt_id': 'synthetic', 'active_seconds_unchanged': True,
          'this_receipt_persistence_excluded': True,
          'final_checkpoint_and_materialization_included_when_scope_complete': True,
          'overhead': {'scope_complete': timing_complete, 'completion_receipt_file': 'completion_timing.json',
                       'subtract_from_active_seconds': False, 'model_only_latency_measured': False}})
    return record


def history(c, tid, kind):
    source = kind == 'source'; ids = [7 if source else 8, 151668]
    folder = g.task_folder(c, tid) / ('large_history' if source else 'small_history')
    stream = 'source_reasoning' if source else 'small_reasoning'
    h = {'task_id': tid, 'prompt_ids': [1, 2], 'reasoning_ids': ids, 'prefix_ids': [1, 2, ids[0]],
         'bridge_ids': [151668], 'prefix_cache_length': 3, 'seed': seed_for(tid, 0, stream),
         'natural_boundary': True, 'reasoning_capped': False, 'early_eos': False, 'reasoning_seconds': 2.}
    h = durable(folder, 'reasoning', tid, stream, {'prompt_ids': [1, 2]},
                'source' if source else 'receiver', h, c)
    hs = sha(folder / 'source_history.json')
    write(folder / 'history_ready.json', {'task_id': tid, 'sha256': hs,
          'declaration_sha256': c['declaration_sha256'], 'contains_final_answer_prefix': False})
    return h, hs


def draw(c, tid, condition, h, hs, seed):
    cp = c['declaration']['primary_checkpoints'][condition.split('_')[0]]['mapper_sha256'] if '_' in condition else None
    expected = g.contract(c, tid, condition, seed, hs, cp)
    folder = g.task_folder(c, tid) / condition / ('seed_' + str(seed['seed_index']))
    record = {'task_id': tid, 'answer_ids': [9, 151645], 'answer_text': 'immutable candidate',
              'answer_text_with_special_tokens': 'immutable candidate', 'answer_seed': seed['answer_seed'],
              'answer_seconds': 1., 'bridge_seconds': .1, 'first_answer_token_seconds': .2,
              'answer_ended_eos': True, 'answer_capped': False, 'bridge_token_count': len(h['bridge_ids'])}
    record = durable(folder / 'sampler', 'answer', tid, seed['stream'],
        {'history_sha256': digest(h), 'prefix_ids': h['prefix_ids'], 'bridge_ids': h['bridge_ids']},
        'source' if condition == 'A' else 'receiver', record, c, g.cache_identity(expected))
    write(folder / 'draw_identity.json', expected)
    write(folder / 'answer.json', {**record, **expected, 'immutable_base_cache_unchanged': True,
          'clone_no_alias': True, 'runtime_identity_sha256': c['runtime_sha256'],
          'checkpoint_load_seconds': 0., 'checkpoint_load_receipt_path': None,
          'checkpoint_load_receipt_sha256': None})
    assert g.verify_draw(folder, expected, h)
    return folder, expected


def complete_fixture(c):
    for tid in c['declaration']['task_ids']:
        large, hs = history(c, tid, 'source'); small, ss = history(c, tid, 'small')
        folder = g.task_folder(c, tid)
        p = {'task_id': tid, 'declaration_sha256': c['declaration_sha256'],
             'public_prompt_sha256': digest(c['visible'][tid]['prompt']), 'enable_thinking': False,
             'prompt_ids': [1, 2, 3], 'prefix_ids': [1, 2], 'bridge_ids': [3],
             'rendered_template': '<think>\n\n</think>'}
        write(folder / 'prompt_only_template.json', p); ps = sha(folder / 'prompt_only_template.json')
        metric = {'max_abs': 0., 'mean_abs': 0., 'kl': 0., 'top1_equal': True}
        control_path = folder / 'controls/test_attempt.json'
        write(control_path, {'whole_native_tensor_exact': True, 'whole_native_splice_bridge': metric,
              'partial_native_splice_bridge': metric, 'passed': True, 'history_sha256': hs,
              'prompt_tokens': 2, 'history_tokens': 3, 'absolute_suffix_positions': [2, 2], 'no_cropping': True})
        for kind in ['source', 'small', 'receiver']:
            write(c['top'] / 'primary/jobs' / (kind + '_' + tid.replace('/', '__')) / 'complete.json', {
                'experiment_id': c['declaration']['experiment_id'], 'declaration_sha256': c['declaration_sha256'],
                'task_id': tid, 'kind': kind, 'worker_attempt': 'test_attempt',
                'runtime_path': 'workers/test/runtime.json', 'runtime_identity_sha256': c['runtime_sha256'],
                'setup_path': 'workers/test/model_setup.json',
                **({'control_path': str(control_path.relative_to(c['top']))} if kind == 'receiver' else {})})
        for condition in g.CONDITIONS:
            h, sh = (small, ss) if condition == 'B' else (({'prefix_ids': p['prefix_ids'], 'bridge_ids': p['bridge_ids']}, ps) if condition == 'P' else (large, hs))
            for seed in c['seeds']['tasks'][tid]['answers']: draw(c, tid, condition, h, sh, seed)


def declaration_fixture(tmp_path):
    c = context(tmp_path); d = copy.deepcopy(c['declaration'])
    inputs = {'seeds.json': c['seeds'], 'visible.json': list(c['visible'].values()),
              'configs/coding_pilot_v1/pilot.json': {'models': {role: {**spec, 'dtype': 'bfloat16', 'frozen': True} for role, spec in g.MODELS.items()}},
              'policy.json': {'version': 'frozen', 'limit': 1},
              'proof.json': {'files': [{'sha256': cp['mapper_sha256']} for cp in d['primary_checkpoints'].values()]}}
    for name, obj in inputs.items(): write(tmp_path / name, obj)
    for name in g.REQUIRED_IMPLEMENTATION | {'scorer.py'}:
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# frozen source fixture\n')
    for cp in d['primary_checkpoints'].values(): cp['actual_checkpoint_byte_proof'] = 'proof.json'
    d.update(status='FROZEN', generation_authorized_by_this_file=True, task_scope_resolution='owner-resolved',
             seeds_path='seeds.json', visible_path='visible.json', inputs={p: sha(tmp_path / p) for p in inputs},
             implementation={p: sha(tmp_path / p) for p in g.REQUIRED_IMPLEMENTATION | {'scorer.py'}},
             scorer={'policy_path': 'policy.json', 'policy_sha256': sha(tmp_path / 'policy.json'),
                     'policy_identity_sha256': digest(inputs['policy.json']),
                     'implementation_hashes': {'scorer.py': sha(tmp_path / 'scorer.py')}})
    write(tmp_path / 'declaration.json', d)
    return c, d


def test_frozen_declaration_and_inputs_validate(tmp_path):
    _, d = declaration_fixture(tmp_path)
    assert g.validate_declaration(tmp_path, 'declaration.json', sha(tmp_path / 'declaration.json')) == d
    assert d['scorer']['policy_sha256'] != d['scorer']['policy_identity_sha256']


@pytest.mark.parametrize('mutation', ['draft', 'missing_model_hash', 'missing_seed_hash', 'bad_canonical_policy', 'unverified_mapper', 'missing_implementation', 'duplicate_task'])
def test_invalid_declarations_fail_before_model_loading(tmp_path, mutation):
    _, d = declaration_fixture(tmp_path)
    if mutation == 'draft': d['status'] = 'DRAFT_NOT_EXECUTABLE'
    elif mutation == 'missing_model_hash': del d['inputs']['configs/coding_pilot_v1/pilot.json']
    elif mutation == 'missing_seed_hash': del d['inputs']['seeds.json']
    elif mutation == 'bad_canonical_policy': d['scorer']['policy_identity_sha256'] = d['scorer']['policy_sha256']
    elif mutation == 'unverified_mapper': d['primary_checkpoints']['FIXED']['actual_checkpoint_byte_proof'] = None
    elif mutation == 'missing_implementation': del d['implementation']['gearshift/coding_confirmation_sampling.py']
    else: d['task_ids'] *= 2
    write(tmp_path / 'declaration.json', d)
    with pytest.raises(ValueError): g.validate_declaration(tmp_path, 'declaration.json', sha(tmp_path / 'declaration.json'))


@pytest.mark.parametrize('field', ['answers', 'source_reasoning_seed', 'small_reasoning_seed'])
def test_seed_manifest_cannot_change_worker_independent_draws(tmp_path, field):
    c = context(tmp_path); tid = c['declaration']['task_ids'][0]
    if field == 'answers': c['seeds']['tasks'][tid][field] = c['seeds']['tasks'][tid][field][::-1]
    else: c['seeds']['tasks'][tid][field] += 1
    with pytest.raises(ValueError): g.validate_seeds(c['seeds'], [tid])


@pytest.mark.parametrize('group', ['inputs', 'implementation'])
def test_private_path_is_rejected_before_any_input_hash_or_read(tmp_path, monkeypatch, group):
    declaration_fixture(tmp_path)
    path = tmp_path / 'declaration.json'; d = read_json(path)
    d[group]['data/coding_pilot_v1/private/confirmation.json'] = 'f' * 64
    write(path, d); expected = sha(path)
    original_sha, original_read = g.sha, g.read

    def only_declaration(fn, candidate):
        assert Path(candidate) == path, 'No declared input may be opened before private-path rejection'
        return fn(candidate)

    monkeypatch.setattr(g, 'sha', lambda p: only_declaration(original_sha, p))
    monkeypatch.setattr(g, 'read', lambda p: only_declaration(original_read, p))
    with pytest.raises(ValueError, match='Private test files'):
        g.validate_declaration(tmp_path, 'declaration.json', expected)


def test_closure_seals_exact_population_and_is_idempotent(tmp_path):
    c = context(tmp_path, 2); complete_fixture(c)
    result = g.generation_closure(c)
    assert result['answer_count'] == 48 and len(result['histories']) == 4
    assert {r['contract']['condition'] for r in result['answers']} == set(g.CONDITIONS)
    assert len([r for r in result['files'] if r['path'].endswith('/completion_timing.json')]) == 52
    assert g.generation_closure(c) == result
    assert read_json(c['top'] / 'primary/generation_closure.json') == result


def read_json(path): return json.loads(path.read_text())


def test_closure_waits_for_all_jobs_without_creating_seal(tmp_path):
    c = context(tmp_path); complete_fixture(c)
    next((c['top'] / 'primary/jobs').glob('source_*/complete.json')).unlink()
    assert g.generation_closure(c) is None
    assert not (c['top'] / 'primary/generation_closure.json').exists()


@pytest.mark.parametrize('mutation', ['missing_answer', 'extra_answer', 'changed_tokens', 'score_added', 'wrong_history', 'wrong_checkpoint', 'wrong_control', 'wrong_runtime', 'missing_sampler_receipt', 'swapped_B_history', 'job_identity', 'missing_timing', 'wrong_timing_identity', 'wrong_timing_record', 'wrong_setup_identity'])
def test_closure_rejects_invalid_completed_populations(tmp_path, mutation):
    c = context(tmp_path); complete_fixture(c); tid = c['declaration']['task_ids'][0]
    folder = g.task_folder(c, tid); answer = folder / 'ROTATING_M/seed_0/answer.json'
    if mutation == 'missing_answer': answer.unlink()
    elif mutation == 'extra_answer': write(folder / 'UNDECLARED/seed_0/answer.json', {'extra': True})
    elif mutation == 'missing_sampler_receipt': (folder / 'ROTATING_M/seed_0/sampler/complete.json').unlink()
    elif mutation == 'missing_timing': (folder / 'large_history/completion_timing.json').unlink()
    elif mutation in ('wrong_timing_identity', 'wrong_timing_record'):
        p = folder / 'ROTATING_M/seed_0/sampler/completion_timing.json'; row = read_json(p)
        row['identity_sha256' if mutation == 'wrong_timing_identity' else 'record_sha256'] = '0' * 64
        write(p, row)
    elif mutation == 'wrong_setup_identity':
        p = c['status_root'] / 'model_setup.json'; row = read_json(p)
        row['declaration_sha256'] = '0' * 64; write(p, row)
    elif mutation == 'wrong_control':
        p = folder / 'controls/test_attempt.json'; row = read_json(p); row['whole_native_splice_bridge']['max_abs'] = .01; write(p, row)
    elif mutation == 'wrong_runtime': write(c['status_root'] / 'runtime.json', {'torch': 'changed'})
    elif mutation == 'swapped_B_history':
        p = folder / 'B/seed_0/answer.json'; row = read_json(p); row['history_sha256'] = sha(folder / 'large_history/source_history.json'); write(p, row)
    elif mutation == 'job_identity':
        p = next((c['top'] / 'primary/jobs').glob('source_*/complete.json')); row = read_json(p); row['task_id'] = 'another'; write(p, row)
    else:
        row = read_json(answer)
        if mutation == 'changed_tokens': row['answer_ids'][0] += 1
        elif mutation == 'score_added': row['score'] = {'passed': True}
        elif mutation == 'wrong_history': row['history_sha256'] = '0' * 64
        elif mutation == 'wrong_checkpoint': row['checkpoint_sha256'] = '0' * 64
        write(answer, row)
    with pytest.raises((ValueError, FileNotFoundError)): g.generation_closure(c)
    assert not (c['top'] / 'primary/generation_closure.json').exists()


def test_draw_receipt_can_recover_after_raw_answer_commit_without_sampling(tmp_path):
    c = context(tmp_path); tid = c['declaration']['task_ids'][0]; h, hs = history(c, tid, 'source')
    folder, expected = draw(c, tid, 'A', h, hs, c['seeds']['tasks'][tid]['answers'][0])
    (folder / 'draw_complete.json').unlink()
    before = (folder / 'answer.json').read_bytes()
    assert g.verify_draw(folder, expected, h)
    assert (folder / 'answer.json').read_bytes() == before


def test_expired_budget_stops_before_backend_construction(tmp_path, monkeypatch):
    c = context(tmp_path)
    c['guard'] = lambda: (_ for _ in ()).throw(TimeoutError('lease deadline'))
    monkeypatch.setattr(g, 'initialize', lambda c: pytest.fail('loaded models after deadline'))
    with pytest.raises(TimeoutError): g.work(c, 'source')


def test_work_stealing_finishes_descendants_and_closes_primary(tmp_path, monkeypatch):
    c = context(tmp_path); events = []
    monkeypatch.setattr(g, 'initialize', noop)

    def reason(c, tid, kind):
        events.append(kind)
        if kind == 'source': write(g.task_folder(c, tid) / 'large_history/history_ready.json', {'ready': True})
        return {}

    def receiver(c, tid): events.append('receiver'); return {'control_path': 'unused-test-control'}
    monkeypatch.setattr(g, 'reasoning_job', reason); monkeypatch.setattr(g, 'receiver_job', receiver)
    monkeypatch.setattr(g, 'generation_closure', lambda c: {'sealed': True} if len(events) == 3 else None)
    g.work(c, 'receiver')
    assert events == ['source', 'receiver', 'small']
    assert c['events'][-1]['all_primary_generation_complete'] is True


def test_no_eligible_job_is_not_claimed_as_global_completion(tmp_path, monkeypatch):
    c = context(tmp_path); monkeypatch.setattr(g, 'initialize', noop)

    @contextmanager
    def locked(*args): yield False

    monkeypatch.setattr(g, 'claim_job', locked)
    g.work(c, 'source')
    assert c['events'][-1]['all_primary_generation_complete'] is False
    assert c['events'][-1]['gpu_idle_release_requested'] is True


def test_constructor_integration_uses_exact_backend_objects_and_pins(tmp_path, monkeypatch):
    c, d = declaration_fixture(tmp_path)
    c['plan'] = {'declaration_path': 'declaration.json', 'declaration_sha256': sha(tmp_path / 'declaration.json'),
                 'experiment_id': d['experiment_id'], 'code_commit': d.get('source_commit')}
    import transformers
    import gearshift.coding_inference as inference
    import gearshift.coding_gradients as gradients
    calls = []
    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: 'same pipeline'),
                                chat_template='same template', apply_chat_template=lambda *a, **kw: [1, 2])

    def backend(spec):
        obj = SimpleNamespace(name=spec['id'], tokenizer=tokenizer); calls.append(obj); return obj

    class Mapper:
        def __init__(self, source, receiver): assert source is calls[0] and receiver is calls[1]
        def to(self, device): assert device == 'cuda:0'; return self
        def requires_grad_(self, value): assert value is False; return self

    monkeypatch.setattr(g, 'ROOT', tmp_path)
    monkeypatch.setattr(inference, 'Backend', backend); monkeypatch.setattr(gradients, 'AffineMapper', Mapper)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'NVIDIA H200')
    monkeypatch.setattr(torch, '__version__', '2.8.0+cu128'); monkeypatch.setattr(torch.version, 'cuda', '12.8')
    monkeypatch.setattr(transformers, '__version__', '4.57.6')
    monkeypatch.setattr(torch, 'set_num_threads', noop); monkeypatch.setattr(torch, 'use_deterministic_algorithms', noop)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    (c['status_root'] / 'model_setup.json').unlink()  # Constructor writes its own fresh attempt receipt.
    g.initialize(c)
    assert [b.name for b in calls] == [g.MODELS['source']['id'], g.MODELS['receiver']['id']]
    assert set(c['backend_identities']) == {b.name for b in calls}


@pytest.mark.parametrize('mismatch', ['experiment_id', 'code_commit'])
def test_plan_cannot_dispatch_a_different_frozen_experiment_or_source(tmp_path, monkeypatch, mismatch):
    c = context(tmp_path)
    c['plan'] = {'declaration_path': 'unused', 'declaration_sha256': 'a' * 64,
                 'experiment_id': c['declaration']['experiment_id'], 'code_commit': 'b' * 40}
    declaration = dict(c['declaration'], source_commit='b' * 40)
    c['plan'][mismatch] = 'different'
    monkeypatch.setattr(g, 'validate_declaration', lambda *a: declaration)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: pytest.fail('inspected GPU for a mismatched plan'))
    with pytest.raises(ValueError, match='Plan and frozen declaration'):
        g.initialize(c)


@pytest.mark.parametrize('complete_timing', [True, False])
def test_sample_answers_preserves_single_output_source_cost_and_missing_timing(tmp_path, monkeypatch, complete_timing):
    import gearshift.coding_confirmation_sampling as sampling
    c = context(tmp_path); tid = c['declaration']['task_ids'][0]; h, hs = history(c, tid, 'source')
    spec = g.MODELS['receiver']; backend = SimpleNamespace(name=spec['id'])
    c['backend_identities'] = {backend.name: {'model_id': backend.name, 'revision': spec['revision'],
        'runtime_identity_sha256': c['runtime_sha256'], 'declaration_sha256': c['declaration_sha256']}}

    def sampled(backend, h, cache, tid, stream, cap, folder, telemetry, guard, publish):
        assert cap == 4096
        row = {'task_id': tid, 'answer_ids': [9, 151645], 'answer_text': 'saved candidate',
               'answer_seed': seed_for(tid, 0, stream), 'answer_seconds': 1., 'bridge_seconds': .1,
               'answer_ended_eos': True, 'answer_capped': False}
        return durable(folder, 'answer', tid, stream,
            {'history_sha256': digest(h), 'prefix_ids': h['prefix_ids'], 'bridge_ids': h['bridge_ids']},
            'receiver', row, c, backend.sampling_identity['cache_identity_sha256'], complete_timing)

    monkeypatch.setattr(sampling, 'answer_durable', sampled)
    base = ((torch.ones(1, 1, 3, 1), torch.ones(1, 1, 3, 1)),)
    g.sample_answers(c, backend, tid, 'D', h, base, history_sha=hs,
                     timing={'reasoning_seconds': 2., 'native_prefill_seconds': .5, 'mapping_seconds': .2, 'splice_seconds': .1})
    rows = [read_json(p) for p in (g.task_folder(c, tid) / 'D').glob('seed_*/answer.json')]
    assert len(rows) == 3
    for row in rows:
        if complete_timing:
            assert row['single_output_inference_seconds'] == pytest.approx(3.8 + row['cache_clone_seconds'])
        else:
            assert row['single_output_inference_seconds'] is None
        assert row['source_cost_divisor_for_single_output'] == 1


def secondary_checkpoints(c):
    result = {}
    for arm, weight in [('FIXED', 7.), ('ROTATING', 11.)]:
        folder = c['repo_root'] / 'replication/arms' / arm / 'checkpoints/step_1024'; folder.mkdir(parents=True)
        torch.save({'arm': arm, 'step': 1024, 'state_dict': {'weight': weight}}, folder / 'mapper.pt')
        identity = {'experiment_id': c['declaration']['experiment_id'] + '/replication', 'arm': arm,
                    'configuration': {'training_seed': 20260919},
                    'selected_checkpoint_sha256': '0b9700ffd9cb38c23bcfa1327181b32992b128c6b95078214328382f86b8d68f'}
        mapper_sha = sha(folder / 'mapper.pt')
        manifest = {'checkpoint_identity': identity, 'checkpoint_identity_sha256': digest(identity),
                    'arm': arm, 'step': 1024, 'schedule_position': 1024,
                    'complete_resumable': True, 'verified_roundtrip': True,
                    'files': {'mapper.pt': {'sha256': mapper_sha}}}
        write(folder / 'manifest.json', manifest)
        result[arm] = {'mapper_path': str((folder / 'mapper.pt').relative_to(c['repo_root'])),
                       'mapper_sha256': mapper_sha, 'manifest_path': str((folder / 'manifest.json').relative_to(c['repo_root'])),
                       'manifest_sha256': sha(folder / 'manifest.json'), 'checkpoint_identity': identity}
    c['secondary_checkpoints'] = result


def test_secondary_receiver_reuses_primary_history_controls_and_only_runs_four_forms(tmp_path, monkeypatch):
    from gearshift.core import CacheInjector
    c = context(tmp_path); complete_fixture(c); secondary_checkpoints(c)
    tid = c['declaration']['task_ids'][0]; before = {str(p): sha(p) for p in (c['top'] / 'primary').rglob('*') if p.is_file()}
    prefills = []

    class Backend:
        def __init__(self, role): self.role = role
        def prefill_chunked(self, ids):
            prefills.append((self.role, list(ids)))
            pairs = ((torch.ones(1, 1, len(ids), 1), torch.ones(1, 1, len(ids), 1)),)
            return SimpleNamespace(past_key_values=CacheInjector.create(pairs, clone=True))
        def forward(self, *a, **kw): pytest.fail('secondary repeated native controls')

    class Mapper:
        def load_state_dict(self, state, strict): self.weight = state['weight']
        def __call__(self, pairs): return tuple(tuple(x * self.weight for x in pair) for pair in pairs)

    c.update(source=Backend('source'), receiver=Backend('receiver'), mapper=Mapper(),
             loaded_mapper=('primary', 'FIXED', c['declaration']['primary_checkpoints']['FIXED']['mapper_sha256']))
    calls = []

    def sampled(c, backend, tid, condition, h, base, **kw):
        calls.append((condition, kw['cohort'], float(base[0][0][0, 0, -1, 0]), kw['checkpoint_sha']))

    monkeypatch.setattr(g, 'sample_answers', sampled)
    receipt = g.receiver_job(c, tid, cohort='secondary')
    assert [r[0] for r in calls] == ['FIXED_M', 'ROTATING_M', 'FIXED_H', 'ROTATING_H']
    assert all(r[1] == 'secondary' for r in calls)
    assert [r[2] for r in calls] == [7., 11., 7., 11.]
    assert prefills == [('source', [1, 2, 7]), ('receiver', [1, 2])]
    assert {str(p): sha(p) for p in (c['top'] / 'primary').rglob('*') if p.is_file()} == before
    assert read_json(c['top'] / receipt['control_reuse_path'])['primary_control_sha256']


@pytest.mark.parametrize('mutation', ['wrong_seed', 'wrong_endpoint', 'wrong_initializer', 'wrong_hash', 'uncommitted'])
def test_secondary_checkpoint_admission_rejects_wrong_replication(tmp_path, mutation):
    c = context(tmp_path); secondary_checkpoints(c); cp = c['secondary_checkpoints']['FIXED']
    p = c['repo_root'] / cp['manifest_path']; manifest = read_json(p)
    if mutation == 'wrong_hash': cp['manifest_sha256'] = '0' * 64
    else:
        if mutation == 'wrong_seed': manifest['checkpoint_identity']['configuration']['training_seed'] = 20260915
        elif mutation == 'wrong_endpoint': manifest['step'] = 512
        elif mutation == 'wrong_initializer': manifest['checkpoint_identity']['selected_checkpoint_sha256'] = 'f' * 64
        else: manifest['complete_resumable'] = False
        manifest['checkpoint_identity_sha256'] = digest(manifest['checkpoint_identity'])
        cp['checkpoint_identity'] = manifest['checkpoint_identity']
        write(p, manifest); cp['manifest_sha256'] = sha(p)
    with pytest.raises(ValueError): g.checkpoint_for(c, 'FIXED', 'secondary')


def test_freezer_output_satisfies_generation_gate_in_isolated_fixture(tmp_path, monkeypatch):
    from scripts import coding_confirmation_freeze as freeze
    _, draft = declaration_fixture(tmp_path)
    tids = ['atcoder/test_' + str(i) for i in range(200)]
    visible_path = 'data/coding_pilot_v1/visible/confirmation.json'
    write(tmp_path / visible_path, [{'task_id': t, 'prompt': 'public fixture ' + t, 'prompt_ids': [1, 2]} for t in tids])
    original = {'task_ids': tids, 'original_reserved40_subset_ids': tids[:40],
                'visible_input_expected_sha256': sha(tmp_path / visible_path)}
    write(tmp_path / 'evidence/coding_pilot_v1/data_gate.json', {
        'passed': True, 'rows': [{'task_id': t, 'split': 'confirmation'} for t in tids],
        'identity': {'private_files': {'confirmation': '1' * 64},
                     'visible_files': {'confirmation': sha(tmp_path / visible_path)}}})
    initializer = '0b9700ffd9cb38c23bcfa1327181b32992b128c6b95078214328382f86b8d68f'
    draft.update(status='DRAFT_NOT_EXECUTABLE', generation_authorized_by_this_file=False,
                 exposure_audit_hashes={}, secondary_training={'initializer_sha256': initializer})
    for name, value in {
        'task_manifest.json': original, 'declaration.draft.json': draft,
        'seeds.draft.json': seeds(tids), 'analysis_plan.draft.json': {'status': 'draft'},
        'protocol.draft.json': {'status': 'draft'}, 'replication_declaration.json': {'seed': 20260919},
        'replication_schedules.json': {'fixture_only': True},
    }.items(): write(tmp_path / freeze.CONFIG / name, value)
    proof = {'files': [{'sha256': initializer}] +
             [{'sha256': cp['mapper_sha256']} for cp in draft['primary_checkpoints'].values()]}
    write(tmp_path / freeze.EVIDENCE / 'checkpoint_byte_verification.json', proof)
    calibration = tmp_path / freeze.EVIDENCE / 'scorer_repair/calibration'
    write(calibration / 'calibration_receipt.json', {'passed': True, 'test_fixture_only': True})
    policy = {'policy_status': 'frozen', 'calibration_receipt_sha256': sha(calibration / 'calibration_receipt.json')}
    write(calibration / 'frozen_policy.json', policy)
    write(calibration / 'scorer_identity.json', {'scorer_version': 'unit_test_only', 'policy_sha256': digest(policy),
          'files': {'scorer.py': sha(tmp_path / 'scorer.py')}})
    for relative in freeze.IMPLEMENTATION:
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# frozen integration test fixture only\n')
    decision = {'authority': 'direct_owner_reply', 'owner_reply_text': 'TEST FIXTURE ONLY',
                'decision': 'protect_40', 'experiment_id': draft['experiment_id']}
    write(tmp_path / 'test_owner_fixture.json', decision)

    def committed_output(args, cwd, text=False):
        if args[1:] == ['rev-parse', 'HEAD']: return 'f' * 40
        return (tmp_path / args[-1].split(':', 1)[1]).read_bytes()

    monkeypatch.setattr(freeze.subprocess, 'check_output', committed_output)
    result = freeze.freeze('test_owner_fixture.json', repo=tmp_path)
    relative = str(Path(result['declaration_path']).relative_to(tmp_path))
    d = g.validate_declaration(tmp_path, relative, result['declaration_sha256'])
    assert d['task_count'] == 160 and d['primary_answer_count'] == 3840 and d['secondary_answer_count'] == 1920
    assert Path(result['declaration_path']).is_relative_to(tmp_path)
