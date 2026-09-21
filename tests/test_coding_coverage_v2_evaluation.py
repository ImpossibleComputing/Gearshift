import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gearshift.coding_control import digest, sha, write
from scripts import coding_coverage_v2_evaluate as evaluation
from scripts.coding_coverage_v2_report import cluster_statistics


def raw_record(contract):
    return {**contract, 'answer_ids': [7, 0], 'answer_text': 'pass', 'answer_text_with_special_tokens': 'pass',
            'immutable_base_cache_unchanged': True, 'clone_no_alias': True, 'answer_ended_eos': True,
            'answer_capped': False, 'attempt_id': 'attempt1', 'answer_seconds': .1}


def job_fixture(root, n=21):
    tids = ['task/'+str(i) for i in range(n)]
    seeds = {t: [{'seed_index': s, 'stream': 'seed'+str(s), 'answer_seed': s+9} for s in range(3)] for t in tids}
    job = {'experiment_id': 'coverage_v2_test', 'job_id': 'START_0_tasks', 'arm': 'START', 'step': 0,
           'task_ids': tids, 'forms': ['M'], 'include_controls': False, 'validation_kl': False,
           'checkpoint': {'path': 'start.pt', 'sha256': 'a'*64}}
    folder = root/'jobs'/job['job_id']; write(folder/'job.json', job)
    files = []; controls = []
    for tid in tids:
        task = folder/'validation'/tid.replace('/', '__'); write(task/'source_history.json', {'task_id': tid})
        for seed in seeds[tid]:
            dest = task/'START_M'/('seed_'+str(seed['seed_index']))
            contract = evaluation.draw_contract(job, tid, 'START_M', 'M', seed, sha(task/'source_history.json'))
            write(dest/'draw_identity.json', contract); write(dest/'answer.json', raw_record(contract))
            evaluation.completed_draw(dest, contract)
            files.append({'path': str((dest/'answer.json').relative_to(folder)), 'sha256': sha(dest/'answer.json'),
                          'receipt_sha256': sha(dest/'complete.json')})
        control = folder/'attempts/attempt1'/tid.replace('/', '__')/'native_splice_control.json'
        write(control, {'task_id': tid, 'passed': True})
        controls.append({'path': str(control.relative_to(folder)), 'sha256': sha(control)})
    write(folder/'generation_complete.json', {'job_identity_sha256': digest(job), 'files': files,
           'kl_files': [], 'control_files': controls, 'hidden_tests_loaded': False})
    return {'experiment_id': 'coverage_v2_test', 'jobs': [job]}, seeds


def test_committed_draw_requires_exact_contract_and_hash(tmp_path):
    plan, seeds = job_fixture(tmp_path, 1); job = plan['jobs'][0]
    folder = tmp_path/'jobs'/job['job_id']/'validation/task__0'; dest = folder/'START_M/seed_0'
    contract = evaluation.draw_contract(job, 'task/0', 'START_M', 'M', seeds['task/0'][0], sha(folder/'source_history.json'))
    assert evaluation.completed_draw(dest, contract)['answer_ids'] == [7, 0]
    modified = {**contract, 'checkpoint_sha256': 'b'*64}
    with pytest.raises(ValueError, match='contract'):
        evaluation.completed_draw(dest, modified)
    r = evaluation.read(dest/'answer.json'); r['answer_ids'] = [8, 0]; write(dest/'answer.json', r)
    with pytest.raises(ValueError, match='identity changed'):
        evaluation.completed_draw(dest, contract)


def test_generation_closure_rejects_missing_job_or_duplicate_shard(tmp_path):
    plan, _ = job_fixture(tmp_path, 1)
    assert len(evaluation.generation_closure(tmp_path, plan)['files']) == 3
    duplicate = copy.deepcopy(plan); duplicate['jobs'] *= 2
    with pytest.raises(ValueError, match='Duplicate job'):
        evaluation.generation_closure(tmp_path, duplicate)
    (tmp_path/'jobs/START_0_tasks/generation_complete.json').unlink()
    with pytest.raises(FileNotFoundError):
        evaluation.generation_closure(tmp_path, plan)


def test_closure_requires_a_passed_native_control(tmp_path):
    plan, _ = job_fixture(tmp_path, 1); p = tmp_path/'jobs/START_0_tasks/generation_complete.json'
    complete = evaluation.read(p); complete['control_files'] = []; write(p, complete)
    with pytest.raises(ValueError, match='native splice'):
        evaluation.generation_closure(tmp_path, plan)


def test_scoring_waits_for_entire_generation_and_shards_without_rewrites(tmp_path, monkeypatch):
    from gearshift import coding_sandbox as sandbox
    root = tmp_path/'evaluation'; plan, _ = job_fixture(root)
    private_path = tmp_path/'data/coding_pilot_v1/private/coverage_generalization.json'
    write(private_path, {f'task/{i}': {} for i in range(21)})
    write(tmp_path/'evidence/coding_pilot_v1/sandbox_gate.json', {'passed': True})
    opened = []; raw_read = evaluation.read
    def tracked(path):
        if Path(path) == private_path:
            opened.append('private')
        return raw_read(path)
    monkeypatch.setattr(evaluation, 'read', tracked)
    calls = []
    monkeypatch.setattr(sandbox, 'score', lambda code, spec, guard: (calls.append(code) or {'category': 'pass', 'passed': True}))
    c = {'repo_root': tmp_path, 'root': root, 'attempt_id': 'scorer1', 'guard': lambda: None, 'publish': lambda **kw: None,
         'private_tests_sha256': sha(private_path)}
    manifest_path = root/'jobs/START_0_tasks/generation_complete.json'; original = manifest_path.read_bytes(); manifest_path.unlink()
    with pytest.raises(FileNotFoundError):
        evaluation.score_jobs(c, plan)
    assert not opened and not calls
    manifest_path.write_bytes(original)
    raw_hashes = {str(p): sha(p) for p in root.glob('jobs/*/validation/*/*/seed_*/answer.json')}
    for shard in range(4):
        evaluation.score_jobs(c, plan, shard_index=shard, shard_count=4)
    assert len(calls) == 63 and len(opened) == 4
    assert not (root/'scored_answer_manifest.json').exists()
    complete = evaluation.finalize_scores(c, plan)
    assert len(complete['files']) == 63
    assert raw_hashes == {str(p): sha(p) for p in root.glob('jobs/*/validation/*/*/seed_*/answer.json')}
    evaluation.score_jobs(c, plan, shard_index=2, shard_count=4)
    assert len(calls) == 63  # Existing exact score receipts are also reused.
    c['private_tests_sha256'] = '0'*64
    with pytest.raises(ValueError, match='Private test bytes'):
        evaluation.score_jobs(c, plan, shard_index=0, shard_count=4)


def test_finished_sampler_state_is_recovered_without_another_draw(tmp_path):
    seed = {'stream': 'stream', 'answer_seed': 11, 'seed_index': 0}
    state = {'state': 'complete', 'task_id': 'task', 'stream': 'stream', 'seed': 11, 'cap': 4096,
             'historical_prefix_length': 4, 'answer_ids': [4, 0], 'rng_initial': [1], 'rng_state': [2]}
    write(tmp_path/'resume.json', state)
    b = SimpleNamespace(eos=[0], tokenizer=SimpleNamespace(decode=lambda ids, **kw: 'answer'))
    history = {'prefix_ids': [1, 2, 3, 4], 'bridge_ids': [5]}
    row = evaluation._recover_finished_sampler(tmp_path, b, history, 'task', seed)
    assert row['answer_ids'] == [4, 0] and row['answer_seconds'] is None
    assert row['timing_recovered_without_measurement']
    bad = {**seed, 'answer_seed': 12}
    with pytest.raises(ValueError, match='exact draw identity'):
        evaluation._recover_finished_sampler(tmp_path, b, history, 'task', bad)


def test_incomplete_sampler_attempt_is_preserved(tmp_path):
    dest = tmp_path/'condition/seed_0'; write(dest/'resume.json', {'state': 'aborted', 'answer_ids': [5, 6]})
    evaluation._archive_incomplete(dest)
    assert not dest.exists()
    saved = list((dest.parent/'interrupted_attempts').glob('seed_0_*/resume.json'))
    assert len(saved) == 1 and evaluation.read(saved[0])['answer_ids'] == [5, 6]


def test_actual_job_wrapper_resumes_exact_draws_and_checks_both_kl_panels(tmp_path, monkeypatch):
    from gearshift import coding_coverage_runtime, coding_inference, coding_recovery_answer, core
    # Keep real torch tensors/splicing/version checks. Stub only the heavy model
    # and the sampler so interruptions exercise actual durable orchestration.
    class Cache:
        def __init__(self, pairs): self.pairs = pairs
    class Tokenizer:
        def convert_tokens_to_ids(self, s): return 999
        def decode(self, ids, **kw): return '<think>\n\n</think>'
        def apply_chat_template(self, *a, **kw): return [1, 2, 3]
    class Backend:
        eos = [0]; tokenizer = Tokenizer()
        def prefill_chunked(self, ids):
            pair = (torch.ones(1, 1, len(ids), 2), torch.ones(1, 1, len(ids), 2))
            return SimpleNamespace(past_key_values=Cache((pair,)))
        def forward(self, ids, cache): return SimpleNamespace(logits=torch.ones(1, 1, 8))
    class Mapper(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
        def forward(self, pairs): return tuple(tuple(x*self.weight for x in pair) for pair in pairs)
    monkeypatch.setattr(core.CacheExtractor, 'tensors', lambda cache: cache.pairs)
    monkeypatch.setattr(core.CacheInjector, 'create', lambda pairs, clone=True: Cache(tuple(tuple(x.clone() for x in pair) for pair in pairs)))
    monkeypatch.setattr(coding_inference, 'sync', lambda: None)
    monkeypatch.setattr(coding_coverage_runtime.CoverageRuntime, '__init__', lambda self, *a: None)
    monkeypatch.setattr(coding_coverage_runtime.CoverageRuntime, 'kl', lambda self, obj, positions, *a, **kw: torch.tensor([.1]*len(positions)))
    mapper = Mapper(); torch.save({'state_dict': mapper.state_dict()}, tmp_path/'start.pt')
    d = {'validation_task_ids': ['task/0'], 'selected_checkpoint_sha256': sha(tmp_path/'start.pt')}
    h = {'prompt_ids': [1, 2], 'prefix_ids': [1, 2, 3, 4], 'bridge_ids': [5], 'reasoning_seconds': .2}
    seeds = {'task/0': [{'seed_index': s, 'stream': 's'+str(s), 'answer_seed': s+9} for s in range(3)]}
    monkeypatch.setattr(evaluation, 'load_inputs', lambda c: (d, {'task/0': {'source_history': h}},
        {'task/0': {'prompt_ids': [1,2], 'prompt': 'write code'}}, seeds, {'task/0': {'legacy': [0,1], 'broad': [0,2]}}))
    calls = []; interrupt = [True]
    def sample(b, history, cache, tid, stream, cap, dest, telemetry, guard, publish):
        if len(calls) == 1 and interrupt[0]:
            raise RuntimeError('worker interrupted')
        calls.append((str(dest), stream))
        return {'task_id': tid, 'answer_ids': [7,0], 'answer_text': 'pass', 'answer_text_with_special_tokens': 'pass',
                'answer_seed': int(stream[1:])+9, 'answer_seconds': .1, 'answer_ended_eos': True, 'answer_capped': False}
    monkeypatch.setattr(coding_recovery_answer, 'answer_instrumented', sample)
    telemetry = SimpleNamespace(reset=lambda *a: None, sample=lambda **kw: None, failure=lambda *a: None)
    c = {'repo_root': tmp_path, 'root': tmp_path/'evaluation', 'attempt_id': 'first', 'identity': {'runtime': 'test'},
         'guard': lambda: None, 'publish': lambda **kw: None, 'telemetry': telemetry}
    job = {'experiment_id': 'v2', 'job_id': 'start_task', 'arm': 'START', 'step': 0, 'task_ids': ['task/0'],
           'forms': ['M','H'], 'include_controls': True, 'validation_kl': True,
           'checkpoint': {'path': 'start.pt', 'sha256': d['selected_checkpoint_sha256']}}
    with pytest.raises(RuntimeError, match='interrupted'):
        evaluation.run_job(c, Backend(), Backend(), mapper, job)
    first_path = c['root']/'jobs/start_task/validation/task__0/START_M/seed_0/answer.json'; first_hash = sha(first_path)
    interrupt[0] = False; c['attempt_id'] = 'otherpod_second'; c['identity'] = {'runtime': 'test', 'pod_id': 'replacement-pod'}
    result = evaluation.run_job(c, Backend(), Backend(), mapper, job)
    assert len(calls) == 12 and len(result['files']) == 12
    assert sha(first_path) == first_hash and evaluation.read(first_path)['attempt_id'] == 'first'
    assert len(result['kl_files']) == 1 and len(result['control_files']) == 2
    assert evaluation.run_job(c, Backend(), Backend(), mapper, job) == result
    assert len(calls) == 12
    kl = evaluation.read(c['root']/'jobs/start_task'/result['kl_files'][0]['path'])
    assert kl['legacy']['positions'] == [0,1] and kl['broad']['positions'] == [0,2]


def statistics_rows():
    tids = [f't{i}' for i in range(21)]; rows = []
    for i, tid in enumerate(tids):
        for condition in ['FIXED', 'ROTATING']:
            for s in range(3):
                passed = i < 10 or (condition == 'ROTATING' and i == 10 and s == 0)
                rows.append({'task_id': tid, 'condition': condition, 'seed_index': s, 'passed': passed,
                    'category': 'pass' if passed else 'test_assertion', 'missing_requested_entrypoint': False,
                    'syntax_valid': True, 'EOS': True, 'capped': False, 'answer_tokens': 20,
                    'repetition_4gram_fraction': 0., **{k: .1 for k in (
                        'answer_seconds','native_prefill_seconds','mapping_seconds','splice_seconds','historical_receiver_prefill_tokens',
                        'source_inclusive_estimate_seconds','source_cache_reconstruction_seconds','cache_clone_seconds')}})
    return tids, rows


def test_bootstrap_clusters_tasks_and_uses_one_joint_resample_matrix():
    tids, rows = statistics_rows()
    result, means, indices = cluster_statistics(rows, tids, ['FIXED','ROTATING'], [('ROTATING','FIXED')])
    assert np.shape(indices) == (10000, 21)
    assert result['conditions']['FIXED']['passed_draws'] == 30
    assert result['conditions']['ROTATING']['passed_draws'] == 31
    delta = result['contrasts']['ROTATING-FIXED']
    assert delta['difference'] == pytest.approx(1/63)
    expected = np.asarray([r['ROTATING']-r['FIXED'] for r in means])[indices].mean(axis=1)
    assert delta['ci95'] == pytest.approx(np.quantile(expected, [.025,.975]).tolist())
    assert result['task_clusters'] == 21 and result['seeds_per_task'] == 3


@pytest.mark.parametrize('change', ['missing','duplicate','different_seed'])
def test_bootstrap_rejects_nonmatched_population(change):
    tids, rows = statistics_rows()
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows.append(rows[0])
    else: rows[0]['seed_index'] = 3
    with pytest.raises(ValueError):
        cluster_statistics(rows, tids, ['FIXED','ROTATING'], [('ROTATING','FIXED')])


def test_full_compact_report_regenerates_without_weights_or_private_tests(tmp_path):
    """Exercise the complete231-job/1512-draw layout using synthetic receipts."""
    import json
    from scripts.coding_coverage_v2_report import report
    from gearshift.coding_coverage import PUBLICATION
    def put(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj))
    root = tmp_path/'results/v2/evaluation'; tids = [f'task/{i}' for i in range(21)]
    seeds = {t: [{'seed_index': s, 'stream': f's{s}', 'answer_seed': s+9} for s in range(3)] for t in tids}
    panels = {t: {'legacy': [0,1], 'broad': [0,2]} for t in tids}
    put(tmp_path/'seeds.json', {'validation': seeds}); put(tmp_path/'panels.json', panels)
    d = {'publication_commit': PUBLICATION, 'validation_task_ids': tids, 'selected_checkpoint_sha256': 'a'*64,
         'answer_seeds_path': 'seeds.json', 'answer_seeds_sha256': sha(tmp_path/'seeds.json'),
         'panels_path': 'panels.json', 'panels_sha256': sha(tmp_path/'panels.json'), 'analysis': {'bootstrap_seed': 20260918}}
    put(tmp_path/'declaration.json', d)
    put(tmp_path/'data/coding_pilot_v1/visible/coverage_generalization.json', [{'task_id': t, 'prompt': 'write a program'} for t in tids])
    plan = {'experiment_id': 'synthetic_v2', 'jobs': [], 'declaration_path': 'declaration.json',
            'declaration_sha256': sha(tmp_path/'declaration.json'), 'training_roots': {a: 'results/v2/'+a for a in ('FIXED','ROTATING')}}
    for arm, step in [('START',0)]+[(a,s) for s in (128,256,512,768,1024) for a in ('FIXED','ROTATING')]:
        for i, tid in enumerate(tids):
            job = {'experiment_id': plan['experiment_id'], 'job_id': f'{arm}_{step:04d}_{i:02d}', 'arm': arm, 'step': step,
                   'task_ids': [tid], 'forms': ['M','H'], 'include_controls': arm == 'START', 'validation_kl': True,
                   'checkpoint': {'path': 'absent_weights.pt', 'sha256': 'a'*64 if arm == 'START' else digest([arm,step])}}
            plan['jobs'].append(job); folder = root/'jobs'/job['job_id']; task = folder/'validation'/tid.replace('/', '__')
            put(folder/'job.json', job); put(task/'source_history.json', {'task_id': tid, 'reasoning': 'saved'})
            files = []
            for condition, form in [(arm+'_M','M'),(arm+'_H','H')]+([('D','D'),('P','P')] if arm == 'START' else []):
                for seed in seeds[tid]:
                    dest = task/condition/('seed_'+str(seed['seed_index']))
                    contract = evaluation.draw_contract(job, tid, condition, form, seed, sha(task/'source_history.json'))
                    put(dest/'draw_identity.json', contract); put(dest/'answer.json', raw_record(contract))
                    put(dest/'complete.json', {'contract_sha256': digest(contract), 'answer_sha256': sha(dest/'answer.json')})
                    files.append({'path': str((dest/'answer.json').relative_to(folder)), 'sha256': sha(dest/'answer.json'),
                                  'receipt_sha256': sha(dest/'complete.json')})
            control = folder/'attempts/attempt1'/tid.replace('/', '__')/'native_splice_control.json'
            put(control, {'task_id': tid, 'passed': True})
            kl = {'task_id': tid, 'arm': arm, 'step': step, 'job_identity_sha256': digest(job), 'checkpoint_sha256': job['checkpoint']['sha256'],
                  **{p: {'positions': panels[tid][p], 'per_position_kl': [.2,.2], 'mean_kl': .2} for p in ('legacy','broad')}}
            put(task/'validation_kl.json', kl)
            put(folder/'generation_complete.json', {'job_identity_sha256': digest(job), 'files': files, 'hidden_tests_loaded': False,
                'kl_files': [{'path': str((task/'validation_kl.json').relative_to(folder)), 'sha256': sha(task/'validation_kl.json')}],
                'control_files': [{'path': str(control.relative_to(folder)), 'sha256': sha(control)}]})
    closure = evaluation.generation_closure(root, plan); put(root/'generation_closure.json', closure); scores = []
    for item in closure['files']:
        path = (root/item['path']).parent/'score.json'
        put(path, {'answer_sha256': item['sha256'], 'generation_closure_sha256': digest(closure), 'code': 'pass',
                   'score': {'passed': True, 'category': 'pass'}, 'hidden_tests_loaded_after_all_generation': True})
        scores.append({'path': str(path.relative_to(root)), 'sha256': sha(path), 'answer_path': item['path'], 'answer_sha256': item['sha256']})
    put(root/'scored_answer_manifest.json', {'generation_closure_sha256': digest(closure), 'hidden_tests_loaded_after_all_generation': True, 'files': scores})
    for arm in ('FIXED','ROTATING'):
        train = tmp_path/plan['training_roots'][arm]
        put(train/'training_steps.json', [{'step': i+1, 'arm': arm, 'gradient_predictions': 32, 'normalization_per_arm': 32,
            'cache_gradients_finite_nonzero': True, 'wall_seconds': 1., 'cache_preparation_seconds': .2,
            'continuation_forward_backward_seconds': .7, 'optimizer_seconds': .1} for i in range(1024)])
        put(train/'training_complete.json', {'full_target_completed': True, 'completed_updates': 1024, 'scored_positions': 32768})
        put(train/'training_identity.json', {'identity': 'synthetic'}); put(train/'checkpoint_manifest.json', {'checkpoints': []})
        put(train/'resume_preflight.json', {'passed': True})
        for step in (0,128,256,512,768,1024):
            put(train/'coverage'/f'step_{step:04d}.json', {'steps': step, 'scored_positions': 32*step,
                'unique_task_positions': 32 if step else 0, 'tasks': {tids[0]: {'positions': {'0': 32*step} if step else {},
                'by_category': {'code_body': {'scored': 32*step, 'unique': 32 if step else 0}}}}})
    first = report(root, plan, repo_root=tmp_path, output=tmp_path/'report1')
    second = report(root, plan, repo_root=tmp_path, output=tmp_path/'report2')
    assert first == second and first['fresh_draws'] == 1512 and first['full_1024_per_arm_completed']
    assert first['primary']['difference'] == 0
    files1 = {p.name: sha(p) for p in (tmp_path/'report1').iterdir()}
    files2 = {p.name: sha(p) for p in (tmp_path/'report2').iterdir()}
    assert files1 == files2
    assert not list(tmp_path.rglob('*.pt')) and not (tmp_path/'data/coding_pilot_v1/private').exists()
