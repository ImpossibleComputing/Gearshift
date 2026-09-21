"""Predeclared inference and diagnostic accounting, synthetic outcomes only."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from gearshift.coding_control import digest, sha, write
from scripts import coding_confirmation_report as report


def analysis(n):
    value = report.read(Path(__file__).resolve().parents[1] / 'configs/coding_pilot_v1/confirmation_01/analysis_plan.draft.json')
    return {**value, 'status': 'FROZEN', 'task_count': n}


def rows(tasks):
    return [{'task_id': t, 'condition': c, 'seed_index': s, 'passed': False,
             'missing': False, 'category': 'test_assertion'}
            for t in tasks for c in report.CONDITIONS for s in range(3)]


def test_exact_joint_task_bootstrap_and_three_draw_mean_not_best_of_three():
    tasks = ['z', 'a', 'middle']; records = rows(tasks)
    for r in records:
        r['passed'] = r['condition'] == 'ROTATING_M' and r['seed_index'] <= tasks.index(r['task_id'])
    summary, task_rows, gains, indices = report.cluster_statistics(records, tasks, analysis(3))
    expected_indices = np.random.Generator(np.random.PCG64(20260919)).integers(0, 3, (10000, 3))
    assert np.array_equal(indices, expected_indices)
    assert summary['bootstrap_indices_sha256'] == digest(expected_indices.tolist())
    primary = summary['contrasts']['ROTATING_M-FIXED_M']
    values = np.array([1 / 3, 2 / 3, 1.])
    assert primary['difference'] == pytest.approx(2 / 3)
    assert primary['ci95'] == np.quantile(values[expected_indices].mean(axis=1), [.025, .975], method='linear').tolist()
    assert summary['task_ids'] == tasks and primary['guaranteed_task_gains'] == 3
    assert len(task_rows) == 24 and len(gains) == 21
    shuffled, _, _, shuffled_indices = report.cluster_statistics(list(reversed(records)), tasks, analysis(3))
    assert shuffled == summary and np.array_equal(shuffled_indices, indices)


def test_six_secondary_intervals_use_joint_bonferroni_quantiles():
    tasks = ['a', 'b', 'c', 'd']; records = rows(tasks)
    for r in records:
        r['passed'] = r['condition'] == 'ROTATING_H' and (tasks.index(r['task_id']) > r['seed_index'])
    summary, _, _, indices = report.cluster_statistics(records, tasks, analysis(4))
    assert len(summary['contrasts']) == 7
    boot = np.array([0, 1/3, 2/3, 1.])[indices].mean(axis=1)
    for pair in report.SECONDARY:
        c = summary['contrasts']['-'.join(pair)]
        assert c['familywise_ci'] == np.quantile(boot, [.05/12, 1-.05/12], method='linear').tolist()
        assert c['ci95'] == np.quantile(boot, [.025, .975], method='linear').tolist()


def test_exact_zero_bootstrap_boundary_cannot_round_into_positive_evidence(monkeypatch):
    tasks = ['a', 'b', 'c', 'd']; records = rows(tasks)
    positive, negative = [0, 2, 2, 2], [0, 2, 3, 1]
    # Different thirds patterns with equal integer totals. Subtracting float
    # means incorrectly produces positive evidence for this possible resample.
    assert (np.array(positive) / 3).mean() - (np.array(negative) / 3).mean() > 0
    for r in records:
        counts = positive if r['condition'] == 'ROTATING_M' else negative if r['condition'] == 'FIXED_M' else [0] * 4
        r['passed'] = r['seed_index'] < counts[tasks.index(r['task_id'])]

    class AdversarialResamples:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 4, (10000, 4))
            return np.tile(np.arange(4, dtype=np.int64), (10000, 1))

    # The separate exact-PCG64 test fixes production resampling. This controlled
    # index matrix isolates the zero-percentile numerical boundary, no epsilon.
    monkeypatch.setattr(report.np.random, 'Generator', lambda bitgen: AdversarialResamples())
    summary, _, _, _ = report.cluster_statistics(records, tasks, analysis(4))
    primary = summary['contrasts']['ROTATING_M-FIXED_M']
    assert primary['difference'] == 0.
    assert primary['ci95'] == [0., 0.]
    assert primary['supports_positive_difference'] is False


def test_missing_keeps_full_denominators_and_extreme_pair_sensitivities():
    records = rows(['a', 'b'])
    for r in records:
        if r['task_id'] == 'a' and r['condition'] in report.PRIMARY and r['seed_index'] == 0:
            r.update(passed=None, missing=True, category='infrastructure_failure')
    summary, _, gains, _ = report.cluster_statistics(records, ['a', 'b'], analysis(2))
    c = summary['contrasts']['ROTATING_M-FIXED_M']
    assert c['difference'] is None and c['ci95'] is None and c['supports_positive_difference'] is None
    assert c['draws_per_condition'] == 6 and c['possible_mean_bounds'] == pytest.approx([-1/6, 1/6])
    assert c['sensitivity_lower_ci95'] == pytest.approx([-1/3, 0])
    assert c['sensitivity_upper_ci95'] == pytest.approx([0, 1/3])
    assert c['unresolved_tasks'] == 1 and c['guaranteed_task_ties'] == 1
    assert summary['conditions']['ROTATING_M']['pass_rate'] is None
    assert summary['conditions']['A']['pass_rate'] == 0
    assert [r for r in gains if r['contrast'] == 'ROTATING_M-FIXED_M'][0]['state'] == 'unresolved'


@pytest.mark.parametrize('change', ['missing_row', 'duplicate', 'extra_seed', 'numeric_boolean', 'unmarked_null'])
def test_nondeclared_or_ambiguous_population_is_rejected(change):
    records = rows(['a'])
    if change == 'missing_row': records.pop()
    elif change == 'duplicate': records.append(dict(records[0]))
    elif change == 'extra_seed': records[-1]['seed_index'] = 3
    elif change == 'numeric_boolean': records[0]['passed'] = 1
    else: records[0]['passed'] = None
    with pytest.raises(ValueError): report.cluster_statistics(records, ['a'], analysis(1))


def test_unfrozen_or_changed_analysis_rejected():
    a = analysis(1); a['status'] = 'draft'
    with pytest.raises(ValueError): report.cluster_statistics(rows(['a']), ['a'], a)
    a = analysis(1); a['bootstrap']['seed'] += 1
    with pytest.raises(ValueError): report.cluster_statistics(rows(['a']), ['a'], a)


def timed():
    raw = {'condition': 'ROTATING_H', 'source_cost_divisor_for_single_output': 1,
        'reasoning_seconds': 120., 'answer_seconds': 20., 'native_prefill_seconds': 5.,
        'mapping_seconds': 1., 'splice_seconds': 2., 'cache_clone_seconds': 3.,
        'bridge_seconds': 4., 'first_answer_token_seconds': 5., 'source_cache_reconstruction_seconds': 60.,
        'reasoning_model': 'source', 'inference_timing_complete': True,
        'single_output_inference_seconds': 151., 'sampler_timing': {'complete': True, 'reconstruction_seconds': 7.}}
    history = {'reasoning_seconds': 120., 'sampler_timing': {'complete': True,
               'initial_prefill_seconds': 10., 'reconstruction_seconds': 11.}}
    return raw, history


def test_full_source_h_native_prefill_and_bridge_are_counted_exactly_once():
    raw, history = timed(); result = report.timing_record(raw, history, {})
    assert result['single_output_inference_seconds'] == 151.
    assert result['reasoning_generation_including_checkpoint_io_seconds'] == 110.
    assert result['history_only_amortized_three_answer_seconds'] == 71.
    assert result['source_cache_reconstruction_seconds'] == 60.
    assert result['answer_resume_reconstruction_seconds'] == 7.
    assert result['checkpoint_io_seconds'] is None and result['model_loading_seconds'] is None
    raw['single_output_inference_seconds'] += raw['bridge_seconds']
    with pytest.raises(ValueError, match='full stage costs'): report.timing_record(raw, history, {})


def test_b_uses_own_history_and_p_has_no_reasoning():
    raw, history = timed(); raw.update(condition='B', reasoning_model='receiver')
    assert report.timing_record(raw, history, {})['reasoning_model'] == 'receiver'
    wrong = copy.deepcopy(history); wrong['reasoning_seconds'] = 200.
    with pytest.raises(ValueError, match='reasoning cost'): report.timing_record(raw, wrong, {})
    raw.update(condition='P', reasoning_model=None, reasoning_seconds=0., single_output_inference_seconds=31.)
    assert report.timing_record(raw, None, {})['single_output_inference_seconds'] == 31.


def test_interrupted_timing_stays_null_and_no_source_amortization_in_primary():
    raw, history = timed(); raw.update(inference_timing_complete=False, single_output_inference_seconds=None)
    raw['sampler_timing']['complete'] = False
    assert report.timing_record(raw, history, {})['single_output_inference_seconds'] is None
    raw['source_cost_divisor_for_single_output'] = 3
    with pytest.raises(ValueError, match='full reasoning'): report.timing_record(raw, history, {})


@pytest.mark.parametrize('damage', ['wrong_model', 'missing_stage', 'oversized_prefill', 'oversized_bridge'])
def test_invalid_timing_cannot_silently_produce_deployment_estimate(damage):
    raw, history = timed()
    if damage == 'wrong_model': raw['reasoning_model'] = 'receiver'
    elif damage == 'missing_stage': raw.update(native_prefill_seconds=None, single_output_inference_seconds=None)
    elif damage == 'oversized_prefill': history['sampler_timing']['initial_prefill_seconds'] = 121.
    else: raw['bridge_seconds'] = 21.
    with pytest.raises(ValueError): report.timing_record(raw, history, {})


def test_static_interface_names_and_repetition_without_execution():
    prompt = report.INTERFACE + 'class Solution:\n    def solve(self, x): pass\n'
    valid = 'class Solution:\n    def solve(self, x): return x\n'
    assert report.interface_diagnostics(valid, prompt)['required_names_present'] is True
    assert report.interface_diagnostics('def solve(x): return x', prompt)['required_names_present'] is False
    assert report.interface_diagnostics('def broken(', prompt)['syntax_valid'] is False
    assert report.interface_diagnostics('print(1)', 'stdin program')['required_names_present'] is None
    assert report.repeated_fourgrams([1] * 8) == 4/5


def test_terminal_overhead_requires_sealed_exact_record_identity_and_preserves_unknown_tail(tmp_path):
    record = {'sampler_identity_sha256': 'identity', 'sampler_timing': {'overhead': {
        'completion_receipt_file': 'completion_timing.json'}}}
    overhead = {'scope_complete': False, 'scope': 'Recorded calls only', 'exclusions': ['Unknown crash tail'],
                **{name: .1 for name in report.OVERHEAD_SECONDS}}
    value = {'schema': 1, 'state': 'complete', 'identity_sha256': 'identity',
             'record_sha256': digest(record), 'terminal_attempt_id': 'synthetic-attempt', 'overhead': overhead}
    relative = 'sampler/completion_timing.json'; write(tmp_path / relative, value)
    files = {relative: {'sha256': sha(tmp_path / relative)}}
    measured = report.overhead_receipt(tmp_path, files, 'sampler', record)
    assert measured['scope_complete'] is False
    result = report.overhead_for_draw(measured, measured)
    assert result['recorded_checkpoint_io_seconds'] == .2
    assert result['checkpoint_io_seconds'] is None
    assert result['answer_overhead_scope_complete'] is False
    with pytest.raises(ValueError, match='not generation-sealed'):
        report.overhead_receipt(tmp_path, {}, 'sampler', record)
    value['record_sha256'] = 'changed'; write(tmp_path / relative, value)
    with pytest.raises(ValueError, match='hash changed'):
        report.overhead_receipt(tmp_path, files, 'sampler', record)


def test_prompt_only_overhead_has_no_source_cost_and_keeps_checkpoint_setup_out_of_latency():
    overhead = {'scope_complete': True, **{name: 1. for name in report.OVERHEAD_SECONDS}}
    result = report.overhead_for_draw(overhead, None, prompt_only=True)
    assert result['reasoning_recorded_checkpoint_persistence_seconds'] == 0.
    assert result['checkpoint_io_seconds'] == 1.
    raw, history = timed(); raw['checkpoint_load_seconds'] = 300.
    measured = report.timing_record(raw, history, {})
    assert measured['checkpoint_load_seconds'] == 300.
    assert measured['single_output_inference_seconds'] == 151.


def synthetic_bundle(repo):
    """Small complete public-record bundle; no scorer/candidate execution."""
    task = 'synthetic/task'; root = repo / 'results'; plan_path = 'results/primary/scoring/scoring_plan.json'
    source = Path(__file__).resolve().parents[1] / 'gearshift/coding_sandbox.py'
    (repo / 'gearshift').mkdir(); (repo / 'gearshift/coding_sandbox.py').write_bytes(source.read_bytes())
    scorer = {'policy_sha256': 'policy-canonical', 'files': {'gearshift/coding_sandbox.py': sha(source)}}
    write(repo / 'scorer.json', scorer)
    seed_rows = [{'seed_index': s, 'answer_seed': 100 + s, 'stream': 'answer_' + str(s)} for s in range(3)]
    write(repo / 'seeds.json', {'tasks': {task: {'answers': seed_rows}}})
    write(repo / 'analysis.json', analysis(1))
    write(repo / 'visible.json', [{'task_id': task, 'prompt': 'Write a stdin program.'}])
    d = {'status': 'FROZEN', 'generation_authorized_by_this_file': True, 'experiment_id': 'synthetic',
        'task_ids': [task], 'task_count': 1, 'primary_conditions': report.CONDITIONS,
        'primary_answer_count': 24, 'primary_training_seed': 20260915,
        'source_commit': 'synthetic-source', 'publication_commit': 'synthetic-publication',
        'primary_checkpoints': {arm: {'step': 1024, 'mapper_sha256': arm.lower() + '-synthetic-sha'}
                                for arm in ('FIXED', 'ROTATING')},
        'analysis_path': 'analysis.json', 'seeds_path': 'seeds.json', 'visible_path': 'visible.json',
        'inputs': {p: sha(repo / p) for p in ['analysis.json', 'seeds.json', 'visible.json']},
        'scorer': {'identity_path': 'scorer.json', 'identity_file_sha256': sha(repo / 'scorer.json'),
                   'policy_identity_sha256': 'policy-canonical'}}
    write(repo / 'declaration.json', d); dsha = sha(repo / 'declaration.json')
    _, h = timed()
    h.update(task_id=task, reasoning_ids=[1, 151668], natural_boundary=True, reasoning_capped=False, early_eos=False)
    files = []; histories = []

    def add(relative, value):
        write(root / relative, value)
        files.append({'path': relative, 'bytes': (root / relative).stat().st_size, 'sha256': sha(root / relative)})

    for kind, folder in [('source', 'large_history'), ('small', 'small_history')]:
        relative = f'primary/tasks/synthetic__task/{folder}/source_history.json'
        add(relative, h); histories.append({'task_id': task, 'kind': kind, 'sha256': sha(root / relative)})
    add('workers/synthetic/model_setup.json', {'declaration_sha256': dsha,
        'charged_to_single_output_inference': False, 'model_loading_seconds': {'source': 10., 'receiver': 3.},
        'mapper_initialization_seconds': 2., 'started_epoch': 1., 'completed_epoch': 16.})
    mapper_setup = 'primary/tasks/synthetic__task/mapper_loads/attempt_ROTATING_H.json'
    add(mapper_setup, {'declaration_sha256': dsha, 'charged_to_single_output_inference': False,
        'task_id': task, 'condition': 'ROTATING_H', 'checkpoint_load_seconds': 7.})
    answers = []; records = []
    for condition in report.CONDITIONS:
        for seed in seed_rows:
            contract = {'task_id': task, 'condition': condition, **seed, 'cohort': 'primary',
                        'declaration_sha256': dsha}
            raw, _ = timed(); raw.update(contract, answer_text='```python\nprint(1)\n```',
                                        answer_ids=[11, 151645], answer_ended_eos=True, answer_capped=False,
                                        checkpoint_load_seconds=0., checkpoint_load_receipt_path=None,
                                        checkpoint_load_receipt_sha256=None)
            if condition == 'ROTATING_H':
                raw.update(checkpoint_load_seconds=7., checkpoint_load_receipt_path=mapper_setup,
                           checkpoint_load_receipt_sha256=sha(root / mapper_setup))
            if condition == 'P': raw.update(reasoning_seconds=0., reasoning_model=None, single_output_inference_seconds=31.)
            elif condition == 'B': raw['reasoning_model'] = 'receiver'
            relative = f'primary/tasks/synthetic__task/{condition}/seed_{seed["seed_index"]}/answer.json'
            add(relative, raw)
            answers.append({'contract': contract, 'contract_sha256': digest(contract),
                            'path': relative, 'answer_sha256': sha(root / relative)})
            code = report.extract(raw['answer_text'])
            records.append({'record_id': digest(contract), 'contract': contract, 'contract_sha256': digest(contract),
                'answer_path': relative, 'answer_sha256': sha(root / relative),
                'code_sha256': hashlib.sha256(code.encode()).hexdigest(), **{k: contract[k] for k in ['task_id', 'condition', 'seed_index', 'answer_seed']}})
    closure = {'experiment_id': 'synthetic', 'declaration_sha256': dsha, 'task_ids': [task], 'task_count': 1,
        'conditions': report.CONDITIONS, 'answer_count': 24, 'histories': histories, 'answers': answers,
        'files': files, 'all_primary_generation_complete': True, 'scoring_requires_this_complete_seal': True}
    closure['closure_sha256'] = digest(closure); write(root / 'primary/generation_closure.json', closure)
    plan = {'cohort': 'primary', 'result_root': 'results', 'experiment_id': 'synthetic',
        'task_ids': [task], 'conditions': report.CONDITIONS, 'expected_answers': 24,
        'declaration_path': 'declaration.json', 'declaration_sha256': dsha,
        'scorer_identity': scorer, 'scorer_identity_sha256': digest(scorer),
        'generation_closure_path': 'primary/generation_closure.json',
        'generation_closure_sha256': closure['closure_sha256'],
        'generation_closure_identity_sha256': closure['closure_sha256'],
        'generation_closure_file_sha256': sha(root / 'primary/generation_closure.json'),
        'manifest_path': 'primary/scoring/scored_answer_manifest.json', 'records': records}
    write(repo / plan_path, plan); manifest_files = []
    for record in records:
        code = report.extract(report.read(root / record['answer_path'])['answer_text'])
        outcome = record['condition'] == 'ROTATING_M'
        score = {'passed': outcome, 'missing': False, 'category': 'pass' if outcome else 'test_assertion',
                 'scorer_identity': scorer, 'receipts': [{'attempts': [{'wall_seconds': .1}]}]}
        binding = {'scoring_plan_identity_sha256': digest(plan), 'scoring_plan_sha256': sha(repo / plan_path),
            'declaration_sha256': dsha, 'generation_closure_identity_sha256': closure['closure_sha256'],
            'generation_closure_file_sha256': plan['generation_closure_file_sha256'],
            'scorer_identity_sha256': digest(scorer),
            **{k: record[k] for k in ['record_id', 'answer_sha256', 'contract_sha256', 'code_sha256']}}
        receipt = {'binding': binding, 'contract': record['contract'], 'code': code, 'score_v2': score,
            'hidden_tests_loaded_after_primary_generation': True,
            **{k: record[k] for k in ['task_id', 'condition', 'seed_index', 'answer_seed', 'answer_path', 'answer_sha256']}}
        path = 'primary/scoring/scores/' + record['record_id'] + '/score.json'; write(root / path, receipt)
        manifest_files.append({'path': path, 'sha256': sha(root / path), 'missing': False, 'passed': outcome,
            **{k: record[k] for k in ['answer_path', 'answer_sha256', 'task_id', 'condition', 'seed_index', 'answer_seed']}})
    manifest = {'experiment_id': 'synthetic', 'cohort': 'primary', 'declaration_sha256': dsha,
        'generation_closure_sha256': closure['closure_sha256'], 'generation_closure_identity_sha256': closure['closure_sha256'],
        'generation_closure_file_sha256': plan['generation_closure_file_sha256'], 'scorer_identity_sha256': digest(scorer),
        'scoring_plan_path': plan_path, 'scoring_plan_sha256': sha(repo / plan_path), 'scoring_plan_identity_sha256': digest(plan),
        'task_ids': [task], 'conditions': report.CONDITIONS, 'expected_answers': 24, 'committed_answers': 24,
        'missing_answers': 0, 'hidden_tests_loaded_after_primary_generation': True, 'raw_answers_unchanged': True,
        'files': manifest_files}
    write(root / plan['manifest_path'], manifest)
    return plan_path, root, plan, manifest


def test_report_regenerates_from_complete_compact_bundle_and_preserves_inputs(tmp_path):
    path, root, plan, manifest = synthetic_bundle(tmp_path)
    before = {p: sha(p) for p in tmp_path.rglob('*.json')}
    summary = report.report(path, repo=tmp_path)
    assert summary['answers'] == 24 and summary['missing_draws'] == 0
    assert summary['contrasts']['ROTATING_M-FIXED_M']['difference'] == 1.
    output = root / 'primary/report'
    assert (output / 'REPORT.md').exists() and (output / 'per_task_seed.csv').exists()
    markdown = (output / 'REPORT.md').read_text()
    assert 'Bonferroni 99.1667% interval (pp)' in markdown
    assert all(f'{a} − {b}' in markdown for a, b in report.SECONDARY)
    assert sha(tmp_path / 'declaration.json') in markdown
    assert 'fixed-synthetic-sha' in markdown and 'rotating-synthetic-sha' in markdown
    assert summary['setup_measurements']['worker_setup_receipts'] == 1
    assert summary['setup_measurements']['mapper_checkpoint_receipts'] == 1
    assert len(summary['setup_measurements']['records']) == 2  # Three answer references share one load.
    setup = summary['setup_measurements']['records']
    assert next(r for r in setup if r['kind'] == 'worker_models')['source_loading_seconds'] == 10.
    assert next(r for r in setup if r['kind'] == 'mapper_checkpoint')['checkpoint_load_seconds'] == 7.
    assert summary['diagnostics']['ROTATING_H']['timing_all_outcomes']['single_output_inference_seconds']['mean'] == 151.
    with np.load(output / 'bootstrap_indices.npz', allow_pickle=False) as saved:
        assert saved['indices'].shape == (10000, 1)
    assert all(sha(p) == expected for p, expected in before.items())
    assert report.read(output / 'report_manifest.json')['generation_or_candidate_execution_performed'] is False


@pytest.mark.parametrize('damage', ['raw', 'receipt', 'missing', 'closure', 'order', 'private_barrier'])
def test_report_rejects_changed_or_incomplete_evidence_before_creating_output(tmp_path, damage):
    path, root, plan, manifest = synthetic_bundle(tmp_path)
    first = manifest['files'][0]
    if damage == 'raw': write(root / first['answer_path'], {'corrupt': True})
    elif damage == 'receipt':
        value = report.read(root / first['path']); value['binding']['generation_closure_identity_sha256'] = 'wrong'
        write(root / first['path'], value); first['sha256'] = sha(root / first['path'])
    elif damage == 'missing': manifest['files'].pop()
    elif damage == 'closure':
        value = report.read(root / plan['generation_closure_path']); value['all_primary_generation_complete'] = False
        write(root / plan['generation_closure_path'], value)
    elif damage == 'order': manifest['files'].reverse()
    else: manifest['hidden_tests_loaded_after_primary_generation'] = False
    write(root / plan['manifest_path'], manifest)
    with pytest.raises(ValueError): report.report(path, repo=tmp_path)
    assert not (root / 'primary/report').exists()


def test_compact_report_preserves_explicit_missing_score_and_full_population(tmp_path):
    path, root, plan, manifest = synthetic_bundle(tmp_path)
    first = next(r for r in manifest['files'] if r['condition'] == 'ROTATING_M')
    receipt = report.read(root / first['path'])
    receipt['score_v2'].update(passed=None, missing=True, category='infrastructure_failure')
    write(root / first['path'], receipt); first.update(passed=None, missing=True, sha256=sha(root / first['path']))
    manifest['missing_answers'] = 1; write(root / plan['manifest_path'], manifest)
    summary = report.report(path, repo=tmp_path)
    assert summary['answers'] == 24 and summary['missing_draws'] == 1
    assert summary['contrasts']['ROTATING_M-FIXED_M']['difference'] is None
    assert summary['conditions']['ROTATING_M']['possible_mean_bounds'] == pytest.approx([2/3, 1])
    assert 'Primary estimation is incomplete' in (root / 'primary/report/REPORT.md').read_text()


def test_report_output_cannot_overwrite_preserved_other_experiments(tmp_path):
    path, root, plan, manifest = synthetic_bundle(tmp_path)
    with pytest.raises(ValueError, match='report namespace'):
        report.report(path, repo=tmp_path, output='prior_pilot/report')
    assert not (tmp_path / 'prior_pilot').exists()
