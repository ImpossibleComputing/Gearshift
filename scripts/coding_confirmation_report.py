#!/usr/bin/env python3
"""Regenerate the predeclared primary report from compact immutable receipts.

No model loading, private-test access, candidate execution or result selection.
All null outcomes remain missing; the denominator is always every frozen task
and its three answer draws. Secondary training is intentionally independent.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import digest, sha, write
from gearshift.coding_sandbox import extract

CONDITIONS = ['A', 'B', 'D', 'P', 'FIXED_M', 'ROTATING_M', 'FIXED_H', 'ROTATING_H']
PRIMARY = ['ROTATING_M', 'FIXED_M']
SECONDARY = [['ROTATING_H', x] for x in ['FIXED_H', 'D', 'P', 'B', 'A', 'ROTATING_M']]
Q95 = [.025, .975]
QFAMILY = [.05 / 12, 1 - .05 / 12]
INTERFACE = 'Implement the provided Python interface. Return the result; do not read stdin.\n'
OVERHEAD_SECONDS = ['checkpoint_persistence_seconds', 'checkpoint_preparation_seconds',
                    'telemetry_seconds', 'progress_publish_seconds',
                    'result_materialization_seconds', 'identity_binding_seconds']


def read(path): return json.loads(Path(path).read_text())


def scoped(root, relative):
    root = Path(root).resolve(); relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Evidence path must be relative and scoped')
    path = root / relative
    for parent in [path, *path.parents]:
        if parent == root: break
        if parent.is_symlink(): raise ValueError('Linked evidence is not permitted')
    path.resolve().relative_to(root)
    return path


def checked(root, relative, expected):
    path = scoped(root, relative)
    if not isinstance(expected, str) or sha(path) != expected:
        raise ValueError('Evidence hash changed: ' + str(relative))
    return read(path)


def validate_analysis(a, task_ids):
    b = a['bootstrap']
    if (a.get('status') != 'FROZEN' or a.get('task_count') != len(task_ids) or
            a.get('answer_draws_per_task_condition') != 3 or a.get('endpoint') != 1024 or
            a.get('primary_contrast') != PRIMARY or a.get('secondary_contrasts') != SECONDARY or
            b.get('resamples') != 10000 or b.get('seed') != 20260919 or
            b.get('rng') != 'numpy.random.Generator(numpy.random.PCG64(20260919))' or
            b.get('quantile_method') != 'linear' or b.get('secondary_family_size') != 6 or
            b.get('primary_and_descriptive_interval_quantiles') != Q95 or
            b.get('secondary_familywise_interval_quantiles') != QFAMILY):
        raise ValueError('Frozen analysis specification differs from the implemented method')
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError('Frozen task order must be nonempty and unique')


def quantiles(values, q): return np.quantile(values, q, method='linear').tolist()


def cluster_statistics(rows, task_ids, analysis):
    """One exact joint task resample matrix, including all three draws per task."""
    validate_analysis(analysis, task_ids)
    lookup = {}
    for row in rows:
        key = row['task_id'], row['condition'], row['seed_index']
        if key in lookup: raise ValueError('Duplicate task/condition/seed score')
        value = row['passed']
        if value is not None and type(value) is not bool:
            raise ValueError('Only binary outcomes or explicit null missingness are valid')
        if type(row.get('missing')) is not bool or row['missing'] is not (value is None):
            raise ValueError('Outcome and explicit missingness disagree')
        lookup[key] = row
    expected = {(t, c, s) for t in task_ids for c in CONDITIONS for s in range(3)}
    if set(lookup) != expected: raise ValueError('Incomplete or unexpected declared score population')
    lower = np.zeros((len(task_ids), len(CONDITIONS)), dtype=np.int64)
    upper = lower.copy(); task_rows = []
    for i, tid in enumerate(task_ids):
        for j, condition in enumerate(CONDITIONS):
            values = [lookup[tid, condition, seed]['passed'] for seed in range(3)]
            successes = sum(v is True for v in values); missing = sum(v is None for v in values)
            lower[i, j] = successes; upper[i, j] = successes + missing
            task_rows.append({'task_id': tid, 'condition': condition, 'draws': 3,
                'passed_draws': successes, 'missing_draws': missing,
                'pass_rate': None if missing else successes / 3,
                'success_lower': successes / 3, 'success_upper': (successes + missing) / 3})
    indices = np.random.Generator(np.random.PCG64(20260919)).integers(
        0, len(task_ids), size=(10000, len(task_ids)))
    # Sum integer successes first. Subtract paired integer sums before dividing,
    # so an exact zero contrast cannot become a tiny positive floating endpoint.
    blo, bhi = lower[indices].sum(axis=1), upper[indices].sum(axis=1)
    denominator = 3 * len(task_ids)
    conditions = {}
    for j, condition in enumerate(CONDITIONS):
        selected = [lookup[t, condition, s] for t in task_ids for s in range(3)]
        missing = sum(r['missing'] for r in selected)
        conditions[condition] = {'draws': len(selected), 'tasks': len(task_ids),
            'passed_draws': sum(r['passed'] is True for r in selected), 'missing_draws': missing,
            'point_estimate_complete': missing == 0,
            'pass_rate': None if missing else float(lower[:, j].sum() / denominator),
            'ci95': None if missing else quantiles(blo[:, j] / denominator, Q95),
            'possible_mean_bounds': [float(lower[:, j].sum() / denominator), float(upper[:, j].sum() / denominator)],
            'sensitivity_lower_ci95': quantiles(blo[:, j] / denominator, Q95),
            'sensitivity_upper_ci95': quantiles(bhi[:, j] / denominator, Q95),
            'categories': dict(collections.Counter(r.get('category', 'unspecified') for r in selected))}
    contrasts = {}; gains = []
    for positive, negative in [PRIMARY, *SECONDARY]:
        ja, jb = CONDITIONS.index(positive), CONDITIONS.index(negative)
        lo, hi = lower[:, ja] - upper[:, jb], upper[:, ja] - lower[:, jb]
        bootlo, boothi = (blo[:, ja] - bhi[:, jb]) / denominator, (bhi[:, ja] - blo[:, jb]) / denominator
        complete = not conditions[positive]['missing_draws'] and not conditions[negative]['missing_draws']
        is_primary = [positive, negative] == PRIMARY
        q = Q95 if is_primary else QFAMILY
        ci = quantiles(bootlo, q) if complete else None
        name = positive + '-' + negative
        contrasts[name] = {'positive': positive, 'negative': negative,
            'role': 'primary' if is_primary else 'secondary_six_comparison_family',
            'tasks': len(task_ids), 'draws_per_condition': 3 * len(task_ids),
            'point_estimate_complete': complete, 'difference': float(lo.sum() / denominator) if complete else None,
            'ci95': quantiles(bootlo, Q95) if complete else None,
            'familywise_ci': ci if not is_primary else None,
            'inference_interval': ci, 'inference_quantiles': q,
            'supports_positive_difference': bool(ci[0] > 0) if ci else None,
            'possible_mean_bounds': [float(lo.sum() / denominator), float(hi.sum() / denominator)],
            'sensitivity_lower_ci95': quantiles(bootlo, Q95),
            'sensitivity_upper_ci95': quantiles(boothi, Q95),
            'sensitivity_lower_inference_interval': quantiles(bootlo, q),
            'sensitivity_upper_inference_interval': quantiles(boothi, q),
            'guaranteed_task_gains': int((lo > 0).sum()), 'guaranteed_task_losses': int((hi < 0).sum()),
            'guaranteed_task_ties': int(((lo == 0) & (hi == 0)).sum()),
            'unresolved_tasks': int(((lo <= 0) & (hi >= 0) & (lo != hi)).sum())}
        for i, tid in enumerate(task_ids):
            gains.append({'contrast': name, 'task_id': tid, 'difference_lower': float(lo[i] / 3),
                'difference_upper': float(hi[i] / 3), 'state': 'gain' if lo[i] > 0 else 'loss' if hi[i] < 0
                else 'tie' if lo[i] == hi[i] == 0 else 'unresolved'})
    return {'task_ids': task_ids, 'task_clusters': len(task_ids), 'seeds_per_task': 3,
        'bootstrap_seed': 20260919, 'joint_resamples': 10000,
        'bootstrap_indices_sha256': digest(indices.tolist()), 'numpy_version': np.__version__,
        'conditions': conditions, 'contrasts': contrasts,
        'method': 'Equal task weights, mean of all three fixed draws; paired percentile task-cluster bootstrap with one joint index matrix. No best-of-three or complete-case primary.',
        'missingness': 'Null is not failure. Sensitivity intervals replace missing outcomes by extreme 0/1 values and are not observed complete-data inference.',
        'multiplicity': analysis['bootstrap']['multiplicity_note'],
        'interpretive_limits': ['An interval containing zero is inconclusive, not equivalence.',
            'H beating P alone does not identify a causal contribution of the particular transferred reasoning.',
            'Primary training-seed inference is conditional on the fixed initializer, corpus and training seed; any second seed is reported separately.']}, task_rows, gains, indices


def interface_diagnostics(code, prompt):
    """Static interface evidence only; never executes a submitted program."""
    starter = prompt.split(INTERFACE, 1)[1] if INTERFACE in prompt else None
    wanted = set(re.findall(r'^\s*(?:async\s+)?def (\w+)\(', starter or '', re.M))
    expects_solution = bool(re.search(r'\bclass\s+Solution\b', starter or ''))
    try: tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return {'syntax_valid': False, 'callable_interface_expected': starter is not None,
                'required_names_present': False if starter is not None else None}
    if expects_solution:
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Solution']
        names = {n.name for c in classes for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        matched = bool(classes) and wanted <= names
    else:
        names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        matched = wanted <= names
    return {'syntax_valid': True, 'callable_interface_expected': starter is not None,
            'required_names_present': matched if starter is not None else None}


def repeated_fourgrams(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(max(0, len(ids) - 3))]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.


def seconds(value):
    if value is None: return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Timing must be nonnegative finite seconds or null')
    return float(value)


def timing_record(raw, history, score):
    """Keep deployment estimates distinct from interrupted/setup/fleet costs."""
    condition = raw['condition']; htiming = history.get('sampler_timing', {}) if history else {}
    atiming = raw.get('sampler_timing', {})
    names = ['reasoning_seconds', 'answer_seconds', 'native_prefill_seconds', 'mapping_seconds',
             'splice_seconds', 'cache_clone_seconds', 'bridge_seconds', 'first_answer_token_seconds',
             'source_cache_reconstruction_seconds', 'reasoning_resume_reconstruction_seconds',
             'checkpoint_load_seconds']
    row = {name: seconds(raw.get(name)) for name in names}
    if raw.get('source_cost_divisor_for_single_output') != 1:
        raise ValueError('Single-answer cost must charge the full reasoning history')
    expected_model = None if condition == 'P' else 'receiver' if condition == 'B' else 'source'
    if raw.get('reasoning_model') != expected_model:
        raise ValueError('Timing reasoning model differs from the declared condition')
    if condition == 'P':
        if row['reasoning_seconds'] != 0 or history is not None:
            raise ValueError('Prompt-only timing cannot contain a source reasoning cost')
        prefill = 0.
    else:
        if history is None or row['reasoning_seconds'] != seconds(history.get('reasoning_seconds')):
            raise ValueError('Full source/small reasoning cost changed')
        prefill = seconds(htiming.get('initial_prefill_seconds'))
    if prefill is not None and row['reasoning_seconds'] is not None and prefill > row['reasoning_seconds'] + 1e-8:
        raise ValueError('Reasoning prefill exceeds its enclosing active-time measurement')
    if row['bridge_seconds'] is not None and row['answer_seconds'] is not None and row['bridge_seconds'] > row['answer_seconds'] + 1e-8:
        raise ValueError('Answer bridge exceeds its enclosing active-time measurement')
    row['reasoning_prompt_prefill_seconds'] = prefill
    row['reasoning_generation_including_checkpoint_io_seconds'] = (
        None if prefill is None or row['reasoning_seconds'] is None else max(0., row['reasoning_seconds'] - prefill))
    row['answer_resume_reconstruction_seconds'] = seconds(atiming.get('reconstruction_seconds'))
    row['reasoning_resume_reconstruction_seconds'] = seconds(htiming.get('reconstruction_seconds')) if history else 0.
    complete = raw.get('inference_timing_complete') is True
    if complete and (atiming.get('complete') is not True or (history and htiming.get('complete') is not True)):
        raise ValueError('Timing falsely claims complete interrupted measurements')
    parts = [row[n] for n in ['reasoning_seconds', 'answer_seconds', 'native_prefill_seconds',
                              'mapping_seconds', 'splice_seconds', 'cache_clone_seconds']]
    if complete and any(x is None for x in parts):
        raise ValueError('Complete inference timing lacks a required stage measurement')
    total = sum(parts) if complete and all(x is not None for x in parts) else None
    supplied = seconds(raw.get('single_output_inference_seconds'))
    if (total is None) != (supplied is None) or (total is not None and not math.isclose(total, supplied, rel_tol=1e-10, abs_tol=1e-8)):
        raise ValueError('Single-output timing does not match recorded full stage costs')
    row.update(inference_timing_complete=complete, single_output_inference_seconds=total,
        reasoning_model=raw.get('reasoning_model'),
        history_only_amortized_three_answer_seconds=(None if total is None else total - row['reasoning_seconds'] * 2 / 3),
        checkpoint_io_seconds=None, model_loading_seconds=None)
    attempts = [a for test in score.get('receipts', []) for a in test.get('attempts', [])]
    measured = [seconds(a.get('wall_seconds')) for a in attempts]
    row['cpu_scoring_recorded_wall_seconds'] = sum(v for v in measured if v is not None) if attempts else None
    row['cpu_scoring_attempts_without_wall_timing'] = sum(v is None for v in measured)
    return row


def metric_summary(values):
    finite = [v for v in values if v is not None]
    return {'measured': len(finite), 'unmeasured': len(values) - len(finite),
            'mean': float(np.mean(finite)) if finite else None,
            'median': float(np.median(finite)) if finite else None}


def overhead_receipt(root, files, folder, record):
    """Read only the immutable terminal sidecar included in the whole seal."""
    snapshot = record.get('sampler_timing', {}).get('overhead')
    if snapshot is None: return None
    if snapshot.get('completion_receipt_file') != 'completion_timing.json':
        raise ValueError('Sampler overhead lacks its immutable completion receipt')
    relative = str(Path(folder) / 'completion_timing.json')
    if relative not in files: raise ValueError('Sampler completion overhead is not generation-sealed')
    value = checked(root, relative, files[relative]['sha256']); overhead = value['overhead']
    if (value.get('schema') != 1 or value.get('state') != 'complete' or
            value.get('identity_sha256') != record['sampler_identity_sha256'] or
            value.get('record_sha256') != digest(record) or
            type(overhead.get('scope_complete')) is not bool):
        raise ValueError('Sampler overhead identity or completeness differs')
    return {'path': relative, 'sha256': files[relative]['sha256'],
        'sampler_identity_sha256': value['identity_sha256'], 'terminal_attempt_id': value['terminal_attempt_id'],
        'scope_complete': overhead['scope_complete'],
        **{name: seconds(overhead.get(name)) for name in OVERHEAD_SECONDS},
        'scope': overhead.get('scope'), 'exclusions': overhead.get('exclusions')}


def overhead_for_draw(answer, reasoning, prompt_only=False):
    row = {}
    for prefix, receipt in [('answer', answer), ('reasoning', reasoning)]:
        for name in OVERHEAD_SECONDS:
            row[prefix + '_recorded_' + name] = (receipt[name] if receipt else
                0. if prefix == 'reasoning' and prompt_only else None)
        row[prefix + '_overhead_scope_complete'] = (receipt['scope_complete'] if receipt else
            True if prefix == 'reasoning' and prompt_only else None)
    values = [row[p + '_recorded_checkpoint_persistence_seconds'] for p in ('answer', 'reasoning')]
    complete = all(row[p + '_overhead_scope_complete'] is True for p in ('answer', 'reasoning'))
    row['recorded_checkpoint_io_seconds'] = sum(values) if all(v is not None for v in values) else None
    row['checkpoint_io_seconds'] = row['recorded_checkpoint_io_seconds'] if complete else None
    return row


def diagnostics_summary(rows):
    result = {}
    timing_names = [n for n in rows[0] if n.endswith('_seconds')] if rows else []
    for condition in CONDITIONS:
        selected = [r for r in rows if r['condition'] == condition]
        result[condition] = {'draws': len(selected),
            'categories': dict(collections.Counter(r['category'] for r in selected)),
            'answer_eos': sum(r['answer_ended_eos'] for r in selected),
            'answer_cap': sum(r['answer_capped'] for r in selected),
            'syntax_invalid': sum(not r['syntax_valid'] for r in selected),
            'required_interface_names_missing': sum(r['required_names_present'] is False for r in selected),
            'answer_tokens': metric_summary([r['answer_tokens'] for r in selected]),
            'repeated_fourgram_fraction': metric_summary([r['repeated_fourgram_fraction'] for r in selected]),
            'timing_all_outcomes': {n: metric_summary([r[n] for r in selected]) for n in timing_names},
            'timing_by_outcome': {label: {n: metric_summary([r[n] for r in selected if r['passed'] is value])
                for n in timing_names} for label, value in [('passed', True), ('failed', False), ('missing', None)]}}
    return result


def csv_write(path, rows):
    if not rows: raise ValueError('Cannot emit an empty declared table')
    with Path(path).open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def render(summary):
    primary = summary['contrasts']['ROTATING_M-FIXED_M']
    if primary['point_estimate_complete']:
        ci = primary['ci95']
        result = f"ROTATING_M − FIXED_M: {100 * primary['difference']:+.2f} percentage points (paired 95% interval {100 * ci[0]:+.2f} to {100 * ci[1]:+.2f})."
        decision = 'Supports a positive difference under the frozen primary rule.' if primary['supports_positive_difference'] else 'Does not establish a positive difference; an interval spanning zero is inconclusive, not equivalence.'
    else:
        lo, hi = primary['possible_mean_bounds']
        result = f'Primary estimation is incomplete because of missing infrastructure outcomes. Full-denominator possible mean contrast: {100*lo:+.2f} to {100*hi:+.2f} percentage points.'
        decision = 'No complete-case primary estimate or definitive primary claim is substituted.'
    identities = summary['identities']
    lines = ['# Fresh-task confirmation', '', result, '', decision, '',
        f"Experiment `{summary['experiment_id']}`; primary training seed `{summary['training_seed']}`; declaration SHA-256 `{summary['bindings']['declaration_sha256']}`.", '',
        f"Scientific source commit `{identities['source_commit']}`; preserved publication commit `{identities['publication_commit']}`.", '',
        f"Population: {summary['task_clusters']} frozen task clusters; three fixed answer draws per task and condition. {summary['missing_draws']} explicit missing outcomes across all conditions.", '',
        '| Condition | Passed / declared draws | Missing | Mean pass rate |', '|---|---:|---:|---:|']
    for name, row in summary['conditions'].items():
        rate = 'incomplete' if row['pass_rate'] is None else f"{100*row['pass_rate']:.2f}%"
        lines.append(f"| {name} | {row['passed_draws']} / {row['draws']} | {row['missing_draws']} | {rate} |")
    lines += ['', 'The six prespecified hybrid comparisons use Bonferroni 99.1667% intervals, with ordinary 95% intervals retained as descriptive. All comparisons share the exact saved 10,000 task-cluster resamples.', '',
        '| Contrast | Difference (pp) | Descriptive 95% interval (pp) | Bonferroni 99.1667% interval (pp) |',
        '|---|---:|---:|---:|']
    for pair in SECONDARY:
        row = summary['contrasts']['-'.join(pair)]
        if row['point_estimate_complete']:
            difference = f"{100*row['difference']:+.2f}"
            ordinary = f"[{100*row['ci95'][0]:+.2f}, {100*row['ci95'][1]:+.2f}]"
            adjusted = f"[{100*row['familywise_ci'][0]:+.2f}, {100*row['familywise_ci'][1]:+.2f}]"
        else:
            difference = f"incomplete; possible [{100*row['possible_mean_bounds'][0]:+.2f}, {100*row['possible_mean_bounds'][1]:+.2f}]"
            ordinary = adjusted = 'incomplete; sensitivity intervals in summary.json'
        lines.append(f"| {pair[0]} − {pair[1]} | {difference} | {ordinary} | {adjusted} |")
    lines += ['', 'Frozen mapper identities:', '']
    for arm, checkpoint in identities['primary_checkpoints'].items():
        lines.append(f"- {arm}: step `{checkpoint['step']}`, mapper SHA-256 `{checkpoint['mapper_sha256']}`.")
    lines += ['', 'Recorded single-output stage-sum estimates are descriptive seconds, with the full history charged to each answer:', '',
        '| Condition | All outcomes mean | Passing answers mean | Failing answers mean | Unmeasured draws |',
        '|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        diagnostic = summary['diagnostics'][condition]
        all_times = diagnostic['timing_all_outcomes']['single_output_inference_seconds']
        passing = diagnostic['timing_by_outcome']['passed']['single_output_inference_seconds']['mean']
        failing = diagnostic['timing_by_outcome']['failed']['single_output_inference_seconds']['mean']
        fmt = lambda value: 'unmeasured' if value is None else f'{value:.2f}'
        lines.append(f"| {condition} | {fmt(all_times['mean'])} | {fmt(passing)} | {fmt(failing)} | {all_times['unmeasured']} |")
    setup = summary['setup_measurements']
    lines += ['', f"Separate measured setup: {setup['worker_setup_receipts']} unique worker model-loading receipts and {setup['mapper_checkpoint_receipts']} unique mapper-loading receipts; repeated references from three draws are counted once. Details are in setup_measurements.csv when present.", '',
        'Missingness sensitivities retain every task and all three draw denominators. Their intervals concern extreme assigned outcomes, not observed complete-data inference.', '',
        'Timing charges the full reasoning history to each single-answer estimate; B uses its independent small-model history, and H includes receiver-native prompt prefill. Bridge time is already inside answer time. History-only three-answer amortization is a separately labelled deployment estimate. Cache reconstruction, worker model setup and checkpoint loading remain separate. Closure-bound sampler sidecars report measured checkpoint persistence, preparation, telemetry, progress publication and materialization overhead; none is added again to the unchanged active sampler time or subtracted to invent model-only latency. Sidecar/attempt-receipt writes and unknown crash tails remain excluded or incomplete as recorded. Unmeasured fleet, rental and interrupted setup costs are not invented.', '',
        'Timing is split by pass/failure/missing outcomes. Short failed answers are not evidence of useful speedup, and fleet parallelism is not a per-answer latency gain. Static interface checks inspect syntax and required public names only; correctness comes from the frozen scorer.', '',
        'H beating P alone does not establish the causal contribution of the particular transferred reasoning. A second training seed, when available, must remain a separate report; this primary report does not wait for it.', '']
    return '\n'.join(lines)


def load_records(plan_path, repo=ROOT):
    """Validate provenance and whole-population closure before looking at scores."""
    repo = Path(repo).resolve(); plan_path = scoped(repo, plan_path)
    plan = read(plan_path); plan_sha = sha(plan_path); plan_identity = digest(plan)
    if plan.get('cohort') != 'primary':
        raise ValueError('This report is primary-only; secondary training must be reported separately')
    root = scoped(repo, plan['result_root'])
    d = checked(repo, plan['declaration_path'], plan['declaration_sha256'])
    tids = d['task_ids']
    if (d.get('status') != 'FROZEN' or d.get('generation_authorized_by_this_file') is not True or
            d.get('primary_conditions') != CONDITIONS or d.get('task_count') != len(tids) or
            plan['experiment_id'] != d['experiment_id'] or plan['task_ids'] != tids or
            plan['conditions'] != CONDITIONS or plan['expected_answers'] != 24 * len(tids) or
            d['primary_answer_count'] != 24 * len(tids)):
        raise ValueError('Frozen primary population or experiment identity differs')
    analysis = checked(repo, d['analysis_path'], d['inputs'][d['analysis_path']])
    validate_analysis(analysis, tids)
    seeds = checked(repo, d['seeds_path'], d['inputs'][d['seeds_path']])
    visible_rows = checked(repo, d['visible_path'], d['inputs'][d['visible_path']])
    visible = {r['task_id']: r for r in visible_rows}
    if len(visible) != len(visible_rows) or not set(tids) <= set(visible):
        raise ValueError('Frozen public prompt population differs')
    scorer = checked(repo, d['scorer']['identity_path'], d['scorer']['identity_file_sha256'])
    if plan['scorer_identity'] != scorer or plan['scorer_identity_sha256'] != digest(scorer):
        raise ValueError('Frozen scorer identity differs')
    if scorer['policy_sha256'] != d['scorer']['policy_identity_sha256']:
        raise ValueError('Canonical scorer policy identity differs')
    # The unchanged extractor is the only scorer implementation used here.
    if any(sha(base / 'gearshift/coding_sandbox.py') != scorer['files']['gearshift/coding_sandbox.py']
           for base in (repo, ROOT)):
        raise ValueError('Frozen candidate extraction implementation changed')
    closure = checked(root, plan['generation_closure_path'], plan['generation_closure_file_sha256'])
    closure_identity = closure.get('closure_sha256')
    if closure_identity != digest({k: v for k, v in closure.items() if k != 'closure_sha256'}):
        raise ValueError('Embedded closure identity differs')
    if (plan['generation_closure_identity_sha256'] != closure_identity or
            plan['generation_closure_sha256'] != closure_identity or
            closure.get('declaration_sha256') != plan['declaration_sha256'] or
            closure.get('experiment_id') != d['experiment_id'] or closure.get('task_ids') != tids or
            closure.get('conditions') != CONDITIONS or closure.get('task_count') != len(tids) or
            closure.get('answer_count') != plan['expected_answers'] or
            closure.get('all_primary_generation_complete') is not True or
            closure.get('scoring_requires_this_complete_seal') is not True):
        raise ValueError('Whole-primary generation closure differs')
    files = {}
    for item in closure['files']:
        path = scoped(root, item['path'])
        if item['path'] in files or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Closed generation artifact changed or duplicated')
        files[item['path']] = item
    expected = [(t, c, s) for t in tids for c in CONDITIONS for s in range(3)]
    sealed = {}
    for answer in closure['answers']:
        contract = answer['contract']; key = tuple(contract[n] for n in ('task_id', 'condition', 'seed_index'))
        if key in sealed or answer['contract_sha256'] != digest(contract):
            raise ValueError('Duplicate or changed sealed draw contract')
        if answer['path'] not in files or answer['answer_sha256'] != files[answer['path']]['sha256']:
            raise ValueError('Answer is not in the whole-generation file inventory')
        sealed[key] = answer
    if list(sealed) != expected:
        raise ValueError('Sealed task/condition/seed population or frozen order differs')
    manifest_path = scoped(root, plan['manifest_path']); manifest = read(manifest_path)
    if (manifest.get('experiment_id') != d['experiment_id'] or
            manifest.get('declaration_sha256') != plan['declaration_sha256'] or
            manifest.get('scoring_plan_path') != str(plan_path.relative_to(repo)) or
            manifest.get('scoring_plan_sha256') != plan_sha or
            manifest.get('scoring_plan_identity_sha256') != plan_identity or
            manifest.get('generation_closure_identity_sha256') != closure_identity or
            manifest.get('generation_closure_sha256') != closure_identity or
            manifest.get('generation_closure_file_sha256') != plan['generation_closure_file_sha256'] or
            manifest.get('scorer_identity_sha256') != digest(scorer) or
            manifest.get('cohort') != 'primary' or manifest.get('task_ids') != tids or
            manifest.get('conditions') != CONDITIONS or manifest.get('expected_answers') != len(expected) or
            manifest.get('committed_answers') != len(expected) or
            manifest.get('hidden_tests_loaded_after_primary_generation') is not True or
            manifest.get('raw_answers_unchanged') is not True):
        raise ValueError('Score manifest does not bind the frozen plan, declaration and closure')
    actual = [(r['task_id'], r['condition'], r['seed_index']) for r in manifest['files']]
    if actual != expected: raise ValueError('Scored population or task/condition/seed order differs')
    if len({r['path'] for r in manifest['files']}) != len(expected):
        raise ValueError('Duplicate score receipt path')
    if [(r['task_id'], r['condition'], r['seed_index']) for r in plan['records']] != expected:
        raise ValueError('Scoring plan task/condition/seed order differs')
    histories = {}; history_rows = []; history_overhead = {}; overhead_rows = []
    for tid in tids:
        for role, folder in [('source', 'large_history'), ('small', 'small_history')]:
            relative = 'primary/tasks/' + tid.replace('/', '__') + '/' + folder + '/source_history.json'
            if relative not in files: raise ValueError('Required original history is absent from the closure')
            history = read(root / relative); histories[tid, role] = history
            if history['task_id'] != tid: raise ValueError('History task identity differs')
            overhead = overhead_receipt(root, files, Path(relative).parent, history)
            history_overhead[tid, role] = overhead
            if overhead:
                overhead_rows.append({'task_id': tid, 'role': role, 'condition': None, 'seed_index': None, **overhead})
            history_rows.append({'task_id': tid, 'role': role, 'reasoning_tokens': len(history['reasoning_ids']),
                'natural_boundary': history['natural_boundary'], 'reasoning_capped': history['reasoning_capped'],
                'early_eos': history['early_eos'], 'reasoning_seconds': history.get('reasoning_seconds'),
                'timing_complete': history.get('sampler_timing', {}).get('complete') is True,
                'history_path': relative, 'history_sha256': files[relative]['sha256']})
    rows = []
    for item, key, planned in zip(manifest['files'], expected, plan['records']):
        answer = sealed[key]; contract = answer['contract']; tid, condition, seed_index = key
        seed = seeds['tasks'][tid]['answers'][seed_index]
        if (seed['seed_index'] != seed_index or contract['answer_seed'] != seed['answer_seed'] or
                contract['stream'] != seed['stream'] or contract['cohort'] != 'primary' or
                contract['declaration_sha256'] != plan['declaration_sha256'] or
                item['answer_path'] != answer['path'] or item['answer_sha256'] != answer['answer_sha256']):
            raise ValueError('Frozen seed, cohort, raw-answer or declaration binding differs')
        raw = read(root / answer['path'])
        if any(raw.get(k) != v for k, v in contract.items()): raise ValueError('Raw answer contract differs')
        scored = checked(root, item['path'], item['sha256']); binding = scored['binding']
        code = extract(raw['answer_text']); code_sha = hashlib.sha256(code.encode()).hexdigest()
        wanted = {'scoring_plan_identity_sha256': plan_identity, 'scoring_plan_sha256': plan_sha,
            'declaration_sha256': plan['declaration_sha256'], 'generation_closure_identity_sha256': closure_identity,
            'generation_closure_file_sha256': plan['generation_closure_file_sha256'],
            'scorer_identity_sha256': digest(scorer), 'record_id': answer['contract_sha256'],
            'answer_sha256': answer['answer_sha256'], 'contract_sha256': answer['contract_sha256'], 'code_sha256': code_sha}
        if (planned['contract'] != contract or planned['record_id'] != answer['contract_sha256'] or
                planned['answer_path'] != answer['path'] or planned['answer_sha256'] != answer['answer_sha256'] or
                planned['code_sha256'] != code_sha):
            raise ValueError('Scoring plan record differs from the sealed draw')
        if (any(binding.get(k) != v for k, v in wanted.items()) or scored.get('contract') != contract or
                scored.get('answer_path') != answer['path'] or scored.get('answer_sha256') != answer['answer_sha256'] or
                scored.get('code') != code or scored.get('hidden_tests_loaded_after_primary_generation') is not True):
            raise ValueError('Score receipt/extraction is not bound to the complete primary population')
        for field in ('task_id', 'condition', 'seed_index', 'answer_seed'):
            if scored.get(field) != contract[field] or item.get(field) != contract[field]:
                raise ValueError('Score receipt task/condition/seed identity differs')
        score = scored['score_v2']
        if (type(score.get('missing')) is not bool or score['missing'] is not (score.get('passed') is None) or
                (score.get('passed') is not None and type(score['passed']) is not bool) or
                item['passed'] != score['passed'] or item['missing'] != score['missing'] or
                score.get('scorer_identity') != scorer):
            raise ValueError('Score outcome or scorer identity differs')
        if score['missing'] and score.get('category') != 'infrastructure_failure':
            raise ValueError('Only explicit infrastructure outcomes may be missing')
        history = None if condition == 'P' else histories[tid, 'small' if condition == 'B' else 'source']
        answer_overhead = None
        if raw.get('sampler_timing', {}).get('overhead') is not None:
            sampler_folder = Path(answer['path']).parent / 'sampler'
            record_path = str(sampler_folder / 'answer_record.json')
            if record_path not in files: raise ValueError('Durable answer record is not generation-sealed')
            answer_overhead = overhead_receipt(root, files, sampler_folder, read(root / record_path))
            overhead_rows.append({'task_id': tid, 'role': 'answer', 'condition': condition,
                                  'seed_index': seed_index, **answer_overhead})
        reason_overhead = None if condition == 'P' else history_overhead[tid, 'small' if condition == 'B' else 'source']
        ids = raw['answer_ids']
        if not ids or type(raw['answer_ended_eos']) is not bool or type(raw['answer_capped']) is not bool or raw['answer_ended_eos'] == raw['answer_capped']:
            raise ValueError('Answer token/stopping diagnostics differ')
        row = {'task_id': tid, 'condition': condition, 'seed_index': seed_index, 'answer_seed': seed['answer_seed'],
            'passed': score['passed'], 'missing': score['missing'], 'category': score['category'],
            **interface_diagnostics(code, visible[tid]['prompt']), 'answer_tokens': len(ids),
            'answer_ended_eos': raw['answer_ended_eos'], 'answer_capped': raw['answer_capped'],
            'repeated_fourgram_fraction': repeated_fourgrams(ids), **timing_record(raw, history, score),
            **overhead_for_draw(answer_overhead, reason_overhead, condition == 'P'),
            'answer_path': answer['path'], 'answer_sha256': answer['answer_sha256'],
            'score_path': item['path'], 'score_sha256': item['sha256']}
        rows.append(row)
    if manifest.get('missing_answers') != sum(r['missing'] for r in rows):
        raise ValueError('Manifest missing-outcome count differs')
    setup_rows = []
    for relative, entry in files.items():
        path = Path(relative)
        if path.name != 'model_setup.json' and path.parent.name != 'mapper_loads': continue
        value = read(root / relative)
        if value.get('declaration_sha256') != plan['declaration_sha256'] or value.get('charged_to_single_output_inference') is not False:
            raise ValueError('Setup measurement is not bound separately from inference')
        worker = path.name == 'model_setup.json'
        loading = value.get('model_loading_seconds', {}) if worker else {}
        setup_rows.append({'kind': 'worker_models' if worker else 'mapper_checkpoint',
            'path': relative, 'sha256': entry['sha256'], 'task_id': value.get('task_id'),
            'condition': value.get('condition'),
            'source_loading_seconds': seconds(loading.get('source')),
            'receiver_loading_seconds': seconds(loading.get('receiver')),
            'mapper_initialization_seconds': seconds(value.get('mapper_initialization_seconds')),
            'checkpoint_load_seconds': seconds(value.get('checkpoint_load_seconds')),
            'started_epoch': value.get('started_epoch'), 'completed_epoch': value.get('completed_epoch')})
    return {'repo': repo, 'root': root, 'plan': plan, 'declaration': d, 'analysis': analysis,
        'rows': rows, 'history_rows': history_rows, 'setup_rows': setup_rows,
        'overhead_rows': overhead_rows, 'bindings': {
            'scoring_plan_path': str(plan_path.relative_to(repo)), 'scoring_plan_sha256': plan_sha,
            'scoring_plan_identity_sha256': plan_identity, 'declaration_sha256': plan['declaration_sha256'],
            'analysis_plan_sha256': d['inputs'][d['analysis_path']], 'scored_manifest_sha256': sha(manifest_path),
            'generation_closure_identity_sha256': closure_identity,
            'generation_closure_file_sha256': plan['generation_closure_file_sha256'],
            'scorer_identity_sha256': digest(scorer)}}


def report(plan_path, *, repo=ROOT, output=None):
    loaded = load_records(plan_path, repo)
    d, rows = loaded['declaration'], loaded['rows']
    summary, task_rows, gains, indices = cluster_statistics(rows, d['task_ids'], loaded['analysis'])
    summary.update(experiment_id=d['experiment_id'], cohort='primary', training_seed=d['primary_training_seed'],
        answers=len(rows), missing_draws=sum(r['missing'] for r in rows), bindings=loaded['bindings'],
        identities={'source_commit': d['source_commit'], 'publication_commit': d['publication_commit'],
            'primary_checkpoints': {arm: {key: cp[key] for key in ['step', 'mapper_sha256']}
                for arm, cp in d['primary_checkpoints'].items()}},
        diagnostics=diagnostics_summary(rows), history_diagnostics=loaded['history_rows'],
        setup_measurements={'scope': 'Each unique closure-bound completed setup receipt, counted once. Unclosed/interrupted worker attempts are not assumed free or included by inference.',
            'records': loaded['setup_rows'], 'not_added_to_single_output_inference_seconds': True,
            'worker_setup_receipts': sum(r['kind'] == 'worker_models' for r in loaded['setup_rows']),
            'mapper_checkpoint_receipts': sum(r['kind'] == 'mapper_checkpoint' for r in loaded['setup_rows'])},
        sampler_overhead_measurements={'scope': 'Each immutable generation-sealed completion sidecar once; source histories are not counted again for each reused condition or seed.',
            'records': loaded['overhead_rows'], 'not_subtracted_from_active_seconds': True,
            'incomplete_scope_receipts': sum(r['scope_complete'] is False for r in loaded['overhead_rows'])},
        separately_unmeasured_costs=['completion-sidecar and attempt-receipt persistence tails',
            'unrecorded interrupted setup', 'fleet_elapsed', 'rental_cost', 'cumulative_spending'],
        timing_estimate_not_measured_end_to_end_latency=True, secondary_report_not_required=True)
    default_relative = str((loaded['root'] / 'primary/report').relative_to(loaded['repo']))
    out = scoped(loaded['repo'], output if output is not None else default_relative)
    # Restrict writes to this report namespace; preserved pilot/v2/scorer reports
    # and immutable generation/scoring inputs cannot be overwritten by CLI paths.
    if not out.resolve().is_relative_to((loaded['root'] / 'primary/report').resolve()):
        raise ValueError('Output must remain within this primary report namespace')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'bootstrap_indices.npz', indices=indices)
    summary['bootstrap_archive_sha256'] = sha(out / 'bootstrap_indices.npz')
    write(out / 'summary.json', summary)
    csv_write(out / 'per_task_seed.csv', rows); csv_write(out / 'per_task.csv', task_rows)
    csv_write(out / 'task_gains_losses.csv', gains); csv_write(out / 'histories.csv', loaded['history_rows'])
    if loaded['setup_rows']: csv_write(out / 'setup_measurements.csv', loaded['setup_rows'])
    if loaded['overhead_rows']: csv_write(out / 'sampler_overhead.csv', loaded['overhead_rows'])
    condition_rows = [{'condition': condition, **{k: value[k] for k in [
        'draws', 'tasks', 'passed_draws', 'missing_draws', 'point_estimate_complete', 'pass_rate']},
        'ci95_lower': value['ci95'][0] if value['ci95'] else None,
        'ci95_upper': value['ci95'][1] if value['ci95'] else None,
        'possible_mean_lower': value['possible_mean_bounds'][0], 'possible_mean_upper': value['possible_mean_bounds'][1]}
        for condition, value in summary['conditions'].items()]
    csv_write(out / 'conditions.csv', condition_rows)
    (out / 'REPORT.md').write_text(render(summary))
    inventory = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(out.iterdir()) if p.is_file() and p.name != 'report_manifest.json']
    write(out / 'report_manifest.json', {'experiment_id': d['experiment_id'], 'cohort': 'primary',
        'source_bindings': loaded['bindings'], 'report_script_sha256': sha(Path(__file__)),
        'generation_or_candidate_execution_performed': False, 'files': inventory})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = report(args.plan, output=args.output)
    print(json.dumps({'experiment_id': result['experiment_id'], 'answers': result['answers'],
                      'missing': result['missing_draws'], 'primary': result['contrasts']['ROTATING_M-FIXED_M']}, indent=2))
