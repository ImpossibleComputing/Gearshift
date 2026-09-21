#!/usr/bin/env python3
"""Regenerate coverage-v2 inference from compact records, without weights/tests."""
import argparse
import ast
import collections
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from gearshift.coding_control import digest, sha, write
from gearshift.coding_coverage import DECLARATION, PUBLICATION
from gearshift.coding_sandbox import extract
from scripts.coding_coverage_v2_evaluate import generation_closure, validate_job, draw_contract, validate_raw

ROOT = Path(__file__).resolve().parents[1]
TIMINGS = ('answer_seconds', 'native_prefill_seconds', 'mapping_seconds', 'splice_seconds',
           'historical_receiver_prefill_tokens', 'source_inclusive_estimate_seconds',
           'source_cache_reconstruction_seconds', 'cache_clone_seconds')


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    if rows:
        with Path(path).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def repeated_fourgrams(ids):
    grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
    return (len(grams)-len(set(grams)))/len(grams) if grams else 0.


def interface(code, prompt):
    wanted = re.findall(r'^\s*(?:async\s+)?def (\w+)\(', prompt, re.M)
    expects_solution = bool(re.search(r'\bclass\s+Solution\b', prompt))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, bool(wanted) or expects_solution
    if expects_solution:
        solutions = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'Solution']
        functions = {f.name for n in solutions for f in n.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
        missing = not solutions or not set(wanted) <= functions
    else:
        functions = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        missing = bool(wanted) and not set(wanted) <= functions
    return True, missing


def label(arm, step, form):
    if form in ('D', 'P'):
        return form
    return 'START_' + form if arm == 'START' else f'{arm}_{step:04d}_{form}'


def cluster_statistics(rows, task_ids, labels, contrasts, bootstrap_seed=20260918):
    """One joint resample matrix for every condition, curve and contrast."""
    lookup = {}
    for row in rows:
        key = row['task_id'], row['condition'], row['seed_index']
        if key in lookup:
            raise ValueError('Duplicate task/condition/seed draw')
        lookup[key] = row
    expected = {(t, a, s) for t in task_ids for a in labels for s in range(3)}
    if set(lookup) != expected:
        raise ValueError('Incomplete or unexpected matched validation population')
    means = np.asarray([[sum(lookup[t, a, s]['passed'] for s in range(3))/3 for a in labels] for t in task_ids])
    indices = np.random.default_rng(bootstrap_seed).integers(0, len(task_ids), size=(10000, len(task_ids)))
    boot = means[indices].mean(axis=1)
    summary = {'task_clusters': len(task_ids), 'seeds_per_task': 3, 'task_ids': task_ids, 'conditions': {}, 'contrasts': {},
               'bootstrap_seed': bootstrap_seed, 'bootstrap_indices_sha256': digest(indices.tolist()),
               'method': 'Equal task means over three predeclared draws. 10000 identical joint paired task-cluster resamples for all conditions/checkpoints; percentile 95% intervals with linear interpolation. Exploratory, unadjusted. No best-of-three. An interval containing zero is not equivalence.'}
    task_rows = [{'task_id': tid, **{a: float(means[i, j]) for j, a in enumerate(labels)}} for i, tid in enumerate(task_ids)]
    for j, name in enumerate(labels):
        rr = [r for r in rows if r['condition'] == name]
        summary['conditions'][name] = {'pass_rate': float(means[:, j].mean()), 'ci95': np.quantile(boot[:, j], [.025, .975]).tolist(),
            'passed_draws': sum(r['passed'] for r in rr), 'draws': len(rr), 'task_clusters': len(task_ids),
            'failure_categories': dict(collections.Counter(r['category'] for r in rr)),
            'syntax_failures': sum(not r['syntax_valid'] for r in rr), 'entrypoint_failures': sum(r['missing_requested_entrypoint'] for r in rr),
            'EOS': sum(r['EOS'] for r in rr), 'caps': sum(r['capped'] for r in rr),
            'timing_unavailable_draws': sum(r['answer_seconds'] is None for r in rr),
            'means': {k: statistics.fmean(v) if (v := [r[k] for r in rr if r[k] is not None]) else None
                      for k in ('answer_tokens', 'repetition_4gram_fraction', *TIMINGS)}}
    for a, b in contrasts:
        ia, ib = labels.index(a), labels.index(b); delta = means[:, ia]-means[:, ib]
        summary['contrasts'][a+'-'+b] = {'difference': float(delta.mean()),
            'ci95': np.quantile(boot[:, ia]-boot[:, ib], [.025, .975]).tolist(),
            'tasks_positive': int((delta > 0).sum()), 'tasks_negative': int((delta < 0).sum()), 'tasks_tied': int((delta == 0).sum()),
            'per_task_difference': delta.tolist()}
    return summary, task_rows, indices.tolist()


def _coverage_records(training_roots, repo):
    rows = []; task_positions = []; training = {}; timings = {}
    for arm in ('FIXED', 'ROTATING'):
        if arm not in training_roots:
            continue
        root = repo/Path(training_roots[arm])
        complete = read(root/'training_complete.json') if (root/'training_complete.json').exists() else None
        logs = read(root/'training_steps.json')
        if any(r['step'] != i+1 or r['arm'] != arm or r['gradient_predictions'] != 32 or
               r['normalization_per_arm'] != 32 or not r['cache_gradients_finite_nonzero'] for i, r in enumerate(logs)):
            raise ValueError('Training log has a gap, changed normalization, or failed gradients')
        training[arm] = {'completion': complete, 'logged_updates': len(logs), 'identity': read(root/'training_identity.json'),
                         'resume_verification': read(root/'resume_preflight.json') if (root/'resume_preflight.json').exists() else None,
                         'checkpoint_manifest': read(root/'checkpoint_manifest.json')}
        if not training[arm]['resume_verification'] or not training[arm]['resume_verification']['passed']:
            raise ValueError('Missing passed actual-model checkpoint resume preflight')
        if complete and (complete['completed_updates'] != len(logs) or complete['scored_positions'] != 32*len(logs)):
            raise ValueError('Training completion receipt differs from the update log')
        timings[arm] = {'logged_updates': len(logs), 'gradient_predictions': sum(r['gradient_predictions'] for r in logs),
                       'all_cache_gradients_finite_nonzero': all(r['cache_gradients_finite_nonzero'] for r in logs),
                       **{k: sum(r[k] for r in logs) for k in ('wall_seconds', 'cache_preparation_seconds', 'continuation_forward_backward_seconds', 'optimizer_seconds')}}
        for path in sorted((root/'coverage').glob('step_*.json')):
            r = read(path)
            if r['scored_positions'] != 32*r['steps']:
                raise ValueError('Coverage totals differ from 32 scored positions per update')
            categories = collections.defaultdict(lambda: {'scored': 0, 'unique': 0})
            for tid, task in r['tasks'].items():
                for category, counts in task['by_category'].items():
                    categories[category]['scored'] += counts['scored']; categories[category]['unique'] += counts['unique']
                for position, count in task['positions'].items():
                    task_positions.append({'arm': arm, 'step': r['steps'], 'task_id': tid, 'answer_position': int(position), 'scored_count': count})
            rows.append({'arm': arm, 'step': r['steps'], 'scored_positions': r['scored_positions'],
                         'unique_task_positions': r['unique_task_positions'], 'by_category': dict(categories),
                         'source_path': str(path.relative_to(repo)), 'source_sha256': sha(path)})
    return training, timings, rows, task_positions


def render(out, summary, kl_rows, coverage, primary_step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.hashsalt': 'gearshift-coverage-v2'})
    def save(fig, name):
        fig.tight_layout(); fig.savefig(out/(name+'.png'), dpi=170); fig.savefig(out/(name+'.svg'), metadata={'Date': None}); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, form in zip(axes, ('M', 'H')):
        for arm, color in (('FIXED', '#405bb7'), ('ROTATING', '#a33c32')):
            curve = [(0, summary['conditions']['START_'+form])]
            curve += sorted((int(name.split('_')[1]), r) for name, r in summary['conditions'].items() if name.startswith(arm+'_') and name.endswith('_'+form))
            y = np.array([r['pass_rate']*100 for _, r in curve])
            lo = np.array([r['ci95'][0]*100 for _, r in curve]); hi = np.array([r['ci95'][1]*100 for _, r in curve])
            ax.plot([s for s, _ in curve], y, marker='o', label=arm, color=color)
            ax.fill_between([s for s, _ in curve], lo, hi, alpha=.10, color=color)
        for name, color in (('D', '#397052'), ('P', '#9c8251')):
            ax.axhline(summary['conditions'][name]['pass_rate']*100, label=name, color=color, linestyle='--')
        ax.set(title='Pure mapped cache (M)' if form == 'M' else 'Native prompt + mapped reasoning (H)',
               xlabel='Additional updates per arm', ylabel='Mean task success (%)', ylim=(0, 105)); ax.legend()
    save(fig, 'validation_behavior_curves')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, panel in zip(axes, ('legacy', 'broad')):
        for arm, color in (('FIXED', '#405bb7'), ('ROTATING', '#a33c32')):
            by_step = collections.defaultdict(list)
            for row in kl_rows:
                if row['panel'] == panel and row['arm'] in (arm, 'START'):
                    by_step[row['step']].append(row['mean_kl'])
            curve = sorted((step, statistics.fmean(values)) for step, values in by_step.items())
            ax.plot([s for s, _ in curve], [v for _, v in curve], marker='o', label=arm, color=color)
        ax.set(title='Legacy windows' if panel == 'legacy' else 'Fixed broad panel', xlabel='Additional updates per arm', ylabel='Mean task KL'); ax.legend()
    save(fig, 'validation_kl_curves')
    if coverage:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for arm, color in (('FIXED', '#405bb7'), ('ROTATING', '#a33c32')):
            rr = sorted([r for r in coverage if r['arm'] == arm], key=lambda r: r['step'])
            axes[0].plot([r['step'] for r in rr], [r['unique_task_positions'] for r in rr], marker='o', label=arm, color=color)
        axes[0].set(xlabel='Additional updates per arm', ylabel='Unique task / answer-position pairs', title='Direct supervision coverage'); axes[0].legend()
        categories = ['prose', 'signature', 'code_body', 'return_statement', 'EOS']
        x = np.arange(len(categories))
        for arm, shift, color in (('FIXED', -.18, '#405bb7'), ('ROTATING', .18, '#a33c32')):
            matches = [r for r in coverage if r['arm'] == arm and r['step'] == primary_step]
            if matches:
                axes[1].bar(x+shift, [matches[0]['by_category'].get(k, {}).get('scored', 0) for k in categories], width=.36, label=arm, color=color)
        axes[1].set_xticks(x, [c.replace('_', ' ') for c in categories], rotation=25)
        axes[1].set(ylabel='Scored positions', title=f'Primary checkpoint exposure ({primary_step} updates)'); axes[1].legend()
        save(fig, 'training_coverage')


def report(result_root, plan, *, repo_root=ROOT, output=None, training_roots=None, primary_step=1024, engineering_failure=None):
    repo = Path(repo_root).resolve(); root = repo/Path(result_root)
    if isinstance(plan, (str, Path)):
        plan = read(repo/Path(plan))
    if primary_step != 1024 and not engineering_failure:
        raise ValueError('Only an explicit engineering-failure receipt can replace the primary 1024 checkpoint')
    out = Path(output) if output is not None else root/'report'; out.mkdir(parents=True, exist_ok=True)
    declaration = plan.get('declaration_path', DECLARATION); d = read(repo/declaration)
    if plan.get('declaration_sha256') and sha(repo/declaration) != plan['declaration_sha256']:
        raise ValueError('Frozen declaration differs')
    if d['publication_commit'] != PUBLICATION:
        raise ValueError('Frozen publication identity differs')
    for key in ('panels', 'answer_seeds'):
        if sha(repo/d[key+'_path']) != d[key+'_sha256']:
            raise ValueError('Frozen analysis inputs changed')
    seeds = read(repo/d['answer_seeds_path'])['validation']; panels = read(repo/d['panels_path'])
    visible = {r['task_id']: r for r in read(repo/'data/coding_pilot_v1/visible/coverage_generalization.json')}
    closure = generation_closure(root, plan)
    if read(root/'generation_closure.json') != closure:
        raise ValueError('Global generation closure differs')
    manifest = read(root/'scored_answer_manifest.json')
    if manifest['generation_closure_sha256'] != digest(closure) or manifest['hidden_tests_loaded_after_all_generation'] is not True:
        raise ValueError('Scoring is not bound to whole-plan generation closure')
    jobs = {j['job_id']: j for j in plan['jobs']}
    checkpoints = {}; free_steps = set(); kl_population = collections.defaultdict(set)
    for job in jobs.values():
        validate_job(job, d)
        key = job['arm'], job['step']; checkpoint = job['checkpoint']['sha256']
        if key in checkpoints and checkpoints[key] != checkpoint:
            raise ValueError('Task shards evaluated different mapper checkpoints')
        checkpoints[key] = checkpoint
        if job['arm'] != 'START' and job.get('forms', ['M', 'H']):
            free_steps.add(job['step'])
    if not {128, 512, primary_step} <= free_steps and not engineering_failure:
        raise ValueError('Mandatory free-running learning curve checkpoints are incomplete')
    rows = []; raw_paths = set()
    for item in manifest['files']:
        score_path = root/item['path']; path = root/item['answer_path']; r = read(path); scored = read(score_path)
        if sha(path) != item['answer_sha256'] or sha(score_path) != item['sha256']:
            raise ValueError('Scored answer evidence changed')
        if item['answer_path'] in raw_paths:
            raise ValueError('Duplicate score receipt')
        raw_paths.add(item['answer_path'])
        if (scored['answer_sha256'] != item['answer_sha256'] or scored['generation_closure_sha256'] != digest(closure) or
                scored['hidden_tests_loaded_after_all_generation'] is not True or scored['code'] != extract(r['answer_text'])):
            raise ValueError('Scorer receipt/extraction changed')
        job = jobs[Path(item['answer_path']).parts[1]]; tid = r['task_id']; folder = path.parents[2]
        seed = seeds[tid][r['seed_index']]
        validate_raw(r, draw_contract(job, tid, r['condition'], r['form'], seed, sha(folder/'source_history.json')))
        syntax, missing = interface(scored['code'], visible[tid]['prompt'])
        row = {'task_id': tid, 'condition': label(job['arm'], job['step'], r['form']), 'arm': job['arm'],
               'step': job['step'], 'form': r['form'], 'seed_index': r['seed_index'], 'answer_seed': r['answer_seed'],
               'passed': bool(scored['score']['passed']), 'category': scored['score']['category'], 'syntax_valid': syntax,
               'missing_requested_entrypoint': missing, 'answer_tokens': len(r['answer_ids']), 'EOS': r['answer_ended_eos'],
               'capped': r['answer_capped'], 'repetition_4gram_fraction': repeated_fourgrams(r['answer_ids']),
               **{k: r.get(k, 0.) for k in TIMINGS}, 'raw_path': str(path.relative_to(repo)), 'raw_sha256': item['answer_sha256'],
               'score_path': str(score_path.relative_to(repo)), 'score_sha256': item['sha256'], 'attempt_id': r['attempt_id']}
        rows.append(row)
    if raw_paths != {r['path'] for r in closure['files']}:
        raise ValueError('Scored manifest omits or adds generation records')
    labels = ['START_M', 'START_H'] + [label(a, s, f) for s in sorted(free_steps) for a in ('FIXED', 'ROTATING') for f in ('M', 'H')] + ['D', 'P']
    contrasts = [('START_H', 'START_M'), ('START_H', 'D'), ('START_H', 'P')]
    for step in sorted(free_steps):
        fm, fh, rm, rh = [label(a, step, f) for a, f in [('FIXED','M'), ('FIXED','H'), ('ROTATING','M'), ('ROTATING','H')]]
        contrasts += [(rm, fm), (rh, fh), (fm, 'START_M'), (rm, 'START_M'), (fh, 'START_H'), (rh, 'START_H'), (fh, fm), (rh, rm)]
        contrasts += [(h, control) for h in (fh, rh) for control in ('D', 'P')]
    summary, task_rows, indices = cluster_statistics(rows, d['validation_task_ids'], labels, contrasts, d['analysis']['bootstrap_seed'])
    kl_rows = []
    for item in closure['kl_files']:
        row = read(root/item['path']); job = jobs[Path(item['path']).parts[1]]
        if row['job_identity_sha256'] != digest(job) or row['checkpoint_sha256'] != job['checkpoint']['sha256']:
            raise ValueError('KL checkpoint identity differs')
        key = row['arm'], row['step']
        if row['task_id'] in kl_population[key]:
            raise ValueError('Duplicate validation KL task')
        kl_population[key].add(row['task_id'])
        for panel in ('legacy', 'broad'):
            r = row[panel]
            if r['positions'] != panels[row['task_id']][panel] or len(r['positions']) != len(r['per_position_kl']) or not all(math.isfinite(v) for v in r['per_position_kl']):
                raise ValueError('KL does not use the exact frozen panel')
            if not math.isclose(r['mean_kl'], statistics.fmean(r['per_position_kl']), rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError('KL mean differs from raw per-position values')
            kl_rows.append({'arm': row['arm'], 'step': row['step'], 'task_id': row['task_id'], 'panel': panel,
                            'mean_kl': r['mean_kl'], 'predictions': len(r['positions'])})
    expected_kl = {('START', 0)} | {(a, s) for a in ('FIXED','ROTATING') for s in (128,256,512,768,1024) if s <= primary_step}
    if set(kl_population) != expected_kl or any(v != set(d['validation_task_ids']) for v in kl_population.values()):
        raise ValueError('Both KL panels must cover all 21 histories at every declared common checkpoint')
    training, timing, coverage, positions = _coverage_records(training_roots or plan.get('training_roots', {}), repo)
    full = all((training.get(a, {}).get('completion') or {}).get('full_target_completed') for a in ('FIXED','ROTATING'))
    if primary_step == 1024 and not full:
        raise ValueError('Primary1024 report requires both full clean training completion receipts')
    primary_key = label('ROTATING',primary_step,'M')+'-'+label('FIXED',primary_step,'M')
    primary = summary['contrasts'][primary_key]
    comparison = {'primary_step': primary_step, 'primary_contrast': primary_key, 'primary': primary,
                  'full_1024_per_arm_completed': full, 'engineering_failure': engineering_failure,
                  'checkpoint_selection': 'Predeclared 1024, or last common durable checkpoint solely upon documented engineering failure.',
                  'prior_four_case_diagnostic': {'START_M': .4, 'SEEN_FIT_M': .75, 'D': 1., 'task_clusters': 4, 'seeds_per_task': 5,
                      'new_training_or_draws': False, 'interpretation': 'Prior seen-training-case evidence only, not validation or new-v2 evidence.'},
                  'validation': summary, 'training': training, 'training_timing': timing, 'coverage': coverage,
                  'experiment_id': plan['experiment_id'], 'plan_sha256': digest(plan), 'fresh_draws': len(rows),
                  'publication_commit': PUBLICATION, 'confirmation_used': False}
    write(out/'summary.json', comparison); write(out/'validation_summary.json', summary)
    write(out/'task_bootstrap_indices.json', indices); write(out/'training_coverage.json', coverage)
    csv_write(out/'validation_draws.csv', rows); csv_write(out/'validation_task_means.csv', task_rows)
    csv_write(out/'validation_kl_by_task.csv', kl_rows); csv_write(out/'training_answer_position_exposure.csv', positions)
    render(out, summary, kl_rows, coverage, primary_step)
    delta = primary['difference']*100; lo, hi = [100*x for x in primary['ci95']]
    conclusion = ('The exploratory paired interval is above zero, supporting better held-out free-running behavior for ROTATING on these 21 previously examined validation tasks.' if lo > 0 else
                  'The exploratory paired interval is below zero, supporting worse held-out free-running behavior for ROTATING on these 21 previously examined validation tasks.' if hi < 0 else
                  'The exploratory paired interval includes zero; this run does not establish that ROTATING improves over FIXED. It also does not establish equivalence.')
    lines = ['# Coverage generalization v2 results', '',
             f'At the common {primary_step}-update checkpoint, ROTATING minus FIXED in pure mapped-cache free-running code success is **{delta:+.1f} percentage points**, paired 95% interval **[{lo:+.1f}, {hi:+.1f}]**. '+conclusion, '',
             f'Full clean 1,024/1,024 training completed: **{str(full).lower()}**. Fresh scored validation answers: **{len(rows)}**. '
             'Every sampled answer used one of the exact three declared seeds per task. Checkpoints were immutable and evaluated asynchronously; sampled validation outcomes never selected the primary endpoint.', '',
             '## Scientific scope', '',
             'Both arms cleanly restart from the same original selected update-96 mapper with identical fresh AdamW, constant LR scheduler, RNG seed and task schedule. The exact 104 training histories, 32 scored positions per update, fixed and rotating selection recipes, frozen source/receiver model revisions, BF16 arithmetic, natural handoff, source-written teacher references, extraction and isolated hidden-test scoring are retained. Full prefixes and intervening continuation tokens are processed; free-running generation receives no teacher-answer prefix.', '',
             'The 21 validation tasks are held out from gradients but already informed earlier selection and analysis. They are exploratory validation, not untouched confirmation. The reserved 200-task and second-seed confirmation sets remain untouched. Bootstrap inference averages the three draws within task and jointly resamples 21 task IDs 10,000 times, using exactly the same resample indices for every condition and contrast. Intervals are unadjusted. 63 draws are not 63 independent tasks, and best-of-three is not pass@1.', '',
             '## Primary checkpoint results', '', '| Condition | Passed draws | Mean task success | 95% task interval |', '|---|---:|---:|---:|']
    primary_labels = ['START_M','START_H']+[label(a,primary_step,f) for a in ('FIXED','ROTATING') for f in ('M','H')]+['D','P']
    for name in primary_labels:
        r = summary['conditions'][name]
        lines.append(f'| {name} | {r["passed_draws"]}/{r["draws"]} | {r["pass_rate"]:.1%} | [{r["ci95"][0]:.1%}, {r["ci95"][1]:.1%}] |')
    lines += ['', 'M translates the complete historical cache. H preserves the receiver-native original prompt cache and translates the saved reasoning suffix. D natively prefills the exact full source history. P uses the unchanged prompt-only non-thinking template. START is generated fresh in this experiment and shared between the two identical initial mapper states.', '',
              '| Paired task contrast | Difference, pp | 95% interval, pp |','|---|---:|---:|']
    for name, r in summary['contrasts'].items():
        if f'_{primary_step:04d}_' in name or name.startswith('START_'):
            lines.append(f'| {name} | {r["difference"]*100:+.1f} | [{r["ci95"][0]*100:+.1f}, {r["ci95"][1]*100:+.1f}] |')
    lines += ['', '## Execution and controls', '',
              'Per-task/seed raw answers, token IDs, source trajectories and score receipts are immutable and hash-bound. Incomplete worker attempts are retained. A finished sampler transaction can be recovered from its durable tokens/RNG without another draw; unavailable timing is recorded explicitly. All planned generation closes before private tests are opened. Native/native splice controls, absolute positions, isolated cloned caches and exact frozen seeds are checked. No program repairs, function renaming, later-block rescue or favorable-seed selection occur.', '',
              'Training checkpoints contain mapper, optimizer, scheduler, update/schedule position, RNG and precision state with atomic manifests and verified reload. Resume checks and gradient logs are included with each arm. Durable RunPod jobs do not depend on a laptop/Studio heartbeat or a continuing SSH session; mirror connectivity is optional.', '',
              'Both KL panels are separate teacher-forced fidelity measures; neither establishes program correctness. Learning-curve tables and figures retain every available declared common checkpoint. Coverage JSON/CSV reports total and unique task/position exposure, prose, signature, code-body, return and EOS categories plus answer-position distributions. Categories are descriptive and did not choose training positions.', '',
              'The earlier four-case fit remains a side diagnostic: START_M 40%, fitted M 75%, native D 100% over five seeds per seen task. It was not retrained and is not evidence of unseen generalization.', '',
              '## Artifact and publication boundary', '',
              f'Experiment identity: `{plan["experiment_id"]}`. Frozen publication remains `gearshift-progress-01` at `{PUBLICATION}`. The first article and previous results are unchanged. No push or publication.', '',
              'This report regenerates from compact evidence without model weights, mapper weights, private tests or generated-code execution. Resource receipts, current Git identity, conservative cumulative cost, failures and external heavy-artifact retrieval inventory accompany the review bundle.']
    if engineering_failure:
        lines += ['', 'Engineering fallback: '+json.dumps(engineering_failure, sort_keys=True)]
    (out/'COVERAGE_GENERALIZATION_V2_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--result-root', required=True); parser.add_argument('--plan', required=True)
    parser.add_argument('--repo-root', default=str(ROOT)); parser.add_argument('--output'); parser.add_argument('--primary-step', type=int, default=1024)
    parser.add_argument('--engineering-failure'); args = parser.parse_args()
    report(args.result_root, args.plan, repo_root=args.repo_root, output=args.output, primary_step=args.primary_step,
           engineering_failure=read(args.engineering_failure) if args.engineering_failure else None)
