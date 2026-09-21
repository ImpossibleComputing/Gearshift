#!/usr/bin/env python3
"""Separate second-training-seed queue; the frozen primary runtime is imported unchanged.

Only public inputs are read. No controls, histories, answer draws, task IDs or
checkpoints are selected using scores. The primary declaration is never amended.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, digest, sha, write
from scripts import coding_confirmation_generate as primary
from scripts.coding_coverage_v2_worker import claim_job

CONDITIONS = ['FIXED_M', 'ROTATING_M', 'FIXED_H', 'ROTATING_H']
SEED = 20260919
START_SHA = '0b9700ffd9cb38c23bcfa1327181b32992b128c6b95078214328382f86b8d68f'
REPLICATION = 'configs/coding_pilot_v1/confirmation_01/replication_declaration.json'
IMPLEMENTATION = {
    'scripts/coding_confirmation_secondary_generate.py',
    'scripts/coding_confirmation_secondary_supervisor.py',
    'scripts/coding_confirmation_secondary_score.py',
    'scripts/coding_confirmation_secondary_report.py',
    'scripts/coding_confirmation_replication_supervisor.py',
    'scripts/coding_confirmation_replication_stage.py',
    'scripts/coding_confirmation_replication.py',
    'gearshift/coding_coverage_v2_lease.py',
}
COMPONENTS = ['mapper', 'optimizer', 'scheduler', 'update', 'schedule_position',
              'python_rng', 'numpy_rng', 'torch_cpu_rng', 'torch_cuda_rng',
              'scaler_explicitly_none_for_bf16', 'checkpoint_identity', 'training_log']


def analysis_binding(d):
    return {'primary_analysis_path': d['analysis_path'],
        'primary_analysis_sha256': d['inputs'][d['analysis_path']],
        'bootstrap': {'rng': 'numpy.random.Generator(numpy.random.PCG64(20260919))', 'resamples': 10000,
            'index_generation': 'integers(0, task_count, size=(10000, task_count))',
            'task_order': 'Same200-task frozen declaration order', 'quantile_method': 'linear',
            'quantiles': [.025, .975], 'identical_primary_index_matrix_required': True},
        'contrasts': [['ROTATING_M', 'FIXED_M'], ['ROTATING_H', 'FIXED_H'], ['ROTATING_H', 'ROTATING_M'],
                      ['ROTATING_H', 'D'], ['ROTATING_H', 'P'], ['ROTATING_H', 'B'], ['ROTATING_H', 'A']],
        'native_controls': 'Reuse original primary A/B/D/P raw answers and frozen scoring receipts; no new draws or rescoring.',
        'inference': 'Exploratory unadjusted95% task-paired intervals; the original primary inference family is unchanged.',
        'missingness': 'Use the unchanged frozen primary full-population conservative sensitivity policy.',
        'pool_or_choose_best_training_seed': False, 'primary_report_waits_for_replication': False}


def read(path): return json.loads(Path(path).read_text())


def source_manifest(repo, output):
    """Compact source-only git proof prepared locally, before remote staging."""
    repo = Path(repo).resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    for name in IMPLEMENTATION:
        if subprocess.check_output(['git', 'show', commit + ':' + name], cwd=repo) != (repo / name).read_bytes():
            raise ValueError('Commit the tested secondary adapter before freezing: ' + name)
    value = {'schema': 1, 'adapter_commit': commit, 'files': {p: sha(repo / p) for p in sorted(IMPLEMENTATION)}}
    bind(destination(repo, output), value)
    return value


def verify_source_manifest(repo, path, expected_sha256):
    file = primary.scoped_file(repo, path)
    if sha(file) != expected_sha256: raise ValueError('Pinned secondary source manifest hash differs')
    value = read(file)
    if (value.get('schema') != 1 or not re.fullmatch('[0-9a-f]{40}', value.get('adapter_commit', '')) or
            set(value.get('files', {})) != IMPLEMENTATION):
        raise ValueError('Secondary source manifest identity or coverage differs')
    for name, expected in value['files'].items():
        if sha(primary.scoped_file(repo, name)) != expected:
            raise ValueError('Staged secondary adapter bytes differ: ' + name)
    return value


def destination(root, relative):
    root = Path(root).resolve(); p = Path(relative)
    if p.is_absolute() or not p.parts or '..' in p.parts:
        raise ValueError('Output path must remain within its declared root')
    p = root / p
    for item in [p, *p.parents]:
        if item == root: break
        if item.is_symlink(): raise ValueError('Linked scientific output is not permitted')
    return p


def checkpoint(repo, d, arm, manifest_relative, *, verify_weights):
    """Validate pinned original96 provenance and the complete final training state."""
    repo = Path(repo); path = primary.scoped_file(repo, manifest_relative); m = read(path)
    rd_path = primary.scoped_file(repo, REPLICATION)
    if d['inputs'].get(REPLICATION) != sha(rd_path): raise ValueError('Replication declaration is not primary-bound')
    rd = read(rd_path); models = read(primary.scoped_file(repo, d['model_config_path']))['models']
    if (rd.get('training_seed') != SEED or rd.get('selected_checkpoint_sha256') != START_SHA or
            rd.get('target_additional_updates') != 1024 or len(rd.get('training_task_ids', [])) != 104 or
            rd.get('experiment_id') != d['experiment_id'] or
            d['inputs'].get(rd['schedules_path']) != rd['schedules_sha256'] or
            sha(primary.scoped_file(repo, rd['schedules_path'])) != rd['schedules_sha256']):
        raise ValueError('Frozen paired training recipe or schedules differ')
    identity = m.get('checkpoint_identity', {})
    expected = {'experiment_id': d['experiment_id'] + '/replication', 'arm': arm,
        'config_sha256': sha(rd_path), 'schedules_sha256': rd['schedules_sha256'],
        'corpus_sha256': rd['corpus_manifest_sha256'], 'selected_checkpoint_sha256': START_SHA,
        'models': models, 'configuration': {'optimizer': rd['optimizer'], 'dtype': 'bfloat16',
            'attention': 'sdpa', 'gradient_positions': 32, 'full_context': True,
            'scheduler': 'constant', 'training_seed': SEED, 'schedule_position_offset': 96,
            'replication_id': rd['replication_id']}}
    if (any(identity.get(k) != v for k, v in expected.items()) or
            not re.fullmatch('[0-9a-f]{40}', identity.get('code_commit', '')) or
            m.get('checkpoint_identity_sha256') != digest(identity) or m.get('arm') != arm or
            m.get('step') != 1024 or m.get('schedule_position') != 1024 or
            m.get('complete_resumable') is not True or m.get('verified_roundtrip') is not True or
            m.get('state_components') != COMPONENTS or set(m.get('files', {})) != {'mapper.pt', 'full.pt'} or
            path.parent != repo / rd['result_root'] / 'arms' / arm / 'checkpoints/step_1024'):
        raise ValueError('Secondary checkpoint provenance, endpoint or resumable state differs')
    for name, item in m['files'].items():
        if (type(item.get('bytes')) is not int or item['bytes'] <= 0 or
                not re.fullmatch('[0-9a-f]{64}', item.get('sha256', ''))):
            raise ValueError('Invalid checkpoint byte inventory')
        if verify_weights:
            p = primary.scoped_file(repo, str((path.parent / name).relative_to(repo)))
            if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
                raise ValueError('Secondary checkpoint bytes differ')
    return {'step': 1024, 'manifest_path': str(path.relative_to(repo)), 'manifest_sha256': sha(path),
        'mapper_path': str((path.parent / 'mapper.pt').relative_to(repo)),
        'mapper_sha256': m['files']['mapper.pt']['sha256'], 'mapper_bytes': m['files']['mapper.pt']['bytes'],
        'full_checkpoint_path': str((path.parent / 'full.pt').relative_to(repo)),
        'full_checkpoint_sha256': m['files']['full.pt']['sha256'], 'full_checkpoint_bytes': m['files']['full.pt']['bytes'],
        'checkpoint_identity': identity, 'actual_mapper_and_full_bytes_verified_at_freeze': True}


def prepare(repo, declaration_path, declaration_sha256, fixed_manifest, rotating_manifest, output,
            *, source_manifest_path=None, source_manifest_sha256=None):
    """Freeze public-only secondary bindings after both final checkpoints commit."""
    repo = Path(repo).resolve()
    d = primary.validate_declaration(repo, declaration_path, declaration_sha256)
    if (d['task_count'] != 200 or d.get('secondary_conditions') != CONDITIONS or
            d.get('secondary_answer_count') != 2400 or d.get('secondary_training_seed') != SEED):
        raise ValueError('Secondary requires the owner-approved200-task four-condition population')
    if (source_manifest_path is None) != (source_manifest_sha256 is None):
        raise ValueError('Source manifest path and caller-pinned hash are required together')
    if source_manifest_path is not None:
        source = verify_source_manifest(repo, source_manifest_path, source_manifest_sha256)
        commit = source['adapter_commit']
        source_proof = {'method': 'pinned_compact_source_manifest', 'path': source_manifest_path, 'sha256': source_manifest_sha256}
    else:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        for name in IMPLEMENTATION:
            if subprocess.check_output(['git', 'show', commit + ':' + name], cwd=repo) != (repo / name).read_bytes():
                raise ValueError('Commit the tested secondary adapter before freezing: ' + name)
        source_proof = {'method': 'local_git_commit'}
    checkpoints = {arm: checkpoint(repo, d, arm, manifest, verify_weights=True)
                   for arm, manifest in [('FIXED', fixed_manifest), ('ROTATING', rotating_manifest)]}
    if checkpoints['FIXED']['checkpoint_identity']['code_commit'] != checkpoints['ROTATING']['checkpoint_identity']['code_commit']:
        raise ValueError('Paired training source commits differ')
    value = {'schema': 1, 'status': 'FROZEN', 'experiment_id': d['experiment_id'], 'cohort': 'secondary',
        'declaration_path': declaration_path, 'declaration_sha256': declaration_sha256,
        'primary_source_commit': d['source_commit'], 'adapter_commit': commit,
        'adapter_source_proof': source_proof,
        'training_seed': SEED, 'task_ids': d['task_ids'], 'task_count': 200, 'conditions': list(CONDITIONS),
        'answer_count': 2400, 'answer_draws_per_task_condition': 3, 'endpoint': 1024,
        'checkpoints': checkpoints, 'implementation': {p: sha(repo / p) for p in sorted(IMPLEMENTATION)},
        'replication_analysis': analysis_binding(d),
        'source_history_policy': 'Reuse exact primary committed source history and native controls; no reasoning or baseline generation.',
        'primary_artifacts_modified': False, 'private_tests_loaded': False,
        'checkpoint_selection': 'Both predeclared final endpoints; no score-dependent choice.'}
    bind(destination(repo, output), value)
    return value


def validate_secondary(repo, plan, *, verify_weights=True):
    repo = Path(repo).resolve()
    d = primary.validate_declaration(repo, plan['declaration_path'], plan['declaration_sha256'])
    path = primary.scoped_file(repo, plan['secondary_declaration_path'])
    if sha(path) != plan['secondary_declaration_sha256']: raise ValueError('Secondary declaration hash differs')
    s = read(path)
    required = {'schema': 1, 'status': 'FROZEN', 'experiment_id': d['experiment_id'], 'cohort': 'secondary',
        'declaration_path': plan['declaration_path'], 'declaration_sha256': plan['declaration_sha256'],
        'primary_source_commit': d['source_commit'], 'training_seed': SEED, 'task_ids': d['task_ids'],
        'task_count': 200, 'conditions': CONDITIONS, 'answer_count': 2400,
        'answer_draws_per_task_condition': 3, 'endpoint': 1024,
        'replication_analysis': analysis_binding(d),
        'primary_artifacts_modified': False, 'private_tests_loaded': False}
    if (any(s.get(k) != v for k, v in required.items()) or d['task_count'] != 200 or
            d.get('secondary_conditions') != CONDITIONS or d.get('secondary_answer_count') != 2400 or
            d.get('secondary_training_seed') != SEED or
            plan.get('code_commit') != d['source_commit'] or plan.get('experiment_id') != d['experiment_id'] or
            not re.fullmatch('[0-9a-f]{40}', s.get('adapter_commit', '')) or
            set(s.get('implementation', {})) != IMPLEMENTATION or set(s.get('checkpoints', {})) != {'FIXED', 'ROTATING'}):
        raise ValueError('Frozen secondary identity or population differs')
    proof = s.get('adapter_source_proof', {})
    if proof.get('method') == 'pinned_compact_source_manifest':
        source = verify_source_manifest(repo, proof['path'], proof['sha256'])
        if source['adapter_commit'] != s['adapter_commit'] or source['files'] != s['implementation']:
            raise ValueError('Secondary adapter source proof differs from its frozen declaration')
    elif proof != {'method': 'local_git_commit'}:
        raise ValueError('Secondary adapter source proof is missing')
    for name, expected in s['implementation'].items():
        if sha(primary.scoped_file(repo, name)) != expected or sha(ROOT / name) != expected:
            raise ValueError('Secondary adapter implementation differs: ' + name)
    for arm, cp in s['checkpoints'].items():
        actual = checkpoint(repo, d, arm, cp['manifest_path'], verify_weights=verify_weights)
        if actual != cp: raise ValueError('Frozen secondary checkpoint inventory differs')
    if s['checkpoints']['FIXED']['checkpoint_identity']['code_commit'] != s['checkpoints']['ROTATING']['checkpoint_identity']['code_commit']:
        raise ValueError('Paired training source commits differ')
    return d, s


def public_context(repo, plan_path, plan_sha256, *, verify_weights=False):
    repo = Path(repo).resolve(); path = primary.scoped_file(repo, plan_path)
    if sha(path) != plan_sha256: raise ValueError('Secondary dispatch plan hash differs')
    plan = read(path); d, s = validate_secondary(repo, plan, verify_weights=verify_weights)
    rows = read(primary.scoped_file(repo, d['visible_path']))
    visible = {r['task_id']: r for r in rows if r['task_id'] in d['task_ids']}
    if len({r['task_id'] for r in rows}) != len(rows) or set(visible) != set(d['task_ids']):
        raise ValueError('Public task population differs')
    top = destination(repo, plan['result_root'])
    return {'repo_root': repo, 'root': top / 'secondary', 'top': top, 'plan': plan,
        'declaration': d, 'declaration_sha256': plan['declaration_sha256'],
        'secondary_declaration': s, 'secondary_declaration_sha256': plan['secondary_declaration_sha256'],
        'secondary_checkpoints': s['checkpoints'], 'seeds': read(repo / d['seeds_path']), 'visible': visible}


def task_folder(c, tid): return c['top'] / 'secondary/tasks' / tid.replace('/', '__')
def job_path(c, tid): return c['top'] / 'secondary/jobs' / ('receiver_' + tid.replace('/', '__')) / 'complete.json'


def verify_job(c, tid):
    path = job_path(c, tid)
    if not path.exists(): return None
    row = read(path)
    expected = {'experiment_id': c['declaration']['experiment_id'], 'cohort': 'secondary',
        'declaration_sha256': c['declaration_sha256'], 'secondary_declaration_sha256': c['secondary_declaration_sha256'],
        'task_id': tid, 'kind': 'receiver', 'training_seed': SEED}
    if any(row.get(k) != v for k, v in expected.items()): raise ValueError('Secondary job identity differs')
    return row


def generation_closure(c, seal=True):
    """Seal2,400 secondary draws plus exact references to the sealed primary.

    May be run CPU-only after workers drain. Requires primary closure, never
    primary scores; no private bytes or model weights are needed for resealing.
    """
    d, top = c['declaration'], Path(c['top']); primary.validate_seeds(c['seeds'], d['task_ids'])
    jobs = [(tid, verify_job(c, tid)) for tid in d['task_ids']]
    if any(row is None for _, row in jobs): return None
    first = primary.generation_closure(c, seal=False)
    if first is None or not (top / 'primary/generation_closure.json').exists(): return None
    if read(top / 'primary/generation_closure.json') != first: raise ValueError('Primary generation seal differs')
    files = {}; answers = []; histories = []; expected_answers = set()

    def include(path):
        path = Path(path); relative = str(path.relative_to(top)); path = primary.scoped_file(top, relative)
        files[relative] = {'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)}

    def sampler(folder, kind):
        for name in ['identity.json', 'resume.json', 'complete.json', 'completion_timing.json',
                     'source_history.json' if kind == 'reasoning' else 'answer_record.json']:
            include(folder / name)

    include(top / 'primary/generation_closure.json')
    for tid, row in jobs:
        folder = task_folder(c, tid); include(job_path(c, tid))
        runtime_path = primary.scoped_file(top, row['runtime_path']); runtime = read(runtime_path)
        setup_path = primary.scoped_file(top, row['setup_path']); setup = read(setup_path)
        if (digest(runtime) != row['runtime_identity_sha256'] or
                runtime.get('torch') != '2.8.0+cu128' or runtime.get('transformers') != '4.57.6' or
                runtime.get('attention') != 'sdpa' or runtime.get('dtype') != 'bfloat16' or
                'H200' not in runtime.get('gpu', '') or runtime.get('TF32') is not False or
                runtime.get('deterministic_algorithms') is not True or
                setup_path != runtime_path.parent / 'model_setup.json' or
                setup.get('declaration_sha256') != c['declaration_sha256'] or
                setup.get('runtime_identity_sha256') != row['runtime_identity_sha256'] or
                setup.get('charged_to_single_output_inference') is not False or
                set(setup.get('model_loading_seconds', {})) != {'source', 'receiver'} or
                any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in [*setup['model_loading_seconds'].values(), setup.get('mapper_initialization_seconds')])):
            raise ValueError('Secondary runtime or model setup differs')
        include(runtime_path); include(setup_path)
        h, hs, history_folder = primary.verify_history(c, tid, 'source')
        sampler(history_folder, 'reasoning'); include(history_folder / 'history_ready.json')
        histories.append({'task_id': tid, 'kind': 'source', 'sha256': hs, 'reused_from_primary': True})
        old = primary.verify_job_receipt(c, tid, 'receiver')
        control_path = primary.scoped_file(top, old['control_path'])
        if control_path.parent != (primary.task_folder(c, tid) / 'controls').resolve():
            raise ValueError('Primary native control belongs to another task')
        primary.validate_control_record(read(control_path), h, hs); include(control_path)
        reuse_path = primary.scoped_file(top, row['control_reuse_path'])
        expected_reuse = {'primary_control_path': old['control_path'], 'primary_control_sha256': sha(control_path),
                          'source_history_sha256': hs, 'declaration_sha256': c['declaration_sha256']}
        if reuse_path != folder / 'control_reuse.json' or read(reuse_path) != expected_reuse:
            raise ValueError('Secondary control/history reuse differs')
        include(reuse_path)
        for condition in CONDITIONS:
            cp = c['secondary_checkpoints'][condition.split('_')[0]]['mapper_sha256']
            for seed in c['seeds']['tasks'][tid]['answers']:
                expected = primary.contract(c, tid, condition, seed, hs, cp, 'secondary')
                dest = folder / condition / ('seed_' + str(seed['seed_index']))
                if not primary.verify_draw(dest, expected, h, require_receipt=True):
                    raise ValueError('Secondary job lacks a declared answer')
                raw = read(dest / 'answer.json'); seconds = raw.get('checkpoint_load_seconds')
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError('Secondary mapper setup timing invalid')
                load = raw.get('checkpoint_load_receipt_path')
                if load is not None:
                    path = primary.scoped_file(top, load); receipt = read(path)
                    required = {'task_id': tid, 'condition': condition, 'cohort': 'secondary',
                        'mapper_sha256': cp, 'declaration_sha256': c['declaration_sha256'],
                        'checkpoint_load_seconds': seconds, 'charged_to_single_output_inference': False}
                    if (path.parent != (folder / 'mapper_loads').resolve() or
                            sha(path) != raw.get('checkpoint_load_receipt_sha256') or
                            any(receipt.get(k) != v for k, v in required.items())):
                        raise ValueError('Secondary mapper load receipt differs')
                    include(path)
                elif seconds != 0 or raw.get('checkpoint_load_receipt_sha256') is not None:
                    raise ValueError('Secondary mapper load lacks receipt')
                for name in ['draw_identity.json', 'answer.json', 'draw_complete.json']: include(dest / name)
                sampler(dest / 'sampler', 'answer')
                relative = str((dest / 'answer.json').relative_to(top)); expected_answers.add(relative)
                answers.append({'contract': expected, 'contract_sha256': digest(expected), 'path': relative,
                                'answer_sha256': files[relative]['sha256']})
    actual = {str(p.relative_to(top)) for p in (top / 'secondary').rglob('answer.json')}
    actual_jobs = {p for p in (top / 'secondary/jobs').glob('*/complete.json')}
    if (actual != expected_answers or actual_jobs != {job_path(c, t) for t in d['task_ids']} or
            list((top / 'secondary').rglob('source_history.json')) or
            len(answers) != d['secondary_answer_count'] or len(answers) != 12 * d['task_count']):
        raise ValueError('Undeclared or missing secondary population')
    value = {'schema': 1, 'experiment_id': d['experiment_id'], 'cohort': 'secondary', 'training_seed': SEED,
        'declaration_sha256': c['declaration_sha256'], 'secondary_declaration_sha256': c['secondary_declaration_sha256'],
        'task_ids': d['task_ids'], 'task_count': d['task_count'], 'conditions': CONDITIONS,
        'answer_count': len(answers), 'answers': answers, 'histories': histories,
        'checkpoints': c['secondary_checkpoints'],
        'primary_generation_closure_path': 'primary/generation_closure.json',
        'primary_generation_closure_sha256': first['closure_sha256'],
        'primary_generation_closure_file_sha256': sha(top / 'primary/generation_closure.json'),
        'reused_baseline_conditions': ['A', 'B', 'D', 'P'],
        'all_secondary_generation_complete': True, 'scoring_requires_this_complete_seal': True,
        'files': [files[k] for k in sorted(files)]}
    value['closure_sha256'] = digest(value)
    if seal: bind(top / 'secondary/generation_closure.json', value)
    return value


def work(c):
    import torch
    c['guard']()
    _, s = validate_secondary(ROOT, c['plan'], verify_weights=True)
    primary.initialize(c)
    c.update(root=c['top'] / 'secondary', secondary_declaration=s,
             secondary_declaration_sha256=c['plan']['secondary_declaration_sha256'], secondary_checkpoints=s['checkpoints'])
    tasks = sorted(c['declaration']['task_ids'], key=lambda t: (-len(c['visible'][t]['prompt_ids']), t))
    with torch.no_grad():
        while True:
            c['guard'](); did_work = False
            for tid in tasks:
                if verify_job(c, tid) is not None or primary.verify_job_receipt(c, tid, 'receiver') is None: continue
                with claim_job(c, 'receiver_' + tid.replace('/', '__')) as claimed:
                    if not claimed or verify_job(c, tid) is not None: continue
                    c['guard'](); c['telemetry'].reset('secondary_' + tid)
                    started = time.time(); extra = primary.receiver_job(c, tid, cohort='secondary')
                    bind(job_path(c, tid), {'experiment_id': c['declaration']['experiment_id'], 'cohort': 'secondary',
                        'declaration_sha256': c['declaration_sha256'], 'secondary_declaration_sha256': c['secondary_declaration_sha256'],
                        'task_id': tid, 'kind': 'receiver', 'training_seed': SEED,
                        'worker_attempt': c['attempt_id'], 'finished_epoch': time.time(), 'wall_seconds': time.time() - started,
                        'runtime_identity_sha256': c['runtime_sha256'],
                        'runtime_path': str((c['status_root'] / 'runtime.json').relative_to(c['top'])),
                        'setup_path': str((c['status_root'] / 'model_setup.json').relative_to(c['top'])), **extra})
                    c['publish'](stage='secondary_job_complete', task_id=tid)
                    did_work = True; gc.collect(); torch.cuda.empty_cache(); break
            if not did_work:
                closure = generation_closure(c)
                c['publish'](state='complete', stage='secondary_generation_sealed' if closure else 'no_ready_unclaimed_secondary_jobs',
                             all_secondary_generation_complete=bool(closure), gpu_idle_release_requested=True)
                return


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['source-manifest', 'prepare', 'run', 'seal'])
    for name in ['plan', 'plan-sha256', 'worker-id', 'declaration', 'declaration-sha256', 'fixed-manifest', 'rotating-manifest', 'output',
                 'source-manifest', 'source-manifest-sha256']:
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.action == 'source-manifest':
        value = source_manifest(ROOT, args.output)
        print(json.dumps({'path': args.output, 'sha256': sha(ROOT / args.output), 'adapter_commit': value['adapter_commit']})); return
    if args.action == 'prepare':
        value = prepare(ROOT, args.declaration, args.declaration_sha256, args.fixed_manifest, args.rotating_manifest, args.output,
                        source_manifest_path=args.source_manifest, source_manifest_sha256=args.source_manifest_sha256)
        print(json.dumps({'secondary_declaration_path': args.output, 'sha256': sha(ROOT / args.output), 'answer_count': value['answer_count']})); return
    if args.action == 'seal':
        value = generation_closure(public_context(ROOT, args.plan, args.plan_sha256))
        print(json.dumps({'complete': value is not None})); return
    from scripts.coding_confirmation_runtime import build_context
    c = None
    try:
        c = build_context(args.plan, 'confirmation_secondary_generation', args.worker_id); work(c)
    except BaseException as exc:
        if c:
            c['telemetry'].failure(exc)
            write(c['status_root'] / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
                  'traceback': traceback.format_exc(), 'epoch': time.time()})
            c['publish'](state='failed', stage='secondary_generation_failed', error=str(exc))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)): raise
        raise SystemExit(65 if isinstance(exc, (ValueError, FloatingPointError, AssertionError)) else 1) from exc


if __name__ == '__main__': main()
