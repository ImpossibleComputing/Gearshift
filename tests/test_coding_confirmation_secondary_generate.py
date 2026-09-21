"""Public synthetic transactions only: no model execution or private tests."""
import copy
import importlib.util
from pathlib import Path

import pytest

from gearshift.coding_control import digest, sha, write
from scripts import coding_confirmation_secondary_generate as s

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('secondary_primary_fixture', ROOT / 'tests/test_coding_confirmation_generate.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def checkpoint_fixture(repo, d, arm):
    schedules = 'replication_schedules.json'; write(repo / schedules, {'seed': s.SEED})
    rd = {'training_seed': s.SEED, 'selected_checkpoint_sha256': s.START_SHA, 'target_additional_updates': 1024,
        'training_task_ids': list(range(104)), 'experiment_id': d['experiment_id'], 'schedules_path': schedules,
        'schedules_sha256': sha(repo / schedules), 'corpus_manifest_sha256': 'f' * 64,
        'optimizer': {'name': 'AdamW', 'lr': .00001}, 'replication_id': 'replication_seed_20260919',
        'result_root': 'results/replication'}
    write(repo / s.REPLICATION, rd)
    d['inputs'].update({s.REPLICATION: sha(repo / s.REPLICATION), schedules: sha(repo / schedules)})
    d['model_config_path'] = 'configs/coding_pilot_v1/pilot.json'
    identity = {'experiment_id': d['experiment_id'] + '/replication', 'arm': arm, 'code_commit': 'a' * 40,
        'config_sha256': sha(repo / s.REPLICATION), 'schedules_sha256': rd['schedules_sha256'],
        'corpus_sha256': rd['corpus_manifest_sha256'], 'selected_checkpoint_sha256': s.START_SHA,
        'models': s.read(repo / d['model_config_path'])['models'],
        'configuration': {'optimizer': rd['optimizer'], 'dtype': 'bfloat16', 'attention': 'sdpa',
            'gradient_positions': 32, 'full_context': True, 'scheduler': 'constant',
            'training_seed': s.SEED, 'schedule_position_offset': 96, 'replication_id': rd['replication_id']}}
    folder = repo / rd['result_root'] / 'arms' / arm / 'checkpoints/step_1024'; folder.mkdir(parents=True)
    for name in ['mapper.pt', 'full.pt']: (folder / name).write_bytes((arm + name + 'synthetic bytes').encode())
    manifest = {'checkpoint_identity': identity, 'checkpoint_identity_sha256': digest(identity),
        'arm': arm, 'step': 1024, 'schedule_position': 1024, 'complete_resumable': True, 'verified_roundtrip': True,
        'state_components': s.COMPONENTS,
        'files': {n: {'bytes': (folder / n).stat().st_size, 'sha256': sha(folder / n)} for n in ['mapper.pt', 'full.pt']}}
    write(folder / 'manifest.json', manifest)
    return str((folder / 'manifest.json').relative_to(repo))


def frozen_fixture(tmp_path, monkeypatch):
    _, d = f.declaration_fixture(tmp_path)
    tids = ['atcoder/task_' + str(i) for i in range(200)]
    write(tmp_path / 'seeds.json', f.seeds(tids))
    write(tmp_path / 'visible.json', [{'task_id': t, 'prompt': 'public', 'prompt_ids': [1, 2]} for t in tids])
    d.update(task_ids=tids, task_count=200, primary_answer_count=4800, source_commit='b' * 40,
             secondary_conditions=s.CONDITIONS, secondary_answer_count=2400, secondary_training_seed=s.SEED)
    d['analysis_path'] = 'analysis.json'; write(tmp_path / 'analysis.json', {'status': 'FROZEN', 'task_count': 200})
    d['inputs']['analysis.json'] = sha(tmp_path / 'analysis.json')
    paths = {arm: checkpoint_fixture(tmp_path, d, arm) for arm in ['FIXED', 'ROTATING']}
    d['inputs'].update({name: sha(tmp_path / name) for name in ['seeds.json', 'visible.json']})
    for name in s.IMPLEMENTATION:
        dest = tmp_path / name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('# synthetic frozen adapter source for metadata validation\n')
        if name in d['implementation']: d['implementation'][name] = sha(dest)
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    write(tmp_path / 'declaration.json', d)
    monkeypatch.setattr(s.subprocess, 'check_output', lambda args, **kw: 'c' * 40 if args[1] == 'rev-parse'
                        else (tmp_path / args[-1].split(':', 1)[1]).read_bytes())
    addendum = s.prepare(tmp_path, 'declaration.json', sha(tmp_path / 'declaration.json'),
                         paths['FIXED'], paths['ROTATING'], 'secondary.json')
    plan = {'declaration_path': 'declaration.json', 'declaration_sha256': sha(tmp_path / 'declaration.json'),
        'secondary_declaration_path': 'secondary.json', 'secondary_declaration_sha256': sha(tmp_path / 'secondary.json'),
        'experiment_id': d['experiment_id'], 'code_commit': d['source_commit'], 'result_root': 'results'}
    write(tmp_path / 'dispatch.json', plan)
    return d, addendum, plan


def test_secondary_freeze_verifies_both_full_and_mapper_bytes_and_cpu_reader_needs_no_weights(tmp_path, monkeypatch):
    d, addendum, plan = frozen_fixture(tmp_path, monkeypatch)
    assert addendum['answer_count'] == 2400 and len(addendum['task_ids']) == 200
    assert s.validate_secondary(tmp_path, plan) == (d, addendum)
    for cp in addendum['checkpoints'].values():
        (tmp_path / cp['mapper_path']).unlink(); (tmp_path / cp['full_checkpoint_path']).unlink()
    assert s.public_context(tmp_path, 'dispatch.json', sha(tmp_path / 'dispatch.json'))['secondary_checkpoints'] == addendum['checkpoints']
    with pytest.raises(ValueError, match='missing'): s.validate_secondary(tmp_path, plan, verify_weights=True)


def test_remote_freeze_uses_caller_pinned_source_manifest_without_git(tmp_path, monkeypatch):
    d, addendum, plan = frozen_fixture(tmp_path, monkeypatch)
    source = s.source_manifest(tmp_path, 'adapter_source.json'); source_sha = sha(tmp_path / 'adapter_source.json')
    monkeypatch.setattr(s.subprocess, 'check_output', lambda *a, **kw: pytest.fail('Remote freeze must not require git'))
    cps = addendum['checkpoints']
    remote = s.prepare(tmp_path, 'declaration.json', plan['declaration_sha256'],
        cps['FIXED']['manifest_path'], cps['ROTATING']['manifest_path'], 'remote_secondary.json',
        source_manifest_path='adapter_source.json', source_manifest_sha256=source_sha)
    assert remote['adapter_commit'] == source['adapter_commit']
    plan.update(secondary_declaration_path='remote_secondary.json', secondary_declaration_sha256=sha(tmp_path / 'remote_secondary.json'))
    assert s.validate_secondary(tmp_path, plan)[1] == remote
    with pytest.raises(ValueError, match='manifest hash'):
        s.verify_source_manifest(tmp_path, 'adapter_source.json', '0' * 64)
    (tmp_path / next(iter(s.IMPLEMENTATION))).write_text('changed staged adapter')
    with pytest.raises(ValueError, match='adapter bytes'): s.verify_source_manifest(tmp_path, 'adapter_source.json', source_sha)


@pytest.mark.parametrize('mutation', ['seed', 'initializer', 'step', 'schedule', 'missing_optimizer', 'full_bytes', 'mapper_bytes'])
def test_bad_secondary_checkpoint_fails_before_any_model_setup(tmp_path, monkeypatch, mutation):
    d, addendum, plan = frozen_fixture(tmp_path, monkeypatch)
    cp = addendum['checkpoints']['FIXED']; path = tmp_path / cp['manifest_path']; m = s.read(path)
    if mutation == 'full_bytes': (tmp_path / cp['full_checkpoint_path']).write_bytes(b'corrupt')
    elif mutation == 'mapper_bytes': (tmp_path / cp['mapper_path']).write_bytes(b'corrupt')
    else:
        if mutation == 'seed': m['checkpoint_identity']['configuration']['training_seed'] = 20260915
        elif mutation == 'initializer': m['checkpoint_identity']['selected_checkpoint_sha256'] = '0' * 64
        elif mutation == 'step': m['step'] = 768
        elif mutation == 'schedule': m['schedule_position'] = 512
        else: m['state_components'].remove('optimizer')
        m['checkpoint_identity_sha256'] = digest(m['checkpoint_identity']); write(path, m)
    with pytest.raises(ValueError): s.checkpoint(tmp_path, d, 'FIXED', cp['manifest_path'], verify_weights=True)


@pytest.mark.parametrize('mutation', ['task_count', 'condition', 'seed_count', 'primary_commit', 'implementation'])
def test_secondary_addendum_cannot_change_population_or_frozen_source(tmp_path, monkeypatch, mutation):
    _, addendum, plan = frozen_fixture(tmp_path, monkeypatch)
    if mutation == 'task_count': addendum['task_count'] = 160
    elif mutation == 'condition': addendum['conditions'].append('D')
    elif mutation == 'seed_count': addendum['answer_draws_per_task_condition'] = 4
    elif mutation == 'primary_commit': plan['code_commit'] = addendum['adapter_commit']
    else: addendum['implementation'][next(iter(addendum['implementation']))] = '0' * 64
    write(tmp_path / 'secondary.json', addendum); plan['secondary_declaration_sha256'] = sha(tmp_path / 'secondary.json')
    with pytest.raises(ValueError): s.validate_secondary(tmp_path, plan)


def secondary_fixture(tmp_path, count=2):
    c = f.context(tmp_path, count); f.complete_fixture(c)
    c['declaration'].update(secondary_answer_count=count * 12, secondary_conditions=s.CONDITIONS)
    s.primary.generation_closure(c)
    c.update(root=c['top'] / 'secondary', secondary_declaration_sha256='f' * 64,
             secondary_checkpoints={arm: {'mapper_sha256': ch * 64} for arm, ch in [('FIXED', '1'), ('ROTATING', '2')]})
    for tid in c['declaration']['task_ids']:
        h, hs, _ = s.primary.verify_history(c, tid, 'source'); folder = s.task_folder(c, tid)
        old = s.primary.verify_job_receipt(c, tid, 'receiver'); control = c['top'] / old['control_path']
        write(folder / 'control_reuse.json', {'primary_control_path': old['control_path'], 'primary_control_sha256': sha(control),
            'source_history_sha256': hs, 'declaration_sha256': c['declaration_sha256']})
        write(s.job_path(c, tid), {'experiment_id': c['declaration']['experiment_id'], 'cohort': 'secondary',
            'declaration_sha256': c['declaration_sha256'], 'secondary_declaration_sha256': c['secondary_declaration_sha256'],
            'training_seed': s.SEED, 'kind': 'receiver', 'task_id': tid, 'runtime_path': 'workers/test/runtime.json',
            'runtime_identity_sha256': c['runtime_sha256'], 'setup_path': 'workers/test/model_setup.json',
            'control_reuse_path': str((folder / 'control_reuse.json').relative_to(c['top']))})
        for condition in s.CONDITIONS:
            for seed in c['seeds']['tasks'][tid]['answers']:
                cp = c['secondary_checkpoints'][condition.split('_')[0]]['mapper_sha256']
                expected = s.primary.contract(c, tid, condition, seed, hs, cp, 'secondary')
                dest = folder / condition / ('seed_' + str(seed['seed_index']))
                original = {'task_id': tid, 'answer_ids': [9, 151645], 'answer_text': 'synthetic secondary',
                    'answer_seed': seed['answer_seed'], 'answer_ended_eos': True, 'answer_capped': False}
                record = f.durable(dest / 'sampler', 'answer', tid, seed['stream'],
                    {'history_sha256': digest(h), 'prefix_ids': h['prefix_ids'], 'bridge_ids': h['bridge_ids']},
                    'receiver', original, c, s.primary.cache_identity(expected))
                write(dest / 'draw_identity.json', expected)
                write(dest / 'answer.json', {**record, **expected, 'immutable_base_cache_unchanged': True,
                    'clone_no_alias': True, 'runtime_identity_sha256': c['runtime_sha256'],
                    'checkpoint_load_seconds': 0., 'checkpoint_load_receipt_path': None, 'checkpoint_load_receipt_sha256': None})
                s.primary.verify_draw(dest, expected, h)
    return c


def test_secondary_closure_exact_order_reuses_primary_without_mutation(tmp_path):
    c = secondary_fixture(tmp_path)
    before = {str(p): sha(p) for p in (c['top'] / 'primary').rglob('*') if p.is_file()}
    closed = s.generation_closure(c)
    assert closed['answer_count'] == 24 and closed['reused_baseline_conditions'] == ['A', 'B', 'D', 'P']
    assert [(r['contract']['task_id'], r['contract']['condition'], r['contract']['seed_index']) for r in closed['answers']] == [
        (t, condition, seed) for t in c['declaration']['task_ids'] for condition in s.CONDITIONS for seed in range(3)]
    assert len([r for r in closed['files'] if r['path'].endswith('/completion_timing.json')]) == 26
    assert s.generation_closure(c) == closed
    assert before == {str(p): sha(p) for p in (c['top'] / 'primary').rglob('*') if p.is_file()}


def test_secondary_mapper_io_receipt_is_closed_and_bound_to_condition_and_checkpoint(tmp_path):
    c = secondary_fixture(tmp_path, 1); tid = c['declaration']['task_ids'][0]; folder = s.task_folder(c, tid)
    path = folder / 'mapper_loads/attempt_FIXED_M.json'
    receipt = {'task_id': tid, 'condition': 'FIXED_M', 'cohort': 'secondary',
        'mapper_sha256': c['secondary_checkpoints']['FIXED']['mapper_sha256'],
        'declaration_sha256': c['declaration_sha256'], 'checkpoint_load_seconds': .25,
        'charged_to_single_output_inference': False}
    write(path, receipt)
    for seed in range(3):
        draw = folder / 'FIXED_M' / ('seed_' + str(seed)); row = s.read(draw / 'answer.json')
        row.update(checkpoint_load_seconds=.25, checkpoint_load_receipt_path=str(path.relative_to(c['top'])),
                   checkpoint_load_receipt_sha256=sha(path)); write(draw / 'answer.json', row)
        expected = s.read(draw / 'draw_identity.json'); (draw / 'draw_complete.json').unlink()
        s.primary.verify_draw(draw, expected)
    closure = s.generation_closure(c, seal=False)
    assert str(path.relative_to(c['top'])) in {r['path'] for r in closure['files']}
    receipt['cohort'] = 'primary'; write(path, receipt)
    with pytest.raises(ValueError, match='mapper load'): s.generation_closure(c)


@pytest.mark.parametrize('mutation', ['missing_draw', 'extra_draw', 'new_history', 'reuse', 'timing', 'job_seed', 'source_control', 'checkpoint', 'setup'])
def test_secondary_closure_rejects_invalid_population_or_reused_evidence(tmp_path, mutation):
    c = secondary_fixture(tmp_path, 1); tid = c['declaration']['task_ids'][0]; folder = s.task_folder(c, tid)
    draw = folder / 'FIXED_M/seed_0'
    if mutation == 'missing_draw': (draw / 'answer.json').unlink()
    elif mutation == 'extra_draw': write(folder / 'D/seed_0/answer.json', {'undeclared': True})
    elif mutation == 'new_history': write(folder / 'large_history/source_history.json', {'reroll': True})
    elif mutation == 'reuse': write(folder / 'control_reuse.json', {'source_history_sha256': '0' * 64})
    elif mutation == 'timing': (draw / 'sampler/completion_timing.json').unlink()
    elif mutation == 'job_seed':
        row = s.read(s.job_path(c, tid)); row['training_seed'] = 20260915; write(s.job_path(c, tid), row)
    elif mutation == 'source_control':
        path = c['top'] / s.primary.verify_job_receipt(c, tid, 'receiver')['control_path']; row = s.read(path)
        row['whole_native_tensor_exact'] = False; write(path, row)
    elif mutation == 'checkpoint': c['secondary_checkpoints']['FIXED']['mapper_sha256'] = '0' * 64
    else:
        row = s.read(c['status_root'] / 'model_setup.json'); row['mapper_initialization_seconds'] = -1
        write(c['status_root'] / 'model_setup.json', row)
    with pytest.raises((ValueError, FileNotFoundError)): s.generation_closure(c)
    assert not (c['top'] / 'secondary/generation_closure.json').exists()


def test_closure_waits_for_primary_seal_and_all_secondary_jobs(tmp_path):
    c = secondary_fixture(tmp_path, 1); p = c['top'] / 'primary/generation_closure.json'; saved = p.read_bytes(); p.unlink()
    assert s.generation_closure(c) is None
    p.write_bytes(saved); s.job_path(c, c['declaration']['task_ids'][0]).unlink()
    assert s.generation_closure(c) is None


def test_queue_calls_only_secondary_receiver_and_uses_separate_claims(tmp_path, monkeypatch):
    c = secondary_fixture(tmp_path, 1); tid = c['declaration']['task_ids'][0]; s.job_path(c, tid).unlink()
    c['plan'] = {'secondary_declaration_sha256': c['secondary_declaration_sha256']}
    monkeypatch.setattr(s, 'validate_secondary', lambda *a, **kw: (c['declaration'], {'checkpoints': c['secondary_checkpoints']}))
    monkeypatch.setattr(s.primary, 'initialize', lambda context: None)
    calls = []
    def receiver(context, task, cohort):
        assert context['root'] == context['top'] / 'secondary'
        calls.append((task, cohort))
        return {'control_reuse_path': str((s.task_folder(c, tid) / 'control_reuse.json').relative_to(c['top']))}
    monkeypatch.setattr(s.primary, 'receiver_job', receiver)
    monkeypatch.setattr(s.primary, 'reasoning_job', lambda *a, **kw: pytest.fail('Secondary must never reason'))
    s.work(c)
    assert calls == [(tid, 'secondary')]
    assert (c['top'] / 'secondary/claims' / ('receiver_' + tid.replace('/', '__') + '.lock')).exists()
    assert not (c['top'] / 'claims').exists()
    s.work(c); assert len(calls) == 1  # No completed draw or job reroll.
