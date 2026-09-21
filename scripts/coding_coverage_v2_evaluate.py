#!/usr/bin/env python3
"""Immutable asynchronous coverage jobs; raw sampling and private scoring are separate.

The numerical calls, sampler, templates and native splice controls match the
preceding coverage evaluator. Only task/checkpoint scheduling and durable output
transactions differ. The caller owns device setup, deadlines and job claims.
"""
import gc
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import bind, digest, sha, write

ROOT = Path(__file__).resolve().parents[1]
STEPS = (0, 128, 256, 512, 768, 1024)


def read(path):
    return json.loads(Path(path).read_text())


def _repo(c):
    return Path(c.get('repo_root', ROOT)).resolve()


def _rel(path, repo):
    return str(Path(path).resolve().relative_to(repo))


def _safe(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,180}', value):
        raise ValueError('Unsafe or missing job/attempt identity')
    return value


def load_inputs(c):
    """Only public prompts and saved source-written trajectories are loaded here."""
    if '_v2_evaluation_inputs' in c:
        return c['_v2_evaluation_inputs']
    from gearshift.coding_coverage import DECLARATION
    from scripts.coding_partial_corpus import load_corpus
    repo = _repo(c)
    declaration_path = c.get('declaration_path', DECLARATION)
    expected = c.get('declaration_sha256', c.get('spec', {}).get('declaration_sha256'))
    if not expected or sha(repo / declaration_path) != expected:
        raise ValueError('Evaluation declaration hash missing or changed')
    d = read(repo / declaration_path)
    for rel, expected_hash in d['unchanged_source_files'].items():
        if sha(repo / rel) != expected_hash:
            raise ValueError('Frozen numerical/scoring implementation changed: ' + rel)
    for name in ('corpus_manifest', 'answer_seeds', 'panels'):
        rel = d[name] if name == 'corpus_manifest' else d[name + '_path']
        if sha(repo / rel) != d[name + '_sha256']:
            raise ValueError('Frozen evaluation input changed: ' + name)
    histories, _, _ = load_corpus(repo, repo / d['corpus_manifest'], features=False)
    val = [h for h in histories if h['split'] == 'validation']
    if [h['task_id'] for h in val] != d['validation_task_ids']:
        raise ValueError('Validation membership/order differs')
    seeds = read(repo / d['answer_seeds_path'])['validation']
    if set(seeds) != set(d['validation_task_ids']) or any(
            [x['seed_index'] for x in seeds[t]] != [0, 1, 2] for t in seeds):
        raise ValueError('Expected three exact previously declared seeds per validation task')
    visible = {x['task_id']: x for x in read(repo / 'data/coding_pilot_v1/visible/coverage_generalization.json')}
    data = (d, {h['task_id']: h for h in val}, visible, seeds, read(repo / d['panels_path']))
    c['_v2_evaluation_inputs'] = data
    return data


def validate_job(job, d):
    _safe(job['job_id']); _safe(job['experiment_id'])
    if job['arm'] not in ('START', 'FIXED', 'ROTATING') or job['step'] not in STEPS:
        raise ValueError('Undeclared checkpoint arm/update')
    if (job['arm'] == 'START') != (job['step'] == 0):
        raise ValueError('Step zero is one shared START; trained arms use positive updates')
    tids = job['task_ids']
    if not tids or len(tids) != len(set(tids)) or not set(tids) <= set(d['validation_task_ids']):
        raise ValueError('Job is not a unique shard of the 21 validation histories')
    forms = job.get('forms', ['M', 'H'])
    if len(forms) != len(set(forms)) or not set(forms) <= {'M', 'H'}:
        raise ValueError('Undeclared generation form')
    if job.get('include_controls', False) and job['arm'] != 'START':
        raise ValueError('D/P controls are generated once under START jobs')
    if not forms and not job.get('include_controls', False) and not job.get('validation_kl', True):
        raise ValueError('Empty evaluation job')
    cp = job['checkpoint']
    if not isinstance(cp.get('sha256'), str) or len(cp['sha256']) != 64:
        raise ValueError('Immutable mapper hash missing')
    if job['arm'] == 'START' and cp['sha256'] != d['selected_checkpoint_sha256']:
        raise ValueError('START must use the original selected update-96 mapper')
    return [(job['arm'] + '_' + f, f) for f in forms] + ([('D', 'D'), ('P', 'P')] if job.get('include_controls', False) else [])


def draw_contract(job, tid, condition, form, seed, history_sha):
    return {'job_identity_sha256': digest(job), 'experiment_id': job['experiment_id'],
            'task_id': tid, 'condition': condition, 'form': form, 'step': job['step'],
            'scope': 'validation', 'seed_index': seed['seed_index'], 'stream': seed['stream'],
            'answer_seed': seed['answer_seed'], 'checkpoint_sha256': job['checkpoint']['sha256'] if form in ('M', 'H') else None,
            'source_history_sha256': history_sha if form != 'P' else None,
            'teacher_answer_prefix_supplied': False}


def validate_raw(row, contract):
    if any(row.get(k) != v for k, v in contract.items()):
        raise ValueError('Existing answer does not match immutable task/seed/checkpoint contract')
    if 'score' in row or 'code' in row:
        raise ValueError('Raw generated answers cannot be modified by private scoring')
    if not isinstance(row.get('answer_ids'), list) or not row['answer_ids']:
        raise ValueError('Missing raw answer token IDs')
    if not row.get('immutable_base_cache_unchanged') or not row.get('clone_no_alias'):
        raise ValueError('Answer is missing cache isolation checks')


def completed_draw(dest, contract):
    """A committed draw is read, hashed and reused; it is never silently rerolled."""
    dest = Path(dest)
    if not (dest / 'answer.json').exists():
        if (dest / 'complete.json').exists():
            raise ValueError('Draw receipt exists without its raw answer')
        return None
    row = read(dest / 'answer.json'); validate_raw(row, contract)
    receipt = {'contract_sha256': digest(contract), 'answer_sha256': sha(dest / 'answer.json')}
    bind(dest / 'complete.json', receipt)
    return row


def _archive_incomplete(dest):
    """Keep interrupted attempts, including partial tokens/RNG, outside canonical draws."""
    dest = Path(dest)
    if not dest.exists():
        return
    # A finished sampler checkpoint is recoverable without another model draw.
    if (dest / 'resume.json').exists() and read(dest / 'resume.json').get('state') == 'complete':
        return
    base = dest.parent / 'interrupted_attempts'
    base.mkdir(parents=True, exist_ok=True)
    target = base / (dest.name + '_' + str(time.time_ns()))
    os.rename(dest, target)


def _recover_finished_sampler(dest, backend, history, tid, seed):
    p = Path(dest) / 'resume.json'
    if not p.exists():
        return None
    r = read(p)
    if r.get('state') != 'complete':
        return None
    if (r['task_id'], r['stream'], r['seed'], r['cap'], r['historical_prefix_length']) != (
            tid, seed['stream'], seed['answer_seed'], 4096, len(history['prefix_ids'])):
        raise ValueError('Finished sampler state differs from exact draw identity')
    tokens = r['answer_ids']
    if not tokens or (tokens[-1] not in backend.eos and len(tokens) != 4096):
        raise ValueError('Sampler complete state has neither EOS nor the declared cap')
    return {'task_id': tid, 'answer_ids': tokens,
            'answer_text': backend.tokenizer.decode(tokens, skip_special_tokens=True),
            'answer_text_with_special_tokens': backend.tokenizer.decode(tokens, skip_special_tokens=False),
            'answer_seed': seed['answer_seed'], 'rng_initial': r['rng_initial'], 'rng_final': r['rng_state'],
            'answer_seconds': None, 'bridge_seconds': None, 'first_answer_token_seconds': None,
            'answer_ended_eos': tokens[-1] in backend.eos, 'answer_capped': tokens[-1] not in backend.eos,
            'bridge_token_count': len(history['bridge_ids']), 'timing_recovered_without_measurement': True,
            'instrumentation': 'Recovered completed durable sampler tokens/RNG after interruption before answer-record commit; no new sampling and timing unavailable.'}


def _answer_manifest(root, job, seeds):
    files = []
    for tid in job['task_ids']:
        folder = root / 'validation' / tid.replace('/', '__')
        for condition, form in [(job['arm'] + '_' + f, f) for f in job.get('forms', ['M', 'H'])] + ([('D', 'D'), ('P', 'P')] if job.get('include_controls', False) else []):
            for seed in seeds[tid]:
                dest = folder / condition / ('seed_' + str(seed['seed_index']))
                contract = draw_contract(job, tid, condition, form, seed, sha(folder / 'source_history.json'))
                if completed_draw(dest, contract) is None:
                    raise ValueError('Generation job is missing a committed draw')
                files.append({'path': str((dest / 'answer.json').relative_to(root)), 'sha256': sha(dest / 'answer.json'),
                              'receipt_sha256': sha(dest / 'complete.json')})
    return files


def run_job(c, source, receiver, mapper, job):
    """Called on a persistent evaluator GPU; checkpoints and raw answers stay immutable.

    c requires root, identity, attempt_id, guard, publish, telemetry and the frozen
    declaration hash (top-level or spec). repo_root defaults to this checkout.
    Each job is exclusively claimed by the caller, including across retries.
    """
    import torch
    from gearshift.coding_coverage_runtime import CoverageRuntime
    from gearshift.coding_recovery_answer import answer_instrumented
    from gearshift.coding_post_progress import splice
    from gearshift.coding_inference import sync, logit_metrics
    from gearshift.core import CacheExtractor, CacheInjector

    d, lookup, visible, seeds, panels = load_inputs(c)
    conditions = validate_job(job, d)
    repo = _repo(c); root = Path(c['root']) / 'jobs' / job['job_id']
    attempt = _safe(c['attempt_id']); root.mkdir(parents=True, exist_ok=True)
    bind(root / 'job.json', job)
    attempt_root = root / 'attempts' / attempt
    bind(attempt_root / 'identity.json', {'job_identity_sha256': digest(job), 'attempt_id': attempt, 'runtime_identity': c['identity']})
    cp = repo / job['checkpoint']['path']
    if sha(cp) != job['checkpoint']['sha256']:
        raise ValueError('Immutable evaluation checkpoint changed')
    if (root / 'generation_complete.json').exists():
        complete = read(root / 'generation_complete.json')
        if complete['job_identity_sha256'] != digest(job) or complete['files'] != _answer_manifest(root, job, seeds):
            raise ValueError('Completed generation job manifest changed')
        for item in complete['kl_files'] + complete.get('control_files', []):
            if sha(root/item['path']) != item['sha256']:
                raise ValueError('Completed validation/control evidence changed')
        return complete
    state = torch.load(cp, map_location='cpu', weights_only=True)
    if job['arm'] != 'START' and (state['step'] != job['step'] or state['arm'] != job['arm']):
        raise ValueError('Checkpoint payload arm/update differs')
    mapper.load_state_dict(state['state_dict'], strict=True); del state
    mapper.requires_grad_(False)
    runtime = CoverageRuntime(source, receiver, mapper, c['guard'], c['telemetry'])
    started = time.time()
    try:
        with torch.no_grad():
            for index, tid in enumerate(job['task_ids']):
                c['guard'](); obj = lookup[tid]; h = obj['source_history']
                folder = root / 'validation' / tid.replace('/', '__')
                bind(folder / 'source_history.json', h)
                history_sha = sha(folder / 'source_history.json')
                if h['prompt_ids'] != visible[tid]['prompt_ids'] or h['prefix_ids'][:len(h['prompt_ids'])] != h['prompt_ids']:
                    raise ValueError('Visible prompt/full saved history mismatch')
                missing = [(name, form, seed) for name, form in conditions for seed in seeds[tid]
                           if completed_draw(folder / name / ('seed_' + str(seed['seed_index'])),
                                             draw_contract(job, tid, name, form, seed, history_sha)) is None]
                kl_path = folder / 'validation_kl.json'
                if not missing and (not job.get('validation_kl', True) or kl_path.exists()):
                    continue
                nprompt = len(h['prompt_ids']); n = len(h['prefix_ids'])
                c['publish'](stage='coverage_v2_generation', job_id=job['job_id'], task_id=tid, completed_tasks=index)
                c['telemetry'].reset(job['job_id'] + '/' + tid)
                sync(); before = time.monotonic(); out = source.prefill_chunked(h['prefix_ids'])
                sp = CacheExtractor.tensors(out.past_key_values); del out; sync(); source_prep = time.monotonic() - before
                before = time.monotonic(); out = receiver.prefill_chunked(h['prefix_ids'])
                tp = CacheExtractor.tensors(out.past_key_values); del out; sync(); native_prep = time.monotonic() - before
                before = time.monotonic(); out = receiver.prefill_chunked(h['prompt_ids'])
                prompt = CacheExtractor.tensors(out.past_key_values); del out; sync(); prompt_prep = time.monotonic() - before
                sliced = tuple(tuple(x[..., :nprompt, :] for x in pair) for pair in tp)
                nn = splice(sliced, tp, nprompt)
                exact = all(torch.equal(x, y) for p, q in zip(nn, tp) for x, y in zip(p, q))
                a = receiver.forward(h['bridge_ids'], CacheInjector.create(tp, clone=True)).logits.detach().clone()
                z = receiver.forward(h['bridge_ids'], CacheInjector.create(nn, clone=True)).logits.detach().clone()
                metric = logit_metrics(a, z)
                partial = splice(prompt, tp, nprompt)
                p = receiver.forward(h['bridge_ids'], CacheInjector.create(partial, clone=True)).logits.detach().clone()
                partial_metric = logit_metrics(a, p)
                prefix_diff = max(float((x-y).abs().max()) for pp, qq in zip(prompt, sliced) for x, y in zip(pp, qq))
                control = {'task_id': tid, 'whole_native_split_tensor_exact': exact, 'whole_native_splice_bridge': metric,
                           'actual_partial_prompt_cache_max_abs': prefix_diff, 'actual_partial_prompt_splice_bridge': partial_metric,
                           'passed': exact and metric['max_abs'] == 0, 'attempt_id': attempt}
                write(attempt_root / tid.replace('/', '__') / 'native_splice_control.json', control)
                if not control['passed']:
                    raise RuntimeError('Native/native splice changed whole-history execution')
                del sliced, nn, a, z, p, partial; gc.collect()
                versions = [x._version for pair in (*sp, *tp, *prompt) for x in pair]
                bind(folder / 'boundary.json', {'task_id': tid, 'prompt_tokens': nprompt, 'full_prefix_tokens': n,
                     'reasoning_suffix_tokens': n-nprompt, 'absolute_suffix_positions': [nprompt, n-1],
                     'bridge_ids': h['bridge_ids'], 'teacher_answer_prefix_supplied': False,
                     'source_history_sha256': history_sha, 'no_cropping': True})
                if job.get('validation_kl', True) and not kl_path.exists():
                    row = {'task_id': tid, 'arm': job['arm'], 'step': job['step'], 'job_identity_sha256': digest(job),
                           'checkpoint_sha256': job['checkpoint']['sha256'], 'attempt_id': attempt}
                    before = time.monotonic()
                    for panel in ('legacy', 'broad'):
                        values = runtime.kl(obj, panels[tid][panel], 'natural_handoff_boundary', sp, tp, require_grad=False)
                        vals = values.flatten().cpu().tolist()
                        if not vals or not all(math.isfinite(v) for v in vals):
                            raise FloatingPointError('Nonfinite validation KL')
                        row[panel] = {'positions': panels[tid][panel], 'per_position_kl': vals, 'mean_kl': sum(vals)/len(vals)}
                        del values
                    row['wall_seconds'] = time.monotonic() - before
                    bind(kl_path, row)
                # Counterbalance using the original global task index, independent of sharding.
                shift = d['validation_task_ids'].index(tid) % max(1, len(conditions))
                ordered = conditions[shift:] + conditions[:shift]
                bind(folder / 'condition_order.json', [x[0] for x in ordered])
                for condition, form in ordered:
                    todo = [seed for seed in seeds[tid] if (condition, form, seed) in missing]
                    if not todo:
                        continue
                    c['guard'](); c['publish'](stage='coverage_v2_generation', job_id=job['job_id'], task_id=tid, condition=condition)
                    mapping = splicing = 0.; ph = h
                    prefill = native_prep if form == 'D' else 0.; prefill_tokens = n if form == 'D' else 0
                    if form in ('M', 'H'):
                        sync(); before = time.monotonic(); mapped = mapper(sp); sync(); mapping = time.monotonic()-before
                        if form == 'H':
                            before = time.monotonic(); base = splice(prompt, mapped, nprompt); sync(); splicing = time.monotonic()-before
                            del mapped; prefill = prompt_prep; prefill_tokens = nprompt
                        else:
                            base = mapped; del mapped
                    elif form == 'D':
                        base = tp
                    else:
                        ids = receiver.tokenizer.apply_chat_template([{'role': 'user', 'content': visible[tid]['prompt']}], tokenize=True,
                                                                     add_generation_prompt=True, enable_thinking=False)
                        text = receiver.tokenizer.decode(ids, skip_special_tokens=False)
                        if '<think>\n\n</think>' not in text:
                            raise ValueError('Non-thinking template differs')
                        ph = {'prefix_ids': ids[:-1], 'bridge_ids': ids[-1:]}
                        bind(folder / 'prompt_only_template.json', {'task_id': tid, 'prompt_ids': ids, 'rendered_template': text,
                             'enable_thinking': False, 'separate_thinking_stage_requested': False,
                             'prefill_prefix_ids': ph['prefix_ids'], 'bridge_ids': ph['bridge_ids']})
                        sync(); before = time.monotonic(); out = receiver.prefill_chunked(ph['prefix_ids'])
                        base = CacheExtractor.tensors(out.past_key_values); del out; sync()
                        prefill = time.monotonic()-before; prefill_tokens = len(ph['prefix_ids'])
                    for seed in todo:
                        c['guard'](); dest = folder / condition / ('seed_' + str(seed['seed_index']))
                        contract = draw_contract(job, tid, condition, form, seed, history_sha)
                        _archive_incomplete(dest)
                        # Contract is durable BEFORE the sampler writes any partial/completed tokens.
                        bind(dest / 'draw_identity.json', contract)
                        bind(dest / 'draw_attempt.json', {'attempt_id': attempt, 'runtime_identity_sha256': digest(c['identity'])}) if not (dest / 'draw_attempt.json').exists() else None
                        base_versions = [x._version for pair in base for x in pair]
                        row = _recover_finished_sampler(dest, receiver, ph, tid, seed)
                        clone_seconds = 0.
                        if row is None:
                            sync(); before = time.monotonic(); cache = CacheInjector.create(base, clone=True); sync(); clone_seconds = time.monotonic()-before
                            if any(x.data_ptr() == y.data_ptr() for pp, qq in zip(base, CacheExtractor.tensors(cache)) for x, y in zip(pp, qq)):
                                raise RuntimeError('Generation cache aliases immutable base')
                            row = answer_instrumented(receiver, ph, cache, tid, seed['stream'], 4096, dest,
                                                      c['telemetry'], c['guard'], c['publish']); del cache
                        if row['answer_seed'] != seed['answer_seed'] or base_versions != [x._version for pair in base for x in pair]:
                            raise RuntimeError('Answer seed or immutable base cache drift')
                        source_time = 0. if form == 'P' else h['reasoning_seconds']
                        row.update(contract)
                        row.update(runtime_identity_sha256=digest(c['identity']), attempt_id=attempt,
                            draw_attempt=read(dest / 'draw_attempt.json'), mapping_seconds=mapping, splice_seconds=splicing,
                            native_prefill_seconds=prefill, historical_receiver_prefill_tokens=prefill_tokens,
                            cache_clone_seconds=clone_seconds, source_cache_reconstruction_seconds=source_prep if form != 'P' else 0.,
                            source_reasoning_seconds=source_time,
                            source_inclusive_estimate_seconds=None if row['answer_seconds'] is None else source_time+mapping+splicing+prefill+row['answer_seconds'],
                            separate_thinking_stage_requested=False,
                            thinking_open_token_in_answer=receiver.tokenizer.convert_tokens_to_ids('<think>') in row['answer_ids'],
                            immutable_base_cache_unchanged=True, clone_no_alias=True)
                        write(dest / 'answer.json', row); completed_draw(dest, contract); gc.collect()
                    del base; gc.collect(); torch.cuda.empty_cache()
                    if versions != [x._version for pair in (*sp, *tp, *prompt) for x in pair]:
                        raise RuntimeError('Saved historical pairs mutated')
                del sp, tp, prompt; gc.collect(); torch.cuda.empty_cache()
                write(root / 'progress.json', {'job_identity_sha256': digest(job), 'last_completed_task': tid,
                      'completed_tasks': index+1, 'expected_tasks': len(job['task_ids']), 'hidden_tests_loaded': False})
        result = {'job_identity_sha256': digest(job), 'experiment_id': job['experiment_id'], 'files': _answer_manifest(root, job, seeds),
                  'kl_files': [{'path': str(p.relative_to(root)), 'sha256': sha(p)} for p in sorted((root/'validation').glob('*/validation_kl.json'))],
                  'control_files': [{'path': str(p.relative_to(root)), 'sha256': sha(p)} for p in sorted((root/'attempts').glob('*/*/native_splice_control.json'))],
                  'hidden_tests_loaded': False, 'teacher_answer_prefix_supplied': False, 'confirmation_used': False}
        if job.get('validation_kl', True) and len(result['kl_files']) != len(job['task_ids']):
            raise ValueError('Incomplete validation KL task shard')
        bind(root / 'generation_complete.json', result)
        write(attempt_root / 'complete.json', {'job_identity_sha256': digest(job), 'wall_seconds': time.time()-started})
        return result
    except BaseException as exc:
        write(attempt_root / ('failure_' + str(time.time_ns()) + '.json'), {'exception': type(exc).__name__, 'error': str(exc),
              'traceback': traceback.format_exc(), 'job_identity_sha256': digest(job)})
        raise


def generation_closure(result_root, plan):
    """Verify exact whole-plan membership before private tests can be loaded."""
    root = Path(result_root); jobs = plan['jobs']; seen = set(); files = []; kl = []; receipts = []
    if len({j['job_id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job in immutable generation plan')
    for job in jobs:
        folder = root/'jobs'/job['job_id']; complete = read(folder/'generation_complete.json')
        if read(folder/'job.json') != job or complete['job_identity_sha256'] != digest(job) or complete['hidden_tests_loaded'] is not False:
            raise ValueError('Generation completion identity/phase differs')
        expected = {(t, name, s) for t in job['task_ids'] for name in
                    [job['arm']+'_'+f for f in job.get('forms', ['M','H'])] + (['D','P'] if job.get('include_controls', False) else []) for s in range(3)}
        actual = set()
        for item in complete['files']:
            p = folder/item['path']; row = read(p)
            if sha(p) != item['sha256'] or sha(p.parent/'complete.json') != item['receipt_sha256']:
                raise ValueError('Raw generated answer or receipt changed')
            contract = read(p.parent/'draw_identity.json'); validate_raw(row, contract)
            if contract['job_identity_sha256'] != digest(job):
                raise ValueError('Draw belongs to another generation job')
            key = (row['task_id'], row['condition'], row['seed_index'])
            if key in actual:
                raise ValueError('Duplicate task/condition/seed within job')
            actual.add(key)
            across = (job['step'], *key)
            if across in seen:
                raise ValueError('Duplicate task/condition/seed across job shards')
            seen.add(across); files.append({'path': str(p.relative_to(root)), 'sha256': item['sha256']})
        if actual != expected:
            raise ValueError('Incomplete or unexpected generation job population')
        for item in complete['kl_files']:
            p = folder/item['path']
            if sha(p) != item['sha256']:
                raise ValueError('Validation KL evidence changed')
            kl.append({'path': str(p.relative_to(root)), 'sha256': item['sha256']})
        controls = []
        for item in complete.get('control_files', []):
            p = folder/item['path']
            if sha(p) != item['sha256']:
                raise ValueError('Native splice control evidence changed')
            record = read(p)
            if record['passed']:
                controls.append(record['task_id'])
        if set(controls) != set(job['task_ids']):
            raise ValueError('Missing passed native splice control for a generation task')
        receipts.append({'job_id': job['job_id'], 'completion_sha256': sha(folder/'generation_complete.json')})
    return {'experiment_id': plan['experiment_id'], 'plan_sha256': digest(plan), 'jobs': receipts, 'files': files, 'kl_files': kl,
            'all_planned_generation_complete': True, 'hidden_tests_loaded': False}


def score_jobs(c, plan, *, shard_index=0, shard_count=1, owned_job_ids=None):
    """Run on a CPU helper only after closure. Raw answers are never overwritten."""
    from gearshift.coding_sandbox import extract, score
    repo = _repo(c); root = Path(c['root'])
    closure = generation_closure(root, plan)  # Must happen before even opening the private file.
    bind(root/'generation_closure.json', closure)
    if not read(repo/'evidence/coding_pilot_v1/sandbox_gate.json')['passed']:
        raise RuntimeError('Frozen isolated sandbox gate is not passed')
    private_path = repo/'data/coding_pilot_v1/private/coverage_generalization.json'
    if c.get('private_tests_sha256') and sha(private_path) != c['private_tests_sha256']:
        raise ValueError('Private test bytes differ from the frozen scorer identity')
    private = read(private_path)
    task_ids = {read(root/x['path'])['task_id'] for x in closure['files']}
    if len(task_ids) != 21 or not task_ids <= set(private):
        raise ValueError('Private scorer population differs from the 21 declared validation tasks')
    if not isinstance(shard_count, int) or not 1 <= shard_count or not 0 <= shard_index < shard_count:
        raise ValueError('Invalid scoring shard')
    if owned_job_ids is not None and (shard_count != 1 or not set(owned_job_ids) <= {j['job_id'] for j in plan['jobs']}):
        raise ValueError('Invalid owned scoring jobs')
    selected = [item for i, item in enumerate(closure['files']) if
                (Path(item['path']).parts[1] in owned_job_ids if owned_job_ids is not None else i % shard_count == shard_index)]
    shard = 'jobs_' + digest(sorted(owned_job_ids))[:16] if owned_job_ids is not None else f'shard_{shard_index:03d}_of_{shard_count:03d}'
    scored = []
    for item in selected:
        c['guard'](); path = root/item['path']; row = read(path); dest = path.parent/'score.json'
        if dest.exists():
            result = read(dest)
            if result['answer_sha256'] != item['sha256'] or result['generation_closure_sha256'] != digest(closure):
                raise ValueError('Existing score receipt belongs to other answers/closure')
        else:
            c['publish'](stage='coverage_v2_isolated_scoring', task_id=row['task_id'], condition=row['condition'], scored=len(scored), expected=len(selected))
            code = extract(row['answer_text']); score_result = score(code, private[row['task_id']], c['guard'])
            if score_result['category'] in ('sandbox_unavailable', 'missing_tests'):
                raise RuntimeError('Private scorer is unavailable; this is not a model failure')
            result = {'answer_sha256': item['sha256'], 'generation_closure_sha256': digest(closure),
                      'code': code, 'score': score_result, 'scorer_attempt_id': c['attempt_id'],
                      'hidden_tests_loaded_after_all_generation': True}
            bind(dest, result)
        scored.append({'path': str(dest.relative_to(root)), 'sha256': sha(dest), 'answer_path': item['path'], 'answer_sha256': item['sha256']})
        write(root/'scoring_shards'/(shard+'_progress.json'), {'scored': len(scored), 'expected': len(selected)})
    result = {'experiment_id': plan['experiment_id'], 'generation_closure_sha256': digest(closure), 'files': scored,
              'hidden_tests_loaded_after_all_generation': True, 'answers_rerolled': 0, 'confirmation_used': False}
    bind(root/'scoring_shards'/(shard+'.json'), result)
    if shard_count == 1 and owned_job_ids is None:
        finalize_scores(c, plan)
    return result


def finalize_scores(c, plan):
    """Verify disjoint CPU shard outputs and atomically publish the full score manifest."""
    root = Path(c['root']); closure = generation_closure(root, plan); files = []
    if read(root/'generation_closure.json') != closure:
        raise ValueError('Scoring closure identity changed')
    for item in closure['files']:
        path = (root/item['path']).parent/'score.json'; score_record = read(path)
        if (score_record['answer_sha256'] != item['sha256'] or
            score_record['generation_closure_sha256'] != digest(closure) or
            score_record['hidden_tests_loaded_after_all_generation'] is not True or
            score_record['score']['category'] in ('sandbox_unavailable', 'missing_tests')):
            raise ValueError('Completed score is not bound to the full generation closure')
        files.append({'path': str(path.relative_to(root)), 'sha256': sha(path),
                      'answer_path': item['path'], 'answer_sha256': item['sha256']})
    result = {'experiment_id': plan['experiment_id'], 'generation_closure_sha256': digest(closure), 'files': files,
              'hidden_tests_loaded_after_all_generation': True, 'answers_rerolled': 0, 'confirmation_used': False}
    bind(root/'scored_answer_manifest.json', result)
    return result
