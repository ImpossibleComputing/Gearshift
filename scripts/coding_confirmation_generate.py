#!/usr/bin/env python3
"""Public-input-only confirmation generation with deterministic durable job claims.

Workers consume available reasoning and answer jobs, then exit when no work is
ready. A worker completing the last source history can finish its descendants;
GPU allocations are never kept alive solely to wait on a console or queue.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, digest, seed_for, sha, write
from scripts.coding_coverage_v2_worker import claim_job

CONDITIONS = ['A', 'B', 'D', 'P', 'FIXED_M', 'ROTATING_M', 'FIXED_H', 'ROTATING_H']
MODELS = {'source': {'id': 'Qwen/Qwen3-32B', 'revision': '9216db5781bf21249d130ec9da846c4624c16137'},
          'receiver': {'id': 'Qwen/Qwen3-8B', 'revision': 'b968826d9c46dd6066d109eabc6255188de91218'}}
ANSWER_STREAMS = ['answer_small', 'coverage_answer_1', 'coverage_answer_2']
REQUIRED_IMPLEMENTATION = {
    'scripts/coding_confirmation_generate.py', 'gearshift/coding_confirmation_sampling.py',
    'gearshift/coding_inference.py', 'gearshift/core.py', 'gearshift/coding_gradients.py',
    'gearshift/coding_post_progress.py', 'scripts/coding_coverage_v2_worker.py',
    'gearshift/coding_control.py', 'scripts/coding_confirmation_runtime.py', 'gearshift/coding_confirmation_lease.py'}


def read(path):
    return json.loads(Path(path).read_text())


def scoped_file(root, relative):
    root = Path(root).resolve(); path = Path(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError('Input path must remain inside its declared root')
    result = root / path
    cursor = result
    while cursor != root:
        if cursor.is_symlink(): raise ValueError('Scientific input cannot be a symlink')
        cursor = cursor.parent
    if not result.is_file(): raise ValueError('Scientific input is missing: ' + str(relative))
    return result


def validate_seeds(seeds, task_ids):
    if seeds.get('namespace') != 'coding_pilot_v1' or seeds.get('source_stream') != 'source_reasoning' or seeds.get('small_reasoning_stream') != 'small_reasoning':
        raise ValueError('Frozen reasoning seed streams differ')
    for tid in task_ids:
        row = seeds['tasks'][tid]
        if row['source_reasoning_seed'] != seed_for(tid, 0, 'source_reasoning') or row['small_reasoning_seed'] != seed_for(tid, 0, 'small_reasoning'):
            raise ValueError('Frozen reasoning seed values differ')
        expected = [{'seed_index': i, 'stream': stream, 'answer_seed': seed_for(tid, 0, stream)}
                    for i, stream in enumerate(ANSWER_STREAMS)]
        if row['answers'] != expected:
            raise ValueError('Exactly three worker-independent answer seeds required')


def task_folder(c, tid):
    return c['top'] / 'primary' / 'tasks' / tid.replace('/', '__')


def validate_declaration(repo, path, expected):
    repo = Path(repo)
    if sha(scoped_file(repo, path)) != expected:
        raise ValueError('Frozen declaration changed')
    d = read(repo / path)
    if d['status'] != 'FROZEN' or d['generation_authorized_by_this_file'] is not True:
        raise ValueError('Draft declarations cannot authorize generation')
    if (type(d['task_count']) is not int or d['task_count'] <= 0 or
            d['task_count'] != len(d['task_ids']) or len(set(d['task_ids'])) != d['task_count'] or
            any(not re.fullmatch(r'(?:atcoder|leetcode|codeforces)/[A-Za-z0-9_]+', t) for t in d['task_ids'])):
        raise ValueError('Task population is not exact and unique')
    if d['primary_conditions'] != CONDITIONS or d['primary_answer_count'] != 24 * d['task_count']:
        raise ValueError('Undeclared primary condition/draw population')
    # Validate names before hashing any inputs: hashing opens file contents too.
    if any('private' in Path(name).parts for name in (*d['inputs'], *d['implementation'])):
        raise ValueError('Private test files cannot be generation inputs')
    for rel, expected_hash in d['inputs'].items():
        if sha(scoped_file(repo, rel)) != expected_hash:
            raise ValueError('Frozen input changed: ' + rel)
    for rel, expected_hash in d['implementation'].items():
        if sha(scoped_file(repo, rel)) != expected_hash:
            raise ValueError('Frozen generation implementation changed: ' + rel)
    if not REQUIRED_IMPLEMENTATION <= d['implementation'].keys():
        raise ValueError('Generation implementation freeze is incomplete')
    if not d['scorer']['policy_sha256'] or not d['task_scope_resolution']:
        raise ValueError('Missing scorer freeze or task-scope resolution')
    for key in ['seeds_path', 'visible_path']:
        if d[key] not in d['inputs']:
            raise ValueError('Consumed input is not bound by declaration: ' + key)
    model_path = d.get('model_config_path', 'configs/coding_pilot_v1/pilot.json')
    if model_path not in d['inputs']:
        raise ValueError('Consumed model configuration is not hash-bound')
    models = read(scoped_file(repo, model_path))['models']
    for role, spec in MODELS.items():
        if any(models[role].get(k) != v for k, v in spec.items()) or models[role].get('dtype') != 'bfloat16' or models[role].get('frozen') is not True:
            raise ValueError('Frozen language model identity differs')
    policy = d['scorer'].get('policy_path')
    if policy not in d['inputs'] or d['inputs'][policy] != d['scorer']['policy_sha256']:
        raise ValueError('Scorer policy is not bound to declaration inputs')
    if d['scorer'].get('policy_identity_sha256') != digest(read(repo / policy)):
        raise ValueError('Scorer canonical policy identity differs from file identity')
    if not d['scorer'].get('implementation_hashes'):
        raise ValueError('Scorer implementation freeze missing')
    for rel, want in d['scorer']['implementation_hashes'].items():
        if d['implementation'].get(rel) != want and d['inputs'].get(rel) != want:
            raise ValueError('Scorer implementation is not bound')
    validate_seeds(read(repo / d['seeds_path']), d['task_ids'])
    for arm in ['FIXED', 'ROTATING']:
        cp = d['primary_checkpoints'][arm]
        if cp.get('step') != 1024 or not cp.get('actual_checkpoint_byte_proof'):
            raise ValueError('Final checkpoint or actual-byte verification missing')
        proof = cp['actual_checkpoint_byte_proof']
        if not isinstance(proof, str) or proof not in d['inputs']:
            raise ValueError('Actual checkpoint byte proof is not hash-bound')
        if cp['mapper_sha256'] not in {r['sha256'] for r in read(repo / proof)['files']}:
            raise ValueError('Actual checkpoint byte proof lacks final mapper hash')
    return d


def contract(c, tid, condition, seed, history_sha, checkpoint_sha=None, cohort='primary'):
    return {'experiment_id': c['declaration']['experiment_id'], 'cohort': cohort,
            'declaration_sha256': c['declaration_sha256'], 'task_id': tid,
            'condition': condition, 'seed_index': seed['seed_index'],
            'answer_seed': seed['answer_seed'], 'stream': seed['stream'],
            'history_sha256': history_sha, 'checkpoint_sha256': checkpoint_sha,
            'teacher_answer_prefix_supplied': False, 'endpoint': 1024 if checkpoint_sha else None}


def verify_sampler(folder, kind, task_id, stream, expected_conditioning, model_role, declaration_sha, cache_identity=None):
    """Validate a durable completed transaction without executing its output."""
    folder = Path(folder)
    identity = read(folder / 'identity.json'); identity_sha = digest(identity)
    if identity.get('sampler') != kind or identity.get('task_id') != task_id or identity.get('stream') != stream:
        raise ValueError('Sampler task/stream identity differs')
    if identity.get('seed') != seed_for(task_id, 0, stream) or identity.get('cap') != (24576 if kind == 'reasoning' else 4096):
        raise ValueError('Sampler seed or cap differs')
    if identity.get('conditioning') != expected_conditioning:
        raise ValueError('Sampler prompt/history conditioning differs')
    model = identity['model']; spec = MODELS[model_role]
    if model.get('model_id') != spec['id'] or model.get('revision') != spec['revision'] or model.get('declaration_sha256') != declaration_sha:
        raise ValueError('Sampler pinned model/declaration differs')
    if cache_identity is not None and model.get('cache_identity_sha256') != cache_identity:
        raise ValueError('Sampler cache identity differs')
    if (identity.get('prefill_chunk') != 512 or identity.get('continuation_forward_chunk') != 1 or
            identity.get('sampling') != {'temperature': .6, 'top_p': .95, 'top_k': 20} or
            identity.get('eos') != [151643, 151645] or identity.get('closing_think') != 151668):
        raise ValueError('Sampler numerical or decoding contract differs')
    state = read(folder / 'resume.json'); checksum = state.pop('resume_sha256', None)
    name = 'source_history.json' if kind == 'reasoning' else 'answer_record.json'
    record = read(folder / name); receipt = read(folder / 'complete.json')
    if checksum != digest(state) or state.get('state') != 'complete' or state.get('identity_sha256') != identity_sha:
        raise ValueError('Sampler transaction is incomplete or changed')
    if state.get('record') != record or state.get('record_sha256') != digest(record):
        raise ValueError('Sampler result and complete transaction differ')
    want = {'identity_sha256': identity_sha, 'record_sha256': digest(record),
            'record_file_sha256': sha(folder / name), 'record_file': name}
    if receipt != want or record.get('sampler_identity_sha256') != identity_sha:
        raise ValueError('Sampler completion receipt differs')
    timing = read(folder / 'completion_timing.json')
    overhead = timing.get('overhead', {})
    if (timing.get('schema') != 1 or timing.get('state') != 'complete' or
            timing.get('identity_sha256') != identity_sha or timing.get('record_sha256') != digest(record) or
            timing.get('active_seconds_unchanged') is not True or
            timing.get('this_receipt_persistence_excluded') is not True or
            timing.get('final_checkpoint_and_materialization_included_when_scope_complete') is not True or
            type(overhead.get('scope_complete')) is not bool or
            overhead.get('completion_receipt_file') != 'completion_timing.json' or
            overhead.get('subtract_from_active_seconds') is not False or
            overhead.get('model_only_latency_measured') is not False):
        raise ValueError('Sampler completion timing identity or scope differs')
    ids = record['reasoning_ids' if kind == 'reasoning' else 'answer_ids']
    rng_key = 'rng_after_reasoning' if kind == 'reasoning' else 'rng_final'
    if ids != state['tokens'] or record[rng_key] != state['rng_state'] or record['rng_initial'] != state['rng_initial']:
        raise ValueError('Sampler committed token/RNG state differs')
    if not ids or len(ids) > identity['cap'] or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError('Invalid committed sampler token population')
    terminal = ids[-1] == 151668 if kind == 'reasoning' else ids[-1] in (151643, 151645)
    if state.get('forwarded_tokens') != len(ids) - terminal:
        raise ValueError('Sampler completed cache position differs')
    if kind == 'answer' and (any(t in (151643, 151645) for t in ids[:-1]) or
            record.get('answer_ended_eos') is not terminal or record.get('answer_capped') is not (not terminal) or
            (not terminal and len(ids) != 4096)):
        raise ValueError('Answer stopping or cap differs')
    return record, identity


def cache_identity(expected):
    return digest({'history': expected['history_sha256'], 'condition': expected['condition'],
                   'checkpoint': expected['checkpoint_sha256'], 'cohort': expected['cohort']})


def verify_draw(folder, expected, history=None, require_receipt=False):
    folder = Path(folder)
    p = folder / 'answer.json'
    if not p.exists():
        return False
    row = read(p)
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError('Committed answer identity differs')
    if 'score' in row or 'code' in row:
        raise ValueError('Private scoring cannot modify a raw answer')
    if not row['immutable_base_cache_unchanged'] or not row['clone_no_alias']:
        raise ValueError('Missing cache isolation proof')
    if not row.get('answer_ids'):
        raise ValueError('Raw answer token IDs missing')
    if read(folder / 'draw_identity.json') != expected:
        raise ValueError('Draw identity file differs')
    if history is None:
        conditioning = read(folder / 'sampler/identity.json')['conditioning']
    else:
        conditioning = {'history_sha256': digest(history), 'prefix_ids': history['prefix_ids'], 'bridge_ids': history['bridge_ids']}
    original, identity = verify_sampler(folder / 'sampler', 'answer', expected['task_id'], expected['stream'], conditioning,
        'source' if expected['condition'] == 'A' else 'receiver', expected['declaration_sha256'], cache_identity(expected))
    if any(row.get(k) != v for k, v in original.items()) or row.get('runtime_identity_sha256') != identity['model']['runtime_identity_sha256']:
        raise ValueError('Raw answer changed its durable sampler result')
    receipt = {'contract_sha256': digest(expected), 'answer_sha256': sha(p),
               'sampler_complete_sha256': sha(folder / 'sampler/complete.json')}
    if require_receipt:
        if read(folder / 'draw_complete.json') != receipt: raise ValueError('Draw completion receipt differs')
    else:
        bind(folder / 'draw_complete.json', receipt)
    return True


def sample_answers(c, backend, tid, condition, history, base, *, history_sha,
                   checkpoint_sha=None, timing=None, cohort='primary'):
    import torch
    from gearshift.core import CacheInjector, CacheExtractor
    from gearshift.coding_inference import sync
    from gearshift.coding_confirmation_sampling import answer_durable
    timing = timing or {}
    folder = c['top'] / cohort / 'tasks' / tid.replace('/', '__') / condition
    for seed in c['seeds']['tasks'][tid]['answers']:
        identity = contract(c, tid, condition, seed, history_sha, checkpoint_sha, cohort)
        dest = folder / ('seed_' + str(seed['seed_index']))
        if verify_draw(dest, identity, history):
            continue
        bind(dest / 'draw_identity.json', identity)
        c['publish'](stage='answer_generation', task_id=tid, condition=condition, seed_index=seed['seed_index'])
        backend.sampling_identity = {**c['backend_identities'][backend.name],
            'cache_identity_sha256': cache_identity(identity)}
        versions = [x._version for pair in base for x in pair]
        sync(); started = time.monotonic()
        cache = CacheInjector.create(base, clone=True)
        sync(); clone_seconds = time.monotonic() - started
        if any(x.data_ptr() == y.data_ptr() for pp, qq in zip(base, CacheExtractor.tensors(cache))
               for x, y in zip(pp, qq)):
            raise ValueError('Private draw cache aliases the immutable base')
        row = answer_durable(backend, history, cache, tid, seed['stream'], 4096,
                             dest / 'sampler', c['telemetry'], c['guard'], c['publish'])
        del cache
        if row['answer_seed'] != seed['answer_seed'] or versions != [x._version for pair in base for x in pair]:
            raise ValueError('Seed or historical cache mutated')
        row = dict(row)  # Keep the immutable sampler transaction as its own record.
        row.update(identity)
        row.update(timing)
        row.setdefault('checkpoint_load_seconds', 0.)
        row.setdefault('checkpoint_load_receipt_path', None)
        row.setdefault('checkpoint_load_receipt_sha256', None)
        row.update(cache_clone_seconds=clone_seconds, immutable_base_cache_unchanged=True,
                   clone_no_alias=True, runtime_identity_sha256=c['runtime_sha256'],
                   attempt_id=c['attempt_id'], source_cost_divisor_for_single_output=1,
                   experimental_answer_draws_sharing_history=3)
        reasoning = timing.get('reasoning_seconds', 0)
        timing_complete = timing.get('reasoning_timing_complete', True) and row.get('sampler_timing', {}).get('complete') is True
        row['inference_timing_complete'] = timing_complete
        row['single_output_inference_seconds_kind'] = 'sum_of_recorded_stage_costs_single_output_estimate'
        row['single_output_inference_seconds'] = None if not timing_complete or reasoning is None or row.get('answer_seconds') is None else (
            reasoning + row['answer_seconds'] + timing.get('native_prefill_seconds', 0)
            + timing.get('mapping_seconds', 0) + timing.get('splice_seconds', 0) + clone_seconds)
        write(dest / 'answer.json', row)
        verify_draw(dest, identity, history)
        gc.collect()


def reasoning_job(c, tid, kind):
    from gearshift.core import CacheExtractor
    from gearshift.coding_confirmation_sampling import reason_durable
    source = kind == 'source'
    backend = c['source'] if source else c['receiver']
    stream = c['seeds']['source_stream'] if source else c['seeds']['small_reasoning_stream']
    visible = c['visible'][tid]
    folder = task_folder(c, tid) / ('large_history' if source else 'small_history')
    backend.sampling_identity = c['backend_identities'][backend.name]
    c['publish'](stage='source_reasoning' if source else 'small_reasoning', task_id=tid)
    h, cache = reason_durable(backend, visible['prompt_ids'], tid, stream, 24576,
                             folder, c['telemetry'], c['guard'], c['publish'])
    if h['prompt_ids'] != visible['prompt_ids']:
        raise ValueError('Source prompt differs from frozen public input')
    # This file is the immutable fan-out boundary. It contains no final answer.
    history_sha = sha(folder / 'source_history.json')
    bind(folder / 'history_ready.json', {'task_id': tid, 'sha256': history_sha,
         'declaration_sha256': c['declaration_sha256'], 'contains_final_answer_prefix': False})
    base = CacheExtractor.tensors(cache)
    sample_answers(c, backend, tid, 'A' if source else 'B', h, base, history_sha=history_sha,
                   timing={'reasoning_seconds': h.get('reasoning_seconds'),
                           'reasoning_timing_complete': h.get('sampler_timing', {}).get('complete') is True,
                           'reasoning_model': 'source' if source else 'receiver',
                           'native_prefill_seconds': 0., 'mapping_seconds': 0., 'splice_seconds': 0.,
                           'reasoning_resume_reconstruction_seconds': h.get('sampler_timing', {}).get('reconstruction_seconds', 0.),
                           'native_reasoning_state_used': True})
    del base, cache
    return {}


def checkpoint_for(c, arm, cohort):
    if cohort == 'primary': return c['declaration']['primary_checkpoints'][arm]
    if cohort != 'secondary': raise ValueError('Unknown generation cohort')
    cp = c['secondary_checkpoints'][arm]; repo = Path(c.get('repo_root', ROOT))
    manifest_path = scoped_file(repo, cp['manifest_path'])
    if sha(manifest_path) != cp['manifest_sha256']:
        raise ValueError('Replication manifest hash differs')
    m = read(manifest_path); identity = cp['checkpoint_identity']
    if (m.get('checkpoint_identity') != identity or m.get('checkpoint_identity_sha256') != digest(identity) or
            m.get('arm') != arm or m.get('step') != 1024 or m.get('schedule_position') != 1024 or
            m.get('complete_resumable') is not True or m.get('verified_roundtrip') is not True or
            identity.get('experiment_id') != c['declaration']['experiment_id'] + '/replication' or
            identity.get('arm') != arm or identity.get('configuration', {}).get('training_seed') != 20260919 or
            identity.get('selected_checkpoint_sha256') != '0b9700ffd9cb38c23bcfa1327181b32992b128c6b95078214328382f86b8d68f' or
            m.get('files', {}).get('mapper.pt', {}).get('sha256') != cp['mapper_sha256']):
        raise ValueError('Replication final checkpoint identity or completeness differs')
    if scoped_file(repo, cp['mapper_path']).parent != manifest_path.parent:
        raise ValueError('Replication mapper is outside its committed checkpoint')
    return cp


def validate_control_record(control, history, history_sha):
    metric = control['whole_native_splice_bridge']
    if (control.get('passed') is not True or control.get('whole_native_tensor_exact') is not True or
            metric.get('max_abs') != 0 or metric.get('kl') != 0 or metric.get('top1_equal') is not True or
            control.get('history_sha256') != history_sha or control.get('prompt_tokens') != len(history['prompt_ids']) or
            control.get('history_tokens') != len(history['prefix_ids']) or control.get('no_cropping') is not True or
            control.get('absolute_suffix_positions') != [len(history['prompt_ids']), len(history['prefix_ids']) - 1]):
        raise ValueError('Native/native control or absolute position evidence differs')
    for panel in ['whole_native_splice_bridge', 'partial_native_splice_bridge']:
        if any(not isinstance(control[panel].get(k), (int, float)) or not math.isfinite(control[panel][k])
               for k in ['max_abs', 'mean_abs', 'kl']):
            raise ValueError('Nonfinite native-control evidence')


def receiver_job(c, tid, cohort='primary'):
    import torch
    from gearshift.core import CacheExtractor, CacheInjector
    from gearshift.coding_post_progress import splice
    from gearshift.coding_inference import sync, logit_metrics
    if cohort not in ('primary', 'secondary'): raise ValueError('Unknown generation cohort')
    folder = c['top'] / cohort / 'tasks' / tid.replace('/', '__')
    h, hs, _ = verify_history(c, tid, 'source')
    if cohort == 'secondary':
        # No baseline reruns and no writes to the primary history/control tree.
        receipt = verify_job_receipt(c, tid, 'receiver')
        if receipt is None: raise ValueError('Primary native-control completion is not ready')
        control_path = scoped_file(c['top'], receipt['control_path'])
        if control_path.parent != (task_folder(c, tid) / 'controls').resolve():
            raise ValueError('Primary native control belongs to another task')
        validate_control_record(read(control_path), h, hs)
        for arm in ['FIXED', 'ROTATING']: checkpoint_for(c, arm, cohort)
        bind(folder / 'control_reuse.json', {'primary_control_path': receipt['control_path'],
             'primary_control_sha256': sha(control_path), 'source_history_sha256': hs,
             'declaration_sha256': c['declaration_sha256']})
    source, receiver, mapper = c['source'], c['receiver'], c['mapper']
    nprompt, n = len(h['prompt_ids']), len(h['prefix_ids'])
    c['guard'](); sync(); before = time.monotonic(); out = source.prefill_chunked(h['prefix_ids'])
    sp = CacheExtractor.tensors(out.past_key_values); del out; sync(); reconstruction = time.monotonic() - before
    tp = (); native_prefill = 0.
    if cohort == 'primary':
        c['guard'](); before = time.monotonic(); out = receiver.prefill_chunked(h['prefix_ids'])
        tp = CacheExtractor.tensors(out.past_key_values); del out; sync(); native_prefill = time.monotonic() - before
    c['guard'](); before = time.monotonic(); out = receiver.prefill_chunked(h['prompt_ids'])
    prompt = CacheExtractor.tensors(out.past_key_values); del out; sync(); prompt_prefill = time.monotonic() - before
    if cohort == 'primary':
        sliced = tuple(tuple(x[..., :nprompt, :] for x in pair) for pair in tp)
        whole = splice(sliced, tp, nprompt)
        exact = all(torch.equal(x, y) for p, q in zip(whole, tp) for x, y in zip(p, q))
        a = receiver.forward(h['bridge_ids'], CacheInjector.create(tp, clone=True)).logits.detach().clone()
        z = receiver.forward(h['bridge_ids'], CacheInjector.create(whole, clone=True)).logits.detach().clone()
        metric = logit_metrics(a, z)
        partial = splice(prompt, tp, nprompt)
        p = receiver.forward(h['bridge_ids'], CacheInjector.create(partial, clone=True)).logits.detach().clone()
        control = {'whole_native_tensor_exact': exact, 'whole_native_splice_bridge': metric,
                   'partial_native_splice_bridge': logit_metrics(a, p),
                   'prompt_tokens': nprompt, 'history_tokens': n,
                   'passed': exact and metric['max_abs'] == 0, 'history_sha256': hs,
                   'absolute_suffix_positions': [nprompt, n-1], 'no_cropping': True}
        bind(folder / 'controls' / (c['attempt_id'] + '.json'), control)
        validate_control_record(control, h, hs)
        del sliced, whole, a, z, p, partial
    versions = [x._version for pair in (*sp, *tp, *prompt) for x in pair]
    conditions = CONDITIONS[2:] if cohort == 'primary' else ['FIXED_M', 'ROTATING_M', 'FIXED_H', 'ROTATING_H']
    shift = c['declaration']['task_ids'].index(tid) % len(conditions)
    for condition in conditions[shift:] + conditions[:shift]:
        c['guard']()
        form = condition[-1]
        mapping = splicing = 0.
        checkpoint_load_seconds = 0.; checkpoint_load_path = None
        cp_sha = None; ph = h; history_sha = hs
        prefill = native_prefill if condition == 'D' else 0.
        if form in ('M', 'H'):
            arm = condition.split('_')[0]
            cp = checkpoint_for(c, arm, cohort)
            cp_sha = cp['mapper_sha256']
            mapper_key = (cohort, arm, cp_sha)
            if c.get('loaded_mapper') != mapper_key:
                sync(); checkpoint_started = time.monotonic()
                path = scoped_file(Path(c.get('repo_root', ROOT)), cp['mapper_path'])
                if sha(path) != cp_sha: raise ValueError('Frozen final mapper bytes differ')
                payload = torch.load(path, map_location='cpu', weights_only=True)
                if payload['arm'] != arm or payload['step'] != 1024: raise ValueError('Mapper endpoint differs')
                mapper.load_state_dict(payload['state_dict'], strict=True)
                c['loaded_mapper'] = mapper_key; del payload
                sync(); checkpoint_load_seconds = time.monotonic() - checkpoint_started
                checkpoint_load_path = folder / 'mapper_loads' / (c['attempt_id'] + '_' + condition + '.json')
                bind(checkpoint_load_path, {'task_id': tid, 'condition': condition, 'cohort': cohort,
                    'mapper_sha256': cp_sha, 'declaration_sha256': c['declaration_sha256'],
                    'checkpoint_load_seconds': checkpoint_load_seconds,
                    'scope': 'File hash verification, CPU deserialization, GPU state copy and synchronization.',
                    'charged_to_single_output_inference': False})
            sync(); before = time.monotonic(); mapped = mapper(sp); sync(); mapping = time.monotonic() - before
            if form == 'H':
                before = time.monotonic(); base = splice(prompt, mapped, nprompt); sync(); splicing = time.monotonic() - before
                del mapped; prefill = prompt_prefill
            else: base = mapped
        elif condition == 'D': base = tp
        else:
            ids = receiver.tokenizer.apply_chat_template([{'role': 'user', 'content': c['visible'][tid]['prompt']}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            rendered = receiver.tokenizer.decode(ids, skip_special_tokens=False)
            if '<think>\n\n</think>' not in rendered: raise ValueError('Non-thinking template changed')
            ph = {'prefix_ids': ids[:-1], 'bridge_ids': ids[-1:]}
            bind(folder / 'prompt_only_template.json', {'task_id': tid, 'declaration_sha256': c['declaration_sha256'],
                 'public_prompt_sha256': digest(c['visible'][tid]['prompt']), 'enable_thinking': False,
                 'prompt_ids': ids, 'rendered_template': rendered, **ph})
            history_sha = sha(folder / 'prompt_only_template.json')
            sync(); before = time.monotonic(); out = receiver.prefill_chunked(ph['prefix_ids'])
            base = CacheExtractor.tensors(out.past_key_values); del out; sync(); prefill = time.monotonic() - before
        sample_answers(c, receiver, tid, condition, ph, base, history_sha=history_sha, checkpoint_sha=cp_sha, cohort=cohort,
            timing={'reasoning_seconds': 0. if condition == 'P' else h.get('reasoning_seconds'),
                    'reasoning_timing_complete': True if condition == 'P' else h.get('sampler_timing', {}).get('complete') is True,
                    'reasoning_model': None if condition == 'P' else 'source',
                    'native_prefill_seconds': prefill, 'mapping_seconds': mapping, 'splice_seconds': splicing,
                    'checkpoint_load_seconds': checkpoint_load_seconds,
                    'checkpoint_load_receipt_path': str(checkpoint_load_path.relative_to(c['top'])) if checkpoint_load_path else None,
                    'checkpoint_load_receipt_sha256': sha(checkpoint_load_path) if checkpoint_load_path else None,
                    'source_cache_reconstruction_seconds': 0. if condition == 'P' else reconstruction})
        del base
        if form == 'M': del mapped
        gc.collect(); torch.cuda.empty_cache()
        if versions != [x._version for pair in (*sp, *tp, *prompt) for x in pair]:
            raise ValueError('Historical cache mutated')
    del sp, tp, prompt
    if cohort == 'secondary':
        return {'control_reuse_path': str((folder / 'control_reuse.json').relative_to(c['top']))}
    return {'control_path': str((folder / 'controls' / (c['attempt_id'] + '.json')).relative_to(c['top']))}


def verify_history(c, tid, kind):
    source = kind == 'source'
    folder = task_folder(c, tid) / ('large_history' if source else 'small_history')
    stream = 'source_reasoning' if source else 'small_reasoning'
    h, _ = verify_sampler(folder, 'reasoning', tid, stream,
        {'prompt_ids': c['visible'][tid]['prompt_ids']}, 'source' if source else 'receiver', c['declaration_sha256'])
    ready = {'task_id': tid, 'sha256': sha(folder / 'source_history.json'),
             'declaration_sha256': c['declaration_sha256'], 'contains_final_answer_prefix': False}
    if read(folder / 'history_ready.json') != ready:
        raise ValueError('History fan-out receipt differs')
    ids = h['reasoning_ids']; natural = ids[-1] == 151668; early = ids[-1] in (151643, 151645)
    if any(t in (151643, 151645, 151668) for t in ids[:-1]):
        raise ValueError('Reasoning history continues after termination')
    prefix = c['visible'][tid]['prompt_ids'] + (ids[:-1] if natural else ids)
    if (h['prompt_ids'] != c['visible'][tid]['prompt_ids'] or h['prefix_ids'] != prefix or
            h['bridge_ids'] != [151668] or h['prefix_cache_length'] != len(prefix) or
            h['natural_boundary'] is not natural or h['early_eos'] is not early or
            h['reasoning_capped'] is not (not natural and not early) or
            h['seed'] != seed_for(tid, 0, stream) or (not natural and not early and len(ids) != 24576)):
        raise ValueError('Reasoning handoff boundary/seed/cap differs')
    if any(k in h for k in ['teacher_answer', 'answer_ids', 'answer_text', 'code', 'score']):
        raise ValueError('Final-answer material entered a shared reasoning history')
    return h, ready['sha256'], folder


def verify_job_receipt(c, tid, kind):
    path = c['top'] / 'primary/jobs' / (kind + '_' + tid.replace('/', '__')) / 'complete.json'
    if not path.exists():
        return None
    row = read(path)
    if any(row.get(k) != v for k, v in {'experiment_id': c['declaration']['experiment_id'],
            'declaration_sha256': c['declaration_sha256'], 'task_id': tid, 'kind': kind}.items()):
        raise ValueError('Generation job completion identity differs')
    return row


def generation_closure(c, seal=True):
    """Seal exactly the frozen primary population; reads metadata, never scores.

Returns None while any expected job has no completed receipt. Once all jobs
claim completion, any missing/extra/changed artifact is a validity failure.
"""
    d = c['declaration']; top = Path(c['top'])
    validate_seeds(c['seeds'], d['task_ids'])
    jobs = [(tid, kind, verify_job_receipt(c, tid, kind)) for tid in d['task_ids']
            for kind in ['source', 'small', 'receiver']]
    if any(row is None for _, _, row in jobs): return None
    files = {}; answers = []; histories = []; expected_answers = set(); expected_histories = set()

    def include(path):
        path = Path(path)
        relative = str(path.relative_to(top)); path = scoped_file(top, relative)
        entry = {'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)}
        files[relative] = entry
        return entry

    def sampler_files(folder, kind):
        names = ['identity.json', 'resume.json', 'complete.json', 'completion_timing.json',
                 'source_history.json' if kind == 'reasoning' else 'answer_record.json']
        for name in names: include(folder / name)

    job_map = {(tid, kind): row for tid, kind, row in jobs}
    for tid, kind, row in jobs:
        include(top / 'primary/jobs' / (kind + '_' + tid.replace('/', '__')) / 'complete.json')
        runtime_path = scoped_file(top, row['runtime_path'])
        runtime = read(runtime_path)
        if digest(runtime) != row['runtime_identity_sha256']:
            raise ValueError('Worker runtime receipt differs')
        if (runtime.get('torch') != '2.8.0+cu128' or runtime.get('transformers') != '4.57.6' or
                runtime.get('attention') != 'sdpa' or runtime.get('dtype') != 'bfloat16' or
                'H200' not in runtime.get('gpu', '') or runtime.get('TF32') is not False or
                runtime.get('deterministic_algorithms') is not True):
            raise ValueError('Worker numerical runtime differs')
        include(runtime_path)
        setup_path = scoped_file(top, row['setup_path'])
        if setup_path != runtime_path.parent / 'model_setup.json':
            raise ValueError('Worker setup receipt is outside its runtime attempt')
        setup = read(setup_path)
        if (setup.get('declaration_sha256') != c['declaration_sha256'] or
                setup.get('runtime_identity_sha256') != row['runtime_identity_sha256'] or
                setup.get('charged_to_single_output_inference') is not False or
                set(setup.get('model_loading_seconds', {})) != {'source', 'receiver'} or
                any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in [*setup['model_loading_seconds'].values(), setup.get('mapper_initialization_seconds')])):
            raise ValueError('Worker model setup measurement identity differs')
        include(setup_path)
    for tid in d['task_ids']:
        large, large_sha, large_folder = verify_history(c, tid, 'source')
        small, small_sha, small_folder = verify_history(c, tid, 'small')
        for role, h, hs, folder in [('source', large, large_sha, large_folder), ('small', small, small_sha, small_folder)]:
            sampler_files(folder, 'reasoning'); include(folder / 'history_ready.json')
            expected_histories.add(str((folder / 'source_history.json').relative_to(top)))
            histories.append({'task_id': tid, 'kind': role, 'sha256': hs})
        folder = task_folder(c, tid)
        control_rel = job_map[(tid, 'receiver')]['control_path']
        control_path = scoped_file(top, control_rel)
        if control_path.parent != (folder / 'controls').resolve():
            raise ValueError('Receiver control path does not belong to task')
        control = read(control_path)
        validate_control_record(control, large, large_sha)
        include(control_path)
        ppath = folder / 'prompt_only_template.json'; p = read(ppath)
        if (p.get('task_id') != tid or p.get('declaration_sha256') != c['declaration_sha256'] or
                p.get('public_prompt_sha256') != digest(c['visible'][tid]['prompt']) or p.get('enable_thinking') is not False or
                not p.get('prompt_ids') or p.get('prefix_ids') != p['prompt_ids'][:-1] or
                p.get('bridge_ids') != p['prompt_ids'][-1:] or '<think>\n\n</think>' not in p.get('rendered_template', '')):
            raise ValueError('Prompt-only pinned-template identity differs')
        psha = sha(ppath); include(ppath)
        ph = {'prefix_ids': p['prefix_ids'], 'bridge_ids': p['bridge_ids']}
        for condition in CONDITIONS:
            h, hs = (small, small_sha) if condition == 'B' else ((ph, psha) if condition == 'P' else (large, large_sha))
            cp = d['primary_checkpoints'][condition.split('_')[0]]['mapper_sha256'] if '_' in condition else None
            for seed in c['seeds']['tasks'][tid]['answers']:
                expected = contract(c, tid, condition, seed, hs, cp)
                dest = folder / condition / ('seed_' + str(seed['seed_index']))
                if not verify_draw(dest, expected, h, require_receipt=True):
                    raise ValueError('Job completed without every declared answer')
                raw = read(dest / 'answer.json')
                load_path = raw.get('checkpoint_load_receipt_path')
                load_seconds = raw.get('checkpoint_load_seconds')
                if type(load_seconds) not in (int, float) or not math.isfinite(load_seconds) or load_seconds < 0:
                    raise ValueError('Checkpoint setup timing missing or invalid')
                if load_path is not None:
                    path = scoped_file(top, load_path); receipt = read(path)
                    if (path.parent != (folder / 'mapper_loads').resolve() or
                            sha(path) != raw.get('checkpoint_load_receipt_sha256') or
                            receipt.get('task_id') != tid or receipt.get('condition') != condition or
                            receipt.get('cohort') != 'primary' or receipt.get('mapper_sha256') != cp or
                            receipt.get('declaration_sha256') != c['declaration_sha256'] or
                            receipt.get('checkpoint_load_seconds') != load_seconds or
                            receipt.get('charged_to_single_output_inference') is not False):
                        raise ValueError('Mapper setup receipt does not bind this draw')
                    include(path)
                elif load_seconds != 0 or raw.get('checkpoint_load_receipt_sha256') is not None:
                    raise ValueError('Checkpoint loading was measured without a receipt')
                for name in ['draw_identity.json', 'answer.json', 'draw_complete.json']: include(dest / name)
                sampler_files(dest / 'sampler', 'answer')
                rel = str((dest / 'answer.json').relative_to(top)); expected_answers.add(rel)
                answers.append({'contract': expected, 'contract_sha256': digest(expected),
                                'path': rel, 'answer_sha256': files[rel]['sha256']})
    actual_answers = {str(p.relative_to(top)) for p in (top / 'primary').rglob('answer.json')}
    actual_histories = {str(p.relative_to(top)) for name in ['large_history', 'small_history']
                        for p in (top / 'primary/tasks').glob('*/' + name + '/source_history.json')}
    if actual_answers != expected_answers or actual_histories != expected_histories:
        raise ValueError('Undeclared or missing primary answer/history population')
    if len(answers) != d['primary_answer_count'] or len(answers) != 24 * d['task_count']:
        raise ValueError('Final primary answer population differs')
    closure = {'schema': 1, 'experiment_id': d['experiment_id'], 'declaration_sha256': c['declaration_sha256'],
               'task_ids': d['task_ids'], 'task_count': d['task_count'], 'conditions': CONDITIONS,
               'answer_count': len(answers), 'histories': histories, 'answers': answers,
               'files': [files[k] for k in sorted(files)], 'all_primary_generation_complete': True,
               'scoring_requires_this_complete_seal': True}
    closure['closure_sha256'] = digest(closure)
    if seal: bind(top / 'primary/generation_closure.json', closure)
    return closure


def initialize(c):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import transformers
    from gearshift.coding_inference import Backend, sync
    from gearshift.coding_gradients import AffineMapper
    d = validate_declaration(ROOT, c['plan']['declaration_path'], c['plan']['declaration_sha256'])
    if d['experiment_id'] != c['plan']['experiment_id'] or d.get('source_commit') != c['plan'].get('code_commit'):
        raise ValueError('Plan and frozen declaration experiment/source identities differ')
    if torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
        raise ValueError('Exactly one matching H200 must be visible')
    if torch.__version__ != '2.8.0+cu128' or torch.version.cuda != '12.8' or transformers.__version__ != '4.57.6':
        raise ValueError('Pinned numerical runtime differs')
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    models = read(ROOT / d.get('model_config_path', 'configs/coding_pilot_v1/pilot.json'))['models']
    c['declaration'] = d; c['declaration_sha256'] = c['plan']['declaration_sha256']
    c['seeds'] = read(ROOT / d['seeds_path'])
    rows = read(ROOT / d['visible_path'])
    if len({r['task_id'] for r in rows}) != len(rows): raise ValueError('Duplicate public task rows')
    c['visible'] = {r['task_id']: r for r in rows if r['task_id'] in d['task_ids']}
    if set(c['visible']) != set(d['task_ids']): raise ValueError('Public task coverage differs')
    if any(len(r['prompt_ids']) > 8192 for r in c['visible'].values()):
        raise ValueError('Oversized prompt requires explicit failure accounting; no cropping')
    runtime = {'torch': torch.__version__, 'transformers': transformers.__version__,
               'attention': 'sdpa', 'dtype': 'bfloat16', 'gpu': torch.cuda.get_device_name(0),
               'TF32': False, 'deterministic_algorithms': True}
    c['runtime_sha256'] = digest(runtime)
    bind(c['status_root'] / 'runtime.json', runtime)
    setup_started = time.time(); loads = {}
    for role in ('source', 'receiver'):
        c['guard'](); sync(); started = time.monotonic()
        c[role] = Backend(models[role]); sync()
        loads[role] = time.monotonic() - started
    c['guard']()
    if (c['source'].tokenizer.backend_tokenizer.to_str() != c['receiver'].tokenizer.backend_tokenizer.to_str() or
            c['source'].tokenizer.chat_template != c['receiver'].tokenizer.chat_template):
        raise ValueError('Pinned tokenizers differ')
    for row in c['visible'].values():
        ids = c['source'].tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                tokenize=True, add_generation_prompt=True, enable_thinking=True)
        if ids != row['prompt_ids']: raise ValueError('Frozen prompt/tokenizer mismatch')
    sync(); started = time.monotonic()
    c['mapper'] = AffineMapper(c['source'], c['receiver']).to('cuda:0')
    c['mapper'].requires_grad_(False)
    sync(); mapper_initialization = time.monotonic() - started
    bind(c['status_root'] / 'model_setup.json', {
        'declaration_sha256': c['declaration_sha256'], 'runtime_identity_sha256': c['runtime_sha256'],
        'started_epoch': setup_started, 'completed_epoch': time.time(),
        'model_loading_seconds': loads, 'mapper_initialization_seconds': mapper_initialization,
        'charged_to_single_output_inference': False})
    c['backend_identities'] = {s['id']: {'model_id': s['id'], 'revision': s['revision'],
        'runtime_identity_sha256': c['runtime_sha256'], 'declaration_sha256': c['declaration_sha256']}
        for s in models.values()}


def work(c, preference):
    import torch
    c['guard']()
    initialize(c)
    # Length is public. No observed correctness or score enters scheduling.
    tasks = sorted(c['declaration']['task_ids'], key=lambda t: (-len(c['visible'][t]['prompt_ids']), t))
    kinds = {'source': ['source', 'small', 'receiver'], 'small': ['small', 'source', 'receiver'],
             'receiver': ['receiver', 'source', 'small']}[preference]
    with torch.no_grad():
        while True:
            c['guard']()
            did_work = False
            for kind in kinds:
                for tid in tasks:
                    job_id = kind + '_' + tid.replace('/', '__')
                    complete = c['top'] / 'primary/jobs' / job_id / 'complete.json'
                    if verify_job_receipt(c, tid, kind) is not None: continue
                    if kind == 'receiver' and not (task_folder(c, tid) / 'large_history/history_ready.json').exists(): continue
                    with claim_job(c, job_id) as claimed:
                        if not claimed or verify_job_receipt(c, tid, kind) is not None: continue
                        c['guard'](); c['telemetry'].reset(job_id)
                        started = time.time()
                        if kind == 'receiver': extra = receiver_job(c, tid)
                        else: extra = reasoning_job(c, tid, kind)
                        bind(complete, {'experiment_id': c['declaration']['experiment_id'],
                            'declaration_sha256': c['declaration_sha256'], 'task_id': tid,
                            'kind': kind, 'worker_attempt': c['attempt_id'], 'finished_epoch': time.time(),
                            'wall_seconds': time.time() - started,
                            'runtime_identity_sha256': c['runtime_sha256'],
                            'runtime_path': str((c['status_root'] / 'runtime.json').relative_to(c['top'])),
                            'setup_path': str((c['status_root'] / 'model_setup.json').relative_to(c['top'])), **extra})
                        c['publish'](stage='job_complete', task_id=tid, kind=kind)
                        did_work = True
                        gc.collect(); torch.cuda.empty_cache()
                        break
                if did_work: break
            if not did_work:
                closure = generation_closure(c)
                c['publish'](state='complete', stage='primary_generation_sealed' if closure else 'no_ready_unclaimed_jobs',
                             all_primary_generation_complete=bool(closure), gpu_idle_release_requested=True)
                return


def main():
    from scripts.coding_confirmation_runtime import build_context
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True); p.add_argument('--worker-id', required=True)
    p.add_argument('--preference', choices=['source', 'small', 'receiver'], default='source')
    a = p.parse_args()
    c = None
    try:
        c = build_context(a.plan, 'confirmation_generation', a.worker_id)
        work(c, a.preference)
    except BaseException as exc:
        if c:
            c['telemetry'].failure(exc)
            write(c['status_root'] / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
                  'traceback': traceback.format_exc(), 'epoch': time.time()})
            c['publish'](state='failed', stage='generation_failed', error=str(exc))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)): raise
        raise SystemExit(65 if isinstance(exc, (ValueError, FloatingPointError, AssertionError)) else 1) from exc


if __name__ == '__main__': main()
