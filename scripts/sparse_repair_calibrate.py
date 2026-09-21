#!/usr/bin/env python3
"""Bounded real-context CUDA calibration; no screen generation or private tests.

This runner is NOT numerically validated until its actual H200 output passes.
It consumes only the four frozen calibration histories. It creates no provider
resources and must be launched by the existing durable job/supervisor mechanism.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import signal
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import sha, write, seed_for

WEIGHTS_RECEIPT_SHA256 = '71dcbbe22c51dc4d9f8b976db58ac59fe221589340eb7dfd9620bee5a2b4e896'
MAPPER_SHA256 = '0e3caa7111ad8861f8e68e361ade6732e4dca5e587ca4f1ed8ba072a7ca9e386'
REQUIRED_IMPLEMENTATION = ('gearshift/sparse_repair.py', 'scripts/sparse_repair_calibrate.py',
    'gearshift/coding_inference.py', 'gearshift/core.py', 'gearshift/coding_gradients.py',
    'gearshift/coding_post_progress.py', 'gearshift/coding_control.py',
    'gearshift/coding_confirmation_sampling.py')
PROBE_POSITIONS = (0, 32, 128)
CALIBRATION_CAP = 129
ALTERED_SHAPE_TOLERANCE = {'maximum_absolute_logit_difference': .125, 'maximum_KL': .0001, 'top1_must_match': True}


def read(path):
    return json.loads(Path(path).read_text())


def safe_public_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or 'private' in path.relative_to(Path(root).resolve()).parts:
        raise ValueError('Input escapes public repository scope')
    return path


def validate_inputs(root, declaration_path, expected_sha, mapper_path, task_ids=None):
    """Before importing/loading models: hash bindings and exact saved-token checks."""
    root = Path(root).resolve()
    p = safe_public_path(root, declaration_path)
    if sha(p) != expected_sha:
        raise ValueError('Frozen declaration hash differs')
    d = read(p)
    if d.get('inputs'):
        protocol_path = safe_public_path(root, d['inputs']['protocol_path'])
        if sha(protocol_path) != d['inputs']['protocol_sha256']:
            raise ValueError('Pinned protocol bytes differ')
        if any(read(protocol_path)['protocol'].get(k) != v for k,v in d['protocol'].items()):
            raise ValueError('Copied frozen protocol differs from completed confirmation')
    if d.get('local_probe_positions', list(PROBE_POSITIONS)) != list(PROBE_POSITIONS):
        raise ValueError('Frozen local probe rule differs')
    cases = d['calibration_tasks']
    if len(cases) != 4 or len({r['task_id'] for r in cases}) != 4:
        raise ValueError('Exactly four unique frozen calibration cases required')
    screen = {r['task_id'] for r in d['screen_tasks']}
    if screen & {r['task_id'] for r in cases}:
        raise ValueError('Calibration overlaps screen')
    requested = {r['task_id'] for r in cases} if task_ids is None else set(task_ids)
    if not requested or not requested <= {r['task_id'] for r in cases}:
        raise ValueError('Only frozen calibration tasks may be executed')
    if sha(mapper_path) != MAPPER_SHA256:
        raise ValueError('Required primary ROTATING mapper bytes differ')
    result = []
    for row in cases:
        p = safe_public_path(root, row['history_path'])
        if sha(p) != row['history_sha256']:
            raise ValueError('Saved history hash differs')
        h = read(p)
        if h['task_id'] != row['task_id'] or h['bridge_ids'] != [151668]:
            raise ValueError('History task or boundary differs')
        prefix = h['prompt_ids'] + (h['reasoning_ids'][:-1] if h['natural_boundary'] else h['reasoning_ids'])
        if prefix != h['prefix_ids'] or len(prefix) != h['prefix_cache_length']:
            raise ValueError('Saved history structure differs; never regenerate')
        if h['natural_boundary'] and h['reasoning_ids'][-1] != 151668:
            raise ValueError('Natural closing-think identity differs')
        if row['historical_reasoning_positions'] != len(prefix) - len(h['prompt_ids']):
            raise ValueError('Frozen reasoning budget differs')
        if 'calibration_seeds' in d and d['calibration_seeds'][row['task_id']] != seed_for(row['task_id'], 0, 'sparse_repair_01_calibration_D'):
            raise ValueError('Frozen calibration seed differs')
        if row['task_id'] in requested:
            result.append((row, h))
    # Generation hosts must not accidentally contain the usual private input root.
    if (root / 'data/coding_pilot_v1/private').exists():
        raise ValueError('Private-test directory must not be present on generation worker')
    return d, result


def pass_control(metrics, bit_exact, *, altered_shape=False):
    if not altered_shape:
        return bool(bit_exact)
    return bool(metrics['max_abs'] <= .125 and metrics['kl'] <= .0001 and metrics['top1_equal'])


def memory_record(torch):
    free, total = torch.cuda.mem_get_info()
    return {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'free_bytes': free, 'total_bytes': total,
            'warning_only': True, 'not_a_dispatch_veto': True}


def tensor_bytes(pairs):
    return sum(x.numel() * x.element_size() for pair in pairs for x in pair)


def fingerprints(pairs):
    import torch
    return [[hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
             for t in pair] for pair in pairs]


def timed(function):
    from gearshift.coding_inference import sync
    sync(); begin = time.monotonic(); result = function(); sync()
    return result, time.monotonic() - begin


def runtime_gate():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import transformers
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
        raise ValueError('Exactly one visible H200 required')
    if str(torch.__version__) != '2.8.0+cu128' or torch.version.cuda != '12.8' or transformers.__version__ != '4.57.6':
        raise ValueError('Pinned CUDA/Torch/Transformers runtime differs')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'transformers': transformers.__version__,
            'gpu': torch.cuda.get_device_name(0), 'BF16': True, 'TF32': False, 'deterministic_algorithms': True,
            'attention': 'sdpa'}


def verify_weights(receipt_path):
    """No downloads/substitutions: independently hash all 22 pinned snapshot shards."""
    if sha(receipt_path) != WEIGHTS_RECEIPT_SHA256:
        raise ValueError('Historical weight identity receipt differs')
    receipt = read(receipt_path)
    rows = receipt['shards']
    if len(rows) != 22 or {r['role'] for r in rows} != {'source', 'receiver'}:
        raise ValueError('Pinned 17+5 shard inventory required')
    checked = []
    for row in rows:
        p = Path(row['path'])
        if not p.is_file() or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Frozen model shard bytes unavailable or changed: ' + str(p))
        checked.append({k: row[k] for k in ('role', 'path', 'bytes', 'sha256')})
    return {'receipt_sha256': sha(receipt_path), 'files': checked, 'actual_bytes_rehashed': True,
            'total_bytes': sum(r['bytes'] for r in rows)}


def generate_calibration_D(backend, h, native, folder, declaration_sha, guard):
    """Save every sampled token and RNG state; restart reconstructs exact progress."""
    import torch
    from gearshift.core import CacheInjector
    from gearshift.coding_inference import sample, sync
    path = folder / 'native_D_progress.json'
    seed = seed_for(h['task_id'], 0, 'sparse_repair_01_calibration_D')
    identity = {'task_id': h['task_id'], 'declaration_sha256': declaration_sha, 'seed': seed,
                'cap': CALIBRATION_CAP, 'kind': 'calibration_only_not_screen'}
    rng = torch.Generator(device=backend.device).manual_seed(seed)
    state = read(path) if path.exists() else {**identity, 'answer_ids': [], 'complete': False, 'eos': False}
    if any(state.get(k) != v for k, v in identity.items()):
        raise ValueError('Calibration resume identity differs')
    tokens = state['answer_ids']
    if state['complete']:
        state['answer_text'] = backend.tokenizer.decode(tokens, skip_special_tokens=False)
        state['probe_positions_available'] = [p for p in PROBE_POSITIONS if p < len(tokens)]
        state['answer_limit_note'] = '129 tokens maximum; calibration trace only, not complete-answer quality'
        write(folder / 'native_D_trace.json', state)
        return state
    cache = CacheInjector.create(native, clone=True)
    out = backend.forward(h['bridge_ids'], cache)
    for token in tokens:
        guard(); out = backend.forward([token], out.past_key_values)
    if 'rng_state' in state:
        rng.set_state(torch.tensor(state['rng_state'], dtype=torch.uint8))
    sync(); begin = time.monotonic()
    if not state['complete']:
        for _ in range(len(tokens), CALIBRATION_CAP):
            guard(); token = sample(out.logits, rng); tokens.append(token)
            eos = token in backend.eos
            state.update(answer_ids=tokens, rng_state=rng.get_state().cpu().tolist(), eos=eos,
                         complete=eos or len(tokens) == CALIBRATION_CAP,
                         measured_generation_segment_seconds=time.monotonic() - begin)
            write(path, state)
            if state['complete']:
                break
            out = backend.forward([token], out.past_key_values)
    sync(); state['answer_text'] = backend.tokenizer.decode(tokens, skip_special_tokens=False)
    state['probe_positions_available'] = [p for p in PROBE_POSITIONS if p < len(tokens)]
    state['answer_limit_note'] = '129 tokens maximum; calibration trace only, not complete-answer quality'
    write(folder / 'native_D_trace.json', state)
    del cache, out
    return state


def replay(backend, pairs, ids, positions, guard):
    import torch
    from gearshift.core import CacheInjector, CacheExtractor
    cache = CacheInjector.create(pairs, clone=True)
    if not all(a.data_ptr() != b.data_ptr() for pa, pb in zip(pairs, CacheExtractor.tensors(cache)) for a, b in zip(pa, pb)):
        raise ValueError('Mutable decode cache aliases immutable backing')
    logits = {}
    initial = cache.get_seq_length()
    for step, token in enumerate(ids):
        guard()
        if cache.get_seq_length() != initial + step:
            raise ValueError('Repeated/skipped token or position drift')
        out = backend.forward([token], cache); cache = out.past_key_values
        if cache.get_seq_length() != initial + step + 1:
            raise ValueError('Wrong post-forward cache length')
        if step in positions:
            logits[step] = out.logits.detach().clone()
    del out, cache
    return logits


def compare(a, b, altered_shape=False):
    import torch
    from gearshift.coding_inference import logit_metrics
    rows = []
    if set(a) != set(b):
        raise ValueError('Control probe coverage differs')
    for position in sorted(a):
        metric = logit_metrics(a[position], b[position]); exact = torch.equal(a[position], b[position])
        rows.append({'answer_position': position, **metric, 'bit_exact': exact,
                     'passed': pass_control(metric, exact, altered_shape=altered_shape)})
    return {'positions': rows, 'altered_shape': altered_shape, 'passed': all(r['passed'] for r in rows)}


def make_same_query_probe(receiver, native, mapped, layout, task_id, seed, records):
    """Inspect identical current queries on native D's own fixed trace only."""
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from gearshift.sparse_repair import native_reasoning_mass, select_indices, intervene, SelectionIdentity
    sdpa = ALL_ATTENTION_FUNCTIONS['sdpa']
    def probe(layer, step, query, live, mask, scaling):
        if step not in PROBE_POSITIONS:
            return
        module = receiver.model.model.layers[layer].self_attn
        reference = sdpa(module, query, live[0], live[1], mask, dropout=0., scaling=scaling)[0]
        mass = native_reasoning_mass(query, native[layer][0], live[0], layout,
                    scaling=scaling, query_position=live[0].shape[-2]-1, attention_mask=mask)
        for fraction in (.05, .10, .25):
            indices = select_indices(mass, layout, fraction, identity=SelectionIdentity(task_id, seed, layer, step))
            retained = mass.gather(1, indices-layout.prompt_length).sum(1)
            for mode in ('N', 'M', 'R'):
                changed = intervene(live, native[layer], mapped[layer], layout, indices, mode)
                altered_mask = mask
                if mode in ('N', 'M'):
                    allowed = changed.allowed.repeat_interleave(query.shape[1]//live[0].shape[1], dim=1)
                    altered_mask = allowed if mask is None else (mask & allowed if mask.dtype==torch.bool else mask.masked_fill(~allowed, -torch.inf))
                out = sdpa(module, query, changed.keys, changed.values, altered_mask, dropout=0., scaling=scaling)[0]
                delta = (out.float()-reference.float())
                records.append({'layer':layer, 'answer_position':step, 'mode':mode, 'fraction':fraction,
                    'identical_prefix_and_current_query':True, 'attention_output_max_abs':float(delta.abs().max()),
                    'attention_output_relative_l2':float(delta.norm()/reference.float().norm().clamp_min(1e-30)),
                    'selected_mass_sum_per_kv_group':retained.cpu().tolist(),
                    'all_reasoning_mass_sum_per_kv_group':mass.sum(1).cpu().tolist(),
                    'reference_query_source':'calibration D own sampled answer prefix'})
    return probe


def run_case(c, row, h):
    import torch
    from gearshift.core import CacheExtractor
    from gearshift.coding_post_progress import splice
    from gearshift.sparse_repair import HistoryLayout
    from gearshift.sparse_repair import SparseRepairController
    folder = c['output'] / h['task_id'].replace('/', '__'); folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        done = read(folder / 'complete.json')
        if done['declaration_sha256'] != c['declaration_sha'] or read(folder / 'calibration.json').get('implementation_hashes') != c['implementation_hashes']:
            raise ValueError('Complete case identity differs')
        if sha(folder / 'calibration.json') != done['calibration_sha256']:
            raise ValueError('Completed calibration bytes differ')
        return read(folder / 'calibration.json')
    torch.cuda.reset_peak_memory_stats(); start = time.monotonic(); timing = {}
    c['guard'](); write(folder / 'status.json', {'stage': 'cache_reconstruction', 'task_id': h['task_id']})
    out, timing['source_cache_reconstruction_seconds'] = timed(lambda: c['source'].prefill_chunked(h['prefix_ids']))
    source_pairs = CacheExtractor.tensors(out.past_key_values); del out
    source_bytes = tensor_bytes(source_pairs)
    mapped, timing['mapper_seconds'] = timed(lambda: c['mapper'](source_pairs))
    del source_pairs; gc.collect(); torch.cuda.empty_cache()
    out, timing['native_cache_construction_seconds'] = timed(lambda: c['receiver'].prefill_chunked(h['prefix_ids']))
    native = CacheExtractor.tensors(out.past_key_values); del out
    out, timing['native_prompt_prefill_seconds'] = timed(lambda: c['receiver'].prefill_chunked(h['prompt_ids']))
    prompt = CacheExtractor.tensors(out.past_key_values); del out
    hybrid, timing['hybrid_splice_seconds'] = timed(lambda: splice(prompt, mapped, len(h['prompt_ids'])))
    native_splice, timing['native_native_splice_seconds'] = timed(lambda: splice(prompt, native, len(h['prompt_ids'])))
    before, timing['backing_fingerprint_before_seconds'] = timed(lambda: [fingerprints(x) for x in (native, mapped, prompt, hybrid, native_splice)])
    layout = HistoryLayout(len(h['prompt_ids']), len(h['prefix_ids']))
    dtrace, timing['D_calibration_trace_wall_seconds'] = timed(lambda: generate_calibration_D(c['receiver'], h, native, folder, c['declaration_sha'], c['guard']))
    positions = [p for p in PROBE_POSITIONS if p < len(dtrace['answer_ids'])]
    ids = h['bridge_ids'] + dtrace['answer_ids'][:max(positions)]
    controller = SparseRepairController(c['receiver'].model, native, mapped, layout,
                  mode='disabled', fraction=.1, task_id=h['task_id'], draw_seed=dtrace['seed'], trace_enabled=False, profile=False)
    baseline, timing['dense_native_replay_seconds'] = timed(lambda: replay(c['receiver'], native, ids, positions, c['guard']))
    controls = {}
    same_query_probes = []
    controller.probe_callback = make_same_query_probe(c['receiver'], native, mapped, layout, h['task_id'], dtrace['seed'], same_query_probes)
    with controller:
        hooked, timing['disabled_hook_replay_seconds'] = timed(lambda: replay(c['receiver'], native, ids, positions, c['guard']))
        controls['unmodified_hook_vs_deployed_dense'] = compare(baseline, hooked)
        controller.probe_callback = None
        write(folder / 'same_query_attention_probes.json', same_query_probes)
        controller.set_condition('N', fraction=1., reset=True)
        full_n, timing['N100_replay_seconds'] = timed(lambda: replay(c['receiver'], native, ids, positions, c['guard']))
        controls['N100_vs_dense_native'] = compare(baseline, full_n)
        controller.set_condition('disabled', reset=True)
        dense_h, timing['dense_hybrid_replay_seconds'] = timed(lambda: replay(c['receiver'], hybrid, ids, positions, c['guard']))
        controller.set_condition('R', fraction=0., reset=True)
        zero_r, timing['R0_replay_seconds'] = timed(lambda: replay(c['receiver'], hybrid, ids, positions, c['guard']))
        controls['R0_vs_existing_H'] = compare(dense_h, zero_r)
        controller.set_condition('disabled', reset=True)
        nn, timing['native_splice_replay_seconds'] = timed(lambda: replay(c['receiver'], native_splice, ids, positions, c['guard']))
        controller.set_condition('R', fraction=1., reset=True)
        full_r, timing['R100_replay_seconds'] = timed(lambda: replay(c['receiver'], hybrid, ids, positions, c['guard']))
        controls['R100_vs_matched_native_native_splice'] = compare(nn, full_r)
    rounding = compare(baseline, nn, altered_shape=True)
    del hooked, full_n, dense_h, zero_r, nn, full_r
    # Output-distribution probes share exact token prefixes, but intervened
    # later-layer queries may differ; same-query attention probes are separate.
    probes = []; traces = []
    for mode in (('N', 'M', 'R') if all(x['passed'] for x in controls.values()) else ()):
        for fraction in (.05, .10, .25):
            c['guard']()
            controller = SparseRepairController(c['receiver'].model, native, mapped, layout,
                          mode=mode, fraction=fraction, task_id=h['task_id'], draw_seed=dtrace['seed'],
                          trace_enabled=True, profile=True)
            with controller:
                logits, elapsed = timed(lambda: replay(c['receiver'], native if mode == 'N' else hybrid, ids, positions, c['guard']))
            detail = compare(baseline, logits, altered_shape=True)
            probes.append({'mode': mode, 'fraction': fraction, 'matched_prefix_not_identical_later_layer_query': True,
                           'positions': detail['positions'], 'wall_seconds': elapsed})
            traces.append({'mode': mode, 'fraction': fraction, 'summary': controller.summary(), 'records': controller.trace_records()})
            del logits
            write(folder / 'local_distribution_probes.json', probes)
            write(folder / 'selected_budget_traces.json', traces)
    # Short same-semantics repeated whole-run timing, with bulky tracing disabled.
    timing_repeats = []
    timing_ids = ids[:min(16, len(ids))]
    for mode, frac, base in ([('disabled', .1, hybrid), ('R', .1, hybrid)] if all(x['passed'] for x in controls.values()) else []):
        for repeat in range(4):
            stage_profile = repeat == 3
            controller = SparseRepairController(c['receiver'].model, native, mapped, layout,
                          mode=mode, fraction=frac, task_id=h['task_id'], draw_seed=dtrace['seed'],
                          trace_enabled=False, profile=stage_profile)
            with controller:
                logits, elapsed = timed(lambda: replay(c['receiver'], base, timing_ids, (0,), c['guard']))
            timing_repeats.append({'condition': mode, 'fraction': frac, 'repeat': repeat,
                'decode_steps': len(timing_ids), 'wall_seconds': elapsed, 'stage_profile':stage_profile, 'component_profile': controller.summary()})
            del logits
    after, timing['backing_fingerprint_after_seconds'] = timed(lambda: [fingerprints(x) for x in (native, mapped, prompt, hybrid, native_splice)])
    intact = before == after
    result = {'task_id': h['task_id'], 'history_sha256': row['history_sha256'], 'declaration_sha256': c['declaration_sha'],
              'implementation_hashes': c['implementation_hashes'], 'runtime': c['runtime'],
              'controls': controls, 'all_same_path_controls_passed': all(x['passed'] for x in controls.values()),
              'independent_prompt_shape_rounding': rounding, 'backing_caches_unchanged': intact,
              'history_length': layout.history_length, 'prompt_length': layout.prompt_length,
              'probe_positions': positions, 'missing_declared_probe_positions': sorted(set(PROBE_POSITIONS)-set(positions)),
              'timing': timing, 'timing_repeats': timing_repeats, 'actual_calibration_wall_seconds': time.monotonic()-start,
              'same_query_attention_probe_count':len(same_query_probes),
              'memory': memory_record(torch), 'backing_cache_bytes_measured_tensor_storage': {
                  'source_peak_then_freed': source_bytes, 'native': tensor_bytes(native), 'mapped': tensor_bytes(mapped),
                  'native_prompt': tensor_bytes(prompt), 'hybrid': tensor_bytes(hybrid), 'native_native_splice': tensor_bytes(native_splice)},
              'passed': all(x['passed'] for x in controls.values()) and intact,
              'limitations': ['Masked dense kernels; no sparse latency savings claimed.',
                             'Calibration trace is limited to129 sampled tokens, not complete-answer quality.',
                             'Next-token distribution probes use common prefixes; later-layer queries can differ.',
                             'Independent prompt-prefill rounding is reported separately, never a tolerance for same-shape mismatch.']}
    write(folder / 'calibration.json', result)
    write(folder / 'complete.json', {'declaration_sha256': c['declaration_sha'], 'calibration_sha256': sha(folder/'calibration.json'), 'passed': result['passed']})
    del native, mapped, prompt, hybrid, native_splice, baseline, controller
    gc.collect(); torch.cuda.empty_cache()
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--declaration', required=True)
    p.add_argument('--declaration-sha256', required=True)
    p.add_argument('--mapper', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--task-id', action='append')
    p.add_argument('--maximum-wall-seconds', type=float, default=7200)
    p.add_argument('--weights-receipt', default='evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/resources/primary_pair01/model_weights_verified.json')
    args = p.parse_args(argv); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def guard():
        if stop_requested or time.monotonic()-started >= args.maximum_wall_seconds:
            raise TimeoutError('Bounded calibration wall budget reached; retain resume state')
    if not 0 < args.maximum_wall_seconds <= 7200:
        raise ValueError('Calibration wall cap must be within7200seconds')
    try:
        d, cases = validate_inputs(ROOT, args.declaration, args.declaration_sha256, args.mapper, args.task_id)
        write(output/'execution_contract.json', {'declaration_sha256': args.declaration_sha256, 'mapper_sha256': MAPPER_SHA256,
            'calibration_task_ids': [r['task_id'] for r,h in cases], 'no_screen_generation': True,
            'probe_positions': PROBE_POSITIONS, 'calibration_token_cap': CALIBRATION_CAP,
            'same_shape_gate': 'bit exact logits', 'altered_shape_only_tolerance': ALTERED_SHAPE_TOLERANCE,
            'implementation_hashes': {p:sha(ROOT/p) for p in REQUIRED_IMPLEMENTATION},
            'maximum_wall_seconds': args.maximum_wall_seconds})
        runtime = runtime_gate(); write(output/'runtime.json', runtime)
        weights, elapsed = timed(lambda: verify_weights(ROOT/args.weights_receipt))
        weights['verification_seconds'] = elapsed; write(output/'weights_verified.json', weights); guard()
        import torch
        from gearshift.coding_inference import Backend
        from gearshift.coding_gradients import AffineMapper
        protocol = read(ROOT/'configs/coding_pilot_v1/confirmation_01/protocol.json')['protocol']
        source, source_load = timed(lambda: Backend(protocol['models']['source'])); guard()
        receiver, receiver_load = timed(lambda: Backend(protocol['models']['receiver'])); guard()
        if source.tokenizer.backend_tokenizer.to_str() != receiver.tokenizer.backend_tokenizer.to_str():
            raise ValueError('Pinned tokenizers differ')
        mapper = AffineMapper(source, receiver); payload = torch.load(args.mapper, map_location='cpu', weights_only=True)
        mapper.load_state_dict(payload['state_dict'], strict=True); mapper.requires_grad_(False); del payload
        write(output/'model_setup.json', {'source_load_seconds': source_load, 'receiver_load_seconds': receiver_load})
        context = {'source': source, 'receiver': receiver, 'mapper': mapper, 'output': output,
                   'declaration_sha': args.declaration_sha256, 'guard': guard, 'runtime':runtime,
                   'implementation_hashes':{p:sha(ROOT/p) for p in REQUIRED_IMPLEMENTATION}}
        results = []
        with torch.no_grad():
            for row, h in cases:
                guard(); results.append(run_case(context, row, h))
                write(output/'summary.json', {'cases': results, 'complete': False, 'passed': False})
                if not results[-1]['passed']:
                    raise RuntimeError('Actual-model calibration control failed; screen is blocked')
        write(output/'summary.json', {'cases': results, 'complete': True, 'passed': all(r['passed'] for r in results),
                                      'executed_case_count': len(results), 'full_calibration_case_count':4,
                                      'all_four_executed_here':len(results)==4, 'wall_seconds':time.monotonic()-started})
    except BaseException as exc:
        write(output/'failure.json', {'error': str(exc), 'exception': type(exc).__name__, 'traceback': traceback.format_exc(),
                                     'wall_seconds':time.monotonic()-started, 'screen_authorized':False})
        raise


if __name__ == '__main__':
    main()
