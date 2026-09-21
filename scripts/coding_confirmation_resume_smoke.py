#!/usr/bin/env python3
"""Bounded public-only H200 sampler restart proof; never reads benchmark records.

prepare is CPU-only. run starts fresh Python processes for baseline, each hard
exit, and resume. For a cross-host check run the phase commands individually,
copy the *whole* compact smoke folder before resume, then verify there. Model
weights and the original declared mapper are inputs, never copied into proof.
No numerical source or sampling policy is patched. A short smoke is evidence
for its exact cases/runtime only, not universal or long-context equivalence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, digest, sha, write

SCRIPT = 'scripts/coding_confirmation_resume_smoke.py'
DECLARATION = 'configs/coding_pilot_v1/confirmation_01/declaration.json'
DECLARATION_SHA = '16387d5361c795121929fd9244821f12ee005f9b36b2febddca5d9ef89f7536f'
TASK = 'public-synthetic-resume-smoke-even-integers-v1'
PROMPT = ('This is a public synthetic software smoke test. Reason carefully about '
          'how to list the first 120 positive even integers without missing any. '
          'Then write a self-contained Python program with one explicit list '
          'literal containing all 120 integers, followed by print(sum(values)). '
          'Do not use range, loops, comprehensions, or multiplication.')
CASES = ['source_reasoning', 'receiver_native', 'receiver_mapped', 'receiver_hybrid']
FROZEN_FILES = ['gearshift/coding_confirmation_sampling.py', 'gearshift/coding_inference.py',
                'gearshift/core.py', 'gearshift/coding_gradients.py',
                'gearshift/coding_post_progress.py', 'gearshift/coding_control.py',
                'gearshift/coding_confirmation_lease.py']
MODELS = {'source': {'id': 'Qwen/Qwen3-32B', 'revision': '9216db5781bf21249d130ec9da846c4624c16137'},
          'receiver': {'id': 'Qwen/Qwen3-8B', 'revision': 'b968826d9c46dd6066d109eabc6255188de91218'}}
EXIT_INTERRUPTED = 86


def read(path):
    return json.loads(Path(path).read_text())


def scoped(repo, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Expected a relative repository input')
    if 'private' in relative.parts:
        raise ValueError('Private inputs are forbidden')
    path = Path(repo) / relative
    if not path.resolve().is_relative_to(Path(repo).resolve()):
        raise ValueError('Input escapes repository')
    return path


def prepare(repo, output, *, cap=128, interrupt_after=1, mapper_arm='ROTATING'):
    repo, output = Path(repo), Path(output)
    if type(cap) is not int or not 2 <= cap <= 256:
        raise ValueError('Smoke cap must be between 2 and 256')
    if interrupt_after not in (1, 65, 129, 193) or interrupt_after >= cap:
        raise ValueError('Interruption must be an existing durable progress boundary below cap')
    if mapper_arm not in ('FIXED', 'ROTATING'):
        raise ValueError('Only an original primary mapper may be used')
    if sha(repo / DECLARATION) != DECLARATION_SHA:
        raise ValueError('Original primary declaration differs')
    declaration = read(repo / DECLARATION)
    hashes = {p: declaration['implementation'][p] for p in FROZEN_FILES}
    for p, expected in hashes.items():
        if sha(scoped(repo, p)) != expected:
            raise ValueError('Frozen numerical source differs: ' + p)
    model_path = declaration['model_config_path']
    if sha(scoped(repo, model_path)) != declaration['inputs'][model_path]:
        raise ValueError('Pinned model configuration differs')
    models = read(repo / model_path)['models']
    if any(any(models[r].get(k) != v for k, v in spec.items()) for r, spec in MODELS.items()):
        raise ValueError('Pinned model identities differ')
    checkpoint = declaration['primary_checkpoints'][mapper_arm]
    if checkpoint['step'] != 1024:
        raise ValueError('Only step 1024 is allowed')
    plan = {'schema': 1, 'purpose': 'synthetic_public_sampler_resume_smoke',
            'task_id': TASK, 'prompt': PROMPT, 'cases': CASES, 'cap': cap,
            'interrupt_after': interrupt_after, 'models': MODELS,
            'source_stream': 'source_reasoning', 'answer_stream': 'answer_small',
            'primary_declaration_path': DECLARATION,
            'primary_declaration_sha256': DECLARATION_SHA,
            'frozen_implementation': hashes, 'harness_sha256': sha(repo / SCRIPT),
            'mapper': {'arm': mapper_arm, 'step': 1024, 'path': checkpoint['mapper_path'],
                       'sha256': checkpoint['mapper_sha256'], 'bytes': checkpoint['mapper_bytes']},
            'comparison': 'Exact full tokens, generator states, checkpoint/reconstructed/final logits and KV bytes.',
            'interruption': 'os._exit(86) after fsynced unchanged-sampler checkpoint; no cache tensors saved.',
            'limitations': ['One public synthetic prompt and one seed per path.',
                           'At most 256 tokens per draw; not a long-context or universal resume proof.',
                           'No benchmark outputs, private tests, scoring, training, or model-policy changes.',
                           'An early terminal before the fixed cut is inconclusive and cannot pass; no rerolls.']}
    bind(output / 'plan.json', plan)
    return {'plan': str(output / 'plan.json'), 'plan_sha256': sha(output / 'plan.json')}


def validate_plan(repo, folder, expected):
    repo, folder = Path(repo), Path(folder)
    if sha(folder / 'plan.json') != expected:
        raise ValueError('Smoke plan hash differs')
    p = read(folder / 'plan.json')
    if (p['purpose'] != 'synthetic_public_sampler_resume_smoke' or p['task_id'] != TASK
            or p['prompt'] != PROMPT or p['cases'] != CASES or p['models'] != MODELS
            or p['source_stream'] != 'source_reasoning' or p['answer_stream'] != 'answer_small'
            or type(p['cap']) is not int or not 2 <= p['cap'] <= 256
            or p['interrupt_after'] not in (1, 65, 129, 193) or p['interrupt_after'] >= p['cap']):
        raise ValueError('Bounded synthetic protocol differs')
    if p['primary_declaration_path'] != DECLARATION or p['primary_declaration_sha256'] != DECLARATION_SHA:
        raise ValueError('Primary provenance differs')
    if sha(repo / DECLARATION) != DECLARATION_SHA:
        raise ValueError('Original primary declaration changed')
    d = read(repo / DECLARATION)
    if set(p['frozen_implementation']) != set(FROZEN_FILES):
        raise ValueError('Incomplete numerical code binding')
    for rel, expected_hash in p['frozen_implementation'].items():
        if expected_hash != d['implementation'][rel] or sha(scoped(repo, rel)) != expected_hash:
            raise ValueError('Frozen numerical source changed: ' + rel)
    if sha(repo / SCRIPT) != p['harness_sha256']:
        raise ValueError('Smoke harness changed')
    checkpoint = d['primary_checkpoints'][p['mapper']['arm']]
    if p['mapper'] != {'arm': p['mapper']['arm'], 'step': 1024, 'path': checkpoint['mapper_path'],
                       'sha256': checkpoint['mapper_sha256'], 'bytes': checkpoint['mapper_bytes']}:
        raise ValueError('Mapper identity differs')
    return p


def tensor_fingerprint(tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'bytes_sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def cache_fingerprint(cache):
    from gearshift.core import CacheExtractor
    rows = [[tensor_fingerprint(k), tensor_fingerprint(v)]
            for k, v in CacheExtractor.tensors(cache, clone=False)]
    return {'sequence_length': cache.get_seq_length(), 'layers': rows, 'sha256': digest(rows)}


class Observer:
    """Read-only caller instrumentation; forward arguments/results are unchanged."""
    def __init__(self, backend):
        self.backend, self.last = backend, None
        self.target = self.reconstructed = None
        self.forward_lengths = []
        original = backend.forward

        def forward(ids, *args, **kwargs):
            result = original(ids, *args, **kwargs)
            self.last = result
            self.forward_lengths.append(len(ids))
            if self.target is not None and result.past_key_values.get_seq_length() == self.target:
                if self.reconstructed is not None:
                    raise ValueError('Reconstruction boundary was observed twice')
                self.reconstructed = self.snapshot()
            return result

        backend.forward = forward

    def arm(self, length=None):
        self.target, self.reconstructed, self.forward_lengths = length, None, []

    def snapshot(self):
        if self.last is None:
            raise ValueError('No model forward observed')
        from gearshift.coding_confirmation_sampling import _fingerprint
        return {'cache': cache_fingerprint(self.last.past_key_values),
                'next_logits_sha256': _fingerprint(self.last.logits)}


class Telemetry:
    def __init__(self, folder):
        self.folder = Path(folder)

    def sample(self, **values):
        write(self.folder / 'telemetry.json', {'epoch': time.time(), **values})

    def failure(self, exc):
        write(self.folder / 'failure.json', {'epoch': time.time(), 'exception': type(exc).__name__,
                                           'message': str(exc)})


def runtime():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import transformers
    if torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
        raise ValueError('Exactly one matching H200 must be visible')
    if torch.__version__ != '2.8.0+cu128' or torch.version.cuda != '12.8' or transformers.__version__ != '4.57.6':
        raise ValueError('Pinned numerical runtime differs')
    free, total = torch.cuda.mem_get_info()
    if free < .95 * total:
        raise ValueError('Smoke requires an otherwise idle reserved H200 before model loading')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {'torch': torch.__version__, 'transformers': transformers.__version__,
            'attention': 'sdpa', 'dtype': 'bfloat16', 'gpu': torch.cuda.get_device_name(0),
            'TF32': False, 'deterministic_algorithms': True}


def allocation_guard(folder, lease_path, lease_sha256):
    from gearshift.coding_confirmation_lease import load_lease, control_root
    lease = load_lease(lease_path, lease_sha256)
    if os.environ.get('RUNPOD_POD_ID') != lease['pod_id']:
        raise ValueError('Smoke worker is not on the reserved pod')
    if not Path(folder).resolve().is_relative_to(Path(lease['allowed_result_root']).resolve()):
        raise ValueError('Smoke output is outside the reserved result root')

    def guard():
        if time.time() >= lease['deadline_epoch'] - 120:
            raise TimeoutError('Reserved allocation deadline')
        if (control_root(lease) / 'lease_guard/STOP').exists():
            raise TimeoutError('Allocation stop marker')
    guard()
    return lease, guard


def execution_identity(lease, numerical_runtime):
    # These identify the physical attempt and deliberately do not seed sampling.
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version',
                                          '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    return {'pid': os.getpid(), 'host': socket.gethostname(), 'pod_id': lease['pod_id'],
            'process_started_ns': time.time_ns(), 'gpu_driver_inventory': driver,
            'runtime': numerical_runtime, 'runtime_sha256': digest(numerical_runtime)}


def load_models(p, cases, runtime_sha):
    from gearshift.coding_inference import Backend
    roles = {'source' if c == 'source_reasoning' else 'receiver' for c in cases}
    if any(c in ('receiver_mapped', 'receiver_hybrid') for c in cases):
        roles.add('source')
    result = {}
    for role in ('source', 'receiver'):
        if role not in roles:
            continue
        backend = Backend(p['models'][role])
        if backend.config._commit_hash != p['models'][role]['revision']:
            raise ValueError('Loaded model revision differs')
        backend.sampling_identity = {**p['models'][role], 'model_id': backend.name,
                                     'runtime_identity_sha256': runtime_sha,
                                     'declaration_sha256': digest(p)}
        result[role] = (backend, Observer(backend))
    if len(result) == 2:
        a, b = result['source'][0], result['receiver'][0]
        if (a.tokenizer.backend_tokenizer.to_str() != b.tokenizer.backend_tokenizer.to_str()
                or a.tokenizer.chat_template != b.tokenizer.chat_template):
            raise ValueError('Pinned tokenizers differ')
    return result


def answer_base(repo, p, case, models, history):
    import torch
    from gearshift.core import CacheExtractor, CacheInjector
    from gearshift.coding_gradients import AffineMapper
    from gearshift.coding_post_progress import splice
    receiver = models['receiver'][0]
    if case == 'receiver_native':
        out = receiver.prefill_chunked(history['prefix_ids'])
        pairs = CacheExtractor.tensors(out.past_key_values)
        return CacheInjector.create(pairs, clone=True)
    source = models['source'][0]
    path = scoped(repo, p['mapper']['path'])
    if sha(path) != p['mapper']['sha256'] or path.stat().st_size != p['mapper']['bytes']:
        raise ValueError('Original step-1024 mapper bytes differ')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['step'] != 1024 or payload['arm'] != p['mapper']['arm']:
        raise ValueError('Original mapper payload identity differs')
    mapper = AffineMapper(source, receiver).to(receiver.device)
    mapper.load_state_dict(payload['state_dict'], strict=True)
    mapper.requires_grad_(False)
    source_out = source.prefill_chunked(history['prefix_ids'])
    # Same full-history mapping and native prompt splice as frozen receiver_job.
    mapped = mapper(CacheExtractor.tensors(source_out.past_key_values))
    if case == 'receiver_hybrid':
        prompt = receiver.prefill_chunked(history['prompt_ids'])
        mapped = splice(CacheExtractor.tensors(prompt.past_key_values), mapped,
                        len(history['prompt_ids']))
    return CacheInjector.create(mapped, clone=True)


def state_values(state):
    return {k: state[k] for k in ('identity_sha256', 'tokens', 'forwarded_tokens',
                                 'rng_initial', 'rng_state', 'logits_sha256')}


def run_case(repo, folder, p, case, phase, models, attempt, guard, exit_fn=os._exit):
    import torch
    from gearshift.coding_confirmation_sampling import reason_durable, answer_durable
    from gearshift.coding_inference import memory_record
    folder = Path(folder)
    dest = folder / ('baseline' if phase == 'baseline' else 'restarted') / case
    control = folder / 'proof' / case
    if (dest / 'resume.json').exists() and phase != 'resume':
        raise ValueError('Smoke trajectory already exists; no rerolls or overwrites')
    if (control / (phase + '_started.json')).exists():
        raise ValueError('Smoke phase already attempted; preserve evidence instead of retrying')
    cut = read(control / 'interruption.json') if phase == 'resume' else None
    if cut:
        if sha(dest / 'resume.json') != cut['resume_file_sha256']:
            raise ValueError('Interrupted durable state changed before resume')
        if read(dest / 'resume.json')['state'] != 'running':
            raise ValueError('Expected an actual abrupt-exit durable state')
    bind(control / (phase + '_started.json'), {'case': case, 'phase': phase, **attempt})
    role = 'source' if case == 'source_reasoning' else 'receiver'
    backend, observer = models[role]
    # Release caller-only references left over from other cases before cache setup.
    for _, observed in models.values():
        observed.last = None; observed.arm()
    guard()
    history = base = None
    with torch.no_grad():
        if case == 'source_reasoning':
            prompt_ids = backend.tokenizer.apply_chat_template([{'role': 'user', 'content': PROMPT}],
                            tokenize=True, add_generation_prompt=True, enable_thinking=True)
            if not 1 <= len(prompt_ids) <= 512:
                raise ValueError('Synthetic prompt unexpectedly oversized')
        else:
            source_proof = read(folder / 'proof/source_reasoning/baseline_result.json')
            if (source_proof['record_path'] != 'baseline/source_reasoning/source_history.json'
                    or sha(folder / source_proof['record_path']) != source_proof['record_sha256']):
                raise ValueError('Synthetic conditioning history changed')
            history = read(folder / 'baseline/source_reasoning/source_history.json')
            if history['task_id'] != TASK or len(history['reasoning_ids']) > p['cap']:
                raise ValueError('Only the synthetic baseline history may condition answers')
            base = answer_base(repo, p, case, models, history)
            backend.sampling_identity['cache_identity_sha256'] = digest({
                'case': case, 'history_sha256': digest(history),
                'mapper_sha256': p['mapper']['sha256'] if case != 'receiver_native' else None})
        base_fingerprint = cache_fingerprint(base) if base is not None else None
        observer.arm(cut['boundary']['cache']['sequence_length'] if cut else None)
        telemetry = Telemetry(control / phase)

        def publish(**progress):
            guard()
            if not memory_record()['passed']:
                raise RuntimeError('Smoke VRAM headroom gate failed')
            if phase != 'interrupt' or progress.get('generated_tokens') != p['interrupt_after']:
                return
            state = read(dest / 'resume.json')
            if state['state'] != 'running' or len(state['tokens']) != p['interrupt_after']:
                raise ValueError('Durable checkpoint does not match predetermined interruption')
            marker = {'case': case, 'attempt': attempt, 'plan_sha256': sha(folder / 'plan.json'),
                      'resume_file_sha256': sha(dest / 'resume.json'),
                      'state': state_values(state), 'base_cache': base_fingerprint,
                      'boundary': observer.snapshot(), 'expected_exit_code': EXIT_INTERRUPTED,
                      'cache_tensors_persisted': False, 'epoch': time.time()}
            if marker['boundary']['next_logits_sha256'] != state['logits_sha256']:
                raise ValueError('Observer and frozen checkpoint logits disagree')
            bind(control / 'interrupted_state.json', state)
            bind(control / 'interruption.json', marker)
            exit_fn(EXIT_INTERRUPTED)
            raise RuntimeError('Hard-exit hook unexpectedly returned')

        if case == 'source_reasoning':
            record, cache = reason_durable(backend, prompt_ids, TASK, p['source_stream'], p['cap'],
                                           dest, telemetry, guard, publish)
        else:
            record = answer_durable(backend, history, base, TASK, p['answer_stream'], p['cap'],
                                     dest, telemetry, guard, publish)
        if phase == 'interrupt':
            bind(control / 'interruption_not_exercised.json', {
                'reason': 'Terminal reached before prescribed cut; no reroll permitted.',
                'attempt': attempt, 'case': case})
            raise RuntimeError('Predetermined interruption was not exercised')
        final_state = read(dest / 'resume.json')
        memory = memory_record()
        if not memory['passed']:
            raise RuntimeError('Smoke VRAM headroom gate failed')
        receipt = {'schema': 1, 'case': case, 'phase': phase,
                   'plan_sha256': sha(folder / 'plan.json'), 'attempt': attempt,
                   'state': state_values(final_state), 'base_cache': base_fingerprint,
                   'final': observer.snapshot(), 'reconstructed_boundary': observer.reconstructed,
                   'resume_count': record['sampler_timing']['resume_count'],
                   'record_path': str((dest / ('source_history.json' if case == 'source_reasoning' else 'answer_record.json')).relative_to(folder)),
                   'final_resume_sha256': sha(dest / 'resume.json'),
                   'forward_input_lengths': observer.forward_lengths, 'memory': memory,
                   'started_from_saved_state': bool(cut), 'saved_state_file_sha256': cut['resume_file_sha256'] if cut else None,
                   'no_cache_tensors_loaded': True}
        receipt['record_sha256'] = sha(folder / receipt['record_path'])
        bind(control / (phase + '_result.json'), receipt)
    for _, observed in models.values():
        observed.last = None; observed.arm()
    return receipt


def phase(repo, folder, plan_sha, mode, case, lease_path, lease_sha):
    if Path(repo).resolve() != ROOT.resolve():
        raise ValueError('Execute the harness inside the staged repository it verifies')
    p = validate_plan(repo, folder, plan_sha)
    lease, guard = allocation_guard(folder, lease_path, lease_sha)
    if mode == 'interrupt' and case == 'all':
        raise ValueError('Hard-exit phase must name exactly one case')
    cases = CASES if case == 'all' else [case]
    numerical = runtime()
    attempt = execution_identity(lease, numerical)
    guard()
    models = load_models(p, cases, digest(numerical))
    for current in cases:
        run_case(repo, folder, p, current, mode, models, attempt, guard)


def compare_case(folder, p, case):
    folder = Path(folder); root = folder / 'proof' / case
    a, cut, b = (read(root / name) for name in ('baseline_result.json', 'interruption.json', 'resume_result.json'))
    for row in (a, cut, b):
        if row['case'] != case or row['plan_sha256'] != sha(folder / 'plan.json'):
            raise ValueError('Smoke result identity differs')
    for row, prefix in ((a, 'baseline'), (b, 'restarted')):
        expected_name = 'source_history.json' if case == 'source_reasoning' else 'answer_record.json'
        if row['record_path'] != prefix + '/' + case + '/' + expected_name:
            raise ValueError('Unexpected synthetic record path')
        if sha(folder / row['record_path']) != row['record_sha256']:
            raise ValueError('Completed raw synthetic record changed')
        resume_path = (folder / row['record_path']).parent / 'resume.json'
        state = read(resume_path)
        if (sha(resume_path) != row['final_resume_sha256'] or state['state'] != 'complete'
                or state_values(state) != row['state']):
            raise ValueError('Completed durable state changed')
    if sha(root / 'interrupted_state.json') != cut['resume_file_sha256']:
        raise ValueError('Archived interruption checkpoint changed')
    check = {
        'full_token_sequence_equal': a['state']['tokens'] == b['state']['tokens'],
        'full_generator_state_equal': a['state']['rng_initial'] == b['state']['rng_initial'] and a['state']['rng_state'] == b['state']['rng_state'],
        'sampler_identity_equal': a['state']['identity_sha256'] == b['state']['identity_sha256'] == cut['state']['identity_sha256'],
        'next_logits_exactly_equal': a['state']['logits_sha256'] == b['state']['logits_sha256'] == a['final']['next_logits_sha256'] == b['final']['next_logits_sha256'],
        'all_final_KV_bytes_equal': a['final']['cache'] == b['final']['cache'],
        'all_reconstructed_KV_bytes_equal': cut['boundary'] == b['reconstructed_boundary'],
        'base_cache_bytes_equal': a['base_cache'] == cut['base_cache'] == b['base_cache'],
        'interrupted_prefix_preserved': cut['state']['tokens'] == a['state']['tokens'][:p['interrupt_after']] and len(cut['state']['tokens']) == p['interrupt_after'],
        'saved_checkpoint_reloaded': b['saved_state_file_sha256'] == cut['resume_file_sha256'] and b['started_from_saved_state'] and b['resume_count'] == 1 and a['resume_count'] == 0,
        'separate_processes': len({(x['pod_id'], x['pid']) for x in (a['attempt'], cut['attempt'], b['attempt'])}) == 3,
        'same_numerical_runtime': a['attempt']['runtime'] == cut['attempt']['runtime'] == b['attempt']['runtime'],
        'bounded_tokens': len(a['state']['tokens']) <= p['cap'] <= 256,
        'VRAM_headroom_passed': a['memory']['passed'] and b['memory']['passed'],
        'no_KV_checkpoint_loaded': cut['cache_tensors_persisted'] is False and b['no_cache_tensors_loaded'] is True,
    }
    if (root / 'interruption_not_exercised.json').exists():
        check['actual_interruption_exercised'] = False
    checksums = {str(path.relative_to(folder)): sha(path) for path in sorted(root.rglob('*.json'))}
    return {'case': case, 'checks': check, 'passed': all(check.values()),
            'generated_tokens': len(a['state']['tokens']), 'interrupted_after': p['interrupt_after'],
            'cross_pod_resume': cut['attempt']['pod_id'] != b['attempt']['pod_id'],
            'runtime': a['attempt']['runtime'], 'evidence': checksums}


def verify(repo, folder, plan_sha):
    p = validate_plan(repo, folder, plan_sha)
    rows = [compare_case(folder, p, case) for case in CASES]
    result = {'schema': 1, 'purpose': p['purpose'], 'plan_sha256': plan_sha,
              'all_cases_passed': all(r['passed'] for r in rows), 'cases': rows,
              'cross_pod_resume_for_all_cases': all(r['cross_pod_resume'] for r in rows),
              'scope': 'Observed bounded sampler resume equivalence only. No training-resume, quality, '
                       'long-context, arbitrary interruption-point, or universal numerical-correctness claim.',
              'limitations': p['limitations']}
    bind(Path(folder) / 'SMOKE_PROOF.json', result)
    return result


def run(repo, folder, plan_sha, lease_path, lease_sha, timeout=3600):
    if not 0 < timeout <= 7200:
        raise ValueError('Smoke wall-clock budget must be at most two hours')
    validate_plan(repo, folder, plan_sha)
    deadline = time.monotonic() + timeout
    phases = [('baseline', 'all'), *(('interrupt', case) for case in CASES), ('resume', 'all')]
    receipts = []
    for mode, case in phases:
        command = [sys.executable, str(Path(repo) / SCRIPT), 'phase', '--repo', str(repo),
                   '--folder', str(folder), '--plan-sha256', plan_sha, '--phase', mode,
                   '--case', case, '--lease', str(lease_path), '--lease-sha256', lease_sha]
        log = Path(folder) / 'launch' / (mode + '_' + case + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise TimeoutError('Bounded smoke deadline; worker killed without retry')
        expected = EXIT_INTERRUPTED if mode == 'interrupt' else 0
        row = {'phase': mode, 'case': case, 'pid': child.pid, 'exit_code': code, 'expected_exit_code': expected}
        receipts.append(row)
        write(Path(folder) / 'launch/process_receipts.json', receipts)
        if code != expected:
            raise RuntimeError('Smoke worker did not produce expected completion: ' + str(row))
    return verify(repo, folder, plan_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_p = sub.add_parser('prepare')
    prepare_p.add_argument('--repo', type=Path, default=ROOT)
    prepare_p.add_argument('--output', type=Path, required=True)
    prepare_p.add_argument('--cap', type=int, default=128)
    prepare_p.add_argument('--interrupt-after', type=int, default=1)
    prepare_p.add_argument('--mapper-arm', choices=['FIXED', 'ROTATING'], default='ROTATING')
    for name in ('run', 'phase', 'verify'):
        q = sub.add_parser(name); q.add_argument('--repo', type=Path, default=ROOT)
        q.add_argument('--folder', type=Path, required=True)
        q.add_argument('--plan-sha256', required=True)
        if name != 'verify':
            q.add_argument('--lease', type=Path, required=True); q.add_argument('--lease-sha256', required=True)
        if name == 'run': q.add_argument('--timeout-seconds', type=int, default=3600)
        if name == 'phase':
            q.add_argument('--phase', choices=['baseline', 'interrupt', 'resume'], required=True)
            q.add_argument('--case', choices=['all', *CASES], required=True)
    a = parser.parse_args()
    if a.command == 'prepare':
        result = prepare(a.repo, a.output, cap=a.cap, interrupt_after=a.interrupt_after, mapper_arm=a.mapper_arm)
    elif a.command == 'run':
        result = run(a.repo, a.folder, a.plan_sha256, a.lease, a.lease_sha256, a.timeout_seconds)
    elif a.command == 'phase':
        result = phase(a.repo, a.folder, a.plan_sha256, a.phase, a.case, a.lease, a.lease_sha256)
    else:
        result = verify(a.repo, a.folder, a.plan_sha256)
    print(json.dumps(result, indent=2))
    if isinstance(result, dict) and result.get('all_cases_passed') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
