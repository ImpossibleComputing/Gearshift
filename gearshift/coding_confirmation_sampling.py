"""Durable, identity-bound samplers using the original numerical call schedule.

Callers own immutable cache construction, cloning, and scientific job claims.
Every backend must expose ``sampling_identity`` with model_id, revision,
runtime_identity_sha256 and declaration_sha256. Answers additionally require
cache_identity_sha256 identifying their native/mapped/P conditioning cache.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .coding_control import bind, digest, seed_for, sha, write
from .coding_inference import generator, sample, sync


_OVERHEAD_KINDS = ('checkpoint_persistence', 'checkpoint_preparation', 'telemetry',
                   'progress_publish', 'result_materialization', 'identity_binding')
_OVERHEAD_FIELDS = tuple(kind + suffix for kind in _OVERHEAD_KINDS
                         for suffix in ('_seconds', '_calls', '_failed_calls'))


def _valid_overhead(value):
    if not isinstance(value, dict) or value.get('schema') != 1 or type(value.get('scope_complete')) is not bool:
        return False
    for name in _OVERHEAD_FIELDS:
        number = value.get(name)
        if name.endswith('_seconds'):
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                return False
        elif type(number) is not int or number < 0:
            return False
    return True


def _read(path):
    return json.loads(Path(path).read_text())


def _fingerprint(logits):
    if logits is None:
        return None
    value = logits.detach().contiguous().cpu()
    return digest({'shape': list(value.shape), 'dtype': str(value.dtype),
                   'bytes_sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()})


@contextmanager
def _lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.sampler.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _model_identity(backend, answer):
    model = dict(getattr(backend, 'sampling_identity', {}))
    required = ['model_id', 'revision', 'runtime_identity_sha256', 'declaration_sha256']
    if answer:
        required.append('cache_identity_sha256')
    if any(not isinstance(model.get(k), str) or not model[k] for k in required):
        raise ValueError('Explicit pinned backend sampling_identity is required')
    if getattr(backend, 'name', model['model_id']) != model['model_id']:
        raise ValueError('Backend model name differs from sampling identity')
    return {key: model[key] for key in required}


class _Session:
    def __init__(self, backend, task_id, stream, cap, folder, kind, conditioning, telemetry):
        if type(cap) is not int or cap <= 0:
            raise ValueError('Positive integer sampling cap required')
        self.backend, self.folder, self.kind = backend, folder, kind
        self.telemetry = telemetry
        self.overhead = {name: 0. if name.endswith('_seconds') else 0 for name in _OVERHEAD_FIELDS}
        self.overhead_complete = True
        self.overhead_gaps = []
        self.prior_attempt_receipts = []
        self.last_resume_sha256 = None
        self.identity = {'schema': 1, 'sampler': kind, 'task_id': task_id,
                         'stream': stream, 'seed': seed_for(task_id, 0, stream), 'cap': cap,
                         'model': _model_identity(backend, kind == 'answer'),
                         'conditioning': conditioning, 'prefill_chunk': 512,
                         'continuation_forward_chunk': 1,
                         'sampling': {'temperature': .6, 'top_p': .95, 'top_k': 20},
                         'eos': sorted(backend.eos), 'closing_think': 151668}
        with self.measure('identity_binding'):
            self.ident = bind(folder / 'identity.json', self.identity)
        self.rng = generator(task_id, stream, backend.device)
        self.initial = self.rng.get_state().cpu().tolist()
        self.tokens = []
        self.saved = None
        self.base_seconds = self.reconstruction_seconds = self.prefill_seconds = 0.
        self.bridge_seconds = 0.
        self.first = None
        self.started = None
        self.complete_timing = True
        self.resume_count = 0
        self.snap = {'count': 0, 'forwarded': 0, 'rng': self.initial, 'logits': None}
        path = folder / 'resume.json'
        if path.exists():
            state = _read(path)
            checksum = state.pop('resume_sha256', None)
            if checksum != digest(state) or state.get('identity_sha256') != self.ident:
                raise ValueError('Durable sampler state hash or identity differs')
            if state['rng_initial'] != self.initial:
                raise ValueError('Initial sampler RNG differs')
            self.tokens = state['tokens']
            if (not isinstance(self.tokens, list) or len(self.tokens) > cap or
                    any(type(t) is not int or t < 0 for t in self.tokens)):
                raise ValueError('Invalid committed sampler tokens')
            terminals = set(backend.eos) | ({151668} if kind == 'reasoning' else set())
            if any(t in terminals for t in self.tokens[:-1]):
                raise ValueError('Committed trajectory continues after terminal token')
            maximum = len(self.tokens)
            if self.tokens and (self.tokens[-1] == 151668 if kind == 'reasoning'
                                else self.tokens[-1] in backend.eos):
                maximum -= 1
            if not 0 <= state['forwarded_tokens'] <= maximum:
                raise ValueError('Invalid committed forward position')
            self.rng.set_state(torch.tensor(state['rng_state'], dtype=torch.uint8, device='cpu'))
            self.saved = state
            self.last_resume_sha256 = checksum
            self.restore_overhead(state, checksum)
            timing = state['timing']
            self.base_seconds = timing['active_seconds']
            self.reconstruction_seconds = timing['reconstruction_seconds']
            self.prefill_seconds = timing['initial_prefill_seconds']
            self.bridge_seconds = timing['initial_bridge_seconds']
            self.first = timing['first_answer_token_seconds']
            self.complete_timing = timing['complete'] and state['state'] != 'running'
            self.resume_count = timing['resume_count'] + (state['state'] != 'complete')
            self.snap = {'count': len(self.tokens), 'forwarded': state['forwarded_tokens'],
                         'rng': state['rng_state'], 'logits': None}
        elif (folder / 'complete.json').exists() or (folder / self.record_name).exists():
            raise ValueError('Sampler result exists without its durable transaction')
        elif any((folder / 'attempts').glob('*.json')):
            self.overhead_complete = False
            self.overhead_gaps.append('Prior attempt had no durable sampling checkpoint; its overhead is not inferred.')
        self.call_started = time.monotonic()
        self.call_epoch = time.time()
        self.start_tokens = len(self.tokens)
        self.rebuild_started = None
        self.rebuild_this_call = 0.
        self.attempt_path = folder / 'attempts' / f'{time.time_ns()}.json'
        self.attempt('started')

    @contextmanager
    def measure(self, kind):
        """Elapsed operation wall time only: no new device sync or model call."""
        before = time.perf_counter()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.overhead[kind + '_seconds'] += time.perf_counter() - before
            self.overhead[kind + '_calls'] += 1
            self.overhead[kind + '_failed_calls'] += int(failed)

    def overhead_snapshot(self):
        return {'schema': 1, **self.overhead, 'scope_complete': self.overhead_complete,
                'attempt_id': self.attempt_path.name,
                'completion_receipt_file': 'completion_timing.json',
                'prior_attempt_receipts': list(self.prior_attempt_receipts),
                'unmeasured_gaps': list(self.overhead_gaps),
                'scope': 'Cumulative known measurements along this durable transaction through this snapshot. '
                         'Persistence measures resume.json write(), including JSON serialization, file/directory fsync '
                         'and atomic replacement. Preparation measures checkpoint assembly/logit fingerprint/hash work. '
                         'Telemetry and publish measure the unchanged callback calls; identity binding is outside active time. '
                         'Materialization measures record/complete receipt binding and hashing. Failed calls contribute observed time.',
                'exclusions': ['Attempt and completion timing receipt writes, lock acquisition, checkpoint reads and uninstrumented setup.',
                               'Any call currently in progress; final checkpoint/materialization follow the result snapshot '
                               'and are available only in completion_timing.json.',
                               'Later completed-result retrievals are separate attempts and never mutate completion_timing.json.'],
                'subtract_from_active_seconds': False,
                'model_only_latency_measured': False}

    def restore_overhead(self, state, checksum):
        """Prefer a closed attempt; a crash snapshot is an explicitly partial sum.

        A checkpoint cannot contain the duration of writing itself. A matching
        closed attempt records that duration afterwards. Without it, retain only
        already measured counters and never invent the missing write/crash tail.
        Timing metadata never chooses tokens, replay positions or RNG state.
        """
        snapshot = state.get('timing', {}).get('overhead')
        measured = snapshot if _valid_overhead(snapshot) else None
        closed = False
        attempt_id = snapshot.get('attempt_id') if isinstance(snapshot, dict) else None
        if isinstance(attempt_id, str) and Path(attempt_id).name == attempt_id and attempt_id.endswith('.json'):
            path = self.folder / 'attempts' / attempt_id
            try:
                receipt = _read(path)
                if (receipt.get('identity_sha256') == self.ident and receipt.get('state') in ('failed', 'complete')
                        and receipt.get('last_resume_sha256') == checksum and _valid_overhead(receipt.get('overhead'))):
                    measured = receipt['overhead']; closed = True
                    self.prior_attempt_receipts = list(measured.get('prior_attempt_receipts', []))
                    self.prior_attempt_receipts.append({'path': str(path.relative_to(self.folder)), 'sha256': sha(path)})
            except (OSError, ValueError, TypeError):
                pass
        if measured is not None:
            for name in _OVERHEAD_FIELDS:
                self.overhead[name] += measured[name]
            self.overhead_gaps = list(measured.get('unmeasured_gaps', []))
            if not closed:
                self.prior_attempt_receipts = list(measured.get('prior_attempt_receipts', []))
        self.overhead_complete = bool(measured and measured['scope_complete'] and closed)
        if not closed:
            self.overhead_gaps.append('No matching closed-attempt timing receipt; checkpoint self-write and any crash tail are unknown.')

    def attempt(self, state, **extra):
        reconstruction = self.rebuild_this_call
        if self.rebuild_started is not None:
            reconstruction = time.monotonic() - self.rebuild_started
        write(self.attempt_path, {
            'identity_sha256': self.ident, 'state': state, 'started_epoch': self.call_epoch,
            'sampler_call_wall_seconds': time.monotonic() - self.call_started,
            'initial_committed_tokens': self.start_tokens,
            'last_consistent_tokens': self.snap['count'],
            'last_resume_sha256': self.last_resume_sha256,
            'overhead': self.overhead_snapshot(),
            'reconstruction_seconds_this_call': reconstruction,
            'interrupted_running_receipt_means_timing_incomplete': state == 'started',
            **extra})

    @property
    def record_name(self):
        return 'source_history.json' if self.kind == 'reasoning' else 'answer_record.json'

    def timing(self):
        return {'active_seconds': self.base_seconds + (time.monotonic() - self.started if self.started is not None else 0.),
                'reconstruction_seconds': self.reconstruction_seconds,
                'initial_prefill_seconds': self.prefill_seconds,
                'initial_bridge_seconds': self.bridge_seconds,
                'first_answer_token_seconds': self.first, 'complete': self.complete_timing,
                'resume_count': self.resume_count, 'overhead': self.overhead_snapshot()}

    def checkpoint(self, state='running', record=None):
        # snap is replaced only at a consistent token/RNG/logit boundary. A
        # failure inside sample/forward can never save an advanced RNG without
        # the corresponding sampled token, or claim an incomplete forward.
        with self.measure('checkpoint_preparation'):
            s = self.snap
            row = {'schema': 1, 'identity_sha256': self.ident, 'state': state,
                   'tokens': self.tokens[:s['count']], 'forwarded_tokens': s['forwarded'],
                   'rng_initial': self.initial, 'rng_state': s['rng'],
                   'logits_sha256': _fingerprint(s['logits']), 'timing': self.timing(),
                   'cache_checkpointed': False}
            if record is not None:
                row.update(record=record, record_sha256=digest(record))
            row['resume_sha256'] = digest(row)
        with self.measure('checkpoint_persistence'):
            write(self.folder / 'resume.json', row)
        self.last_resume_sha256 = row['resume_sha256']

    def commit(self, logits, forwarded):
        self.snap = {'count': len(self.tokens), 'forwarded': forwarded,
                     'rng': self.rng.get_state().cpu().tolist(), 'logits': logits}

    def verify_reconstruction(self, logits):
        if self.saved and self.saved['logits_sha256'] is not None:
            if _fingerprint(logits) != self.saved['logits_sha256']:
                write(self.folder / 'reconstruction_failure.json', {
                    'identity_sha256': self.ident, 'expected': self.saved['logits_sha256'],
                    'actual': _fingerprint(logits), 'new_sampling_performed': False})
                raise ValueError('Exact sampler reconstruction logits differ')

    def finish(self, record):
        self.checkpoint('complete', record)
        self.materialize(record)
        self.completion_timing(record)
        self.attempt('complete', completed_result_reused=False)
        return record

    def materialize(self, record):
        with self.measure('result_materialization'):
            bind(self.folder / self.record_name, record)
            bind(self.folder / 'complete.json', {'identity_sha256': self.ident,
                 'record_sha256': digest(record), 'record_file_sha256': sha(self.folder / self.record_name),
                 'record_file': self.record_name})

    def completion_timing(self, record):
        """Immutable post-commit measurement, excluding its own persistence.

        Generation closure must hash this sidecar together with the result. Its
        absence on recovery is not evidence of zero final-commit I/O; the closed
        attempt/snapshot distinction above determines measured-scope completeness.
        """
        path = self.folder / 'completion_timing.json'
        if path.exists():
            receipt = _read(path)
            if receipt.get('identity_sha256') != self.ident or receipt.get('record_sha256') != digest(record):
                raise ValueError('Completion timing belongs to a different sampler result')
            return receipt
        receipt = {'schema': 1, 'state': 'complete', 'identity_sha256': self.ident,
                   'record_sha256': digest(record), 'terminal_attempt_id': self.attempt_path.name,
                   'complete_transaction_recovered': bool(self.saved and self.saved['state'] == 'complete'),
                   'overhead': self.overhead_snapshot(),
                   'final_checkpoint_and_materialization_included_when_scope_complete': True,
                   'this_receipt_persistence_excluded': True,
                   'active_seconds_unchanged': True}
        bind(path, receipt)
        return receipt

    def completed_record(self):
        if not self.saved or self.saved['state'] != 'complete':
            return None
        record = self.saved.get('record')
        if not isinstance(record, dict) or digest(record) != self.saved.get('record_sha256'):
            raise ValueError('Completed sampler transaction has no verified result')
        key = 'reasoning_ids' if self.kind == 'reasoning' else 'answer_ids'
        rng_key = 'rng_after_reasoning' if self.kind == 'reasoning' else 'rng_final'
        if record[key] != self.tokens or record[rng_key] != self.saved['rng_state']:
            raise ValueError('Completed result differs from sampler transaction')
        self.materialize(record)
        self.completion_timing(record)
        return record

    def failed(self, exc):
        # Never overwrite a previously verified complete transaction or a bad
        # checkpoint with a partially reconstructed state.
        transaction_complete = False
        try:
            durable = _read(self.folder / 'resume.json')
            checksum = durable.pop('resume_sha256', None)
            transaction_complete = durable.get('state') == 'complete' and checksum == digest(durable)
        except (OSError, ValueError):
            pass
        if self.started is not None and not transaction_complete:
            try:
                self.checkpoint('interrupted')
            except BaseException:
                pass
        try:
            with self.measure('telemetry'):
                self.telemetry.failure(exc)
        except BaseException:
            pass
        try:
            self.attempt('failed', error_type=type(exc).__name__,
                         complete_transaction_preserved=transaction_complete)
        except BaseException:
            pass


def _progress(s, guard, publish, forwarded):
    s.checkpoint()
    with s.measure('telemetry'):
        s.telemetry.sample(stage=s.kind + '_generation', task_id=s.identity['task_id'],
                           generated_tokens=len(s.tokens), forwarded_tokens=forwarded)
    with s.measure('progress_publish'):
        publish(stage=s.kind + '_generation', task_id=s.identity['task_id'], generated_tokens=len(s.tokens))
    guard()


def _replay_tokens(backend, tokens, count, cache, logits, guard):
    for i, token in enumerate(tokens[:count]):
        out = backend.forward([token], cache)
        cache, logits = out.past_key_values, out.logits
        if i % 64 == 0:
            guard()
    return cache, logits


@torch.no_grad()
def reason_durable(backend, prompt_ids, task_id, stream, cap, folder, telemetry, guard, publish):
    """Return a saved history and exact native pre-bridge cache; never resample it."""
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError('Nonempty original prompt required')
    folder = Path(folder)
    with _lock(folder):
        s = _Session(backend, task_id, stream, cap, folder, 'reasoning', {'prompt_ids': prompt_ids}, telemetry)
        completed = s.completed_record()
        try:
            guard(); sync(); before = time.monotonic()
            if s.saved:
                s.rebuild_started = before
            out = backend.prefill_chunked(prompt_ids, chunk=512)
            cache, logits = out.past_key_values, out.logits
            sync(); prefill = time.monotonic() - before
            forwarded = 0
            if s.saved:
                cache, logits = _replay_tokens(backend, s.tokens, s.saved['forwarded_tokens'], cache, logits, guard)
                forwarded = s.saved['forwarded_tokens']
                s.verify_reconstruction(logits)
                # The last committed sample can have failed before its forward.
                n = len(s.tokens) - bool(s.tokens and s.tokens[-1] == 151668)
                for token in s.tokens[forwarded:n]:
                    out = backend.forward([token], cache)
                    cache, logits = out.past_key_values, out.logits
                    forwarded += 1
                sync(); reconstruction = time.monotonic() - before
                s.rebuild_this_call, s.rebuild_started = reconstruction, None
                if completed is not None:
                    if cache.get_seq_length() != len(completed['prefix_ids']):
                        raise ValueError('Reconstructed completed history length differs')
                    receipt = {'identity_sha256': s.ident, 'completed_result_reused': True,
                               'new_samples': 0, 'cache_reconstruction_seconds': reconstruction,
                               'epoch': time.time()}
                    write(folder / 'retrievals' / f'{time.time_ns()}.json', receipt)
                    s.attempt('complete', completed_result_reused=True, new_samples=0)
                    return completed, cache
                s.reconstruction_seconds += reconstruction
            else:
                s.prefill_seconds = prefill
                s.base_seconds = prefill
            s.commit(logits, forwarded)
            s.started = time.monotonic()
            s.checkpoint()
            terminal = bool(s.tokens and (s.tokens[-1] == 151668 or s.tokens[-1] in backend.eos))
            while len(s.tokens) < cap and not terminal:
                token = sample(logits, s.rng)
                s.tokens.append(token)
                s.commit(logits, forwarded)
                terminal = token == 151668 or token in backend.eos
                if token == 151668:
                    break
                # Original reasoning forwards EOS once after detecting it; all
                # ordinary tokens, including the final capped token, are cached.
                out = backend.forward([token], cache)
                cache, logits = out.past_key_values, out.logits
                forwarded += 1
                s.commit(logits, forwarded)
                if not terminal and (len(s.tokens) - 1) % 64 == 0:
                    _progress(s, guard, publish, forwarded)
            natural = bool(s.tokens and s.tokens[-1] == 151668)
            early = bool(s.tokens and s.tokens[-1] in backend.eos and not natural)
            prefix = prompt_ids + (s.tokens[:-1] if natural else s.tokens)
            if cache.get_seq_length() != len(prefix):
                raise ValueError('Natural handoff cache length differs')
            sync(); timing = s.timing()
            record = {'task_id': task_id, 'prompt_ids': prompt_ids, 'reasoning_ids': s.tokens,
                      'prefix_ids': prefix, 'bridge_ids': [151668], 'natural_boundary': natural,
                      'reasoning_capped': not natural and not early, 'early_eos': early,
                      'prefix_cache_length': len(prefix), 'seed': seed_for(task_id, 0, stream),
                      'rng_initial': s.initial, 'rng_after_reasoning': s.snap['rng'],
                      'reasoning_seconds': timing['active_seconds'],
                      'reasoning_text': backend.tokenizer.decode(s.tokens, skip_special_tokens=False),
                      'sampler_identity_sha256': s.ident, 'sampler_timing': timing}
            return s.finish(record), cache
        except BaseException as exc:
            s.failed(exc)
            raise


@torch.no_grad()
def answer_durable(backend, history, cache, task_id, stream, cap, folder, telemetry, guard, publish):
    """Return an exact committed answer, reconstructing only unfinished draws.

The caller supplies a fresh private copy of the same native/mapped base cache
on each attempt. Its identity must be set on backend.sampling_identity.
"""
    folder = Path(folder)
    conditioning = {'history_sha256': digest(history), 'prefix_ids': history['prefix_ids'],
                    'bridge_ids': history['bridge_ids']}
    with _lock(folder):
        s = _Session(backend, task_id, stream, cap, folder, 'answer', conditioning, telemetry)
        completed = s.completed_record()
        if completed is not None:
            s.attempt('complete', completed_result_reused=True, new_samples=0)
            return completed
        try:
            if cache.get_seq_length() != len(history['prefix_ids']):
                raise ValueError('Answer base cache length differs')
            guard(); sync(); before = time.monotonic()
            if s.saved:
                s.rebuild_started = before
            out = backend.forward(history['bridge_ids'], cache)
            cache, logits = out.past_key_values, out.logits
            sync(); bridge = time.monotonic() - before
            forwarded = 0
            if s.saved:
                cache, logits = _replay_tokens(backend, s.tokens, s.saved['forwarded_tokens'], cache, logits, guard)
                forwarded = s.saved['forwarded_tokens']
                s.verify_reconstruction(logits)
                n = len(s.tokens) - bool(s.tokens and s.tokens[-1] in backend.eos)
                for token in s.tokens[forwarded:n]:
                    out = backend.forward([token], cache)
                    cache, logits = out.past_key_values, out.logits
                    forwarded += 1
                sync(); reconstruction = time.monotonic() - before
                s.rebuild_this_call, s.rebuild_started = reconstruction, None
                s.reconstruction_seconds += reconstruction
            else:
                s.bridge_seconds = bridge
                s.base_seconds = bridge
            s.commit(logits, forwarded)
            s.started = time.monotonic()
            s.checkpoint()
            ended = bool(s.tokens and s.tokens[-1] in backend.eos)
            while len(s.tokens) < cap and not ended:
                token = sample(logits, s.rng)
                s.tokens.append(token)
                if s.first is None:
                    sync(); s.first = s.timing()['active_seconds']
                s.commit(logits, forwarded)
                if token in backend.eos:
                    ended = True
                    break
                out = backend.forward([token], cache)
                cache, logits = out.past_key_values, out.logits
                forwarded += 1
                s.commit(logits, forwarded)
                if (len(s.tokens) - 1) % 64 == 0:
                    _progress(s, guard, publish, forwarded)
            sync(); timing = s.timing()
            record = {'task_id': task_id, 'answer_ids': s.tokens,
                      'answer_text': backend.tokenizer.decode(s.tokens, skip_special_tokens=True),
                      'answer_text_with_special_tokens': backend.tokenizer.decode(s.tokens, skip_special_tokens=False),
                      'answer_seed': seed_for(task_id, 0, stream), 'rng_initial': s.initial,
                      'rng_final': s.snap['rng'], 'answer_seconds': timing['active_seconds'],
                      'bridge_seconds': s.bridge_seconds, 'first_answer_token_seconds': s.first,
                      'answer_ended_eos': ended, 'answer_capped': not ended,
                      'bridge_token_count': len(history['bridge_ids']),
                      'sampler_identity_sha256': s.ident, 'sampler_timing': timing,
                      'instrumentation': 'Active time still includes checkpoint/telemetry overhead. Measured overhead is separate, '
                                         'not subtracted into model-only latency; completion_timing.json includes final commit I/O. '
                                         'Cache reconstruction and incomplete crash timing remain separate.'}
            return s.finish(record)
        except BaseException as exc:
            s.failed(exc)
            raise
