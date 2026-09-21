"""Frozen coverage comparison: exact paired exposure, no score-based choices."""
from collections import Counter, OrderedDict
from pathlib import Path
import json, random
from .coding_control import sha, digest, seed_for

DECLARATION = 'configs/coding_pilot_v1/coverage_generalization/declaration.json'
OWNER = 'configs/coding_pilot_v1/coverage_generalization/OWNER_INSTRUCTION.txt'
CONFIG = 'configs/coding_pilot_v1/coverage_generalization'
EVIDENCE = 'evidence/coding_pilot_v1/coverage_generalization'
START_SHA = '0b9700ffd9cb38c23bcfa1327181b32992b128c6b95078214328382f86b8d68f'
PUBLICATION = '64725974fa55459350d1c9d09037bab64d0c5ec6'
ARMS = ('FIXED', 'ROTATING')


def grouped(item):
    result = OrderedDict()
    for part in item['segments']:
        result.setdefault(part['task_id'], []).extend(part['positions'])
    return result


class WindowCycle:
    """Shuffle contiguous eight-token windows, exhaust a cycle before repeats.

    Partial last windows are real tokens, never padded. If a new cycle starts
    within an update, postpone positions already in that update instead of
    counting duplicate predictions. No queued token is discarded.
    """
    def __init__(self, start, length, seed):
        self.windows = [list(range(a, min(a + 8, length))) for a in range(start, length, 8)]
        self.rng = random.Random(seed)
        self.queue = []
        self.cycle = -1

    def take(self, count, excluded=()):
        excluded = set(excluded)
        if count > sum(len(w) for w in self.windows) - len(excluded.intersection(p for w in self.windows for p in w)):
            raise ValueError('Insufficient distinct eligible positions')
        chosen, draws = [], []
        while len(chosen) < count:
            if not self.queue:
                self.cycle += 1
                windows = self.windows.copy()
                self.rng.shuffle(windows)
                self.queue = [(p, w[0], self.cycle) for w in windows for p in w]
            index = next((i for i, (p, _, _) in enumerate(self.queue) if p not in excluded), None)
            if index is None:
                raise AssertionError('Cycle cannot supply unique predictions')
            p, window, cycle = self.queue.pop(index)
            excluded.add(p); chosen.append(p)
            draws.append({'position': p, 'window_start': window, 'cycle': cycle})
        return chosen, draws


def paired_schedules(histories, updates=1024, seed=20260915, skip=96):
    from .coding_training import make_schedule
    lookup = {h['task_id']: h for h in histories}
    fixed = make_schedule(histories, seed, updates + skip)[skip:]
    queues = {}
    rotating = []
    for step, item in enumerate(fixed, 1):
        item['original_schedule_step'] = item['step']; item['step'] = step
        parts = []; used = {}
        for part in item['segments']:
            tid = part['task_id']; n = len(lookup[tid]['teacher_answer']['answer_ids'])
            used.setdefault(tid, set()); count = len(part['positions'])
            # The original four independent buckets often contribute only eight
            # positions of a task. Retain its early bucket; rotate other buckets.
            # Tasks of 9..32 tokens have no later fixed bucket: cycle all their
            # windows (including early tokens) so their EOS remains eligible.
            if part['anchor'] == 0 and not 8 < n <= 32:
                positions = part['positions'].copy(); draws = []
                rule = 'unchanged_early_bucket'
            else:
                start = 0 if n <= 32 else 8
                if tid not in queues:
                    queues[tid] = WindowCycle(start, n, seed_for(tid, seed, 'coverage_window_order'))
                positions, draws = queues[tid].take(count, used[tid])
                rule = 'whole_short_answer_cycle' if start == 0 else 'tail_window_cycle'
            used[tid].update(positions)
            parts.append({'task_id': tid, 'anchor': part['anchor'], 'positions': positions,
                          'rotation_rule': rule, 'window_draws': draws})
        new = {'step': step, 'segments': parts, 'predictions': item['predictions']}
        check_pair(item, new, lookup)
        rotating.append(new)
    return {'FIXED': fixed, 'ROTATING': rotating}


def check_pair(fixed, rotating, lookup):
    a, b = grouped(fixed), grouped(rotating)
    if list(a) != list(b) or {k: len(v) for k, v in a.items()} != {k: len(v) for k, v in b.items()}:
        raise ValueError('Paired task order or contribution differs')
    for item, groups in [(fixed, a), (rotating, b)]:
        if item['predictions'] != 32 or sum(map(len, groups.values())) != 32:
            raise ValueError('Every paired update must have exactly32 predictions')
        for tid, positions in groups.items():
            n = len(lookup[tid]['teacher_answer']['answer_ids'])
            if len(positions) != len(set(positions)) or not all(0 <= p < n for p in positions):
                raise ValueError('Duplicate, padding or unavailable prediction')


def validation_panels(histories):
    from .coding_training import OFFSETS, valid_positions
    result = {}
    for h in histories:
        ids = h['teacher_answer']['answer_ids']; n = len(ids)
        legacy = [p for a in OFFSETS for p in valid_positions(ids, a)]
        broad = set(range(min(8, n)))
        # Fifteen evenly spaced full eight-token windows, ending at the actual
        # last answer token. Short overlapping windows are deduplicated.
        for j in range(15):
            a = round(j * max(0, n - 8) / 14)
            broad.update(range(a, min(a + 8, n)))
        result[h['task_id']] = {'legacy': legacy, 'broad': sorted(broad)}
    return result


def seed_manifest(task_ids, count):
    streams = ['answer_small'] + [f'coverage_answer_{i}' for i in range(1, count)]
    return {tid: [{'seed_index': i, 'stream': stream, 'answer_seed': seed_for(tid, 0, stream)}
                  for i, stream in enumerate(streams)] for tid in task_ids}


def choose_endpoint(preflight, experiment_seconds=24*3600):
    """Timing-only rule frozen before task-quality generation. No losses used."""
    rows = preflight['representative_updates']
    mean_pair_seconds = sum(r['wall_seconds'] for r in rows) / len(rows)
    # Three stress updates are a memory/shape gate; all retained separately.
    if not preflight['memory_and_gradient_checks_passed']:
        raise ValueError('Preflight engineering failure needs inspection')
    forecasts = []
    for endpoint in [1024, 512, 256, 128]:
        checkpoint_count = 1 + 2 * sum(x <= endpoint for x in [128, 256, 512, 1024])
        training = 1.5 * mean_pair_seconds * endpoint
        validation = 1.5 * preflight['validation_seconds'] * checkpoint_count
        total = training + validation + 6*3600 + 90*60
        forecasts.append({'endpoint': endpoint, 'training_seconds': training,
                          'validation_seconds': validation, 'evaluation_reserve_seconds': 6*3600,
                          'bootstrap_and_export_reserve_seconds': 90*60, 'total_seconds': total})
    eligible = [r for r in forecasts if r['total_seconds'] <= experiment_seconds]
    if not eligible:
        raise ValueError('Even128 matched updates do not fit with evaluation reserve')
    return {'endpoint': eligible[0]['endpoint'], 'forecasts': forecasts,
            'chosen_before_task_quality_generation': True, 'quality_scores_consulted': False,
            'rule': 'Largest common checkpoint fitting 1.5x measured training/validation, plus6h evaluation and90min setup/export.',
            'preflight_sha256': digest(preflight)}


def validate_coverage_plan(plan, root):
    from .coding_parallel import verify_gate, confirmation_input
    root = Path(root); d = json.loads((root/DECLARATION).read_text())
    proof = verify_gate(root, 'coverage_declaration', plan['gates']['coverage_declaration'])
    if proof['inputs'].get(DECLARATION) != sha(root/DECLARATION) or proof['inputs'].get(OWNER) != sha(root/OWNER):
        raise ValueError('Coverage instructions unbound')
    if plan.get('scope_amendment') != 'coverage_generalization_v1' or plan.get('scientific_scope_unchanged') is not False:
        raise ValueError('Distinct coverage identity required')
    if len(plan['workers']) != 1 or any(confirmation_input(p) for p in plan['files']):
        raise ValueError('One paired worker; reserved confirmation forbidden')
    if d['selected_checkpoint_sha256'] != START_SHA or sha(root/d['selected_checkpoint']) != START_SHA:
        raise ValueError('Original selected96 required')
    if d['publication_commit'] != PUBLICATION or d['max_additional_gpu_hours'] != 32 or d['max_additional_usd'] != 200:
        raise ValueError('Publication or budget contract differs')
    if plan['task_ids'] != d['training_task_ids'] + d['validation_task_ids']:
        raise ValueError('Exact104/21 corpus required')
    for w in plan['workers']:
        if w['worker_fields'].get('declaration_sha256') != sha(root/DECLARATION):
            raise ValueError('Worker declaration differs')
    if plan['stage'] == 'coverage_preflight':
        if any('/private/' in p for p in plan['files']) or plan['workers'][0]['maximum_seconds'] > 7200:
            raise ValueError('Preflight has no hidden tests and at most2h')
    elif plan['stage'] in ['coverage_experiment', 'coverage_evaluation_recovery']:
        verify_gate(root, 'coverage_endpoint', plan['gates']['coverage_endpoint'])
        endpoint = json.loads((root/plan['workers'][0]['worker_fields']['endpoint_path']).read_text())
        if endpoint['endpoint'] not in [128,256,512,1024] or endpoint['quality_scores_consulted'] is not False:
            raise ValueError('Invalid common endpoint')
        if plan['stage'] == 'coverage_evaluation_recovery':
            verify_gate(root, 'coverage_recovery', plan['gates']['coverage_recovery'])
            fields = plan['workers'][0]['worker_fields']
            if sha(root/fields['recovery_path']) != fields['recovery_sha256']:
                raise ValueError('Recovery decision changed')
            receipt = json.loads((root/fields['recovery_path']).read_text())
            validate_recovery(root, receipt)
            if receipt['target_updates'] != endpoint['endpoint'] or plan['workers'][0]['maximum_seconds'] > 10*3600:
                raise ValueError('Recovery cannot change training target or exceed ten hours')
    else:
        raise ValueError('Unknown coverage stage')


def validate_recovery(root, receipt):
    """Accept only the latest durable paired state, never a loss-selected state."""
    root = Path(root)
    if receipt.get('training_resumed') is not False or receipt.get('quality_scores_consulted') is not False:
        raise ValueError('Recovery is evaluation only, selected before quality scores')
    source = root/receipt['source_result_root']
    source.resolve().relative_to((root/'results/coding_pilot_v1').resolve())
    for name in ['training_steps.json','training_failure.json','validation_curve.json','training_identity.json',
                 'optimizer_reset.json','frozen_schedule_prefix.json','frozen_endpoint.json','answer_span_audit.json','planned_exposure.json']:
        if str((source/name).relative_to(root)) not in receipt['inputs']:
            raise ValueError('Unbound original training record: '+name)
    for rel, expected in receipt['inputs'].items():
        p = root/rel
        p.resolve().relative_to(root.resolve())
        if not p.is_file() or p.is_symlink() or sha(p) != expected:
            raise ValueError('Recovery evidence or mapper changed: '+rel)
    pairs = []
    for p in (source/'paired_checkpoints').glob('step_*.json'):
        if str(p.relative_to(root)) not in receipt['inputs']:
            raise ValueError('Unbound paired checkpoint')
        row = json.loads(p.read_text())
        if row.get('paired_commit') is not True or set(row['arms']) != set(ARMS):
            raise ValueError('Incomplete paired checkpoint')
        pairs.append(row)
    if not pairs:
        raise ValueError('No durable paired checkpoint')
    chosen = max(pairs, key=lambda p:p['step'])
    if chosen['step'] <= 0 or chosen['step'] != receipt['primary_common_checkpoint']:
        raise ValueError('Must evaluate latest durable nonzero common checkpoint')
    for arm in ARMS:
        cp = chosen['arms'][arm]
        if receipt['inputs'].get(cp['path']) != cp['sha256'] or sha(root/cp['path']) != cp['sha256']:
            raise ValueError('Paired mapper identity differs')
    steps = json.loads((source/'training_steps.json').read_text())
    if [r['step'] for r in steps] != list(range(1,len(steps)+1)) or len(steps) != receipt['completed_logged_updates']:
        raise ValueError('Training log sequence differs')
    if not chosen['step'] <= len(steps) < receipt['target_updates']:
        raise ValueError('Invalid incomplete training schedule')
    for row in steps:
        if row['normalization_per_arm'] != 32 or set(row['arms']) != set(ARMS):
            raise ValueError('Unmatched saved training update')
        if any(a['gradient_predictions'] != 32 or a['cache_gradients_finite_nonzero'] is not True for a in row['arms'].values()):
            raise ValueError('Saved training gradient checks failed')
    for scope in ['validation','seen_training']:
        if list((source/scope).glob('*/*/seed_*/answer.json')):
            raise ValueError('Unexpected prior quality generation in recovery source')
    return chosen
