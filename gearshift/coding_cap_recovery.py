"""Recover the single interrupted cap amendment without another scientific draw."""
import json
from pathlib import Path

from gearshift.coding_cap_amendment import NEW_CAP, replay_constraints, validate_parent
from gearshift.coding_control import digest, sha
from gearshift.coding_resume import load_parent, verify_prefix
from gearshift.coding_reuse import compatible_parent, verify_completed


def validate_recovery(parent_root, original_root, repo):
    parent_root, original_root, repo = map(Path, (parent_root, original_root, repo))
    original = validate_parent(original_root, repo)
    parent = compatible_parent(parent_root, repo)
    declaration = repo / 'configs/coding_pilot_v1/cap_amendment_v1.json'
    if (parent.get('reasoning_cap') != NEW_CAP or parent.get('cap_amendment_number') != 1
            or parent.get('cap_amendment_sha256') != sha(declaration)
            or parent.get('parent_identity_sha256') != sha(original_root / 'identity.json')):
        raise ValueError('Recovery must preserve the original single cap amendment')
    if (parent_root / 'baseline_gate.json').exists():
        raise ValueError('A completed scientific outcome is not an interrupted run')
    ids = original['task_ids']
    folders = {p.name for p in (parent_root / 'tasks').iterdir()}
    if folders != {t.replace('/', '__') for t in ids}:
        raise ValueError('Interrupted cohort membership differs')
    complete_ids = []
    rows = []
    for tid in ids:
        folder = parent_root / 'tasks' / tid.replace('/', '__')
        if not (folder / 'complete.json').exists():
            continue
        row = verify_completed(folder, parent, tid)
        for kind in ['source', 'small']:
            history = json.loads((folder / (kind + '_history.json')).read_text())
            if history['reasoning_capped'] != row[kind + '_capped']:
                raise ValueError('Cap label differs from completed history')
            if history['reasoning_capped'] and len(history['reasoning_ids']) != NEW_CAP:
                raise ValueError('Amended capped history has the wrong token budget')
        complete_ids.append(tid)
        rows.append(row)
    # This recovery is bounded to the observed 39-case provider interruption.
    if complete_ids != ids[:39] or ids[-1] not in original['repeat_task_ids']:
        raise ValueError('This recovery requires the exact 39-case checkpoint')
    progress = json.loads((parent_root / 'progress.json').read_text())
    counts = {a: sum(r['pass'][a] for r in rows) for a in ['A', 'B', 'D']}
    if progress['completed_tasks'] != 39 or progress['passes'] != counts:
        raise ValueError('Interrupted progress differs from committed measurements')
    folder = parent_root / 'tasks' / ids[-1].replace('/', '__')
    for p in folder.glob('partial_*.json'):
        partial = json.loads(p.read_text())
        if partial['task_id'] != ids[-1] or partial['identity_sha256'] != digest(parent):
            raise ValueError('Interrupted token identity differs')
    prefixes, complete, _ = recovery_constraints(original_root, original['parent_identity'], parent_root, ids[-1])
    return {
        'parent_identity': parent, 'original_identity': original['parent_identity'],
        'task_ids': ids, 'completed_task_ids': complete_ids, 'remaining_task_ids': ids[39:],
        'counts': counts, 'source_cap_count': sum(r['source_capped'] for r in rows),
        'small_cap_count': sum(r['small_capped'] for r in rows),
        'parent_identity_file_sha256': sha(parent_root / 'identity.json'),
        'original_identity_file_sha256': sha(original_root / 'identity.json'),
        'saved_segment_lengths': {k: len(v) for k, v in prefixes.items()},
        'completed_segments': list(complete), 'cap_amendment_number': 1,
        'reasoning_cap': NEW_CAP,
    }


def recovery_constraints(original_root, original_identity, interrupted_root, task_id):
    prefixes, complete, lineage = replay_constraints(original_root, original_identity, task_id)
    recovered, recovered_complete = load_parent(interrupted_root, task_id)
    for segment, tokens in recovered.items():
        old = prefixes.get(segment, [])
        verify_prefix(tokens, old)
        if len(tokens) > len(old):
            prefixes[segment] = tokens
        if segment in complete and len(tokens) > len(complete[segment]['tokens']):
            raise ValueError('Interrupted stream extends a completed original segment')
    for segment, record in recovered_complete.items():
        verify_prefix(record['tokens'], prefixes.get(segment, []), final=True)
        if segment in complete and record['tokens'] != complete[segment]['tokens']:
            raise ValueError('Completed recovery segment differs from original')
        complete[segment] = record
    return prefixes, complete, {
        **lineage, 'interrupted_identity_file_sha256': sha(Path(interrupted_root) / 'identity.json'),
        'recovery_only': True, 'additional_cap_increase': False,
    }
