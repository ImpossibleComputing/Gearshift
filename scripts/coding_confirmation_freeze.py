#!/usr/bin/env python3
"""Bind the confirmation recipe after an explicit owner task-scope decision.

This does not generate, train, score, provision, or infer a decision from silence.
Existing frozen files are verified for equality and are never overwritten.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, digest, sha

CONFIG = 'configs/coding_pilot_v1/confirmation_01'
EVIDENCE = 'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z'
SCORING_IMPLEMENTATION = [
    'scripts/coding_confirmation_score.py', 'scripts/coding_confirmation_report.py',
    'scripts/coding_scorer_repair_calibrate.py',
]
IMPLEMENTATION = [
    'gearshift/core.py', 'gearshift/coding_control.py', 'gearshift/coding_inference.py',
    'gearshift/coding_gradients.py', 'gearshift/coding_post_progress.py',
    'gearshift/coding_confirmation_sampling.py', 'gearshift/coding_confirmation_lease.py',
    'scripts/coding_confirmation_runtime.py', 'scripts/coding_confirmation_generate.py',
    'scripts/coding_confirmation_freeze.py', 'scripts/coding_coverage_v2_worker.py',
    'scripts/coding_confirmation_generation_supervisor.py',
] + SCORING_IMPLEMENTATION


def read(path):
    return json.loads(Path(path).read_text())


def resolve_tasks(manifest, resolution):
    if resolution.get('authority') != 'direct_owner_reply' or not resolution.get('owner_reply_text'):
        raise ValueError('Explicit recorded owner reply is required')
    choice = resolution.get('decision')
    if choice not in ('all_200', 'protect_40'):
        raise ValueError('Task scope has not been resolved')
    tids = manifest['task_ids']
    reserve = set(manifest['original_reserved40_subset_ids'])
    if len(tids) != 200 or len(set(tids)) != 200 or len(reserve) != 40 or not reserve <= set(tids):
        raise ValueError('Original200/40 membership differs from the audited conflict')
    selected = tids if choice == 'all_200' else [t for t in tids if t not in reserve]
    return list(selected)


def private_identity_from_gate(gate, original_ids, selected_ids, experiment_id):
    """Declare historical private-file bytes without loading any hidden tests.

    The original file is retained intact on the private scorer. Only resolved
    task IDs may be executed; preserving40 does not require rewriting that file.
    """
    original = [r['task_id'] for r in gate['rows'] if r['split'] == 'confirmation']
    private_sha = gate['identity']['private_files']['confirmation']
    if (gate.get('passed') is not True or len(original) != 200 or
            len(set(original)) != 200 or set(original) != set(original_ids) or
            not selected_ids or len(set(selected_ids)) != len(selected_ids) or
            not set(selected_ids) <= set(original) or not re.fullmatch('[0-9a-f]{64}', private_sha)):
        raise ValueError('Historical private-file identity or selected task scope differs')
    return {'experiment_id': experiment_id, 'status': 'FROZEN',
        'task_ids': list(selected_ids), 'task_count': len(selected_ids),
        'private_file_task_ids': list(original_ids), 'private_file_task_count': 200,
        'private_tests_sha256': private_sha, 'private_file_rewritten': False,
        'execution_scope': 'Only task_ids; no unselected task may be scored.',
        'private_test_values_included': False,
        'provenance': 'Exact original private confirmation file pinned by the historical data gate.'}


def freeze(resolution_path, repo=ROOT):
    repo = Path(repo).resolve()
    resolution_path = Path(resolution_path)
    if not resolution_path.is_absolute(): resolution_path = repo / resolution_path
    resolution_rel = str(resolution_path.resolve().relative_to(repo))
    resolution = read(resolution_path)
    original = read(repo / CONFIG / 'task_manifest.json')
    tids = resolve_tasks(original, resolution)
    d = read(repo / CONFIG / 'declaration.draft.json')
    if resolution.get('experiment_id') != d['experiment_id']:
        raise ValueError('Owner decision refers to another experiment')
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    for relative in IMPLEMENTATION:
        committed = subprocess.check_output(['git', 'show', code_commit + ':' + relative], cwd=repo)
        if committed != (repo / relative).read_bytes():
            raise ValueError('Commit the tested implementation before freezing: ' + relative)

    policy_path = EVIDENCE + '/scorer_repair/calibration/frozen_policy.json'
    identity_path = EVIDENCE + '/scorer_repair/calibration/scorer_identity.json'
    calibration_path = EVIDENCE + '/scorer_repair/calibration/calibration_receipt.json'
    policy, identity, calibration = [read(repo / p) for p in (policy_path, identity_path, calibration_path)]
    if policy.get('policy_status') != 'frozen' or calibration.get('passed') is not True:
        raise ValueError('Successful frozen scorer calibration is required')
    if digest(policy) != identity['policy_sha256'] or sha(repo / calibration_path) != policy['calibration_receipt_sha256']:
        raise ValueError('Calibrated policy identity differs')
    for path, expected in identity['files'].items():
        if sha(repo / path) != expected: raise ValueError('Frozen scorer source differs: ' + path)

    proof_path = EVIDENCE + '/checkpoint_byte_verification.json'
    proof = read(repo / proof_path)
    observed = {p['sha256'] for p in proof['files']}
    for cp in d['primary_checkpoints'].values():
        if cp['mapper_sha256'] not in observed: raise ValueError('Actual mapper byte proof missing')
        cp['actual_checkpoint_byte_proof'] = proof_path
    if d['secondary_training']['initializer_sha256'] not in observed:
        raise ValueError('Original initializer byte proof missing')

    gate_path = 'evidence/coding_pilot_v1/data_gate.json'
    gate = read(repo / gate_path)
    if gate['identity']['visible_files']['confirmation'] != original['visible_input_expected_sha256']:
        raise ValueError('Historical gate and public confirmation file differ')
    private_identity = private_identity_from_gate(gate, original['task_ids'], tids, d['experiment_id'])
    private_identity.update(historical_gate_path=gate_path, historical_gate_sha256=sha(repo / gate_path))
    private_identity_path = CONFIG + '/private_test_identity.json'
    bind(repo / private_identity_path, private_identity)

    resolved = {'experiment_id': d['experiment_id'], 'status': 'FROZEN', 'task_ids': tids,
        'task_count': len(tids), 'original_manifest_path': CONFIG + '/task_manifest.json',
        'original_manifest_sha256': sha(repo / CONFIG / 'task_manifest.json'),
        'resolution_path': resolution_rel, 'resolution_sha256': sha(resolution_path),
        'original_reserved40_used': resolution['decision'] == 'all_200',
        'task_substitution_or_resampling': False}
    seeds = read(repo / CONFIG / 'seeds.draft.json')
    seeds.update(status='FROZEN', task_selection='Exact owner-resolved frozen manifest',
                 tasks={t: seeds['tasks'][t] for t in tids})
    analysis = read(repo / CONFIG / 'analysis_plan.draft.json')
    analysis.update(status='FROZEN', task_count=len(tids))
    protocol = read(repo / CONFIG / 'protocol.draft.json')
    protocol['status'] = 'FROZEN'
    for name, value in [('tasks.json', resolved), ('seeds.json', seeds),
                        ('analysis_plan.json', analysis), ('protocol.json', protocol)]:
        bind(repo / CONFIG / name, value)
    visible_path = 'data/coding_pilot_v1/visible/confirmation.json'
    if sha(repo / visible_path) != original['visible_input_expected_sha256']:
        raise ValueError('Frozen public confirmation file differs')
    paths = [CONFIG + '/' + n for n in ('tasks.json', 'seeds.json', 'analysis_plan.json', 'protocol.json')]
    paths += [visible_path, 'configs/coding_pilot_v1/pilot.json', resolution_rel, proof_path,
              gate_path, private_identity_path,
              policy_path, identity_path, calibration_path,
              CONFIG + '/replication_declaration.json', CONFIG + '/replication_schedules.json']
    paths += list(d['exposure_audit_hashes']) + list(identity['files'])
    d.update(status='FROZEN', generation_authorized_by_this_file=True,
        task_count=len(tids), task_ids=tids, primary_answer_count=24*len(tids),
        secondary_answer_count=12*len(tids), task_scope_resolution=resolution,
        seeds_path=CONFIG + '/seeds.json', visible_path=visible_path,
        model_config_path='configs/coding_pilot_v1/pilot.json',
        protocol_path=CONFIG + '/protocol.json', analysis_path=CONFIG + '/analysis_plan.json',
        source_commit=code_commit, implementation={p: sha(repo / p) for p in IMPLEMENTATION},
        inputs={p: sha(repo / p) for p in paths}, pending_finalization=[],
        scorer={'version': identity['scorer_version'], 'policy_path': policy_path,
                'policy_sha256': sha(repo / policy_path), 'policy_identity_sha256': identity['policy_sha256'],
                'identity_path': identity_path, 'identity_file_sha256': sha(repo / identity_path),
                'implementation_hashes': {**identity['files'], **{p: sha(repo / p) for p in SCORING_IMPLEMENTATION}},
                'private_tests_sha256': private_identity['private_tests_sha256'],
                'private_test_identity_path': private_identity_path,
                'calibration_receipt': calibration_path,
                'calibration_sha256': sha(repo / calibration_path), 'frozen_before_confirmation_scoring': True})
    d['secondary_training'].update(schedules_path=CONFIG + '/replication_schedules.json',
        schedules_sha256=sha(repo / CONFIG / 'replication_schedules.json'))
    if len(tids) != 200:
        d['secondary_training']['evaluation_population_amendment'] = (
            'Owner task-scope decision supersedes the conditional200-task evaluation count in the earlier training declaration. '
            'The already frozen training recipe, seed, initializer and updates are unchanged.')
    destination = repo / CONFIG / 'declaration.json'
    bind(destination, d)
    return {'declaration_path': str(destination), 'declaration_sha256': sha(destination),
            'task_count': len(tids), 'primary_answers': d['primary_answer_count'],
            'secondary_answers': d['secondary_answer_count']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner-resolution', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.owner_resolution), indent=2))
