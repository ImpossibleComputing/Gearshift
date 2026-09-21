#!/usr/bin/env python3
"""Frozen second-seed adapter for the unchanged coverage-v2 training path.

The primary confirmation pair is never trained here. This separate pair starts
at the original selected96 tensors and changes only the declared training seed
and its deterministic paired task/window schedules. No scores/private tests are
inputs, and this module contains no allocation or spending-policy implementation.
"""
from __future__ import annotations
import argparse
import copy
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from gearshift.coding_control import bind, digest, sha, write
from gearshift.coding_coverage import ARMS, START_SHA, PUBLICATION, paired_schedules, check_pair

EXPERIMENT = 'confirmation_01_20260919T094418Z'
SEED = 20260919
OLD_SEED = 20260915
SKIP = 96
UPDATES = 1024
CHECKPOINTS = [0, 128, 256, 512, 768, 1024]
CONFIG = 'configs/coding_pilot_v1/confirmation_01'
DECLARATION = CONFIG + '/replication_declaration.json'
SCHEDULES = CONFIG + '/replication_schedules.json'
PLAN = CONFIG + '/replication_plan.json'
INVENTORY = CONFIG + '/replication_checkpoint_inventory.json'
BASE = 'configs/coding_pilot_v1/coverage_generalization_v2/declaration.json'
CORPUS = 'results/coding_pilot_v1/partial_corpus_20260917_v1/corpus_manifest.json'
RESULT = 'results/coding_pilot_v1/' + EXPERIMENT + '/replication'
UNCHANGED = (
    'gearshift/coding_coverage.py', 'gearshift/coding_coverage_runtime.py',
    'gearshift/coding_coverage_v2_checkpoint.py', 'scripts/coding_coverage_v2_worker.py',
    'scripts/coding_coverage_v2_train.py', 'scripts/coding_coverage_experiment.py',
    'scripts/coding_partial_corpus.py',
)
RETAIN = (
    'publication_commit', 'corpus_manifest', 'corpus_manifest_sha256', 'training_task_ids',
    'validation_task_ids', 'corpus_limitation', 'selected_checkpoint', 'selected_checkpoint_sha256',
    'model_config_sha256', 'unchanged_source_files', 'optimizer', 'target_additional_updates',
    'checkpoint_updates', 'panels_path', 'panels_sha256', 'answer_seeds_path', 'answer_seeds_sha256',
    'fixed', 'rotating', 'validation_panels',
)


def read(path):
    return json.loads(Path(path).read_text())


def load_training_inputs(repo=ROOT):
    from coding_partial_corpus import load_corpus
    repo = Path(repo); base = read(repo / BASE)
    if base['publication_commit'] != PUBLICATION or base['selected_checkpoint_sha256'] != START_SHA:
        raise ValueError('Original publication or selected96 identity changed')
    if sha(repo / 'configs/coding_pilot_v1/pilot.json') != base['model_config_sha256']:
        raise ValueError('Pinned model/decoding configuration changed')
    if base['corpus_manifest'] != CORPUS or sha(repo / CORPUS) != base['corpus_manifest_sha256']:
        raise ValueError('Frozen104-history corpus identity changed')
    histories, _, _ = load_corpus(repo, repo / CORPUS, features=False)
    training = [h for h in histories if h['split'] == 'training']
    validation = [h for h in histories if h['split'] == 'validation']
    if len(training) != 104 or [h['task_id'] for h in training] != base['training_task_ids']:
        raise ValueError('Replication must use the same ordered104 training histories')
    if len(validation) != 21 or [h['task_id'] for h in validation] != base['validation_task_ids']:
        raise ValueError('Legacy setup validation membership changed')
    return base, training, validation


def schedule_proof(training, schedules):
    lookup = {h['task_id']: h for h in training}
    if set(schedules) != set(ARMS) or any(len(schedules[a]) != UPDATES for a in ARMS):
        raise ValueError('Expected exactly1024 updates in each secondary arm')
    for index in range(UPDATES):
        fixed, rotating = schedules['FIXED'][index], schedules['ROTATING'][index]
        if fixed['step'] != index + 1 or rotating['step'] != index + 1:
            raise ValueError('Secondary schedule update positions are not contiguous')
        check_pair(fixed, rotating, lookup)
    return {'training_seed': SEED, 'previous_training_seed': OLD_SEED,
            'updates_per_arm': UPDATES, 'gradient_positions_per_update': 32,
            'gradient_positions_per_arm': UPDATES * 32, 'training_tasks': len(training),
            'paired_task_order_and_contributions_verified': True,
            'same_missing_position_and_rotation_rules': True,
            'schedule_position_offset': SKIP,
            'recipe': 'Existing paired_schedules(histories,updates=1024,seed=20260919,skip=96); only seed changes.'}


def freeze(repo=ROOT):
    repo = Path(repo); base, training, validation = load_training_inputs(repo)
    schedules = paired_schedules(training, updates=UPDATES, seed=SEED, skip=SKIP)
    proof = schedule_proof(training, schedules)
    # Immutable creation: rerunning may verify identical bytes/content, never
    # replace an existing frozen seed declaration after observing outcomes.
    bind(repo / SCHEDULES, schedules)
    declaration = {k: copy.deepcopy(base[k]) for k in RETAIN}
    declaration.update(schema=1, experiment_id=EXPERIMENT, replication_id='replication_seed_20260919',
        role='secondary_training_seed_replication', training_seed=SEED, prior_training_seed=OLD_SEED,
        owner_instruction_sha256=sha(repo / CONFIG / 'OWNER_INSTRUCTION.txt'),
        prior_v2_declaration=BASE, prior_v2_declaration_sha256=sha(repo / BASE),
        schedules_path=SCHEDULES, schedules_sha256=sha(repo / SCHEDULES),
        task_order='Same paired_schedules construction and96-update index offset as v2; new seed20260919 changes task permutation and rotating window order jointly across the pair.',
        result_root=RESULT, target_additional_updates=UPDATES, checkpoint_updates=CHECKPOINTS,
        optimizer_state='Fresh identical AdamW/constant scheduler/RNG in both arms; original selected96 has no optimizer state. Complete subsequent checkpoints use unchanged v2 atomic save/resume implementation.',
        numerical_implementation={p: sha(repo / p) for p in UNCHANGED},
        schedule_proof=proof,
        preflight={'actual_model_exact_resume_required': True, 'updates_before_interrupt': 2,
                   'compared_next_update': 3, 'pristine_all_state_restore_required': True,
                   'preflight_updates_do_not_enter_main_training': True},
        evaluation={'generation_in_this_runner': False, 'confirmation_tasks': 200, 'answer_seeds_per_task': 3,
                    'forms': ['FIXED_M', 'ROTATING_M', 'FIXED_H', 'ROTATING_H'], 'expected_answers': 2400,
                    'checkpoint': 1024, 'answer_seed_policy': 'Reuse the exact primary confirmation task/stream/seed manifest; training seed never enters answer-seed derivation.',
                    'source_history_policy': 'Reuse the exact frozen primary large-model reasoning histories/native controls; no new reasoning draw for the secondary models.',
                    'primary_confirmation_does_not_wait_for_this_secondary': True,
                    'validation_setup_note': 'Inherited21-task panels/answer-seed file are integrity inputs for unchanged setup only; training runner generates no validation outputs or scores.'},
        cache_reuse={'saved_histories_reused': True, 'compatible_persisted_v2_kv_artifacts': False,
                     'tensor_cache_reuse_enabled': False, 'prefix_chunk_size': 512,
                     'reason': 'V2 persisted no full KV tensors. Earlier DiskLRU identities lack pinned model revisions, so no older tensor cache is admitted. Retain exact full-prefix reconstruction without adding a cache optimization experiment.',
                     'large_transient_caches_persisted': False},
        resource_policy={'provider_lifecycle_in_this_runner': False, 'parent_owns_budget_and_allocation_guards': True,
                         'old_v2_1000_usd_lease_not_used': True, 'execution': 'Two equivalent H200 processes in parallel, one visible GPU per arm.',
                         'soft_spending_target_is_not_a_training_kill_switch': True},
        confirmation_private_tests_are_training_inputs=False, reserved_40_tasks_used=False,
        comparison_scope='Two optimization seeds conditional on a common original initializer/corpus; no seed selection or sweep.')
    bind(repo / DECLARATION, declaration)
    inventory = checkpoint_inventory(declaration)
    bind(repo / INVENTORY, inventory)
    plan = {'schema': 1, 'experiment_id': EXPERIMENT, 'role': 'secondary_training_seed_replication',
            'declaration_path': DECLARATION, 'declaration_sha256': sha(repo / DECLARATION),
            'result_root': RESULT, 'training_seed': SEED, 'target_updates': UPDATES,
            'checkpoint_updates': CHECKPOINTS, 'starting_mapper': inventory['original_initializer'],
            'checkpoint_inventory_path': INVENTORY, 'checkpoint_inventory_sha256': sha(repo / INVENTORY),
            'required_worker_api': 'run_replication(c,arm); c supplies local guard/publish/telemetry and code_commit.',
            'parallel_arms': [{'arm': a, 'one_visible_H200_required': True} for a in ARMS],
            'numerical_implementation': declaration['numerical_implementation'],
            'hidden_tests_or_confirmation_answers_permitted': False}
    bind(repo / PLAN, plan)
    return {'declaration': DECLARATION, 'declaration_sha256': sha(repo / DECLARATION),
            'schedules': SCHEDULES, 'schedules_sha256': sha(repo / SCHEDULES),
            'plan': PLAN, 'plan_sha256': sha(repo / PLAN), 'proof': proof}


def checkpoint_inventory(d):
    return {'experiment_id': EXPERIMENT, 'replication_id': d['replication_id'],
        'original_initializer': {'path': d['selected_checkpoint'], 'sha256': START_SHA,
            'retrieval_path': '/Users/qeetbastudio/Projects/Gearshift-heavy-backups/coverage_generalization_v2_20260918T233720Z/original_selected96/mapper_step_0096.pt',
            'verification': 'Saved prior off-pod backup inventory; worker rehashes actual staged bytes and verifies tensor equality before any update.'},
        'expected_outputs': [{'arm': arm, 'step': step,
            'directory': RESULT + f'/arms/{arm}/checkpoints/step_{step:04d}',
            'mapper_file': 'mapper.pt', 'full_resumable_file': 'full.pt', 'manifest_file': 'manifest.json',
            'hash_status': 'Generated only after immutable atomic checkpoint commits; never invented in advance.'}
            for arm in ARMS for step in CHECKPOINTS],
        'required_full_state': ['mapper', 'AdamW', 'constant_scheduler', 'Python_RNG', 'NumPy_RNG',
            'torch_CPU_RNG', 'torch_CUDA_RNG', 'schedule_position', 'training_log', 'code/config/corpus/model_identities'],
        'retention': 'All six common checkpoints per arm, plus separate actual-model resume-preflight evidence.',
        'primary_mapper_weights_never_overwritten': True}


def verify(repo=ROOT):
    repo = Path(repo); base, training, validation = load_training_inputs(repo)
    d = read(repo / DECLARATION); plan = read(repo / PLAN)
    if d.get('training_seed') != SEED or d.get('prior_training_seed') != OLD_SEED or d['training_seed'] == OLD_SEED:
        raise ValueError('The one predeclared new training seed changed')
    if d.get('role') != 'secondary_training_seed_replication' or d.get('result_root') != RESULT:
        raise ValueError('Replication identity/result namespace changed')
    if d['schedules_path'] != SCHEDULES or d['schedules_sha256'] != sha(repo / SCHEDULES):
        raise ValueError('Frozen secondary schedules changed')
    if d['prior_v2_declaration_sha256'] != sha(repo / BASE):
        raise ValueError('Original v2 declaration changed')
    if d['owner_instruction_sha256'] != sha(repo / CONFIG / 'OWNER_INSTRUCTION.txt'):
        raise ValueError('Owner instruction snapshot changed')
    for key in RETAIN:
        if key not in ('checkpoint_updates',) and d[key] != base[key]:
            raise ValueError('Replication changed the frozen recipe: ' + key)
    for key, expected in d['numerical_implementation'].items():
        if sha(repo / key) != expected:
            raise ValueError('Unchanged v2 training implementation differs: ' + key)
    schedules = read(repo / SCHEDULES)
    if schedules != paired_schedules(training, updates=UPDATES, seed=SEED, skip=SKIP):
        raise ValueError('Saved schedules do not equal the declared original recipe with new seed')
    schedule_proof(training, schedules)
    if d['checkpoint_updates'] != CHECKPOINTS or d['target_additional_updates'] != UPDATES:
        raise ValueError('Final1024 endpoint/common checkpoint schedule changed')
    if plan['declaration_sha256'] != sha(repo / DECLARATION) or plan['starting_mapper']['sha256'] != START_SHA:
        raise ValueError('Replication plan declaration/initializer mismatch')
    if plan['checkpoint_inventory_sha256'] != sha(repo / INVENTORY):
        raise ValueError('Replication checkpoint inventory changed')
    return d, plan


def make_checkpoint_identity(d, code_commit, arm):
    if arm not in ARMS or not isinstance(code_commit, str) or not re.fullmatch(r'[0-9a-f]{40}', code_commit):
        raise ValueError('Exact source commit and declared arm are required')
    return {'experiment_id': EXPERIMENT + '/replication', 'arm': arm, 'code_commit': code_commit,
            'config_sha256': sha(ROOT / DECLARATION), 'schedules_sha256': d['schedules_sha256'],
            'corpus_sha256': d['corpus_manifest_sha256'], 'selected_checkpoint_sha256': START_SHA,
            'models': read(ROOT / 'configs/coding_pilot_v1/pilot.json')['models'],
            'configuration': {'optimizer': d['optimizer'], 'dtype': 'bfloat16', 'attention': 'sdpa',
                'gradient_positions': 32, 'full_context': True, 'scheduler': 'constant',
                'training_seed': SEED, 'schedule_position_offset': SKIP,
                'replication_id': d['replication_id']}}


def run_replication(c, arm):
    """Use the parent's allocation-only context, never the old v2 dollar cap.

    Required parent fields: guard, publish, telemetry, identity, status_root,
    code_commit. The root is checked/replaced with this arm's new namespace.
    Parent may supply resume_checkpoint to continue the last complete save.
    """
    d, plan = verify(ROOT)
    if arm not in ARMS:
        raise ValueError('Unknown replication arm')
    code_commit = c.get('code_commit') or c.get('spec', {}).get('code_commit') or c.get('plan', {}).get('code_commit')
    if not code_commit:
        raise ValueError('Parent training context is missing source code_commit')
    for field in ('guard', 'publish', 'telemetry', 'identity', 'status_root'):
        if field not in c:
            raise ValueError('Parent training context is missing: ' + field)
    root = ROOT / RESULT / 'arms' / arm
    if 'root' in c and Path(c['root']).resolve() != root.resolve():
        raise ValueError('Parent replication output root differs; historical artifacts cannot be overwritten')
    context = {**c, 'root': root, 'repo_root': ROOT, 'declaration': d, 'plan': plan,
        'config': read(ROOT / 'configs/coding_pilot_v1/pilot.json'),
        'declaration_path': DECLARATION, 'declaration_sha256': sha(ROOT / DECLARATION),
        'spec': {'arm': arm, 'code_commit': code_commit, 'training_seed': SEED,
                 'target_updates': UPDATES, 'checkpoint_updates': CHECKPOINTS,
                 'declaration_sha256': sha(ROOT / DECLARATION),
                 'checkpoint_identity': make_checkpoint_identity(d, code_commit, arm)},
    }
    # Existing setup asks for prior numerical-control provenance. Bind the same
    # already audited evidence; no private scores enter setup or optimization.
    context['plan'] = {**plan, 'prior_controls': {
        'source_declaration': BASE, 'source_declaration_sha256': d['prior_v2_declaration_sha256'],
        'training_audit': 'evidence/coding_pilot_v1/coverage_generalization_v2_20260918T233720Z/monitoring/independent_training_audit.json'}}
    if c.get('resume_checkpoint'):
        context['spec']['resume_checkpoint'] = c['resume_checkpoint']
    from coding_coverage_v2_worker import setup
    from coding_coverage_v2_train import train
    state = setup(context, arm)
    return train(context, state, arm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'verify'])
    args = parser.parse_args()
    print(json.dumps(freeze() if args.action == 'freeze' else {'verified': True, 'declaration': verify()[0]['replication_id']}, indent=2))


if __name__ == '__main__': main()
