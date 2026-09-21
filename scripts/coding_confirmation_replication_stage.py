#!/usr/bin/env python3
"""Inventory/check compact replication inputs; never provision or launch a pod."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from gearshift.coding_control import digest, sha, write

CONFIG = 'configs/coding_pilot_v1/confirmation_01'
DECLARATION = CONFIG + '/replication_declaration.json'
BACKUP = 'evidence/coding_pilot_v1/coverage_generalization_v2_20260918T233720Z/backups/external_heavy_backup_manifest.json'
AUDIT = 'evidence/coding_pilot_v1/coverage_generalization_v2_20260918T233720Z/monitoring/independent_training_audit.json'


def read(path):
    return json.loads(Path(path).read_text())


def local_file(repo, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe staged relative path')
    path = Path(repo) / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError('Staged input missing or symlink: ' + str(relative))
    path.resolve().relative_to(Path(repo).resolve())
    return path


def make_manifest(repo=ROOT):
    repo = Path(repo); d = read(repo / DECLARATION)
    # Include compact source trees to preserve legacy bare-import dependencies.
    # These are code only: no dataset, environment, credentials or tensor glob.
    paths = {str(p.relative_to(repo)) for folder in ('gearshift', 'scripts')
             for p in (repo / folder).rglob('*.py') if '__pycache__' not in p.parts}
    paths.update({DECLARATION, CONFIG + '/OWNER_INSTRUCTION.txt',
                  CONFIG + '/replication_schedules.json', CONFIG + '/replication_plan.json',
                  CONFIG + '/replication_checkpoint_inventory.json',
                  CONFIG + '/replication_launch_contract.md',
                  'configs/coding_pilot_v1/pilot.json', d['prior_v2_declaration'],
                  d['corpus_manifest'], d['panels_path'], d['answer_seeds_path'], BACKUP, AUDIT,
                  *d['unchanged_source_files'], *d['numerical_implementation']})
    for optional in ('requirements.txt', 'requirements.lock.txt', 'pyproject.toml'):
        if (repo / optional).is_file(): paths.add(optional)
    corpus = read(repo / d['corpus_manifest'])
    compact = {k: v for k, v in corpus['files'].items() if not k.endswith('.pt')}
    paths.update(compact)
    files = {}
    for relative in sorted(paths):
        path = local_file(repo, relative)
        receipt = {'sha256': sha(path), 'bytes': path.stat().st_size}
        if relative in compact and receipt != compact[relative]:
            raise ValueError('Frozen source corpus input differs: ' + relative)
        files[relative] = receipt
    original = next(x for x in read(repo / BACKUP)['files']
                    if x['runtime_relative_path'] == d['selected_checkpoint'])
    if original['sha256'] != d['selected_checkpoint_sha256'] or not original['verified']:
        raise ValueError('Original selected96 backup identity differs')
    return {'schema': 1, 'experiment_id': d['experiment_id'], 'role': 'replication_train',
            'declaration_sha256': sha(repo / DECLARATION), 'files': files,
            'files_sha256': digest(files), 'compact_file_count': len(files),
            'compact_bytes': sum(x['bytes'] for x in files.values()),
            'external_files': {d['selected_checkpoint']: {
                'sha256': original['sha256'], 'bytes': original['bytes'],
                'backup_path': original['backup_path']}},
            'model_weights': read(repo / 'configs/coding_pilot_v1/pilot.json')['models'],
            'model_cache': '/workspace/hf',
            'private_tests_and_confirmation_answers_included': False,
            'paired_feature_or_kv_tensors_included': False,
            'post_snapshot_source_changes_require_fresh_manifest': True}


def verify_manifest(repo, manifest, require_initializer=True):
    repo = Path(repo)
    if digest(manifest['files']) != manifest['files_sha256']:
        raise ValueError('Stage inventory content hash differs')
    entries = dict(manifest['files'])
    if require_initializer: entries.update(manifest['external_files'])
    for relative, expected in entries.items():
        path = local_file(repo, relative)
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('Staged input checksum differs: ' + relative)
    return {'verified': True, 'files_verified': len(entries),
            'initializer_verified': require_initializer,
            'model_weights_need_separate_pinned_shard_verification': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory', 'verify'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--compact-only', action='store_true',
                        help='Local package check only; do not use for launch approval.')
    args = parser.parse_args(argv)
    if args.action == 'inventory':
        result = make_manifest(); write(Path(args.manifest), result)
        print(json.dumps({k: result[k] for k in ('compact_file_count', 'compact_bytes', 'files_sha256')}))
    else:
        print(json.dumps(verify_manifest(ROOT, read(args.manifest), not args.compact_only)))


if __name__ == '__main__': main()
