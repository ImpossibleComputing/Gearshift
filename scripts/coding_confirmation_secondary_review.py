#!/usr/bin/env python3
"""Package the separate second-seed report from saved public records only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import coding_confirmation_review as public
from scripts import coding_confirmation_secondary_report as secondary

ZIP_NAME = 'gearshift_confirmation_secondary_review.zip'
REPORT_NAME = 'CONFIRMATION_SECONDARY_RESULTS.md'


def verify_report(plan_path, primary_plan, repo=ROOT):
    """Regenerate every secondary report file without executing candidates."""
    repo = Path(repo).resolve()
    plan = public.read(public.public_path(repo, plan_path))
    root = public.reporter.scoped(repo, plan['result_root'])
    saved = root / 'secondary/report'
    manifest = public.read(public.public_path(repo, str((saved / 'FILE_MANIFEST.json').relative_to(repo))))
    names = {row['path'] for row in manifest['files']}
    required = {'summary.json', REPORT_NAME, 'per_draw.csv', 'per_task.csv',
                'gains_losses.csv', 'joint_bootstrap_indices.json'}
    if (manifest.get('cohort') != 'secondary' or manifest.get('training_seed') != 20260919
            or len(names) != len(manifest['files']) or not required <= names
            or any(len(public.relative(name).parts) != 1 for name in names)):
        raise ValueError('Secondary report manifest identity or membership differs')
    if {p.name for p in saved.iterdir()} != names | {'FILE_MANIFEST.json'}:
        raise ValueError('Unexpected secondary report files')
    for row in manifest['files']:
        path = public.public_path(repo, str((saved / row['path']).relative_to(repo)))
        if path.stat().st_size != row['bytes'] or public.sha(path) != row['sha256']:
            raise ValueError('Secondary report file changed: ' + row['path'])
    with tempfile.TemporaryDirectory(prefix='gearshift-secondary-report-check-') as scratch:
        regenerated = Path(scratch) / 'report'
        summary = secondary.report(plan_path, repo=repo, primary_plan=primary_plan,
                                   output=regenerated)
        if summary['native_reference_draws_reused'] != 12 * summary['task_clusters']:
            raise ValueError('All four original primary native controls are required')
        for path in regenerated.iterdir():
            if path.is_symlink() or not (saved / path.name).is_file() or path.read_bytes() != (saved / path.name).read_bytes():
                raise ValueError('Secondary report differs from frozen saved-record regeneration: ' + path.name)
        if {p.name for p in regenerated.iterdir()} != names | {'FILE_MANIFEST.json'}:
            raise ValueError('Regenerated secondary report population differs')
    return plan, root, summary


def export(plan_path, primary_plan, primary_delivery, delivery, *, repo=ROOT):
    repo = Path(repo).resolve(); out = Path(delivery).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError('Delivery already exists; choose a new directory')
    plan, root, summary = verify_report(plan_path, primary_plan, repo)
    files, loaded, primary_summary, _, _ = public.gate(repo, primary_plan)
    d = loaded['declaration']; result_name = str(root.relative_to(repo))
    if (summary['experiment_id'] != d['experiment_id'] or summary['task_ids'] != d['task_ids']
            or summary['primary_mapper_scores_pooled_or_selected'] is not False):
        raise ValueError('Secondary report is not a separate analysis of the same tasks')
    original = Path(primary_delivery).resolve()
    original_receipt = public.read(original / 'DELIVERY_RECEIPT.json')
    if (original_receipt.get('experiment_id') != d['experiment_id']
            or original_receipt.get('all_outputs_verified') is not True
            or public.sha(original / public.ZIP_NAME) != original_receipt['archive_sha256']):
        raise ValueError('Previously delivered primary archive identity differs')
    original_manifest = public.read(original / public.MANIFEST)
    if public.sha(original / public.MANIFEST) != original_receipt['inventory_sha256']:
        raise ValueError('Original primary inventory identity differs')
    old_files = {r['path']: r for r in original_manifest['files']}
    preserved = {}
    for name in ('CONFIRMATION_RESULTS.md', 'SCORER_REPAIR_RESULTS.md'):
        data = (original / name).read_bytes(); row = old_files[name]
        if len(data) != row['bytes'] or public.hash_bytes(data) != row['sha256']:
            raise ValueError('Original delivered report changed: ' + name)
        preserved[name] = data
    evidence_name = 'evidence/coding_pilot_v1/' + d['experiment_id']
    for name in ('gearshift', 'scripts', 'tests', 'configs', evidence_name, result_name):
        files.tree(name)
    for name in ('README.md', 'RESULTS.md', 'pyproject.toml', 'uv.lock', 'requirements.lock',
                 'requirements.txt', 'LICENSE', '.gitignore'):
        if (repo / name).is_file(): files.add(name)
    files.add(plan_path)
    for name in ('declaration_path', 'secondary_declaration_path', 'generation_plan_path'):
        files.add(plan[name], plan[name.replace('_path', '_sha256')])
    closure = public.read(files.add(result_name + '/' + plan['generation_closure_path'],
                                    plan['generation_closure_file_sha256']))
    for row in closure['files']: files.add(result_name + '/' + row['path'], row['sha256'])
    scored = public.read(files.add(result_name + '/' + plan['manifest_path'], summary['scored_manifest_sha256']))
    for row in scored['files']: files.add(result_name + '/' + row['path'], row['sha256'])
    repair = public.add_repair(files, result_name, public.ORIGINAL_EVALUATION)
    now = datetime.now(timezone.utc).isoformat()
    metadata = {'schema_version': 1, 'experiment_id': d['experiment_id'], 'cohort': 'secondary',
        'export_utc': now, 'training_seed': 20260919, 'checkpoint_step': 1024,
        'task_count': summary['task_clusters'], 'secondary_answers': summary['secondary_answers'],
        'secondary_missing_answers': summary['secondary_missing_answers'],
        'native_reference_draws_reused': summary['native_reference_draws_reused'],
        'primary_answers': primary_summary['answers'], 'primary_missing_answers': primary_summary['missing_draws'],
        'training_seeds_pooled_or_selected': False, 'original_primary_delivery_unchanged': True,
        'original_primary_archive_sha256': original_receipt['archive_sha256'],
        'primary_declaration_sha256': summary['declaration_sha256'],
        'secondary_declaration_sha256': summary['secondary_declaration_sha256'],
        'scored_manifest_sha256': summary['scored_manifest_sha256'],
        'secondary_report_files_regenerated_and_identical': True,
        'candidate_or_model_execution_performed': False,
        'private_tests_or_weights_included': False, 'scorer_repair': repair,
        'git': public.git_identity(repo, d), 'packager_sha256': public.sha(Path(__file__)),
        'saved_resource_accounting': public.saved_resources(files, evidence_name),
        'resource_status_note': 'Saved receipts carry observation times; export time is not a live provider check.',
        'excluded_paths': sorted(set(files.excluded))}
    reproduction = f'''# Reproduce the separate confirmation reports

Install the included analysis dependencies from the project lockfile. Run from
the extracted root; these commands read saved public records, without weights,
private tests, model generation or candidate execution.

```sh
python scripts/coding_confirmation_report.py --plan {primary_plan}
python scripts/coding_confirmation_secondary_report.py --plan {plan_path} --primary-plan {primary_plan} --output regenerated_secondary_report
```

The secondary output directory must be new. The same 200 tasks and original
A/B/D/P measurements are reused; they are not additional tasks or replication
draws. Training seeds remain separate. Secondary intervals are explicitly
exploratory and unadjusted. The original primary report is preserved verbatim.
The included file manifest inventories every other ZIP member. Its own hash and
the ZIP hash are in the external delivery receipt. Heavy checkpoint retrieval
paths and hashes are in the frozen manifests and verified backup receipts.
'''
    generated = {**preserved, REPORT_NAME: (root / 'secondary/report' / REPORT_NAME).read_bytes(),
        'SECONDARY_REVIEW_DELIVERY.json': public.encoded(metadata),
        'PRIMARY_DELIVERY_RECEIPT.json': (original / 'DELIVERY_RECEIPT.json').read_bytes(),
        'REPRODUCE.md': reproduction.encode()}
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.secondary-review-', dir=out.parent))
    try:
        inventory = []; total = 0; archive = stage / ZIP_NAME
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            for name in sorted(set(files.files) | set(generated)):
                if name in files.files and name in generated: raise ValueError('Generated review path collision')
                content = generated[name] if name in generated else files.bytes(name)
                total += len(content)
                if total > public.MAX_TOTAL_BYTES: raise ValueError('Compact review exceeds total size limit')
                public.scan(name, content); z.writestr(name, content)
                inventory.append({'path': name, 'bytes': len(content), 'sha256': public.hash_bytes(content)})
            manifest = public.encoded({'schema_version': 1, 'experiment_id': d['experiment_id'],
                'cohort': 'secondary', 'export_utc': now, 'files': inventory,
                'uncompressed_bytes_excluding_inventory': total,
                'inventory_self_hash': 'Recorded in external DELIVERY_RECEIPT.json.'})
            z.writestr(public.MANIFEST, manifest)
        verified = public.verify_archive(archive, public.hash_bytes(manifest))
        for name, content in generated.items(): (stage / name).write_bytes(content)
        (stage / public.MANIFEST).write_bytes(manifest)
        receipt = {**metadata, **verified, 'all_outputs_verified': True,
            'zip_path': str(out / ZIP_NAME), 'results_path': str(out / REPORT_NAME),
            'archive_member_count': len(inventory) + 1}
        (stage / 'DELIVERY_RECEIPT.json').write_bytes(public.encoded(receipt))
        out.mkdir()
        for path in stage.iterdir():
            if path.name != 'DELIVERY_RECEIPT.json': os.replace(path, out / path.name)
        os.replace(stage / 'DELIVERY_RECEIPT.json', out / 'DELIVERY_RECEIPT.json')
        stage.rmdir(); return receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--primary-plan', required=True)
    parser.add_argument('--primary-delivery', required=True); parser.add_argument('--delivery', required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.plan, args.primary_plan, args.primary_delivery, args.delivery), indent=2))
