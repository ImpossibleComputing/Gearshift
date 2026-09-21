#!/usr/bin/env python3
"""Verify and import a stopped history stage for review; never resume execution."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import digest, sha
from gearshift.coding_parallel import relative_path, safe_id
from gearshift.coding_snapshot import verify_archive
from gearshift.coding_training import validate_history
from coding_parallel_report import read, write


def copy_exact(source, destination):
    if source.is_symlink() or not source.is_file():
        raise ValueError('Unsafe source')
    if destination.exists():
        if destination.is_symlink() or sha(destination) != sha(source):
            raise ValueError('Refusing to overwrite historical evidence: ' + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.import.tmp')
    shutil.copyfile(source, temporary)
    if sha(temporary) != sha(source):
        raise ValueError('Copy changed')
    temporary.replace(destination)


def verify_task(folder, identity, task):
    receipt = read(folder / 'complete.json')
    if receipt['task_id'] != task['task_id'] or receipt['identity_sha256'] != digest(identity):
        raise ValueError('Task identity differs')
    for name, expected in receipt['files'].items():
        if Path(name).name != name or sha(folder / name) != expected:
            raise ValueError('Task output changed')
    history = read(folder / 'source_history.json')
    teacher = read(folder / 'teacher_answer.json')
    if history['prompt_ids'] != task['prompt_ids'] or receipt['split'] != task['split']:
        raise ValueError('Task membership or prompt changed')
    validate_history({'task_id': task['task_id'], 'source_history': history, 'teacher_answer': teacher})
    if len(history['reasoning_ids']) != receipt['source_reasoning_tokens']:
        raise ValueError('Reasoning count differs')
    return receipt, history, teacher


def collect(repo, run_id):
    safe_id(run_id)
    base = repo / 'evidence/coding_pilot_v1'
    control = base / 'control/parallel' / run_id
    plan = read(control / 'plan.json')
    done = read(control / 'complete.json')
    identity = digest(plan)
    if plan['stage'] != 'histories' or done.get('passed') is not False or done['stage_identity'] != identity:
        raise ValueError('Expected the exact completed failed history stage')
    watchdog = read(base / 'control/watchdog_status.json')
    if watchdog['active_gpu_count'] != 0 or time.time() - watchdog['epoch'] > 120:
        raise ValueError('Fresh zero-GPU cleanup confirmation required')
    resources = {r['id']: r for r in read(base / 'control/ledger.json')['resources']}
    expected = {}
    for split in ['training', 'validation']:
        for row in read(repo / f'data/coding_pilot_v1/visible/{split}.json'):
            if row['task_id'] in expected:
                raise ValueError('Duplicate fixed task')
            expected[row['task_id']] = {**row, 'split': split}
    assigned = [tid for worker in plan['workers'] for tid in worker['task_ids']]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(expected):
        raise ValueError('Stage does not cover the exact fixed cohort')
    rows = []; workers = []; imports = {}; feature_inventory = {}
    for worker in plan['workers']:
        wid = worker['worker_id']; safe_id(wid)
        backup = base / 'parallel_backups' / run_id / wid
        proof = read(control / wid / 'backup_verified.json')
        if not proof['verified'] or not proof['worker_confirmed_absent'] or proof['stage_identity'] != identity:
            raise ValueError('Missing final backup confirmation')
        if len(set(proof['copies'])) < 2:
            raise ValueError('Two distinct final copies required')
        for name in proof['copies']:
            if sha(repo / relative_path(name)) != proof['sha256']:
                raise ValueError('Final backup differs')
        for rid in [proof['pod_id'], proof['volume_id']]:
            if not resources[rid].get('absent_epoch'):
                raise ValueError('Worker resource still active')
        result_rel = f'results/coding_pilot_v1/{run_id}/{wid}'
        evidence_rel = f'evidence/coding_pilot_v1/parallel/{run_id}/{wid}'
        with tempfile.TemporaryDirectory(prefix='history-stop-import-') as tmp:
            stage = Path(tmp)
            manifest = verify_archive(backup / 'final.tar.gz', stage)
            scope = manifest['scope']
            if scope['stage_identity'] != identity or scope['worker_id'] != wid or scope['run_id'] != run_id:
                raise ValueError('Snapshot scope differs')
            root = stage / result_rel; evidence = stage / evidence_rel
            status = read(evidence / 'worker_status.json'); native = read(root / 'native_gate.json')
            ident = read(root / 'identity.json')
            if status['state'] not in ['complete', 'failed'] or status['worker_id'] != wid or status['stage_identity'] != identity:
                raise ValueError('Worker is not finalized')
            if ident['stage_identity'] != identity or ident['worker_id'] != wid:
                raise ValueError('Worker identity differs')
            if not native['passed'] or native['identity_sha256'] != digest(ident):
                raise ValueError('Native control identity differs')
            complete_ids = set(); expected_features = {}
            for tid in worker['task_ids']:
                folder = root / 'tasks' / tid.replace('/', '__')
                row = {'worker': wid, 'task_id': tid, 'split': expected[tid]['split'], 'status': 'not_started',
                       'source_reasoning_tokens': 0, 'answer_tokens': 0, 'source_capped': None,
                       'answer_capped': None, 'reused': False, 'transaction_sha256': None,
                       'source_history_sha256': None, 'partial_prefix_sha256': None}
                if (folder / 'complete.json').exists():
                    receipt, history, answer = verify_task(folder, ident, expected[tid])
                    complete_ids.add(tid)
                    for name, wanted in receipt.get('feature_files', {}).items():
                        if Path(name).name != name:
                            raise ValueError('Unsafe task feature path')
                        expected_features[str(folder.relative_to(stage) / name)] = wanted
                    if (task_feature := receipt.get('feature_files', {})) != ({'paired_features.pt': task_feature.get('paired_features.pt')} if expected[tid]['split'] == 'training' else {}):
                        raise ValueError('Required training feature missing or validation feature unexpected')
                    row.update(status='complete', source_reasoning_tokens=len(history['reasoning_ids']),
                        answer_tokens=len(answer['answer_ids']), source_capped=history['reasoning_capped'],
                        answer_capped=answer['answer_capped'], reused=receipt.get('reused_without_generation', False),
                        transaction_sha256=sha(folder / 'complete.json'), source_history_sha256=sha(folder / 'source_history.json'))
                elif (folder / 'inflight_source_reasoning.json').exists():
                    partial = read(folder / 'inflight_source_reasoning.json')
                    if partial['task_id'] != tid or partial['identity_sha256'] != digest(ident):
                        raise ValueError('Partial prefix identity differs')
                    row.update(status='partial', source_reasoning_tokens=len(partial['tokens']),
                               partial_prefix_sha256=sha(folder / 'inflight_source_reasoning.json'))
                rows.append(row)
            if status['state'] == 'complete':
                final = read(root / 'complete.json')
                if complete_ids != set(worker['task_ids']) or set(final['tasks']) != complete_ids or final['task_count'] != len(complete_ids) or final['identity_sha256'] != digest(ident):
                    raise ValueError('Successful worker is incomplete')
                for tid, wanted in final['tasks'].items():
                    if sha(root / 'tasks' / tid.replace('/', '__') / 'complete.json') != wanted:
                        raise ValueError('Final task receipt changed')
            else:
                failure = read(root / 'failure.json')
                if failure['error'] != 'Actual worker memory/headroom gate failed':
                    raise ValueError('Unreviewed failure type')
            feature_manifest = read(root / 'feature_manifest.json')['files']
            if feature_manifest != expected_features:
                raise ValueError('Feature inventory does not cover exact complete tasks')
            for name, wanted in feature_manifest.items():
                relative_path(name)
                if not name.startswith(result_rel + '/') or Path(name).suffix != '.pt':
                    raise ValueError('Unexpected feature file')
                src = backup / 'mapper_checkpoints' / name
                if proof['mapper_checkpoints'][name]['sha256'] != wanted or sha(src) != wanted:
                    raise ValueError('Feature backup changed')
                feature_inventory[name] = {'sha256': wanted, 'bytes': src.stat().st_size,
                                           'backup': str(src.relative_to(repo)), 'excluded_from_review_zip': True}
                copy_exact(src, repo / name)
            for name, record in manifest['files'].items():
                if any(name.startswith(prefix + '/') for prefix in [result_rel, evidence_rel]):
                    copy_exact(stage / name, repo / relative_path(name)); imports[name] = record
            workers.append({'worker': wid, 'state': status['state'], 'complete_histories': len(complete_ids),
                'task_count': len(worker['task_ids']), 'failure_task': status.get('task_id') if status['state'] == 'failed' else None,
                'failure_epoch': status['heartbeat_epoch'] if status['state'] == 'failed' else None,
                'archive_sha256': proof['sha256'], 'pod_id': proof['pod_id'], 'volume_id': proof['volume_id'],
                'resources_confirmed_absent': True})
    counts = {state: sum(r['status'] == state for r in rows) for state in ['complete', 'partial', 'not_started']}
    return {'run_id': run_id, 'stage_identity': identity, 'stage_completion_sha256': sha(control / 'complete.json'),
        'counts': counts, 'split_counts': {split: {state: sum(r['split'] == split and r['status'] == state for r in rows)
            for state in counts} for split in ['training', 'validation']}, 'workers': workers,
        'reused_complete_histories': sum(r['reused'] for r in rows), 'task_rows': rows,
        'imported_compact_files': imports, 'excluded_feature_inventory': feature_inventory,
        'cost_snapshot': watchdog, 'study_complete': False, 'training_started': False,
        'confirmation_tasks_run': 0, 'second_seed_tasks_run': 0,
        'memory_failure_readings': 'Not recorded by the worker guard; exact condition and cause unresolved.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); repo = Path(__file__).resolve().parents[1]
    output = repo / relative_path(str(args.output))
    if output.exists():
        raise ValueError('Use a fresh report directory; never overwrite a review')
    summary = collect(repo, args.run)
    output.mkdir(parents=True)
    write(output / 'history_summary.json', summary)
    write(output / 'feature_inventory.json', summary['excluded_feature_inventory'])
    with (output / 'history_tasks.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary['task_rows'][0])); writer.writeheader(); writer.writerows(summary['task_rows'])
    print(json.dumps({'output': str(output), 'counts': summary['counts'], 'split_counts': summary['split_counts'],
                      'features': len(summary['excluded_feature_inventory']), 'workers': summary['workers']}, indent=2))

if __name__ == '__main__':
    main()
