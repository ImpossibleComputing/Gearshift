import importlib.util
from pathlib import Path
import shutil
import sys
import tarfile
import time
import pytest
from gearshift.coding_control import digest, sha, write
from gearshift.coding_snapshot import capture


def module():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('history_stop_report', scripts / 'coding_history_stop_report.py')
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def fixture(tmp_path):
    m = module(); run = 'test'; wid = 'history01'; base = tmp_path / 'evidence/coding_pilot_v1'
    control = base / 'control/parallel' / run; evidence = base / 'parallel' / run / wid
    result = tmp_path / 'results/coding_pilot_v1' / run / wid
    plan = {'stage': 'histories', 'workers': [{'worker_id': wid, 'task_ids': ['p/1', 'p/2']}]}
    ident = {'stage_identity': digest(plan), 'worker_id': wid}
    write(control / 'plan.json', plan); write(control / 'complete.json', {'passed': False, 'stage_identity': digest(plan)})
    for split, tid in [('training', 'p/1'), ('validation', 'p/2')]:
        write(tmp_path / f'data/coding_pilot_v1/visible/{split}.json', [{'task_id': tid, 'prompt_ids': [1]}])
    write(result / 'identity.json', ident); write(result / 'native_gate.json', {'passed': True, 'identity_sha256': digest(ident)})
    write(result / 'failure.json', {'error': 'Actual worker memory/headroom gate failed'})
    write(evidence / 'worker_status.json', {'state': 'failed', 'worker_id': wid, 'stage_identity': digest(plan), 'task_id': 'p/2', 'heartbeat_epoch': time.time()})
    folder = result / 'tasks/p__1'; feature = folder / 'paired_features.pt'; folder.mkdir(parents=True); feature.write_bytes(b'sampled features')
    history = {'task_id': 'p/1', 'prompt_ids': [1], 'reasoning_ids': [2, 151668], 'prefix_ids': [1, 2],
               'prefix_cache_length': 2, 'bridge_ids': [151668], 'natural_boundary': True, 'reasoning_capped': False}
    answer = {'task_id': 'p/1', 'answer_ids': [4, 151645], 'answer_capped': False}
    write(folder / 'source_history.json', history); write(folder / 'teacher_answer.json', answer)
    write(folder / 'complete.json', {'task_id': 'p/1', 'identity_sha256': digest(ident), 'split': 'training',
        'source_reasoning_tokens': 2, 'files': {name: sha(folder / name) for name in ['source_history.json', 'teacher_answer.json']},
        'feature_files': {'paired_features.pt': sha(feature)}})
    write(result / 'feature_manifest.json', {'files': {str(feature.relative_to(tmp_path)): sha(feature)}})
    write(result / 'tasks/p__2/inflight_source_reasoning.json', {'task_id': 'p/2', 'identity_sha256': digest(ident), 'tokens': [5, 6, 7]})
    write(evidence / 'worker_spec.json', {'run_id': run, 'worker_id': wid, 'stage': 'histories',
        'result_root': str(result.relative_to(tmp_path)), 'worker_root': str(evidence.relative_to(tmp_path)), 'stage_identity': digest(plan)})
    snapshot = tmp_path / 'snapshot'; capture(tmp_path, snapshot, worker_spec=evidence / 'worker_spec.json')
    backup = base / 'parallel_backups' / run / wid; backup.mkdir(parents=True)
    archive = backup / 'final.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in snapshot.rglob('*'):
            if path.is_file(): tar.add(path, arcname=path.relative_to(snapshot))
    shutil.copyfile(archive, backup / 'latest.tar.gz')
    copy = backup / 'mapper_checkpoints' / feature.relative_to(tmp_path); copy.parent.mkdir(parents=True); shutil.copyfile(feature, copy)
    write(control / wid / 'backup_verified.json', {'verified': True, 'worker_confirmed_absent': True,
        'stage_identity': digest(plan), 'copies': [str((backup / name).relative_to(tmp_path)) for name in ['final.tar.gz', 'latest.tar.gz']],
        'sha256': sha(archive), 'pod_id': 'pod', 'volume_id': 'volume',
        'mapper_checkpoints': {str(feature.relative_to(tmp_path)): {'sha256': sha(feature)}}})
    write(base / 'control/watchdog_status.json', {'active_gpu_count': 0, 'epoch': time.time()})
    write(base / 'control/ledger.json', {'resources': [{'id': name, 'absent_epoch': time.time()} for name in ['pod', 'volume']]})
    return m, result, copy


def test_partial_report_preserves_exact_records_without_claiming_completion(tmp_path):
    m, root, _ = fixture(tmp_path); before = (root / 'tasks/p__1/source_history.json').read_bytes()
    result = m.collect(tmp_path, 'test')
    assert result['counts'] == {'complete': 1, 'partial': 1, 'not_started': 0}
    assert result['study_complete'] is False and result['training_started'] is False
    assert result['task_rows'][1]['source_reasoning_tokens'] == 3
    assert (root / 'tasks/p__1/source_history.json').read_bytes() == before


def test_feature_corruption_is_rejected_before_reporting_success(tmp_path):
    m, _, copy = fixture(tmp_path); copy.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Feature backup changed'): m.collect(tmp_path, 'test')


def test_import_refuses_to_overwrite_changed_existing_evidence(tmp_path):
    m, root, _ = fixture(tmp_path); path = root / 'tasks/p__1/source_history.json'; path.write_text('changed')
    with pytest.raises(ValueError, match='overwrite historical'): m.collect(tmp_path, 'test')
    assert path.read_text() == 'changed'


def test_active_compute_blocks_final_export_snapshot(tmp_path):
    m, _, _ = fixture(tmp_path)
    write(tmp_path / 'evidence/coding_pilot_v1/control/watchdog_status.json', {'active_gpu_count': 1, 'epoch': time.time()})
    with pytest.raises(ValueError, match='zero-GPU'): m.collect(tmp_path, 'test')
