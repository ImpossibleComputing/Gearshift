import json
import sys

import pytest

from gearshift.coding_control import digest, sha, write
from gearshift.coding_reuse import REQUIRED, reuse_completed
from gearshift.coding_snapshot import stream
from scripts import coding_import_snapshot as importer


def completed(folder, identity, task_id, text):
    for name in REQUIRED:
        write(folder / name, {
            'task_id': task_id, 'score': {'passed': True},
            'verbatim_text': text, 'token_ids': [1, 2, 3],
        })
    write(folder / 'complete.json', {
        'task_id': task_id, 'identity_sha256': digest(identity),
        'pass': {'A': True, 'B': True, 'D': True},
        'files': {p.name: sha(p) for p in folder.iterdir()},
    })


def fixture(tmp_path):
    source = tmp_path / 'source'
    runtime = {'packages': {'torch': 'fixed-version'}}
    write(source / 'evidence/coding_pilot_v1/before_cap_v1/runtime_lock.json', runtime)
    original = source / 'results/coding_pilot_v1/preflight_v5'
    amended = source / 'results/coding_pilot_v1/preflight_cap_v1'
    original_id = {'run': 'original', 'runtime_sha256': digest(runtime)}
    amended_id = {'run': 'cap_amendment', 'runtime_sha256': digest(runtime)}
    for root, identity in [(original, original_id), (amended, amended_id)]:
        write(root / 'identity.json', identity)
        write(root / 'native_gate.json', {'passed': True})
        write(root / 'progress.json', {'completed_tasks': 2})
    for task in ['unaffected', 'affected']:
        completed(original / 'tasks' / task, original_id, task, 'original reasoning\n')
    reuse_completed(
        original / 'tasks/unaffected', amended / 'tasks/unaffected',
        original_id, amended_id, 'unaffected',
        original.relative_to(source) / 'tasks/unaffected',
    )
    completed(amended / 'tasks/affected', amended_id, 'affected', 'extended reasoning\n')
    archive = tmp_path / 'snapshot.tar.gz'
    with archive.open('wb') as output:
        stream(source, output)
    return source, archive


def run_import(monkeypatch, destination, archive):
    monkeypatch.setattr(importer, 'ROOT', destination)
    monkeypatch.setattr(sys, 'argv', ['coding_import_snapshot.py', str(archive)])
    importer.main()


def test_import_retains_original_and_amended_trajectories_and_lineage(tmp_path, monkeypatch):
    source, archive = fixture(tmp_path)
    destination = tmp_path / 'destination'
    run_import(monkeypatch, destination, archive)
    for namespace in ['preflight_v5', 'preflight_cap_v1']:
        relative = 'results/coding_pilot_v1/' + namespace
        for p in (source / relative / 'tasks').rglob('*.json'):
            assert (destination / p.relative_to(source)).read_bytes() == p.read_bytes()
        assert (destination / relative / 'progress_at_2_tasks.json').exists()
    receipt = json.loads(next((destination / 'evidence/coding_pilot_v1/imports').glob('*.json')).read_text())
    assert {r['namespace'] for r in receipt['imported']} == {'preflight_v5', 'preflight_cap_v1'}
    assert all(r['completed_tasks'] == 2 for r in receipt['imported'])
    before = {p.relative_to(destination): p.read_bytes() for p in destination.rglob('*') if p.is_file()}
    run_import(monkeypatch, destination, archive)
    assert before == {p.relative_to(destination): p.read_bytes() for p in destination.rglob('*') if p.is_file()}


def test_import_refuses_to_replace_a_saved_amended_trajectory(tmp_path, monkeypatch):
    _, archive = fixture(tmp_path)
    destination = tmp_path / 'destination'
    run_import(monkeypatch, destination, archive)
    trajectory = destination / 'results/coding_pilot_v1/preflight_cap_v1/tasks/affected/source_history.json'
    trajectory.write_text('prior local evidence')
    with pytest.raises(ValueError, match='Refusing to replace existing evidence'):
        run_import(monkeypatch, destination, archive)
    assert trajectory.read_text() == 'prior local evidence'
