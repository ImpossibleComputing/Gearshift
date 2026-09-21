"""CPU-only deployment, checkpoint restart and durable release contracts."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from gearshift.coding_control import digest, write
from gearshift.coding_confirmation_lease import sha256_file, verify_release_receipt
from scripts import coding_confirmation_replication_supervisor as supervisor
from scripts import coding_confirmation_replication_stage as stage
from scripts import coding_confirmation_replication as replication


def identity(arm='FIXED'):
    return {'experiment_id': replication.EXPERIMENT + '/replication', 'arm': arm,
            'code_commit': 'a' * 40, 'seed': 20260919}


def checkpoint(root, arm='FIXED', step=128):
    folder = Path(root) / 'checkpoints' / f'step_{step:04d}'; folder.mkdir(parents=True)
    for name in ('mapper.pt', 'full.pt'): (folder / name).write_bytes((name + arm + str(step)).encode())
    components = ['mapper', 'optimizer', 'scheduler', 'update', 'schedule_position',
                  'python_rng', 'numpy_rng', 'torch_cpu_rng', 'torch_cuda_rng',
                  'scaler_explicitly_none_for_bf16', 'checkpoint_identity', 'training_log']
    m = {'checkpoint_identity': identity(arm), 'checkpoint_identity_sha256': digest(identity(arm)),
         'arm': arm, 'step': step, 'schedule_position': step, 'complete_resumable': True,
         'verified_roundtrip': True, 'state_components': components,
         'files': {n: {'bytes': (folder / n).stat().st_size, 'sha256': sha256_file(folder / n)}
                   for n in ('full.pt', 'mapper.pt')}}
    write(folder / 'manifest.json', m)
    return folder


def test_stage_inventory_verifies_all_compact_inputs_and_requires_original_initializer(tmp_path):
    manifest = stage.make_manifest()
    assert manifest['compact_file_count'] > 391
    assert not any('/private/' in p or p.endswith('.pt') for p in manifest['files'])
    assert manifest['external_files'][replication.read(replication.ROOT / replication.DECLARATION)['selected_checkpoint']]['sha256'] == replication.START_SHA
    assert stage.verify_manifest(replication.ROOT, manifest, False)['verified']
    with pytest.raises(ValueError, match='missing'):
        stage.verify_manifest(tmp_path, manifest)


def test_environment_preserves_verified_pod_and_discards_credentials(monkeypatch):
    monkeypatch.setenv('RUNPOD_API_KEY', 'secret'); monkeypatch.setenv('HTTPS_PROXY', 'https://user:secret@example')
    monkeypatch.delenv('RUNPOD_POD_ID', raising=False)
    env = supervisor.child_environment({'pod_id': 'verifiedpod'}, 1)
    assert env['RUNPOD_POD_ID'] == 'verifiedpod' and env['CUDA_VISIBLE_DEVICES'] == '1'
    assert env['HF_HOME'] == '/workspace/hf' and env['HF_HUB_OFFLINE'] == '1'
    assert 'RUNPOD_API_KEY' not in env and 'HTTPS_PROXY' not in env


def test_resume_latest_complete_only_and_preserve_unfinished_preflight(tmp_path):
    preflight = tmp_path / 'resume_preflight'; preflight.mkdir()
    (preflight / 'partial.pt').write_bytes(b'original failure bytes')
    assert supervisor.prepare_restart(tmp_path, identity()) is None
    saved = list((tmp_path / 'attempts').rglob('partial.pt'))
    assert len(saved) == 1 and saved[0].read_bytes() == b'original failure bytes'
    checkpoint(tmp_path, step=0); latest = checkpoint(tmp_path, step=128)
    assert supervisor.prepare_restart(tmp_path, identity()) == latest
    (latest / 'full.pt').write_bytes(b'broken')
    with pytest.raises(ValueError, match='bytes'): supervisor.prepare_restart(tmp_path, identity())


def test_checkpoint_rejects_cross_arm_identity_and_missing_rng(tmp_path):
    folder = checkpoint(tmp_path)
    with pytest.raises(ValueError, match='identity'): supervisor.checkpoint_files(folder, identity('ROTATING'))
    m = supervisor.read(folder / 'manifest.json'); m['state_components'].remove('torch_cuda_rng')
    write(folder / 'manifest.json', m)
    with pytest.raises(ValueError, match='resumable'): supervisor.checkpoint_files(folder, identity())


def test_complete_release_copies_full_states_and_archives_and_guard_reverifies(tmp_path):
    root = tmp_path / 'results'; root.mkdir()
    lease = {'experiment_id': replication.EXPERIMENT, 'pod_id': 'pod1',
             'allowed_result_root': str(root), 'control_relative': 'allocations/pod1'}
    folder = root / 'allocations/pod1/replication_supervisor'; folder.mkdir(parents=True)
    for arm in replication.ARMS:
        arm_root = root / 'replication/arms' / arm
        for step in replication.CHECKPOINTS: checkpoint(arm_root, arm, step)
        write(arm_root / 'training_complete.json', {'full_target_completed': True, 'completed_updates': 1024})
    # Other confirmation writers must never enter this pod's release snapshot.
    (root / 'primary_mutable.json').write_text('not included')
    receipt = supervisor.durable_release(root, lease, {a: identity(a) for a in replication.ARMS}, 'complete', folder)
    assert receipt['gpu_release_verified'] and verify_release_receipt(lease)
    manifest = supervisor.read(root / receipt['manifest_path'])
    assert all(x['path'] != 'primary_mutable.json' for x in manifest['files'])
    copies = [x for x in manifest['files'] if '/full_states/' in x['path']]
    assert len(copies) == 24 and len([x for x in manifest['files'] if x['path'].endswith('.tar.gz')]) == 2
    (root / copies[0]['path']).write_bytes(b'corrupt backup')
    with pytest.raises(ValueError, match='changed'): verify_release_receipt(lease)


def test_incomplete_training_never_claims_complete_release(tmp_path):
    lease = {'experiment_id': replication.EXPERIMENT, 'pod_id': 'pod1',
             'allowed_result_root': str(tmp_path), 'control_relative': 'allocations/pod1'}
    folder = tmp_path / 'allocations/pod1/replication_supervisor'; folder.mkdir(parents=True)
    checkpoint(tmp_path / 'replication/arms/FIXED', step=0)
    with pytest.raises(ValueError, match='every declared'):
        supervisor.durable_release(tmp_path, lease, {a: identity(a) for a in replication.ARMS}, 'complete', folder)
    assert not (tmp_path / 'allocations/pod1/gpu_release_verified.json').exists()


def test_bounded_restarts_and_same_group_offline_launch(tmp_path, monkeypatch):
    s = supervisor.Supervisor.__new__(supervisor.Supervisor)
    s.repo = tmp_path; s.plan_path = tmp_path / 'dispatch.json'; s.root = tmp_path / 'results'
    s.folder = s.root / 'supervisor'; s.lease = {'pod_id': 'pod1'}
    s.identities = {a: identity(a) for a in replication.ARMS}
    s.children = {}; s.attempts = {'FIXED': 0, 'ROTATING': 0}; s.guard = lambda: None; s.event = lambda *a, **k: None
    spawned = []
    def popen(command, **kwargs):
        assert kwargs['start_new_session'] is False and kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0'
        assert kwargs['env']['RUNPOD_POD_ID'] == 'pod1'
        spawned.append(command); return SimpleNamespace(pid=123)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', popen)
    monkeypatch.setattr(supervisor.os, 'getpgid', lambda pid: os.getpgrp())
    for _ in range(3):
        s.start('FIXED')
        with pytest.raises(ValueError, match='live worker'): s.start('FIXED')
        s.children.clear()
    with pytest.raises(RuntimeError, match='restarts exhausted'): s.start('FIXED')
    assert len(spawned) == 3


def test_corrupt_checkpoint_is_preserved_as_failure_without_false_resume_claim(tmp_path):
    lease = {'experiment_id': replication.EXPERIMENT, 'pod_id': 'pod1',
             'allowed_result_root': str(tmp_path), 'control_relative': 'allocations/pod1'}
    folder = tmp_path / 'allocations/pod1/replication_supervisor'; folder.mkdir(parents=True)
    cp = checkpoint(tmp_path / 'replication/arms/FIXED', step=0)
    (cp / 'full.pt').write_bytes(b'corrupt failure evidence')
    result = supervisor.durable_release(tmp_path, lease, {a: identity(a) for a in replication.ARMS}, 'failed_preserved', folder)
    assert result['outcome'] == 'failed_preserved' and verify_release_receipt(lease)
    assert list(folder.glob('failure_checkpoint_*.json'))
