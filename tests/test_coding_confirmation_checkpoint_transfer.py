"""Synthetic transport and endpoint provenance checks; no models or cloud calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import coding_confirmation_checkpoint_transfer as t

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('transfer_fixture', ROOT / 'tests/test_coding_confirmation_secondary_generate.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def endpoint(tmp_path, monkeypatch):
    _, addendum, _ = f.frozen_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(t, 'DECLARATION', 'declaration.json')
    monkeypatch.setattr(t, 'DECLARATION_SHA', t.sha(tmp_path / 'declaration.json'))
    monkeypatch.setattr(t, 'RESULT', 'results/replication/arms')
    monkeypatch.setattr(t, 'TRAINING_COMMIT', 'a' * 40)
    for arm, cp in addendum['checkpoints'].items():
        folder = tmp_path / t.RESULT / arm / 'checkpoints/step_1024/export'
        f.write(folder / 'training_complete.json', {'arm': arm, 'target_updates': 1024,
            'completed_updates': 1024, 'full_target_completed': True, 'scored_positions': 32768,
            'last_durable_checkpoint': 1024, 'checkpoint_identity_sha256': f.digest(cp['checkpoint_identity']),
            'hidden_tests_loaded': False, 'free_running_answers_generated': 0})
        f.write(folder / 'training_steps.json', [{'step': n, 'arm': arm} for n in range(1, 1025)])
    return addendum


def test_complete_pair_has_exact_four_bound_weight_files(tmp_path, monkeypatch):
    endpoint(tmp_path, monkeypatch)
    rows = t.expected_weights(tmp_path)
    assert len(rows) == 4
    for row in rows: t.check_file(tmp_path / row['path'], row)


@pytest.mark.parametrize('mutation', ['step_gap', 'wrong_arm', 'incomplete', 'score_exposure', 'training_commit', 'declaration_pin'])
def test_rejects_changed_provenance_or_completion(tmp_path, monkeypatch, mutation):
    endpoint(tmp_path, monkeypatch)
    root = tmp_path / t.RESULT / 'FIXED/checkpoints/step_1024/export'
    if mutation in ('step_gap', 'wrong_arm'):
        path = root / 'training_steps.json'; value = t.read(path)
        if mutation == 'step_gap': value.pop(800)
        else: value[2]['arm'] = 'ROTATING'
    elif mutation in ('incomplete', 'score_exposure'):
        path = root / 'training_complete.json'; value = t.read(path)
        value['completed_updates' if mutation == 'incomplete' else 'hidden_tests_loaded'] = 1023 if mutation == 'incomplete' else True
    elif mutation == 'training_commit':
        monkeypatch.setattr(t, 'TRAINING_COMMIT', 'd' * 40)
    else: monkeypatch.setattr(t, 'DECLARATION_SHA', '0' * 64)
    if mutation not in ('training_commit', 'declaration_pin'): f.write(path, value)
    with pytest.raises(ValueError): t.expected_weights(tmp_path)


def transport(tmp_path, monkeypatch, content=b'exact final endpoint bytes'):
    monkeypatch.setattr(t, 'ROOT', tmp_path)
    path = tmp_path / 'results/checkpoint/mapper.pt'; path.parent.mkdir(parents=True)
    import hashlib
    row = {'path': str(path.relative_to(tmp_path)), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    monkeypatch.setattr(t.base, 'ssh_args', lambda _: [])
    return path, row, content


def test_partial_download_resumes_exact_bytes_then_idempotently_verifies(tmp_path, monkeypatch):
    path, row, content = transport(tmp_path, monkeypatch)
    partial = path.with_name(path.name + '.partial-' + row['sha256']); partial.write_bytes(content[:7])
    def remote(args, stdout, **kw):
        assert args[0].startswith('tail -c +8 ')
        stdout.write(content[7:]); return SimpleNamespace(returncode=0)
    monkeypatch.setattr(t.subprocess, 'run', remote)
    t.fetch_weight({}, row)
    assert path.read_bytes() == content and not partial.exists()
    monkeypatch.setattr(t.subprocess, 'run', lambda *a, **k: pytest.fail('Verified file must not be downloaded again'))
    t.fetch_weight({}, row)


@pytest.mark.parametrize('fault', ['changed_existing', 'bad_stream', 'interrupted', 'linked_parent'])
def test_transport_preserves_bad_or_interrupted_evidence_without_publishing(tmp_path, monkeypatch, fault):
    path, row, content = transport(tmp_path, monkeypatch)
    if fault == 'changed_existing': path.write_bytes(b'keep this mismatch')
    if fault == 'linked_parent':
        path.parent.rmdir(); elsewhere = tmp_path / 'elsewhere'; elsewhere.mkdir(); path.parent.symlink_to(elsewhere)
    def remote(args, stdout, **kw):
        stdout.write(b'corrupt' if fault == 'bad_stream' else content[:3])
        return SimpleNamespace(returncode=1 if fault == 'interrupted' else 0, stderr=b'interrupted')
    monkeypatch.setattr(t.subprocess, 'run', remote)
    with pytest.raises((ValueError, RuntimeError)): t.fetch_weight({}, row)
    if fault == 'changed_existing': assert path.read_bytes() == b'keep this mismatch'
    else: assert not path.exists()
    if fault == 'interrupted': assert path.with_name(path.name + '.partial-' + row['sha256']).read_bytes() == content[:3]


@pytest.mark.parametrize('existing_mismatch', [False, True])
def test_regional_upload_verifies_actual_persisted_bytes_and_never_overwrites(tmp_path, monkeypatch, existing_mismatch):
    endpoint(tmp_path, monkeypatch)
    rows = t.expected_weights(tmp_path)
    for arm in ['FIXED', 'ROTATING']:
        f.write(tmp_path / t.RESULT / arm / 'checkpoints/step_1024/export/checkpoint_manifest.json', {'fixture': True})
    area = tmp_path / 'evidence'
    f.write(area / 'verified_local.json', {'files': rows, 'actual_mapper_and_full_bytes_verified': True,
        'metadata': {p: {'bytes': (tmp_path/p).stat().st_size, 'sha256': t.sha(tmp_path/p)} for p in t.metadata_names()}})
    f.write(tmp_path / t.public.CONFIG / 'regional_partition.json', {'shard_storage': {'origin': {'network_volume_id': 'fixture'}}})
    monkeypatch.setattr(t, 'ROOT', tmp_path); monkeypatch.setattr(t, 'AREA', area)
    monkeypatch.setattr(t, 'expected_weights', lambda: rows)
    monkeypatch.setattr(t.public, 'REMOTE', str(tmp_path / 'remote'))
    monkeypatch.setattr(t.base, 'api', lambda _: {'status': 'RUNNING', 'mounts': {'network': [{'volumeId': 'fixture', 'path': '/workspace'}]}})
    monkeypatch.setattr(t.base, 'ssh_args', lambda _: [])
    actual_run = t.subprocess.run
    monkeypatch.setattr(t.subprocess, 'run', lambda args, **kw: actual_run(['/bin/sh', '-c', args[0]], **kw))
    first = tmp_path / 'remote' / rows[0]['path']
    if existing_mismatch:
        first.parent.mkdir(parents=True); first.write_bytes(b'preserve existing evidence')
        with pytest.raises(RuntimeError, match='overwrite forbidden'): t.upload('test', 'origin')
        assert first.read_bytes() == b'preserve existing evidence'
        assert not list(area.glob('verified_origin_*.json'))
    else:
        result = t.upload('test', 'origin')
        assert result['files_verified'] == 12
        for row in t.read(result['saved'])['files']:
            t.check_file(row['path'], row)
            assert row['persisted_bytes_verified'] is True


def test_final_metadata_exports_preserve_earlier_local_inventory(tmp_path, monkeypatch):
    import base64
    endpoint(tmp_path, monkeypatch); rows=t.expected_weights(tmp_path)
    for arm in ['FIXED','ROTATING']:
        f.write(tmp_path/t.RESULT/arm/'checkpoints/step_1024/export/checkpoint_manifest.json',{'step':1024})
        f.write(tmp_path/t.RESULT/arm/'checkpoint_manifest.json',{'step':768,'prior_observation':True})
        f.write(tmp_path/t.RESULT/arm/'training_steps.json',[{'step':768,'prior_observation':True}])
    old={p:p.read_bytes() for p in (tmp_path/t.RESULT).glob('*/*.json')}
    files={p:base64.b64encode((tmp_path/t.local_metadata_path(p)).read_bytes()).decode() for p in t.source_metadata_names()}
    monkeypatch.setattr(t,'ROOT',tmp_path);monkeypatch.setattr(t,'AREA',tmp_path/'receipt')
    monkeypatch.setattr(t,'expected_weights',lambda:rows)
    monkeypatch.setattr(t,'endpoint_metadata',lambda _: {'files':files,'missing':[]})
    monkeypatch.setattr(t.base,'api',lambda _: {'mounts':{'network':[{'volumeId':'o0ndo35b3f','path':'/workspace'}]}})
    result=t.fetch('rekbqruqox2bk9')
    assert result['ready'] and result['files_verified']==4
    assert all(p.read_bytes()==value for p,value in old.items())
    receipt=t.read(result['receipt'])
    assert set(receipt['metadata'])==t.metadata_names()
    assert set(receipt['metadata_source_paths'].values())==t.source_metadata_names()
