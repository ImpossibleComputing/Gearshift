import io
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import coding_coverage_v2_dispatch as dispatch
from gearshift.coding_coverage_v2_lease import atomic_json


def archive(tmp_path, rows):
    output = tmp_path / 'archive.tar.gz'
    with tarfile.open(output, 'w:gz') as target:
        for name, content, kind in rows:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(content)
                target.addfile(member, io.BytesIO(content))
            else:
                member.linkname = '/root/.gearshift-runpod-key'
                target.addfile(member)
    return output


@pytest.mark.parametrize('name', ['../escape', '/root/key', 'gearshift/../../escape', '.runpod/config.toml',
                                 '.env', dispatch.RESULT_ROOT + '/lease_guard/status.json',
                                 dispatch.RESULT_ROOT + '/gpu_release_verified.json'])
def test_archive_rejects_traversal_secrets_and_live_control_state(tmp_path, name):
    path = archive(tmp_path, [(name, b'payload', tarfile.REGTYPE)])
    with pytest.raises(ValueError):
        dispatch.validate_archive(path, private=False, lease_sha256='a'*64, guard_sha256='b'*64)


def test_archive_rejects_symlink_and_normalized_path_overwrite(tmp_path):
    path = archive(tmp_path, [('gearshift/link', b'', tarfile.SYMTYPE)])
    with pytest.raises(ValueError, match='ordinary'):
        dispatch.validate_archive(path, private=False, lease_sha256='a'*64, guard_sha256='b'*64)
    path = archive(tmp_path, [('./' + dispatch.EVIDENCE + '/allocation_lease.json', b'changed', tarfile.REGTYPE)])
    with pytest.raises(ValueError, match='immutable'):
        dispatch.validate_archive(path, private=False, lease_sha256='a'*64, guard_sha256='b'*64)


def test_private_tests_have_a_separate_archive_boundary(tmp_path):
    path = archive(tmp_path, [('data/coding_pilot_v1/private/coverage_generalization.json', b'{}', tarfile.REGTYPE)])
    dispatch.validate_archive(path, private=True, lease_sha256='a'*64, guard_sha256='b'*64)
    with pytest.raises(ValueError, match='boundary'):
        dispatch.validate_archive(path, private=False, lease_sha256='a'*64, guard_sha256='b'*64)


def test_recorded_or_ambiguous_allocation_cannot_silently_provision_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, 'ROOT', tmp_path)
    evidence = tmp_path / dispatch.EVIDENCE
    atomic_json(tmp_path / 'configs/coding_pilot_v1/coverage_generalization_v2/declaration.json', {
        'budget_start': {'upper_usd': 335.93, 'gpu_hours': 58.94, 'epoch': 1000}})
    monkeypatch.setattr(dispatch.time, 'time', lambda: 1000)
    calls = []
    def fail(*args):
        calls.append(args)
        raise RuntimeError('unsupported center; provider allocation state not confirmed')
    monkeypatch.setattr(dispatch, 'cli', fail)
    with pytest.raises(RuntimeError, match='unsupported'):
        dispatch.allocate(SimpleNamespace(region='invalid-region'))
    failures = list((evidence / 'allocation_attempts').glob('*/failure.json'))
    assert len(failures) == 1
    assert dispatch.read(failures[0])['stage'] == 'volume_creation'
    with pytest.raises(ValueError, match='reconciliation'):
        dispatch.allocate(SimpleNamespace(region='new-region'))
    assert len(calls) == 1


@pytest.mark.parametrize('price', [float('nan'), float('inf'), None, 19])
def test_bad_live_quote_blocks_before_private_key_upload(tmp_path, monkeypatch, price):
    monkeypatch.setattr(dispatch, 'ROOT', tmp_path)
    lease = {'experiment_id': dispatch.EXPERIMENT, 'pod_id': 'owned-pod',
             'allocation_epoch': 1000, 'deadline_epoch': 73000,
             'upper_hourly_usd': 22.20, 'gpu_count': 4, 'baseline_usd': 335.93,
             'baseline_gpu_hours': 58.94, 'total_cap_usd': 1000,
             'total_cap_gpu_hours': 500, 'cleanup_reserve_usd': 40,
             'allowed_result_root': dispatch.REMOTE_ROOT+'/'+dispatch.RESULT_ROOT,
             'network_volume_id': 'owned-volume'}
    atomic_json(tmp_path / dispatch.EVIDENCE / 'allocation_lease.json', lease)
    monkeypatch.setattr(dispatch, 'cli', lambda *args: {
        'id': 'owned-pod', 'gpuCount': 4, 'networkVolumeId': 'owned-volume', 'costPerHr': price})
    monkeypatch.setattr(dispatch, 'ssh', lambda *args, **kwargs: pytest.fail('key/guard must not reach an unverified allocation'))
    with pytest.raises(ValueError, match='quote'):
        dispatch.arm(SimpleNamespace())


def test_remote_identity_must_match_pod_before_credentials_are_sent(monkeypatch):
    monkeypatch.setattr(dispatch, 'ssh', lambda *args, **kwargs: b'{"pod_id":"different-pod"}')
    with pytest.raises(ValueError, match='SSH endpoint'):
        dispatch.verify_remote_identity(SimpleNamespace(), {'pod_id':'owned-pod'})


def test_valid_four_h200_quote_fits_conservative_rate():
    assert 18.36 * 1.2 + .15 < 22.20
