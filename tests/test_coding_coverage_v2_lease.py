import copy
import json
import os
from pathlib import Path
import signal
import urllib.error

import pytest

from gearshift.coding_coverage_v2_lease import (
    PROVIDER_BASE, atomic_json, control_root, delete_own_pod, linux_process_identity,
    load_lease, read_private_key, redacted_error, run_guard, sha256_file,
    signal_registered_supervisor, verify_lease, verify_release_receipt,
)


def lease(tmp_path):
    return {'experiment_id': 'coverage_v2_test', 'pod_id': 'owned-pod',
            'allocation_epoch': 1000, 'deadline_epoch': 73000,
            'upper_hourly_usd': 22.20, 'gpu_count': 4, 'baseline_usd': 335.93,
            'baseline_gpu_hours': 58.94, 'total_cap_usd': 1000,
            'total_cap_gpu_hours': 500, 'cleanup_reserve_usd': 40,
            'allowed_result_root': str(tmp_path / 'results' / 'experiment')}


def release_receipt(value):
    root = Path(value['allowed_result_root'])
    root.mkdir(parents=True, exist_ok=True)
    result = root / 'complete.json'
    atomic_json(result, {'complete': True})
    manifest = {k: value[k] for k in ('experiment_id', 'pod_id')}
    manifest['files'] = [{'path': 'complete.json', 'bytes': result.stat().st_size,
                          'sha256': sha256_file(result)}]
    atomic_json(root / 'release_manifest.json', manifest)
    receipt = {k: value[k] for k in ('experiment_id', 'pod_id')}
    receipt.update(gpu_release_verified=True, all_gpu_workers_stopped=True,
                   durable_backup_verified=True, manifest_path='release_manifest.json',
                   manifest_sha256=sha256_file(root / 'release_manifest.json'))
    atomic_json(root / 'gpu_release_verified.json', receipt)
    return root, receipt


def test_total_rate_and_gpu_reservations_are_not_double_counted(tmp_path):
    result = verify_lease(lease(tmp_path))
    assert result['reserved_total_usd'] == pytest.approx(819.93)
    assert result['reserved_total_gpu_hours'] == pytest.approx(138.94)


def test_two_pod_reservation_includes_other_lease_once(tmp_path):
    item = lease(tmp_path)
    item.update(gpu_count=2, upper_hourly_usd=11.10, other_reserved_usd=222,
                other_reserved_gpu_hours=40, control_relative='allocations/owned-pod')
    result = verify_lease(item)
    assert result['reserved_total_usd'] == pytest.approx(819.93)
    assert result['reserved_total_gpu_hours'] == pytest.approx(138.94)
    assert control_root(item) == Path(item['allowed_result_root']) / 'allocations/owned-pod'
    for field, value in [('other_reserved_usd', -1), ('other_reserved_gpu_hours', float('nan')),
                         ('other_reserved_usd', 500), ('other_reserved_gpu_hours', 402),
                         ('control_relative', 'allocations/other-pod'),
                         ('control_relative', '../outside')]:
        with pytest.raises(ValueError):
            verify_lease(dict(item, **{field: value}))


def test_one_pod_deadline_never_writes_shared_or_other_pod_stop(tmp_path):
    item = lease(tmp_path)
    item.update(control_relative='allocations/owned-pod', deadline_epoch=1010)
    root = Path(item['allowed_result_root'])
    atomic_json(root / 'allocations/other-pod/lease_guard/status.json', {'state': 'healthy'})
    timer = FakeClock(1000)
    run_guard(item, 'secret', clock=timer.now, monotonic=timer.now,
              sleep=timer.sleep, grace_seconds=0, delete=lambda *_: True,
              signal_supervisor=lambda _: False)
    assert (control_root(item) / 'lease_guard/STOP').exists()
    assert not (root / 'lease_guard/STOP').exists()
    assert not (root / 'allocations/other-pod/lease_guard/STOP').exists()
    assert json.loads((root / 'allocations/other-pod/lease_guard/status.json').read_text()) == {'state': 'healthy'}


def test_common_or_other_pod_release_receipt_cannot_release_own_scoped_pod(tmp_path):
    item = lease(tmp_path)
    root, _ = release_receipt(item)
    item['control_relative'] = 'allocations/owned-pod'
    assert verify_release_receipt(item) is False
    other = root / 'allocations/other-pod'
    other.mkdir(parents=True)
    (root / 'gpu_release_verified.json').rename(other / 'gpu_release_verified.json')
    assert verify_release_receipt(item) is False


@pytest.mark.parametrize(('field', 'value'), [
    ('pod_id', '../other'), ('pod_id', ''), ('gpu_count', True), ('gpu_count', 0),
    ('upper_hourly_usd', float('nan')), ('upper_hourly_usd', float('inf')),
    ('baseline_usd', -1), ('total_cap_usd', 1001), ('total_cap_gpu_hours', 501),
    ('cleanup_reserve_usd', 0), ('deadline_epoch', 1000),
    ('upper_hourly_usd', 40), ('baseline_gpu_hours', 430),
    ('allowed_result_root', '/'), ('allowed_result_root', 'relative/path'),
    ('allowed_result_root', '/workspace/../root'),
])
def test_rejects_invalid_or_unfunded_lease(tmp_path, field, value):
    item = lease(tmp_path)
    item[field] = value
    with pytest.raises(ValueError):
        verify_lease(item)


def test_lease_hash_is_pinned_before_guard_arms(tmp_path):
    path = tmp_path / 'lease.json'
    item = lease(tmp_path)
    atomic_json(path, item)
    digest = sha256_file(path)
    assert load_lease(path, digest) == item
    item['deadline_epoch'] += 60
    atomic_json(path, item)
    with pytest.raises(ValueError, match='hash mismatch'):
        load_lease(path, digest)


def test_completed_release_requires_all_identity_and_artifact_checks(tmp_path):
    item = lease(tmp_path)
    assert verify_release_receipt(item) is False
    root, receipt = release_receipt(item)
    assert verify_release_receipt(item) is True
    for field, value in [('pod_id', 'other-pod'), ('experiment_id', 'other'),
                         ('gpu_release_verified', False), ('all_gpu_workers_stopped', False),
                         ('durable_backup_verified', False), ('manifest_path', '../../outside')]:
        wrong = dict(receipt, **{field: value})
        atomic_json(root / 'gpu_release_verified.json', wrong)
        with pytest.raises(ValueError):
            verify_release_receipt(item)
    atomic_json(root / 'gpu_release_verified.json', receipt)
    (root / 'complete.json').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        verify_release_receipt(item)


def test_release_manifest_cannot_escape_by_symlink(tmp_path):
    item = lease(tmp_path)
    root, _ = release_receipt(item)
    outside = tmp_path / 'outside.json'
    outside.write_bytes((root / 'complete.json').read_bytes())
    (root / 'complete.json').unlink()
    (root / 'complete.json').symlink_to(outside)
    with pytest.raises(ValueError, match='symlinks'):
        verify_release_receipt(item)


class FakeClock:
    def __init__(self, value):
        self.value = value
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


def test_no_network_or_signals_before_deadline_and_delete_retries_are_redacted(tmp_path):
    item = lease(tmp_path)
    item['deadline_epoch'] = 1010
    root = Path(item['allowed_result_root'])
    atomic_json(root / 'gpu_release_verified.json', {'pod_id': 'unauthorized'})
    timer = FakeClock(1000)
    attempts, signals = [], []

    def remove(value, key):
        assert value['pod_id'] == 'owned-pod'
        assert timer.now() >= 1130  # deadline plus atomic-checkpoint grace
        attempts.append(timer.now())
        if len(attempts) < 7:
            raise OSError('secret-key-in-error https://provider/?api_key=secret')
        return True

    def stop(value):
        signals.append(timer.now())
        return True

    result = run_guard(item, 'secret', clock=timer.now, monotonic=timer.now,
                       sleep=timer.sleep, delete=remove, signal_supervisor=stop)
    assert result['delete_attempts'] == 7
    assert signals == [1130]
    assert attempts == [1130, 1131, 1133, 1137, 1145, 1161, 1191]
    assert max(timer.sleeps) <= 30
    assert 'secret' not in (root / 'lease_guard/events.jsonl').read_text()
    assert json.loads((root / 'lease_guard/STOP').read_text())['reason'] == 'immutable_allocation_deadline'


def test_verified_completion_releases_without_waiting_or_signaling(tmp_path):
    item = lease(tmp_path)
    release_receipt(item)
    timer = FakeClock(1000)
    removed = []
    result = run_guard(item, 'secret', clock=timer.now, monotonic=timer.now,
                       sleep=timer.sleep, delete=lambda value, key: removed.append(value['pod_id']) or True,
                       signal_supervisor=lambda _: pytest.fail('completed worker must not be signaled'))
    assert result['reason'] == 'verified_completion'
    assert removed == ['owned-pod']
    assert timer.sleeps == []


def test_wall_clock_rollback_does_not_extend_immutable_lease(tmp_path):
    item = lease(tmp_path)
    item['deadline_epoch'] = 1010
    timer = FakeClock(1000)
    reads = [0]

    def wall_time():
        reads[0] += 1
        return timer.now() if reads[0] <= 2 else timer.now() - 10000

    attempts = []
    run_guard(item, 'secret', clock=wall_time, monotonic=timer.now,
              sleep=timer.sleep, grace_seconds=0,
              delete=lambda *args: attempts.append(timer.now()) or True,
              signal_supervisor=lambda _: False)
    assert attempts == [1010]


class Response:
    status = 204
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def test_delete_targets_only_own_pod_and_requires_absence_proof(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append((request.method, request.full_url, timeout))
        assert request.headers['Authorization'] == 'Bearer secret'
        if request.method == 'GET':
            raise urllib.error.HTTPError(request.full_url, 404, 'absent', {}, None)
        return Response()

    assert delete_own_pod(lease(tmp_path), 'secret', opener=opener) is True
    assert calls == [('DELETE', PROVIDER_BASE + 'owned-pod', 20),
                     ('GET', PROVIDER_BASE + 'owned-pod', 20)]
    assert delete_own_pod(lease(tmp_path), 'secret', opener=lambda *a, **k: Response()) is False


def test_key_stays_outside_results_with_owner_only_permissions(tmp_path):
    item = lease(tmp_path)
    path = tmp_path / 'private-key'
    path.write_text('secret')
    path.chmod(0o600)
    assert read_private_key(path, item['allowed_result_root']) == 'secret'
    path.chmod(0o644)
    with pytest.raises(ValueError, match='owner-only'):
        read_private_key(path, item['allowed_result_root'])
    root = Path(item['allowed_result_root'])
    root.mkdir(parents=True)
    secret = root / 'key'
    secret.write_text('secret')
    secret.chmod(0o600)
    with pytest.raises(ValueError, match='scientific result root'):
        read_private_key(secret, root)


def fake_process(tmp_path, pid):
    root = tmp_path / 'proc'
    (root / str(pid)).mkdir(parents=True)
    # After state(field3), pgrp is offset2 and starttime(field22) offset19.
    fields = ['S', '1', str(pid)] + ['0'] * 16 + ['123456'] + ['0'] * 10
    (root / str(pid) / 'stat').write_text(f'{pid} (python complicated ) name) ' + ' '.join(fields))
    boot = root / 'sys/kernel/random/boot_id'
    boot.parent.mkdir(parents=True)
    boot.write_text('boot-identity')
    return root


def test_signal_requires_matching_boot_pid_start_and_owned_process_group(tmp_path):
    item = lease(tmp_path)
    pid = max(os.getpid(), os.getpgrp()) + 10000
    proc = fake_process(tmp_path, pid)
    registration = dict(linux_process_identity(pid, proc), experiment_id=item['experiment_id'], pod_id=item['pod_id'])
    path = Path(item['allowed_result_root']) / 'lease_guard/supervisor_registration.json'
    atomic_json(path, registration)
    signals = []
    assert signal_registered_supervisor(item, proc_root=proc,
                                        killpg=lambda *args: signals.append(args)) is True
    assert signals == [(pid, signal.SIGTERM)]
    for field, value in [('start_ticks', 999), ('boot_id', 'newboot'), ('pgid', pid + 1),
                         ('pod_id', 'other-pod'), ('pid', 1), ('pid', os.getpid())]:
        atomic_json(path, dict(registration, **{field: value}))
        with pytest.raises(ValueError):
            signal_registered_supervisor(item, proc_root=proc,
                                         killpg=lambda *_: pytest.fail('unsafe process signal'))


def test_http_error_logging_does_not_capture_provider_body_or_url():
    error = urllib.error.HTTPError('https://secret-key.example', 403, 'secret', {}, None)
    assert redacted_error(error) == {'error_type': 'HTTPError', 'http_status': 403}
