"""Independent, standard-library-only RunPod lease guard.

This process does not supervise scientific progress or require a controller
heartbeat. It contacts the provider only after an immutable deadline or a
verified release receipt. A provider outage can delay billing termination;
stopping local processes is never reported as proof that billing stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import time
import urllib.error
import urllib.request


PROVIDER_BASE = 'https://rest.runpod.io/v1/pods/'
SAFE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def verify_lease(lease):
    """Validate immutable allocation-wide reservations, without double counting.

    upper_hourly_usd is the TOTAL fleet/lease rate, not a per-GPU rate. It
    includes the existing retained storage if the dispatcher assigned that
    liability here. baseline_usd is usage already accrued before allocation.
    """
    if not isinstance(lease, dict):
        raise ValueError('lease must be an object')
    for field in ('experiment_id', 'pod_id'):
        if not isinstance(lease.get(field), str) or not SAFE_ID.fullmatch(lease[field]):
            raise ValueError(f'invalid {field}')
    for field in ('allocation_epoch', 'deadline_epoch', 'upper_hourly_usd',
                  'baseline_usd', 'baseline_gpu_hours', 'total_cap_usd',
                  'total_cap_gpu_hours', 'cleanup_reserve_usd'):
        value = lease.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid {field}')
    if lease['total_cap_usd'] != 1000 or lease['total_cap_gpu_hours'] != 500:
        raise ValueError('lease must preserve cumulative hard ceilings')
    if lease['cleanup_reserve_usd'] != 40:
        raise ValueError('lease must reserve forty dollars for cleanup')
    if type(lease.get('gpu_count')) is not int or not 1 <= lease['gpu_count'] <= 8:
        raise ValueError('invalid gpu_count')
    hours = (lease['deadline_epoch'] - lease['allocation_epoch']) / 3600
    if hours <= 0 or lease['upper_hourly_usd'] <= 0:
        raise ValueError('lease duration and rate must be positive')
    for field in ('other_reserved_usd', 'other_reserved_gpu_hours'):
        value = lease.get(field, 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid {field}')
    reserved = (lease['baseline_usd'] + hours * lease['upper_hourly_usd']
                + lease['cleanup_reserve_usd'] + lease.get('other_reserved_usd', 0))
    gpu_hours = (lease['baseline_gpu_hours'] + hours * lease['gpu_count']
                 + lease.get('other_reserved_gpu_hours', 0))
    if reserved >= lease['total_cap_usd'] or gpu_hours >= lease['total_cap_gpu_hours']:
        raise ValueError('lease worst-case reservation reaches a cumulative hard ceiling')
    root = lease.get('allowed_result_root')
    if not isinstance(root, str) or not Path(root).is_absolute() or '..' in Path(root).parts:
        raise ValueError('allowed_result_root must be an absolute scoped path')
    if len(Path(root).parts) < 3:
        raise ValueError('allowed_result_root is not sufficiently scoped')
    relative = lease.get('control_relative', '')
    if relative not in ('', 'allocations/' + lease['pod_id']):
        raise ValueError('control_relative must name this allocation only')
    return {'reserved_total_usd': reserved, 'reserved_total_gpu_hours': gpu_hours,
            'allocation_hours': hours}


def control_root(lease):
    """Per-pod control files are separate from the shared scientific results."""
    relative = lease.get('control_relative', '')
    if relative not in ('', 'allocations/' + lease['pod_id']):
        raise ValueError('control_relative must name this allocation only')
    root = Path(lease['allowed_result_root'])
    candidate = root / relative
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError('control root cannot traverse a symlink')
        cursor = cursor.parent
    return candidate


def load_lease(path, expected_sha256):
    if not isinstance(expected_sha256, str) or not SHA256.fullmatch(expected_sha256):
        raise ValueError('immutable lease SHA-256 is required')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('immutable lease hash mismatch')
    lease = json.loads(raw)
    verify_lease(lease)
    return lease


def _scoped_file(root, relative):
    if not isinstance(relative, str):
        raise ValueError('manifest path must be a string')
    path = Path(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError('manifest path must remain inside allowed_result_root')
    root = Path(root).resolve()
    candidate = root / path
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError('symlinks are not allowed in release evidence')
        cursor = cursor.parent
    if not candidate.resolve().is_relative_to(root) or not candidate.is_file():
        raise ValueError('release evidence file is missing or outside the result root')
    return candidate


def verify_release_receipt(lease):
    """Require a matching completion receipt and verify every listed durable file.

    The supervisor owns this receipt. Models and private tests do not belong in
    this manifest; it inventories completed scientific records/checkpoints and
    their on-volume backup copies. Volume removal is never authorized here.
    """
    root = Path(lease['allowed_result_root'])
    receipt_path = control_root(lease) / 'gpu_release_verified.json'
    if not receipt_path.exists():
        return False
    receipt = json.loads(_scoped_file(root, str(receipt_path.relative_to(root))).read_text())
    for field in ('experiment_id', 'pod_id'):
        if receipt.get(field) != lease[field]:
            raise ValueError('release receipt identity mismatch')
    for field in ('gpu_release_verified', 'all_gpu_workers_stopped', 'durable_backup_verified'):
        if receipt.get(field) is not True:
            raise ValueError('release receipt does not authorize GPU release')
    manifest_path = _scoped_file(root, receipt.get('manifest_path'))
    if sha256_file(manifest_path) != receipt.get('manifest_sha256'):
        raise ValueError('release manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('experiment_id') != lease['experiment_id'] or manifest.get('pod_id') != lease['pod_id']:
        raise ValueError('release manifest identity mismatch')
    files = manifest.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('release manifest must contain durable files')
    seen = set()
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get('path'), str):
            raise ValueError('invalid release manifest record')
        if record['path'] in seen:
            raise ValueError('duplicate release manifest path')
        seen.add(record['path'])
        path = _scoped_file(root, record['path'])
        if type(record.get('bytes')) is not int or record['bytes'] < 0:
            raise ValueError('invalid release manifest size')
        if path.stat().st_size != record['bytes'] or sha256_file(path) != record.get('sha256'):
            raise ValueError('durable release evidence has changed')
    return True


def read_private_key(path, allowed_result_root):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('API key must be a private absolute regular file')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise ValueError('API key file must be owner-only and owned by the guard user')
    if path.resolve().is_relative_to(Path(allowed_result_root).resolve()):
        raise ValueError('API key must not be under scientific result root')
    value = path.read_text().strip()
    if not value or any(character.isspace() for character in value):
        raise ValueError('API key file contains no valid token')
    return value


def redacted_error(error):
    """Never persist exception bodies/URLs, which may echo authorization data."""
    result = {'error_type': type(error).__name__}
    if isinstance(error, urllib.error.HTTPError):
        result['http_status'] = error.code
    return result


def delete_own_pod(lease, api_key, opener=urllib.request.urlopen):
    """Delete only the immutable pod ID, then require a provider absence proof."""
    verify_lease(lease)
    url = PROVIDER_BASE + lease['pod_id']
    headers = {'Authorization': 'Bearer ' + api_key, 'User-Agent': 'gearshift-lease-guard'}
    for method in ('DELETE', 'GET'):
        request = urllib.request.Request(url, method=method, headers=headers)
        try:
            with opener(request, timeout=20) as response:
                status = response.status
                # No response content is needed, retained, or logged.
                if not 200 <= status < 300:
                    raise RuntimeError('provider did not acknowledge request')
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return True
            raise
    # An acknowledged DELETE is not enough if GET still finds the allocation.
    return False


def linux_process_identity(pid, proc_root=Path('/proc')):
    """Bind a registered process group to a boot and process creation time."""
    if type(pid) is not int or pid <= 1:
        raise ValueError('invalid supervisor PID')
    proc_root = Path(proc_root)
    raw = (proc_root / str(pid) / 'stat').read_text()
    # comm may contain spaces and parentheses. Remaining fields start at state.
    suffix = raw[raw.rfind(')') + 2:].split()
    return {'pid': pid, 'pgid': int(suffix[2]), 'start_ticks': int(suffix[19]),
            'boot_id': (proc_root / 'sys/kernel/random/boot_id').read_text().strip()}


def signal_registered_supervisor(lease, *, proc_root=Path('/proc'), killpg=os.killpg):
    root = Path(lease['allowed_result_root'])
    path = control_root(lease) / 'lease_guard/supervisor_registration.json'
    if not path.exists():
        return False
    registration = json.loads(_scoped_file(root, str(path.relative_to(root))).read_text())
    if any(registration.get(field) != lease[field] for field in ('experiment_id', 'pod_id')):
        raise ValueError('supervisor registration identity mismatch')
    pid = registration.get('pid')
    if type(pid) is not int or pid <= 1 or pid == os.getpid():
        raise ValueError('unsafe supervisor PID')
    if registration.get('pgid') != pid or pid == os.getpgrp():
        raise ValueError('supervisor must own a separate process group')
    try:
        current = linux_process_identity(pid, proc_root)
    except FileNotFoundError:
        return False
    if any(registration.get(field) != current[field] for field in ('pid', 'pgid', 'start_ticks', 'boot_id')):
        raise ValueError('supervisor process identity changed; refusing to signal')
    killpg(pid, signal.SIGTERM)
    return True


def run_guard(lease, api_key, *, poll_seconds=5, grace_seconds=120,
              clock=time.time, monotonic=time.monotonic, sleep=time.sleep,
              delete=delete_own_pod, signal_supervisor=signal_registered_supervisor):
    """Run independently until provider confirms that this allocation is absent."""
    reservation = verify_lease(lease)
    if not 0 < poll_seconds <= 30 or not 0 <= grace_seconds <= 120:
        raise ValueError('invalid guard polling or shutdown grace')
    root = Path(lease['allowed_result_root'])
    control = control_root(lease) / 'lease_guard'
    control.mkdir(parents=True, exist_ok=True)

    def record(state, **details):
        value = {'epoch': clock(), 'pid': os.getpid(), 'experiment_id': lease['experiment_id'],
                 'pod_id': lease['pod_id'], 'state': state, **details}
        atomic_json(control / 'status.json', value)
        with (control / 'events.jsonl').open('a') as output:
            output.write(json.dumps(value, allow_nan=False) + '\n')
            output.flush()
            os.fsync(output.fileno())

    record('armed', deadline_epoch=lease['deadline_epoch'], **reservation,
           provider_scheduled_expiry=False, controller_heartbeat_required=False)
    # A later wall-clock rollback cannot extend the already reserved lease.
    monotonic_deadline = monotonic() + max(0, lease['deadline_epoch'] - clock())
    last_bad_receipt = None
    while clock() < lease['deadline_epoch'] and monotonic() < monotonic_deadline:
        try:
            if verify_release_receipt(lease):
                reason = 'verified_completion'
                break
        except Exception as error:
            # Bad or half-written receipts cannot stop a healthy job.
            fingerprint = redacted_error(error)
            if fingerprint != last_bad_receipt:
                record('unauthorized_release_ignored', **fingerprint)
                last_bad_receipt = fingerprint
        sleep(min(poll_seconds, max(0, lease['deadline_epoch'] - clock()),
                  max(0, monotonic_deadline - monotonic())))
    else:
        reason = 'immutable_allocation_deadline'

    atomic_json(control / 'STOP', {'epoch': clock(), 'reason': reason,
                                  'experiment_id': lease['experiment_id'], 'pod_id': lease['pod_id']})
    record('stop_requested', reason=reason)
    if reason == 'immutable_allocation_deadline':
        # Workers check 120 seconds earlier. This extra bounded grace permits an
        # in-flight atomic checkpoint; it is covered by the cleanup reservation.
        grace_end = monotonic() + grace_seconds
        while monotonic() < grace_end:
            sleep(min(poll_seconds, grace_end - monotonic()))
        try:
            signaled = signal_supervisor(lease)
            record('supervisor_signal', signaled=bool(signaled))
        except Exception as error:
            record('supervisor_signal_refused', **redacted_error(error))

    delay = 1
    attempts = 0
    while True:
        attempts += 1
        try:
            absent = delete(lease, api_key)
            if absent:
                record('provider_confirmed_absent', reason=reason, delete_attempts=attempts)
                return {'state': 'provider_confirmed_absent', 'delete_attempts': attempts, 'reason': reason}
            record('delete_not_yet_confirmed', delete_attempts=attempts)
        except Exception as error:
            record('delete_retry', delete_attempts=attempts, **redacted_error(error))
        # Retry forever after the deadline rather than abandoning a billed pod.
        # This is not a guarantee against a provider-wide control-plane outage.
        sleep(delay)
        delay = min(30, delay * 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lease', required=True)
    parser.add_argument('--lease-sha256', required=True)
    parser.add_argument('--key-file', required=True)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--grace-seconds', type=float, default=120)
    args = parser.parse_args()
    lease = load_lease(args.lease, args.lease_sha256)
    key = read_private_key(args.key_file, lease['allowed_result_root'])
    # No key enters command-line arguments, result files, or worker environments.
    run_guard(lease, key, poll_seconds=args.poll_seconds, grace_seconds=args.grace_seconds)


if __name__ == '__main__':
    main()
