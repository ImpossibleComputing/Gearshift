#!/usr/bin/env python3
"""Probe actual cross-host flock exclusion on the intended shared volume.

No accelerator, model, experiment, private test or provider API is imported.
Run hold on one host and try --expect blocked on another, then release the holder
and repeat try --expect acquired. Success on one host alone is not evidence that
a network volume coordinates locks across hosts. The lock inode is never removed.
"""
import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import time
import uuid


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def probe(lock, mode, receipt, *, ttl_seconds=60., release_file=None, expect=None, probe_id=None):
    if mode not in ('hold', 'try') or not 0 < ttl_seconds <= 60:
        raise ValueError('Use hold/try and a bounded TTL in (0,60] seconds')
    if expect not in (None, 'acquired', 'blocked') or (mode == 'hold' and expect == 'blocked'):
        raise ValueError('Invalid expectation for this probe mode')
    lock = Path(lock).absolute(); lock.parent.mkdir(parents=True, exist_ok=True)
    if release_file and Path(release_file).exists():
        raise ValueError('Use a fresh release-file path for this bounded probe')
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0), 0o600)
    acquired = False; stop = {'signal': None}; handlers = {}
    started = time.time(); monotonic = time.monotonic(); st = os.fstat(fd)
    base = {'probe_id': probe_id or uuid.uuid4().hex, 'mode': mode, 'lock_path': str(lock),
            'hostname': socket.gethostname(), 'pid': os.getpid(), 'started_epoch': started,
            'device': st.st_dev, 'inode': st.st_ino, 'mechanism': 'flock_LOCK_EX_LOCK_NB',
            'lock_inode_never_unlinked': True, 'ttl_seconds': ttl_seconds}
    def emit(state, **kwargs):
        row = {**base, 'state': state, 'epoch': time.time(), **kwargs}
        save(receipt, row); print(json.dumps(row, sort_keys=True), flush=True); return row
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); acquired = True
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            emit('blocked', acquired=False, expected=expect, expected_outcome_matched=expect == 'blocked')
            return 0 if expect == 'blocked' else 75
        matched = expect in (None, 'acquired')
        if mode == 'try':
            emit('acquired', acquired=True, expected=expect, expected_outcome_matched=matched)
            return 0 if matched else 1
        emit('holding', acquired=True, expected=expect, expected_outcome_matched=matched,
             deadline_epoch=started+ttl_seconds)
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda number, frame: stop.update(signal=number))
        reason = 'ttl_expired'
        while time.monotonic()-monotonic < ttl_seconds:
            if stop['signal']:
                reason = 'signal'; break
            if release_file and Path(release_file).exists():
                reason = 'release_file'; break
            time.sleep(min(.1, max(0., ttl_seconds-(time.monotonic()-monotonic))))
        fcntl.flock(fd, fcntl.LOCK_UN); acquired = False
        emit('released', acquired=False, release_reason=reason, held_seconds=time.monotonic()-monotonic,
             signal=stop['signal'])
        return 0
    except BaseException as exc:
        emit('failed', acquired=acquired, exception=type(exc).__name__, error=str(exc))
        raise
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['hold','try'], required=True)
    p.add_argument('--lock', required=True); p.add_argument('--receipt', required=True)
    p.add_argument('--ttl-seconds', type=float, default=60); p.add_argument('--release-file')
    p.add_argument('--expect', choices=['acquired','blocked']); p.add_argument('--probe-id')
    a = p.parse_args()
    raise SystemExit(probe(a.lock, a.mode, a.receipt, ttl_seconds=a.ttl_seconds,
                           release_file=a.release_file, expect=a.expect, probe_id=a.probe_id))
