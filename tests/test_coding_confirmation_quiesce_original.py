"""Synthetic local stop/snapshot checks: no process signal or provider action."""
import copy
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

import pytest

from gearshift.coding_control import digest, sha, write
from scripts import coding_confirmation_quiesce_original as s

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'; (repo / 'scripts').mkdir(parents=True)
    shutil.copyfile(ROOT / s.SCRIPT, repo / s.SCRIPT)
    (repo / s.SUPERVISOR).write_text('# synthetic frozen supervisor\n')
    tids = ['atcoder/task_' + str(i) for i in range(200)]
    d = {'experiment_id': s.EXPERIMENT, 'task_count': 200, 'primary_answer_count': 4800,
         'task_ids': tids, 'source_commit': 'a' * 40}
    monkeypatch.setattr(s, 'declaration', lambda repo: copy.deepcopy(d))
    root = repo / 'results/coding_pilot_v1' / s.EXPERIMENT
    leases = {}; now = time.time()
    for pod, rel in s.ORIGINALS.items():
        lease = {'experiment_id': s.EXPERIMENT, 'pod_id': pod, 'allocation_epoch': now - 20,
                 'deadline_epoch': now + 3600, 'upper_hourly_usd': 12, 'baseline_usd': 100,
                 'total_cap_usd': 2500, 'cleanup_reserve_usd': 40, 'gpu_count': 2,
                 'allowed_result_root': str(root), 'network_volume_id': 'original-volume',
                 'control_relative': 'allocations/' + pod}
        write(repo / rel, lease); leases[pod] = lease
        control = s.control_root(lease)
        stop = control / 'generation_supervisor/workers_stopped.json'
        write(stop, {'all_registered_workers_stopped': True, 'process_group_ownership_verified': True})
        backup = control / 'release_backups/one/compact_a.tar.gz'
        backup.parent.mkdir(parents=True); backup.write_bytes(b'synthetic durable backup')
        rows = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in [stop, backup]]
        manifest = control / 'release_manifest.json'
        write(manifest, {'experiment_id': s.EXPERIMENT, 'pod_id': pod, 'files': rows})
        write(control / 'gpu_release_verified.json', {'experiment_id': s.EXPERIMENT, 'pod_id': pod,
              'gpu_release_verified': True, 'all_gpu_workers_stopped': True, 'durable_backup_verified': True,
              'manifest_path': str(manifest.relative_to(root)), 'manifest_sha256': sha(manifest)})
    result = s.prepare(repo, 'evidence/control_plan.json')
    partial = root / 'primary/tasks/atcoder__task_0/large_history/resume.json'
    write(partial, {'tokens': [1, 2, 3], 'state': 'interrupted'})
    write(root / 'claims/source_atcoder__task_0.owner.json', {'state': 'released'})
    write(root / 'claims/history/source_atcoder__task_0/previous.json', {'state': 'released'})
    pod = next(iter(s.ORIGINALS))
    worker = root / 'workers' / (pod + '_generation_0') / 'attempts/original'
    runtime = {'torch': 'synthetic-runtime'}
    write(worker / 'runtime.json', runtime)
    write(worker / 'model_setup.json', {'declaration_sha256': s.DECLARATION_SHA,
          'runtime_identity_sha256': digest(runtime)})
    write(worker / 'telemetry.jsonl', {'public_measurement': 1})
    write(root / 'primary/jobs/source_atcoder__task_1/complete.json', {
          'experiment_id': s.EXPERIMENT, 'declaration_sha256': s.DECLARATION_SHA,
          'runtime_path': str((worker / 'runtime.json').relative_to(root)),
          'setup_path': str((worker / 'model_setup.json').relative_to(root)),
          'runtime_identity_sha256': digest(runtime)})
    absent = []
    for pod in s.ORIGINALS:
        path = repo / 'evidence' / (pod + '_absent.json')
        write(path, {'pod_id': pod, 'source': 'runpod_provider_inspection',
              'state': 'provider_confirmed_absent', 'observed_epoch': now,
              'verification_method': 'get_pod_404', 'http_status': 404})
        absent.append({'pod_id': pod, 'path': str(path.relative_to(repo)), 'sha256': sha(path)})
    write(repo / 'evidence/absence.json', {'receipts': absent})
    monkeypatch.setenv('RUNPOD_POD_ID', s.TRAINING_POD)
    return {'repo': repo, 'root': root, 'd': d, 'leases': leases, 'plan': result,
            'absence': 'evidence/absence.json', 'absence_sha': sha(repo / 'evidence/absence.json')}


def snapshot(c, output='evidence/snapshot'):
    return s.snapshot(c['repo'], c['plan']['plan_path'], c['plan']['plan_sha256'],
                      c['absence'], c['absence_sha'], output)


def live_supervisor(c, tmp_path, monkeypatch):
    pod = next(iter(s.ORIGINALS)); lease = c['leases'][pod]
    proc = tmp_path / 'proc'; pid = 9876
    fields = ['S', '1', str(pid)] + ['0'] * 17; fields[19] = '123456'
    (proc / str(pid)).mkdir(parents=True)
    (proc / str(pid) / 'stat').write_text(str(pid) + ' (python worker) ' + ' '.join(fields))
    (proc / 'sys/kernel/random').mkdir(parents=True)
    (proc / 'sys/kernel/random/boot_id').write_text('original-boot')
    (proc / str(pid) / 'cwd').symlink_to(c['repo'], target_is_directory=True)
    argv = [str(c['repo'] / '.pilot-venv/bin/python'), '-u', s.SUPERVISOR, '--plan', 'dispatch.json']
    (proc / str(pid) / 'cmdline').write_bytes(b'\0'.join(v.encode() for v in argv) + b'\0')
    write(c['repo'] / 'dispatch.json', {'experiment_id': s.EXPERIMENT,
          'declaration_sha256': s.DECLARATION_SHA, 'result_root': str(c['root'].relative_to(c['repo'])),
          'lease_path': s.ORIGINALS[pod], 'lease_sha256': sha(c['repo'] / s.ORIGINALS[pod])})
    write(s.control_root(lease) / 'lease_guard/supervisor_registration.json', {
          'experiment_id': s.EXPERIMENT, 'pod_id': pod, 'pid': pid, 'pgid': pid,
          'start_ticks': 123456, 'boot_id': 'original-boot'})
    monkeypatch.setenv('RUNPOD_POD_ID', pod)
    return lease, proc, pid


def test_snapshot_is_exact_public_archive_and_sharding_compatible(prepared):
    c = prepared
    before = {str(p.relative_to(c['root'])): sha(p) for p in c['root'].rglob('*') if p.is_file()}
    result = snapshot(c)
    folder = c['repo'] / 'evidence/snapshot'
    q = s.read(folder / 'quiescence.json'); fragment = s.read(folder / 'partition_inventory_fragment.json')
    copied_index = str((folder / 'provider_absence_index.json').relative_to(c['repo']))
    copied_absence = s.verify_absence(c['repo'], copied_index, sha(c['repo'] / copied_index), c['leases'])
    assert all(path.is_relative_to(folder) for path in copied_absence.values())
    assert result['touched_task_count'] == 2
    assert [p['pod_id'] for p in fragment['original_generation_allocations']] == list(s.ORIGINALS)
    assert fragment['excluded_training_pod_ids'] == [s.TRAINING_POD]
    assert all('workers' not in Path(row['path']).parts for row in q['files'])
    with tarfile.open(c['repo'] / result['archive']['path']) as archive:
        names = archive.getnames()
        assert any(name.endswith('/model_setup.json') for name in names)
        assert any(name.endswith('/runtime.json') for name in names)
        assert any('claims/history/' in name for name in names)
        assert not any('private' in Path(name).parts for name in names)
        assert not any(name.endswith('.lock') for name in names)
        for member in archive.getmembers(): assert member.isfile()
    after = {str(p.relative_to(c['root'])): sha(p) for p in c['root'].rglob('*') if p.is_file()}
    assert after == before
    partition = {'schema': 1, 'experiment_id': s.EXPERIMENT, 'declaration_sha256': s.DECLARATION_SHA,
        'source_commit': c['d']['source_commit'], 'task_ids': c['d']['task_ids'], 'task_count': 200,
        'whole_task_assignment': True, 'static_after_freeze': True, 'origin_shard_id': 'origin',
        'shards': {'origin': c['d']['task_ids'][:100], 'new': c['d']['task_ids'][100:]},
        'shard_storage': {'origin': {'network_volume_id': 'old', 'allowed_result_root': '/workspace/origin'},
                          'new': {'network_volume_id': 'new', 'allowed_result_root': '/workspace/new'}}, **fragment}
    assert s.sharded.validate_partition(c['d'], partition, q) == partition['shards']
    with pytest.raises(FileExistsError): snapshot(c)


@pytest.mark.parametrize('mutation', ['wrong_reader', 'one_absent', 'not_absent', 'backup_changed', 'stop_false', 'runtime_missing', 'heavy', 'symlink'])
def test_snapshot_refuses_unproved_or_changed_inputs(prepared, monkeypatch, mutation):
    c = prepared; pod = next(iter(s.ORIGINALS)); control = s.control_root(c['leases'][pod])
    if mutation == 'wrong_reader': monkeypatch.setenv('RUNPOD_POD_ID', pod)
    elif mutation in ('one_absent', 'not_absent'):
        index = s.read(c['repo'] / c['absence'])
        if mutation == 'one_absent': index['receipts'].pop()
        else:
            row = index['receipts'][0]; path = c['repo'] / row['path']; val = s.read(path)
            val['http_status'] = 503; write(path, val); row['sha256'] = sha(path)
        write(c['repo'] / c['absence'], index); c['absence_sha'] = sha(c['repo'] / c['absence'])
    elif mutation == 'backup_changed': next((control / 'release_backups').rglob('*.tar.gz')).write_bytes(b'changed')
    elif mutation == 'stop_false': write(control / 'generation_supervisor/workers_stopped.json', {'all_registered_workers_stopped': False})
    elif mutation == 'runtime_missing': next((c['root'] / 'workers').rglob('runtime.json')).unlink()
    elif mutation == 'heavy': (c['root'] / 'primary/tasks/atcoder__task_0/mapper.pt.tmp').write_bytes(b'heavy')
    else: (c['root'] / 'primary/tasks/atcoder__task_0/linked').symlink_to(c['repo'])
    with pytest.raises((ValueError, FileNotFoundError)): snapshot(c)
    assert not (c['repo'] / 'evidence/snapshot/SNAPSHOT_COMPLETE.json').exists()


def test_changed_source_during_archive_prevents_completion(prepared, monkeypatch):
    c = prepared; original = s.make_archive
    def changed(root, archive, rows):
        original(root, archive, rows)
        write(root / 'primary/tasks/atcoder__task_0/new.json', {'unexpected': True})
    monkeypatch.setattr(s, 'make_archive', changed)
    with pytest.raises(ValueError, match='population changed'): snapshot(c)
    assert not (c['repo'] / 'evidence/snapshot/quiescence.json').exists()


def test_inspect_checks_registered_live_identity_and_exact_entrypoint(prepared, tmp_path, monkeypatch):
    lease, proc, pid = live_supervisor(prepared, tmp_path, monkeypatch)
    result = s.inspect_supervisor(lease, proc_root=proc)
    assert result['pid'] == pid and result['start_ticks'] == 123456
    path = s.control_root(lease) / 'lease_guard/supervisor_registration.json'
    row = s.read(path); row['start_ticks'] += 1; write(path, row)
    with pytest.raises(ValueError, match='process identity changed'):
        s.inspect_supervisor(lease, proc_root=proc)


def test_inspect_rejects_training_entrypoint_even_with_matching_pid(prepared, tmp_path, monkeypatch):
    lease, proc, pid = live_supervisor(prepared, tmp_path, monkeypatch)
    path = proc / str(pid) / 'cmdline'
    path.write_bytes(path.read_bytes().replace(s.SUPERVISOR.encode(), b'scripts/coding_confirmation_replication_supervisor.py'))
    with pytest.raises(ValueError, match='frozen original generation supervisor'):
        s.inspect_supervisor(lease, proc_root=proc)


def test_stop_needs_execute_never_targets_training_and_does_not_retry(prepared, monkeypatch):
    c = prepared; calls = []
    monkeypatch.setattr(s, 'inspect_supervisor', lambda lease: {'pid': 7654, 'pod_id': lease['pod_id']})
    monkeypatch.setattr(s, 'signal_verified_pid', lambda *a: calls.append(a))
    args = (c['repo'], c['plan']['plan_path'], c['plan']['plan_sha256'])
    pod = next(iter(s.ORIGINALS))
    with pytest.raises(ValueError, match='Explicit'): s.stop(*args, pod, 'evidence/stop')
    with pytest.raises(ValueError, match='Only'): s.stop(*args, s.TRAINING_POD, 'evidence/stop', execute=True)
    assert calls == []
    result = s.stop(*args, pod, 'evidence/stop', execute=True)
    assert result['signal_sent'] and len(calls) == 1
    with pytest.raises(FileExistsError): s.stop(*args, pod, 'evidence/stop', execute=True)
    assert len(calls) == 1


@pytest.mark.parametrize('drift', [False, True])
def test_pidfd_signal_rechecks_identity_and_targets_only_supervisor(prepared, monkeypatch, drift):
    before = {'pid': 7654, 'pgid': 7654, 'start_ticks': 55, 'boot_id': 'boot',
              'registration_sha256': 'a', 'dispatch_sha256': 'b'}
    after = dict(before)
    if drift: after['start_ticks'] += 1
    calls, closed = [], []
    monkeypatch.setattr(s.os, 'pidfd_open', lambda pid, flags: 54321, raising=False)
    monkeypatch.setattr(s.signal, 'pidfd_send_signal', lambda *a: calls.append(a), raising=False)
    monkeypatch.setattr(s.os, 'close', lambda fd: closed.append(fd))
    monkeypatch.setattr(s, 'inspect_supervisor', lambda *a, **kw: after)
    if drift:
        with pytest.raises(ValueError, match='changed before signal'): s.signal_verified_pid({}, before)
        assert calls == []
    else:
        s.signal_verified_pid({}, before)
        assert calls == [(54321, s.signal.SIGTERM, None, 0)]
    assert closed == [54321]
