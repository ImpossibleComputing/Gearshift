"""Secondary release tests use only synthetic compact bytes and fake processes."""
import fcntl
import importlib.util
from pathlib import Path

import pytest

from gearshift.coding_control import write, sha
from gearshift.coding_confirmation_lease import verify_release_receipt
from scripts import coding_confirmation_secondary_supervisor as s

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('secondary_supervisor_primary_fixture', ROOT / 'tests/test_coding_confirmation_generation_supervisor.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def pending(root, pod='pod1'):
    job = 'receiver_atcoder__unfinished'
    owner = root / 'secondary/claims' / (job + '.owner.json')
    write(owner, {'attempt_id': pod + '_secondary_0_attempt1'})
    (owner.parent / (job + '.lock')).touch()
    p = root / 'secondary/tasks/atcoder__unfinished/FIXED_M/seed_0/sampler/resume.json'
    write(p, {'committed_tokens': [1, 2], 'rng': [3, 4]})
    return job, p


def test_secondary_partial_snapshot_holds_own_lock_and_preserves_exact_resume(tmp_path):
    job, p = pending(tmp_path); folder = tmp_path / 'allocations/pod1/secondary_supervisor'
    with (tmp_path / 'secondary/claims' / (job + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s.preserve_partial_jobs(tmp_path, folder, 'pod1')
        assert not folder.exists()
    s.preserve_partial_jobs(tmp_path, folder, 'pod1')
    target = folder / 'partial_snapshots' / p.relative_to(tmp_path)
    assert target.read_bytes() == p.read_bytes()
    write(p, {'reclaimed': True}); assert s.read(target)['committed_tokens'] == [1, 2]


def test_other_allocation_partial_or_primary_work_is_not_snapshotted(tmp_path):
    pending(tmp_path, 'otherpod'); f.pending(tmp_path, 'pod1')
    folder = tmp_path / 'snapshot'; s.preserve_partial_jobs(tmp_path, folder, 'pod1')
    assert not folder.exists()


def test_closed_secondary_draw_timing_and_control_reuse_are_backed_up(tmp_path):
    original = f.draw(tmp_path)
    target = tmp_path / 'secondary/tasks/atcoder__a/FIXED_M/seed_0'; target.parent.mkdir(parents=True)
    original.rename(target)
    reuse = target.parents[1] / 'control_reuse.json'; write(reuse, {'primary_control_sha256': 'f' * 64})
    files = s.closed_files(tmp_path, tmp_path / 'empty_control', 'pod1')
    assert reuse in files and target / 'sampler/completion_timing.json' in files


def test_secondary_release_verifies_archives_partial_progress_and_own_workers(tmp_path):
    f.history(tmp_path); pending(tmp_path)
    folder = tmp_path / 'allocations/pod1/secondary_supervisor'; folder.mkdir(parents=True)
    write(folder / 'workers_stopped.json', {'stopped': True})
    status = tmp_path / 'workers/pod1_secondary_0/attempts/a/status.json'; write(status, {'stopped': True})
    lease = {'experiment_id': 'test', 'pod_id': 'pod1', 'allowed_result_root': str(tmp_path),
             'control_relative': 'allocations/pod1'}
    s.release(tmp_path, lease, folder, 'ready_secondary_jobs_drained')
    assert verify_release_receipt(lease)
    rows = s.read(tmp_path / 'allocations/pod1/release_manifest.json')['files']
    assert str(status.relative_to(tmp_path)) in {r['path'] for r in rows}
    assert any('partial_snapshots/secondary/tasks/' in r['path'] for r in rows)
    archives = [r for r in rows if r['path'].endswith('.tar.gz')]
    assert len(archives) == 2 and archives[0]['sha256'] == archives[1]['sha256']
    (tmp_path / archives[0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError): verify_release_receipt(lease)


def run_fixture(tmp_path, monkeypatch, processes):
    root = tmp_path / 'results'; lease = {'experiment_id': 'test', 'pod_id': 'pod1',
        'deadline_epoch': s.time.time() + 1000, 'gpu_count': 1,
        'allowed_result_root': str(root), 'control_relative': 'allocations/pod1'}
    plan = {'experiment_id': 'test', 'result_root': 'results', 'lease_path': 'lease.json', 'lease_sha256': 'test'}
    write(tmp_path / 'plan.json', plan)
    monkeypatch.setattr(s, 'ROOT', tmp_path); monkeypatch.setattr(s, 'load_lease', lambda *a: lease)
    monkeypatch.setattr(s, 'validate_guard', lambda *a: None)
    monkeypatch.setattr(s.secondary, 'validate_secondary', lambda *a, **kw: None)
    monkeypatch.setattr(s, 'linux_process_identity', lambda pid: {'pid': pid})
    monkeypatch.setattr(s.os, 'getpid', lambda: 100); monkeypatch.setattr(s.os, 'getpgrp', lambda: 100)
    monkeypatch.setattr(s.os, 'getpgid', lambda pid: 100); monkeypatch.setattr(s.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(s.time, 'sleep', lambda *a: None)
    commands = []; queue = iter(processes)
    def popen(command, **kw):
        commands.append((command, kw)); return next(queue)
    monkeypatch.setattr(s.subprocess, 'Popen', popen)
    released = []
    def release(root, lease, folder, outcome):
        assert all(p.poll() is not None for p in processes)
        released.append(outcome)
    monkeypatch.setattr(s, 'release', release)
    return root / 'allocations/pod1/secondary_supervisor', commands, released


def test_supervisor_launches_only_secondary_offline_same_group_and_exits_when_drained(tmp_path, monkeypatch):
    proc = f.FakeProcess(exitcode=0); folder, commands, released = run_fixture(tmp_path, monkeypatch, [proc])
    monkeypatch.setenv('RUNPOD_API_KEY', 'must-not-leak')
    s.run('plan.json')
    command, kw = commands[0]
    assert command[1:3] == ['scripts/coding_confirmation_secondary_generate.py', 'run']
    assert kw['start_new_session'] is False and kw['env']['CUDA_VISIBLE_DEVICES'] == '0'
    assert kw['env']['RUNPOD_POD_ID'] == 'pod1' and kw['env']['HF_HUB_OFFLINE'] == '1'
    assert 'RUNPOD_API_KEY' not in kw['env']
    assert released == ['ready_secondary_jobs_drained']


def test_secondary_retry_is_bounded_to_two_restarts_and_preserves_all_attempts(tmp_path, monkeypatch):
    processes = [f.FakeProcess(exitcode=1) for _ in range(3)]
    folder, commands, released = run_fixture(tmp_path, monkeypatch, processes); s.run('plan.json')
    assert len(commands) == 3 and len(list(folder.glob('gpu_0_attempt_*/exited.json'))) == 3
    assert released == ['failed_preserved'] and 'restart allowance exhausted' in s.read(folder / 'failure.json')['error']


def test_scientific_failure_never_retries(tmp_path, monkeypatch):
    folder, commands, released = run_fixture(tmp_path, monkeypatch, [f.FakeProcess(exitcode=65)]); s.run('plan.json')
    assert len(commands) == 1 and released == ['failed_preserved']


def test_startup_receipt_failure_stops_registered_child_before_release(tmp_path, monkeypatch):
    proc = f.FakeProcess(); folder, commands, released = run_fixture(tmp_path, monkeypatch, [proc])
    original = s.atomic_json
    def failing(path, value):
        if Path(path).name == 'started.json': raise OSError('synthetic write failure')
        return original(path, value)
    monkeypatch.setattr(s, 'atomic_json', failing); s.run('plan.json')
    assert proc.terminated and proc.waited and released == ['failed_preserved']


def test_unverified_secondary_process_group_withholds_idle_release(tmp_path, monkeypatch):
    proc = f.FakeProcess(); folder, commands, released = run_fixture(tmp_path, monkeypatch, [proc])
    monkeypatch.setattr(s.os, 'getpgid', lambda pid: 999); s.run('plan.json')
    assert proc.terminated and proc.waited and released == []
    assert (folder / 'release_withheld.json').exists()
