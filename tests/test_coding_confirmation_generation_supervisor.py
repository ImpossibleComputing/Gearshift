"""Release must protect resumable progress without racing another allocation."""
import fcntl
from pathlib import Path
import pytest
from gearshift.coding_control import write, sha, digest
from gearshift.coding_confirmation_lease import verify_release_receipt
from scripts import coding_confirmation_generation_supervisor as sup


def sampler(folder, kind):
    name = 'source_history.json' if kind == 'reasoning' else 'answer_record.json'
    for n in ['identity.json', 'resume.json', name]: write(folder / n, {'test': n})
    receipt = {'record_file_sha256': sha(folder / name), 'record_sha256': digest({'test': name}),
               'identity_sha256': digest({'test': 'identity.json'})}
    write(folder / 'complete.json', receipt)
    write(folder / 'completion_timing.json', {key: receipt[key] for key in ('record_sha256', 'identity_sha256')})


def history(root):
    folder = root / 'primary/tasks/atcoder__a/large_history'
    sampler(folder, 'reasoning')
    write(folder / 'history_ready.json', {'sha256': sha(folder / 'source_history.json')})
    return folder


def draw(root):
    folder = root / 'primary/tasks/atcoder__a/A/seed_0'
    sampler(folder / 'sampler', 'answer')
    write(folder / 'answer.json', {'immutable': True})
    write(folder / 'draw_identity.json', {'immutable': True})
    write(folder / 'draw_complete.json', {'answer_sha256': sha(folder / 'answer.json'),
        'sampler_complete_sha256': sha(folder / 'sampler/complete.json')})
    return folder


def test_only_committed_sampler_transactions_enter_shared_snapshot(tmp_path):
    h = history(tmp_path); a = draw(tmp_path)
    pending = tmp_path / 'primary/tasks/atcoder__b/large_history'
    sampler(pending, 'reasoning')  # Not yet advertised for fan-out.
    files = sup.closed_files(tmp_path, tmp_path / 'supervisor', 'pod1')
    assert h / 'resume.json' in files and a / 'sampler/resume.json' in files
    assert h / 'completion_timing.json' in files and a / 'sampler/completion_timing.json' in files
    assert not any(p.is_relative_to(pending) for p in files)
    (h / 'source_history.json').write_text('{}')
    with pytest.raises(ValueError, match='history changed'): sup.closed_files(tmp_path, tmp_path / 'supervisor', 'pod1')


def test_changed_completed_draw_cannot_authorize_release(tmp_path):
    a = draw(tmp_path); write(a / 'answer.json', {'corrupt': True})
    with pytest.raises(ValueError, match='draw changed'): sup.closed_files(tmp_path, tmp_path / 'supervisor', 'pod1')


def pending(root, pod='pod1'):
    job = 'source_atcoder__unfinished'
    owner = root / 'claims' / (job + '.owner.json')
    write(owner, {'attempt_id': pod + '_generation_0_attempt1'})
    (root / 'claims' / (job + '.lock')).touch()
    p = root / 'primary/tasks/atcoder__unfinished/large_history/resume.json'
    write(p, {'tokens': [1, 2], 'rng_state': [4, 5]})
    return job, p


def test_partial_snapshot_holds_job_lock_and_preserves_exact_sampler_bytes(tmp_path):
    job, p = pending(tmp_path)
    folder = tmp_path / 'allocations/pod1/generation_supervisor'; folder.mkdir(parents=True)
    with (tmp_path / 'claims' / (job + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sup.preserve_partial_jobs(tmp_path, folder, 'pod1')
        assert not (folder / 'partial_snapshots').exists()
    sup.preserve_partial_jobs(tmp_path, folder, 'pod1')
    saved = folder / 'partial_snapshots' / p.relative_to(tmp_path)
    assert saved.read_bytes() == p.read_bytes()
    write(p, {'reclaimed_by_another_worker': True})
    assert sup.read(saved)['tokens'] == [1, 2]


def test_other_allocation_partial_progress_is_never_read_for_backup(tmp_path):
    pending(tmp_path, 'otherpod')
    folder = tmp_path / 'supervisor'
    sup.preserve_partial_jobs(tmp_path, folder, 'pod1')
    assert not folder.exists()


def test_release_verifies_two_archives_and_guard_rechecks_all_bytes(tmp_path):
    history(tmp_path); draw(tmp_path); pending(tmp_path)
    lease = {'experiment_id': 'test', 'pod_id': 'pod1',
             'allowed_result_root': str(tmp_path), 'control_relative': 'allocations/pod1'}
    folder = tmp_path / 'allocations/pod1/generation_supervisor'; folder.mkdir(parents=True)
    write(folder / 'workers_stopped.json', {'all_stopped': True})
    sup.release(tmp_path, lease, folder, 'ready_jobs_drained')
    assert verify_release_receipt(lease)
    manifest = sup.read(tmp_path / 'allocations/pod1/release_manifest.json')
    archives = [r for r in manifest['files'] if r['path'].endswith('.tar.gz')]
    assert len(archives) == 2 and archives[0]['sha256'] == archives[1]['sha256']
    (tmp_path / archives[1]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'): verify_release_receipt(lease)


def test_guard_rejects_cross_pod_and_wrong_deadline_before_worker_start(tmp_path, monkeypatch):
    monkeypatch.setenv('RUNPOD_POD_ID', 'pod1'); monkeypatch.setattr(sup.os, 'kill', lambda *a: None)
    lease = {'experiment_id': 'test', 'pod_id': 'pod1', 'deadline_epoch': 100}
    state = {**lease, 'state': 'armed', 'pid': 10}
    write(tmp_path / 'lease_guard/status.json', state)
    assert sup.validate_guard(lease, tmp_path) == state
    monkeypatch.setenv('RUNPOD_POD_ID', 'otherpod')
    with pytest.raises(ValueError, match='different pod'): sup.validate_guard(lease, tmp_path)
    monkeypatch.setenv('RUNPOD_POD_ID', 'pod1'); state['deadline_epoch'] = 101
    write(tmp_path / 'lease_guard/status.json', state)
    with pytest.raises(ValueError, match='not armed'): sup.validate_guard(lease, tmp_path)


class FakeProcess:
    pid = 101

    def __init__(self, exitcode=None):
        self.returncode = exitcode
        self.terminated = False
        self.waited = False

    def poll(self): return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        assert self.returncode is not None, 'Supervisor must stop live workers before waiting'
        self.waited = True
        return self.returncode


def run_fixture(tmp_path, monkeypatch, proc):
    root = tmp_path / 'results'
    lease = {'experiment_id': 'test', 'pod_id': 'pod1', 'deadline_epoch': sup.time.time() + 1000,
             'gpu_count': 1, 'allowed_result_root': str(root), 'control_relative': 'allocations/pod1'}
    plan = {'experiment_id': 'test', 'result_root': 'results', 'lease_path': 'lease.json',
            'lease_sha256': 'test', 'preferences': ['source']}
    write(tmp_path / 'plan.json', plan)
    monkeypatch.setattr(sup, 'ROOT', tmp_path)
    monkeypatch.setattr(sup, 'load_lease', lambda *args: lease)
    monkeypatch.setattr(sup, 'validate_guard', lambda *args: None)
    monkeypatch.setattr(sup, 'linux_process_identity', lambda pid: {'pid': pid})
    monkeypatch.setattr(sup.os, 'getpid', lambda: 100)
    monkeypatch.setattr(sup.os, 'getpgrp', lambda: 100)
    monkeypatch.setattr(sup.os, 'getpgid', lambda pid: 100)
    monkeypatch.setattr(sup.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(sup.subprocess, 'Popen', lambda *args, **kw: proc)
    monkeypatch.setattr(sup.time, 'sleep', lambda *args: None)
    released = []

    def release(root, lease, folder, outcome):
        assert proc.poll() is not None, 'Cannot release a pod with an untracked live worker'
        released.append(outcome)

    monkeypatch.setattr(sup, 'release', release)
    return root / 'allocations/pod1/generation_supervisor', released


def test_receipt_write_failure_after_spawn_stops_and_reaps_worker_before_release(tmp_path, monkeypatch):
    proc = FakeProcess()
    folder, released = run_fixture(tmp_path, monkeypatch, proc)
    original = sup.atomic_json

    def fail_started(path, value):
        if Path(path).name == 'started.json': raise OSError('Synthetic receipt write failure')
        return original(path, value)

    monkeypatch.setattr(sup, 'atomic_json', fail_started)
    sup.run('plan.json')
    assert proc.terminated and proc.waited
    assert released == ['failed_preserved']
    assert sup.read(folder / 'failure.json')['type'] == 'OSError'


def test_escaped_worker_is_reaped_but_cannot_authorize_idle_release(tmp_path, monkeypatch):
    proc = FakeProcess()
    folder, released = run_fixture(tmp_path, monkeypatch, proc)
    monkeypatch.setattr(sup.os, 'getpgid', lambda pid: 999)
    sup.run('plan.json')
    assert proc.terminated and proc.waited and released == []
    assert sup.read(folder / 'release_withheld.json')['immutable_lease_guard_retains_authority']
    assert not sup.read(folder / 'workers_stopped.json')['process_group_ownership_verified']


def test_worker_already_exited_before_group_inspection_is_not_restarted(tmp_path, monkeypatch):
    proc = FakeProcess(exitcode=0)
    folder, released = run_fixture(tmp_path, monkeypatch, proc)

    def vanished(pid): raise ProcessLookupError('Synthetic fast successful exit')

    monkeypatch.setattr(sup.os, 'getpgid', vanished)
    sup.run('plan.json')
    assert released == ['ready_jobs_drained'] and not proc.terminated
    assert sup.read(folder / 'gpu_0_attempt_1/exited.json')['returncode'] == 0


@pytest.mark.parametrize('error', [ProcessLookupError, PermissionError])
def test_live_worker_with_unverifiable_group_cannot_authorize_idle_release(tmp_path, monkeypatch, error):
    proc = FakeProcess()
    folder, released = run_fixture(tmp_path, monkeypatch, proc)

    def vanished(pid): raise error('Synthetic process inspection failure')

    monkeypatch.setattr(sup.os, 'getpgid', vanished)
    sup.run('plan.json')
    assert proc.terminated and proc.waited and released == []
    assert (folder / 'release_withheld.json').exists()
