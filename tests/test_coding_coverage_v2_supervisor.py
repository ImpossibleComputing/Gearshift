import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import coding_coverage_v2_supervisor as supervisor
from gearshift.coding_coverage_v2_lease import atomic_json, control_root, sha256_file, verify_release_receipt


def fixture(tmp_path):
    root = tmp_path / 'results/v2'
    root.mkdir(parents=True)
    declaration = tmp_path / 'declaration.json'
    atomic_json(declaration, {'frozen': True})
    lease = {'experiment_id': 'v2_test', 'pod_id': 'owned-pod',
             'allocation_epoch': 1000, 'deadline_epoch': 73000,
             'upper_hourly_usd': 22.20, 'gpu_count': 4, 'baseline_usd': 335.93,
             'baseline_gpu_hours': 58.94, 'total_cap_usd': 1000,
             'total_cap_gpu_hours': 500, 'cleanup_reserve_usd': 40,
             'allowed_result_root': str(root)}
    plan = {'experiment_id': 'v2_test', 'result_root': 'results/v2',
            'declaration_path': 'declaration.json', 'declaration_sha256': sha256_file(declaration),
            'allocation_lease_path': 'lease.json',
            'expected_jobs': [{'job_id': 'task_b', 'arm': 'START', 'step': 0},
                              {'job_id': 'task_a', 'arm': 'FIXED', 'step': 1024}]}
    atomic_json(tmp_path / 'plan.json', plan)
    atomic_json(tmp_path / 'lease.json', lease)
    return root, lease, plan


def save_jobs(root, plan):
    for expected in plan['expected_jobs']:
        job = {**expected, 'experiment_id': plan['experiment_id'], 'checkpoint': {'sha256': 'a' * 64}}
        directory = root / 'evaluation/jobs' / job['job_id']
        atomic_json(directory / 'job.json', job)
        atomic_json(directory / 'generation_complete.json', {
            'job_identity_sha256': supervisor.digest(job), 'hidden_tests_loaded': False})


def multipod_fixture(tmp_path, role='evaluation_helper', count=2):
    root, lease, plan = fixture(tmp_path)
    lease.update(gpu_count=count, upper_hourly_usd=count * 5.55,
                 other_reserved_usd=(4-count)*5.55*20,
                 other_reserved_gpu_hours=(4-count)*20,
                 control_relative='allocations/owned-pod')
    plan.update(execution_role=role, local_gpu_count=count)
    atomic_json(tmp_path / 'plan.json', plan)
    atomic_json(tmp_path / 'lease.json', lease)
    return root, lease, plan


def save_checkpoints(root):
    for arm in ('FIXED', 'ROTATING'):
        for step in (0, 128, 256, 512, 768, 1024):
            folder = root / 'arms' / arm / 'checkpoints' / f'step_{step:04d}'
            folder.mkdir(parents=True, exist_ok=True)
            files = {}
            for name in ('full.pt', 'mapper.pt'):
                path = folder / name
                path.write_bytes((arm + str(step) + name).encode())
                files[name] = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
            atomic_json(folder / 'manifest.json', {'arm': arm, 'step': step,
                'complete_resumable': True, 'verified_roundtrip': True, 'files': files})


@pytest.mark.parametrize(('role', 'count'), [('primary', 2), ('evaluation_helper', 1), ('evaluation_helper', 2)])
def test_multipod_plan_isolated_control_and_worker_identity(tmp_path, role, count):
    root, lease, plan = multipod_fixture(tmp_path, role, count)
    instance = supervisor.Supervisor(tmp_path, 'plan.json', 'lease.json')
    assert instance.control == root / 'allocations/owned-pod/supervisor'
    assert instance.worker_id('eval_gpu0') == 'owned-pod_eval_gpu0'
    assert supervisor.verify_plan(tmp_path, plan, lease) == root
    wrong = dict(plan, local_gpu_count=4)
    with pytest.raises(ValueError, match='GPU count'):
        supervisor.verify_plan(tmp_path, wrong, lease)


def test_helper_release_ignores_concurrent_primary_scoring_and_control(tmp_path):
    root, lease, plan = multipod_fixture(tmp_path)
    save_jobs(root, plan)
    save_checkpoints(root)
    atomic_json(root / 'evaluation/jobs/task_a/score.json', {'score': 1})
    atomic_json(root / 'workers/primary_train/status.json', {'mutable': True})
    atomic_json(root / 'bootstrap/status.json', {'mutable': True})
    atomic_json(root / 'allocations/primary/lease_guard/status.json', {'state': 'healthy'})
    receipt = supervisor.prepare_gpu_release(root, lease, outcome='complete', helper=True)
    assert receipt['manifest_path'] == 'allocations/owned-pod/release_manifest.json'
    assert verify_release_receipt(lease)
    manifest = supervisor.read(control_root(lease) / 'release_manifest.json')
    names = {row['path'] for row in manifest['files']}
    assert 'arms/FIXED/checkpoints/step_1024/full.pt' in names
    assert 'evaluation/jobs/task_a/job.json' in names
    assert not any('score.json' in name or name.startswith(('workers/', 'bootstrap/')) for name in names)
    atomic_json(root / 'evaluation/jobs/task_a/score.json', {'score': 0})
    atomic_json(root / 'workers/primary_train/status.json', {'mutable': 'changed'})
    atomic_json(root / 'allocations/primary/lease_guard/status.json', {'state': 'scoring'})
    atomic_json(root / 'evaluation/scored_answer_manifest.json', {'scored': True})
    assert verify_release_receipt(lease)
    assert not (root / 'gpu_release_verified.json').exists()
    assert not (root / 'allocations/primary/gpu_release_verified.json').exists()


def test_helper_success_cannot_release_missing_full_checkpoint(tmp_path):
    root, lease, plan = multipod_fixture(tmp_path)
    save_jobs(root, plan)
    save_checkpoints(root)
    (root / 'arms/ROTATING/checkpoints/step_1024/full.pt').write_bytes(b'broken')
    with pytest.raises(ValueError, match='checkpoint hash'):
        supervisor.prepare_gpu_release(root, lease, outcome='complete', helper=True)
    assert not (control_root(lease) / 'gpu_release_verified.json').exists()


def test_scoped_primary_manifest_ignores_other_running_allocations(tmp_path):
    root, lease, _ = multipod_fixture(tmp_path, role='primary')
    atomic_json(root / 'execution_complete.json', {'completed': True})
    atomic_json(root / 'allocations/helper/lease_guard/status.json', {'state': 'backing_up'})
    atomic_json(root / 'workers/helper/status.json', {'running': True})
    atomic_json(root / 'bootstrap/status.json', {'running': True})
    supervisor.prepare_gpu_release(root, lease, outcome='complete')
    atomic_json(root / 'allocations/helper/lease_guard/status.json', {'state': 'released'})
    atomic_json(root / 'workers/helper/status.json', {'running': False})
    assert verify_release_receipt(lease)


def test_final_plan_requires_whole_frozen_population_and_preserves_order(tmp_path):
    root, lease, plan = fixture(tmp_path)
    assert supervisor.verify_plan(tmp_path, plan, lease) == root
    assert not supervisor.all_generation_complete(root, plan)
    with pytest.raises(ValueError, match='population'):
        supervisor.assemble_evaluation_plan(root, plan)
    save_jobs(root, plan)
    result = supervisor.assemble_evaluation_plan(root, plan)
    assert [job['job_id'] for job in result['jobs']] == ['task_b', 'task_a']
    assert result['declaration_path'] == plan['declaration_path']
    assert result['declaration_sha256'] == plan['declaration_sha256']
    assert result['training_roots'] == {'FIXED': 'results/v2/arms/FIXED', 'ROTATING': 'results/v2/arms/ROTATING'}
    assert supervisor.all_generation_complete(root, plan)
    assert supervisor.assemble_evaluation_plan(root, plan) == result
    atomic_json(root / 'evaluation/jobs/unplanned/job.json', {'job_id': 'unplanned'})
    with pytest.raises(ValueError, match='population'):
        supervisor.assemble_evaluation_plan(root, plan)


def test_completion_identity_or_phase_change_blocks_scoring_plan(tmp_path):
    root, _, plan = fixture(tmp_path)
    save_jobs(root, plan)
    path = root / 'evaluation/jobs/task_a/generation_complete.json'
    complete = supervisor.read(path)
    complete['hidden_tests_loaded'] = True
    atomic_json(path, complete)
    with pytest.raises(ValueError, match='completion'):
        supervisor.assemble_evaluation_plan(root, plan)


def test_worker_environment_is_offline_and_has_no_credentials(monkeypatch):
    for name in ['RUNPOD_API_KEY', 'OPENAI_API_KEY', 'HF_TOKEN', 'AWS_SECRET_ACCESS_KEY', 'SOME_PASSWORD']:
        monkeypatch.setenv(name, 'secret')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    env = supervisor.child_environment(2)
    assert not any(value == 'secret' for value in env.values())
    assert env['CUDA_VISIBLE_DEVICES'] == '2'
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'
    assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert supervisor.child_environment(None)['CUDA_VISIBLE_DEVICES'] == ''


def test_gpu_release_keeps_heavy_files_and_verifies_two_compact_copies(tmp_path):
    root, lease, _ = fixture(tmp_path)
    atomic_json(root / 'execution_complete.json', {'completed': True})
    (root / 'full_checkpoint.pt').write_bytes(b'heavy optimizer and RNG state')
    atomic_json(root / 'lease_guard/status.json', {'mutable': 1})
    atomic_json(root / 'supervisor/status.json', {'mutable': 1})
    receipt = supervisor.prepare_gpu_release(root, lease, outcome='complete')
    assert verify_release_receipt(lease)
    assert receipt['network_volume_deletion_authorized'] is False
    manifest = supervisor.read(root / 'release_manifest.json')
    names = {row['path'] for row in manifest['files']}
    assert 'full_checkpoint.pt' in names
    assert 'release_backups/compact_a.tar.gz' in names
    assert 'release_backups/compact_b.tar.gz' in names
    assert 'lease_guard/status.json' not in names
    assert 'supervisor/status.json' not in names
    import tarfile
    with tarfile.open(root / 'release_backups/compact_a.tar.gz') as archive:
        assert 'full_checkpoint.pt' not in archive.getnames()
        assert 'execution_complete.json' in archive.getnames()
    atomic_json(root / 'lease_guard/status.json', {'mutable': 2})
    atomic_json(root / 'supervisor/status.json', {'mutable': 2})
    assert verify_release_receipt(lease)
    (root / 'full_checkpoint.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='changed'):
        verify_release_receipt(lease)


def test_failed_attempt_logs_are_preserved_in_release(tmp_path):
    root, lease, _ = fixture(tmp_path)
    path = root / 'supervisor/attempts/train_FIXED/attempt_01/worker.log'
    path.parent.mkdir(parents=True)
    path.write_text('engineering failure')
    atomic_json(root / 'supervisor/failure_123.json', {'exception': 'RuntimeError'})
    receipt = supervisor.prepare_gpu_release(root, lease, outcome='failed_preserved')
    assert receipt['outcome'] == 'failed_preserved'
    assert verify_release_receipt(lease)
    manifest = supervisor.read(root / 'release_manifest.json')
    assert str(path.relative_to(root)) in {row['path'] for row in manifest['files']}


def test_restart_retains_worker_id_role_gpu_and_never_exceeds_three_attempts(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path)
    instance = supervisor.Supervisor(tmp_path, 'plan.json', 'lease.json')

    class Finished:
        def poll(self):
            return 1

    folder = root / 'supervisor/attempts/eval_gpu2/attempt_01'
    folder.mkdir(parents=True)
    instance.children['eval_gpu2'] = {'process': Finished(), 'folder': folder,
        'metadata': {'role': 'evaluate', 'gpu': 2, 'options': {}, 'attempt': 1}}
    calls = []
    monkeypatch.setattr(instance, 'start', lambda *args, **kwargs: calls.append((args, kwargs)))
    instance.poll()
    assert calls == [(('eval_gpu2', 'evaluate', 2), {})]
    instance.children['eval_gpu2'] = {'process': Finished(), 'folder': folder,
        'metadata': {'role': 'evaluate', 'gpu': 2, 'options': {}, 'attempt': 3}}
    with pytest.raises(RuntimeError, match='attempt 3'):
        instance.poll()
    assert len(calls) == 1


def test_scientific_rejection_does_not_restart(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path)
    instance = supervisor.Supervisor(tmp_path, 'plan.json', 'lease.json')
    folder = root / 'supervisor/attempts/train_FIXED/attempt_01'
    folder.mkdir(parents=True)

    class FailedControl:
        def poll(self):
            return 65

    instance.children['train_FIXED'] = {'process': FailedControl(), 'folder': folder,
        'metadata': {'role': 'train', 'gpu': 0, 'options': {'arm': 'FIXED'}, 'attempt': 1}}
    monkeypatch.setattr(instance, 'start', lambda *args, **kwargs: pytest.fail('scientific rejection retried'))
    with pytest.raises(RuntimeError, match='returncode 65'):
        instance.poll()


def test_full_lifecycle_runs_parallel_gpu_jobs_then_cpu_scoring_and_durable_release(tmp_path, monkeypatch):
    root, lease, plan = fixture(tmp_path)
    instance = supervisor.Supervisor(tmp_path, 'plan.json', 'lease.json')
    events = []
    next_pid = [200]
    monkeypatch.setattr(supervisor.os, 'getpid', lambda: 100)
    monkeypatch.setattr(supervisor.os, 'getpgrp', lambda: 100)
    monkeypatch.setattr(supervisor.os, 'getpgid', lambda pid: 100)
    monkeypatch.setattr(supervisor, 'linux_process_identity', lambda pid: {
        'pid': pid, 'pgid': pid, 'start_ticks': 10, 'boot_id': 'test-boot'})
    monkeypatch.setattr(supervisor.time, 'time', lambda: 1010)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: None)
    monkeypatch.setenv('RUNPOD_API_KEY', 'credential-that-must-not-leak')

    class Child:
        def __init__(self, command, **kwargs):
            self.role = command[command.index('--role') + 1]
            self.worker = command[command.index('--worker-id') + 1]
            self.pid = next_pid[0]
            next_pid[0] += 1
            self.finished = False
            assert kwargs['start_new_session'] is False
            assert 'RUNPOD_API_KEY' not in kwargs['env']
            if self.role in {'score', 'finalize'}:
                assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
                assert supervisor.all_generation_complete(root, plan)
                assert not any(item.startswith('running_gpu:') for item in events)
            else:
                events.append('running_gpu:' + self.worker)
            events.append('start:' + self.worker)

        def poll(self):
            if self.finished:
                return 0
            # All initial workers must have been started before waiting on any.
            assert len([item for item in events if item.startswith('start:')]) >= 4
            if self.role == 'train':
                arm = self.worker.removeprefix('train_')
                atomic_json(root / 'arms' / arm / 'training_complete.json', {
                    'full_target_completed': True, 'completed_updates': 1024})
            elif self.role == 'evaluate':
                save_jobs(root, plan)
            elif self.role == 'finalize':
                atomic_json(root / 'evaluation/scored_answer_manifest.json', {
                    'experiment_id': plan['experiment_id'], 'hidden_tests_loaded_after_all_generation': True})
            if self.role in {'train', 'evaluate'}:
                events.remove('running_gpu:' + self.worker)
            self.finished = True
            events.append('exit:' + self.worker)
            return 0

    monkeypatch.setattr(supervisor.subprocess, 'Popen', Child)
    receipt = instance.run()
    assert receipt['outcome'] == 'complete'
    assert verify_release_receipt(lease)
    assert {item for item in events if item.startswith('start:')} == {
        'start:train_FIXED', 'start:train_ROTATING', 'start:eval_gpu2', 'start:eval_gpu3',
        'start:score_cpu0', 'start:score_cpu1', 'start:score_cpu2', 'start:score_cpu3', 'start:finalize_cpu'}
    assert supervisor.read(root / 'execution_complete.json')['full_1024_1024_training_completed'] is True


@pytest.mark.parametrize(('role', 'count'), [('primary', 2), ('evaluation_helper', 1), ('evaluation_helper', 2)])
def test_multi_pod_lifecycle_uses_only_local_gpus_and_own_release(tmp_path, monkeypatch, role, count):
    root, lease, plan = multipod_fixture(tmp_path, role, count)
    if role == 'evaluation_helper':
        save_checkpoints(root)
    instance = supervisor.Supervisor(tmp_path, 'plan.json', 'lease.json')
    started, finished = [], []
    monkeypatch.setattr(supervisor.os, 'getpid', lambda: 100)
    monkeypatch.setattr(supervisor.os, 'getpgrp', lambda: 100)
    monkeypatch.setattr(supervisor.os, 'getpgid', lambda _: 100)
    monkeypatch.setattr(supervisor, 'linux_process_identity', lambda pid: {
        'pid': pid, 'pgid': pid, 'start_ticks': 10, 'boot_id': 'test-boot'})
    monkeypatch.setattr(supervisor.time, 'time', lambda: 1010)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: None)

    class Child:
        def __init__(self, command, **kwargs):
            self.role = command[command.index('--role') + 1]
            self.worker = command[command.index('--worker-id') + 1]
            self.arm = command[command.index('--arm') + 1] if '--arm' in command else None
            self.pid = 200 + len(started)
            self.done = False
            device = kwargs['env']['CUDA_VISIBLE_DEVICES']
            assert self.worker.startswith(lease['pod_id'] + '_')
            if self.role in {'score', 'finalize'}:
                assert role == 'primary'
                assert device == ''
                assert supervisor.all_generation_complete(root, plan)
            else:
                assert 0 <= int(device) < count
            if role == 'evaluation_helper':
                assert self.role == 'evaluate'
            started.append((self.worker, self.role, device))

        def poll(self):
            if self.done:
                return 0
            assert len(started) >= count
            if self.role == 'train':
                atomic_json(root / 'arms' / self.arm / 'training_complete.json', {
                    'full_target_completed': True, 'completed_updates': 1024})
            elif self.role == 'evaluate':
                save_jobs(root, plan)
            elif self.role == 'finalize':
                atomic_json(root / 'evaluation/scored_answer_manifest.json', {
                    'experiment_id': plan['experiment_id'], 'hidden_tests_loaded_after_all_generation': True})
            self.done = True
            finished.append(self.worker)
            return 0

    monkeypatch.setattr(supervisor.subprocess, 'Popen', Child)
    receipt = instance.run()
    assert receipt['outcome'] == 'complete'
    assert verify_release_receipt(lease)
    assert not (root / 'gpu_release_verified.json').exists()
    assert not (root / 'lease_guard/supervisor_registration.json').exists()
    assert (control_root(lease) / 'lease_guard/supervisor_registration.json').is_file()
    if role == 'evaluation_helper':
        assert len(started) == count
        assert not (root / 'evaluation/evaluation_plan.json').exists()
        assert not (root / 'execution_complete.json').exists()
    else:
        assert [row[1] for row in started[:2]] == ['train', 'train']
        assert sum(row[1] == 'evaluate' for row in started) == 2
        assert sum(row[1] == 'score' for row in started) == 4
        assert supervisor.read(root / 'execution_complete.json')['full_1024_1024_training_completed'] is True
