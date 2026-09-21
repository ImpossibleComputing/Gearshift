"""Operational supervisor tests use fake processes and compact public bytes."""
import importlib.util
from pathlib import Path

import pytest

from gearshift.coding_control import sha, write
from scripts import coding_confirmation_sharded_supervisor as s

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sharded_supervisor_fixture', ROOT / 'tests/test_coding_confirmation_generation_supervisor.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def setup(tmp_path, monkeypatch, processes):
    root = tmp_path / 'results'; lease = {'experiment_id': 'test', 'pod_id': 'newpod',
        'deadline_epoch': s.time.time() + 1000, 'gpu_count': 1,
        'allowed_result_root': str(root), 'control_relative': 'allocations/newpod'}
    plan = {'execution_mode': 'generation', 'experiment_id': 'test', 'result_root': 'results', 'lease_path': 'lease.json',
            'lease_sha256': 'test', 'preferences': ['source']}
    write(tmp_path / 'plan.json', plan)
    monkeypatch.setattr(s, 'ROOT', tmp_path); monkeypatch.setattr(s, 'load_lease', lambda *a: lease)
    monkeypatch.setattr(s, 'validate_guard', lambda *a: None); monkeypatch.setattr(s.sharded, 'public_context', lambda *a: None)
    monkeypatch.setattr(s.sharded, 'validate_generation_store', lambda *a, **kw: None)
    monkeypatch.setattr(s, 'linux_process_identity', lambda pid: {'pid': pid})
    monkeypatch.setattr(s.os, 'getpid', lambda: 100); monkeypatch.setattr(s.os, 'getpgrp', lambda: 100)
    monkeypatch.setattr(s.os, 'getpgid', lambda pid: 100); monkeypatch.setattr(s.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(s.time, 'sleep', lambda *a: None)
    commands = []; iterator = iter(processes)
    def popen(command, **kw): commands.append((command, kw)); return next(iterator)
    monkeypatch.setattr(s.subprocess, 'Popen', popen)
    released = []
    def release(root, lease, folder, outcome, plan_path):
        assert all(p.poll() is not None for p in processes); released.append(outcome)
    monkeypatch.setattr(s, 'release', release)
    return root / 'allocations/newpod/sharded_generation_supervisor', commands, released, sha(tmp_path / 'plan.json')


def test_launch_pins_plan_and_keeps_original_worker_namespace_for_safe_backup(tmp_path, monkeypatch):
    folder, commands, released, expected = setup(tmp_path, monkeypatch, [f.FakeProcess(exitcode=0)])
    s.run('plan.json', expected); command, kw = commands[0]
    assert command[1:3] == ['scripts/coding_confirmation_sharded_generate.py', 'run']
    assert command[command.index('--plan-sha256') + 1] == expected
    assert command[command.index('--worker-id') + 1] == 'newpod_generation_0'
    assert kw['start_new_session'] is False and kw['env']['HF_HUB_OFFLINE'] == '1'
    assert released == ['ready_sharded_jobs_drained']


def test_changed_dispatch_rejected_before_worker_launch(tmp_path, monkeypatch):
    folder, commands, released, expected = setup(tmp_path, monkeypatch, [])
    with pytest.raises(ValueError, match='dispatch plan'): s.run('plan.json', '0' * 64)
    assert commands == [] and released == []


def test_restart_allowance_and_scientific_failure_are_bounded(tmp_path, monkeypatch):
    folder, commands, released, expected = setup(tmp_path, monkeypatch, [f.FakeProcess(exitcode=1) for _ in range(3)])
    s.run('plan.json', expected)
    assert len(commands) == 3 and released == ['failed_preserved']
    assert len(list(folder.glob('gpu_0_attempt_*/exited.json'))) == 3


def test_scientific_failure_cannot_reroll(tmp_path, monkeypatch):
    folder, commands, released, expected = setup(tmp_path, monkeypatch, [f.FakeProcess(exitcode=65)])
    s.run('plan.json', expected)
    assert len(commands) == 1 and released == ['failed_preserved']


def test_receipt_failure_reaps_child_and_unverified_group_withholds_release(tmp_path, monkeypatch):
    proc = f.FakeProcess(); folder, commands, released, expected = setup(tmp_path, monkeypatch, [proc])
    monkeypatch.setattr(s.os, 'getpgid', lambda pid: 999); s.run('plan.json', expected)
    assert proc.terminated and proc.waited and released == []
    assert (folder / 'release_withheld.json').exists()


def test_release_preserves_partition_and_source_proofs_before_original_backup(tmp_path, monkeypatch):
    root = tmp_path / 'results'; folder = root / 'allocations/pod/sharded_generation_supervisor'
    write(tmp_path / 'plan.json', {'partition_path': 'partition.json', 'adapter_source_manifest_path': 'source.json',
        'lease_path': 'lease.json', 'provider_mount_receipt_path': 'mount.json'})
    write(tmp_path / 'partition.json', {'quiescence_path': 'quiescence.json', 'original_generation_allocations': [{'lease_path': 'old_lease.json'}]})
    write(tmp_path / 'old_lease.json', {'original': True})
    write(tmp_path / 'source.json', {'frozen': True})
    write(tmp_path / 'lease.json', {'immutable': True}); write(tmp_path / 'mount.json', {'provider_inspection_path': 'observation.json'})
    write(tmp_path / 'observation.json', {'source': 'runpod_provider_inspection'})
    write(tmp_path / 'quiescence.json', {'allocation_receipts': []})
    write(root / 'sharding/shards/origin.json', {'authentic_task_shard': True})
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    called = []
    def release(root, lease, folder, outcome):
        proof = folder / 'operational_provenance'
        for relative in ['plan.json', 'partition.json', 'source.json', 'quiescence.json', 'lease.json', 'mount.json', 'observation.json', 'old_lease.json', 'results/sharding/shards/origin.json']:
            assert (proof / relative).read_bytes() == (tmp_path / relative).read_bytes()
        called.append(True)
    monkeypatch.setattr(s, 'primary_release', release)
    s.release(root, {}, folder, 'complete', tmp_path / 'plan.json')
    assert called == [True]


def test_supervisor_rejects_merge_dispatch_before_any_child(tmp_path, monkeypatch):
    folder, commands, released, expected = setup(tmp_path, monkeypatch, [])
    plan = s.read(tmp_path / 'plan.json'); plan['execution_mode'] = 'merge'; write(tmp_path / 'plan.json', plan)
    with pytest.raises(ValueError, match='generation dispatch'): s.run('plan.json', sha(tmp_path / 'plan.json'))
    assert commands == [] and released == []


def test_supervisor_rejects_unverified_current_mount_before_any_child(tmp_path, monkeypatch):
    folder, commands, released, expected = setup(tmp_path, monkeypatch, [])
    def invalid(*a, **kw): raise ValueError('Wrong physical result store')
    monkeypatch.setattr(s.sharded, 'validate_generation_store', invalid)
    s.run('plan.json', expected)
    assert commands == [] and released == ['failed_preserved']
