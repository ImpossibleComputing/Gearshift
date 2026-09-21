"""Verify secondary launch ownership and recovery without GPUs or provider calls."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import coding_confirmation_secondary_regional_stage as s
from gearshift.coding_confirmation_lease import control_root


def launch_fixture(tmp_path, monkeypatch):
    lease = {'experiment_id': 'test', 'pod_id': 'newpod', 'allowed_result_root': str(tmp_path / 'results'),
             'control_relative': 'allocations/newpod'}
    control = control_root(lease); (control / 'lease_guard').mkdir(parents=True)
    identity = {'pid': 9876, 'pgid': 9876, 'start_ticks': 77, 'boot_id': 'boot'}
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    monkeypatch.setattr(s.regional, 'public_context', lambda *a, **kw: {})
    monkeypatch.setattr(s.regional, 'verify_allocation', lambda *a, **kw: (lease, {}))
    monkeypatch.setattr(s, 'preflight', lambda *a: ({}, lease, {'ready_task_count': 3}))
    monkeypatch.setattr(s, 'linux_process_identity', lambda *a: identity)
    monkeypatch.setattr(s, 'verify_launch', lambda *a: None)
    monkeypatch.setattr(s.common, 'all_device_environment', lambda *a: {'SAFE': '1'})
    monkeypatch.setattr(s.subprocess, 'check_output', lambda *a, **kw: '')
    calls = []
    def start(command, **kw):
        assert s.SUPERVISOR in command
        assert not (control / 'secondary_sharded_supervisor').exists()
        assert (control / 'secondary_launch_intent.json').exists()
        calls.append((command, kw)); (control / 'secondary_sharded_supervisor').mkdir()
        s.bind_json(control / 'lease_guard/supervisor_registration.json', {
            **identity, 'experiment_id': 'test', 'pod_id': 'newpod'})
        return SimpleNamespace(pid=9876, poll=lambda: None)
    monkeypatch.setattr(s.subprocess, 'Popen', start)
    return lease, control, identity, calls


def test_launch_is_detached_secondary_only_and_idempotent_while_gpu_busy(tmp_path, monkeypatch):
    lease, control, identity, calls = launch_fixture(tmp_path, monkeypatch)
    first = s.launch_remote('plan.json', 'a' * 64)
    assert first['guard_registered'] and first['process_identity'] == identity
    assert calls[0][1]['start_new_session'] and calls[0][1]['env'] == {'SAFE': '1'}
    monkeypatch.setattr(s.subprocess, 'check_output', lambda *a, **kw: pytest.fail('Running launch must not need idle GPU'))
    second = s.launch_remote('plan.json', 'a' * 64)
    assert second['already_running'] and len(calls) == 1


@pytest.mark.parametrize('prior', ['intent', 'registration', 'owned_folder'])
def test_unknown_launch_or_existing_primary_ownership_cannot_be_replaced(tmp_path, monkeypatch, prior):
    _, control, _, calls = launch_fixture(tmp_path, monkeypatch)
    if prior == 'intent': s.bind_json(control / 'secondary_launch_intent.json', {'unknown': True})
    elif prior == 'registration': s.bind_json(control / 'lease_guard/supervisor_registration.json', {'primary': True})
    else: (control / 'secondary_sharded_supervisor').mkdir()
    with pytest.raises(ValueError, match='ownership'): s.launch_remote('plan.json', 'a' * 64)
    assert not calls


def test_busy_gpu_or_invalid_checkpoint_prevents_launch_intent(tmp_path, monkeypatch):
    _, control, _, calls = launch_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(s.subprocess, 'check_output', lambda *a, **kw: '1234\n')
    with pytest.raises(ValueError, match='GPU compute'): s.launch_remote('plan.json', 'a' * 64)
    assert not calls and not (control / 'secondary_launch_intent.json').exists()
    def fail(*a): raise ValueError('Final checkpoint hash differs')
    monkeypatch.setattr(s, 'preflight', fail)
    with pytest.raises(ValueError, match='checkpoint'): s.launch_remote('plan.json', 'a' * 64)
    assert not calls and not (control / 'secondary_launch_intent.json').exists()


def test_failed_spawn_is_not_blindly_retried(tmp_path, monkeypatch):
    _, control, _, calls = launch_fixture(tmp_path, monkeypatch)
    def fail(*a, **kw): raise OSError('Connection interrupted')
    monkeypatch.setattr(s.subprocess, 'Popen', fail)
    with pytest.raises(OSError): s.launch_remote('plan.json', 'a' * 64)
    assert (control / 'secondary_launch_intent.json').exists()
    with pytest.raises(ValueError, match='ambiguous'): s.launch_remote('plan.json', 'a' * 64)


@pytest.mark.parametrize('mutation', ['reused_pid', 'primary_command', 'other_plan', 'registration'])
def test_exact_process_and_guard_registration_are_required(tmp_path, monkeypatch, mutation):
    lease = {'experiment_id': 'test', 'pod_id': 'pod', 'allowed_result_root': str(tmp_path / 'results'),
             'control_relative': 'allocations/pod'}
    identity = {'pid': 1234, 'pgid': 1234, 'start_ticks': 77, 'boot_id': 'boot'}
    current = dict(identity); registration = {**identity, 'experiment_id': 'test', 'pod_id': 'pod'}
    command = [s.SUPERVISOR, '--plan', 'plan.json', '--plan-sha256', 'a' * 64]
    if mutation == 'reused_pid': current['start_ticks'] = 99
    elif mutation == 'primary_command': command[0] = 'scripts/coding_confirmation_sharded_supervisor.py'
    elif mutation == 'other_plan': command[2] = 'other.json'
    else: registration['pod_id'] = 'other'
    proc = tmp_path / 'proc'; (proc / '1234').mkdir(parents=True)
    (proc / '1234/cmdline').write_bytes(b'\0'.join(x.encode() for x in command))
    s.bind_json(control_root(lease) / 'lease_guard/supervisor_registration.json', registration)
    monkeypatch.setattr(s, 'linux_process_identity', lambda *a: current)
    with pytest.raises(ValueError): s.verify_launch(lease, 'plan.json', 'a' * 64, identity, proc)
