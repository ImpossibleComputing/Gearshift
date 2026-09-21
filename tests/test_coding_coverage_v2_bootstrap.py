"""Bootstrap contracts without apt, downloads, GPU allocation, or provider calls."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from gearshift.coding_coverage_v2_lease import atomic_json, sha256_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import coding_coverage_v2_bootstrap as bootstrap


def test_bootstrap_environment_drops_credentials_and_offline_supervisor_is_explicit(monkeypatch):
    monkeypatch.setenv('RUNPOD_API_KEY', 'never-inherit')
    monkeypatch.setenv('HF_TOKEN', 'never-inherit')
    monkeypatch.setenv('PIP_INDEX_URL', 'https://user:password@example.invalid/simple')
    monkeypatch.setenv('HTTPS_PROXY', 'https://user:password@example.invalid')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    online = bootstrap.child_environment()
    offline = bootstrap.child_environment(offline=True)
    for name in ['RUNPOD_API_KEY', 'HF_TOKEN', 'PIP_INDEX_URL', 'HTTPS_PROXY', 'CUDA_VISIBLE_DEVICES']:
        assert name not in online and name not in offline
    assert online['HF_HOME'] == '/workspace/hf'
    assert online['HF_HUB_OFFLINE'] == '0'
    assert offline['HF_HUB_OFFLINE'] == offline['TRANSFORMERS_OFFLINE'] == offline['HF_DATASETS_OFFLINE'] == '1'


def test_commands_preserve_exact_sandbox_and_pinned_runtime_with_system_torch():
    commands = bootstrap.command_plan('/workspace/GearshiftV2', 4, '/opt/conda/bin/python')
    byname = {name: command for name, command, timeout in commands}
    assert byname['venv'] == ['/opt/conda/bin/python', '-m', 'venv', '--system-site-packages', '.pilot-venv']
    assert byname['sandbox_probe'] == ['/usr/bin/python3', 'scripts/coding_sandbox_probe.py']
    assert byname['pinned_model_download'][-1] == 'scripts/coding_download_models.py'
    assert set(byname['pinned_packages'][4:]) == set(bootstrap.PACKAGES)
    assert not any(arg.startswith('torch==') for arg in byname['pinned_packages'])
    assert 'torch.cuda.device_count()==4' in byname['runtime_check'][-1]
    assert 'matplotlib==3.11.2' in byname['pinned_packages']
    assert 'import scripts.coding_coverage_v2_report as report' in byname['report_dependency_check'][-1]
    assert 'fig.savefig' in byname['report_dependency_check'][-1]
    assert commands.index(next(x for x in commands if x[0] == 'report_dependency_check')) < commands.index(next(x for x in commands if x[0] == 'pinned_model_download'))
    assert commands.index(next(x for x in commands if x[0] == 'runtime_check')) < commands.index(next(x for x in commands if x[0] == 'pinned_model_download'))


def test_real_local_volume_probe_checks_fsync_rename_and_other_process_lock(tmp_path):
    receipt = bootstrap.probe_volume(tmp_path / 'probe')
    assert receipt['atomic_rename_verified'] and receipt['file_fsync_verified'] and receipt['directory_fsync_verified']
    assert receipt['cross_process_flock_exclusion_verified'] and receipt['flock_release_verified']
    assert receipt['probe_bytes'] == 32768


def test_mount_check_rejects_container_overlay_and_accepts_workspace_mount(tmp_path):
    info = tmp_path / 'mountinfo'
    info.write_text('19 1 0:12 / / rw - overlay overlay rw\n')
    with pytest.raises(ValueError, match='temporary container'): bootstrap.mounted_volume('/workspace/GearshiftV2', 'volume1', info)
    info.write_text(info.read_text() + '21 19 0:44 / /workspace rw - nfs4 storage.example:/volume rw\n')
    result = bootstrap.mounted_volume('/workspace/GearshiftV2', 'volume1', info)
    assert result['separate_workspace_mount_verified'] and result['network_volume_id'] == 'volume1'
    assert 'storage.example' not in json.dumps(result)


def build_fixture(tmp_path, monkeypatch):
    now = [1000.]
    monkeypatch.setattr(bootstrap.time, 'time', lambda: now[0])
    atomic_json(tmp_path / 'declaration.json', {'frozen': True})
    atomic_json(tmp_path / 'public.json', {'input': 'unchanged'})
    lease = {'experiment_id': 'bootstrap_test', 'pod_id': 'pod1', 'network_volume_id': 'volume1',
             'allocation_epoch': 900., 'deadline_epoch': 3000., 'upper_hourly_usd': 20.,
             'baseline_usd': 336., 'baseline_gpu_hours': 59., 'total_cap_usd': 1000.,
             'total_cap_gpu_hours': 500., 'cleanup_reserve_usd': 40., 'gpu_count': 4,
             'allowed_result_root': str(tmp_path / 'results/new')}
    atomic_json(tmp_path / 'lease.json', lease)
    plan = {'experiment_id': lease['experiment_id'], 'network_volume_id': 'volume1', 'code_commit': 'a' * 40,
            'result_root': 'results/new', 'allocation_lease_path': 'lease.json',
            'allocation_lease_sha256': sha256_file(tmp_path / 'lease.json'),
            'declaration_path': 'declaration.json', 'declaration_sha256': sha256_file(tmp_path / 'declaration.json'),
            'expected_jobs': [{'job_id': 'job1'}],
            'input_manifest': {'public.json': {'sha256': sha256_file(tmp_path / 'public.json'), 'bytes': (tmp_path / 'public.json').stat().st_size}}}
    atomic_json(tmp_path / 'plan.json', plan)
    obj = bootstrap.Bootstrap(tmp_path, 'plan.json', 'lease.json')
    return obj, now


def test_setup_failure_preserves_receipt_and_never_launches_supervisor(tmp_path, monkeypatch):
    obj, _ = build_fixture(tmp_path, monkeypatch)
    events = []
    monkeypatch.setattr(obj, 'verify_guard', lambda: {'armed': True})
    monkeypatch.setattr(obj, 'existing_supervisor', lambda: None)
    monkeypatch.setattr(bootstrap, 'mounted_volume', lambda *a: {'mounted': True})
    monkeypatch.setattr(bootstrap, 'probe_volume', lambda *a: {'probe': True})
    def failure(*args): events.append('command'); raise RuntimeError('setup command failed')
    monkeypatch.setattr(obj, 'run_command', failure)
    monkeypatch.setattr(obj, 'launch_supervisor', lambda: events.append('forbidden_launch'))
    with pytest.raises(RuntimeError, match='setup command'): obj.run()
    assert events == ['command']
    receipt = json.loads((obj.attempt / 'failure.json').read_text())
    assert receipt['provider_lifecycle_action_taken'] is False
    assert json.loads((obj.control / 'latest.json').read_text())['state'] == 'failed'


def test_existing_supervisor_skips_install_download_and_new_launch(tmp_path, monkeypatch):
    obj, _ = build_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(obj, 'verify_guard', lambda: {'armed': True})
    monkeypatch.setattr(obj, 'existing_supervisor', lambda: {'pid': 44, 'pgid': 44})
    monkeypatch.setattr(obj, 'run_command', lambda *a: pytest.fail('Setup repeated during scientific execution'))
    monkeypatch.setattr(obj, 'launch_supervisor', lambda: pytest.fail('Duplicate supervisor'))
    assert obj.run()['state'] == 'already_running'


def test_guard_must_be_armed_for_exact_lease_before_setup(tmp_path, monkeypatch):
    obj, _ = build_fixture(tmp_path, monkeypatch)
    atomic_json(obj.root / 'lease_guard/status.json', {'state': 'armed', 'experiment_id': 'wrong',
        'pod_id': 'pod1', 'deadline_epoch': 3000., 'pid': os.getpid()})
    with pytest.raises(ValueError, match='not armed'): obj.verify_guard()
    atomic_json(obj.root / 'lease_guard/status.json', {'state': 'armed', 'experiment_id': 'bootstrap_test',
        'pod_id': 'pod1', 'deadline_epoch': 3000., 'pid': os.getpid()})
    assert obj.verify_guard()['guard_independent_of_bootstrap']


def test_command_timeout_is_bounded_by_remaining_allocation_and_environment_is_clean(tmp_path, monkeypatch):
    obj, now = build_fixture(tmp_path, monkeypatch); now[0] = 2800.
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs)); kwargs['stdout'].write(b'preserved command output\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(bootstrap.subprocess, 'run', fake_run)
    row = obj.run_command('test', ['python', 'public-script.py'], 1800)
    assert calls[0][1]['timeout'] == 80.
    assert 'RUNPOD_API_KEY' not in calls[0][1]['env']
    assert row['log_bytes'] > 0 and row['returncode'] == 0


def test_verified_setup_precedes_detached_dispatch(tmp_path, monkeypatch):
    obj, _ = build_fixture(tmp_path, monkeypatch); events = []
    monkeypatch.setattr(obj, 'verify_guard', lambda: {'armed': True})
    monkeypatch.setattr(obj, 'existing_supervisor', lambda: None)
    monkeypatch.setattr(bootstrap, 'mounted_volume', lambda *a: {'mounted': True})
    monkeypatch.setattr(bootstrap, 'probe_volume', lambda *a: {'probe': True})
    monkeypatch.setattr(obj, 'run_command', lambda label, *a: events.append(label))
    def verify():
        events.append('verified'); atomic_json(obj.attempt / 'setup_verification.json', {'passed': True}); return {'passed': True}
    monkeypatch.setattr(obj, 'verify_receipts', verify)
    monkeypatch.setattr(obj, 'publish_shared_setup', lambda: events.append('shared_ready'))
    monkeypatch.setattr(obj, 'launch_supervisor', lambda: (events.append('launched') or {'pid': 44}))
    result = obj.run()
    assert events[-3:] == ['verified', 'shared_ready', 'launched'] and result['state'] == 'dispatched'


def multipod_objects(tmp_path, monkeypatch):
    original, now = build_fixture(tmp_path, monkeypatch)
    base_plan, base_lease = original.plan, original.lease
    result = []
    for pod, role in [('primarypod', 'primary'), ('helperpod', 'evaluation_helper')]:
        lease = {**base_lease, 'pod_id': pod, 'control_relative': 'allocations/' + pod, 'gpu_count': 2}
        lease_rel = 'allocations/' + pod + '/lease.json'
        atomic_json(tmp_path / lease_rel, lease)
        plan = {**base_plan, 'allocation_lease_path': lease_rel,
                'allocation_lease_sha256': sha256_file(tmp_path / lease_rel),
                'execution_role': role, 'local_gpu_count': 2}
        plan_rel = 'dispatch_' + pod + '.json'; atomic_json(tmp_path / plan_rel, plan)
        result.append(bootstrap.Bootstrap(tmp_path, plan_rel, lease_rel))
    return (*result, now)


def publish_fake_ready(primary):
    atomic_json(primary.repo / '.pilot-venv/pyvenv.cfg', {'same_base_image': True})
    proof = {'sandbox_passed': True, 'workers_run_offline': True, 'runtime_packages': list(bootstrap.PACKAGES)}
    atomic_json(primary.attempt / 'setup_verification.json', proof)
    atomic_json(primary.attempt / 'sandbox_gate.json', {'passed': True})
    for name in ('source_weight_pins.json', 'receiver_weight_pins.json', 'weight_verification.json'):
        atomic_json(primary.attempt / 'download_receipts' / name, {'verified': True})
    return primary.publish_shared_setup()


def test_multipod_guard_and_registration_paths_are_allocation_local(tmp_path, monkeypatch):
    primary, helper, now = multipod_objects(tmp_path, monkeypatch)
    assert primary.control != helper.control
    assert primary.root == helper.root
    atomic_json(primary.allocation_control / 'lease_guard/STOP', {'reason': 'primary allocation only'})
    helper.guard()
    with pytest.raises(TimeoutError): primary.guard()
    atomic_json(helper.allocation_control / 'lease_guard/status.json', {'state': 'armed',
        'experiment_id': helper.plan['experiment_id'], 'pod_id': helper.lease['pod_id'],
        'deadline_epoch': helper.lease['deadline_epoch'], 'pid': os.getpid()})
    assert helper.verify_guard()['guard_pid'] == os.getpid()


def test_helper_hash_verifies_shared_ready_and_never_installs_downloads_or_probes_sandbox(tmp_path, monkeypatch):
    primary, helper, _ = multipod_objects(tmp_path, monkeypatch)
    publish_fake_ready(primary); commands = []
    monkeypatch.setattr(helper, 'verify_guard', lambda: {'armed': True})
    monkeypatch.setattr(helper, 'existing_supervisor', lambda: None)
    monkeypatch.setattr(bootstrap, 'mounted_volume', lambda *a: {'mounted': True})
    monkeypatch.setattr(bootstrap, 'probe_volume', lambda *a: {'probe': True})
    monkeypatch.setattr(helper, 'run_command', lambda label, command, timeout: commands.append((label, command)))
    monkeypatch.setattr(helper, 'launch_supervisor', lambda: {'pid': 55})
    monkeypatch.setattr(helper, 'verify_receipts', lambda: pytest.fail('Helper rewrites shared setup receipts'))
    result = helper.run()
    assert result['state'] == 'dispatched'
    assert [name for name, _ in commands] == ['runtime_check', 'report_dependency_check', 'package_inventory']
    assert 'torch.cuda.device_count()==2' in commands[0][1][-1]
    assert json.loads((helper.attempt / 'shared_setup_verified.json').read_text())['pip_download_sandbox_mutations_performed'] is False


def test_helper_rejects_tampered_ready_payload_and_referenced_evidence(tmp_path, monkeypatch):
    primary, helper, _ = multipod_objects(tmp_path, monkeypatch)
    receipt = publish_fake_ready(primary)
    changed = json.loads(json.dumps(receipt)); changed['payload']['code_commit'] = 'changed'
    with pytest.raises(ValueError, match='payload hash'): helper.verify_shared_setup(changed)
    atomic_json(primary.attempt / 'download_receipts/source_weight_pins.json', {'corrupted': True})
    with pytest.raises(ValueError, match='evidence hash'): helper.verify_shared_setup(receipt)


def test_primary_retry_never_reinstalls_after_shared_ready_is_published(tmp_path, monkeypatch):
    primary, _, _ = multipod_objects(tmp_path, monkeypatch)
    publish_fake_ready(primary)
    monkeypatch.setattr(primary, 'verify_guard', lambda: {'armed': True})
    monkeypatch.setattr(primary, 'existing_supervisor', lambda: None)
    monkeypatch.setattr(bootstrap, 'mounted_volume', lambda *a: {'mounted': True})
    monkeypatch.setattr(bootstrap, 'probe_volume', lambda *a: {'probe': True})
    monkeypatch.setattr(primary, 'run_command', lambda *a: pytest.fail('Shared environment modified during retry'))
    monkeypatch.setattr(primary, 'launch_supervisor', lambda: {'pid': 66})
    assert primary.run()['state'] == 'dispatched'


def test_helper_setup_commands_are_explicitly_offline(tmp_path, monkeypatch):
    _, helper, _ = multipod_objects(tmp_path, monkeypatch); environments = []
    def fake_run(command, **kwargs):
        environments.append(kwargs['env']); return SimpleNamespace(returncode=0)
    monkeypatch.setattr(bootstrap.subprocess, 'run', fake_run)
    helper.run_command('runtime_check', ['python', '-c', 'pass'], 100)
    assert environments[0]['HF_HUB_OFFLINE'] == environments[0]['TRANSFORMERS_OFFLINE'] == '1'
