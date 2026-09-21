import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import coding_coverage_v2_optional_helpers as helpers
from gearshift.coding_coverage_v2_lease import atomic_json, sha256_file, verify_lease


def controller(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'; repo.mkdir()
    key = tmp_path / 'key'; key.write_text('secret-private-token'); key.chmod(0o600)
    primary = {'experiment_id': 'coverage_v2_test', 'pod_id': 'primary-pod',
        'allocation_epoch': 1000, 'deadline_epoch': 73000, 'gpu_count': 2,
        'upper_hourly_usd': 11.10, 'other_reserved_usd': 222,
        'other_reserved_gpu_hours': 40, 'baseline_usd': 337,
        'baseline_gpu_hours': 58.94, 'total_cap_usd': 1000,
        'total_cap_gpu_hours': 500, 'cleanup_reserve_usd': 40,
        'network_volume_id': 'owned-volume', 'control_relative': 'allocations/primary-pod',
        'allowed_result_root': str(repo / 'results/v2')}
    plan = {'experiment_id': primary['experiment_id'], 'execution_role': 'primary',
        'local_gpu_count': 2, 'result_root': 'results/v2', 'allocation_lease_path': 'evidence/lease.json',
        'allocation_lease_sha256': 'c'*64,
        'expected_jobs': [{'job_id': 'pending'}], 'code_commit': 'frozen',
        'declaration_path': 'configs/declaration.json', 'declaration_sha256': 'a'*64,
        'input_manifest': {'unchanged': {'sha256': 'b'*64}}}
    atomic_json(repo / 'evidence/lease.json', primary)
    plan['allocation_lease_sha256'] = sha256_file(repo / 'evidence/lease.json')
    atomic_json(repo / 'evidence/plan.json', plan)
    args = SimpleNamespace(plan='evidence/plan.json', lease='evidence/lease.json', key_file=str(key),
        ssh_key=str(tmp_path / 'private-ssh-key'), cli='fake-cli', provider_config='/private/config', region='AP-JP-1')
    monkeypatch.setattr(helpers.time, 'time', lambda: 2000)
    return helpers.Controller(args, repo)


def attempt(c, number, *, count=1, pod=True):
    folder = c.directory / f'attempt_{number:04d}'; folder.mkdir()
    intent = {'name': c.prefix+f'{number:04d}', 'gpu_count': count, 'epoch': 2000}
    atomic_json(folder / 'intent.json', intent)
    row = {'id': f'helper-{number}', 'name': intent['name'], 'gpuCount': count,
           'networkVolumeId': 'owned-volume', 'publicIp': '192.0.2.1', 'portMappings': {'22': 10022},
           'costPerHr': 4.59*count}
    if pod: atomic_json(folder / 'pod.json', row)
    return folder, row


def test_helper_deadline_never_extends_and_reservations_cover_full_fleet(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    for count in (1, 2):
        value = helpers.helper_lease(c.primary, 'new-helper', count, 2000)
        assert value['deadline_epoch'] == c.primary['deadline_epoch']
        assert value['control_relative'] == 'allocations/new-helper'
        assert value['other_reserved_usd'] == pytest.approx(5.55*(4-count)*20)
        assert verify_lease(value)['reserved_total_usd'] < 822
    with pytest.raises(ValueError):
        helpers.helper_lease(c.primary, 'new-helper', 3, 2000)
    with pytest.raises(ValueError):
        helpers.helper_lease(c.primary, 'new-helper', 1, 72900)


def test_optional_controller_refuses_changed_primary_immutable_lease(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    value = helpers.read(c.lease_path); value['deadline_epoch'] += 60
    atomic_json(c.lease_path, value)
    with pytest.raises(ValueError, match='hash mismatch'):
        helpers.Controller(c.args, c.repo)


def test_helper_plan_changes_only_orchestration_fields(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    result = helpers.helper_plan(c.plan, 'helper/lease.json', 1, 'd'*64)
    changed = {key for key in c.plan if c.plan[key] != result[key]}
    assert changed == {'allocation_lease_path', 'allocation_lease_sha256', 'execution_role', 'local_gpu_count'}
    assert result['allocation_lease_sha256'] == 'd'*64
    result['expected_jobs'][0]['job_id'] = 'changed-copy'
    assert c.plan['expected_jobs'][0]['job_id'] == 'pending'


def test_helper_plan_loads_through_actual_frozen_bootstrap_constructor(tmp_path, monkeypatch):
    from coding_coverage_v2_bootstrap import Bootstrap
    c = controller(tmp_path, monkeypatch)
    declaration = c.repo / 'configs/declaration.json'
    atomic_json(declaration, {'frozen': True})
    c.plan['declaration_sha256'] = sha256_file(declaration)
    lease_path = c.directory / 'helper_lease.json'
    lease = helpers.helper_lease(c.primary, 'real-helper', 1, 2000)
    atomic_json(lease_path, lease)
    relative_lease = str(lease_path.relative_to(c.repo))
    plan = helpers.helper_plan(c.plan, relative_lease, 1, sha256_file(lease_path))
    plan_path = c.directory / 'helper_plan.json'; atomic_json(plan_path, plan)
    bootstrap = Bootstrap(c.repo, str(plan_path.relative_to(c.repo)), relative_lease)
    assert bootstrap.execution_role == 'evaluation_helper'
    assert bootstrap.local_gpu_count == 1
    assert bootstrap.lease['pod_id'] == 'real-helper'


def test_ambiguous_create_requires_two_separated_inventory_checks(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    path, _ = attempt(c, 1, pod=False)
    assert not c.reconcile([])
    assert not c.reconcile([])
    monkeypatch.setattr(helpers.time, 'time', lambda: 2061)
    assert c.reconcile([])
    assert helpers.read(path / 'absent_confirmed.json')['two_fresh_inventory_checks']


def test_reconciliation_adopts_exact_owned_name_and_refuses_duplicate_or_unknown(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    path, row = attempt(c, 1, pod=False)
    monkeypatch.setattr(helpers.dispatch, 'cli', lambda *a: row)
    assert c.reconcile([row])
    assert helpers.read(path / 'pod.json')['id'] == row['id']
    with pytest.raises(ValueError, match='duplicate'):
        c.reconcile([row, dict(row, id='other')])
    with pytest.raises(ValueError, match='unregistered'):
        c.reconcile([row, dict(row, id='unknown', name=c.prefix+'9999')])
    with pytest.raises(ValueError, match='identity'):
        monkeypatch.setattr(helpers.dispatch, 'cli', lambda *a: dict(row, networkVolumeId='unrelated-volume'))
        c.reconcile([row])


def test_condensed_list_is_expanded_before_volume_identity_verification(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    path, row = attempt(c, 1, pod=False)
    listing = {key: row[key] for key in ('id', 'name', 'gpuCount')}
    calls = []
    monkeypatch.setattr(helpers.dispatch, 'cli', lambda args, *command: calls.append(command) or row)
    assert c.reconcile([listing])
    assert calls == [('pod', 'get', row['id'])]
    assert helpers.read(path / 'pod.json')['networkVolumeId'] == 'owned-volume'


def test_actual_cli_nested_ssh_schema_and_legacy_endpoint(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    path = c.directory / 'allocation_lease.json'
    remote = c.remote_args({'ssh': {'ip': '157.66.255.80', 'port': 11523, 'id': 'connection'}}, path)
    assert (remote.ssh_host, remote.ssh_port) == ('157.66.255.80', 11523)
    legacy = c.remote_args({'publicIp': '192.0.2.1', 'portMappings': {'22': 10022}}, path)
    assert (legacy.ssh_host, legacy.ssh_port) == ('192.0.2.1', 10022)


def test_capacity_accepts_null_absence_and_only_funded_live_quote(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({'data': {'gpuTypes': [{'g1': {'stockStatus': 'Low', 'uninterruptablePrice': 4.59}, 'g2': None}]}}).encode()
    monkeypatch.setattr(helpers.urllib.request, 'urlopen', lambda *a, **k: Response())
    assert c.capacity() == [1]


def test_purchase_ceiling_counts_failed_allocated_slots_and_unresolved_creates(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    path, _ = attempt(c, 1, count=2)
    atomic_json(path / 'boot_failed.json', {'failed': True})
    monkeypatch.setattr(helpers.dispatch, 'cli', lambda *a: pytest.fail('ceiling must precede provider mutation'))
    with pytest.raises(ValueError, match='ceiling'):
        c.acquire(1)


def test_unresolved_create_blocks_another_purchase(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    attempt(c, 1, pod=False)
    monkeypatch.setattr(helpers.dispatch, 'cli', lambda *a: pytest.fail('ambiguous create must block provider mutation'))
    with pytest.raises(ValueError, match='reconciled'):
        c.acquire(1)


def test_creation_uses_ephemeral_public_key_and_never_changes_account_keys(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    monkeypatch.setattr(c, 'ensure_ssh_key', lambda: 'ssh-ed25519 public-only')
    calls = []
    def cli(args, *command):
        calls.append(command)
        return {'id': 'owned-helper', 'name': command[command.index('--name')+1], 'gpuCount': 2}
    monkeypatch.setattr(helpers.dispatch, 'cli', cli)
    folder = c.acquire(2)
    assert calls[0][:2] == ('pod', 'create')
    assert json.loads(calls[0][calls[0].index('--env')+1]) == {'PUBLIC_KEY': 'ssh-ed25519 public-only'}
    assert not any('secret-private-token' in part for part in calls[0])
    assert helpers.read(folder / 'pod.json')['id'] == 'owned-helper'


def test_possible_bootstrap_launch_is_never_terminated_by_optional_controller(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    path, _ = attempt(c, 1)
    atomic_json(path / 'bootstrap_launch_intent.json', {'may_have_launched': True})
    monkeypatch.setattr(helpers, 'delete_own_pod', lambda *a: pytest.fail('controller must not kill possible work'))
    c.preserve_failed_boot(path, OSError('secret-private-token'))
    assert 'secret-private-token' not in (path / 'boot_failed.json').read_text()
    assert helpers.read(path / 'boot_failed.json')['primary_unaffected']


def test_failed_pre_execution_cleanup_deletes_only_own_helper_and_keeps_volume(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    path, pod = attempt(c, 1)
    monkeypatch.setattr(helpers.dispatch, 'cli', lambda *a: pod)
    deleted = []
    monkeypatch.setattr(helpers, 'delete_own_pod', lambda lease, key: deleted.append(lease['pod_id']) or True)
    c.preserve_failed_boot(path, RuntimeError('control failed'))
    assert deleted == [pod['id']]
    proof = helpers.read(path / 'pre_execution_cleanup_proof.json')
    assert proof['network_volume_retained'] and proof['bootstrap_never_dispatched']
    assert (path / 'provider_absence_verified.json').exists()
    assert not (c.root / 'allocations/primary-pod/lease_guard/STOP').exists()


def test_cross_pod_lock_probe_requires_remote_block_then_acquire(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    path, _ = attempt(c, 1)
    script = c.repo / 'scripts/coding_coverage_v2_lock_probe.py'
    script.parent.mkdir()
    import shutil
    shutil.copyfile(Path(helpers.__file__).with_name('coding_coverage_v2_lock_probe.py'), script)
    outcomes = []
    def ssh(remote, command):
        import shlex
        args = shlex.split(command)
        expected = args[args.index('--expect')+1]
        output = Path(args[args.index('--receipt')+1])
        receipt = {'state': expected, 'hostname': 'other-pod'}
        atomic_json(output, receipt)
        outcomes.append(expected)
        return b''
    monkeypatch.setattr(helpers.dispatch, 'ssh', ssh)
    c.lock_probe(SimpleNamespace(), path)
    assert outcomes == ['blocked', 'acquired']
    assert helpers.read(path / 'lock_probe/verified.json')['cross_host_exclusion_verified']


def test_finished_checks_local_work_and_deadline_without_provider_calls(tmp_path, monkeypatch):
    c = controller(tmp_path, monkeypatch)
    assert not c.finished()
    atomic_json(c.root / 'evaluation/jobs/pending/generation_complete.json', {'done': True})
    assert c.finished()
