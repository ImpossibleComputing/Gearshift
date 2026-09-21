import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('parallel_session_under_test', ROOT / 'scripts/coding_parallel_session.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def pod():
    return {'id': 'pod1', 'gpuCount': 1, 'networkVolumeId': 'volume1', 'costPerHr': 4.59,
        'imageName': 'runpod/pytorch@sha256:' + 'a' * 64}


@pytest.mark.parametrize('field,value', [('gpuCount', 2), ('networkVolumeId', 'other'), ('costPerHr', 5.51), ('imageName', 'moving:tag')])
def test_actual_provider_allocation_is_verified(field, value):
    p = pod(); p[field] = value
    with pytest.raises(ValueError): m.check_pod(p, 'volume1', pod()['imageName'])


def test_price_within_ceiling_and_exact_immutable_image():
    m.check_pod(pod(), 'volume1', pod()['imageName'])


def setup_controller(monkeypatch, tmp_path):
    c = tmp_path / 'evidence/coding_pilot_v1/control'; c.mkdir(parents=True)
    monkeypatch.setattr(m, 'ROOT', tmp_path); monkeypatch.setattr(m, 'C', c)
    m.write(c / 'ledger.json', {})
    monkeypatch.setattr(m, 'decision', lambda ledger: {'stop': False})
    return c


def test_existing_stop_prevents_any_provider_mutation(monkeypatch, tmp_path):
    c = setup_controller(monkeypatch, tmp_path); (c / 'STOP').touch()
    calls = []; monkeypatch.setattr(m, 'cli', lambda *args: calls.append(args))
    plan = {'run_id': 'run1', 'image_digest': pod()['imageName']}
    worker = {'worker_id': 'worker1', 'maximum_seconds': 3600}
    with pytest.raises(RuntimeError, match='Dispatch stopped'):
        m.launch_worker(plan, worker, 'identity', threading.Event())
    assert calls == []


def test_create_failure_cleans_only_new_empty_owned_volume(monkeypatch, tmp_path):
    c = setup_controller(monkeypatch, tmp_path)
    calls = []
    def cli(*args):
        calls.append(args)
        if args[:2] == ('network-volume', 'create'): return {'id': 'newvol', 'name': 'owned'}
        if args[:2] == ('pod', 'create'): raise RuntimeError('no capacity')
        if args[:2] == ('network-volume', 'delete'): return {}
        raise AssertionError(args)
    monkeypatch.setattr(m, 'cli', cli)
    plan = {'run_id': 'run1', 'image_digest': pod()['imageName']}
    worker = {'worker_id': 'worker1', 'maximum_seconds': 3600, 'region': 'CA-MTL-3'}
    event = threading.Event()
    with pytest.raises(RuntimeError, match='no capacity'): m.launch_worker(plan, worker, 'identity', event)
    assert not event.is_set()  # Failed allocation does not cancel independent shards.
    assert ('network-volume', 'delete', 'newvol') in calls
    assert not any(args[:2] == ('pod', 'delete') for args in calls)
    assert (c / 'resource_receipts/newvol.json').exists()


def test_ambiguous_create_is_not_retried(monkeypatch, tmp_path):
    setup_controller(monkeypatch, tmp_path)
    calls = []
    def cli(*args):
        calls.append(args); raise TimeoutError('provider response ambiguous')
    monkeypatch.setattr(m, 'cli', cli)
    with pytest.raises(TimeoutError):
        m.launch_worker({'run_id': 'run1', 'image_digest': pod()['imageName']},
            {'worker_id': 'worker1', 'maximum_seconds': 3600, 'region': 'CA-MTL-3'}, 'id', threading.Event())
    assert len(calls) == 1
    assert (tmp_path / 'evidence/coding_pilot_v1/control/parallel/run1/worker1/allocation_intent.json').exists()


def test_process_liveness_uses_exact_bootstrap_command_not_kill_zero(monkeypatch):
    commands = []
    monkeypatch.setattr(m, 'remote', lambda conn, command, timeout: commands.append(command) or '')
    assert not m.process_alive({}, 123, 'evidence/spec.json')
    assert '/proc/123/cmdline' in commands[0]
    assert 'coding_parallel_session.py' in commands[0]
    assert 'evidence/spec.json' in commands[0]


def test_connected_storage_survives_failed_upload_while_pod_is_removed(monkeypatch, tmp_path):
    setup_controller(monkeypatch, tmp_path)
    calls = []
    def cli(*args):
        calls.append(args)
        if args[:2] == ('network-volume', 'create'): return {'id': 'volume1', 'name': 'owned'}
        if args[:2] == ('pod', 'create'): return {**pod(), 'name': 'ownedpod'}
        if args[:2] == ('pod', 'get'): return {**pod(), 'name': 'ownedpod', 'machine': {'gpuTypeId': 'NVIDIA H200'}}
        if args[:2] == ('ssh', 'info'): return {'connection': {'ip': 'localhost', 'port': 22}}
        if args[:2] == ('pod', 'delete'): return {}
        raise AssertionError(args)
    monkeypatch.setattr(m, 'cli', cli)
    monkeypatch.setattr(m, 'pod_detail_rest', lambda pod_id: {**pod(), 'name': 'ownedpod', 'machine': {'gpuTypeId': 'NVIDIA H200'}})
    monkeypatch.setattr(m, 'remote', lambda *args: '')
    def broken(*args): raise RuntimeError('transfer failure')
    monkeypatch.setattr(m, 'upload', broken)
    plan = {'run_id': 'run1', 'stage': 'memory', 'image_digest': pod()['imageName'],
        'approval_sha256': 'a' * 64, 'files': {}, 'gates': {}}
    worker = {'worker_id': 'worker1', 'maximum_seconds': 3600, 'region': 'CA-MTL-3',
        'task_ids': [], 'command': ['scripts/coding_memory_preflight.py']}
    with pytest.raises(RuntimeError, match='transfer failure'):
        m.launch_worker(plan, worker, 'identity', threading.Event())
    assert ('pod', 'delete', 'pod1') in calls
    assert not any(args[:2] == ('network-volume', 'delete') for args in calls)


def test_missing_volume_in_create_is_verified_with_fresh_expanded_get(monkeypatch, tmp_path):
    created = {**pod(), 'name': 'owned', 'env': {'PUBLIC_KEY': 'unneeded'}, 'machine': {'gpuDisplayName': 'H200 SXM'}}
    del created['networkVolumeId']
    detail = {**created, 'networkVolume': {'id': 'volume1', 'name': 'owned-storage'}}
    calls = []
    monkeypatch.setattr(m, 'pod_detail_rest', lambda pod_id: calls.append(pod_id) or detail)
    verified = m.verify_created_pod(created, 'volume1', created['imageName'], tmp_path / 'observed.json')
    assert calls == ['pod1']
    assert verified['networkVolume']['id'] == 'volume1'
    assert 'env' not in verified
    saved = json.loads((tmp_path / 'observed.json').read_text())
    assert saved['pod'] == verified and 'PUBLIC_KEY' not in (tmp_path / 'observed.json').read_text()


@pytest.mark.parametrize('mutation', ['wrong_volume', 'missing_volume', 'wrong_gpu', 'wrong_id'])
def test_fresh_association_must_match_even_when_create_omits_it(monkeypatch, tmp_path, mutation):
    created = {**pod(), 'name': 'owned'}; del created['networkVolumeId']
    detail = {**created, 'networkVolume': {'id': 'volume1'}, 'machine': {'gpuDisplayName': 'H200 SXM'}}
    if mutation == 'wrong_volume': detail['networkVolume']['id'] = 'other'
    elif mutation == 'missing_volume': del detail['networkVolume']
    elif mutation == 'wrong_gpu': detail['machine']['gpuDisplayName'] = 'H100 SXM'
    elif mutation == 'wrong_id': detail['id'] = 'other'
    monkeypatch.setattr(m, 'pod_detail_rest', lambda pod_id: detail)
    with pytest.raises(ValueError): m.verify_created_pod(created, 'volume1', created['imageName'], tmp_path / 'observed.json')
    assert (tmp_path / 'observed.json').exists()


def test_capacity_retry_keeps_other_workers_running_until_new_pod_allocates(monkeypatch, tmp_path):
    from gearshift.coding_capacity_retry import NO_INSTANCES
    c = setup_controller(monkeypatch, tmp_path)
    event = threading.Event(); calls = []; counts = {'volume': 0, 'pod': 0}
    def cli(*args):
        calls.append(args)
        if args[:2] == ('network-volume', 'create'):
            counts['volume'] += 1
            return {'id': 'volume' + str(counts['volume']), 'name': args[3]}
        if args[:2] == ('pod', 'create'):
            counts['pod'] += 1
            assert not event.is_set()
            if counts['pod'] == 1:
                raise RuntimeError('Provider pod create failed: help\n{"error":"' + NO_INSTANCES + '"}')
            return {**pod(), 'name': args[3], 'networkVolumeId': 'volume2'}
        if args[:2] == ('pod', 'get'):
            return {**pod(), 'name': m.PREFIX + 'run1-worker1-capacity02', 'networkVolumeId': 'volume2', 'machine': {'gpuTypeId': 'NVIDIA H200'}}
        if args[:2] in [('pod', 'list'), ('network-volume', 'list')]: return []
        if args[:2] in [('pod', 'delete'), ('network-volume', 'delete')]: return {}
        if args[:2] == ('ssh', 'info'): return {'connection': {'ip': 'localhost', 'port': 22}}
        raise AssertionError(args)
    monkeypatch.setattr(m, 'cli', cli); monkeypatch.setattr(m, 'remote', lambda *args: '')
    monkeypatch.setattr(m, 'pod_detail_rest', lambda pod_id: {**pod(), 'name': m.PREFIX + 'run1-worker1-capacity02',
        'networkVolumeId': 'volume2', 'machine': {'gpuTypeId': 'NVIDIA H200'}})
    def uploaded(*args): raise RuntimeError('Test stops after successful retry allocation')
    monkeypatch.setattr(m, 'upload', uploaded)
    plan = {'run_id': 'run1', 'stage': 'memory', 'image_digest': pod()['imageName'],
        'approval_sha256': 'a' * 64, 'files': {}, 'gates': {}}
    worker = {'worker_id': 'worker1', 'maximum_seconds': 3600, 'region': 'CA-MTL-3',
        'region_candidates': ['CA-MTL-3', 'US-CA-2', 'US-NC-1'], 'task_ids': [], 'command': ['scripts/coding_memory_preflight.py']}
    with pytest.raises(RuntimeError, match='successful retry'): m.launch_worker(plan, worker, 'frozen', event)
    assert counts == {'volume': 2, 'pod': 2}
    assert ('network-volume', 'delete', 'volume1') in calls
    proof = json.loads((c / 'parallel/run1/worker1/capacity_01_verified.json').read_text())
    assert proof['stage_identity'] == 'frozen' and proof['no_gpu_ever_allocated']


def test_capacity_candidates_do_not_retry_ambiguous_provider_failure(monkeypatch, tmp_path):
    setup_controller(monkeypatch, tmp_path); calls = []
    def cli(*args):
        calls.append(args)
        if args[:2] == ('network-volume', 'create'): return {'id': 'volume1', 'name': args[3]}
        if args[:2] == ('pod', 'create'): raise RuntimeError('Provider pod create failed: timeout')
        if args[:2] == ('network-volume', 'delete'): return {}
        raise AssertionError(args)
    monkeypatch.setattr(m, 'cli', cli)
    with pytest.raises(RuntimeError, match='timeout'):
        m.launch_worker({'run_id': 'run1', 'image_digest': pod()['imageName']},
            {'worker_id': 'worker1', 'maximum_seconds': 3600, 'region': 'CA-MTL-3',
                'region_candidates': ['CA-MTL-3', 'US-CA-2', 'US-NC-1']}, 'fixed', threading.Event())
    assert sum(args[:2] == ('pod', 'create') for args in calls) == 1


def test_all_capacity_names_are_available_for_watchdog_preregistration():
    worker = {'worker_id': 'w', 'region_candidates': ['US-CA-2', 'US-NC-1', 'US-CO-1']}
    names = m.allocation_names({'run_id': 'r'}, worker)
    assert len(names) == 3 and len(set(names)) == 3
    assert names[-1].endswith('-capacity03')


def test_only_exact_twice_verified_retained_storage_is_allowed(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    copies=['recovery/first.tar.gz','recovery/second.tar.gz']
    for name in copies:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'complete preserved evidence')
    proof={'volume_id':'kept','both_copies_verified':True,'copies':copies,'archive_sha256':m.sha(tmp_path/copies[0])}
    m.write(tmp_path/'proof.json',proof)
    name=m.PREFIX+'retained-memory-storage'
    plan={'files':{'proof.json':m.sha(tmp_path/'proof.json'),**{f:m.sha(tmp_path/f) for f in copies}},
          'retained_storage':[{'volume_id':'kept','name':name,'recovery_receipt':'proof.json','sha256':m.sha(tmp_path/'proof.json')}]}
    resources={'pods':[],'volumes':[{'id':'kept','name':name}]}
    m.verify_idle_resources(plan,resources)
    with pytest.raises(ValueError,match='pods'):m.verify_idle_resources(plan,{**resources,'pods':[{'id':'running'}]})
    with pytest.raises(ValueError,match='Unaccounted'):m.verify_idle_resources(plan,{**resources,'volumes':resources['volumes']+[{'id':'unknown','name':m.PREFIX+'unknown'}]})
    (tmp_path/copies[1]).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='backup changed'):m.verify_idle_resources(plan,resources)


def test_unlisted_retained_volume_still_blocks_dispatch():
    with pytest.raises(ValueError,match='Unaccounted'):
        m.verify_idle_resources({'files':{}},{'pods':[],'volumes':[{'id':'orphan','name':m.PREFIX+'orphan'}]})
    m.verify_idle_resources({'files':{}},{'pods':[],'volumes':[]})


def test_model_download_gets_one_hour_only_inside_unchanged_allocation():
    download=['python','scripts/coding_download_models.py']
    assert m.bootstrap_command_timeout(download,18000)==3600
    assert m.bootstrap_command_timeout(download,600)==600
    assert m.bootstrap_command_timeout(['apt-get','update'],18000)==1800
    assert m.bootstrap_command_timeout(['apt-get','update'],20)==20


def test_one_failed_shard_does_not_cancel_an_independent_sibling(monkeypatch,tmp_path):
    c=setup_controller(monkeypatch,tmp_path)
    approval=c/'approval.json';approval.write_text('{}');monkeypatch.setattr(m,'APPROVAL',approval)
    plan={'run_id':'independent','workers':[{'worker_id':'bad','region':'US-GA-2'}, {'worker_id':'good','region':'US-GA-2'}]}
    path=tmp_path/'plan.json';m.write(path,plan)
    monkeypatch.setattr(m,'validate_plan',lambda *args:'fixed')
    monkeypatch.setattr(m,'check_dispatch_budget',lambda *args:{})
    monkeypatch.setattr(m,'own_resources',lambda:{'pods':[],'volumes':[]})
    monkeypatch.setattr(m,'quote',lambda:{'securePrice':4.59})
    monkeypatch.setattr(m,'tick',lambda:None)
    monkeypatch.setattr(m.subprocess,'run',lambda *args,**kwargs:type('Result',(),{'returncode':0,'stdout':'state = running'})())
    observed=[]
    def worker(plan,shard,identity,event):
        observed.append((shard['worker_id'],event.is_set()))
        if shard['worker_id']=='bad':raise RuntimeError('Download timed out')
        assert not event.is_set()
        return {'worker_id':'good','passed':True}
    monkeypatch.setattr(m,'launch_worker',worker)
    class Future:
        def __init__(self,fn,args):self.fn,self.args=fn,args
        def result(self):return self.fn(*self.args)
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,*args):return Future(fn,args)
    monkeypatch.setattr(m.concurrent.futures,'ThreadPoolExecutor',Pool)
    monkeypatch.setattr(m.concurrent.futures,'as_completed',lambda futures:iter(futures))
    with pytest.raises(RuntimeError,match='Bounded stage incomplete'):m.run(path)
    done=json.loads((c/'parallel/independent/complete.json').read_text())
    assert observed==[('bad',False),('good',False)]
    assert not done['passed'] and done['workers']==[{'worker_id':'good','passed':True}]
    assert len(done['errors'])==1
