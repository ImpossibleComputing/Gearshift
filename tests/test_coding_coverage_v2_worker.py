"""Worker isolation, fail-closed initialization, and real-process queue ownership."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
import transformers
from gearshift.coding_control import sha, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import coding_coverage_v2_worker as worker


def claim_context(root, attempt='attempt_test'):
    return {'root': root, 'attempt_id': attempt}


def test_flock_excludes_live_process_and_recovers_after_sigkill_without_pid_checks(tmp_path):
    code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from coding_coverage_v2_worker import claim_job
with claim_job({'root':Path(sys.argv[2]),'attempt_id':'killed_attempt'}, 'job_1') as acquired:
    print('acquired' if acquired else 'blocked', flush=True)
    sys.stdin.readline()
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(Path(worker.__file__).parent), str(tmp_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'acquired'
        lock = tmp_path / 'claims/job_1.lock'; inode = lock.stat().st_ino
        with worker.claim_job(claim_context(tmp_path), 'job_1') as acquired:
            assert acquired is False
        child.kill(); child.wait(timeout=10)
        # SIGKILL cannot run the context manager's cleanup. The active metadata
        # remains, but the kernel released ownership and no PID guess is needed.
        assert json.loads((tmp_path / 'claims/job_1.owner.json').read_text())['state'] == 'active'
        with worker.claim_job(claim_context(tmp_path), 'job_1') as acquired:
            assert acquired is True
            assert lock.stat().st_ino == inode
        assert lock.exists() and lock.stat().st_ino == inode
        history = list((tmp_path / 'claims/history/job_1').glob('*.json'))
        assert len(history) == 1
        assert json.loads(history[0].read_text())['previous_owner_did_not_release_cleanly'] is True
    finally:
        if child.poll() is None: child.kill()
        child.communicate(timeout=10)


def test_empty_crash_claim_and_reused_pid_metadata_do_not_block_work(tmp_path):
    claims = tmp_path / 'claims'; claims.mkdir()
    (claims / 'job_1.lock').touch()
    # A live PID written by a departed/reused owner is irrelevant to flock.
    write(claims / 'job_1.owner.json', {'state': 'active', 'pid': os.getpid(), 'attempt_id': 'old'})
    with worker.claim_job(claim_context(tmp_path), 'job_1') as acquired: assert acquired
    (claims / 'job_2.lock').touch()
    with worker.claim_job(claim_context(tmp_path), 'job_2') as acquired: assert acquired


def test_claim_failure_releases_same_inode_and_preserves_audit_metadata(tmp_path):
    with pytest.raises(RuntimeError, match='job failure'):
        with worker.claim_job(claim_context(tmp_path), 'job_1') as acquired:
            assert acquired
            inode = (tmp_path / 'claims/job_1.lock').stat().st_ino
            raise RuntimeError('job failure')
    with worker.claim_job(claim_context(tmp_path, 'retry'), 'job_1') as acquired:
        assert acquired and (tmp_path / 'claims/job_1.lock').stat().st_ino == inode
    assert json.loads((tmp_path / 'claims/job_1.owner.json').read_text())['state'] == 'released'


@pytest.mark.parametrize('job_id', ['../outside', '/absolute', '', 'job/child'])
def test_claim_rejects_paths_outside_its_queue(tmp_path, job_id):
    with pytest.raises(ValueError, match='Unsafe'):
        with worker.claim_job(claim_context(tmp_path), job_id): pass


def test_claim_refuses_symlink_lock_file(tmp_path):
    root = tmp_path / 'evaluation'; (root / 'claims').mkdir(parents=True)
    outside = tmp_path / 'outside'; outside.write_text('untouched')
    (root / 'claims/job_1.lock').symlink_to(outside)
    with pytest.raises(OSError):
        with worker.claim_job(claim_context(root), 'job_1'): pass
    assert outside.read_text() == 'untouched'


def test_queue_does_not_repeat_job_completed_before_lock_acquisition(tmp_path, monkeypatch):
    import coding_coverage_v2_evaluate as evaluator
    logical = {'job_id': 'job_1', 'step': 0}
    c = {'root': tmp_path, 'plan': {'expected_jobs': [logical]}, 'guard': lambda: None,
         'publish': lambda **kw: None, 'attempt_id': 'test'}
    monkeypatch.setattr(worker, 'setup', lambda c: (None, None, None))
    monkeypatch.setattr(worker, 'materialize_job', lambda c, job: job)
    @contextmanager
    def winner_finished(c, job_id):
        write(tmp_path / 'jobs/job_1/generation_complete.json', {'complete': True})
        yield True
    monkeypatch.setattr(worker, 'claim_job', winner_finished)
    def unexpected(*args): raise AssertionError('Completed job was rerun')
    monkeypatch.setattr(evaluator, 'run_job', unexpected)
    worker.evaluator(c)


def setup_fixture(tmp_path, monkeypatch):
    import gearshift.coding_inference as inference
    import gearshift.coding_gradients as gradients
    import scripts.coding_partial_corpus as corpus
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'NVIDIA H200')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (9, 0))
    monkeypatch.setattr(torch, '__version__', '2.8.0+cu128')
    monkeypatch.setattr(transformers, '__version__', '4.57.6')
    settings = []
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: settings.append(('threads', n)))
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', lambda value: settings.append(('deterministic', value)))
    models = {'source': {'id': 'pinned_source', 'revision': 'source_revision'},
              'receiver': {'id': 'pinned_receiver', 'revision': 'receiver_revision'}}
    backend_calls = []
    def backend(spec):
        backend_calls.append(spec)
        return SimpleNamespace(model=torch.nn.Linear(1, 1).requires_grad_(False).eval(),
            tokenizer=SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: 'identical_tokenizer')))
    monkeypatch.setattr(inference, 'Backend', backend)
    mapper = torch.nn.Linear(3, 2)
    saved = {k: v.detach().clone() for k, v in mapper.state_dict().items()}
    monkeypatch.setattr(gradients, 'AffineMapper', lambda source, receiver: torch.nn.Linear(3, 2))
    checkpoint = tmp_path / 'starting.pt'; torch.save({'state_dict': saved}, checkpoint)
    histories = ([{'task_id': f'train/{i}', 'split': 'training'} for i in range(104)] +
                 [{'task_id': f'validation/{i}', 'split': 'validation'} for i in range(21)])
    def load(repo, path, *, features):
        assert features is False
        return histories, [], {}
    monkeypatch.setattr(corpus, 'load_corpus', load)
    d = {'training_task_ids': [h['task_id'] for h in histories[:104]],
         'validation_task_ids': [h['task_id'] for h in histories[104:]],
         'selected_checkpoint': 'starting.pt', 'selected_checkpoint_sha256': sha(checkpoint),
         'optimizer': {'lr': 1e-5, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0., 'foreach': False, 'fused': False}}
    for name in ['schedules', 'panels', 'answer_seeds']:
        write(tmp_path / (name + '.json'), {'record': name})
        d[name + '_path'] = name + '.json'; d[name + '_sha256'] = sha(tmp_path / (name + '.json'))
    write(tmp_path / 'corpus.json', {'public_sources_only': True})
    d['corpus_manifest'] = 'corpus.json'; d['corpus_manifest_sha256'] = sha(tmp_path / 'corpus.json')
    (tmp_path / 'frozen.py').write_text('immutable numerical implementation')
    d['unchanged_source_files'] = {'frozen.py': sha(tmp_path / 'frozen.py')}
    original_read = worker.read
    def public_read(path):
        assert not Path(path).resolve().is_relative_to((tmp_path / 'data/coding_pilot_v1/private').resolve()), 'Private tests loaded during setup'
        return original_read(path)
    monkeypatch.setattr(worker, 'read', public_read)
    c = {'declaration': d, 'config': {'models': models}, 'guard': lambda: None,
         'publish': lambda **kw: None, 'telemetry': SimpleNamespace(sample=lambda **kw: None, reset=lambda *a: None),
         'identity': {}, 'status_root': tmp_path / 'attempt', 'plan': {'prior_controls': {'frozen': True}}}
    return c, models, backend_calls, saved, settings


@pytest.mark.parametrize('arm', ['FIXED', 'ROTATING'])
def test_setup_builds_only_requested_arm_with_exact_initial_state_and_no_private_tests(tmp_path, monkeypatch, arm):
    c, models, backend_calls, saved, settings = setup_fixture(tmp_path, monkeypatch)
    state = worker.setup(c, arm)
    assert len(state) == 10
    assert backend_calls == [models['source'], models['receiver']]
    assert [set(part) for part in state[7:]] == [{arm}, {arm}, {arm}]
    assert not state[9][arm].state
    assert all(torch.equal(value, saved[key]) for key, value in state[7][arm].state_dict().items())
    assert ('deterministic', True) in settings and ('threads', 8) in settings
    assert not torch.backends.cuda.matmul.allow_tf32
    assert c['identity']['runtime']['dtype'] == 'bfloat16'
    assert c['identity']['runtime']['attention'] == 'sdpa'


@pytest.mark.parametrize('count,name', [(0, 'NVIDIA H200'), (2, 'NVIDIA H200'), (1, 'NVIDIA H100')])
def test_setup_rejects_wrong_gpu_count_or_architecture_before_model_loading(tmp_path, monkeypatch, count, name):
    c, _, backend_calls, _, _ = setup_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: count)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: name)
    with pytest.raises(ValueError, match='Exactly one equivalent H200'): worker.setup(c, 'FIXED')
    assert backend_calls == []


@pytest.mark.parametrize('mutation', ['numerical_source', 'schedule', 'corpus', 'split'])
def test_setup_rejects_frozen_input_drift_before_model_loading(tmp_path, monkeypatch, mutation):
    c, _, backend_calls, _, _ = setup_fixture(tmp_path, monkeypatch)
    if mutation == 'numerical_source': (tmp_path / 'frozen.py').write_text('changed')
    elif mutation == 'schedule': write(tmp_path / 'schedules.json', {'changed': True})
    elif mutation == 'corpus': write(tmp_path / 'corpus.json', {'changed': True})
    else: c['declaration']['validation_task_ids'] = ['other']
    with pytest.raises(ValueError): worker.setup(c, 'FIXED')
    assert backend_calls == []


def test_evaluation_setup_creates_no_optimizer(tmp_path, monkeypatch):
    c, _, backend_calls, _, _ = setup_fixture(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs): raise AssertionError('Evaluation created optimizer')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    assert len(worker.setup(c)) == 3
    assert len(backend_calls) == 2


def context_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    original_path = Path
    monkeypatch.setattr(worker, 'Path', lambda value: tmp_path if str(value) == '/workspace' else original_path(value))
    now = [1000.]
    monkeypatch.setattr(worker.time, 'time', lambda: now[0])
    monkeypatch.setattr(worker.signal, 'signal', lambda *a: None)
    write(tmp_path / 'configs/coding_pilot_v1/pilot.json', {'models': {'source': {'revision': 's'}, 'receiver': {'revision': 'r'}}})
    write(tmp_path / 'declaration.json', {'schedules_sha256': '1' * 64, 'corpus_manifest_sha256': '2' * 64,
          'selected_checkpoint_sha256': '3' * 64, 'optimizer': {'lr': 1e-5}})
    write(tmp_path / 'public.json', {'scientific': 'immutable'})
    lease = {'experiment_id': 'test_v2', 'pod_id': 'ownedpod', 'allowed_result_root': str(tmp_path / 'results/test_v2'),
             'allocation_epoch': 900., 'deadline_epoch': 2000., 'upper_hourly_usd': 19.,
             'baseline_usd': 336., 'baseline_gpu_hours': 59., 'total_cap_usd': 1000.,
             'total_cap_gpu_hours': 500., 'cleanup_reserve_usd': 40., 'gpu_count': 4}
    write(tmp_path / 'lease.json', lease)
    plan = {'experiment_id': 'test_v2', 'result_root': 'results/test_v2', 'declaration_path': 'declaration.json',
            'declaration_sha256': sha(tmp_path / 'declaration.json'), 'allocation_lease_path': 'lease.json',
            'allocation_lease_sha256': sha(tmp_path / 'lease.json'), 'code_commit': 'a' * 40,
            'private_tests_sha256': 'f' * 64,
            'input_manifest': {'public.json': {'sha256': sha(tmp_path / 'public.json'), 'bytes': (tmp_path / 'public.json').stat().st_size}}}
    write(tmp_path / 'plan.json', plan)
    return plan, lease, now


def test_worker_guard_requires_only_local_immutable_deadline_or_stop(tmp_path, monkeypatch):
    import socket
    import urllib.request
    _, lease, now = context_fixture(tmp_path, monkeypatch)
    c = worker.context('plan.json', 'evaluate', 'eval_2')
    def network_forbidden(*args, **kwargs): raise AssertionError('Network dependency in worker guard')
    monkeypatch.setattr(socket, 'getaddrinfo', network_forbidden)
    monkeypatch.setattr(urllib.request, 'urlopen', network_forbidden)
    write(c['top'] / 'mirror_connection_failed.json', {'reason': 'offline console'})
    c['guard']()
    # A rewritten lease file cannot extend the already captured deadline.
    write(tmp_path / 'lease.json', {**lease, 'deadline_epoch': 99999999.})
    now[0] = 1879.; c['guard']()
    now[0] = 1880.
    with pytest.raises(TimeoutError, match='deadline'): c['guard']()
    now[0] = 1000.
    write(c['top'] / 'lease_guard/STOP', {'reason': 'local allocation stop'})
    with pytest.raises(TimeoutError, match='lease stop'): c['guard']()


@pytest.mark.parametrize('path', ['public.json', 'declaration.json', 'lease.json'])
def test_worker_context_rejects_modified_manifest_declaration_or_lease(tmp_path, monkeypatch, path):
    context_fixture(tmp_path, monkeypatch)
    write(tmp_path / path, {'modified': True})
    with pytest.raises((ValueError, KeyError)): worker.context('plan.json', 'evaluate', 'eval_2')


def test_async_consumer_rejects_foreign_optimizer_model_or_corpus_identity(tmp_path, monkeypatch):
    from gearshift.coding_control import digest
    plan, _, _ = context_fixture(tmp_path, monkeypatch)
    c = worker.context('plan.json', 'evaluate', 'eval_2')
    assert c['private_tests_sha256'] == plan['private_tests_sha256']
    folder = c['top'] / 'arms/FIXED/checkpoints/step_0128'
    folder.mkdir(parents=True)
    mapper = folder/'mapper.pt'; mapper.write_bytes(b'known immutable checkpoint')
    identity = worker.checkpoint_identity(plan,c['declaration'],'FIXED')
    m = {'arm':'FIXED','step':128,'schedule_position':128,'complete_resumable':True,
         'verified_roundtrip':True,'checkpoint_identity':identity,'checkpoint_identity_sha256':digest(identity),
         'files':{'mapper.pt':{'sha256':sha(mapper),'bytes':mapper.stat().st_size}}}
    logical={'job_id':'fixed_0128_task00','arm':'FIXED','step':128}
    write(folder/'manifest.json',m)
    assert worker.materialize_job(c,logical)['checkpoint']['sha256']==sha(mapper)
    for field in ['corpus_sha256','models','configuration']:
        foreign={**identity,field:'different'}
        write(folder/'manifest.json',{**m,'checkpoint_identity':foreign,'checkpoint_identity_sha256':digest(foreign)})
        with pytest.raises(ValueError,match='scientific identity'):worker.materialize_job(c,logical)
    write(folder/'manifest.json',m)
    mapper.write_bytes(b'corruption')
    with pytest.raises(ValueError,match='checksum'):worker.materialize_job(c,logical)


def test_worker_rejects_a_hash_valid_lease_that_exceeds_cumulative_budget(tmp_path, monkeypatch):
    plan, lease, _ = context_fixture(tmp_path,monkeypatch)
    write(tmp_path/'lease.json',{**lease,'baseline_usd':999.})
    plan['allocation_lease_sha256']=sha(tmp_path/'lease.json');write(tmp_path/'plan.json',plan)
    with pytest.raises(ValueError,match='hard ceiling'):worker.context('plan.json','evaluate','eval_2')


def test_worker_stop_is_local_to_its_pod_on_shared_scientific_volume(tmp_path, monkeypatch):
    plan, lease, now = context_fixture(tmp_path, monkeypatch)
    lease['control_relative'] = 'allocations/' + lease['pod_id']
    write(tmp_path / 'lease.json', lease)
    plan['allocation_lease_sha256'] = sha(tmp_path / 'lease.json'); write(tmp_path / 'plan.json', plan)
    c = worker.context('plan.json', 'evaluate', 'ownedpod_eval_0')
    write(c['top'] / 'allocations/anotherpod/lease_guard/STOP', {'reason': 'other allocation stopped'})
    write(c['top'] / 'lease_guard/STOP', {'reason': 'legacy shared stop is not this lease'})
    c['guard']()
    write(c['allocation_control'] / 'lease_guard/STOP', {'reason': 'own allocation stopped'})
    with pytest.raises(TimeoutError): c['guard']()
