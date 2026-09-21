"""CPU-only synthetic transfer/resume tests; these are not an actual GPU proof."""
import importlib.util
import io
import multiprocessing
import os
from pathlib import Path
import tarfile
import time

import pytest
import torch

from gearshift.coding_control import sha, write
from scripts import coding_confirmation_cross_pod_smoke as c
from scripts import coding_confirmation_resume_smoke as s

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cross_pod_cpu_fixture', ROOT/'tests/test_coding_confirmation_resume_smoke.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


@pytest.fixture(scope='module')
def original(tmp_path_factory):
    folder = tmp_path_factory.mktemp('completed_crosspod_source')/'smoke'
    prepared = s.prepare(ROOT,folder,cap=128,interrupt_after=65)
    plan = s.read(folder/'plan.json')
    for case in s.CASES:
        assert f.spawn(folder,plan,case,'baseline') == 0
        assert f.spawn(folder,plan,case,'interrupt') == 86
        assert f.spawn(folder,plan,case,'resume') == 0
    result = s.verify(ROOT,folder,prepared['plan_sha256'])
    assert result['all_cases_passed'] and not result['cross_pod_resume_for_all_cases']
    archive = folder.parent/'completed.tar.gz'
    with tarfile.open(archive,'w:gz') as output:
        for path in sorted(folder.rglob('*')):
            if path.is_file(): output.add(path,arcname=str(path.relative_to(folder)),recursive=False)
    return folder,archive,sha(archive)


def derived(original,tmp_path,monkeypatch):
    monkeypatch.setattr(c,'valid_runtime',lambda runtime: runtime.get('test_fake_CPU_only') is True)
    folder,archive,expected = original
    target = tmp_path/'cross_pod_smoke'
    result = c.derive(ROOT,archive,expected,'.',target,'synthetic-target-pod')
    return target,result


def test_derivation_preserves_exact_originals_and_restores_only_declared_checkpoints(original,tmp_path,monkeypatch):
    source,archive,expected = original
    before = {str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    target,result = derived(original,tmp_path,monkeypatch)
    manifest = c.validate_derivation(ROOT,target,result['derivation_sha256'],before_resume=True)
    assert sha(target/c.SOURCE_ARCHIVE) == expected
    assert (target/'plan.json').read_bytes() == (source/'plan.json').read_bytes()
    assert not (target/'SMOKE_PROOF.json').exists()
    for case in s.CASES:
        checkpoint = target/'restarted'/case/'resume.json'
        assert checkpoint.read_bytes() == (source/'proof'/case/'interrupted_state.json').read_bytes()
        assert len(s.read(checkpoint)['tokens']) == 65
        assert s.read(checkpoint)['state'] == 'running'
        assert not (checkpoint.parent/'complete.json').exists()
        assert not (checkpoint.parent/'completion_timing.json').exists()
        assert not (target/'proof'/case/'resume_started.json').exists()
        assert not (target/'proof'/case/'resume_result.json').exists()
        attempts = list((checkpoint.parent/'attempts').glob('*.json'))
        assert len(attempts) == 1 and s.read(attempts[0])['state'] == 'started'
    assert before == {str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    assert manifest['baseline_recomputed'] is False and manifest['new_seed_or_prompt'] is False
    assert 'universal' in manifest['scope']


def cpu_target_child(folder,plan,case):
    import gearshift.coding_inference as inference
    torch.set_num_threads(1)
    inference.memory_record = lambda: {'passed':True,'test_fake_CPU_only':True}
    s.answer_base = f.fake_base
    models = {}
    for role in ('source','receiver'):
        backend = f.Backend(); models[role] = (backend,s.Observer(backend))
    attempt = {'pid':os.getpid(),'pod_id':'synthetic-target-pod','host':'cpu-only-target',
               'process_started_ns':time.time_ns(),'runtime':{'test_fake_CPU_only':True}}
    s.run_case(ROOT,folder,plan,case,'resume',models,attempt,lambda:None)


def cpu_target_resume(folder):
    for case in s.CASES:
        proc = multiprocessing.get_context('fork').Process(target=cpu_target_child,args=(folder,s.read(folder/'plan.json'),case))
        proc.start(); proc.join(20)
        if proc.is_alive(): proc.kill(); proc.join(); raise RuntimeError('Bounded synthetic resume timed out')
        assert proc.exitcode == 0


def test_once_only_wrapper_calls_no_baseline_and_compares_against_original(original,tmp_path,monkeypatch):
    target,result = derived(original,tmp_path,monkeypatch); calls = []
    monkeypatch.setattr(s,'allocation_guard',lambda *a: ({'pod_id':'synthetic-target-pod'},lambda:None))
    class Process:
        pid = 1234
        def __init__(self,command,**kwargs):
            calls.append((command,kwargs))
        def wait(self,**kwargs): cpu_target_resume(target); return 0
        def poll(self): return 0
    monkeypatch.setattr(c.subprocess,'Popen',Process)
    proof = c.resume_once(ROOT,target,result['derivation_sha256'],'lease','a'*64)
    assert proof['all_four_cases_passed'] and proof['cross_pod_resume_for_all_cases']
    assert len(calls) == 1
    command,options = calls[0]
    assert command[command.index('--phase')+1] == 'resume' and command[command.index('--case')+1] == 'all'
    assert options['start_new_session'] is False and options['env']['HF_HUB_OFFLINE'] == '1'
    for case in s.CASES:
        state = s.read(target/'restarted'/case/'resume.json')
        assert state['timing']['resume_count'] == 1 and state['timing']['complete'] is False
    with pytest.raises(ValueError): c.resume_once(ROOT,target,result['derivation_sha256'],'lease','a'*64)
    assert len(calls) == 1


@pytest.mark.parametrize('mutation',['baseline','identity','restored_checkpoint','prior_resume_artifact','source_archive','helper'])
def test_any_derivation_input_or_active_artifact_drift_fails_before_resume(original,tmp_path,monkeypatch,mutation):
    target,result = derived(original,tmp_path,monkeypatch)
    paths = {'baseline':'baseline/receiver_native/answer_record.json',
        'identity':'restarted/receiver_native/identity.json','restored_checkpoint':'restarted/receiver_native/resume.json',
        'prior_resume_artifact':'proof/receiver_native/resume_started.json','source_archive':c.SOURCE_ARCHIVE}
    if mutation == 'helper':
        repo=tmp_path/'staged'; (repo/'scripts').mkdir(parents=True); (repo/c.SCRIPT).write_text('different')
    else:
        repo=ROOT; (target/paths[mutation]).write_text('{}')
    with pytest.raises(ValueError): c.validate_derivation(repo,target,result['derivation_sha256'],before_resume=True)


def test_failed_single_attempt_cannot_retry_or_claim_proof(original,tmp_path,monkeypatch):
    target,result = derived(original,tmp_path,monkeypatch); calls=[]
    monkeypatch.setattr(s,'allocation_guard',lambda *a: ({'pod_id':'synthetic-target-pod'},lambda:None))
    class Process:
        pid=1234
        def __init__(self,*a,**kw): calls.append(True)
        def wait(self,**kw): return 1
        def poll(self): return 1
    monkeypatch.setattr(c.subprocess,'Popen',Process)
    with pytest.raises(RuntimeError): c.resume_once(ROOT,target,result['derivation_sha256'],'lease','a'*64)
    assert (target/'cross_pod_execution/failure.json').exists()
    assert not (target/'CROSS_POD_PROOF.json').exists()
    with pytest.raises(ValueError): c.resume_once(ROOT,target,result['derivation_sha256'],'lease','a'*64)
    assert len(calls)==1


def test_wrong_target_pod_rejected_before_attempt(original,tmp_path,monkeypatch):
    target,result = derived(original,tmp_path,monkeypatch)
    monkeypatch.setattr(s,'allocation_guard',lambda *a: ({'pod_id':'wrong-pod'},lambda:None))
    with pytest.raises(ValueError,match='target pod'): c.resume_once(ROOT,target,result['derivation_sha256'],'lease','a'*64)
    assert not (target/'cross_pod_execution').exists()


def test_derivation_never_replaces_folder_or_original_and_requires_archive_hash(original,tmp_path,monkeypatch):
    source,archive,expected=original; monkeypatch.setattr(c,'valid_runtime',lambda r:True)
    with pytest.raises(ValueError,match='archive hash'): c.derive(ROOT,archive,'0'*64,'.',tmp_path/'new','target')
    with pytest.raises(ValueError,match='must be new'): c.derive(ROOT,archive,expected,'.',source,'target')
    with pytest.raises(ValueError,match='different target'): c.derive(ROOT,archive,expected,'.',tmp_path/'new','synthetic-cpu-test')


@pytest.mark.parametrize('kind',['traversal','symlink','weights','duplicate'])
def test_archive_never_extracts_unsafe_or_heavy_members(tmp_path,kind):
    path=tmp_path/'unsafe.tar'
    with tarfile.open(path,'w') as archive:
        name={'traversal':'../plan.json','symlink':'plan.json','weights':'model.safetensors','duplicate':'plan.json'}[kind]
        info=tarfile.TarInfo(name); info.size=2
        if kind=='symlink': info.type=tarfile.SYMTYPE; info.linkname='/etc/passwd'; info.size=0
        archive.addfile(info,io.BytesIO(b'{}'))
        if kind=='duplicate': archive.addfile(info,io.BytesIO(b'{}'))
    with pytest.raises(ValueError): c.archive_files(path,'.')


def test_real_runtime_gate_rejects_cpu_or_changed_numerics():
    assert not c.valid_runtime({'test_fake_CPU_only':True})
    runtime={'torch':'2.8.0+cu128','transformers':'4.57.6','attention':'sdpa','dtype':'bfloat16',
             'gpu':'NVIDIA H200','TF32':False,'deterministic_algorithms':True}
    assert c.valid_runtime(runtime)
    assert not c.valid_runtime({**runtime,'TF32':True})
