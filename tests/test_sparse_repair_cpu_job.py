"""CPU job plan/backup invariants only: no provider calls or process launch."""
import json
import fcntl
import os
from pathlib import Path
import tarfile
import pytest
from scripts import sparse_repair_cpu_job as cpu


def setup(tmp,monkeypatch):
    root=tmp/'results/sparse_repair_01';root.mkdir(parents=True)
    lease={'experiment_id':'sparse_repair_01','pod_id':'cpu01','allocation_epoch':100,'deadline_epoch':10000,
        'upper_hourly_usd':1.5,'gpu_count':0,'baseline_usd':1360,'other_reserved_usd':20,
        'total_cap_usd':2500,'total_cap_gpu_hours':None,'cleanup_reserve_usd':40,
        'allowed_result_root':str(root),'control_relative':'allocations/cpu01','network_volume_id':'lgk3howszi'}
    path=tmp/'lease.json';path.write_text(json.dumps(lease));decl=tmp/'declaration.json';decl.write_text('{}')
    monkeypatch.setattr(cpu,'implementation_hashes',lambda repo=cpu.ROOT:{'mocked':'same'})
    plan={'experiment_id':'sparse_repair_01','job_id':'score01','lease_path':'lease.json',
        'lease_sha256':cpu.sha256_file(path),'declaration_path':'declaration.json',
        'declaration_sha256':cpu.sha256_file(decl),'result_root':'results/sparse_repair_01',
        'cpu_ids':list(range(8)),'implementation_hashes':{'mocked':'same'},'automatic_retries':0}
    return root,lease,plan


def test_plan_accepts_only_matching_cpu_lease_scope_and_eightcores(tmp_path,monkeypatch):
    root,lease,p=setup(tmp_path,monkeypatch)
    assert cpu.validate_plan(p,tmp_path)==(lease,root)
    for change in ({'cpu_ids':[1,2]},{'cpu_ids':[1]*8},{'private_tests':'other/private.json'},
                   {'automatic_retries':1},{'implementation_hashes':{}},{'result_root':'other'}):
        with pytest.raises(ValueError):cpu.validate_plan({**p,**change},tmp_path)
    lease['gpu_count']=1;(tmp_path/'lease.json').write_text(json.dumps(lease));p['lease_sha256']=cpu.sha256_file(tmp_path/'lease.json')
    with pytest.raises(ValueError,match='CPU lease'):cpu.validate_plan(p,tmp_path)


def test_exact_fixed_chain_argv_separates_public_and_private_stages(tmp_path,monkeypatch):
    root,lease,p=setup(tmp_path,monkeypatch);wrapper=root/'allocations/cpu01/sparse_cpu_job/score01';wrapper.mkdir(parents=True)
    (wrapper/'preflight.json').write_text('{}')
    for stage in ('preflight','prepare','run','finalize'):
        argv=cpu.command(stage,p,tmp_path,wrapper)
        assert argv[:3]==[cpu.PYTHON,'scripts/sparse_repair_score.py',stage]
        assert ('--private-tests' in argv)==(stage in ('prepare','run'))
        assert '--cpu-ids' in argv if stage in ('preflight','run') else '--cpu-ids' not in argv
    with pytest.raises(ValueError):cpu.command('seal',p,tmp_path,wrapper)
    assert cpu.command('run',p,tmp_path,wrapper)[-1]==cpu.sha256_file(wrapper/'preflight.json')


def test_existing_live_owner_not_overwritten(tmp_path,monkeypatch):
    p=tmp_path/'lease_guard/supervisor_registration.json';p.parent.mkdir();old={'pid':99,'pgid':99,'start_ticks':2,'boot_id':'a'};p.write_text(json.dumps(old))
    monkeypatch.setattr(cpu,'linux_process_identity',lambda pid:old)
    monkeypatch.setattr(cpu.os,'getpid',lambda:77)
    assert cpu.existing_owner_is_live(tmp_path)
    monkeypatch.setattr(cpu,'linux_process_identity',lambda pid:{**old,'start_ticks':3})
    assert not cpu.existing_owner_is_live(tmp_path)


def test_candidate_scan_exactlauncher_not_arbitrary_python(tmp_path):
    proc=tmp_path/'proc';proc.mkdir();repo=tmp_path/'repo'
    for pid,args in [(10,[b'/usr/bin/python3',b'-I',str(repo/'scripts/coding_sandbox_child_v2.py').encode(),b'/tmp/a/payload.json',b'']),
                     (11,[b'python3',b'/unrelated/script.py',b''])]:
        p=proc/str(pid);p.mkdir();(p/'cmdline').write_bytes(b'\0'.join(args))
    assert cpu.candidate_processes(repo,proc)==[10]


def test_compact_cpu_release_excludes_privatefile_and_unrelated_outputs(tmp_path,monkeypatch):
    root,lease,p=setup(tmp_path,monkeypatch)
    private=tmp_path/'private';private.mkdir();(private/'development.json').write_text('{"secret":"not for archive"}')
    scores=root/'screen/scoring';scores.mkdir(parents=True);(scores/'scored_answer_manifest.json').write_text('{}')
    (scores/'execution.lock').write_text('')
    wrapper=root/'allocations/cpu01/sparse_cpu_job/score01';wrapper.mkdir(parents=True);(wrapper/'exited.json').write_text('{"all_own_cpu_workers_stopped":true}')
    unrelated=root/'unrelated';unrelated.mkdir();(unrelated/'output.json').write_text('{}')
    path=cpu.jobs.compact_release(root,lease,[scores],wrapper,'complete')
    assert cpu.verify_release_receipt(lease)
    records=json.loads(path.read_text());archives=[root/r['path'] for r in records['files'] if r['path'].endswith('.tar.gz')]
    assert len(archives)==2
    with tarfile.open(archives[0]) as archive:
        names=archive.getnames()
        assert 'screen/scoring/scored_answer_manifest.json' in names
        assert not any('private' in n or 'unrelated' in n for n in names)


def test_stopped_proof_does_not_authorize_with_sandbox_launcher(tmp_path,monkeypatch):
    monkeypatch.setattr(cpu.scorer.base,'cleanup_orphans',lambda lease:[])
    monkeypatch.setattr(cpu.jobs,'group_members',lambda group:[])
    monkeypatch.setattr(cpu,'candidate_processes',lambda repo:[91])
    ticks=iter([0,6]);monkeypatch.setattr(cpu.time,'monotonic',lambda:next(ticks))
    result=cpu.stopped_evidence({'pod_id':'synthetic'})
    assert result['all_own_cpu_workers_stopped'] is False
    assert result['remaining_sandbox_launcher_pids']==[91]


def test_held_execution_lock_blocks_release_without_scanning_or_backup(tmp_path,monkeypatch):
    root,lease,p=setup(tmp_path,monkeypatch)
    scores=root/'screen/scoring';scores.mkdir(parents=True)
    wrapper=root/'allocations/cpu01/sparse_cpu_job/score01';wrapper.mkdir(parents=True)
    def forbidden(*args,**kwargs):raise AssertionError('Must not inspect or release another scorer')
    monkeypatch.setattr(cpu,'stopped_evidence',forbidden)
    monkeypatch.setattr(cpu.jobs,'compact_release',forbidden)
    with (scores/'execution.lock').open('a') as other:
        fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert cpu.preserve_after_stop(root,lease,scores,wrapper,'failed_preserved',1) is False
    assert (wrapper/'release_withheld.json').is_file()
    assert json.loads((wrapper/'exited.json').read_text())['execution_lock_busy'] is True
    assert not (cpu.control_root(lease)/'gpu_release_verified.json').exists()


def test_release_keeps_execution_lock_through_stopped_proof_and_backup(tmp_path,monkeypatch):
    root,lease,p=setup(tmp_path,monkeypatch)
    scores=root/'screen/scoring';scores.mkdir(parents=True)
    wrapper=root/'allocations/cpu01/sparse_cpu_job/score01';wrapper.mkdir(parents=True)
    calls=[]
    def locked():
        with (scores/'execution.lock').open('a') as other:
            with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def proof(lease):locked();calls.append('proof');return {'all_own_cpu_workers_stopped':True}
    def backup(*args):locked();calls.append('backup')
    monkeypatch.setattr(cpu,'stopped_evidence',proof)
    monkeypatch.setattr(cpu.jobs,'compact_release',backup)
    assert cpu.preserve_after_stop(root,lease,scores,wrapper,'complete',0) is True
    assert calls==['proof','backup']


def test_wrapper_uses_storage_validator_for_plan_and_hashes_its_code(tmp_path,monkeypatch):
    root,lease,p=setup(tmp_path,monkeypatch);calls=[]
    def validate(repo,value,plan):calls.append((repo,value,plan))
    monkeypatch.setattr(cpu.storage,'validate_cpu_lease',validate)
    assert cpu.validate_plan(p,tmp_path)==(lease,root)
    assert calls==[(tmp_path.resolve(),lease,p)]
    assert 'gearshift/sparse_repair_storage.py' in cpu.IMPLEMENTATION
