"""Owner-authorized cutoff removal preserves budget enforcement and live inference."""
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
auth = importlib.import_module('phase2_execution_authorization')
w = importlib.import_module('phase2_cloud_watchdog')
worker = importlib.import_module('phase2_cuda_worker')
collector = importlib.import_module('phase2_cloud_session')


def task_pod():
    return dict(id='task', name=w.PREFIX+'task', costPerHr=3.49, desiredStatus='RUNNING')


def test_original_authorization_remains_default_when_no_extension(tmp_path):
    original = auth.execution_authorization(tmp_path/'missing.json')
    assert original['shutdown_deadline_utc'] == auth.LEGACY_SHUTDOWN_UTC
    assert original['authorization_end_utc'] == auth.LEGACY_END_UTC


def test_extension_removes_time_limit_but_keeps_finite_spending_limits():
    assert math.isinf(auth.DEADLINE)
    assert auth.deadline_epoch() is None
    assert auth.remaining_timeout() is None
    ledger, targets, reasons = w.evaluate({}, [task_pod()], w.START+26*3600)
    assert not targets and not reasons
    assert 0 < ledger['estimated_upper_bound_usd'] < 900
    assert w.POLICY['hard_cap_usd'] == 1000
    assert w.POLICY['shutdown_threshold_usd'] == 900


def test_extension_still_enforces_budget_and_provider_failure():
    ledger, targets, reasons = w.evaluate({}, [task_pod()], w.START+260*3600)
    assert targets == ['task'] and 'budget_with_two_minute_margin' in reasons
    _, targets, reasons = w.evaluate(ledger, [], w.START+260*3600+30, list_ok=False)
    assert targets == ['task'] and 'provider_listing_failure' in reasons


@pytest.mark.parametrize('field,value', [('hard_cap_usd',2000),('shutdown_threshold_usd',1000),('max_allocated_hourly_usd',100),('owner_message',''),('authorization_type','guessed')])
def test_extension_refuses_changed_limits_or_missing_owner_instruction(tmp_path,field,value):
    data = dict(auth.AUTHORIZATION);data[field]=value
    p=tmp_path/'policy.json';p.write_text(json.dumps(data))
    with pytest.raises(ValueError):auth.execution_authorization(p)


def test_adopted_child_command_and_start_time_are_checked(monkeypatch):
    args=['scripts/phase2_confirmation.py','--config','configs/phase2_cuda.json']
    class Process:
        def create_time(self):return 123.0
        def status(self):return 'running'
        def cmdline(self):return ['python','scripts/phase2_cuda_entry.py',*args]
    monkeypatch.setattr(worker.psutil,'Process',lambda pid:Process())
    assert worker.adopted_process_alive({'child_pid':1,'child_create_time':123.0},args)
    with pytest.raises(RuntimeError,match='reused'):
        worker.adopted_process_alive({'child_pid':1,'child_create_time':122.0},args)
    with pytest.raises(RuntimeError,match='command mismatch'):
        worker.adopted_process_alive({'child_pid':1,'child_create_time':123.0},['unexpected.py'])


def test_adoption_preserves_child_and_requires_verified_completion(tmp_path,monkeypatch):
    args=['scripts/phase2_confirmation.py','--config','configs/phase2_cuda.json']
    record=dict(label='confirmation',command=args,child_pid=55,child_create_time=123.0)
    monkeypatch.setattr(worker,'STATE',tmp_path)
    alive=iter([True,False]);monkeypatch.setattr(worker,'adopted_process_alive',lambda *a:next(alive))
    monkeypatch.setattr(worker.time,'sleep',lambda x:None)
    proof=dict(all_transaction_hashes_verified=True,completion={'sha256':'verified'})
    monkeypatch.setattr(worker,'verify_adopted_completion',lambda label:proof)
    def forbidden(*a,**kw):raise AssertionError('The scientific child must not be spawned or signaled')
    monkeypatch.setattr(worker.subprocess,'Popen',forbidden);monkeypatch.setattr(worker.os,'killpg',forbidden)
    worker.wait_for_adopted_child('confirmation',args,record)
    result=json.loads((tmp_path/'confirmation.json').read_text())
    assert result['adopted_completion_verified'] is True
    assert result['observed_returncode'] is None and result['returncode'] is None
    assert result['completion_proof']==proof and worker.completed_receipt(result)
    assert not worker.completed_receipt({'returncode':None})


def test_adoption_refuses_missing_completion(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='without verified full completion'):
        worker.verify_adopted_completion('confirmation')


def test_collection_resume_never_uploads_or_restarts_existing_worker(tmp_path,monkeypatch):
    monkeypatch.setattr(collector,'ROOT',tmp_path);monkeypatch.setattr(collector,'CLOUD',tmp_path/'cloud')
    monkeypatch.chdir(tmp_path)
    status=tmp_path/'evidence/phase2/cuda_execution/status.json';status.parent.mkdir(parents=True);status.write_text(json.dumps({'state':'complete'}))
    monkeypatch.setattr(sys,'argv',['collector','task','--resume-existing'])
    provider_calls=[];commands=[]
    def provider(*args):
        provider_calls.append(args)
        if args[:2]==('pod','get'):return task_pod()
        if args[:2]==('ssh','info'):return dict(ip='127.0.0.1',port=22)
        assert args==('pod','delete','task');return None
    monkeypatch.setattr(collector,'cli',provider)
    monkeypatch.setattr(collector.subprocess,'Popen',lambda *a,**kw:None)
    def run(command,**kwargs):
        commands.append(command)
        last=command[-1]
        if last.startswith('test -'):return subprocess.CompletedProcess(command,1,'','')
        return subprocess.CompletedProcess(command,0,json.dumps({'state':'complete'}) if last.startswith('cat ') else '','')
    monkeypatch.setattr(collector.subprocess,'run',run)
    collector.main()
    assert not any('tar --' in str(c) or 'bootstrap.py' in str(c) for c in commands)
    receipt=json.loads((tmp_path/'cloud/task/collection_resume.json').read_text())
    assert receipt['uploaded'] is False and receipt['restarted_worker'] is False
    assert ('pod','delete','task') in provider_calls
