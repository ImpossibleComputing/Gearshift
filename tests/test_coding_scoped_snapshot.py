import io
import json
from pathlib import Path

import pytest

from gearshift.coding_control import sha, write
from gearshift.coding_snapshot import capture, capture_lock, stream, verify_archive


def fixture(tmp_path, stage='cap_recovery'):
    root=tmp_path/'repo';worker='evidence/coding_pilot_v1/parallel/current/worker_00'
    result='results/coding_pilot_v1/current/worker_00'
    spec={'run_id':'current','worker_id':'worker_00','stage':stage,'stage_identity':'frozen',
        'worker_root':worker,'result_root':result}
    path=root/worker/'worker_spec.json';write(path,spec)
    write(root/worker/'bootstrap_status.json',{'state':'bootstrapping','command':'download'})
    (root/worker/'bootstrap.log').write_bytes(b'old'*40000+b'current download progress\n')
    write(root/'evidence/coding_pilot_v1/weight_verification.json',{'models':[]})
    parent=root/'results/coding_pilot_v1/inherited/tasks/old/invalid.json'
    parent.parent.mkdir(parents=True);parent.write_text('This old parent must never be recopied')
    write(root/result/'tasks/t/partial_source_reasoning.json',{'token_ids':[1,2,3]})
    return root,path,spec


def test_scoped_snapshot_preserves_current_outputs_without_reading_inherited_parents(tmp_path):
    root,path,spec=fixture(tmp_path)
    buffer=io.BytesIO();stream(root,buffer,worker_spec=path)
    archive=tmp_path/'scoped.tar.gz';archive.write_bytes(buffer.getvalue())
    manifest=verify_archive(archive,tmp_path/'verified')
    names=set(manifest['files'])
    assert not any('/inherited/' in name for name in names)
    assert spec['result_root']+'/tasks/t/partial_source_reasoning.json' in names
    assert 'evidence/coding_pilot_v1/weight_verification.json' in names
    tail=spec['worker_root']+'/bootstrap_tail.txt'
    assert manifest['files'][tail]['bytes']==100000
    assert (tmp_path/'verified'/tail).read_bytes().endswith(b'current download progress\n')
    assert manifest['scope']['worker_spec_sha256']==sha(path)
    assert manifest['scope']['stage_identity']=='frozen'


def test_scoped_transactions_still_reject_changed_committed_output(tmp_path):
    root,path,spec=fixture(tmp_path);folder=root/spec['result_root']/'tasks/t'
    write(folder/'A.json',{'text':'first'})
    write(folder/'complete.json',{'files':{'A.json':sha(folder/'A.json')}})
    write(folder/'A.json',{'text':'changed'})
    with pytest.raises(ValueError,match='transaction'):capture(root,tmp_path/'staged',worker_spec=path)


def test_memory_scope_includes_both_raw_probe_trees(tmp_path):
    root,path,spec=fixture(tmp_path,stage='memory')
    for name in ['memory_v2_both_gpu','memory_v2_source_cpu_diagnostic']:
        write(root/'results/coding_pilot_v1'/name/'memory_gate.json',{'passed':False,'raw_objectives':['ordinary','boundary']})
    files=capture(root,tmp_path/'staged',worker_spec=path)
    assert all(f'results/coding_pilot_v1/{name}/memory_gate.json' in files
        for name in ['memory_v2_both_gpu','memory_v2_source_cpu_diagnostic'])


def test_parallel_or_orphan_snapshot_cannot_start_a_second_capture(tmp_path):
    root,path,spec=fixture(tmp_path)
    with capture_lock(root):
        with pytest.raises(RuntimeError,match='overlapping'):stream(root,io.BytesIO(),worker_spec=path)
    stream(root,io.BytesIO(),worker_spec=path)


def test_snapshot_scope_cannot_be_changed_to_parent_or_other_worker(tmp_path):
    root,path,spec=fixture(tmp_path);spec['result_root']='results/coding_pilot_v1/inherited';write(path,spec)
    with pytest.raises(ValueError,match='roots differ'):capture(root,tmp_path/'staged',worker_spec=path)


def test_symlink_output_is_not_exfiltrated(tmp_path):
    root,path,spec=fixture(tmp_path)
    secret=tmp_path/'secret.json';secret.write_text('{"secret":true}')
    (root/spec['result_root']/'private.json').symlink_to(secret)
    files=capture(root,tmp_path/'staged',worker_spec=path)
    assert not any(name.endswith('/private.json') for name in files)


def test_collector_uses_remote_hard_deadline_and_checks_worker_identity(monkeypatch,tmp_path):
    from test_coding_parallel_session import m
    root,path,spec=fixture(tmp_path)
    buffer=io.BytesIO();stream(root,buffer,worker_spec=path)
    calls=[]
    def run(argv,**kwargs):
        calls.append((argv,kwargs['timeout']));kwargs['stdout'].write(buffer.getvalue())
        return type('Result',(),{'returncode':0,'stderr':b''})()
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'ssh_args',lambda conn:['ssh'])
    monkeypatch.setattr(m.subprocess,'run',run)
    dest=root/'backups';saved=m.collect({},dest,spec)
    assert (saved/spec['worker_root']/'bootstrap_tail.txt').exists()
    assert 'timeout --signal=TERM --kill-after=5s 150s' in calls[0][0][-1]
    assert '--worker-spec '+spec['worker_root']+'/worker_spec.json' in calls[0][0][-1]
    assert calls[0][1]==180
    old_sha=sha(dest/'latest.tar.gz')
    with pytest.raises(ValueError,match='scope differs'):
        m.collect({},dest,{**spec,'worker_id':'another'})
    assert sha(dest/'latest.tar.gz')==old_sha


def test_collector_rejects_changed_spec_even_if_scope_labels_match(monkeypatch,tmp_path):
    from test_coding_parallel_session import m
    root,path,spec=fixture(tmp_path)
    buffer=io.BytesIO();stream(root,buffer,worker_spec=path)
    write(path,{**spec,'changed_condition':'not allowed'})
    def run(argv,**kwargs):
        kwargs['stdout'].write(buffer.getvalue())
        return type('Result',(),{'returncode':0,'stderr':b''})()
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'ssh_args',lambda conn:['ssh'])
    monkeypatch.setattr(m.subprocess,'run',run)
    with pytest.raises(ValueError,match='frozen Studio'):
        m.collect({},root/'backups',spec)
