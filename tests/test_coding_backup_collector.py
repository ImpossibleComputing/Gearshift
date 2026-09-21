import importlib.util,io,sys,tarfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from gearshift.coding_snapshot import stream
from gearshift.coding_control import write

def setup(tmp_path,monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('backup_collector_test',scripts/'coding_cloud_session.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    root=tmp_path/'studio';control=root/'evidence/coding_pilot_v1/control';control.mkdir(parents=True)
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'C',control);monkeypatch.setattr(m,'ssh_args',lambda conn:['ssh'])
    latest=root/'evidence/coding_pilot_v1/pod_backup/latest';latest.mkdir(parents=True);(latest/'stale.txt').write_text('older snapshot')
    return m,root,latest

def test_collector_replaces_snapshot_without_stale_entries(tmp_path,monkeypatch):
    m,root,latest=setup(tmp_path,monkeypatch);live=tmp_path/'pod'
    write(live/'evidence/coding_pilot_v1/status.json',{'step':8});b=io.BytesIO();stream(live,b)
    def fake(*args,**kw):kw['stdout'].write(b.getvalue());return SimpleNamespace(returncode=0,stderr=b'')
    monkeypatch.setattr(m.subprocess,'run',fake);m.collect({})
    assert not (latest/'stale.txt').exists()
    assert (latest/'evidence/coding_pilot_v1/status.json').exists()
    assert (latest.parent/'previous/stale.txt').read_text()=='older snapshot'
    assert (m.C/'backup_ack.json').exists()

def test_corrupt_download_cannot_replace_verified_snapshot(tmp_path,monkeypatch):
    m,root,latest=setup(tmp_path,monkeypatch)
    def fake(*args,**kw):kw['stdout'].write(b'broken transport');return SimpleNamespace(returncode=0,stderr=b'')
    monkeypatch.setattr(m.subprocess,'run',fake)
    with pytest.raises(tarfile.ReadError):m.collect({})
    assert (latest/'stale.txt').read_text()=='older snapshot'
    assert not (m.C/'backup_ack.json').exists()
