import io,json,threading,time
from pathlib import Path
import pytest
from gearshift.coding_control import write,sha
from gearshift.coding_snapshot import capture,stream,verify_archive

def test_snapshot_survives_atomic_directory_churn(tmp_path):
    root=tmp_path/'live';e=root/'evidence/coding_pilot_v1';e.mkdir(parents=True)
    stop=threading.Event()
    def writer():
        i=0
        while not stop.is_set():write(e/'status.json',{'step':i});i+=1
    t=threading.Thread(target=writer);t.start()
    try:
        for i in range(8):
            b=io.BytesIO();stream(root,b);archive=tmp_path/f'{i}.tgz';archive.write_bytes(b.getvalue())
            manifest=verify_archive(archive,tmp_path/f'out{i}')
            for rel in manifest['files']:json.loads((tmp_path/f'out{i}'/rel).read_text())
    finally:stop.set();t.join()

def test_completed_task_hash_mismatch_is_rejected(tmp_path):
    root=tmp_path/'live';p=root/'results/coding_pilot_v1/p/tasks/t';p.mkdir(parents=True)
    write(p/'A.json',{'answer':'one'});h=sha(p/'A.json')
    write(p/'complete.json',{'files':{'A.json':h}})
    write(p/'A.json',{'answer':'two'})
    with pytest.raises(ValueError,match='transaction'):capture(root,tmp_path/'stage')

def test_verified_extraction_has_no_stale_previous_files(tmp_path):
    root=tmp_path/'live';p=root/'evidence/coding_pilot_v1';p.mkdir(parents=True);write(p/'new.json',{'v':1})
    b=io.BytesIO();stream(root,b);archive=tmp_path/'snapshot.tgz';archive.write_bytes(b.getvalue())
    verify_archive(archive,tmp_path/'new_snapshot')
    assert not (tmp_path/'new_snapshot/results/old.json').exists()
