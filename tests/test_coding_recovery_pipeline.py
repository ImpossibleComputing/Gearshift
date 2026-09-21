import fcntl,sys,threading,time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from coding_recovery_pipeline import wait_controller_idle

def test_waits_for_prior_cleanup_without_removing_lock(tmp_path):
    path=tmp_path/'controller.lock';path.write_text('prior controller')
    with path.open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        release=threading.Thread(target=lambda:(time.sleep(.1),fcntl.flock(owner,fcntl.LOCK_UN)))
        release.start();wait_controller_idle(path,timeout=1);release.join()
    assert path.read_text()=='prior controller'

def test_live_controller_is_not_bypassed(tmp_path):
    path=tmp_path/'controller.lock'
    with path.open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(TimeoutError):wait_controller_idle(path,timeout=.03)
    wait_controller_idle(path,timeout=.1)
