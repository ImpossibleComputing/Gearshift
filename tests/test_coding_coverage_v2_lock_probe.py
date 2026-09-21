import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.coding_coverage_v2_lock_probe import probe


def test_separate_process_exclusion_release_and_same_inode(tmp_path):
    script = Path(__file__).resolve().parents[1]/'scripts/coding_coverage_v2_lock_probe.py'
    lock = tmp_path/'shared.lock'; holding = tmp_path/'holding.json'; release = tmp_path/'release'
    process = subprocess.Popen([sys.executable, str(script), '--mode','hold','--lock',str(lock),
        '--receipt',str(holding),'--release-file',str(release),'--ttl-seconds','8','--probe-id','test'],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic()+5
        while not holding.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert holding.exists() and json.loads(holding.read_text())['state'] == 'holding'
        inode = lock.stat().st_ino
        assert probe(lock, 'try', tmp_path/'blocked.json', expect='blocked') == 0
        assert json.loads((tmp_path/'blocked.json').read_text())['state'] == 'blocked'
        release.touch(); _, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        assert json.loads(holding.read_text())['state'] == 'released'
        assert probe(lock, 'try', tmp_path/'acquired.json', expect='acquired') == 0
        assert lock.stat().st_ino == inode
        assert probe(lock, 'try', tmp_path/'unexpected.json', expect='blocked') == 1
    finally:
        if process.poll() is None:
            process.kill(); process.wait()


def test_holder_terminates_at_bounded_ttl(tmp_path):
    started = time.monotonic()
    assert probe(tmp_path/'lock', 'hold', tmp_path/'receipt.json', ttl_seconds=.05) == 0
    receipt = json.loads((tmp_path/'receipt.json').read_text())
    assert receipt['state'] == 'released' and receipt['release_reason'] == 'ttl_expired'
    assert time.monotonic()-started < 1
