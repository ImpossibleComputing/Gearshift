#!/usr/bin/env python3
"""Deadline-bounded CUDA generation only; generated code is graded on the Studio."""
import argparse
import datetime as dt
import psutil
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from phase2_cloud_watchdog import DEADLINE, write
from phase2_execution_authorization import deadline_epoch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATE = ROOT / 'evidence/phase2/cuda_execution'
ADOPTION = None


def commands():
    cfg = ['--config', 'configs/phase2_cuda.json']
    return [
        ('unit_controls', ['-m', 'pytest', '-q', 'tests/test_core.py', 'tests/test_identity.py', 'tests/test_followup.py', 'tests/test_phase2.py']),
        ('cache_controls', ['scripts/phase2_controls.py', *cfg, '--attempt', 'controls_cuda']),
        ('gradient_eos', ['scripts/phase2_cuda_preflight.py']),
        ('characterization', ['scripts/phase2.py', 'characterization', *cfg, '--tasks', 'results/phase2_v1/tasks/characterization.json', '--conditions', 'results/phase2_v1/tasks/characterization_conditions.json']),
        ('training_preparation', ['scripts/phase2_train.py', 'prepare', *cfg]),
        ('training', ['scripts/phase2_train.py', 'train', *cfg]),
        ('confirmation', ['scripts/phase2_confirmation.py', *cfg]),
    ]


def completed_receipt(record):
    return record.get('returncode') == 0 or (record.get('returncode') is None and record.get('adopted_completion_verified') is True)


def verify_adopted_completion(label):
    if label != 'confirmation':
        raise ValueError('Only the current confirmation stage can be adopted')
    from phase2_studio_postprocess import collected
    from gearshift.phase2_io import artifact
    stage = ROOT / 'results/phase2_v1/confirmation_1p7_to_0p6'
    if not collected(stage):
        raise RuntimeError('Adopted child exited without verified full completion')
    return dict(stage=str(stage.relative_to(ROOT)), completion=artifact(stage / 'complete.json'), all_transaction_hashes_verified=True)


def adopted_process_alive(record, args):
    try:
        process = psutil.Process(record['child_pid'])
        if abs(process.create_time() - record['child_create_time']) > .001:
            raise RuntimeError('Adopted child PID was reused')
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        expected = ['scripts/phase2_cuda_entry.py', *args]
        if process.cmdline()[1:] != expected:
            raise RuntimeError('Adopted child command mismatch')
        return True
    except psutil.NoSuchProcess:
        return False


def wait_for_adopted_child(label, args, record):
    if label != 'confirmation' or record['label'] != label or record['command'] != args:
        raise ValueError('Adoption does not match the frozen running stage')
    while adopted_process_alive(record, args):
        write(STATE / 'status.json', dict(state='running', pid=os.getpid(), child_pid=record['child_pid'], label=label,
            heartbeat_epoch=time.time(), deadline_epoch=deadline_epoch(), adopted_running_child=True))
        if time.time() >= DEADLINE:
            raise RuntimeError('Execution authorization ended while adopting worker')
        time.sleep(10)
    proof = verify_adopted_completion(label)
    receipt = dict(command=args, returncode=None, observed_returncode=None, adopted_completion_verified=True,
        completion_proof=proof, start_epoch=record['child_create_time'], end_epoch=time.time(),
        adoption_reason='Owner removed time cutoff; supervisor replaced while the scientific child continued unchanged')
    write(STATE / (label + '.json'), receipt)



def run(label, args):
    stem = STATE / label
    if stem.with_suffix('.json').exists():
        previous = json.loads(stem.with_suffix('.json').read_text())
        if completed_receipt(previous):
            if previous.get('adopted_completion_verified'):
                verify_adopted_completion(label)
            return
        raise RuntimeError('Preserve failed attempt; explicit recovery required: ' + label)
    if time.time() >= DEADLINE:
        raise RuntimeError('Authorization shutdown deadline reached')
    if ADOPTION is not None and ADOPTION['label'] == label:
        wait_for_adopted_child(label, args, ADOPTION)
        return
    start = time.time()
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    with stem.with_suffix('.txt').open('w') as log:
        child = subprocess.Popen([sys.executable, 'scripts/phase2_cuda_entry.py', *args], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            write(STATE / 'status.json', dict(state='running', pid=os.getpid(), child_pid=child.pid, label=label, heartbeat_epoch=time.time(), deadline_epoch=deadline_epoch()))
            if time.time() >= DEADLINE:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    write(stem.with_suffix('.json'), dict(command=args, returncode=child.returncode, start_epoch=start, end_epoch=time.time()))
    if child.returncode:
        raise RuntimeError(label + ' failed')


def main():
    global ADOPTION
    parser = argparse.ArgumentParser()
    parser.add_argument('--adopt-running')
    options = parser.parse_args()
    if options.adopt_running:
        ADOPTION = json.loads(Path(options.adopt_running).read_text())
    os.chdir(ROOT)
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        write(STATE / 'commands.json', dict(commands=commands(), grading='Studio sandbox only; no generated candidate executed in this container'))
        for label, args in commands():
            if label == 'characterization':
                # Existing harness requires this sentinel. Bind it to the actual CUDA controls.
                for name in ['complete.json', 'gradient_eos.json']:
                    assert json.loads((ROOT / 'results/phase2_v1/controls_cuda' / name).read_text())['passed']
                from gearshift.phase2_io import artifact
                write(ROOT / 'results/phase2_v1/controls/complete.json', dict(passed=True, cuda_controls=artifact(ROOT / 'results/phase2_v1/controls_cuda/complete.json'), cuda_gradients=artifact(ROOT / 'results/phase2_v1/controls_cuda/gradient_eos.json')))
            run(label, args)
        # Selection is validation-only; replication uses the predeclared first seed.
        cfg = json.loads((ROOT / 'configs/phase2_cuda.json').read_text())
        selected = json.loads((ROOT / 'results/phase2_v1/training/1p7_to_0p6/selection.json').read_text())
        cfg.update(source='Qwen/Qwen3-4B', source_revision='1cfa9a7208912126459214e8b04321603b3df60c', training_objectives=[selected['selected_objective']])
        cfg['training']['seeds'] = [selected['representative_seed']]
        from gearshift.phase2_io import immutable
        immutable(ROOT / 'configs/phase2_4b_cuda.json', cfg)
        common = ['--config', 'configs/phase2_4b_cuda.json', '--pair', '4b_to_0p6']
        for label, args in [
            ('replication_controls', ['scripts/phase2_controls.py', '--config', 'configs/phase2_4b_cuda.json', '--attempt', 'controls_4b_cuda']),
            ('replication_preparation', ['scripts/phase2_train.py', 'prepare', *common]),
            ('replication_affine_fit', ['scripts/phase2_fit_replication.py', *common]),
            ('replication_training', ['scripts/phase2_train.py', 'train', *common, '--initialization', 'results/phase2_v1/training/4b_to_0p6/affine_initialization.pt']),
            ('replication_confirmation', ['scripts/phase2_confirmation.py', *common]),
        ]:
            run(label, args)
        write(STATE / 'status.json', dict(state='complete', finished_epoch=time.time(), pid=os.getpid()))
    except BaseException as exc:
        write(STATE / 'status.json', dict(state='failed', error=repr(exc), finished_epoch=time.time(), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
