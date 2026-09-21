#!/usr/bin/env python3
"""Grade fully collected CUDA stages on the Studio, then export blind packets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.phase2_io import artifact, digest, read, validate_transaction
from phase2_cloud_watchdog import DEADLINE, write
from phase2_execution_authorization import remaining_timeout

STAGES = ['characterization', 'confirmation_1p7_to_0p6', 'confirmation_4b_to_0p6']
STATE = ROOT / 'evidence/phase2/studio_transfer/postprocess'


def collected(stage):
    """A completion marker can arrive before its files during rsync."""
    if not (stage / 'complete.json').exists() or not (stage / 'manifest.json').exists():
        return False
    complete = read(stage / 'complete.json')
    manifest = read(stage / 'manifest.json')
    identity = manifest['identity']
    if digest(identity) != manifest['identity_sha256'] or complete['identity_sha256'] != manifest['identity_sha256']:
        raise ValueError('Collected stage identity mismatch')
    if complete['questions'] != len(identity['tasks']) or len(complete['files']) != len(identity['tasks']):
        raise ValueError('Completion marker does not cover the frozen task sample')
    for name, expected in complete['files'].items():
        path = stage / name
        if not path.is_relative_to(stage) or '..' in Path(name).parts or Path(name).is_absolute():
            raise ValueError('Invalid completion path')
        if not path.exists() or artifact(path) != expected:
            return False
    objects = [read(stage / name) for name in complete['files']]
    if {obj['task_id'] for obj in objects} != {task['task_id'] for task in identity['tasks']}:
        raise ValueError('Collected task membership mismatch')
    for obj in objects:
        conditions = identity['conditions']
        if isinstance(conditions, dict):
            conditions = conditions[obj['task_id']]
        validate_transaction(obj, manifest['identity_sha256'], obj['task_id'], conditions)
    return True


def main():
    os.chdir(ROOT)
    STATE.mkdir(parents=True, exist_ok=True)
    try:
        for stage in STAGES:
            folder = ROOT / 'results/phase2_v1' / stage
            tasks = ROOT / 'results/phase2_v1/tasks/characterization.json' if stage == 'characterization' else folder / 'tasks.json'
            while time.time() < DEADLINE:
                write(STATE / 'status.json', dict(state='waiting_for_verified_collection', stage=stage, pid=os.getpid(), heartbeat_epoch=time.time()))
                if tasks.exists() and collected(folder):
                    break
                time.sleep(15)
            else:
                write(STATE / 'status.json', dict(state='deadline', stage=stage)); return
            write(STATE / (stage + '_collection.json'), dict(verified_epoch=time.time(), completion=artifact(folder / 'complete.json'), tasks=artifact(tasks), all_question_hashes_verified=True))
            commands = [
                ['scripts/phase2_score.py', stage, '--tasks', str(tasks)],
                ['scripts/phase2_judge.py', 'export', '--stage', stage, '--tasks', str(tasks)],
            ]
            if stage == 'characterization':
                commands[-1].append('--secondary')
            for index, command in enumerate(commands):
                receipt = STATE / f'{stage}_{index}.json'
                if receipt.exists():
                    if read(receipt)['returncode'] == 0:
                        continue
                    raise RuntimeError('Prior failed postprocessing attempt requires explicit recovery')
                write(STATE / 'status.json', dict(state='running', stage=stage, command=command, pid=os.getpid(), heartbeat_epoch=time.time()))
                with (STATE / f'{stage}_{index}.txt').open('w') as output:
                    result = subprocess.run([sys.executable, *command], stdout=output, stderr=subprocess.STDOUT, timeout=remaining_timeout())
                write(receipt, dict(command=command, returncode=result.returncode, finished_epoch=time.time()))
                if result.returncode:
                    raise RuntimeError('Studio postprocessing failed: ' + stage)
        write(STATE / 'status.json', dict(state='complete', finished_epoch=time.time()))
    except BaseException as exc:
        write(STATE / 'status.json', dict(state='failed', error=repr(exc), finished_epoch=time.time()))
        raise


if __name__ == '__main__':
    main()
