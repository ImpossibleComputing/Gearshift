#!/usr/bin/env python3
"""Recover the historical confirmation-test bytes on the dedicated CPU only.

This materializes a missing input, never new tests. Exact historical source,
membership, raw-file, selected-row, and final-file hashes are mandatory. No
candidate is read or executed and no private value is printed or exported.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import sha, write
from scripts.coding_prepare import decode_tests

EXPERIMENT = 'confirmation_01_20260919T094418Z'
EXPECTED = 'eb94b93805f158e5a75d2b16d142374fc19ce9b5912cc0ae9db53c394a2d39f5'


def main():
    if sys.platform != 'linux' or os.environ.get('RUNPOD_POD_ID') != 'r2bdwy2qnys52b':
        raise ValueError('Recovery requires the declared isolated scoring CPU')
    payload = json.load(sys.stdin)
    membership_bytes = payload['membership'].encode()
    gate = payload['gate']
    if hashlib.sha256(membership_bytes).hexdigest() != gate['identity']['membership_sha256']:
        raise ValueError('Historical membership bytes differ')
    if sha(ROOT/'scripts/coding_prepare.py') != gate['identity']['preparation_source_sha256']:
        raise ValueError('Historical preparation implementation differs')
    if gate['identity']['private_files']['confirmation'] != EXPECTED:
        raise ValueError('Historical private-file identity differs')
    if payload['verified_primary_closure_sha256'] != 'e9f439019ac632f22e8ac8b5eb3d9a222b2b1b3d7afc38abfd01579d2a810187':
        raise ValueError('Whole-primary closure must precede private materialization')
    membership = json.loads(membership_bytes)
    selected = membership['selected']['confirmation']
    wanted = {(r['platform'], str(r['question_id'])): r for r in selected}
    if len(wanted) != 200:
        raise ValueError('Historical confirmation population differs')
    raw_dir = ROOT/'data/coding_pilot_v1/private_recovery_raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {r['upstream_file'] for r in selected}
    def fetch(name):
        meta = membership['source_files'][name]
        path = raw_dir/name
        if not path.exists():
            partial = path.with_suffix('.download')
            if partial.exists():
                raise ValueError('Preserve interrupted download before bounded recovery')
            with urllib.request.urlopen(meta['url'], timeout=120) as source, partial.open('xb') as out:
                while chunk := source.read(8*1024**2): out.write(chunk)
                out.flush(); os.fsync(out.fileno())
            if partial.stat().st_size != meta['bytes'] or sha(partial) != meta['sha256']:
                raise ValueError('Frozen raw input bytes differ: '+name)
            partial.rename(path)
        if path.stat().st_size != meta['bytes'] or sha(path) != meta['sha256']:
            raise ValueError('Frozen raw input bytes differ: '+name)
        return name
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, sorted(files)))
    private = {}
    # Match the original preparation's file and row order exactly.
    for name in membership['source_files']:
        if name not in files: continue
        with (raw_dir/name).open('rb') as stream:
            for line in stream:
                row = json.loads(line)
                key = (row['platform'], str(row['question_id']))
                if key not in wanted: continue
                expected = wanted[key]
                variants = [line, line.rstrip(b'\r\n'), json.dumps(row, sort_keys=True).encode(),
                            json.dumps(row, sort_keys=True, separators=(',', ':')).encode()]
                if expected['source_row_sha256'] not in [hashlib.sha256(v).hexdigest() for v in variants]:
                    raise ValueError('Historical selected row differs')
                task = '/'.join(key)
                if task in private: raise ValueError('Duplicate historical task')
                tests = json.loads(row['public_test_cases'])+decode_tests(row['private_test_cases'])
                private[task] = {'fn_name': json.loads(row['metadata']).get('func_name'), 'tests': tests}
    if len(private) != 200: raise ValueError('Incomplete private input reconstruction')
    target = ROOT/'data/coding_pilot_v1/private/confirmation.json'
    recovered = target.with_name('confirmation.recovered.json')
    if target.exists() or recovered.exists(): raise ValueError('Recovery never overwrites an existing input')
    write(recovered, private)
    if sha(recovered) != EXPECTED:
        raise ValueError('Reconstructed bytes do not match the original frozen file; scoring forbidden')
    recovered.rename(target)
    receipt = {'epoch': time.time(), 'experiment_id': EXPERIMENT, 'pod_id': os.environ['RUNPOD_POD_ID'],
        'path': str(target), 'bytes': target.stat().st_size, 'sha256': EXPECTED, 'task_count': 200,
        'recovered_byte_for_byte_from_frozen_sources': True, 'candidate_execution': False,
        'private_values_in_receipt': False, 'primary_closure_sha256': payload['verified_primary_closure_sha256'],
        'membership_sha256': gate['identity']['membership_sha256'],
        'preparation_source_sha256': gate['identity']['preparation_source_sha256'],
        'raw_files': {name: membership['source_files'][name] for name in sorted(files)}}
    write(ROOT/'evidence/coding_pilot_v1'/EXPERIMENT/'resources/scoring_cpu_primary01/private_input_recovered.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'raw_files'}), flush=True)


if __name__ == '__main__': main()
