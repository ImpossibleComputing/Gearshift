#!/usr/bin/env python3
"""Verify and copy the completed replication endpoint; never load or run a model."""
import argparse
import base64
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import coding_confirmation_dispatch as base
from scripts import coding_confirmation_secondary_generate as secondary
from scripts import coding_confirmation_regional_generation as public

SOURCE_ROOT = '/workspace/GearshiftConfirmation'
RESULT = base.RESULT + '/replication/arms'
DECLARATION = public.DECLARATION
DECLARATION_SHA = '16387d5361c795121929fd9244821f12ee005f9b36b2febddca5d9ef89f7536f'
TRAINING_COMMIT = '8b9c431f888db036daa4c2dd0337410cc8b5ceaa'
AREA = base.EVIDENCE / 'secondary_final_checkpoints'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return public.sha(path)


def source_metadata_names():
    return {RESULT + '/' + arm + '/' + name for arm in ['FIXED', 'ROTATING'] for name in
        ['checkpoints/step_1024/manifest.json', 'training_complete.json', 'checkpoint_manifest.json', 'training_steps.json']}


def local_metadata_path(source):
    if source.endswith('/checkpoints/step_1024/manifest.json'): return source
    return str(Path(source).parent / 'checkpoints/step_1024/export' / Path(source).name)


def metadata_names(): return {local_metadata_path(p) for p in source_metadata_names()}


def endpoint_metadata(pod):
    code = '''import base64,json,pathlib
root=pathlib.Path(ROOT);result=ROOT_REL;files={};missing=[]
for arm in ['FIXED','ROTATING']:
 folder=root/result/arm
 for rel in ['checkpoints/step_1024/manifest.json','training_complete.json','checkpoint_manifest.json','training_steps.json']:
  p=folder/rel
  if not p.is_file() or p.is_symlink():missing.append(str(p));continue
  files[str(p.relative_to(root))]=base64.b64encode(p.read_bytes()).decode()
print(json.dumps({'files':files,'missing':missing}))'''
    code = 'ROOT=' + repr(SOURCE_ROOT) + '\nROOT_REL=' + repr(RESULT) + '\n' + code
    r = subprocess.run(base.ssh_args(pod) + ['python3 -c ' + shlex.quote(code)], capture_output=True, text=True, timeout=80)
    if r.returncode: raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout)


def expected_weights(repo=ROOT):
    repo = Path(repo)
    d = secondary.primary.validate_declaration(repo, DECLARATION, DECLARATION_SHA)
    rows = []; identities = []
    for arm in ['FIXED', 'ROTATING']:
        relative = RESULT + '/' + arm + '/checkpoints/step_1024/manifest.json'
        cp = secondary.checkpoint(repo, d, arm, relative, verify_weights=False)
        exported = repo / RESULT / arm / 'checkpoints/step_1024/export'
        complete = read(exported / 'training_complete.json')
        if (complete.get('arm') != arm or complete.get('completed_updates') != 1024 or
                complete.get('target_updates') != 1024 or complete.get('scored_positions') != 32768 or
                complete.get('full_target_completed') is not True or complete.get('last_durable_checkpoint') != 1024 or
                complete.get('checkpoint_identity_sha256') != secondary.digest(cp['checkpoint_identity']) or
                complete.get('hidden_tests_loaded') is not False or complete.get('free_running_answers_generated') != 0):
            raise ValueError('Training completion receipt differs from final checkpoint')
        steps = read(exported / 'training_steps.json')
        if (not isinstance(steps, list) or [r.get('step') for r in steps] != list(range(1, 1025)) or
                any(r.get('arm') != arm for r in steps)):
            raise ValueError('Training log is not the complete ordered 1024-update arm')
        identities.append(cp['checkpoint_identity']['code_commit'])
        m = read(repo / relative)
        rows += [{'arm': arm, 'path': str(Path(relative).parent / name), 'bytes': item['bytes'],
                  'sha256': item['sha256']} for name, item in sorted(m['files'].items())]
    if identities != [TRAINING_COMMIT, TRAINING_COMMIT]: raise ValueError('Paired training source identities differ')
    return rows


def check_file(path, expected):
    path = Path(path)
    if (any(p.is_symlink() for p in [path, *path.parents]) or not path.is_file() or
            path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']):
        raise ValueError('Transferred checkpoint bytes differ: ' + str(path))


def fetch_weight(pod, row):
    target = secondary.destination(ROOT, row['path'])
    temporary = target.with_name(target.name + '.partial-' + row['sha256'])
    for item in [temporary, *temporary.parents]:
        if item.is_symlink(): raise ValueError('Linked checkpoint destination')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists(): check_file(target, row); return
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > row['bytes']: raise ValueError('Partial checkpoint exceeds expected size')
        if offset < row['bytes']:
            command = 'tail -c ' + shlex.quote('+' + str(offset + 1)) + ' ' + shlex.quote(SOURCE_ROOT + '/' + row['path'])
            with temporary.open('ab') as out:
                result = subprocess.run(base.ssh_args(pod) + [command], stdout=out, stderr=subprocess.PIPE, timeout=900)
                out.flush(); os.fsync(out.fileno())
            if result.returncode: raise RuntimeError('Checkpoint transfer interrupted; immutable partial preserved: ' + result.stderr.decode()[-1000:])
        check_file(temporary, row)
        # Hard-link publication cannot replace a concurrently created final path.
        os.link(temporary, target); temporary.unlink()
        fd = os.open(target.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def fetch(pod_id):
    pod = base.safe(base.api('pods/' + pod_id))
    recovery = [read(p) for p in base.EVIDENCE.glob('checkpoint_reader_ap*/lease.json')
                if read(p).get('pod_id') == pod_id]
    reader_valid = len(recovery) == 1 and recovery[0].get('purpose') == 'checkpoint_transfer_reader' and recovery[0].get('network_volume_id') == 'o0ndo35b3f'
    if reader_valid: base.verify_lease(recovery[0])
    if ((pod_id != 'rekbqruqox2bk9' and not reader_valid) or
            pod.get('mounts', {}).get('network') != [{'volumeId': 'o0ndo35b3f', 'path': '/workspace'}]):
        raise ValueError('Training source pod or protected public volume differs')
    metadata = endpoint_metadata(pod)
    if metadata['missing']: return {'ready': False, 'missing': metadata['missing']}
    if set(metadata['files']) != source_metadata_names(): raise ValueError('Endpoint metadata member set differs')
    expected_names = metadata_names()
    for name, encoded in metadata['files'].items():
        public.bind_file(ROOT / local_metadata_path(name), base64.b64decode(encoded, validate=True))
    rows = expected_weights()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda row: fetch_weight(pod, row), rows))
    for row in rows: check_file(ROOT / row['path'], row)
    value = {'experiment_id': base.EXPERIMENT, 'training_seed': secondary.SEED, 'endpoint': 1024,
        'source_pod_id': pod_id, 'source_volume_id': 'o0ndo35b3f', 'source_root': SOURCE_ROOT,
        'original_training_pod_id': 'rekbqruqox2bk9', 'recovery_reader_used': reader_valid,
        'local_root': str(ROOT), 'actual_mapper_and_full_bytes_verified': True, 'files': rows,
        'metadata': {name: {'bytes': (ROOT/name).stat().st_size, 'sha256': sha(ROOT/name)} for name in sorted(expected_names)},
        'metadata_source_paths': {local_metadata_path(name): name for name in sorted(source_metadata_names())},
        'no_training_or_generation_run': True}
    AREA.mkdir(parents=True, exist_ok=True); public.bind_json(AREA / 'verified_local.json', value)
    return {'ready': True, 'files_verified': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'receipt': str(AREA / 'verified_local.json')}


def upload(pod_id, shard):
    partition = read(ROOT / public.CONFIG / 'regional_partition.json')
    store = partition['shard_storage'][shard]; pod = base.safe(base.api('pods/' + pod_id))
    if pod.get('status') != 'RUNNING' or pod.get('mounts', {}).get('network') != [{'volumeId': store['network_volume_id'], 'path': '/workspace'}]:
        raise ValueError('Destination pod does not own the frozen public shard store')
    receipt = read(AREA / 'verified_local.json'); rows = expected_weights()
    if receipt['files'] != rows or receipt.get('actual_mapper_and_full_bytes_verified') is not True:
        raise ValueError('Local endpoint receipt differs')
    if set(receipt['metadata']) != metadata_names(): raise ValueError('Local metadata inventory differs')
    entries = rows + [{'path': p, 'bytes': r['bytes'], 'sha256': r['sha256']} for p, r in sorted(receipt['metadata'].items())]
    proofs = []
    code = '''import fcntl,hashlib,json,os,pathlib,sys,uuid
p=pathlib.Path(PATH)
for item in [p,*p.parents]:
 if item.is_symlink():raise ValueError('Linked checkpoint target')
p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.transfer-'+uuid.uuid4().hex)
h=hashlib.sha256();count=0
with temp.open('xb') as out:
 while True:
  chunk=sys.stdin.buffer.read(8*1024*1024)
  if not chunk:break
  count+=len(chunk)
  if count>SIZE:raise ValueError('Oversized checkpoint stream')
  h.update(chunk);out.write(chunk)
 out.flush();os.fsync(out.fileno())
if count!=SIZE or h.hexdigest()!=HASH:raise ValueError('Checkpoint stream differs')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
with p.with_name(p.name+'.transfer.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if p.exists():
  if p.stat().st_size!=SIZE or digest(p)!=HASH:raise ValueError('Existing checkpoint differs; overwrite forbidden')
  temp.unlink()
 else:os.rename(temp,p)
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 if digest(p)!=HASH:raise ValueError('Persisted checkpoint differs')
print(json.dumps({'path':str(p),'bytes':SIZE,'sha256':HASH,'persisted_bytes_verified':True}))'''
    for row in entries:
        check_file(ROOT / row['path'], row)
        path = public.REMOTE + '/' + row['path']
        source = 'PATH=' + repr(path) + '\nSIZE=' + repr(row['bytes']) + '\nHASH=' + repr(row['sha256']) + '\n' + code
        with (ROOT / row['path']).open('rb') as inp:
            result = subprocess.run(base.ssh_args(pod) + ['python3 -c ' + shlex.quote(source)], stdin=inp,
                                    capture_output=True, text=True, timeout=900)
        if result.returncode: raise RuntimeError('Remote endpoint transfer failed: ' + result.stderr[-2000:])
        proofs.append(json.loads(result.stdout))
    value = {'pod_id': pod_id, 'shard_id': shard, 'network_volume_id': store['network_volume_id'],
        'local_receipt_sha256': sha(AREA / 'verified_local.json'), 'observed_epoch': time.time(),
        'files': proofs, 'all_persisted_bytes_verified': True}
    path = AREA / ('verified_' + shard + '_' + str(time.time_ns()) + '.json'); public.bind_json(path, value)
    return {'saved': str(path), 'files_verified': len(proofs), 'shard_id': shard}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['fetch', 'upload']); p.add_argument('--pod-id', required=True)
    p.add_argument('--shard'); a = p.parse_args()
    print(json.dumps(fetch(a.pod_id) if a.action == 'fetch' else upload(a.pod_id, a.shard)))
