#!/usr/bin/env python3
"""Copy immutable public progress once per physical shard, without pausing jobs."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import coding_confirmation_dispatch as base
from scripts.coding_confirmation_regional_dispatch import inventory
from scripts.coding_confirmation_completed_primary import completed_primary
from scripts.coding_confirmation_replication_supervisor import verify_archive
from gearshift.coding_confirmation_lease import sha256_file

REMOTE = r'''
import io,json,os,pathlib,sys,tarfile,time
repo=pathlib.Path(REPO);sys.path.insert(0,str(repo))
from scripts.coding_confirmation_generation_supervisor import closed_files
from scripts.coding_confirmation_replication_supervisor import file_receipt,verify_archive
from gearshift.coding_confirmation_lease import sha256_file
root=repo/'results/coding_pilot_v1/confirmation_01_20260919T094418Z'
files=set(closed_files(root,root/'__no_control_for_progress_backup__','__no_worker__'))
# Runtime and operational admission proofs are immutable even while workers run.
for name in ['runtime.json','model_setup.json','operational_shard.json','operational_storage.json',
             'operational_secondary_shard.json','operational_secondary_storage.json','secondary_checkpoint_verification.json']:
    files.update((root/'workers').glob('**/'+name))
files.update((root/'sharding').glob('initial_dataset_verified.json'))
rows=[file_receipt(root,p) for p in sorted(files)]
dest=root/'progress_backups'/STAMP;dest.mkdir(parents=True,exist_ok=False)
archive_path=dest/'immutable_progress.tar.gz'
with tarfile.open(archive_path,'w:gz',compresslevel=1) as archive:
    for row in rows:archive.add(root/row['path'],arcname=row['path'],recursive=False)
    raw=json.dumps(rows,sort_keys=True).encode();info=tarfile.TarInfo('COMPACT_MANIFEST.json');info.size=len(raw)
    archive.addfile(info,io.BytesIO(raw))
with archive_path.open('rb') as handle:os.fsync(handle.fileno())
verify_archive(archive_path,rows)
print(json.dumps({'remote_path':str(archive_path),'bytes':archive_path.stat().st_size,
    'sha256':sha256_file(archive_path),'files':len(rows),'observed_epoch':time.time(),
    'scope':'Immutable committed public progress and admission proofs; active partials remain on protected volume.',
    'generation_seal':False,'candidate_execution':False,'private_tests_loaded':False}))
'''


def backup(sid, pod, folder, stamp, remote_repo='/workspace/GearshiftConfirmationPrimary'):
    command = 'python3 -c ' + shlex.quote('STAMP=' + repr(stamp) + '\nREPO=' + repr(remote_repo) + '\n' + REMOTE)
    result = subprocess.run(base.ssh_args(pod) + [command], check=True, capture_output=True, text=True, timeout=240)
    receipt = json.loads(result.stdout)
    target = folder / (sid + '.tar.gz')
    with target.open('xb') as out:
        subprocess.run(base.ssh_args(pod) + ['cat ' + shlex.quote(receipt['remote_path'])],
                       check=True, stdout=out, timeout=240)
    if target.stat().st_size != receipt['bytes'] or sha256_file(target) != receipt['sha256']:
        raise ValueError('Downloaded progress archive differs')
    with tarfile.open(target, 'r:gz') as archive:
        rows = json.load(archive.extractfile('COMPACT_MANIFEST.json'))
    verify_archive(target, rows)
    value = {**receipt, 'shard_id': sid, 'reader_pod_id': pod['id'],
             'provider_mounts': pod['mounts'], 'local_path': str(target), 'local_all_member_hashes_verified': True}
    base.atomic_json(folder / (sid + '.json'), value)
    return value


def main():
    partition = json.loads((ROOT / 'configs/coding_pilot_v1/confirmation_01/regional_partition.json').read_text())
    pods = [base.safe(base.api('pods/' + p['id'])) for p in inventory()
            if p.get('status') == 'RUNNING' and p.get('name', '').startswith('gearshift-confirmation-')]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(time.time_ns())
    folder = base.EVIDENCE / 'regional_progress_backups' / stamp
    folder.mkdir(parents=True, exist_ok=False)
    jobs = []; completed_receipts = []
    owner_path=ROOT/'configs/coding_pilot_v1/confirmation_01/secondary_fr_execution_owner_v1.json'
    owner=json.loads(owner_path.read_text()) if owner_path.exists() else None
    for sid, store in partition['shard_storage'].items():
        matches = [p for p in pods if p.get('mounts', {}).get('network') ==
                   [{'volumeId': store['network_volume_id'], 'path': '/workspace'}]]
        if not matches:
            # This store has no secondary output queue: its sole secondary owner
            # is backed up below. Never use a primary seal to hide missing
            # original-region secondary progress in other stores.
            if owner and owner.get('source_shard')==sid and owner.get('source_secondary_generation_authorized') is False:
                saved=completed_primary(sid,partition,sha256_file(ROOT/'configs/coding_pilot_v1/confirmation_01/regional_partition.json'),base.EVIDENCE)
                if saved is not None:
                    completed_receipts.append({**saved,'shard_id':sid,'bytes':saved['archive_bytes'],
                        'local_path':saved['archive_path'],'sha256':saved['archive_sha256'],
                        'backup_mode':'reverified_existing_completed_packet'});continue
            raise ValueError('No live reader for shard ' + sid)
        jobs.append((sid, matches[0], '/workspace/GearshiftConfirmationPrimary'))
    if owner_path.exists():
        owner=json.loads(owner_path.read_text())
        matches=[p for p in pods if p.get('mounts',{}).get('network')==[{'volumeId':owner['destination_volume_id'],'path':'/workspace'}]]
        if not matches:raise ValueError('No reader for delegated secondary result store')
        jobs.append(('fr_secondary_delegated',matches[0],owner['destination_repo']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(backup, sid, pod, folder, stamp, repo) for sid, pod, repo in jobs]
        receipts = [future.result() for future in futures] + completed_receipts
    base.atomic_json(folder / 'verified.json', {'experiment_id': base.EXPERIMENT, 'shards': receipts,
        'partition_sha256': sha256_file(ROOT / 'configs/coding_pilot_v1/confirmation_01/regional_partition.json'),
        'generation_seal': False})
    print(json.dumps({'saved': str(folder / 'verified.json'), 'files': sum(r['files'] for r in receipts),
                      'bytes': sum(r['bytes'] for r in receipts)}))


if __name__ == '__main__': main()
