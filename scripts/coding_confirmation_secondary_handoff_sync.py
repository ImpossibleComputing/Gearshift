#!/usr/bin/env python3
"""Copy committed France primary inputs to its sole delegated secondary queue.

Transport only. Existing frozen secondary generation and supervision are reused.
No candidate execution, source generation, mutable sampler transfer, or scoring.
"""
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import coding_confirmation_dispatch as b
from scripts.coding_confirmation_regional_dispatch import inventory
from scripts.coding_confirmation_replication_supervisor import verify_archive
OWNER='configs/coding_pilot_v1/confirmation_01/secondary_fr_execution_owner_v1.json'

EXPORT=r'''
import io,json,os,pathlib,sys,tarfile,time
repo=pathlib.Path('/workspace/GearshiftConfirmationPrimary');sys.path.insert(0,str(repo))
from scripts import coding_confirmation_generate as p
from scripts.coding_confirmation_replication_supervisor import file_receipt,verify_archive
from gearshift.coding_control import sha
def primary_inputs(c,tid):
 h,hs,folder=p.verify_history(c,tid,'source');row=p.verify_job_receipt(c,tid,'receiver')
 assert row is not None
 control=p.scoped_file(c['top'],row['control_path']);assert control.parent==(p.task_folder(c,tid)/'controls').resolve()
 p.validate_control_record(json.loads(control.read_text()),h,hs)
 paths=[folder/n for n in ['identity.json','resume.json','complete.json','completion_timing.json','source_history.json','history_ready.json']]
 paths += [control,c['top']/'primary/jobs'/('receiver_'+tid.replace('/','__'))/'complete.json']
 return {str(q.relative_to(c['top'])):sha(q) for q in paths}
owner=json.load(sys.stdin);top=pathlib.Path(owner['source_root'])
assert json.loads((top/'secondary/execution_owner_fr_to_usco_v1.json').read_text())==owner
assert not list((top/'secondary/claims').glob('*.owner.json'))
d=p.validate_declaration(repo,'configs/coding_pilot_v1/confirmation_01/declaration.json',owner['primary_declaration_sha256'])
c={'repo_root':repo,'top':top,'declaration':d,'declaration_sha256':owner['primary_declaration_sha256'],
   'visible':{r['task_id']:r for r in json.loads((repo/d['visible_path']).read_text())}}
files=set();tasks=[]
for tid in owner['task_ids']:
 row=p.verify_job_receipt(c,tid,'receiver')
 if row is None:continue
 files.update(top/name for name in primary_inputs(c,tid));tasks.append(tid)
 files.update(top/row[key] for key in ['runtime_path','setup_path'])
assert tasks
rows=[file_receipt(top,f) for f in sorted(files)]
dest=top/'secondary_input_exports'/STAMP;dest.mkdir(parents=True,exist_ok=False);archive=dest/'inputs.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for row in rows:t.add(top/row['path'],arcname=row['path'],recursive=False)
 raw=json.dumps(rows,sort_keys=True).encode();i=tarfile.TarInfo('COMPACT_MANIFEST.json');i.size=len(raw);t.addfile(i,io.BytesIO(raw))
with archive.open('rb') as f:os.fsync(f.fileno())
verify_archive(archive,rows)
print(json.dumps({'remote_path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(rows),
 'task_ids':tasks,'source_manifest_sha256':owner['primary_declaration_sha256'],'epoch':time.time(),
 'primary_candidates_copied':False,'exact_committed_histories_and_controls':True}))
'''

IMPORT=r'''
import json,os,pathlib,sys,tarfile,time
repo=pathlib.Path('/workspace/GearshiftConfirmationFranceSecondary');sys.path.insert(0,str(repo))
from scripts.coding_confirmation_replication_supervisor import verify_archive,copy_verified
from scripts import coding_confirmation_generate as p
from gearshift.coding_control import bind,sha
def primary_inputs(c,tid):
 h,hs,folder=p.verify_history(c,tid,'source');row=p.verify_job_receipt(c,tid,'receiver')
 assert row is not None
 control=p.scoped_file(c['top'],row['control_path']);assert control.parent==(p.task_folder(c,tid)/'controls').resolve()
 p.validate_control_record(json.loads(control.read_text()),h,hs)
owner=json.loads((repo/OWNER_PATH).read_text());top=pathlib.Path(owner['destination_root'])
incoming=top/'input_sync'/STAMP;incoming.mkdir(parents=True,exist_ok=False);archive=incoming/'inputs.tar.gz'
with archive.open('xb') as out:
 while True:
  chunk=sys.stdin.buffer.read(1024*1024)
  if not chunk:break
  out.write(chunk)
 out.flush();os.fsync(out.fileno())
assert archive.stat().st_size==SIZE and sha(archive)==SHA
with tarfile.open(archive,'r:gz') as t:rows=json.load(t.extractfile('COMPACT_MANIFEST.json'))
verify_archive(archive,rows)
allowed={t.replace('/','__') for t in owner['task_ids']};jobs=[];regular=[]
for row in rows:
 q=pathlib.PurePosixPath(row['path']);assert not q.is_absolute() and '..' not in q.parts
 parts=q.parts
 if parts[:2]==('primary','tasks'):assert parts[2] in allowed
 elif parts[:2]==('primary','jobs'):
  assert len(parts)==4 and parts[2].startswith('receiver_') and parts[2][9:] in allowed and parts[3]=='complete.json'
 elif parts[0]=='workers':assert parts[-1] in ['runtime.json','model_setup.json']
 else:raise ValueError('Input packet crossed allowed scope')
 (jobs if parts[:2]==('primary','jobs') else regular).append(row)
with tarfile.open(archive,'r:gz') as t:
 for row in regular+jobs:
  dest=top/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():
   assert dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256'];continue
  temp=incoming/('file_'+str(time.time_ns()))
  with temp.open('xb') as out:out.write(t.extractfile(row['path']).read());out.flush();os.fsync(out.fileno())
  assert temp.stat().st_size==row['bytes'] and sha(temp)==row['sha256']
  # Exclusive publication; original histories and completion records never overwritten.
  os.link(temp,dest);temp.unlink()
d=p.validate_declaration(repo,'configs/coding_pilot_v1/confirmation_01/declaration.json',owner['primary_declaration_sha256'])
c={'repo_root':repo,'top':top,'declaration':d,'declaration_sha256':owner['primary_declaration_sha256'],
   'visible':{r['task_id']:r for r in json.loads((repo/d['visible_path']).read_text())}}
tasks=[t for t in owner['task_ids'] if p.verify_job_receipt(c,t,'receiver') is not None]
for tid in tasks:primary_inputs(c,tid)
receipt={'epoch':time.time(),'task_ids':tasks,'task_count':len(tasks),'input_files':len(rows),
 'archive_sha256':sha(archive),'completion_markers_published_last':True,'all_histories_and_controls_revalidated':True}
bind(incoming/'verified.json',receipt);print(json.dumps(receipt))
'''


def run():
    owner=json.loads((ROOT/OWNER).read_text())
    assert owner['cohort']=='secondary' and owner['source_secondary_generation_authorized'] is False
    pods=[b.safe(b.api('pods/'+p['id'])) for p in inventory() if p.get('status')=='RUNNING' and p.get('name','').startswith('gearshift-confirmation-')]
    def reader(volume):
        return next(p for p in pods if p.get('mounts',{}).get('network')==[{'volumeId':volume,'path':'/workspace'}])
    source=reader(owner['source_volume_id']);target=reader(owner['destination_volume_id'])
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+str(time.time_ns())
    folder=b.EVIDENCE/'secondary_fr_input_sync'/stamp;folder.mkdir(parents=True)
    started=time.monotonic()
    r=subprocess.run(b.ssh_args(source)+['python3 -c '+shlex.quote('STAMP='+repr(stamp)+'\n'+EXPORT)],
        input=json.dumps(owner),capture_output=True,text=True,timeout=150)
    if r.returncode:raise RuntimeError(r.stderr[-2500:])
    exported=json.loads(r.stdout);b.atomic_json(folder/'export.json',exported)
    f=folder/'inputs.tar.gz'
    with f.open('xb') as out:subprocess.run(b.ssh_args(source)+['cat '+shlex.quote(exported['remote_path'])],stdout=out,check=True,timeout=150)
    assert b.sha256_file(f)==exported['sha256'] and f.stat().st_size==exported['bytes']
    with tarfile.open(f,'r:gz') as t:rows=json.load(t.extractfile('COMPACT_MANIFEST.json'))
    verify_archive(f,rows)
    constants='STAMP='+repr(stamp)+'\nOWNER_PATH='+repr(OWNER)+'\nSIZE='+repr(exported['bytes'])+'\nSHA='+repr(exported['sha256'])+'\n'
    with f.open('rb') as src:r=subprocess.run(b.ssh_args(target)+['python3 -c '+shlex.quote(constants+IMPORT)],stdin=src,capture_output=True,text=True,timeout=150)
    if r.returncode:raise RuntimeError(r.stderr[-2500:])
    imported=json.loads(r.stdout);b.atomic_json(folder/'import.json',imported)
    result={'source_pod':source['id'],'destination_reader_pod':target['id'],'owner_sha256':b.sha256_file(ROOT/OWNER),
        'elapsed_seconds':time.monotonic()-started,'bytes':exported['bytes'],'task_count':imported['task_count'],
        'source_local_destination_hashes_verified':True,'saved':str(folder),'private_tests_or_scores_read':False}
    b.atomic_json(folder/'verified.json',result);print(json.dumps(result))


if __name__=='__main__':run()
