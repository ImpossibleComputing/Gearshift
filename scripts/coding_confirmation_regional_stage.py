#!/usr/bin/env python3
"""Stage public frozen inputs on a new region, without launching confirmation."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import coding_confirmation_dispatch as d
from scripts.coding_confirmation_generate import validate_declaration
from gearshift.coding_control import sha,write
REMOTE='/workspace/GearshiftConfirmationPrimary';CONFIG='configs/coding_pilot_v1/confirmation_01'

def stage(name):
    folder=d.EVIDENCE/name;lease=json.loads((folder/'lease.json').read_text());pod=d.api('pods/'+lease['pod_id'])
    declpath=CONFIG+'/declaration.json';decl=validate_declaration(ROOT,declpath,sha(ROOT/declpath))
    paths=set(subprocess.check_output(['git','ls-files','gearshift/*.py','scripts/*.py'],cwd=ROOT,text=True).splitlines())
    paths.update(decl['inputs']);paths.update(decl['implementation']);paths.add(declpath)
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/CONFIG).glob('*') if p.is_file())
    paths.add(str((folder/'lease.json').relative_to(ROOT)))
    paths.add('scripts/coding_confirmation_regional_bootstrap.py')
    paths.add('evidence/coding_pilot_v1/'+d.EXPERIMENT+'/resources/regional_environment_source.json')
    for role in ('source','receiver'):paths.add(str((d.EVIDENCE/'replication_pair'/(role+'_weight_pins.json')).relative_to(ROOT)))
    if any('private' in Path(p).parts or Path(p).suffix in ('.pt','.safetensors') for p in paths):raise ValueError('Nonpublic/weight staging entry')
    manifest={'experiment_id':d.EXPERIMENT,'source_commit':decl['source_commit'],'declaration_sha256':sha(ROOT/declpath),'files':{p:{'sha256':sha(ROOT/p),'bytes':(ROOT/p).stat().st_size} for p in sorted(paths)}}
    mp=folder/'regional_stage_manifest.json';write(mp,manifest);packet=folder/'regional_stage.tar.gz'
    with tarfile.open(packet,'w:gz') as out:
        for p in sorted(paths|{str(mp.relative_to(ROOT))}):out.add(ROOT/p,arcname=p,recursive=False)
    code='''import sys,tarfile,pathlib,hashlib,json,os,time
root=pathlib.Path(REMOTE)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as tar:
 for m in tar:
  p=pathlib.Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts
  target=root/p;data=tar.extractfile(m).read()
  if target.exists():assert target.is_file() and not target.is_symlink() and target.read_bytes()==data,str(target)
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
m=json.loads((root/MANIFEST).read_text())
for rel,want in m['files'].items():
 p=root/rel;assert p.stat().st_size==want['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==want['sha256'],rel
assert not list(pathlib.Path('/workspace').glob('**/private/*'))
print(json.dumps({'verified_at_epoch':time.time(),'files_verified':len(m['files']),'source_commit':m['source_commit'],'private_test_files_present':False,'confirmation_generation_started':False}))
'''
    constants='REMOTE='+repr(REMOTE)+'\nMANIFEST='+repr(str(mp.relative_to(ROOT)))+'\n'
    with packet.open('rb') as stream:r=subprocess.run(d.ssh_args(pod)+['python3 -c '+shlex.quote(constants+code)],stdin=stream,capture_output=True,timeout=90)
    if r.returncode:raise RuntimeError(r.stderr.decode()[-2000:])
    value=json.loads(r.stdout);write(folder/'regional_stage_verified.json',value);print(json.dumps(value))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');stage(p.parse_args().name)
