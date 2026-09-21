#!/usr/bin/env python3
import sys, json, io, tarfile, subprocess, shlex, time, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import coding_confirmation_dispatch as dispatch
from scripts.coding_confirmation_generate import validate_declaration
from gearshift.coding_control import sha, write
ROOT=dispatch.ROOT; REMOTE='/workspace/GearshiftConfirmationPrimary'; CONFIG='configs/coding_pilot_v1/confirmation_01'
name=sys.argv[1]; prefs=sys.argv[2:]; folder=dispatch.EVIDENCE/name
lease=json.loads((folder/'lease.json').read_text()); pod=dispatch.api('pods/'+lease['pod_id'])
declpath=CONFIG+'/declaration.json'; decl=validate_declaration(ROOT,declpath,sha(ROOT/declpath))
plan={'experiment_id':dispatch.EXPERIMENT,'role':'primary_confirmation_generation','result_root':dispatch.RESULT,'code_commit':decl['source_commit'],'declaration_path':declpath,'declaration_sha256':sha(ROOT/declpath),'lease_path':str((folder/'lease.json').relative_to(ROOT)),'lease_sha256':sha(folder/'lease.json'),'preferences':prefs}
assert len(prefs)==lease['gpu_count']
planpath=folder/'generation_plan.json';write(planpath,plan)
paths={p for p in subprocess.check_output(['git','ls-files','gearshift/*.py','scripts/*.py'],cwd=ROOT,text=True).splitlines()}
paths.update(decl['inputs']);paths.update(decl['implementation']);paths.add(declpath)
paths.update(str(p.relative_to(ROOT)) for p in (ROOT/CONFIG).glob('*') if p.is_file())
paths.update([plan['lease_path'],str(planpath.relative_to(ROOT))])
for role in ('source','receiver'):paths.add(str((dispatch.EVIDENCE/'replication_pair'/(role+'_weight_pins.json')).relative_to(ROOT)))
paths.add(str((dispatch.EVIDENCE/'replication_pair/model_weights_verified.json').relative_to(ROOT)))
manifest={'experiment_id':dispatch.EXPERIMENT,'source_commit':decl['source_commit'],'declaration_sha256':sha(ROOT/declpath),'files':{p:{'sha256':sha(ROOT/p),'bytes':(ROOT/p).stat().st_size} for p in sorted(paths)}}
assert not any('private' in Path(p).parts for p in paths)
manifestpath=folder/'primary_stage_manifest.json';write(manifestpath,manifest)
packet=folder/'primary_stage.tar.gz'
with tarfile.open(packet,'w:gz') as out:
 for p in sorted(paths|{str(manifestpath.relative_to(ROOT))}):out.add(ROOT/p,arcname=p,recursive=False)
code=r'''import sys,tarfile,pathlib,hashlib,json,os,shutil,time
root=pathlib.Path(REMOTE)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as tar:
 for m in tar:
  p=pathlib.Path(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts
  target=root/p; data=tar.extractfile(m).read()
  if target.exists():assert target.read_bytes()==data, str(target)
  else:
   target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix(target.suffix+'.stage');temporary.write_bytes(data);os.replace(temporary,target)
manifest=json.loads((root/MANIFEST).read_text())
for rel,want in manifest['files'].items():
 p=root/rel;assert p.stat().st_size==want['bytes'] and sha(p)==want['sha256'],rel
venv=root/'.pilot-venv'
if not venv.exists():venv.symlink_to('/workspace/GearshiftV2/.pilot-venv',target_is_directory=True)
d=json.loads((root/DECLARATION).read_text()); copies=[]
for arm,cp in d['primary_checkpoints'].items():
 for field,hashfield in [('mapper_path','mapper_sha256'),('manifest_path','manifest_sha256')]:
  rel=cp[field];src=pathlib.Path('/workspace/GearshiftV2')/rel;dest=root/rel
  if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
  assert sha(dest)==cp[hashfield],rel
  copies.append({'path':str(dest),'sha256':cp[hashfield],'bytes':dest.stat().st_size})
assert not list((root/'data').glob('**/private/*'))
out={'verified_at_epoch':time.time(),'pod_id':POD,'compact_files':len(manifest['files']),'primary_checkpoints':copies,'source_commit':d['source_commit'],'model_cache':'/workspace/hf','shared_environment_unchanged':True,'private_tests_present':False}
(root/RECEIPT).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))'''
constants='\n'.join(k+'='+repr(v) for k,v in {'REMOTE':REMOTE,'MANIFEST':str(manifestpath.relative_to(ROOT)),'DECLARATION':declpath,'POD':lease['pod_id'],'RECEIPT':str((folder/'stage_verified.json').relative_to(ROOT))}.items())
with packet.open('rb') as stream:
 r=subprocess.run(dispatch.ssh_args(pod)+['python3 -c '+shlex.quote(constants+'\n'+code)],stdin=stream,capture_output=True,timeout=180)
if r.returncode:print(r.stderr.decode());raise SystemExit(r.returncode)
receipt=json.loads(r.stdout);write(folder/'stage_verified.json',receipt);print(json.dumps(receipt,indent=2))
