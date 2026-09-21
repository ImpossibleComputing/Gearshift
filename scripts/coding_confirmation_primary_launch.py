#!/usr/bin/env python3
import sys,json,shlex,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import coding_confirmation_dispatch as d
from gearshift.coding_control import write,sha
name=sys.argv[1];folder=d.EVIDENCE/name;lease=json.loads((folder/'lease.json').read_text());pod=d.api('pods/'+lease['pod_id'])
plan=str((folder/'generation_plan.json').relative_to(d.ROOT))
code=r'''import json,pathlib,subprocess,sys,os,time
root=pathlib.Path('/workspace/GearshiftConfirmationPrimary');sys.path.insert(0,str(root));os.chdir(root)
from scripts.coding_confirmation_replication_supervisor import child_environment
from gearshift.coding_confirmation_lease import load_lease
from scripts.coding_confirmation_generate import validate_declaration
p=json.loads((root/PLAN).read_text());lease=load_lease(root/p['lease_path'],p['lease_sha256']);d=validate_declaration(root,p['declaration_path'],p['declaration_sha256'])
folder=root/p['result_root']/lease['control_relative'];folder.mkdir(parents=True,exist_ok=True)
receipt=folder/'generation_launch.json'
if receipt.exists():raise RuntimeError('Existing launch receipt; reconcile instead of duplicate launch')
from scripts.coding_confirmation_generation_supervisor import validate_guard
validate_guard(lease,root/p['result_root']/lease['control_relative'])
with (folder/'generation_launcher.log').open('ab') as log:
 proc=subprocess.Popen([str(root/'.pilot-venv/bin/python'),'scripts/coding_confirmation_generation_supervisor.py','--plan',PLAN],cwd=root,env=child_environment(lease,0),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
row={'epoch':time.time(),'pid':proc.pid,'pgid':os.getpgid(proc.pid),'pod_id':lease['pod_id'],'declaration_sha256':p['declaration_sha256'],'source_commit':p['code_commit'],'preferences':p['preferences'],'root':str(root),'primary_target_answers':4800,'secondary_training_not_waited_for':True}
receipt.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))'''
r=subprocess.run(d.ssh_args(pod)+['/workspace/GearshiftV2/.pilot-venv/bin/python -c '+shlex.quote('PLAN='+repr(plan)+'\n'+code)],capture_output=True,text=True,timeout=50)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
row=json.loads(r.stdout);write(folder/'generation_launch.json',row);print(json.dumps(row,indent=2))
