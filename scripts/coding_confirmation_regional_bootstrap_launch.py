#!/usr/bin/env python3
"""Launch a bounded public downloader; generation workers remain offline."""
import argparse,json,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import coding_confirmation_dispatch as d
from gearshift.coding_control import sha

def launch(name,retry_failed=False):
 f=d.EVIDENCE/name;lease=json.loads((f/'lease.json').read_text());pod=d.api('pods/'+lease['pod_id'])
 code='''import json,os,sys,pathlib,subprocess,time
root=pathlib.Path('/workspace/GearshiftConfirmationPrimary');sys.path.insert(0,str(root));os.chdir(root)
from scripts.coding_confirmation_replication_supervisor import child_environment
from gearshift.coding_confirmation_lease import load_lease
lease=load_lease(root/LEASE,SHA);folder=pathlib.Path(lease['allowed_result_root'])/lease['control_relative']
receipt=folder/'bootstrap_launch.json';attempt=1
if receipt.exists():
 assert RETRY
 old=json.loads(receipt.read_text());attempt=old.get('attempt',1)+1;assert attempt<=3
 state=folder/'regional_bootstrap/status.json';assert state.exists() and json.loads(state.read_text())['state']=='failed'
 try:os.kill(old['pid'],0)
 except ProcessLookupError:pass
 else:raise RuntimeError('Prior bootstrap process still exists; reconcile before retry')
 archive=folder/('bootstrap_attempt_'+str(attempt-1));archive.mkdir()
 receipt.rename(archive/'bootstrap_launch.json');(folder/'regional_bootstrap').rename(archive/'regional_bootstrap')
 if (folder/'bootstrap_launcher.log').exists():(folder/'bootstrap_launcher.log').rename(archive/'bootstrap_launcher.log')
env=child_environment(lease,0);env['HF_HUB_OFFLINE']='0';env['TRANSFORMERS_OFFLINE']='0'
# Bootstrap checks the allocation inventory; scientific workers still select
# exactly one leased GPU using the unchanged child_environment function.
env['CUDA_VISIBLE_DEVICES']=','.join(str(i) for i in range(lease['gpu_count']))
with (folder/'bootstrap_launcher.log').open('ab') as log:
 p=subprocess.Popen(['python3','scripts/coding_confirmation_regional_bootstrap.py','--lease',LEASE,'--lease-sha256',SHA],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
row={'pid':p.pid,'epoch':time.time(),'pod_id':lease['pod_id'],'attempt':attempt,'purpose':'public_inputs_and_runtime_only','confirmation_started':False,'public_download_network_enabled':True,'generation_environment_unchanged':True};receipt.write_text(json.dumps(row,indent=2)+'\\n');print(json.dumps(row))
'''
 constants='LEASE='+repr(str((f/'lease.json').relative_to(ROOT)))+'\nSHA='+repr(sha(f/'lease.json'))+'\nRETRY='+repr(retry_failed)+'\n'
 r=subprocess.run(d.ssh_args(pod)+['python3 -c '+shlex.quote(constants+code)],text=True,capture_output=True,timeout=40)
 if r.returncode:raise RuntimeError(r.stderr[-2000:])
 row=json.loads(r.stdout);d.atomic_json(f/('bootstrap_launch_attempt_'+str(row['attempt'])+'.json'),row);print(json.dumps(row))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--retry-failed',action='store_true');a=p.parse_args();launch(a.name,a.retry_failed)
