#!/usr/bin/env python3
"""Bounded public model/environment bootstrap; never samples a benchmark answer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gearshift.coding_confirmation_lease import load_lease,atomic_json
from gearshift.coding_control import sha

def validate_visible_gpus(runtime, lease):
    names = runtime.get('gpu_names')
    count = lease.get('gpu_count')
    if (type(count) is not int or count < 1 or not isinstance(names, list)
            or len(names) != count or any(not isinstance(n, str) or n != 'NVIDIA H200' for n in names)):
        raise ValueError('Visible H200 count differs from the allocation lease')

def main():
    p=argparse.ArgumentParser();p.add_argument('--lease',required=True);p.add_argument('--lease-sha256',required=True);a=p.parse_args()
    lease=load_lease(ROOT/a.lease,a.lease_sha256)
    assert os.environ.get('RUNPOD_POD_ID')==lease['pod_id']
    control=Path(lease['allowed_result_root'])/lease['control_relative']/'regional_bootstrap';control.mkdir(parents=True,exist_ok=False)
    start=time.time()
    def status(state,**more):atomic_json(control/'status.json',{'state':state,'epoch':time.time(),'started_epoch':start,**more})
    def guard():
        if time.time()>min(lease['deadline_epoch']-120,start+3600):raise TimeoutError('Bounded bootstrap allowance exceeded')
    try:
        status('environment_install')
        venv=ROOT/'.pilot-venv'
        if not venv.exists():subprocess.run(['python3','-m','venv','--system-site-packages',str(venv)],check=True,timeout=90)
        python=str(venv/'bin/python')
        pinned=['transformers==4.57.6','accelerate==1.15.0','numpy==2.2.6','safetensors==0.7.0','psutil==7.2.2','huggingface_hub==0.36.2','tokenizers==0.22.2','regex==2026.9.10','fsspec==2024.6.1','tqdm==4.70.1']
        subprocess.run([python,'-m','pip','install','--disable-pip-version-check',*pinned],check=True,timeout=600)
        guard()
        check="import torch,transformers,numpy,json;assert torch.__version__=='2.8.0+cu128';assert transformers.__version__=='4.57.6';assert numpy.__version__=='2.2.6';print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
        runtime=json.loads(subprocess.check_output([python,'-c',check],text=True,timeout=60));validate_visible_gpus(runtime,lease)
        atomic_json(control/'runtime.json',runtime)
        status('public_model_download')
        models=json.loads((ROOT/'configs/coding_pilot_v1/pilot.json').read_text())['models']
        download="import json,sys;from huggingface_hub import snapshot_download;models=json.loads(sys.argv[1]);[snapshot_download(x['id'],revision=x['revision'],cache_dir='/workspace/hf/hub',allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'],max_workers=4) for x in models.values()]"
        subprocess.run([python,'-c',download,json.dumps(models)],check=True,timeout=2400)
        guard();status('weight_hash_verification');checked=[]
        for role,spec in models.items():
            pins=json.loads((ROOT/'evidence/coding_pilot_v1'/lease['experiment_id']/'resources/replication_pair'/f'{role}_weight_pins.json').read_text())
            assert pins['revision']==spec['revision']
            snapshot=Path('/workspace/hf/hub')/('models--'+spec['id'].replace('/','--'))/'snapshots'/spec['revision']
            for name,want in pins['sha256'].items():
                guard();f=snapshot/name;h=hashlib.sha256()
                with f.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
                assert h.hexdigest()==want,str(f)
                checked.append({'path':str(f),'sha256':want,'bytes':f.stat().st_size})
        atomic_json(control/'model_weights_verified.json',{'files':checked,'all_verified':True,'epoch':time.time()})
        status('complete',weight_files=len(checked),elapsed_seconds=time.time()-start,confirmation_generation_started=False)
    except BaseException as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc));raise
if __name__=='__main__':main()
