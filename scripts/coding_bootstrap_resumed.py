#!/usr/bin/env python3
"""Pod-side preflight bootstrap with pinned stack and local deadline."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/coding_pilot_v1'

def main():
    os.chdir(ROOT);os.environ.update(HF_HOME='/workspace/hf',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    deadline=json.loads((E/'allocation_resume_09.json').read_text())['preflight_deadline_epoch']-120
    py=str(ROOT/'.pilot-venv/bin/python')
    commands=[['apt-get','update'],['apt-get','install','-y','--no-install-recommends','python3','python3-venv','libseccomp2'],
        ['/usr/bin/python3','scripts/coding_sandbox_probe.py'],
        [sys.executable,'-m','venv','--system-site-packages','.pilot-venv'],
        [py,'-m','pip','install','transformers==4.57.6','accelerate==1.15.0','numpy==2.2.6','psutil==7.2.2','safetensors==0.7.0'],
        [py,'-c','import torch; assert torch.__version__.startswith("2.8.0"), torch.__version__; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; assert "H200" in torch.cuda.get_device_name(0)'],
        [py,'scripts/coding_download_models.py']]
    for command in commands:
        remaining=deadline-time.time()
        if remaining<=0:raise TimeoutError('Preflight budget exhausted during bootstrap')
        write(E/'worker_status.json',{'state':'bootstrapping','heartbeat_epoch':time.time(),'command':command})
        subprocess.run(command,check=True,timeout=min(remaining,1800))
    os.execv(py,[py,'scripts/coding_preflight_resumed.py'])

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(E/'worker_status.json',{'state':'bootstrap_failed','error':str(e),'epoch':time.time()});traceback.print_exc();sys.exit(1)
