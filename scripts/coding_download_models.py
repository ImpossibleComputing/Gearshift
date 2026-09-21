#!/usr/bin/env python3
import json,sys,time,os
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT","30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT","60")
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha
from huggingface_hub import HfApi,snapshot_download,configure_http_backend
import requests
class BoundedSession(requests.Session):
    def request(self,*args,**kwargs):
        if kwargs.get('timeout') is None:kwargs['timeout']=(15,60)
        return super().request(*args,**kwargs)
configure_http_backend(backend_factory=BoundedSession)
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'configs/coding_pilot_v1/pilot.json').read_text());rows=[]
for role,spec in cfg['models'].items():
    def progress(stage,**extra):
        row={'epoch':time.time(),'role':role,'stage':stage,**extra};write(ROOT/'evidence/coding_pilot_v1/download_progress.json',row);print(json.dumps(row),flush=True)
    progress('metadata')
    info=HfApi(token=False).model_info(spec['id'],revision=spec['revision'],files_metadata=True,timeout=30)
    assert info.sha==spec['revision']
    pinned={s.rfilename:s.lfs.sha256 for s in info.siblings if s.rfilename.endswith('.safetensors') and s.lfs}
    write(ROOT/f'evidence/coding_pilot_v1/{role}_weight_pins.json',{'revision':info.sha,'sha256':pinned})
    progress('download')
    path=Path(snapshot_download(spec['id'],revision=spec['revision'],token=False,
        allow_patterns=['*.safetensors','*.json','*.model'],max_workers=4,etag_timeout=30))
    for name,expected in pinned.items():
        progress('verify_weight_shard',shard=name)
        observed=sha(path/name)
        if observed!=expected:raise ValueError('Model weight hash mismatch')
    rows.append({'role':role,'revision':info.sha,'shards_verified':len(pinned),'bytes':sum((path/n).stat().st_size for n in pinned)})
    write(ROOT/'evidence/coding_pilot_v1/weight_verification.json',{'models':rows,'epoch':time.time()})
    print(role,'weights verified',flush=True)
