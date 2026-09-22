#!/usr/bin/env python3
"""Retrieve exact publicly pinned model dependencies into ignored storage. No inference."""
import argparse,concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def one(item):
 role,spec,file,storage=item
 dest=storage/'models'/role/file['rfilename'];dest.parent.mkdir(parents=True,exist_ok=True)
 expected=file['lfs']['sha256'];size=file['size']
 if dest.exists():
  if dest.stat().st_size!=size or sha(dest)!=expected:raise ValueError('Existing model shard differs')
  return {'role':role,'file':file['rfilename'],'sha256':expected,'bytes':size}
 url=f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{file['rfilename']}";tmp=dest.with_suffix('.partial')
 for attempt in range(3):
  try:
   with urllib.request.urlopen(url,timeout=180) as src,tmp.open('wb') as out:
    while chunk:=src.read(8*1024**2):out.write(chunk)
   if tmp.stat().st_size!=size or sha(tmp)!=expected:raise ValueError('Downloaded model shard integrity mismatch')
   tmp.replace(dest);print('verified',role,file['rfilename'],flush=True)
   return {'role':role,'file':file['rfilename'],'sha256':expected,'bytes':size}
  except (OSError,TimeoutError):
   if attempt==2:raise
   time.sleep(2**attempt)
def main():
 p=argparse.ArgumentParser();p.add_argument('--storage',type=Path,default=ROOT/'data/early_handoff_01');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 protocol=json.loads((ROOT/'configs/coding_pilot_v1/confirmation_01/protocol.json').read_text())['protocol'];jobs=[]
 for role,spec in protocol['models'].items():
  directory=a.storage/'models'/role;directory.mkdir(parents=True,exist_ok=True)
  for filename,meta in protocol['tokenizer_files'][role].items():
   dest=directory/filename
   if not dest.exists():dest.write_bytes(urllib.request.urlopen(meta['url'],timeout=120).read())
   if dest.stat().st_size!=meta['bytes'] or sha(dest)!=meta['sha256']:raise ValueError('Tokenizer/config mismatch')
  url=f"https://huggingface.co/api/models/{spec['id']}/revision/{spec['revision']}?blobs=true"
  meta=json.load(urllib.request.urlopen(url,timeout=120))
  if meta['sha']!=spec['revision']:raise ValueError('Model revision mismatch')
  files=[x for x in meta['siblings'] if x['rfilename'].endswith('.safetensors')]
  index=json.loads((directory/'model.safetensors.index.json').read_text())
  if {x['rfilename'] for x in files}!=set(index['weight_map'].values()):raise ValueError('Index/shard membership mismatch')
  jobs.extend((role,spec,x,a.storage) for x in files)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:verified=list(ex.map(one,jobs))
 out={'models':protocol['models'],'verified_shards':verified,'model_execution_performed':False}
 path=a.storage/'control/model_weights_verified.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
