#!/usr/bin/env python3
"""Fresh normalized-depth affine fitting on new source-specific training histories."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.linalg
import torch
from gearshift.phase2_io import read,write,bind,artifact,digest,snapshot,runtime_identity
from gearshift.phase2_inference import backends
from gearshift.core import CacheExtractor,save_checkpoint
from gearshift.identity import backend_identity


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase2_4b.json');p.add_argument('--pair',default='4b_to_0p6');args=p.parse_args()
    cfg=read(args.config);source,target=backends(cfg);root=Path(cfg['output'])/'training'/args.pair
    prep=root/'preparation';complete=read(prep/'complete.json')
    objects=[read(prep/n) for n in sorted(complete['cases']) if read(prep/n)['task']['split']=='train']
    rng=np.random.default_rng(cfg['seed']);selections={}
    for o in objects:
        n=len(o['trajectory']['prompt_ids'])+len(o['trajectory']['source_reasoning_ids'])
        selections[o['task']['task_id']]=np.sort(rng.choice(n,min(64,n),replace=False)).tolist()
    layers=[round(t*(source.config.num_hidden_layers-1)/max(target.config.num_hidden_layers-1,1)) for t in range(target.config.num_hidden_layers)]
    identity=dict(source=backend_identity(source),target=backend_identity(target),runtime=runtime_identity(),preparation=artifact(prep/'complete.json'),
        positions=selections,source_layers_by_target=layers,ridge_penalty=0.01,standardize=True,
        code=snapshot(['scripts/phase2_fit_replication.py','gearshift/core.py','gearshift/phase2_inference.py','gearshift/phase2_io.py','requirements.lock.txt']))
    bind(root/'affine_fit_manifest.json',identity)
    if (root/'affine_fit_complete.json').exists():
        if artifact(root/'affine_initialization.pt')!=read(root/'affine_fit_complete.json')['checkpoint']:raise ValueError('Fitted checkpoint changed')
        return
    stores={role:{(layer,kind):[] for layer in (sorted(set(layers)) if role=='source' else range(target.config.num_hidden_layers)) for kind in ['k','v']}
            for role in ['source','target']}
    for i,o in enumerate(objects):
        history=o['trajectory']['prompt_ids']+o['trajectory']['source_reasoning_ids'];ix=selections[o['task']['task_id']]
        for role,b in [('source',source),('target',target)]:
            pairs=CacheExtractor.tensors(b.prefill(history).past_key_values)
            for layer,kind in stores[role]:
                x=pairs[layer][0 if kind=='k' else 1]
                if kind=='k':x=b.rope(x,inverse=True)
                stores[role][(layer,kind)].append(CacheExtractor.flatten(x)[ix].float().cpu())
        print(f'fit extraction {i+1}/{len(objects)}',flush=True)
    arrays={role:{key:torch.cat(values).numpy() for key,values in store.items()} for role,store in stores.items()};del stores
    maps={'functional_k':[],'functional_v':[]};metrics=[]
    for tl,sl in enumerate(layers):
        for kind in ['k','v']:
            x=arrays['source'][sl,kind].astype(np.float64);y=arrays['target'][tl,kind].astype(np.float64)
            xm=x.mean(0);ym=y.mean(0);x-=xm;y-=ym;scale=np.maximum(x.std(0),1e-4);x/=scale
            gram=x.T@x/len(x);cross=x.T@y/len(x)
            w=scipy.linalg.solve(gram+.01*np.eye(len(gram)),cross,assume_a='pos')/scale[:,None];bias=ym-xm@w
            pred=(x*scale)@w;nmse=float(np.mean((pred-y)**2)/max(np.mean(y**2),1e-12))
            maps['functional_'+kind].append(dict(sources=[sl],target=tl,kind=kind,content=kind=='k',penalty=.01,
                weight=torch.from_numpy(w.astype(np.float32)),bias=torch.from_numpy(bias.astype(np.float32))))
            metrics.append(dict(target_layer=tl,source_layer=sl,kind=kind,positions=len(x),training_normalized_mse=nmse))
            print(f'fresh affine {kind} {sl}->{tl}: training NMSE {nmse:.4f}',flush=True)
    save_checkpoint(root/'affine_initialization.pt',maps)
    write(root/'affine_fit_complete.json',dict(identity_sha256=digest(identity),checkpoint=artifact(root/'affine_initialization.pt'),metrics=metrics))


if __name__=='__main__':main()
