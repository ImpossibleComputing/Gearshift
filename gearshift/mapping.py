from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

from .core import CacheExtractor, CacheInjector, save_json, save_checkpoint, tensor_metrics, sync
from .data import read_features


class CacheAdapter:
    """Separate affine K/V maps; the target model never receives historical tokens."""
    def __init__(self, source, target, maps, variant='content'):
        self.source,self.target,self.maps,self.variant=source,target,maps,variant
        self.weights={}
        for name, entries in maps.items():
            self.weights[name]=[{k:(v.to(target.device) if isinstance(v,torch.Tensor) else v)
                                 for k,v in m.items()} for m in entries]

    @classmethod
    def load(cls,cfg,source,target,variant='content'):
        return cls(source,target,torch.load(Path(cfg['output'])/'mappers.pt',weights_only=True),variant)

    def transform(self, source_pairs, variant=None):
        variant=variant or self.variant
        if variant in ['naive','zero','random']:
            return self.broken(source_pairs,variant)
        content=variant in ['content','normalized_content','neighbor','lowrank','mlp','weighted','functional']
        prefix='normalized_' if variant.startswith('normalized_') else ''
        kname=prefix+('k_content' if content else 'k_post')
        vname=prefix+'v'
        if variant in ['neighbor','lowrank','mlp','weighted','functional']:
            kname,vname=variant+'_k',variant+'_v'
        pairs=[]
        c=self.target.config
        for layer in range(c.num_hidden_layers):
            mapped=[]
            for kind,name in [(0,kname),(1,vname)]:
                m=self.weights[name][layer]
                features=[]
                for src in m['sources']:
                    x=source_pairs[src][kind]
                    if kind==0 and content:
                        x=self.source.rope(x,inverse=True)
                    features.append(CacheExtractor.flatten(x).float())
                x=torch.cat(features,-1) if len(features)>1 else features[0]
                if 'down' in m:
                    y=(x @ m['down'].to(x.device)) @ m['up'].to(x.device)+m['bias']
                else:
                    y=x @ m['weight']+m['bias']
                if 'mlp_w1' in m:
                    z=torch.nn.functional.gelu((x-m['mean'].to(x.device)) @ m['mlp_w1'].to(x.device)+m['mlp_b1'].to(x.device))
                    y=y+z @ m['mlp_w2'].to(x.device)+m['mlp_b2'].to(x.device)
                y=CacheExtractor.unflatten(y,c.num_key_value_heads,c.head_dim)
                if kind==0 and content:
                    y=self.target.rope(y)
                mapped.append(y.to(self.target.dtype))
            pairs.append(tuple(mapped))
        return tuple(pairs)

    def inject(self,source_pairs,variant=None):
        return CacheInjector.create(self.transform(source_pairs,variant),clone=False)

    def broken(self,source_pairs,variant):
        c=self.target.config
        out=[]
        for i in range(c.num_hidden_layers):
            j=round(i*(len(source_pairs)-1)/max(c.num_hidden_layers-1,1))
            pair=[]
            for x in source_pairs[j]:
                flat=CacheExtractor.flatten(x)
                d=c.num_key_value_heads*c.head_dim
                if flat.shape[-1]<d:
                    flat=torch.nn.functional.pad(flat,(0,d-flat.shape[-1]))
                y=CacheExtractor.unflatten(flat[:,:d],c.num_key_value_heads,c.head_dim).to(self.target.dtype)
                if variant=='zero': y=torch.zeros_like(y)
                elif variant=='random': y=torch.randn_like(y)*y.float().std().to(y.dtype)
                pair.append(y)
            out.append(tuple(pair))
        return tuple(out)


def alignment_screen(cfg,source,target,root,kind,content):
    """All-pairs ridge screening using one fixed 128-d random projection per source.

    Screening is deliberately inexpensive; finalists are subsequently refit at full rank.
    No test observations are used to choose layers or regularization.
    """
    tag='k_content' if content else ('k_post' if kind=='k' else 'v')
    dest=Path(cfg['output'])/f'alignment_{tag}.json'
    if dest.exists(): return json.loads(dest.read_text())
    rng=np.random.default_rng(cfg['seed'])
    ti=np.sort(rng.choice(np.load(root/'train_tokens.npy').size,cfg['alignment_tokens'],replace=False))
    vi=np.sort(rng.choice(np.load(root/'validation_tokens.npy').size,2048,replace=False))
    device=source.device
    ys=torch.cat([read_features(root,'train','target',i,kind,ti,target,content)
                  for i in range(target.config.num_hidden_layers)],dim=1).to(device)
    yv=torch.cat([read_features(root,'validation','target',i,kind,vi,target,content)
                  for i in range(target.config.num_hidden_layers)],dim=1).to(device)
    ym=ys.mean(0)
    yc=ys-ym
    d=target.config.num_key_value_heads*target.config.head_dim
    rows=[]
    for sl in range(source.config.num_hidden_layers):
        x=read_features(root,'train','source',sl,kind,ti,source,content).to(device)
        xv=read_features(root,'validation','source',sl,kind,vi,source,content).to(device)
        projection=torch.from_numpy((rng.normal(size=(x.shape[1],cfg['alignment_rank']))/np.sqrt(x.shape[1])).astype('float32')).to(device)
        x,xv=x@projection,xv@projection
        xm=x.mean(0)
        xc=x-xm
        gram=(xc.T@xc/len(x)).cpu().double().numpy()
        cross=(xc.T@yc/len(x)).cpu().double().numpy()
        penalty=0.01*np.trace(gram)/len(gram)
        w=torch.from_numpy(scipy.linalg.solve(gram+penalty*np.eye(len(gram)),cross,assume_a='pos').astype('float32')).to(device)
        pred=(xv-xm)@w+ym
        row=[tensor_metrics(yv[:,t*d:(t+1)*d],pred[:,t*d:(t+1)*d]) for t in range(target.config.num_hidden_layers)]
        rows.append(row)
        print(f'alignment {tag}: source {sl+1}/{source.config.num_hidden_layers}',flush=True)
    r2=np.array([[m['r2'] for m in row] for row in rows])
    chosen=r2.argmax(0).tolist()
    result=dict(tag=tag,method='ridge after seeded Gaussian projection, rank '+str(cfg['alignment_rank']),
                train_positions=len(ti),validation_positions=len(vi),source_by_target=rows,
                selected_sources=chosen,
                normalized_sources=[round(t*(source.config.num_hidden_layers-1)/max(target.config.num_hidden_layers-1,1)) for t in range(target.config.num_hidden_layers)])
    save_json(dest,result)
    return result


def fit_ridge(cfg,source,target,root,sl,tl,kind,content):
    sources=[sl] if isinstance(sl,int) else sl
    device=source.device
    total=np.load(root/'train_tokens.npy').size
    n=min(total,cfg.get('fit_tokens',total))
    selected=np.sort(np.random.default_rng(cfg['seed']).choice(total,n,replace=False)) if n<total else np.arange(total)
    in_dim=source.config.num_key_value_heads*source.config.head_dim*len(sources)
    out_dim=target.config.num_key_value_heads*target.config.head_dim
    # Float32 accelerator products; float64 accumulation/solve on CPU.
    sumx=np.zeros(in_dim); sumy=np.zeros(out_dim)
    xx=np.zeros((in_dim,in_dim)); xy=np.zeros((in_dim,out_dim))
    for start in range(0,n,8192):
        ix=selected[start:min(start+8192,n)]
        x=torch.cat([read_features(root,'train','source',s,kind,ix,source,content) for s in sources],1).to(device)
        y=read_features(root,'train','target',tl,kind,ix,target,content).to(device)
        sumx+=x.sum(0).cpu().double().numpy(); sumy+=y.sum(0).cpu().double().numpy()
        xx+=(x.T@x).cpu().double().numpy(); xy+=(x.T@y).cpu().double().numpy()
    xm,ym=sumx/n,sumy/n
    xx=xx/n-np.outer(xm,xm); xy=xy/n-np.outer(xm,ym)
    scale=np.sqrt(np.maximum(np.diag(xx),1e-8))
    gram=xx/np.outer(scale,scale); cross=xy/scale[:,None]
    xv=torch.cat([read_features(root,'validation','source',s,kind,backend=source,content=content) for s in sources],1).to(device)
    yv=read_features(root,'validation','target',tl,kind,backend=target,content=content).to(device)
    best=None
    for penalty in cfg['ridge_lambdas']:
        w=scipy.linalg.solve(gram+penalty*np.eye(in_dim),cross,assume_a='pos')/scale[:,None]
        b=ym-xm@w
        wt=torch.from_numpy(w.astype('float32')).to(device)
        bt=torch.from_numpy(b.astype('float32')).to(device)
        metrics=tensor_metrics(yv,xv@wt+bt)
        candidate=dict(sources=sources,target=tl,kind=kind,content=content,penalty=penalty,
                       weight=wt.cpu(),bias=bt.cpu(),metrics=metrics)
        if best is None or metrics['mse']<best['metrics']['mse']: best=candidate
    return best


def train_mappers(cfg,source,target,root):
    from .data import validate_training_caches
    from .identity import bind, config_identity, artifact, digest
    upstream = validate_training_caches(cfg, source, target, root)
    dest=Path(cfg['output'])/'mappers.pt'
    manifest_path=Path(cfg['output'])/'mapper_training_manifest.json'
    progress_path=Path(cfg['output'])/'mapper_training_progress.json'
    identity=dict(schema=2,cache_identity=upstream,config=config_identity(cfg),
                  implementation=artifact(__file__))
    if dest.exists():
        if not manifest_path.exists() or not progress_path.exists():
            raise ValueError('Existing mapper has no validated training provenance; use a new run')
        progress=json.loads(progress_path.read_text())
        if progress['identity_sha256']!=digest(identity) or progress['checkpoint']!=artifact(dest):
            raise ValueError('Mapper checkpoint or upstream training identity changed')
    bind(manifest_path,identity)
    maps=torch.load(dest,weights_only=True) if dest.exists() else {}
    metrics=[]
    for kind,content in [('k',False),('v',False),('k',True)]:
        tag='k_content' if content else ('k_post' if kind=='k' else 'v')
        alignment=alignment_screen(cfg,source,target,root,kind,content)
        for prefix,sources in [('normalized_',alignment['normalized_sources']),('',alignment['selected_sources'])]:
            key=prefix+tag
            entries=maps.setdefault(key,[])
            for tl,sl in enumerate(sources):
                if tl<len(entries): continue
                existing=next((m for group in maps.values() for m in group if m['sources']==[sl]
                    and m['target']==tl and m['kind']==kind and m['content']==content),None)
                fit=existing if existing else fit_ridge(cfg,source,target,root,sl,tl,kind,content)
                entries.append(fit)
                save_checkpoint(dest,maps)
                save_json(progress_path,dict(state='partial',identity_sha256=digest(identity),
                    checkpoint=artifact(dest),completed_layers={k:len(v) for k,v in maps.items()}))
                print(f'fit {key}: {sl} -> {tl}, R2={fit["metrics"]["r2"]:.3f}, lambda={fit["penalty"]}',flush=True)
    for variant,entries in maps.items():
        for tl,m in enumerate(entries):
            metrics.append(dict(variant=variant,target_layer=tl,sources=m['sources'],penalty=m['penalty'],**m['metrics']))
    save_json(Path(cfg['output'])/'reconstruction_validation.json',metrics)
    save_json(progress_path,dict(state='complete',identity_sha256=digest(identity),
        checkpoint=artifact(dest),completed_layers={k:len(v) for k,v in maps.items()}))
