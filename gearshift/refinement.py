"""Capacity escalation after the full-data affine baseline, in a recorded order."""
import copy
from pathlib import Path
import numpy as np
import scipy.linalg
import torch
from .core import save_json, save_checkpoint, tensor_metrics, seed_all
from .data import read_features
from .mapping import fit_ridge


def refine(cfg,source,target,root):
    path=Path(cfg['output'])/'mappers.pt'
    maps=torch.load(path,weights_only=True)
    device=source.device
    log=[]
    total=np.load(root/'train_tokens.npy').size
    indices=np.sort(np.random.default_rng(cfg['seed']).choice(total,min(total,16384),replace=False))
    # Bounded exploratory refinements use the same 16,384 training positions, full validation.
    for phase in ['neighbor','lowrank','mlp','weighted']:
        for kind,base in [('k','k_content'),('v','v')]:
            key=phase+'_'+kind
            entries=maps.setdefault(key,[])
            for tl,original in enumerate(maps[base]):
                if tl<len(entries): continue
                sl=original['sources'][0]
                x=read_features(root,'train','source',sl,kind,indices,source,kind=='k').to(device)
                y=read_features(root,'train','target',tl,kind,indices,target,kind=='k').to(device)
                xv=read_features(root,'validation','source',sl,kind,backend=source,content=kind=='k').to(device)
                yv=read_features(root,'validation','target',tl,kind,backend=target,content=kind=='k').to(device)
                m=copy.deepcopy(original)
                if phase=='neighbor':
                    neighbor=sl+1 if sl+1<source.config.num_hidden_layers else sl-1
                    local={**cfg,'fit_tokens':len(indices),'ridge_lambdas':[0.01,0.1]}
                    m=fit_ridge(local,source,target,root,[sl,neighbor],tl,kind,kind=='k')
                elif phase=='lowrank':
                    # A rank-constrained approximation of the fitted affine map, with recentered bias.
                    u,s,v=scipy.linalg.svd(m['weight'].numpy(),full_matrices=False)
                    rank=min(128,len(s))
                    down=torch.from_numpy(u[:,:rank]*s[:rank]).to(device)
                    up=torch.from_numpy(v[:rank]).to(device)
                    bias=y.mean(0)-(x.mean(0)@down)@up
                    m.update(down=down.cpu(),up=up.cpu(),bias=bias.cpu(),rank=rank)
                    m['metrics']=tensor_metrics(yv,(xv@down)@up+bias)
                elif phase=='mlp':
                    seed_all(cfg['seed']+tl+(100 if kind=='v' else 0))
                    mean=x.mean(0)
                    w=m['weight'].to(device); b=m['bias'].to(device)
                    w1=torch.nn.Parameter(torch.randn(x.shape[1],128,device=device)*0.01)
                    b1=torch.nn.Parameter(torch.zeros(128,device=device))
                    w2=torch.nn.Parameter(torch.zeros(128,y.shape[1],device=device))
                    b2=torch.nn.Parameter(torch.zeros(y.shape[1],device=device))
                    optimizer=torch.optim.AdamW([w1,b1,w2,b2],lr=0.002,weight_decay=0.0001)
                    # Fixed 64 mini-batches; best validation checkpoint, including the affine initialization.
                    best_mse=original['metrics']['mse']; best=None
                    for step in range(64):
                        ix=torch.randint(0,len(x),(512,),device=device)
                        pred=x[ix]@w+b+torch.nn.functional.gelu((x[ix]-mean)@w1+b1)@w2+b2
                        loss=(pred-y[ix]).square().mean()
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                        if (step+1)%16==0:
                            with torch.no_grad():
                                metrics=tensor_metrics(yv,xv@w+b+torch.nn.functional.gelu((xv-mean)@w1+b1)@w2+b2)
                            if metrics['mse']<best_mse:
                                best_mse=metrics['mse']
                                best=dict(mlp_w1=w1.detach().cpu().clone(),mlp_b1=b1.detach().cpu().clone(),
                                    mlp_w2=w2.detach().cpu().clone(),mlp_b2=b2.detach().cpu().clone(),
                                    mean=mean.cpu(),metrics=metrics,mlp_selected_step=step+1)
                    if best: m.update(best)
                    else: m['mlp_selected_step']=0
                elif phase=='weighted':
                    # Learn a convex combination of normalized-depth and selected-layer predictions on TRAIN.
                    other=maps['normalized_'+base][tl]
                    os=other['sources'][0]
                    z=read_features(root,'train','source',os,kind,indices,source,kind=='k').to(device)
                    zv=read_features(root,'validation','source',os,kind,backend=source,content=kind=='k').to(device)
                    a=x@m['weight'].to(device)+m['bias'].to(device)
                    b=z@other['weight'].to(device)+other['bias'].to(device)
                    diff=a-b
                    alpha=float(((y-b)*diff).sum()/diff.square().sum().clamp_min(1e-12))
                    alpha=max(0,min(1,alpha))
                    m['sources']=[sl,os]
                    m['weight']=torch.cat([alpha*m['weight'],(1-alpha)*other['weight']],0)
                    m['bias']=alpha*m['bias']+(1-alpha)*other['bias']
                    m['alpha']=alpha
                    m['metrics']=tensor_metrics(yv,torch.cat([xv,zv],1)@m['weight'].to(device)+m['bias'].to(device))
                entries.append(m)
                save_checkpoint(path,maps)
                print(f'refine {phase} {kind} layer {tl}: R2={m["metrics"]["r2"]:.3f}',flush=True)
    for variant,entries in maps.items():
        if variant.split('_')[0] in ['neighbor','lowrank','mlp','weighted']:
            for tl,m in enumerate(entries):
                log.append(dict(variant=variant,layer=tl,sources=m['sources'],
                    training_positions_for_refinement=len(indices),alpha=m.get('alpha'),
                    mlp_selected_step=m.get('mlp_selected_step'),**m['metrics']))
    save_json(Path(cfg['output'])/'refinement_validation.json',log)
