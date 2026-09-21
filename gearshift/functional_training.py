"""Optional frozen-target functional distillation of affine cache maps."""
import copy
from pathlib import Path
import numpy as np
import torch
from .core import CacheExtractor, CacheInjector, save_json, save_checkpoint, seed_all
from .mapping import CacheAdapter


def train_functional(cfg,source,target,root):
    from .identity import bind, artifact, backend_identity, protect_pilot
    protect_pilot(cfg['output'])
    seed_all(cfg['seed'])
    path=Path(cfg['output'])/'mappers.pt'
    settings=dict(steps=64,learning_rate=1e-5,context_lengths=[128,512],prediction_tokens=8,
        reconstruction_weight=0.01,gradient_clip=1.0,validation_every=8,
        validation_context_length=256,early_stopping_patience=0,min_delta=0.0)
    settings.update(cfg.get('functional',{}))
    bind(Path(cfg['output'])/'functional_training_manifest.json',dict(settings=settings,
         source=backend_identity(source),target=backend_identity(target),initialization=artifact(path),
         train_tokens=artifact(root/'train_tokens.npy'),validation_tokens=artifact(root/'validation_tokens.npy')))
    if (Path(cfg['output'])/'functional_training.json').exists():
        raise ValueError('Functional training already has records; use a new run to preserve them')
    maps=torch.load(path,weights_only=True)
    train=np.load(root/'train_tokens.npy'); validation=np.load(root/'validation_tokens.npy')
    for kind,key in [('k','normalized_k_content'),('v','normalized_v')]:
        maps['functional_'+kind]=copy.deepcopy(maps[key])
    adapter=CacheAdapter(source,target,maps,'functional')
    parameters=[]
    for key in ['functional_k','functional_v']:
        for m in adapter.weights[key]:
            for name in ['weight','bias']:
                m[name]=torch.nn.Parameter(m[name].detach().clone())
                parameters.append(m[name])
    optimizer=torch.optim.Adam(parameters,lr=settings['learning_rate'])
    assert not any(p.requires_grad for p in target.model.parameters())
    assert not any(p.requires_grad for p in source.model.parameters())
    rng=np.random.default_rng(cfg['seed'])

    def pair(row,start,length):
        ids=row[start:start+length].tolist()
        continuation=row[start+length:start+length+settings['prediction_tokens']].tolist()
        if len(continuation)!=settings['prediction_tokens']:raise ValueError('Training block is too short for configured context and prediction budget')
        so=source.prefill(ids); to=target.prefill(ids)
        # Clone OUTSIDE inference_mode so autograd may save historical features.
        sp=CacheExtractor.tensors(so.past_key_values,clone=True)
        tp=CacheExtractor.tensors(to.past_key_values,clone=True)
        with torch.no_grad():
            reference=target.forward(continuation,CacheInjector.create(tp),all_logits=True).logits.float().log_softmax(-1)
        return sp,tp,continuation,reference

    validation_pairs=[pair(row,64,settings['validation_context_length']) for row in validation]
    def evaluate_validation():
        scores=[]
        with torch.no_grad():
            for sp,tp,ids,lr in validation_pairs:
                cache=adapter.inject(sp,'functional')
                lp=target.forward(ids,cache,all_logits=True).logits.float().log_softmax(-1)
                scores.append(float((lr.exp()*(lr-lp)).sum(-1).mean()))
        return float(np.mean(scores))

    initial=evaluate_validation()
    best=initial; best_state=[p.detach().clone() for p in parameters]
    history=[dict(step=0,validation_kl=initial)]
    stale=0
    for step in range(settings['steps']):
        row=train[int(rng.integers(len(train)))]
        length=int(rng.choice(settings['context_lengths']))
        start=int(rng.integers(0,len(row)-length-settings['prediction_tokens']+1))
        sp,tp,ids,lr=pair(row,start,length)
        mapped=adapter.transform(sp,'functional')
        logits=target.forward(ids,CacheInjector.create(mapped,clone=False),all_logits=True).logits.float()
        lp=logits.log_softmax(-1)
        kl=(lr.exp()*(lr-lp)).sum(-1).mean()
        # Small standardized cache penalty discourages catastrophic drift.
        reconstruction=sum((p.float()-t.float()).square().mean()/t.float().square().mean().clamp_min(1e-6)
                           for predicted,actual in zip(mapped,tp) for p,t in zip(predicted,actual))/(2*len(tp))
        loss=kl+settings['reconstruction_weight']*reconstruction
        optimizer.zero_grad(); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,settings['gradient_clip'])
        if not torch.isfinite(norm):
            history.append(dict(step=step+1,failed='Non-finite gradients; best validated checkpoint retained'))
            break
        optimizer.step()
        entry=dict(step=step+1,train_kl=float(kl.detach()),reconstruction=float(reconstruction.detach()),grad_norm=float(norm))
        if (step+1)%settings['validation_every']==0 or step+1==settings['steps']:
            score=evaluate_validation(); entry['validation_kl']=score
            stale=0 if score<best-settings['min_delta'] else stale+1
            if score<best:
                best=score; best_state=[p.detach().clone() for p in parameters]
            print(f'functional training {step+1}/{settings["steps"]}: train KL {float(kl.detach()):.3f}, validation KL {score:.3f}',flush=True)
        history.append(entry)
        save_json(Path(cfg['output'])/'functional_training.json',dict(initial_validation_kl=initial,best_validation_kl=best,
            history=history,settings=settings))
        if settings['early_stopping_patience'] and stale>=settings['early_stopping_patience']:break
    it=iter(best_state)
    for key in ['functional_k','functional_v']:
        for m in maps[key]:
            m['weight']=next(it).detach().cpu(); m['bias']=next(it).detach().cpu()
    save_checkpoint(path,maps)
    save_json(Path(cfg['output'])/'functional_training.json',dict(initial_validation_kl=initial,best_validation_kl=best,
        history=history,settings=settings))
