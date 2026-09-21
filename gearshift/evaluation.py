from __future__ import annotations
import difflib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .core import CacheExtractor, CacheInjector, distributions, memory, save_json, tensor_metrics, timed
from .mapping import CacheAdapter


def sequence_metrics(reference,candidate,labels):
    # Float32 CPU accumulation, full vocabulary. Each row predicts the next held-out token.
    ref=reference.detach().cpu().float()[0]
    pred=candidate.detach().cpu().float()[0]
    lp=pred.log_softmax(-1); lr=ref.log_softmax(-1)
    labels=torch.tensor(labels)
    nll=-lp[torch.arange(len(labels)),labels]
    kl=(lr.exp()*(lr-lp)).sum(-1).clamp_min(0)
    return dict(nll=float(nll.mean()),perplexity=math.exp(float(nll.mean())),
                token_nll=nll.tolist(),token_kl=kl.tolist(),
                teacher_forced_top1=(ref.argmax(-1)==pred.argmax(-1)).float().tolist())


@torch.inference_mode()
def latency_sweep(cfg,source,target,adapter,test):
    rows=[]
    rng=np.random.default_rng(cfg['seed'])
    end=max(cfg['lengths'])
    for length in cfg.get('latency_lengths',cfg['lengths']):
        context=test[0,end-length:end].tolist(); probe=[int(test[0,end])]
        src=source.prefill(context)
        pairs=CacheExtractor.tensors(src.past_key_values)
        # Explicit shape-specific warmup; synchronization lives inside timed().
        conditions=['native','post','content','normalized_content']
        for name in ['lowrank','mlp','functional']:
            if name+'_k' in adapter.maps: conditions.append(name)
        target.prefill(context+probe)
        for variant in conditions[1:]:
            target.forward(probe,adapter.inject(pairs,variant))
        for repeat in range(cfg['latency_repeats']):
            order=list(rng.permutation(conditions))
            for variant in order:
                if variant=='native':
                    out,core=timed(lambda:target.prefill(context),target.device)
                    _,probe_ms=timed(lambda:target.forward(probe,out.past_key_values),target.device)
                    _,fused=timed(lambda:target.prefill(context+probe),target.device)
                else:
                    cache,core=timed(lambda:adapter.inject(pairs,variant),target.device)
                    _,probe_ms=timed(lambda:target.forward(probe,cache),target.device)
                    fused=core+probe_ms
                rows.append(dict(context_length=length,repeat=repeat,condition=variant,
                    handoff_core_ms=core,probe_ms=probe_ms,time_to_first_token_ms=fused,
                    target_prefill_tokens=length if variant=='native' else 0,
                    target_probe_tokens=len(probe),**memory()))
        print(f'latency length {length} done',flush=True)
    pd.DataFrame(rows).to_csv(Path(cfg['output'])/'latency.csv',index=False)
    save_json(Path(cfg['output'])/'latency.json',rows)


@torch.inference_mode()
def evaluate(cfg,source,target,root):
    from .identity import bind, backend_identity, artifact, config_identity
    bind(Path(cfg['output'])/'evaluation_manifest.json', dict(schema=2,
         source=backend_identity(source), target=backend_identity(target),
         tokens=artifact(root/'test_tokens.npy'), mapper=artifact(Path(cfg['output'])/'mappers.pt'),
         config=config_identity(cfg), protocol='shared bridge; teacher-forced continuation v2'))
    if (Path(cfg['output'])/'functional.json').exists():
        raise ValueError('Evaluation records already exist; preserve them and use a new evaluation run')
    adapter=CacheAdapter.load(cfg,source,target)
    test=np.load(root/'test_tokens.npy')
    variants=['native','normalized_content','post','content','zero','random','naive','no_context','oracle_native_k','oracle_native_v']
    for name in ['neighbor','lowrank','mlp','weighted','functional']:
        if name+'_k' in adapter.maps: variants.insert(4,name)
    rows,reconstruction,free=[],[],[]
    end=max(cfg['lengths'])
    for length in cfg['lengths']:
        for example,seq in enumerate(test):
            context=seq[end-length:end].tolist()
            inputs=seq[end:end+cfg['continuation_tokens']].tolist()
            labels=seq[end+1:end+cfg['continuation_tokens']+1].tolist()
            src,source_ms=timed(lambda:source.prefill(context),source.device)
            source_pairs=CacheExtractor.tensors(src.past_key_values)
            native,prefill_ms=timed(lambda:target.prefill(context),target.device)
            native_pairs=CacheExtractor.tensors(native.past_key_values)
            reference=target.forward(inputs,CacheInjector.create(native_pairs),all_logits=True).logits
            native_free=None
            for variant in variants:
                before=target.input_token_count
                if variant=='native':
                    pairs=native_pairs; mapper_ms=0
                elif variant=='no_context':
                    pairs=None; mapper_ms=0
                elif variant.startswith('oracle_'):
                    predicted=adapter.transform(source_pairs,'normalized_content')
                    pairs=tuple((real[0],pred[1]) if variant=='oracle_native_k' else (pred[0],real[1])
                                for real,pred in zip(native_pairs,predicted))
                    mapper_ms=0
                else:
                    pairs,mapper_ms=timed(lambda:adapter.transform(source_pairs,variant),target.device)
                if variant in ['post','content']:
                    for layer,((nk,nv),(pk,pv)) in enumerate(zip(native_pairs,pairs)):
                        for kind,y,p in [('k',nk,pk),('v',nv,pv)]:
                            reconstruction.append(dict(condition=variant,context_length=length,example=example,
                                layer=layer,kind=kind,**tensor_metrics(CacheExtractor.flatten(y),CacheExtractor.flatten(p))))
                cache=CacheInjector.create(pairs) if pairs else None
                out=target.forward(inputs,cache,all_logits=True)
                assert target.input_token_count-before==len(inputs), 'Historical tokens were reprocessed'
                metrics=distributions(reference[:,:1],out.logits[:,:1])
                sm=sequence_metrics(reference,out.logits,labels)
                row=dict(condition=variant,context_length=length,example=example,**metrics,**sm,
                         source_prefill_ms=source_ms,target_prefill_ms=prefill_ms if variant=='native' else 0,
                         mapper_ms=mapper_ms,target_prefill_tokens=length if variant=='native' else 0,
                         target_continuation_input_tokens=len(inputs),**memory())
                rows.append(row)
                if example<2:
                    one=target.forward(inputs[:1],CacheInjector.create(pairs) if pairs else None)
                    generated,_,_=target.generate_from(one.past_key_values,one.logits,cfg['free_tokens'])
                    if variant=='native': native_free=generated
                    free.append(dict(condition=variant,context_length=length,example=example,
                        token_ids=generated,text=target.tokenizer.decode(generated),
                        position_agreement=sum(a==b for a,b in zip(generated,native_free))/max(len(native_free),len(generated)),
                        sequence_similarity=difflib.SequenceMatcher(a=native_free,b=generated,autojunk=False).ratio()))
            print(f'evaluation length={length} example={example+1}/{len(test)}',flush=True)
            save_json(Path(cfg['output'])/'functional.json',rows)
            save_json(Path(cfg['output'])/'free_continuations.json',free)
    save_json(Path(cfg['output'])/'reconstruction_test.json',reconstruction)
    pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,list)} for row in rows]).to_csv(Path(cfg['output'])/'functional.csv',index=False)
    pd.DataFrame(reconstruction).to_csv(Path(cfg['output'])/'reconstruction_test.csv',index=False)
    latency_sweep(cfg,source,target,adapter,test)


@torch.inference_mode()
def evaluate_reconstruction(cfg,source,target,root):
    adapter=CacheAdapter.load(cfg,source,target)
    test=np.load(root/'test_tokens.npy'); end=max(cfg['lengths']); rows=[]
    variants=['normalized_content']+(['functional'] if 'functional_k' in adapter.maps else [])
    for length in cfg['lengths']:
        for example,seq in enumerate(test):
            context=seq[end-length:end].tolist()
            sp=CacheExtractor.tensors(source.prefill(context).past_key_values)
            tp=CacheExtractor.tensors(target.prefill(context).past_key_values)
            for variant in variants:
                pp=adapter.transform(sp,variant)
                for layer,(native,predicted) in enumerate(zip(tp,pp)):
                    for kind,y,p in zip(['k','v'],native,predicted):
                        rows.append(dict(condition=variant,context_length=length,example=example,layer=layer,kind=kind,
                            **tensor_metrics(CacheExtractor.flatten(y),CacheExtractor.flatten(p))))
        print(f'functional reconstruction length={length} done',flush=True)
    save_json(Path(cfg['output'])/'functional_reconstruction_test.json',rows)
    pd.DataFrame(rows).to_csv(Path(cfg['output'])/'functional_reconstruction_test.csv',index=False)
