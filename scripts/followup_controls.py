#!/usr/bin/env python3
"""Real local model controls; writes only the follow-up experiment."""
import sys
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
import torch
from gearshift.core import ModelBackend, CacheExtractor, CacheInjector, distributions, save_json, environment, seed_all
from gearshift.controls import reinjection_control, rope_control
from gearshift.data import extract_pairs
from gearshift.identity import bind, backend_identity, artifact, digest, array_descriptor
from gearshift.followup import source_files


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/followup_v1.json');p.add_argument('--attempt',default='controls_v2');args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text()); seed_all(cfg['seed'])
    root=Path(cfg['output'])/args.attempt; root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    models={r:ModelBackend(cfg[r],cfg['device'],cfg['dtype'],cfg['attention'],cfg[r+'_revision']) for r in ['source','target']}
    bind(root/'manifest.json',dict(config=cfg,backends={r:backend_identity(b) for r,b in models.items()},code=source_files()))
    save_json(root/'environment.json',environment())
    for role,backend in models.items():
        path=root/role;path.mkdir(exist_ok=True); cc={**cfg,'output':str(path)}
        reinjection_control(cc,backend);rope_control(cc,backend,role)
    b=models['target']; ids=b.tokenizer.encode('Check absolute cache positions and a native control prefix. '*600)
    assert len(ids)>2048
    hybrids=[]
    for total,prefix in [(128,32),(512,64),(2048,128)]:
        full=CacheExtractor.tensors(b.prefill(ids[:total]).past_key_values)
        pristine=tuple(tuple(x[...,:prefix,:] for x in pair) for pair in full)
        hybrid=CacheInjector.splice_prefix(pristine,full)
        bridge=ids[total:total+1]
        ref=b.forward(bridge,CacheInjector.create(full)).logits
        candidate=b.forward(bridge,CacheInjector.create(hybrid)).logits
        exact=distributions(ref,candidate)
        assert exact['max_logit_difference']==0
        independent=CacheExtractor.tensors(b.prefill(ids[:prefix]).past_key_values)
        independent_hybrid=CacheInjector.splice_prefix(independent,full)
        near=distributions(ref,b.forward(bridge,CacheInjector.create(independent_hybrid)).logits)
        assert near['kl']<0.005
        hybrids.append(dict(total=total,prefix=prefix,splice_exact=exact,independent_prefix=near))
    save_json(root/'hybrid.json',hybrids)
    # Exercise actual-model cache extraction, full validation and marker corruption rejection.
    token_root=Path('data/followup_controls')/args.attempt;token_root.mkdir(parents=True,exist_ok=True)
    for split,n in [('train',2),('validation',1)]:
        np.save(token_root/f'{split}_tokens.npy',np.asarray([ids[i*16:(i+1)*16] for i in range(n)],dtype=np.int32))
    save_json(token_root/'tokens.json',dict(splits={s:dict(array=array_descriptor(token_root/f'{s}_tokens.npy'),
        identity_sha256=digest(dict(protocol='actual-model control fixture',split=s,tokens=ids[:32]))) for s in ['train','validation']}))
    cc={**cfg,'output':str(root/'extraction'),'cache_store':str(Path('data/followup_control_store')/args.attempt)}
    extract_pairs(cc,models['source'],models['target'],token_root)
    extract_pairs(cc,models['source'],models['target'],token_root)
    victim=token_root/'train_source_0_k.npy'; saved=victim.resolve().read_bytes()
    victim.resolve().write_bytes(saved[:100])
    rejected=False
    try: extract_pairs(cc,models['source'],models['target'],token_root)
    except ValueError: rejected=True
    finally: victim.resolve().write_bytes(saved)
    assert rejected
    save_json(root/'complete.json',dict(passed=True,corruption_rejected=rejected,full_wall_seconds=time.perf_counter()-started))
    print('Actual-model controls passed',flush=True)


if __name__=='__main__':main()
