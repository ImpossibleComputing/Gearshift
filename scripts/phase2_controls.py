#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import argparse
from gearshift.phase2_io import read,write,bind,inference_source,snapshot
from gearshift.phase2_inference import backends,bridge_ids
from gearshift.core import CacheExtractor,CacheInjector,distributions
from gearshift.controls import reinjection_control,rope_control


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase2_v1.json');p.add_argument('--attempt',default='controls');args=p.parse_args()
    cfg=read(args.config);source,target=backends(cfg)
    root=Path(cfg['output'])/args.attempt;root.mkdir(parents=True,exist_ok=True)
    bind(root/'manifest.json',dict(config=cfg,code=inference_source(),control_source=snapshot(['scripts/phase2_controls.py','gearshift/controls.py'])))
    for name,b in [('source',source),('target',target)]:
        dest=root/name;dest.mkdir(exist_ok=True)
        c={**cfg,'output':str(dest),'lengths':[128,512,2048,4096]}
        write(dest/'geometry.json',b.introspect())
        reinjection_control(c,b);rope_control(c,b,name)
    tokens=target.tokenizer.encode('Validate native prefixes and mapped suffix absolute positions. '*600)
    rows=[]
    for total,prefix in [(128,32),(2048,128),(4096,256)]:
        pairs=CacheExtractor.tensors(target.prefill(tokens[:total]).past_key_values)
        head=tuple(tuple(x[...,:prefix,:] for x in pair) for pair in pairs)
        hybrid=CacheInjector.splice_prefix(head,pairs)
        a=target.forward(tokens[total:total+1],CacheInjector.create(pairs)).logits
        b=target.forward(tokens[total:total+1],CacheInjector.create(hybrid)).logits
        metrics=distributions(a,b);assert metrics['max_logit_difference']==0
        rows.append(dict(total=total,prefix=prefix,metrics=metrics))
    write(root/'hybrid.json',rows)
    # Verify role boundaries and disabled-thinking prefix from the real tokenizer template.
    t=target.tokenizer
    native=t.apply_chat_template([dict(role='user',content='A'),dict(role='assistant',content='B'),dict(role='user',content='Render.\n')],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    suffix=t.decode(bridge_ids(t,'Render.'))
    assert native.endswith(suffix+'\n')
    write(root/'protocol.json',dict(chat_template=t.chat_template,bridge_text=suffix,bridge_ids=bridge_ids(t,'Render.'),
        template_suffix_agrees_except_final_newline=True,eos=sorted(target.eos)))
    write(root/'complete.json',dict(passed=True))
    print('All real-model controls passed',flush=True)


if __name__=='__main__':main()
