#!/usr/bin/env python3
"""Two prespecified pilot flip IDs, cache-construction × EOS diagnostic; original files untouched."""
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.core import CacheExtractor,CacheInjector,save_json,timed,seed_all
from gearshift.followup import backends,source_files
from gearshift.identity import bind,artifact,backend_identity
from gearshift.mapping import CacheAdapter
from gearshift.reasoning import numeric_answer


@torch.inference_mode()
def answer(target,pairs,delimiter,budget,eos):
    out=target.forward(delimiter,CacheInjector.create(pairs));tokens=[];margins=[]
    for i in range(budget):
        top=out.logits[0,-1].float().topk(2);token=int(out.logits[0,-1].argmax())
        margins.append(dict(position=i,chosen_token=token,top2_ids=top.indices.tolist(),top2_logits=top.values.tolist(),gap=float(top.values[0]-top.values[1])))
        tokens.append(token);out=target.forward([token],out.past_key_values)
        if token in eos:break
    return dict(tokens=tokens,text=target.tokenizer.decode(tokens,skip_special_tokens=True),margins=margins,
        stopped_on_eos=bool(tokens and tokens[-1] in eos),cap_length=len(tokens)==budget)


@torch.inference_mode()
def main():
    cfg=json.loads(Path('configs/followup_v1.json').read_text());source,target=backends(cfg)
    out=Path(cfg['output'])/'pilot_sensitivity_v2';out.mkdir(parents=True,exist_ok=True)
    pilot=Path('results/qwen3_1.7b_to_0.6b');path=Path('data/qwen3_1.7b_to_0.6b/reasoning_trajectories.jsonl')
    records={r['dataset_index']:r for r in map(json.loads,path.read_text().splitlines())}
    original={r['dataset_index']:r for r in json.loads((pilot/'reasoning.json').read_text()) if r['condition']=='E_functional_kv'}
    replay={r['dataset_index']:r for r in json.loads((pilot/'stopping_audit.json').read_text()) if r['condition']=='E_functional_kv'}
    bind(out/'manifest.json',dict(ids=[1199,317],source=backend_identity(source),target=backend_identity(target),
        mapper=artifact(pilot/'mappers.pt'),trajectory_file=artifact(path),eos_variants=[[151645],sorted(target.eos)],
        answer_budget=48,code=source_files(),decoding='argmax, exactly matching ModelBackend.generate_from; topk is for margin logging only',
        protocol='Teacher-force saved tokens with full versus incremental source cache reconstruction; neither is an original live-cache capture'))
    maps=torch.load(pilot/'mappers.pt',weights_only=True);adapter=CacheAdapter(source,target,{k:maps[k] for k in ['functional_k','functional_v']},'functional');del maps
    rows=[]
    for index in [1199,317]:
        tr=records[index]
        for construction in ['full_prefill','incremental_saved_tokens']:
            def construct():
                if construction=='full_prefill':return source.prefill(tr['prompt_ids']+tr['source_reasoning_ids'])
                result=source.prefill(tr['prompt_ids'])
                for token in tr['source_reasoning_ids']:result=source.forward([token],result.past_key_values)
                return result
            so,ms=timed(construct,source.device);pairs=adapter.transform(CacheExtractor.tensors(so.past_key_values),'functional')
            for label,eos in [('original_eos',{151645}),('corrected_eos',target.eos)]:
                result=answer(target,pairs,tr['delimiter_ids'],48,eos)
                rows.append(dict(dataset_index=index,construction=construction,eos_variant=label,
                    correct=numeric_answer(result['text'])==tr['gold'],gold=tr['gold'],
                    matches_original_text=result['text']==original[index]['answer'],matches_previous_replay_text=result['text']==replay[index]['answer'],
                    reconstruction_ms=ms,**result))
            del pairs,so
        print(f'Sensitivity diagnostic {index} complete',flush=True)
    comparisons=[]
    for index in [1199,317]:
        for eos in ['original_eos','corrected_eos']:
            a,b=[r for r in rows if r['dataset_index']==index and r['eos_variant']==eos]
            different=next((i for i,(x,y) in enumerate(zip(a['tokens'],b['tokens'])) if x!=y),None)
            comparisons.append(dict(dataset_index=index,eos_variant=eos,exact_token_agreement=a['tokens']==b['tokens'],
                aligned_token_agreement=sum(x==y for x,y in zip(a['tokens'],b['tokens']))/max(len(a['tokens']),len(b['tokens'])),
                first_divergence=different,full_margin_at_divergence=a['margins'][different] if different is not None else None,
                incremental_margin_at_divergence=b['margins'][different] if different is not None else None))
    save_json(out/'records.json',rows);save_json(out/'comparisons.json',comparisons)
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
