#!/usr/bin/env python3
"""Actual pinned BF16 receiver-path checks, with immutable real-history caches."""
import argparse,gc,hashlib,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.coding_control import write,sha,digest
from gearshift.coding_training import read,finish_worker
from gearshift.coding_recovery import worker_context,initialize_backends
from gearshift.coding_gradients import continuation,AffineMapper
from gearshift.coding_post_progress import DECLARATION,selected_histories
from gearshift.coding_inference import logit_metrics
from gearshift.core import CacheExtractor,CacheInjector
ROOT=Path(__file__).resolve().parents[1]

def fingerprints(pairs):
    return [[hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for t in pair] for pair in pairs]

def direct(b,pairs,ids,positions,explicit=False):
    cache=CacheInjector.create(pairs,clone=True)
    if not explicit:
        return b.forward(ids,cache,all_logits=True).logits[:,positions]
    n=pairs[0][0].shape[-2];absolute=torch.arange(n,n+len(ids),device=b.device)
    keys=torch.arange(n+len(ids),device=b.device)
    mask=torch.zeros((len(ids),n+len(ids)),device=b.device,dtype=b.dtype)
    mask.masked_fill_(keys[None]>absolute[:,None],torch.finfo(b.dtype).min)
    return b.model(input_ids=b.ids(ids),past_key_values=cache,use_cache=True,attention_mask=mask[None,None],position_ids=absolute[None],cache_position=absolute,logits_to_keep=0).logits[:,positions]

def brief(logits):
    p=logits.float().softmax(-1);v,i=p.topk(8,dim=-1)
    return {'top8_ids':i.cpu().tolist(),'top8_probabilities':v.cpu().tolist(),'logits_sha256':fingerprints(((logits,logits),))[0][0]}

def gradient(b,pairs,ids,path):
    # Last two positions of first-layer value only. All other cache entries are constants.
    leaf=pairs[0][1][...,-2:,:].detach().clone().requires_grad_(True)
    values=torch.cat((pairs[0][1][...,:-2,:].detach(),leaf),dim=-2)
    current=((pairs[0][0].detach(),values),)+tuple(tuple(x.detach() for x in pair) for pair in pairs[1:])
    if path=='manual':z=continuation(b.model,current,ids,positions=[len(ids)-1],checkpoint_layers=True)
    else:z=direct(b,current,ids,[len(ids)-1])
    # Fixed vocabulary coordinates, independent of observed gradients/answers.
    loss=z.float()[0,-1,[13,198,220,1000]].mean()
    g=torch.autograd.grad(loss,leaf)[0].float().detach();del loss,z,leaf,current,values
    return g

def run(c):
    d=read(ROOT/DECLARATION)
    if sha(ROOT/DECLARATION)!=c['spec']['declaration_sha256']:raise ValueError('Declaration differs')
    source,b=initialize_backends(c);mapper=AffineMapper(source,b)
    cp=ROOT/d['selected_checkpoint'];assert sha(cp)==d['selected_checkpoint_sha256']
    mapper.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state_dict']);mapper.requires_grad_(False)
    rows=[];all_passed=True
    for obj in selected_histories(ROOT,d):
        h=obj['source_history'];answer=obj['teacher_answer']['answer_ids'];n=len(h['prefix_ids']);tid=obj['task_id']
        pos=sorted(set([0,min(32,len(answer)-1),min(128,len(answer)-1),min(512,len(answer)-1),len(answer)-1]))
        ids=(h['bridge_ids']+answer[:-1])[:max(pos)+1]
        c['guard']();c['publish'](stage='numerical_cache_reconstruction',task_id=tid);c['telemetry'].reset(tid)
        with torch.no_grad():
            out=source.prefill_chunked(h['prefix_ids']);sp=CacheExtractor.tensors(out.past_key_values);del out
            mapped=tuple(tuple(x.cpu() for x in pair) for pair in mapper(sp));del sp
            out=b.prefill_chunked(h['prefix_ids']);native=CacheExtractor.tensors(out.past_key_values,device='cpu');del out
        for kind,cpu in [('native',native),('mapped',mapped)]:
            c['guard']();c['publish'](stage='numerical_path_comparison',task_id=tid,cache_kind=kind)
            pairs=tuple(tuple(x.to(b.device) for x in pair) for pair in cpu);before=fingerprints(pairs)
            with torch.no_grad():
                deployed=direct(b,pairs,ids,pos);repeat=direct(b,pairs,ids,pos)
                explicit=direct(b,pairs,ids,pos,True)
                manual=continuation(b.model,pairs,ids,positions=pos,checkpoint_layers=False)
                checked=continuation(b.model,pairs,ids,positions=pos,checkpoint_layers=True)
                controls={'deployed_repeat':logit_metrics(deployed,repeat),'deployed_explicit_mask':logit_metrics(deployed,explicit),'manual_checkpoint_toggle':logit_metrics(manual,checked)}
                cross=logit_metrics(manual,explicit);original_cross=logit_metrics(manual,deployed)
                # Measure deployment's actual one-token answer schedule separately;
                # chunk-vs-token rounding is NOT a tolerance for manual-path drift.
                cache=CacheInjector.create(pairs,clone=True);sequential=[]
                for j,token in enumerate(ids):
                    out=b.forward([token],cache);cache=out.past_key_values
                    if j in pos:sequential.append(out.logits.detach().clone())
                    if j%128==0:c['guard']()
                sequential=torch.cat(sequential,dim=1);del out,cache
                rounding=logit_metrics(deployed,sequential)
                details=[{'answer_position':p,'training_vs_deployed':logit_metrics(manual[:,k:k+1],deployed[:,k:k+1]),'batch_vs_token':logit_metrics(deployed[:,k:k+1],sequential[:,k:k+1]),'manual':brief(manual[:,k:k+1]),'deployed':brief(deployed[:,k:k+1])} for k,p in enumerate(pos)]
                envelope=max(controls['deployed_repeat']['max_abs'],controls['manual_checkpoint_toggle']['max_abs'])
                passed=cross['max_abs']<=envelope and original_cross['max_abs']<=max(envelope,controls['deployed_explicit_mask']['max_abs'])
                del deployed,repeat,explicit,manual,checked,sequential
            ga=gradient(b,pairs,ids[:4],'deployed');gb=gradient(b,pairs,ids[:4],'deployed');gm=gradient(b,pairs,ids[:4],'manual')
            gnoise=float((ga-gb).abs().max());gdelta=float((ga-gm).abs().max())
            gradients={'checked':'first-layer values, final two historical positions, first four continuation tokens; mean four fixed logit coordinates; not all mapper parameters','same_path_max_abs':gnoise,'cross_path_max_abs':gdelta,'native_norm':float(ga.norm()),'manual_norm':float(gm.norm()),'finite':bool(torch.isfinite(ga).all() and torch.isfinite(gm).all()),'passed':gdelta<=gnoise and bool(torch.isfinite(gm).all())}
            intact=before==fingerprints(pairs);isolated=CacheInjector.create(pairs,clone=True)
            noalias=all(a.data_ptr()!=z.data_ptr() for p,q in zip(pairs,CacheExtractor.tensors(isolated)) for a,z in zip(p,q));del isolated
            row={'task_id':tid,'cache_kind':kind,'prefix_tokens':n,'answer_positions':pos,'input_ids':ids,'controls':controls,'cross_path':cross,'manual_vs_default_deployed':original_cross,'batch_vs_sequential':rounding,'tolerance_from_repeat_controls':envelope,'cache_unchanged':intact,'clone_no_alias':noalias,'gradients':gradients,'positions':details,'passed':passed and intact and noalias and gradients['passed']}
            rows.append(row);all_passed &= row['passed'];write(c['root']/'numerical_comparison.json',{'rows':rows,'complete':False,'passed':False})
            del pairs,ga,gb,gm;gc.collect();torch.cuda.empty_cache();c['telemetry'].sample(stage='numerical_case_complete')
        del mapped,native
    result={'rows':rows,'complete':True,'passed':all_passed,'deterministic':True,'reference_continuation':'saved source-written answer','native_prefill':'full real prefixes in512-token chunks; original native controls also run to32768','limitations':['Gradient comparison covers one small value-cache slice, not every key/value/mapper parameter.','Batch versus sequential deployment rounding is reported separately; no threshold is tuned to accept it.']}
    write(c['root']/'numerical_comparison.json',result);write(c['root']/'complete.json',{'identity_sha256':digest(c['identity']),'numerical_passed':all_passed,'cases':len(rows),'comparison_sha256':sha(c['root']/'numerical_comparison.json')})
    # A measured discrepancy is a successful diagnostic capture but blocks training.
    finish_worker(c,'complete',stage='numerical_diagnostic_complete',numerical_passed=all_passed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-spec');a=p.parse_args();c=None
    try:c=worker_context(ROOT,a.worker_spec);run(c)
    except BaseException as exc:
        if c:c['telemetry'].failure(exc);write(c['root']/'failure.json',{'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()});finish_worker(c,'failed',error=str(exc))
        raise
