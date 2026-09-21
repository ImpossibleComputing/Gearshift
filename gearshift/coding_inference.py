"""Sampled natural-boundary inference with explicit pre-boundary cache ownership."""
import gc, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gearshift.core import CacheExtractor, CacheInjector, ModelBackend
from gearshift.coding_control import seed_for,write

class Backend(ModelBackend):
    def __init__(self,spec):
        self.name=spec['id'];self.device='cuda:0';self.dtype=torch.bfloat16
        self.tokenizer=AutoTokenizer.from_pretrained(self.name,revision=spec['revision'],local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(self.name,revision=spec['revision'],dtype=self.dtype,
            attn_implementation='sdpa',device_map={'':'cuda:0'},local_files_only=True).eval()
        self.model.requires_grad_(False);self.config=self.model.config
        self.eos={151645,151643};self.input_token_count=0

    @torch.no_grad()
    def prefill_chunked(self,ids,chunk=512):
        cache=None
        for i in range(0,len(ids),chunk):
            out=self.forward(ids[i:i+chunk],cache);cache=out.past_key_values
        return out

def sample(logits,generator,temperature=.6,top_p=.95,top_k=20):
    values,indices=torch.topk(logits[0,-1].float()/temperature,top_k)
    probs=torch.softmax(values,dim=-1)
    # Keep the first token that crosses the nucleus threshold.
    remove=probs.cumsum(-1)-probs>=top_p
    probs=probs.masked_fill(remove,0);probs/=probs.sum()
    return int(indices[torch.multinomial(probs,1,generator=generator)].item())

def generator(task_id,stream,device='cuda:0'):
    return torch.Generator(device=device).manual_seed(seed_for(task_id,0,stream))

def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()

@torch.no_grad()
def reason(backend,prompt_ids,task_id,stream,cap,closing=151668,callback=None):
    rng=generator(task_id,stream,backend.device);initial_state=rng.get_state().cpu().tolist()
    sync();started=time.monotonic();out=backend.prefill_chunked(prompt_ids)
    cache=out.past_key_values;logits=out.logits;tokens=[];natural=False;early_eos=False
    for i in range(cap):
        token=sample(logits,rng);tokens.append(token)
        if token==closing:
            natural=True;break  # This emitted boundary is deliberately NOT cached yet.
        if token in backend.eos:early_eos=True;break
        out=backend.forward([token],cache);cache=out.past_key_values;logits=out.logits
        if callback and i%64==0:callback(tokens)
    if natural:prefix=prompt_ids+tokens[:-1]
    else:
        # An early EOS is preserved as an anomalous capped-like bridge stratum.
        if early_eos:
            out=backend.forward([tokens[-1]],cache);cache=out.past_key_values
        prefix=prompt_ids+tokens
    assert cache.get_seq_length()==len(prefix)
    sync()
    record={'task_id':task_id,'prompt_ids':prompt_ids,'reasoning_ids':tokens,'prefix_ids':prefix,'bridge_ids':[closing],
        'natural_boundary':natural,'reasoning_capped':not natural and not early_eos,'early_eos':early_eos,
        'prefix_cache_length':len(prefix),'seed':seed_for(task_id,0,stream),'rng_initial':initial_state,
        'rng_after_reasoning':rng.get_state().cpu().tolist(),'reasoning_seconds':time.monotonic()-started,
        'reasoning_text':backend.tokenizer.decode(tokens,skip_special_tokens=False)}
    return record,cache

@torch.no_grad()
def answer(backend,history,cache,task_id,stream,cap,callback=None):
    rng=generator(task_id,stream,backend.device);initial=rng.get_state().cpu().tolist()
    assert cache.get_seq_length()==len(history['prefix_ids'])
    sync();start=time.monotonic()
    out=backend.forward(history['bridge_ids'],cache);cache=out.past_key_values;logits=out.logits
    bridge_seconds=time.monotonic()-start;tokens=[];ended=False;first=None
    for i in range(cap):
        token=sample(logits,rng);tokens.append(token)
        if first is None:sync();first=time.monotonic()-start
        if token in backend.eos:ended=True;break
        out=backend.forward([token],cache);cache=out.past_key_values;logits=out.logits
        if callback and i%64==0:callback(tokens)
    sync()
    return {'task_id':task_id,'answer_ids':tokens,'answer_text':backend.tokenizer.decode(tokens,skip_special_tokens=True),
        'answer_text_with_special_tokens':backend.tokenizer.decode(tokens,skip_special_tokens=False),
        'answer_seed':seed_for(task_id,0,stream),'rng_initial':initial,'rng_final':rng.get_state().cpu().tolist(),
        'answer_seconds':time.monotonic()-start,'bridge_seconds':bridge_seconds,'first_answer_token_seconds':first,
        'answer_ended_eos':ended,'answer_capped':not ended,'bridge_token_count':len(history['bridge_ids'])}

def memory_record():
    free,total=torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),free=free,total=total,
        passed=torch.cuda.max_memory_allocated()<.85*total and free>=10*1024**3)

def logit_metrics(a,b):
    a=a.detach().float();b=b.detach().float();pa=a.log_softmax(-1);pb=b.log_softmax(-1)
    return {'max_abs':float((a-b).abs().max()),'mean_abs':float((a-b).abs().mean()),
        'kl':float((pa.exp()*(pa-pb)).sum(-1).mean()),'top1_equal':bool(torch.equal(a.argmax(-1),b.argmax(-1)))}

@torch.no_grad()
def native_controls(backend,root,role,max_prefix=24576):
    root=Path(root);base=backend.tokenizer.encode('def add(a, b):\n    return a + b\n# unicode: café λ\n')
    # Set before examining any cross-model outcomes. Repeat noise is reported,
    # but it never loosens the predeclared ceiling automatically.
    tol={'maximum_absolute_logit_difference':.125,'maximum_KL':.0001,'top1_must_match':True,'continuation_must_match':True}
    write(root/f'{role}_tolerance.json',tol)
    rows=[]
    for n in [128,512,2048,8192,16384,max_prefix]:
        ids=(base*(n//len(base)+1))[:n];torch.cuda.reset_peak_memory_stats()
        native=backend.prefill_chunked(ids);reference=native.logits.detach().clone()
        pairs=CacheExtractor.tensors(native.past_key_values,clone=True)
        injected=CacheInjector.create(pairs,clone=False)
        # Both caches are the identical pre-z prefix; consume z once at absolute n.
        a=backend.forward([151668],native.past_key_values)
        b=backend.forward([151668],injected)
        metric=logit_metrics(a.logits,b.logits)
        continuation=[]
        for _ in range(8):
            x=int(a.logits[0,-1].argmax());y=int(b.logits[0,-1].argmax());continuation.append(x==y)
            a=backend.forward([x],a.past_key_values);b=backend.forward([y],b.past_key_values)
        del a,b,native,injected,pairs
        # Compare uninterrupted-prefix+closing-token with split pre-z execution.
        whole=backend.prefill_chunked(ids+[151668]);wl=whole.logits.detach().clone();del whole
        split=backend.prefill_chunked(ids);split=backend.forward([151668],split.past_key_values)
        split_metric=logit_metrics(wl,split.logits);del split
        # Actual post-RoPE key tensor roundtrip with original absolute positions.
        one=backend.prefill_chunked(ids[:min(n,512)]);k=one.past_key_values.layers[0].keys
        roundtrip=backend.rope(backend.rope(k,inverse=True));rope_error=float((roundtrip-k.float()).abs().max())
        del k,one;gc.collect();torch.cuda.empty_cache()
        mem=memory_record()
        passed=all(m['max_abs']<=.125 and m['kl']<=.0001 and m['top1_equal'] for m in [metric,split_metric]) and all(continuation) and rope_error<.001 and mem['passed']
        rows.append({'prefix_tokens':n,'reinjection':metric,'whole_vs_split':split_metric,'deterministic_continuation_equal':all(continuation),
            'inverse_rope_roundtrip_max_abs':rope_error,'memory':mem,'passed':passed})
        write(root/f'{role}_native_controls.json',{'tolerance':tol,'rows':rows,'passed':all(r['passed'] for r in rows),'complete':n==max_prefix})
        print(role,'native control',n,'passed',passed,flush=True)
        if not passed:raise RuntimeError(f'{role} native control failed at {n}')
    return rows
