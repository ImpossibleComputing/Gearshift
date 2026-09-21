"""Distinguish cache restoration from changing the numerical execution schedule."""
from contextlib import contextmanager
import gc
import types
from pathlib import Path
import torch
import torch.nn.functional as F
from gearshift.core import CacheExtractor,CacheInjector
from gearshift.coding_control import write
from gearshift.coding_inference import logit_metrics,memory_record

@contextmanager
def float_arithmetic_audit(model):
    """Diagnostic only: BF16 weights stay unchanged; arithmetic/caches use FP32.

    Casting one linear's weight at a time avoids retaining an FP32 model copy.
    Quality runs always restore the original implementation before any task.
    """
    saved=[]
    handle=model.model.embed_tokens.register_forward_hook(lambda module,inputs,out:out.float())
    try:
        for module in model.modules():
            if isinstance(module,torch.nn.Linear):
                saved.append((module,module.forward))
                def forward(self,x):
                    return F.linear(x.float(),self.weight.float(),None if self.bias is None else self.bias.float())
                module.forward=types.MethodType(forward,module)
        yield
    finally:
        handle.remove()
        for module,forward in saved:module.forward=forward

@torch.no_grad()
def bulk_difference(backend,ids):
    a=backend.prefill_chunked(ids+[151668]);al=a.logits.clone();del a
    b=backend.prefill_chunked(ids);b=backend.forward([151668],b.past_key_values)
    return logit_metrics(al,b.logits)

@torch.no_grad()
def precision_audit(backend,root,role):
    base=backend.tokenizer.encode('def add(a, b):\n    return a + b\n# unicode: café λ\n')
    ids=(base*(128//len(base)+1))[:128]
    before=backend.prefill_chunked(ids+[151668]);original=before.logits.clone();del before
    bf16=bulk_difference(backend,ids)
    with float_arithmetic_audit(backend.model):fp32=bulk_difference(backend,ids)
    after=backend.prefill_chunked(ids+[151668]);restored=logit_metrics(original,after.logits);del after
    weights_unchanged_dtype=all(p.dtype==torch.bfloat16 for p in backend.model.parameters())
    passed=(fp32['max_abs']<=.005 and abs(fp32['kl'])<=.000001 and fp32['top1_equal']
        and restored['max_abs']==0 and weights_unchanged_dtype)
    row={'passed':passed,'bf16_bulk_vs_incremental':bf16,'fp32_arithmetic_bulk_vs_incremental':fp32,
        'restored_quality_implementation':restored,'all_stored_weights_remain_bfloat16':weights_unchanged_dtype,
        'prespecified_fp32_audit_max_abs':.005,'prespecified_fp32_audit_max_abs_KL':.000001,
        'interpretation':'Bulk replay changes the GEMM/attention call shapes. It is not an uninterrupted autoregressive run. This diagnostic does not replace the failed v1 gate with a pass.'}
    write(Path(root)/f'{role}_precision_audit.json',row)
    print(role,'precision audit',row,flush=True)
    if not passed:raise RuntimeError(f'{role} precision diagnosis inconclusive; stop before quality runs')
    gc.collect();torch.cuda.empty_cache()
    return row

def within_cache_tolerance(m):
    return m['max_abs']<=.125 and abs(m['kl'])<=.0001 and m['top1_equal']

@torch.no_grad()
def matched_native_controls(backend,root,role,max_prefix=24576):
    root=Path(root);base=backend.tokenizer.encode('def add(a, b):\n    return a + b\n# unicode: café λ\n')
    audit=precision_audit(backend,root,role)
    rows=[]
    for n in [128,512,2048,8192,16384,max_prefix]:
        ids=(base*(n//len(base)+1))[:n];torch.cuda.reset_peak_memory_stats()
        for pattern in ['closing_token_pending','capped_final_reasoning_token_pending']:
            # The uninterrupted and paused paths execute exactly the same calls.
            prefix=ids if pattern=='closing_token_pending' else ids[:-1]
            pending=[151668] if pattern=='closing_token_pending' else [ids[-1],151668]
            native=backend.prefill_chunked(prefix)
            pairs=CacheExtractor.tensors(native.past_key_values,clone=True)
            paused=CacheInjector.create(pairs,clone=False)
            assert native.past_key_values.get_seq_length()==paused.get_seq_length()==len(prefix)
            live=native.past_key_values
            for token in pending:
                a=backend.forward([token],live);live=a.past_key_values
                b=backend.forward([token],paused);paused=b.past_key_values
            assert live.get_seq_length()==paused.get_seq_length()==n+1
            metric=logit_metrics(a.logits,b.logits);saved=a.logits.clone()
            continuation=[]
            for _ in range(8):
                x=int(a.logits[0,-1].argmax());y=int(b.logits[0,-1].argmax());continuation.append(x==y)
                a=backend.forward([x],a.past_key_values);b=backend.forward([y],b.past_key_values)
            del a,b,native,live,paused,pairs
            # Independent same-schedule repetition calibrates deterministic noise.
            repeat=backend.prefill_chunked(prefix)
            for token in pending:repeat=backend.forward([token],repeat.past_key_values)
            repeat_metric=logit_metrics(saved,repeat.logits);del repeat,saved
            gc.collect();torch.cuda.empty_cache();mem=memory_record()
            row={'prefix_tokens':n,'pattern':pattern,'reinjection':metric,'same_schedule_repeat':repeat_metric,
                'deterministic_continuation_equal':all(continuation),'expected_cache_length_after_bridge':n+1,
                'memory':mem,'passed':within_cache_tolerance(metric) and within_cache_tolerance(repeat_metric) and all(continuation) and mem['passed']}
            rows.append(row);write(root/f'{role}_matched_native_controls.json',{'schema':2,'rows':rows,'precision_audit_passed':audit['passed'],
                'passed':all(r['passed'] for r in rows),'complete':n==max_prefix and pattern=='capped_final_reasoning_token_pending',
                'unchanged_cache_tolerance':{'max_abs':.125,'abs_KL':.0001,'top1_must_match':True}})
            print(role,'matched control',n,pattern,'passed',row['passed'],flush=True)
            if not row['passed']:raise RuntimeError(f'{role} matched native control failed at {n}')
    # Original inverse/post-RoPE roundtrip remains a required independent check.
    one=backend.prefill_chunked(ids[:512]);k=one.past_key_values.layers[0].keys
    error=float((backend.rope(backend.rope(k,inverse=True))-k.float()).abs().max())
    write(root/f'{role}_rope_roundtrip.json',{'max_abs':error,'passed':error<.001})
    if error>=.001:raise RuntimeError('RoPE roundtrip failed')
    return rows
