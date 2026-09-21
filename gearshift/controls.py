import inspect
from pathlib import Path
import torch
from .core import CacheExtractor, CacheInjector, distributions, save_json, timed


@torch.inference_mode()
def reinjection_control(cfg, backend):
    path = Path(cfg['output'])
    text = ('A scientific experiment compares a measured outcome with an explicit control. '
            'An autoregressive model predicts each next token using earlier context. ')*600
    ids = backend.tokenizer.encode(text)
    rows = []
    for length in cfg['lengths']:
        prefix = ids[:length]
        next_id = ids[length]
        out = backend.prefill(prefix)
        pairs = CacheExtractor.tensors(out.past_key_values, device='cpu', clone=True)
        checkpoint = path/'reinjection_snapshot.pt'
        torch.save(pairs,checkpoint)
        live = backend.forward([next_id],out.past_key_values)
        del out, pairs
        loaded = torch.load(checkpoint,weights_only=True)
        restored = CacheInjector.create(tuple((k.to(backend.device),v.to(backend.device)) for k,v in loaded))
        resumed = backend.forward([next_id], restored)
        exact = distributions(live.logits,resumed.logits)
        full = backend.prefill(prefix+[next_id])
        full_metrics = distributions(full.logits,resumed.logits)
        live_tokens,_,_ = backend.generate_from(live.past_key_values,live.logits,16)
        resumed_tokens,_,_ = backend.generate_from(resumed.past_key_values,resumed.logits,16)
        row = dict(context_length=length, live_vs_disk_reinjection=exact,
                   full_prefill_vs_incremental=full_metrics,
                   deterministic_continuation_identical=live_tokens==resumed_tokens,
                   continuation_tokens=live_tokens)
        rows.append(row)
        save_json(path/'controls.json',rows)
        print(f'control {length}: exact KL={exact["kl"]:.3g}, max diff={exact["max_logit_difference"]:.3g}, full KL={full_metrics["kl"]:.3g}',flush=True)
        assert exact['max_logit_difference'] < 1e-4 and exact['top1_agreement']==1 and live_tokens==resumed_tokens
        assert full_metrics['kl'] < 0.005 and full_metrics['top1_agreement']==1, 'Full-context numerical control failed'
    checkpoint.unlink()
    save_json(path/'controls_passed.json',dict(passed=True,device=backend.device,dtype=str(backend.dtype),lengths=cfg['lengths']))


@torch.inference_mode()
def rope_control(cfg, backend, role):
    captured = []
    attn = backend.model.model.layers[0].self_attn
    handle = attn.k_norm.register_forward_hook(lambda mod,args,out:captured.append(out.detach().transpose(1,2)))
    out = backend.prefill(backend.tokenizer.encode('Verify rotary key position transforms on an actual attention cache.'))
    handle.remove()
    k = out.past_key_values.layers[0].keys
    rotated = backend.rope(captured[0])
    recovered = backend.rope(k,inverse=True)
    roundtrip = backend.rope(recovered)
    result = dict(cached_keys_are_post_rope=True,
        pre_rope_hook_vs_cache_max=float((rotated-k.float()).abs().max()),
        pre_rope_hook_vs_inverse_cache_max=float((captured[0].float()-recovered).abs().max()),
        inverse_forward_roundtrip_max=float((roundtrip-k.float()).abs().max()),
        attention_forward_source=inspect.getsource(type(attn).forward))
    assert result['pre_rope_hook_vs_cache_max'] < 0.05
    assert result['inverse_forward_roundtrip_max'] < 1e-4
    save_json(Path(cfg['output'])/f'rope_control_{role}.json',result)
