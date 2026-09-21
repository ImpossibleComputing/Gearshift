import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,DynamicCache
from gearshift.coding_gradients import continuation
from gearshift.core import CacheExtractor,CacheInjector

def test_checkpoint_recomputation_keeps_prefix_immutable_and_matches_native():
    torch.manual_seed(4)
    config=Qwen3Config(vocab_size=96,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128)
    config._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(config).eval().requires_grad_(False)
    prefix=torch.tensor([[1,2,3,4,5,6,7,8]]);suffix=torch.tensor([[9,10,11,12]])
    with torch.no_grad():native=model(prefix,use_cache=True)
    pairs=tuple(tuple(t.detach().clone().requires_grad_(True) for t in pair) for pair in CacheExtractor.tensors(native.past_key_values))
    snapshot=tuple(tuple(t.detach().clone() for t in pair) for pair in pairs)
    with torch.no_grad():expected=model(suffix,past_key_values=CacheInjector.create(snapshot),use_cache=True).logits
    actual=continuation(model,pairs,suffix,checkpoint_layers=True)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    actual.square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for pair in pairs for t in pair)
    assert all(p.grad is None for p in model.parameters())
    for pair,old in zip(pairs,snapshot):
        for t,s in zip(pair,old):torch.testing.assert_close(t,s,atol=0,rtol=0)
    assert native.past_key_values.get_seq_length()==prefix.shape[1]
