import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from gearshift.coding_control_calibration import float_arithmetic_audit,within_cache_tolerance

def test_arithmetic_audit_restores_all_weights_and_forward_methods():
    torch.manual_seed(9)
    cfg=Qwen3Config(vocab_size=96,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128)
    model=Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval().requires_grad_(False)
    values={n:p.detach().clone() for n,p in model.named_parameters()}
    ids=torch.tensor([[1,2,3,4]])
    with torch.no_grad():before=model(ids).logits
    with pytest.raises(RuntimeError):
        with float_arithmetic_audit(model):
            with torch.no_grad():audited=model(ids)
            assert audited.logits.dtype==torch.float32
            assert audited.past_key_values.layers[0].keys.dtype==torch.float32
            raise RuntimeError('exercise exception restoration')
    with torch.no_grad():after=model(ids).logits
    assert torch.equal(before,after)
    assert all(torch.equal(values[n],p) and p.dtype==torch.bfloat16 for n,p in model.named_parameters())

def test_original_cache_tolerance_is_not_relaxed():
    assert within_cache_tolerance({'max_abs':0,'kl':0,'top1_equal':True})
    assert not within_cache_tolerance({'max_abs':.3125,'kl':.00018576,'top1_equal':True})
