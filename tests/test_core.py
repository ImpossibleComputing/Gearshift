import math
import numpy as np
import torch
from types import SimpleNamespace
from gearshift.core import CacheExtractor, CacheInjector, ModelBackend, distributions, rotate_half, tensor_metrics
from gearshift.mapping import CacheAdapter
from gearshift.reasoning import numeric_answer


def test_cache_clone_is_independent_and_head_order_preserved():
    # Unequal heads/dimensions and identifiable coordinates expose axis transposition bugs.
    k=torch.arange(2*3*7*4).reshape(2,3,7,4).float()
    flat=CacheExtractor.flatten(k)
    assert torch.equal(flat[8],k[1,:,1,:].reshape(-1))
    one=k[:1]
    assert torch.equal(CacheExtractor.unflatten(CacheExtractor.flatten(one),3,4),one)
    cache=CacheInjector.create([(one,one+1000)])
    cache.layers[0].keys[0,0,0,0]=-99
    assert one[0,0,0,0]==0
    cache.update(torch.zeros(1,3,1,4),torch.ones(1,3,1,4),0)
    assert cache.get_seq_length()==8


def test_rotation_inverse_for_different_positions():
    x=torch.randn(1,3,7,8)
    theta=torch.randn(1,1,7,4).repeat(1,1,1,2)
    y=x*theta.cos()+rotate_half(x)*theta.sin()
    recovered=y*theta.cos()-rotate_half(y)*theta.sin()
    torch.testing.assert_close(x,recovered)


def test_distribution_metrics_known_values():
    a=torch.tensor([[[math.log(.8),math.log(.1),math.log(.04),math.log(.03),math.log(.02),math.log(.01)]]])
    b=torch.zeros_like(a)
    same=distributions(a,a)
    assert same['kl']==0 and same['top1_agreement']==1 and same['top5_overlap']==1
    actual=distributions(a,b)
    expected=sum(p*math.log(p*6) for p in [.8,.1,.04,.03,.02,.01])
    assert abs(actual['kl']-expected)<1e-6
    assert 0<=actual['js']<=math.log(2)


def test_constant_mean_baseline_has_zero_r2():
    y=torch.randn(100,8)
    m=tensor_metrics(y,y.mean(0).expand_as(y))
    assert abs(m['r2'])<1e-6
    assert tensor_metrics(y,y)['r2']==1


def test_answer_grading():
    assert numeric_answer('The result is $1,234.00.')==numeric_answer('1234')
    assert numeric_answer('Work: 9+3. \\boxed{-12.5}')==numeric_answer('-12.500')
    assert numeric_answer('No numerical answer.') is None


def test_adapter_with_unequal_layers_heads_and_head_dimensions():
    source=SimpleNamespace(device='cpu',config=SimpleNamespace(num_hidden_layers=3,num_key_value_heads=2,head_dim=4))
    target=SimpleNamespace(device='cpu',dtype=torch.float32,config=SimpleNamespace(num_hidden_layers=2,num_key_value_heads=3,head_dim=6))
    pairs=tuple((torch.arange(56).reshape(1,2,7,4).float()+100*i,
                 torch.arange(56).reshape(1,2,7,4).float()-100*i) for i in range(3))
    maps={name:[dict(sources=[src],weight=torch.arange(144).reshape(8,18).float()/144,
                     bias=torch.arange(18).float()) for src in [2,0]] for name in ['k_post','v']}
    output=CacheAdapter(source,target,maps).transform(pairs,'post')
    assert len(output)==2
    for layer,src in enumerate([2,0]):
        for kind,name in enumerate(['k_post','v']):
            assert output[layer][kind].shape==(1,3,7,6)
            for position in [0,3,6]:
                expected=pairs[src][kind][0,:,position,:].reshape(8)@maps[name][layer]['weight']+maps[name][layer]['bias']
                torch.testing.assert_close(output[layer][kind][0,:,position,:].reshape(18),expected)
    assert CacheInjector.create(output).get_seq_length()==7


def test_generation_honors_all_generation_config_eos_tokens(monkeypatch):
    model=torch.nn.Module()
    model.config=SimpleNamespace(eos_token_id=9)
    model.generation_config=SimpleNamespace(eos_token_id=[9,10])
    monkeypatch.setattr('gearshift.core.AutoTokenizer.from_pretrained',lambda *args,**kwargs:SimpleNamespace())
    monkeypatch.setattr('gearshift.core.AutoModelForCausalLM.from_pretrained',lambda *args,**kwargs:model)
    backend=ModelBackend('test-only',device='cpu')
    assert backend.eos=={9,10}
    logits=torch.zeros(1,1,12); logits[0,0,10]=1
    backend.forward=lambda ids,cache:SimpleNamespace(logits=logits,past_key_values=cache)
    tokens,_,_=backend.generate_from(object(),logits,3)
    assert tokens==[10]
