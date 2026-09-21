"""CPU synthetic invariant tests, NOT Qwen3/CUDA numerical validation."""
import json
import math
import pytest
import torch
from gearshift.sparse_repair import (HistoryLayout, SelectionIdentity, WorkingSetTrace,
                                    intervene, native_reasoning_mass,
                                    select_indices, selected_accounting)


def fixture():
    layout = HistoryLayout(2, 7, page_size=4)
    native = torch.arange(42, dtype=torch.float32).reshape(1, 2, 7, 3)
    mapped = native + 1000
    live = torch.cat((native[..., :2, :], mapped[..., 2:, :], native[..., :2, :] + 100), -2)
    return layout, (native, native + 100), (mapped, mapped + 100), (live, live + 100)


def attention(q, k, v, allowed=None):
    groups = q.shape[1] // k.shape[1]
    k, v = (x.repeat_interleave(groups, 1) for x in (k, v))
    if allowed is not None:
        allowed = allowed.repeat_interleave(groups, 1)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                                          scale=q.shape[-1]**-.5)


@pytest.mark.parametrize('p,want', [(0,0),(.05,1),(.1,1),(.25,2),(1,5)])
def test_budget_ceil(p, want):
    assert HistoryLayout(2,7).count(p) == want


@pytest.mark.parametrize('p', [-1, 1.01, float('inf'), float('nan')])
def test_bad_budget(p):
    with pytest.raises(ValueError): HistoryLayout(1,2).count(p)


def test_zero_reasoning_is_valid():
    layout = HistoryLayout(2,2)
    mass = torch.empty(2,0)
    for method in ['native_mass','random','recent']:
        indices = select_indices(mass,layout,.1,method=method,identity=SelectionIdentity('x',1,0,0))
        assert indices.shape == (2,0)


def test_gqa_sum_of_probabilities_uses_full_context_normalizer_and_own_live_tail():
    layout = HistoryLayout(1,4)
    q = torch.tensor([[[[1.,2.]],[[2.,-1.]],[[3.,0.]],[[0.,1.]]]])
    native = torch.arange(16,dtype=torch.float32).reshape(1,2,4,2)/10
    live = torch.cat((native + 500, torch.tensor([[[[.5,.1]],[[.1,.8]]]])),2)
    mass = native_reasoning_mass(q,native,live,layout,scaling=.7,query_position=4)
    keys = torch.cat((native,live[...,4:,:]),2).repeat_interleave(2,1)
    probs = torch.softmax(q @ keys.transpose(-1,-2)*.7,dim=-1)
    expected = probs.reshape(1,2,2,1,5).sum(2)[0,:,0,1:4]
    torch.testing.assert_close(mass,expected)
    # The denominator is all history + own current state, not reasoning-only.
    assert (mass.sum(-1) < 2).all()
    bad_tail = live.clone();bad_tail[...,4:,:] += 100
    assert not torch.equal(mass,native_reasoning_mass(q,native,bad_tail,layout,scaling=.7,query_position=4))


def test_mask_and_current_query_validation():
    layout, native, _, live = fixture()
    q = torch.ones((1,4,1,3))
    allowed = torch.ones((1,1,1,9),dtype=torch.bool);allowed[...,3] = False
    mass = native_reasoning_mass(q,native[0],live[0],layout,scaling=1,query_position=8,attention_mask=allowed)
    assert (mass[:,1] == 0).all()
    for bad in [q.expand(1,4,2,3), q[:,:3]]:
        with pytest.raises(ValueError):native_reasoning_mass(bad,native[0],live[0],layout,scaling=1,query_position=8)
    with pytest.raises(ValueError):native_reasoning_mass(q,native[0],live[0],layout,scaling=1,query_position=7)
    with pytest.raises(ValueError):native_reasoning_mass(q,native[0],live[0],layout,scaling=1,query_position=8,attention_mask=torch.zeros_like(allowed))


def test_deterministic_ties_absolute_positions_per_head():
    layout = HistoryLayout(2,7)
    mass = torch.tensor([[1.,1.,1.,1.,1.],[0.,1.,2.,3.,4.]])
    assert select_indices(mass,layout,.25).tolist() == [[2,3],[5,6]]


def test_random_reproducible_identity_not_global_rng_or_execution_order():
    layout = HistoryLayout(10,110)
    mass = torch.ones(2,100)
    ident = SelectionIdentity('task',319,7,11)
    one = select_indices(mass,layout,.1,method='random',identity=ident)
    torch.manual_seed(999);torch.rand(55)
    select_indices(mass,layout,.1,method='random',identity=SelectionIdentity('other',1,0,0))
    two = select_indices(mass,layout,.1,method='random',identity=ident)
    assert torch.equal(one,two)
    assert not torch.equal(one[0],one[1])
    other = select_indices(mass,layout,.1,method='random',identity=SelectionIdentity('task',319,7,12))
    assert not torch.equal(one,other)
    assert select_indices(mass,layout,.1,method='recent').tolist() == [list(range(100,110))]*2


def test_complete_pairs_no_alias_no_accumulation_and_protected_positions():
    layout, native, mapped, live = fixture()
    originals = [x.clone() for pair in [native,mapped,live] for x in pair]
    first = torch.tensor([[2],[4]])
    second = torch.tensor([[3],[5]])
    a = intervene(live,native,mapped,layout,first,'R')
    # Even a deliberately polluted live history cannot accumulate previous repairs.
    b = intervene((a.keys,a.values),native,mapped,layout,second,'R')
    for kind, output in enumerate((b.keys,b.values)):
        for head in range(2):
            assert torch.equal(output[0,head,first[head,0]],mapped[kind][0,head,first[head,0]])
            assert torch.equal(output[0,head,second[head,0]],native[kind][0,head,second[head,0]])
        assert torch.equal(output[...,:2,:],live[kind][...,:2,:])
        assert torch.equal(output[...,7:,:],live[kind][...,7:,:])
        assert output.data_ptr() not in [x.data_ptr() for pair in (native,mapped,live) for x in pair]
    assert b.allowed.all()
    a.keys.zero_();b.values.zero_()
    for x, original in zip([x for pair in (native,mapped,live) for x in pair],originals):
        assert torch.equal(x,original)


@pytest.mark.parametrize('mode',['N','M'])
def test_sparse_mask_preserves_native_prompt_own_answer_and_current(mode):
    layout, native, mapped, live = fixture()
    result = intervene(live,native,mapped,layout,torch.tensor([[2],[6]]),mode)
    assert result.allowed[0,0,0].tolist() == [True,True,True,False,False,False,False,True,True]
    assert result.allowed[0,1,0].tolist() == [True,True,False,False,False,False,True,True,True]
    assert torch.equal(result.keys[...,:2,:],live[0][...,:2,:])
    assert torch.equal(result.keys[...,7:,:],live[0][...,7:,:])


def test_repair_recomputes_weights_not_native_weight_reuse():
    torch.manual_seed(45)
    layout=HistoryLayout(2,6)
    native=tuple(torch.randn(1,2,6,3) for _ in range(2))
    mapped=tuple(torch.randn(1,2,6,3) for _ in range(2))
    live=tuple(torch.randn(1,2,7,3) for _ in range(2))
    q=torch.randn(1,4,1,3)
    index=torch.tensor([[2],[4]])
    r=intervene(live,native,mapped,layout,index,'R')
    expected=attention(q,r.keys,r.values,r.allowed)
    native_keys=torch.cat((live[0][...,:2,:],native[0][...,2:,:],live[0][...,6:,:]),2)
    wrong=attention(q,native_keys,r.values,r.allowed)
    assert not torch.allclose(expected,wrong)


def test_synthetic_zero_full_controls_and_sparse_renormalization():
    torch.manual_seed(15)
    layout, native, mapped, live=fixture()
    q=torch.randn(1,4,1,3)
    mass=torch.ones(2,5)
    none=select_indices(mass,layout,0)
    full=select_indices(mass,layout,1)
    zero=intervene(live,native,mapped,layout,none,'R')
    assert torch.equal(zero.keys,live[0])
    torch.testing.assert_close(attention(q,zero.keys,zero.values,zero.allowed),attention(q,*live),rtol=0,atol=0)
    for mode in ['N','R']:
        all_=intervene(live,native,mapped,layout,full,mode)
        matched=tuple(torch.cat((l[...,:2,:],n[...,2:,:],l[...,7:,:]),2) for l,n in zip(live,native))
        torch.testing.assert_close(attention(q,all_.keys,all_.values,all_.allowed),attention(q,*matched),rtol=0,atol=0)
    indices=torch.tensor([[2,4],[3,6]])
    sparse=intervene(live,native,mapped,layout,indices,'N')
    dense=attention(q,sparse.keys,sparse.values,sparse.allowed)
    # Explicit independent group gathers must match dense-mask normalization.
    rows=[]
    for head in range(2):
        kept=torch.tensor([0,1]+indices[head].tolist()+[7,8])
        k=sparse.keys[:,head:head+1].index_select(2,kept)
        v=sparse.values[:,head:head+1].index_select(2,kept)
        rows.append(attention(q[:,2*head:2*head+2],k,v))
    torch.testing.assert_close(dense,torch.cat(rows,1),rtol=1e-6,atol=1e-5)


@pytest.mark.parametrize('bad',[torch.tensor([[1],[3]]),torch.tensor([[7],[3]]),torch.tensor([[2,2],[3,4]]),torch.tensor([[3,2],[3,4]])])
def test_reject_outside_duplicate_unsorted_indices(bad):
    layout,native,mapped,live=fixture()
    with pytest.raises(ValueError):intervene(live,native,mapped,layout,bad,'R')


def test_pair_and_page_accounting_does_not_union_heads():
    layout=HistoryLayout(2,20,page_size=4)
    indices=torch.tensor([[3,4],[3,8]])
    info=selected_accounting(indices,layout,heads=2,head_dim=3,element_size=2)
    assert info['selected_unique_kv_pairs']==4 # token 3 is a DIFFERENT pair per head
    assert info['logical_selected_bytes']==48
    assert info['per_head_page_count_estimated']==4
    assert info['all_heads_shared_page_count_estimated']==3
    assert info['all_heads_shared_page_bytes_estimated']==3*4*12*2
    assert info['physical_page_or_traffic_measurement'] is False


def test_bounded_trace_full_union_and_resume():
    layout=HistoryLayout(2,7,page_size=4)
    trace=WorkingSetTrace(layout,2,3,2,trace_steps=(0,3))
    for step in range(5):
        indices=torch.tensor([[step+2],[step+2]])
        trace.update(step,indices,torch.ones(2,5)/9)
    summary=trace.summary()
    assert summary['cumulative_unique_kv_pairs']==10
    assert summary['peak_instantaneous_selected_pairs']==2
    assert summary['cumulative_reasoning_pair_fraction']==1
    assert len(trace.records)==2
    restored=WorkingSetTrace.from_state_dict(json.loads(json.dumps(trace.state_dict())))
    assert restored.summary()==summary
    with pytest.raises(ValueError):restored.update(4,torch.tensor([[2],[2]]))
    restored.update(5,torch.tensor([[2],[3]]))
    assert restored.summary()['cumulative_unique_kv_pairs']==10


def tiny_qwen_fixture():
    from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(14)
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                      max_position_embeddings=128,attention_dropout=0.0)
    config._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(config).eval()
    model.requires_grad_(False)
    with torch.no_grad():out=model(input_ids=torch.tensor([[1,2,3,4,5,6,7]]),use_cache=True)
    native=tuple((layer.keys.clone(),layer.values.clone()) for layer in out.past_key_values.layers)
    mapped=tuple((k+.2,v-.3) for k,v in native)
    layout=HistoryLayout(2,7)
    hybrid=tuple(tuple(torch.cat((n[...,:2,:],m[...,2:,:]),2) for n,m in zip(np,mp)) for np,mp in zip(native,mapped))
    def cache(pairs):
        result=DynamicCache()
        for layer,(k,v) in enumerate(pairs):result.update(k.clone(),v.clone(),layer)
        return result
    def forward(pair_cache, token=8):
        past=pair_cache.get_seq_length()
        with torch.no_grad():
            return model(input_ids=torch.tensor([[token]]),past_key_values=pair_cache,use_cache=True,
                         attention_mask=torch.ones(1,past+1,dtype=torch.long),
                         cache_position=torch.tensor([past]),position_ids=torch.tensor([[past]]))
    return model,native,mapped,hybrid,layout,cache,forward


def test_qwen_cpu_random_weights_disabled_hook_exact_deployed_path():
    from gearshift.sparse_repair import SparseRepairController
    model,native,mapped,_,layout,cache,forward=tiny_qwen_fixture()
    baseline=forward(cache(native))
    callback=[]
    controller=SparseRepairController(model,native,mapped,layout,mode='disabled',profile=True,
                    probe_callback=lambda layer,step,*args:callback.append((layer,step)))
    with controller:
        hooked=forward(cache(native))
        torch.testing.assert_close(hooked.logits,baseline.logits,rtol=0,atol=0)
        forward(hooked.past_key_values,9)
    assert model.config._attn_implementation=='sdpa'
    assert callback==[(0,0),(1,0),(0,1),(1,1)]
    assert controller.summary()['attention_calls_by_layer']=={'0':2,'1':2}
    assert controller.summary()['stage_seconds']['attention']>0


@pytest.mark.parametrize('mode,p,initial,reference',[
    ('N',1,'native','native'),('R',0,'hybrid','hybrid'),('R',1,'hybrid','native'),
])
def test_qwen_cpu_random_weights_dense_full_and_zero_controls(mode,p,initial,reference):
    from gearshift.sparse_repair import SparseRepairController
    model,native,mapped,hybrid,layout,cache,forward=tiny_qwen_fixture()
    pairs={'native':native,'hybrid':hybrid}
    originals=[t.clone() for pair in native+mapped for t in pair]
    dense=forward(cache(pairs[reference]))
    controller=SparseRepairController(model,native,mapped,layout,mode=mode,fraction=p,
                                    task_id='synthetic',draw_seed=9)
    with controller:
        result=forward(cache(pairs[initial]))
    torch.testing.assert_close(result.logits,dense.logits,rtol=0,atol=0)
    for tensor,old in zip([t for pair in native+mapped for t in pair],originals):
        assert torch.equal(tensor,old)
    expected_count=2*2*layout.count(p)
    assert controller.summary()['cumulative_unique_pairs_all_layers']==expected_count
    assert len(controller.trace_records())==2


def test_controller_restore_after_error_and_reset_state():
    from gearshift.sparse_repair import SparseRepairController
    model,native,mapped,hybrid,layout,cache,forward=tiny_qwen_fixture()
    controller=SparseRepairController(model,native,mapped,layout,mode='N',fraction=.1)
    with pytest.raises(RuntimeError):
        with controller:
            forward(cache(hybrid))
            raise RuntimeError('intentional')
    assert model.config._attn_implementation=='sdpa'
    controller.set_condition('R',fraction=.25,selector='recent',reset=True)
    assert controller.summary()['cumulative_unique_pairs_all_layers']==0
    with controller:
        out=forward(cache(hybrid))
        forward(out.past_key_values,9)
    assert controller.summary()['oracle_assisted'] is False
    assert controller.summary()['attention_calls_by_layer']=={'0':2,'1':2}


def test_trace_head_subset_does_not_subset_accounting_or_resume():
    trace=WorkingSetTrace(HistoryLayout(2,7),2,3,2,trace_steps=(0,),trace_heads=(1,))
    trace.update(0,torch.tensor([[2,3],[4,5]]))
    record=trace.records[0]
    assert record['traced_kv_heads']==[1]
    assert record['selected_indices_absolute']==[[4,5]]
    assert record['selected_unique_kv_pairs']==4
    assert trace.summary()['cumulative_unique_kv_pairs']==4
    restored=WorkingSetTrace.from_state_dict(json.loads(json.dumps(trace.state_dict())))
    assert restored.trace_heads==(1,)
    assert restored.summary()==trace.summary()


def test_controller_trace_layer_subset_keeps_all_layer_union():
    from gearshift.sparse_repair import SparseRepairController
    model,native,mapped,hybrid,layout,cache,forward=tiny_qwen_fixture()
    controller=SparseRepairController(model,native,mapped,layout,mode='R',fraction=.25,
                                    trace_layers=(1,),trace_heads=(0,))
    with controller:forward(cache(hybrid))
    records=controller.trace_records()
    assert records['0']==[]
    assert records['1'][0]['traced_kv_heads']==[0]
    assert controller.summary()['cumulative_unique_pairs_all_layers']==2*2*2
