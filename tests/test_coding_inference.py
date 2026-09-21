from types import SimpleNamespace
import torch
from gearshift.coding_inference import reason,answer,generator,sample

class Cache:
    def __init__(self,n):self.n=n
    def get_seq_length(self):return self.n

class Fake:
    device='cpu'
    eos={151645,151643}
    tokenizer=SimpleNamespace(decode=lambda ids,**kw:str(ids))
    def __init__(self,emissions):self.emissions=iter(emissions);self.fed=[]
    def out(self,n):
        logits=torch.full((1,1,151669),-10000.)
        logits[0,0,next(self.emissions)]=10000.
        return SimpleNamespace(past_key_values=Cache(n),logits=logits)
    def prefill_chunked(self,ids):self.fed.extend(ids);return self.out(len(ids))
    def forward(self,ids,cache):self.fed.extend(ids);return self.out(cache.n+len(ids))

def test_natural_closing_token_emitted_but_not_cached_until_answer():
    b=Fake([42,151668,7,151645])
    history,cache=reason(b,[10,11],'t','source_reasoning',10)
    assert history['prefix_ids']==[10,11,42] and cache.n==3
    assert b.fed==[10,11,42] and history['natural_boundary']
    result=answer(b,history,cache,'t','answer_large',10)
    assert b.fed==[10,11,42,151668,7]
    assert result['answer_ids']==[7,151645]

def test_capped_final_reasoning_token_is_cached_exactly_once():
    b=Fake([42,43,44,7,151645])
    history,cache=reason(b,[10],'t','source_reasoning',2)
    assert history['prefix_ids']==[10,42,43] and cache.n==3
    assert history['reasoning_capped'] and b.fed==[10,42,43]
    answer(b,history,cache,'t','answer_small',10)
    assert b.fed==[10,42,43,151668,7]

def test_receiver_streams_independent_of_scheduling():
    a=generator('t','answer_small','cpu');b=generator('t','answer_small','cpu')
    logits=torch.zeros(1,1,30)
    expected=[sample(logits,a) for _ in range(30)]
    assert expected==[sample(logits,b) for _ in range(30)]
