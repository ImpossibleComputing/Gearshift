import pytest
from gearshift.early_handoff import *

def history(tokens=None):
 return {'generated_ids':[11,22,33,44,55,66,77,CLOSING] if tokens is None else tokens,'natural_boundary':True,'input_ids_sha256':digest([1,2]),'active_seconds':20,'prefill_seconds':1,'token_elapsed_seconds':list(range(2,10))}
@pytest.mark.parametrize('fraction,cut',[(0,0),(10,0),(25,1),(50,3),(75,5),(100,7)])
def test_exact_floor_boundary_no_semantic_snapping(fraction,cut):
 h=history();view=handoff([1,2],h,fraction)
 assert view['source_index']==cut
 assert view['input_ids']==[1,2]+reasoning_body(h)[:cut]+([CLOSING] if fraction==100 else [])
 assert view['answer_only'] is (fraction==100)
 assert h['generated_ids'][-1]==CLOSING

def test_future_reasoning_and_answer_never_exposed():
 h=history();h['source_answer']='secret future answer'
 assert handoff([1,2],h,25)['input_ids']==[1,2,11]
 assert 'source_answer' not in handoff([1,2],h,25)

def test_original_prompt_identity_checked():
 with pytest.raises(ValueError):handoff([9,2],history(),50)

def test_capped_trace_retains_every_token():
 h=history([11,22,33]);h['natural_boundary']=False
 assert handoff([1,2],h,100)['input_ids']==[1,2,11,22,33,CLOSING]

def test_no_premature_closing():
 with pytest.raises(ValueError):reasoning_body(history([11,CLOSING,22,CLOSING]))

def test_original_context_bound():
 assert continuation_cap(20000)==4576
 with pytest.raises(ValueError):continuation_cap(24577)

def test_matched_but_independent_streams():
 assert seed('synthetic',0,'receiver_answer')==seed('synthetic',0,'receiver_answer')
 assert len({seed('synthetic',i,s) for i in range(3) for s in ['receiver_answer','source_answer']})==6

def test_prefix_accounting_not_linear_token_extrapolation():
 h=history();h['token_elapsed_seconds']=[1,2,4,8,9,10,12,20]
 assert source_prefix_seconds(h,3,50)==4
 assert source_prefix_seconds(h,7,100)==20
 assert source_prefix_seconds(h,0,0)==0

def test_finished_evidence_is_immutable(tmp_path):
 p=tmp_path/'done.json';write(p,{'x':1},True);write(p,{'x':1},True)
 with pytest.raises(ValueError):write(p,{'x':2},True)

def test_ledger_rejects_overruns_and_nonfinite_values():
 d={'soft_usd':1000,'hard_usd':1500,'new_compute_usd':0,'reserved_usd':0,'cleanup_reserve_usd':50};assert validate_ledger(d)==50
 for value in [1450,float('nan'),-1]:
  with pytest.raises(ValueError):validate_ledger({**d,'reserved_usd':value})

class FakeCache:
 def __init__(self,n):self.n=n
 def get_seq_length(self):return self.n

class FakeBackend:
 def __init__(self,emission=None):
  import torch
  self.torch=torch;self.role='source';self.device='cpu';self.calls=0;self.emission=emission
 def sync(self):pass
 def memory(self):return {}
 def rng(self,s):return self.torch.Generator().manual_seed(s)
 def prefill(self,ids):return self.forward(ids)
 def forward(self,ids,cache=None):
  from types import SimpleNamespace
  self.calls+=1
  return SimpleNamespace(logits=self.torch.zeros((1,1,1)),past_key_values=FakeCache((cache.n if cache else 0)+len(ids)))
 def sample(self,logits,rng):return self.emission if self.emission is not None else int(self.torch.randint(10,30,(1,),generator=rng))

@pytest.mark.parametrize('terminal,cached',[(CLOSING,2),(151645,3)])
def test_reasoning_terminal_cache_contract(tmp_path,terminal,cached):
 from scripts.early_handoff_worker import generate
 b=FakeBackend(terminal);r,o=generate(b,[1,2],10,123,'reason',tmp_path/'draw',tmp_path)
 assert r['generated_ids']==[terminal] and o.past_key_values.get_seq_length()==cached
 assert r['natural_boundary'] is (terminal==CLOSING)

def test_complete_never_rerolled(tmp_path):
 from scripts.early_handoff_worker import generate
 b=FakeBackend();r,_=generate(b,[1,2],4,123,'reason',tmp_path/'draw',tmp_path)
 calls=b.calls;r2,o=generate(b,[1,2],4,123,'reason',tmp_path/'draw',tmp_path)
 assert r==r2 and o is None and b.calls==calls

def test_exact_committed_token_rng_recovery(tmp_path,monkeypatch):
 from scripts import early_handoff_worker as w
 reference,_=w.generate(FakeBackend(),[1,2],145,777,'reason',tmp_path/'reference',tmp_path)
 def interrupt(storage):
  if (tmp_path/'interrupted/checkpoint.json').exists():raise InterruptedError('synthetic crash')
 monkeypatch.setattr(w,'stop_requested',interrupt)
 with pytest.raises(InterruptedError):w.generate(FakeBackend(),[1,2],145,777,'reason',tmp_path/'interrupted',tmp_path)
 monkeypatch.setattr(w,'stop_requested',lambda s:None)
 recovered,_=w.generate(FakeBackend(),[1,2],145,777,'reason',tmp_path/'interrupted',tmp_path)
 assert recovered['generated_ids']==reference['generated_ids'] and recovered['rng_state']==reference['rng_state']
 assert recovered['recovery_attempts'][0]['committed_tokens']==128

def test_answer_recovery_reuses_original_native_bridge_shape(tmp_path,monkeypatch):
 from scripts import early_handoff_worker as w
 class ShapeBackend(FakeBackend):
  def prefill(self,ids):raise AssertionError('Answer recovery must not re-prefill the reasoning history')
 b=ShapeBackend();initial=b.forward([1,2,3,CLOSING])
 reference,_=w.generate(b,[1,2,3,CLOSING],145,777,'answer',tmp_path/'reference',tmp_path,initial)
 def interrupt(storage):
  if (tmp_path/'interrupted/checkpoint.json').exists():raise InterruptedError('synthetic crash')
 monkeypatch.setattr(w,'stop_requested',interrupt)
 with pytest.raises(InterruptedError):w.generate(b,[1,2,3,CLOSING],145,777,'answer',tmp_path/'interrupted',tmp_path,b.forward([1,2,3,CLOSING]))
 monkeypatch.setattr(w,'stop_requested',lambda s:None)
 recovered,_=w.generate(b,[1,2,3,CLOSING],145,777,'answer',tmp_path/'interrupted',tmp_path,b.forward([1,2,3,CLOSING]))
 assert recovered['generated_ids']==reference['generated_ids'] and recovered['rng_state']==reference['rng_state']

def analysis_rows():
 rows=[]
 for task in ['synthetic_a','synthetic_b']:
  for ci,c in enumerate(CONDITIONS):
   for draw in range(3):
    rows.append({'task_id':task,'condition':c,'draw':draw,'passed':ci>2,'missing':False,'single_output_stage_sum_seconds':10+ci,'failure_category':'pass' if ci>2 else 'test_assertion','receiver_reasoning_tokens':12,'source_token_index':ci})
 return rows

def test_analysis_clusters_tasks_and_preserves_draws():
 from scripts.early_handoff_report import analyze
 s,t=analyze(analysis_rows(),['synthetic_a','synthetic_b'])
 assert s['tasks']==2 and s['draws']==42 and len(t)==14
 assert s['contrasts']['H50-SMALL_ONLY']['difference']==1
 assert s['contrasts']['H50-SMALL_ONLY']['ci95']==[1,1]
 assert s['observed_quality_latency_frontier']==['SMALL_ONLY','H50']
 assert s['oracle']['mean_success']==1

def test_missing_is_not_recast_as_observed_failure():
 from scripts.early_handoff_report import analyze
 rows=analysis_rows();rows[9].update(passed=None,missing=True)
 s,_=analyze(rows,['synthetic_a','synthetic_b'])
 assert s['conditions']['H50']['pass_rate'] is None
 assert s['contrasts']['H50-SMALL_ONLY']['difference'] is None
 assert s['contrasts']['H50-SMALL_ONLY']['paired_task_gains'] is None
 assert s['oracle']['mean_success'] is None

def test_public_raw_field_gate():
 from scripts.early_handoff_hygiene import forbidden
 assert forbidden({'prompt_ids': [1,2]})
 assert forbidden({'nested':list(range(24))})
 assert not forbidden({'task_id':'synthetic','source_token_index':24,'sha256':'0'*64,'ci95':[0,1]})

def test_generation_seal_idempotent_but_immutable(tmp_path):
 from scripts.early_handoff_worker import seal
 tasks=[f'synthetic_{i}' for i in range(40)]
 for task in tasks:
  for c in CONDITIONS:
   for draw in range(3):
    for name in ['complete.json','answer.json']:
     write(tmp_path/'raw'/task/c/f'answer_{draw}'/name,{'synthetic':True})
 d={'population':{'task_ids':tasks}}
 seal(tmp_path,d);original=(tmp_path/'control/generation_seal.json').read_bytes()
 seal(tmp_path,d);assert (tmp_path/'control/generation_seal.json').read_bytes()==original
 write(tmp_path/'raw'/task/'SMALL_ONLY/answer_0/answer.json',{'synthetic':False})
 with pytest.raises(ValueError):seal(tmp_path,d)
