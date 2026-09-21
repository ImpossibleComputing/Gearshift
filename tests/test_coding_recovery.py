import json
from types import SimpleNamespace
import pytest
import torch
from gearshift.coding_recovery import Telemetry,memory_warnings,reason_instrumented
from gearshift.coding_inference import reason

class Cuda:
    def is_available(self):return True
    def mem_get_info(self):return 8*1024**3,100*1024**3
    def memory_allocated(self):return 20*1024**3
    def memory_reserved(self):return 92*1024**3
    def max_memory_allocated(self):return 90*1024**3
    def max_memory_reserved(self):return 93*1024**3
    def memory_stats(self):return {'num_ooms':0}
    def reset_peak_memory_stats(self):pass

class Cache:
    def __init__(self,n):self.n=n
    def get_seq_length(self):return self.n
class Backend:
    device='cpu';eos={99}
    tokenizer=SimpleNamespace(decode=lambda ids,**kw:str(ids))
    def prefill_chunked(self,ids):return self.forward(ids,None)
    def forward(self,ids,cache):
        n=len(ids)+(cache.n if cache else 0)
        return SimpleNamespace(past_key_values=Cache(n),logits=torch.arange(32,dtype=torch.float32).reshape(1,1,32)/20)

def test_peak_and_free_are_durable_warnings(tmp_path):
    t=Telemetry(tmp_path,Cuda());r=t.sample(task_id='task',sequence_length=21057,stage='generation')
    assert len(r['warnings'])==2 and r['allocated']<r['peak_allocated']
    saved=json.loads((tmp_path/'memory_telemetry.jsonl').read_text());assert saved['pid'] and saved['peak_reserved']
    t.reset('new_task');assert json.loads((tmp_path/'memory_latest.json').read_text())['peak_reset_scope']=='new_task'

def test_same_sampling_and_forward_schedule(tmp_path):
    b=Backend();old,cache=reason(b,[1,2],'t','source_reasoning',130)
    t=Telemetry(tmp_path);new,nc=reason_instrumented(b,[1,2],'t','source_reasoning',130,tmp_path/'run',t,lambda:None,lambda **k:None,old['reasoning_ids'][:65])
    for k in ['reasoning_ids','prefix_ids','rng_initial','rng_after_reasoning']:assert old[k]==new[k]
    assert nc.n==cache.n

def test_resume_saved_before_controlled_abort(tmp_path):
    calls=[]
    def guard():
        calls.append(1)
        if len(calls)==2:raise TimeoutError('controlled')
    with pytest.raises(TimeoutError):reason_instrumented(Backend(),[1,2],'t','source_reasoning',130,tmp_path/'run',Telemetry(tmp_path),guard,lambda **k:None)
    r=json.loads((tmp_path/'run/resume.json').read_text());assert r['state']=='aborted' and len(r['tokens'])==1 and r['rng_state'] and r['cache_sequence_length']==3

def test_changed_replay_is_distinct_failure(tmp_path):
    with pytest.raises(RuntimeError,match='diverged'):reason_instrumented(Backend(),[1,2],'t','source_reasoning',130,tmp_path/'run',Telemetry(tmp_path),lambda:None,lambda **k:None,[1000])
    assert json.loads((tmp_path/'run/replay_divergence.json').read_text())['exact_recovery'] is False
    assert not (tmp_path/'run/source_history.json').exists()

def test_instrumented_answer_preserves_rng_and_tokens(tmp_path):
    from gearshift.coding_inference import answer
    from gearshift.coding_recovery_answer import answer_instrumented
    b=Backend();h={'prefix_ids':[1,2],'bridge_ids':[3]}
    a=answer(b,h,Cache(2),'t','answer_small',130)
    r=answer_instrumented(b,h,Cache(2),'t','answer_small',130,tmp_path/'answer',Telemetry(tmp_path),lambda:None,lambda **kw:None)
    for key in ['answer_ids','rng_initial','rng_final','answer_capped']:assert a[key]==r[key]

def test_selection_uses_validation_and_earlier_tie_only():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from coding_recovery_train import choose_checkpoint
    curve=[{'step':s,'rows':[{}]*21,'validation_kl':kl,'checkpoint':{'x':s},'development_passes':p} for s,kl,p in [(1,.8,39),(8,.5,2),(32,.5,40)]]
    assert choose_checkpoint(curve)['step']==8
    curve[1]['rows']=[]
    assert choose_checkpoint(curve)['step']==32
    assert choose_checkpoint([]) is None

def test_last_committed_mapper_backed_up_before_failure(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import coding_recovery_train as train
    from gearshift.coding_control import write,sha
    monkeypatch.setattr(train,'ROOT',tmp_path)
    root=tmp_path/'results/run';root.mkdir(parents=True)
    (root/'mapper_latest.pt').write_bytes(b'committed model payload')
    write(root/'latest_checkpoint.json',{'step':7,'sha256':sha(root/'mapper_latest.pt')})
    train.preserve_latest(root)
    m=json.loads((root/'checkpoint_manifest.json').read_text());assert len(m['files'])==1
    path,expected=next(iter(m['files'].items()));assert sha(tmp_path/path)==expected
    assert json.loads((root/'final_unvalidated_checkpoint.json').read_text())['eligible_for_selection'] is False
