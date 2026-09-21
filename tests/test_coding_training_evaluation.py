import importlib.util,json,threading,types
from pathlib import Path
import pytest
import torch
from gearshift.coding_control import write,sha,digest
from test_coding_training import TinyBackend,TinyMapper


def module():
    spec=importlib.util.spec_from_file_location('evaluation_worker',Path(__file__).parents[1]/'scripts/coding_evaluation_worker.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_exact_development_replay_rejects_changed_tokens_and_rng():
    m=module();row={'prompt_ids':[1],'reasoning_ids':[4,151668],'prefix_ids':[1,4],'bridge_ids':[151668],
        'natural_boundary':True,'reasoning_capped':False,'early_eos':False,'rng_initial':[1],'rng_after_reasoning':[3]}
    m.verify_replayed_history(row,dict(row))
    with pytest.raises(ValueError):m.verify_replayed_history({**row,'rng_after_reasoning':[9]},row)
    with pytest.raises(ValueError):m.verify_replayed_history({**row,'reasoning_ids':[5,151668]},row)


def test_confirmation_requires_frozen_headline_checkpoint_and_reserved_seed(tmp_path):
    m=module();selection=tmp_path/'selection.json'
    write(selection,{'confirmation_used_for_selection':False,'checkpoint_sha256':'chosen','objective':'natural_handoff_boundary'})
    spec={'stage':'confirmation','task_ids':['p/t'],'selection_path':str(selection),'selection_sha256':sha(selection)}
    cfg={'mapper':{'second_seed_confirmation_ids':[]}}
    candidate={'seed':20260915,'checkpoint_sha256':'chosen'}
    assert m.validate_stage_membership(spec,cfg,[{'task_id':'p/t'}],[candidate])['checkpoint_sha256']=='chosen'
    with pytest.raises(ValueError):m.validate_stage_membership(spec,cfg,[{'task_id':'p/t'}],[{**candidate,'checkpoint_sha256':'other'}])
    spec['task_ids']=['p/t','p/t']
    with pytest.raises(ValueError):m.validate_stage_membership(spec,cfg,[{'task_id':'p/t'}],[candidate])


def test_forty_task_selection_uses_pass_latency_then_step_and_is_immutable(tmp_path):
    m=module();root=tmp_path/'shard';identity={'mock':True};write(root/'identity.json',identity)
    write(root/'native_gate.json',{'passed':True,'identity_sha256':digest(identity)})
    candidates=[{'candidate_id':'a','step':256,'checkpoint_sha256':'aa','objective':'ordinary_continuation'},
        {'candidate_id':'b','step':384,'checkpoint_sha256':'bb','objective':'natural_handoff_boundary'}]
    receipts={};ids=[f'p/{i}' for i in range(40)]
    for index,tid in enumerate(ids):
        folder=root/'tasks'/tid.replace('/','__');write(folder/'C.json',{'raw':'unchanged'})
        write(folder/'complete.json',{'task_id':tid,'identity_sha256':digest(identity),'files':{'C.json':sha(folder/'C.json')},
            'candidates':{'a':{'passed':index<20,'source_inclusive_seconds':10.,'checkpoint_sha256':'aa'},
                'b':{'passed':index<21,'source_inclusive_seconds':20.,'checkpoint_sha256':'bb'}}})
        receipts[tid]=sha(folder/'complete.json')
    write(root/'complete.json',{'stage':'development_candidates','identity_sha256':digest(identity),'tasks':receipts,'candidates':candidates})
    result=m.select_development([root],tmp_path/'selected.json',ids)
    assert result['selected_candidate_id']=='b' and result['confirmation_used_for_selection'] is False
    with pytest.raises(ValueError,match='already frozen'):m.select_development([root],tmp_path/'selected.json',ids)
    with pytest.raises(ValueError,match='forty'):m.select_development([root],tmp_path/'incomplete.json',ids[:-1])


def test_complete_mock_confirmation_has_one_shared_source_and_zero_C_prefill(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'ROOT',tmp_path)
    import gearshift.coding_inference as inference
    import gearshift.coding_sandbox as sandbox
    tid='p/t';task={'task_id':tid,'prompt_ids':[1,2]}
    write(tmp_path/'data/coding_pilot_v1/visible/confirmation.json',[task])
    write(tmp_path/'data/coding_pilot_v1/private/confirmation.json',{tid:{'checker_only':True}})
    baseline=tmp_path/'baseline';write(baseline/'identity.json',{'reasoning_cap':24576})
    gate=tmp_path/'sandbox.json';write(gate,{'passed':True,'canonical_development':{'a':True,'b':True}})
    selection=tmp_path/'selection.json';write(selection,{'confirmation_used_for_selection':False,'checkpoint_sha256':'chosen','objective':'natural_handoff_boundary'})
    tiny=TinyMapper();candidate={'candidate_id':'c','checkpoint_sha256':'chosen','seed':20260915,'step':256,
        'objective':'natural_handoff_boundary','state_dict':tiny.state_dict()}
    monkeypatch.setattr(m,'verified_candidates',lambda spec:[candidate])
    monkeypatch.setattr(m,'AffineMapper',lambda source,receiver:TinyMapper())
    source,target=TinyBackend(3),TinyBackend(4);calls=[]
    def reason(backend,prompt,task_id,stream,cap,callback):
        calls.append(('reason',backend.name,stream))
        h={'task_id':task_id,'prompt_ids':prompt,'reasoning_ids':[151668],'prefix_ids':prompt,'bridge_ids':[151668],
            'natural_boundary':True,'reasoning_capped':False,'early_eos':False,'prefix_cache_length':len(prompt),
            'reasoning_seconds':2.,'rng_initial':[1],'rng_after_reasoning':[2]}
        with torch.no_grad():cache=backend.prefill_chunked(prompt).past_key_values
        callback([151668]);return h,cache
    def answer(backend,history,cache,task_id,stream,cap,callback):
        assert cache.get_seq_length()==len(history['prefix_ids']);calls.append(('answer',backend.name,stream))
        backend.input_token_count+=1
        return {'task_id':task_id,'answer_ids':[8,151645],'answer_text':'print(1)','answer_seconds':1.,'answer_capped':False}
    monkeypatch.setattr(inference,'reason',reason);monkeypatch.setattr(inference,'answer',answer)
    monkeypatch.setattr(inference,'memory_record',lambda:{'passed':True})
    monkeypatch.setattr(sandbox,'score',lambda code,private,guard:{'passed':True,'category':'pass','trusted_checker_complete':True})
    root=tmp_path/'results/coding_pilot_v1/mock/worker';root.mkdir(parents=True)
    spec={'stage':'confirmation','task_ids':[tid],'baseline_root':str(baseline),'sandbox_gate':str(gate),
        'selection_path':str(selection),'selection_sha256':sha(selection)}
    thread=threading.Thread(target=lambda:None);thread.start();published=[]
    c={'spec':spec,'root':root,'config':{'mapper':{}},'guard':lambda:None,'publish':lambda **kw:published.append(kw),
        'identity':{'mock':True},'halt':threading.Event(),'thread':thread}
    done=m.run(c,(source,target));folder=root/'tasks/p__t'
    assert done['task_count']==1 and published[-1]['state']=='complete'
    assert sum(row[0]=='reason' and row[2]=='source_reasoning' for row in calls)==1
    assert sum(row[0]=='reason' and row[2]=='small_reasoning' for row in calls)==1
    assert sum(row[0]=='answer' for row in calls)==4
    crow=json.loads((folder/'C.json').read_text());drow=json.loads((folder/'D.json').read_text())
    assert crow['historical_receiver_prefill_tokens']==0
    assert drow['historical_receiver_prefill_tokens']==2
    transaction=json.loads((folder/'complete.json').read_text())
    assert transaction['pass']=={'A':True,'B':True,'C':True,'D':True}
    assert done['tasks'][tid]==sha(folder/'complete.json')
