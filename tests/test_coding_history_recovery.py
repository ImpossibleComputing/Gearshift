import importlib.util,json,threading
from pathlib import Path
import pytest
import torch
from gearshift.coding_control import write,sha,digest
from gearshift.coding_history_recovery import run_recovered,parent_for
from test_coding_training import TinyBackend


def fixture(tmp_path,monkeypatch,*,bad_prefix=False,bad_rng=False):
    import gearshift.coding_inference as inference
    path=Path(__file__).parents[1]/'scripts/coding_history_worker.py'
    spec=importlib.util.spec_from_file_location('history_under_recovery_test',path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);monkeypatch.setattr(worker,'ROOT',tmp_path)
    ids=['p/done','p/partial'];tasks=[{'task_id':tid,'prompt_ids':[1]} for tid in ids]
    write(tmp_path/'data/coding_pilot_v1/visible/training.json',tasks);write(tmp_path/'data/coding_pilot_v1/visible/validation.json',[])
    write(tmp_path/'baseline/identity.json',{'reasoning_cap':24576})
    shared={k:{} for k in ['baseline','implementation','data_identity']}
    shared.update({k:'fixed' for k in ['config_sha256','training_protocol_sha256','memory_gate_sha256','memory_identity_sha256']})
    original={**shared,'stage_identity':'parent'};current={**shared,'stage_identity':'recovered'}
    parent=tmp_path/'results/interrupted';root=tmp_path/'results/recovered';root.mkdir(parents=True)
    write(parent/'identity.json',original);write(parent/'native_gate.json',{'passed':True,'identity_sha256':digest(original)})
    def history(tid,seconds):
        tokens=[3,4,151668]
        return {'task_id':tid,'prompt_ids':[1],'reasoning_ids':tokens,'prefix_ids':[1,3,4],'bridge_ids':[151668],
            'natural_boundary':True,'reasoning_capped':False,'early_eos':False,'prefix_cache_length':3,
            'seed':7,'rng_initial':[1],'rng_after_reasoning':[2],'reasoning_seconds':seconds}
    def answer(tid,seconds):
        return {'task_id':tid,'answer_ids':[8,151645],'answer_seed':9,'rng_initial':[1],'rng_final':[3],
            'answer_ended_eos':True,'answer_capped':False,'bridge_token_count':1,'answer_seconds':seconds,
            'provenance':'Native frozen source A, answer_large stream, one sampled answer; no correctness filtering.'}
    done=parent/'tasks/p__done';write(done/'source_history.json',history(ids[0],100));write(done/'teacher_answer.json',answer(ids[0],20))
    torch.save({'original_feature':True},done/'paired_features.pt')
    write(done/'complete.json',{'task_id':ids[0],'split':'training','identity_sha256':digest(original),
        'files':{name:sha(done/name) for name in ['source_history.json','teacher_answer.json']},
        'feature_files':{'paired_features.pt':sha(done/'paired_features.pt')}})
    partial=parent/'tasks/p__partial'
    write(partial/'source_history.json',history(ids[1],103))
    write(partial/'inflight_source_reasoning.json',{'task_id':ids[1],'tokens':[99] if bad_prefix else [3], 'identity_sha256':digest(original)})
    write(partial/'inflight_teacher_answer.json',{'task_id':ids[1],'tokens':[8], 'identity_sha256':digest(original)})
    original_files={str(p.relative_to(tmp_path)):sha(p) for p in parent.rglob('*') if p.is_file()}
    declaration=tmp_path/'recovery.json';write(declaration,{'all_final_records_recovered':True,'new_random_draw_allowed':False,
        'workers':{'w1':{'task_ids':ids,'parent_root':'results/interrupted','files':original_files}}})
    source,receiver=TinyBackend(3),TinyBackend(4);calls=[]
    def reason(backend,prompt,tid,stream,cap,callback):
        calls.append(('reason',tid));h=history(tid,2)
        if bad_rng:h['rng_after_reasoning']=[99]
        callback([3,4])
        with torch.no_grad():cache=backend.prefill_chunked(h['prefix_ids']).past_key_values
        return h,cache
    def generate_answer(backend,h,cache,tid,stream,cap,callback):
        calls.append(('answer',tid));callback([8]);return answer(tid,1)
    monkeypatch.setattr(inference,'reason',reason);monkeypatch.setattr(inference,'answer',generate_answer)
    thread=threading.Thread(target=lambda:None);thread.start();published=[]
    context={'spec':{'stage':'histories','task_ids':ids,'worker_id':'w1','baseline_root':str(tmp_path/'baseline'),
        'history_recovery_manifest':'recovery.json','history_recovery_manifest_sha256':sha(declaration)},
        'root':root,'identity':current,'guard':lambda:None,'publish':lambda **kw:published.append(kw),'halt':threading.Event(),'thread':thread}
    return context,worker,(source,receiver),calls,original_files,published,parent,root


def test_complete_history_is_byte_preserved_and_partial_stream_is_exactly_replayed(tmp_path,monkeypatch):
    c,m,backends,calls,original,published,parent,root=fixture(tmp_path,monkeypatch)
    result=run_recovered(c,tmp_path,m,backends)
    assert result['task_count']==2 and set(result['tasks'])=={'p/done','p/partial'}
    assert calls==[('reason','p/partial'),('answer','p/partial')]
    assert all(sha(tmp_path/p)==expected for p,expected in original.items())
    for name in ['source_history.json','teacher_answer.json','paired_features.pt']:
        assert sha(root/'tasks/p__done'/name)==sha(parent/'tasks/p__done'/name)
    assert sha(root/'tasks/p__partial/source_history.json')==sha(parent/'tasks/p__partial/source_history.json')
    reused=json.loads((root/'tasks/p__done/complete.json').read_text())
    assert reused['identity_sha256']==digest(c['identity']) and reused['measurement_identity_sha256']==digest(json.loads((parent/'identity.json').read_text()))
    assert sha(root/'tasks/p__done/parent_complete.json')==sha(parent/'tasks/p__done/complete.json')
    features=json.loads((root/'feature_manifest.json').read_text())['files'];assert len(features)==2
    assert published[-1]['completed_tasks']==2 and published[-1]['total_tasks']==2
    assert m.write is write


@pytest.mark.parametrize('mutation',['bad_prefix','bad_rng'])
def test_saved_prefix_or_rng_divergence_stops_without_replacement_draw(tmp_path,monkeypatch,mutation):
    c,m,backends,calls,original,published,parent,root=fixture(tmp_path,monkeypatch,**{mutation:True})
    with pytest.raises(ValueError):run_recovered(c,tmp_path,m,backends)
    assert not (root/'complete.json').exists()
    assert all(sha(tmp_path/p)==expected for p,expected in original.items())
    assert calls==([] if mutation=='bad_prefix' else [('reason','p/partial')])


def test_recovery_rejects_artifact_membership_and_numerical_scope_changes(tmp_path,monkeypatch):
    c,m,backends,calls,original,published,parent,root=fixture(tmp_path,monkeypatch)
    c['spec']['task_ids']=list(reversed(c['spec']['task_ids']))
    with pytest.raises(ValueError,match='membership'):parent_for(tmp_path,c)
    c['spec']['task_ids'].reverse();c['identity']['config_sha256']='changed'
    with pytest.raises(ValueError,match='scope changed'):parent_for(tmp_path,c)
    c['identity']['config_sha256']='fixed';(parent/'tasks/p__done/source_history.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):parent_for(tmp_path,c)
    assert not calls
