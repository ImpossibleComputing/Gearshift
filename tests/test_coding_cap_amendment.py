import copy,json
from pathlib import Path
import pytest
from gearshift.coding_control import write,sha,digest
from gearshift.coding_reuse import NUMERICAL_FILES
from gearshift.coding_cap_amendment import validate_parent,replay_constraints,OLD_CAP
from gearshift.coding_resume import verify_segment


@pytest.fixture
def parent(tmp_path):
    root=tmp_path/'repo';base=root/'results/parent'
    cfg=root/'configs/coding_pilot_v1/pilot.json';write(cfg,{'models':{'source':{'id':'source'},'receiver':{'id':'receiver'}},'protocol':{'development_cap_gate':{'one_allowed_reasoning_cap_increase':24576}}})
    protocol=root/'configs/coding_pilot_v1/control_protocol_v2.json';write(protocol,{'v':2})
    write(root/'data/coding_pilot_v1/identity.json',{'fixed':True})
    write(root/'data/coding_pilot_v1/visible/development.json',[{'task_id':f't{i}'} for i in range(40)])
    for name in NUMERICAL_FILES:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('unchanged numerical implementation')
    identity={'config_sha256':sha(cfg),'data_identity':{'fixed':True},'control_protocol_sha256':sha(protocol),
        'implementation':{n:sha(root/n) for n in NUMERICAL_FILES},'reasoning_cap':OLD_CAP}
    write(base/'identity.json',identity);write(base/'native_gate.json',{'passed':True,'identity_sha256':digest(identity)})
    for i in range(40):
        folder=base/'tasks'/f't{i}'
        for kind,capped in [('source',i<5),('small',i==5)]:
            write(folder/(kind+'_history.json'),{'task_id':f't{i}','reasoning_ids':[7]*OLD_CAP if capped else [1,2,151668],
                'reasoning_capped':capped,'natural_boundary':not capped,'early_eos':False})
        for arm in ['A','B','D']:write(folder/(arm+'.json'),{'task_id':f't{i}','answer_ids':[8,9],'score':{'passed':True}})
        write(folder/'complete.json',{'task_id':f't{i}','identity_sha256':digest(identity),'pass':{'A':True,'B':True,'D':True},
            'source_capped':i<5,'small_capped':i==5,'files':{p.name:sha(p) for p in folder.glob('*.json')}})
    write(base/'baseline_gate.json',{'passed':False,'cap_gate':False,'protocol_cap_amendment_needed':True,'counts':{'A':40,'B':40,'D':40}})
    write(base/'progress.json',{'observed_baseline_seconds_including_parent':1200})
    return root,base,identity


def test_amendment_uses_all_capped_cases_in_fixed_order_regardless_of_success(parent):
    root,base,_=parent;c=validate_parent(base,root)
    assert c['repeat_task_ids']==[f't{i}' for i in range(6)] and len(c['reuse_task_ids'])==34
    assert c['cap_counts']=={'source':5,'small':1}


def test_amendment_cannot_start_from_a_partial_cohort(parent):
    root,base,_=parent;(base/'tasks/t39/complete.json').unlink()
    with pytest.raises(ValueError,match='every original'):validate_parent(base,root)


def test_only_capped_reasoning_and_dependent_answers_may_change(parent):
    _,base,identity=parent
    prefixes,complete,receipt=replay_constraints(base,identity,'t0')
    assert receipt['extended_reasoning_segments']==['source_reasoning']
    assert set(receipt['answers_with_changed_conditioning'])=={'answer_A','answer_D'}
    assert 'answer_B' in complete and 'small_reasoning' in complete
    verify_segment([7]*OLD_CAP+[4,151668],'source_reasoning',prefixes,complete)
    with pytest.raises(ValueError,match='diverged'):verify_segment([6]+[7]*(OLD_CAP-1),'source_reasoning',prefixes,complete)
    with pytest.raises(ValueError):verify_segment([7]*(OLD_CAP-1),'source_reasoning',prefixes,complete)
    with pytest.raises(ValueError):verify_segment([8,9,10],'answer_B',prefixes,complete)


def test_small_cap_does_not_allow_source_answers_to_change(parent):
    _,base,identity=parent;p,c,r=replay_constraints(base,identity,'t5')
    assert r['answers_with_changed_conditioning']==['answer_B']
    assert 'answer_A' in c and 'answer_D' in c and 'source_reasoning' in c


def test_a_second_cap_amendment_is_forbidden(parent):
    root,base,identity=parent;identity['reasoning_cap']=24576
    write(base/'identity.json',identity);write(base/'native_gate.json',{'passed':True,'identity_sha256':digest(identity)})
    with pytest.raises(ValueError,match='Only one'):validate_parent(base,root)


def test_amendment_worker_reuses_unaffected_tasks_and_replays_only_affected(parent,monkeypatch):
    import importlib.util,time,types,torch
    import gearshift.coding_inference as inference
    import gearshift.coding_control_calibration as calibration
    import gearshift.coding_sandbox as sandbox
    root,base,identity=parent
    target=root/'results/coding_pilot_v1/preflight_v5';target.parent.mkdir(parents=True);base.rename(target);base=target
    script=Path(__file__).resolve().parents[1]/'scripts/coding_preflight_cap.py'
    spec=importlib.util.spec_from_file_location('cap_worker_test',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    e=root/'evidence/coding_pilot_v1';out=root/'results/coding_pilot_v1/preflight_cap_v1'
    for name,value in [('ROOT',root),('E',e),('R',out)]:monkeypatch.setattr(m,name,value)
    monkeypatch.chdir(root)
    declaration=root/'configs/coding_pilot_v1/cap_amendment_v1.json';write(declaration,{'test':True})
    approval=e/'experiment_100h_approved.json';write(approval,{'test':True})
    checkpoint=validate_parent(base,root)
    write(e/'allocation_cap_v1.json',{'preflight_deadline_epoch':time.time()+3600,'image_digest':'test-image',
        'approval_sha256':sha(approval),'cap_amendment_sha256':sha(declaration),
        'parent_checkpoint':{k:v for k,v in checkpoint.items() if k!='parent_identity'}})
    write(e/'sandbox_gate.json',{'passed':True,'canonical_development':{'a':True,'b':True}})
    visible=root/'data/coding_pilot_v1/visible/development.json'
    write(visible,[{'task_id':f't{i}','prompt_ids':[3]} for i in range(40)])
    write(root/'data/coding_pilot_v1/private/development.json',{f't{i}':{} for i in range(40)})
    calls=[];controls=[]
    class FakeBackend:
        def __init__(self,spec):
            self.role=spec['id'];self.tokenizer=types.SimpleNamespace(backend_tokenizer=types.SimpleNamespace(to_str=lambda:'same'),encode=lambda _: [3])
        def prefill_chunked(self,ids):return types.SimpleNamespace(past_key_values=None)
    def reason(backend,prompt,tid,stream,cap,callback):
        assert cap==24576;calls.append((tid,stream))
        kind='source' if stream=='source_reasoning' else 'small'
        old=json.loads((base/'tasks'/tid/(kind+'_history.json')).read_text())
        tokens=old['reasoning_ids']+([99,151668] if old['reasoning_capped'] else [])
        callback(tokens)
        return {**old,'reasoning_ids':tokens,'prompt_ids':prompt,'prefix_ids':prompt+tokens[:-1],
            'reasoning_capped':False,'natural_boundary':True,'early_eos':False,'reasoning_seconds':1},None
    def answer(backend,history,cache,tid,stream,cap,callback):
        callback([8,9]);return {'task_id':tid,'answer_ids':[8,9],'answer_text':'print(1)'}
    monkeypatch.setattr(inference,'Backend',FakeBackend);monkeypatch.setattr(inference,'reason',reason);monkeypatch.setattr(inference,'answer',answer)
    monkeypatch.setattr(inference,'memory_record',lambda:{'passed':True});monkeypatch.setattr(inference,'sync',lambda:None)
    monkeypatch.setattr(calibration,'matched_native_controls',lambda backend,path,role,max_prefix:controls.append((role,max_prefix)))
    monkeypatch.setattr(sandbox,'score',lambda code,tests,guard:{'passed':True})
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'test-device')
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(m.platform,'platform',lambda:'test-platform')
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'test')
    before={str(p.relative_to(base)):p.read_bytes() for p in base.rglob('*') if p.is_file()}
    m.main()
    assert controls==[('source',32768),('receiver',32768)]
    assert {tid for tid,_ in calls}=={f't{i}' for i in range(6)} and len(calls)==12
    progress=json.loads((out/'progress.json').read_text())
    assert (progress['completed_tasks'],progress['reused_parent_tasks'],progress['regenerated_tasks'])==(40,34,6)
    assert progress['source_cap_count']==progress['small_cap_count']==0
    assert json.loads((out/'baseline_gate.json').read_text())['protocol_cap_amendment_needed'] is False
    for name,data in before.items():assert (base/name).read_bytes()==data
    for i in range(6,40):
        row=json.loads((out/f'tasks/t{i}/complete.json').read_text())
        assert row['reused_from']['no_new_generation_or_scoring'] and row['measurement_identity_sha256']==digest(identity)
