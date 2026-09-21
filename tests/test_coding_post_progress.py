import json
import pytest
import torch
from gearshift.coding_post_progress import splice,validate_diagnostic_plan,DECLARATION,OWNER
from gearshift.coding_control import sha

def test_hybrid_boundary_keeps_exact_suffix_and_no_alias():
    native=((torch.arange(6.).reshape(1,1,3,2),torch.ones(1,1,3,2)),)
    mapped=((torch.arange(14.).reshape(1,1,7,2)+100,torch.zeros(1,1,7,2)),)
    result=splice(native,mapped,3)
    assert torch.equal(result[0][0][...,:3,:],native[0][0])
    assert torch.equal(result[0][0][...,3:,:],mapped[0][0][...,3:,:])
    result[0][0].zero_()
    assert native[0][0].sum()>0 and mapped[0][0].sum()>0
    with pytest.raises(ValueError):splice(native,mapped,4)

def test_diagnostic_scope_rejects_private_numerical_and_unselected_cases(tmp_path,monkeypatch):
    from gearshift import coding_parallel
    for path in (DECLARATION,OWNER):(tmp_path/path).parent.mkdir(parents=True,exist_ok=True)
    d={'four_training_histories':[{'task_id':str(i)} for i in range(4)],'diagnostic_max_additional_gpu_hours':20}
    (tmp_path/DECLARATION).write_text(json.dumps(d));(tmp_path/OWNER).write_text('owner')
    proof={'inputs':{p:sha(tmp_path/p) for p in (DECLARATION,OWNER)}}
    monkeypatch.setattr(coding_parallel,'verify_gate',lambda *a:proof)
    plan={'stage':'post_progress_numerical','scope_amendment':'post_progress01','scientific_scope_unchanged':False,'gates':{'post_progress_declaration':{}},'files':{},'task_ids':['0','1','2','3'],'workers':[{'maximum_seconds':3600,'worker_fields':{'declaration_sha256':sha(tmp_path/DECLARATION)}}]}
    validate_diagnostic_plan(plan,tmp_path)
    plan['files']['data/coding_pilot_v1/private/training.json']='x'
    with pytest.raises(ValueError,match='no scorer'):validate_diagnostic_plan(plan,tmp_path)
    plan['files']={};plan['task_ids']=['0','1','2','favorable-dev']
    with pytest.raises(ValueError,match='cohort'):validate_diagnostic_plan(plan,tmp_path)

def test_seen_schedule_covers_tail_eos_and_only_four_training_cases():
    from scripts.coding_post_memorization import coverage_schedule
    histories=[{'task_id':str(i),'teacher_answer':{'answer_ids':list(range(n))}} for i,n in enumerate([33,513,640,755])]
    a=coverage_schedule(histories,20260917,100,64)
    assert a==coverage_schedule(histories,20260917,100,64)
    seen={o['task_id']:set() for o in histories}
    for item in a:
        s=item['segments'][0];n=len(histories[int(s['task_id'])]['teacher_answer']['answer_ids']);positions=s['positions']
        assert len(positions)==len(set(positions))==min(64,n)
        assert positions[0]==0 and positions[-1]==n-1
        seen[s['task_id']].update(positions)
    assert [len(seen[o['task_id']]) for o in histories]==[33,513,640,755]
