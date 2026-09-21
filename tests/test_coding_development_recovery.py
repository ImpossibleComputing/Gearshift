import copy,json
import pytest
from gearshift.coding_control import write,sha,digest,PREFIX
from gearshift.coding_development_recovery import verify_coverage

def fixture(root):
    tasks=[str(i) for i in range(40)];write(root/'dataset.json',[{'task_id':t} for t in tasks]);write(root/'selection.json',{'checkpoint':'fixed'})
    write(root/'science.py','frozen')
    workers=[{'worker_id':f'w{i}','task_ids':tasks[i::4],'command':['frozen.py'],'worker_fields':{'selection':'selection.json'}} for i in range(4)]
    parent={'run_id':'parent','stage':'exploratory_development','task_ids':tasks,'workers':workers,'files':{'science.py':sha(root/'science.py')},'image_digest':'fixed'}
    write(root/'parent.json',parent)
    error='Provider pod create failed: \n'+json.dumps({'error':'failed to create pod: There are no longer any instances available.'})
    write(root/'failure.json',{'state':'failed','error':error})
    write(root/'done.json',{'passed':False,'stage_identity':digest(parent),'workers':[{'worker_id':w['worker_id'],'passed':True} for w in workers[:3]],'errors':[error]})
    write(root/'ledger.json',{'resources':[]});roots=[];preserved=[]
    for w in workers[:3]:
        name=w['worker_id'];roots.append(name);identity={'worker_id':name,'stage_identity':digest(parent)};write(root/name/'identity.json',identity)
        write(root/name/'native_gate.json',{'passed':True,'identity_sha256':digest(identity)})
        committed={}
        for tid in w['task_ids']:
            path=root/name/'tasks'/tid;write(path/'C.json',{'passed':False})
            write(path/'complete.json',{'task_id':tid,'identity_sha256':digest(identity),'files':{'C.json':sha(path/'C.json')}})
            committed[tid]=sha(path/'complete.json');preserved.append(tid)
        write(root/name/'complete.json',{'identity_sha256':digest(identity),'selection_sha256':sha(root/'selection.json'),'tasks':committed})
    proof={'dataset_path':'dataset.json','parent_plan':'parent.json','parent_completion':'done.json','failure':'failure.json','failed_worker_id':'w3','ledger_snapshot':'ledger.json','selection':'selection.json','preserved_worker_roots':roots,'preserved_task_ids':sorted(preserved)}
    plan={'workers':[copy.deepcopy(workers[3])],'image_digest':'fixed'}
    return proof,tasks[3::4],plan

def test_exact_complement_preserves_all_completed_draws(tmp_path):
    proof,remaining,plan=fixture(tmp_path)
    assert len(verify_coverage(tmp_path,proof,remaining,plan))==30

@pytest.mark.parametrize('mutation',['redraw','allocated','changed_code','changed_raw','changed_selection','native_failed','other_failure','changed_fields'])
def test_unsafe_capacity_completion_rejected(tmp_path,mutation):
    proof,remaining,plan=fixture(tmp_path)
    if mutation=='redraw':remaining[0]='0'
    elif mutation=='allocated':write(tmp_path/'ledger.json',{'resources':[{'kind':'pod','name':PREFIX+'parent-w3-capacity01','absent_epoch':123}]})
    elif mutation=='changed_code':write(tmp_path/'science.py','changed')
    elif mutation=='changed_raw':write(tmp_path/'w0/tasks/0/C.json',{'passed':True})
    elif mutation=='changed_selection':write(tmp_path/'selection.json',{'checkpoint':'other'})
    elif mutation=='native_failed':
        p=tmp_path/'w0/native_gate.json';r=json.loads(p.read_text());r['passed']=False;write(p,r)
    elif mutation=='other_failure':
        p=tmp_path/'done.json';r=json.loads(p.read_text());r['errors'].append('out of memory');write(p,r)
    elif mutation=='changed_fields':plan['workers'][0]['worker_fields']['seed']=999
    with pytest.raises(ValueError):verify_coverage(tmp_path,proof,remaining,plan)
