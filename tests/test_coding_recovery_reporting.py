"""Incomplete infrastructure coverage must never become forty quality failures."""
import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import coding_recovery_report as report
from gearshift.coding_control import write,digest,sha


def fixture(root):
    write(root/'data/coding_pilot_v1/visible/development.json',[{'task_id':str(i)} for i in range(40)])
    write(root/'evidence/coding_pilot_v1/control/watchdog_status.json',{'upper_usd':260.,'gpu_hours':46.,'active_gpu_count':0,'active_resources':[]})
    worker=root/'results/coding_pilot_v1/dev/w';ident={'id':'test'};write(worker/'identity.json',ident);folder=worker/'tasks/0'
    names=['A_original','B_original','D_original','D_matched','C','C_initial']
    row={'answer_seconds':2.,'bridge_seconds':.1,'answer_ids':[1,2,3],'source_reasoning_seconds':8.,'answer_ended_eos':True,'answer_capped':False,'score':{'passed':True,'category':'pass'}}
    for name in names:write(folder/(name+'.json'),row)
    receipt={'task_id':'0','identity_sha256':digest(ident),'files':{name+'.json':sha(folder/(name+'.json')) for name in names},
             'pass':{a:True for a in ['A','B','C','C_initial','D_matched','D_original']},'D_tokens_match_original':True}
    write(folder/'complete.json',receipt);return worker,folder


def test_missing_coverage_is_not_wrong_output(tmp_path,monkeypatch):
    fixture(tmp_path);monkeypatch.setattr(report,'ROOT',tmp_path)
    r=report.report('dev','training_not_yet_imported')
    assert r['coverage']['committed_complete']==1 and len(r['coverage']['missing_task_ids'])==39
    assert r['arms']['C']['n']==1 and r['arms']['C']['passes']==1 and r['arms']['C']['pass_rate']==1
    assert r['paired'] is None


def test_changed_committed_score_rejected(tmp_path,monkeypatch):
    worker,folder=fixture(tmp_path);monkeypatch.setattr(report,'ROOT',tmp_path)
    write(folder/'C.json',{'score':{'passed':False}})
    with pytest.raises(ValueError,match='changed'):report.report('dev','missing')


def test_repetition_metric_counts_excess_occurrences():
    assert report.repeated_fourgrams([1,2,3])==0
    assert report.repeated_fourgrams([1]*7)==.75


def test_disjoint_attempt_merge_and_duplicate_rejection(tmp_path,monkeypatch):
    import shutil
    worker,folder=fixture(tmp_path);monkeypatch.setattr(report,'ROOT',tmp_path)
    selection={'checkpoint':{'sha256':'trained'},'initialization_sha256':'initial','step':96,'validation_kl':.29}
    write(tmp_path/'train/selection.json',selection)
    for name,h in [('C','trained'),('C_initial','initial')]:
        p=folder/(name+'.json');r=json.loads(p.read_text());r['checkpoint_sha256']=h;write(p,r)
    receipt=json.loads((folder/'complete.json').read_text())
    receipt['files']={name:sha(folder/name) for name in receipt['files']};write(folder/'complete.json',receipt)
    dest=tmp_path/'results/coding_pilot_v1/retry/w';shutil.copytree(worker,dest)
    with pytest.raises(ValueError,match='Duplicate'):report.report(['dev','retry'],'train')
    (dest/'tasks/0').rename(dest/'tasks/1');p=dest/'tasks/1/complete.json';r=json.loads(p.read_text());r['task_id']='1';write(p,r)
    merged=report.report(['dev','retry'],'train')
    assert merged['coverage']['committed_complete']==2 and merged['paired'] is None
    assert merged['arms']['C']['n']==2
    p=dest/'tasks/1/C.json';r=json.loads(p.read_text());r['checkpoint_sha256']='different';write(p,r)
    q=dest/'tasks/1/complete.json';r=json.loads(q.read_text());r['files']['C.json']=sha(p);write(q,r)
    with pytest.raises(ValueError,match='checkpoint bytes'):report.report(['dev','retry'],'train')


def test_independent_small_reasoning_is_included_in_total(tmp_path,monkeypatch):
    worker,folder=fixture(tmp_path);monkeypatch.setattr(report,'ROOT',tmp_path)
    p=folder/'B_original.json';r=json.loads(p.read_text());r['source_reasoning_seconds']=0;r['own_reasoning_seconds']=123.;write(p,r)
    q=folder/'complete.json';r=json.loads(q.read_text());r['files'][p.name]=sha(p);write(q,r)
    d=report.report('dev','missing')['arms']['B']['means']
    assert d['own_reasoning_seconds']==123.
    assert d['total_model_execution_estimate_seconds']==125.
    assert d['source_inclusive_estimate_seconds']==125.
