import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from collections import deque
import pytest
from gearshift.coding_control import sha,write,digest
from gearshift.coding_sandbox_v2 import DEFAULT_POLICY
from scripts import coding_scorer_repair_calibrate as calibration
from scripts import coding_scorer_repair_rescore as rescoring


def trusted_run(key,inp):
    return subprocess.check_output([sys.executable,'-I','-c',calibration.REFERENCES[key]],input=inp,text=True)


def test_public_stress_constructions_have_correct_small_analogues():
    for name,key,inp,expected in calibration.cases(scale=.0001):
        actual=trusted_run(key,inp)
        assert actual.split()==expected.split(),name


def test_linked_list_reference_matches_independent_list_oracle():
    rng=random.Random(39);current=list(range(1,20));initial=current[:];queries=[];nextvalue=100
    for _ in range(100):
        if len(current)>1 and rng.random()<.4:
            x=rng.choice(current);current.remove(x);queries.append(f'2 {x}')
        else:
            x=rng.choice(current);nextvalue+=1;current.insert(current.index(x)+1,nextvalue);queries.append(f'1 {x} {nextvalue}')
    inp=f'{len(initial)}\n'+ ' '.join(map(str,initial))+f'\n{len(queries)}\n'+'\n'.join(queries)+'\n'
    assert list(map(int,trusted_run('abc344_e',inp).split()))==current


def test_position_history_reference_matches_full_body_oracle():
    rng=random.Random(18);body=[(i,0) for i in range(1,15)];queries=[];expected=[];moves={'R':(1,0),'L':(-1,0),'U':(0,1),'D':(0,-1)}
    for _ in range(100):
        if rng.random()<.65:
            d=rng.choice(list(moves));dx,dy=moves[d];x,y=body[0];body=[(x+dx,y+dy),*body[:-1]];queries.append('1 '+d)
        else:
            p=rng.randrange(1,len(body)+1);expected.append(body[p-1]);queries.append(f'2 {p}')
    inp=f'{len(body)} {len(queries)}\n'+'\n'.join(queries)+'\n'
    actual=[tuple(map(int,line.split())) for line in trusted_run('abc335_c',inp).splitlines()]
    assert actual==expected


def bfs_oracle(grid):
    n=len(grid);players=tuple((i,j) for i in range(n) for j in range(n) if grid[i][j]=='P');q=deque([(players,0)]);seen={players}
    while q:
        state,d=q.popleft()
        if state[0]==state[1]:return d
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            target=tuple((i+di,j+dj) if 0<=i+di<n and 0<=j+dj<n and grid[i+di][j+dj]!='#' else (i,j) for i,j in state)
            if target not in seen:seen.add(target);q.append((target,d+1))
    return -1


def test_dense_bfs_reference_matches_independent_tuple_bfs():
    rng=random.Random(339)
    for _ in range(15):
        n=5;g=[['#' if rng.random()<.25 else '.' for _ in range(n)] for _ in range(n)]
        for p in rng.sample(range(n*n),2):g[p//n][p%n]='P'
        inp=f'{n}\n'+'\n'.join(map(''.join,g))+'\n'
        assert int(trusted_run('abc339_d',inp))==bfs_oracle(g)


def fixture_plan(tmp_path):
    root=tmp_path/'old';root.mkdir();raw={'answer_text':'```python\nprint(7)\n```','task_id':'task','condition':'D','form':'D','step':0,'seed_index':0,'answer_seed':17}
    write(root/'answer.json',raw);old={'code':'print(7)','answer_sha256':sha(root/'answer.json'),'score':{'passed':True,'category':'pass'}};write(root/'score.json',old)
    write(root/'scored_answer_manifest.json',{'hidden_tests_loaded_after_all_generation':True,'generation_closure_sha256':'g','files':[{'path':'score.json','sha256':sha(root/'score.json'),'answer_path':'answer.json','answer_sha256':sha(root/'answer.json')}]})
    policy=tmp_path/'policy.json';write(policy,{**DEFAULT_POLICY,'policy_status':'frozen','physical_cpu_ids':[0]});private=tmp_path/'private.json';write(private,{'task':{'fn_name':None,'tests':[{'input':'','output':'7'}]}});out=tmp_path/'new'
    plan=rescoring.freeze_plan(root,private,policy,out,expected=1)
    return root,out,plan,json.loads(private.read_text())


def test_immutable_rescore_resume_skips_exact_transaction_and_rejects_drift(tmp_path,monkeypatch):
    root,out,plan,private=fixture_plan(tmp_path);calls=[]
    monkeypatch.setattr(rescoring,'score',lambda *a,**k:calls.append(1) or {'passed':False,'category':'test_assertion','missing':False})
    record=plan['records'][0];first=rescoring.score_record(record,plan,root,out,private,0);second=rescoring.score_record(record,plan,root,out,private,0)
    assert first==second and len(calls)==1 and json.loads((root/'score.json').read_text())['score']['passed']
    assert rescoring.finalize(out)['committed']==1
    (root/'answer.json').write_text('{}')
    with pytest.raises(ValueError,match='drift'):rescoring.score_record(record,plan,root,out,private,0)


def test_interrupted_transaction_is_missing_without_candidate_reroll(tmp_path,monkeypatch):
    root,out,plan,private=fixture_plan(tmp_path);record=plan['records'][0]
    def crash(*a,**kw):raise RuntimeError('worker crashed')
    monkeypatch.setattr(rescoring,'score',crash)
    with pytest.raises(RuntimeError):rescoring.score_record(record,plan,root,out,private,0)
    result=rescoring.score_record(record,plan,root,out,private,0)
    assert result['score_v2']['passed'] is None and result['score_v2']['missing']
    assert rescoring.finalize(out)['missing']==1


def test_paired_report_does_not_turn_missing_into_failures():
    from scripts.coding_scorer_repair_report import statistics
    rows=[]
    for task in ('a','b'):
        for condition in ('FIXED_1024_M','ROTATING_1024_M'):
            for seed in range(3):
                passed=condition.startswith('ROTATING')
                rows.append({'task_id':task,'condition':condition,'seed_index':seed,'old_passed':passed,'new_passed':None if task=='a' and seed==0 else passed,'old_category':'pass' if passed else 'test_assertion','new_category':'pass' if passed else 'test_assertion'})
    s,t,i=statistics(rows,resamples=100)
    assert s['variants']['old']['contrasts']['ROTATING_1024_M-FIXED_1024_M']['difference']==1
    fixed=s['variants']['new']['conditions']['FIXED_1024_M'];assert fixed['pass_rate'] is None and fixed['missing_draws']==1
    delta=s['variants']['new']['contrasts']['ROTATING_1024_M-FIXED_1024_M'];assert delta['difference'] is None
    assert delta['possible_mean_bounds'][0] < delta['possible_mean_bounds'][1]


def test_report_regenerates_without_private_tests_or_candidate_execution(tmp_path,monkeypatch):
    from scripts.coding_scorer_repair_report import report
    root,out,plan,private=fixture_plan(tmp_path)
    # Expand a tiny complete3-draw control-only plan; sufficient to check archive-only regeneration.
    original=plan['records'][0];plan['records']=[]
    for seed in range(3):
        plan['records'].append({**original,'record_id':f'record{seed}','seed_index':seed})
    plan['expected_answers']=3;write(out/'rescore_plan.json',plan)
    monkeypatch.setattr(rescoring,'score',lambda *a,**k:{'passed':True,'category':'pass','missing':False})
    for record in plan['records']:rescoring.score_record(record,plan,root,out,private,0)
    rescoring.finalize(out);(tmp_path/'private.json').unlink()
    monkeypatch.setattr(rescoring,'score',lambda *a,**k:pytest.fail('No candidate execution in reports'))
    result=report(out)
    assert result['answers']==3 and (out/'report/SCORER_REPAIR_RESULTS.md').exists()


def test_rescore_refuses_private_test_drift_from_original_receipts(tmp_path):
    root,out,plan,private=fixture_plan(tmp_path)
    scorefile=root/'score.json';old=json.loads(scorefile.read_text())
    old['score']['receipts']=[{'test_index':0,'input_sha256':hashlib.sha256(b'').hexdigest(),'expected_sha256':hashlib.sha256(b'7').hexdigest()}]
    write(scorefile,old);manifest=json.loads((root/'scored_answer_manifest.json').read_text());manifest['files'][0]['sha256']=sha(scorefile);write(root/'scored_answer_manifest.json',manifest)
    private['task']['tests'][0]['output']='8';write(tmp_path/'private.json',private)
    with pytest.raises(ValueError,match='Original scored test bytes differ'):
        rescoring.freeze_plan(root,tmp_path/'private.json',tmp_path/'policy.json',tmp_path/'new_bad',expected=1)
