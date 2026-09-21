"""Public synthetic receipts only; no private corpus/candidate execution."""
import hashlib
import json
from pathlib import Path
import pytest
from scripts import sparse_repair_score as s


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def make_public(tmp):
    tasks=[{'task_id':f'task/{i}','history_sha256':f'history{i}'} for i in range(12)]
    d={'screen_tasks':tasks,'seeds':{},'mapper':{'sha256':'mapper'}}
    c={'repo_root':tmp,'declaration':d,'declaration_path':'declaration.json','declaration_sha256':'frozen'}
    top=tmp/'out'
    for task in tasks:
        tid=task['task_id'];folder=top/'screen/tasks'/tid.replace('/','__')
        d['seeds'][tid]=[{'seed_index':i,'stream':f'draw{i}','answer_seed':i+4} for i in range(3)]
        pp=folder/'prompt_only_template.json'
        put(pp,{'task_id':tid,'declaration_sha256':'frozen','enable_thinking':False,
                'rendered_template':'<think>\n\n</think>','prompt_ids':[1,2],'prefix_ids':[1],'bridge_ids':[2]})
        files=[]
        for condition in s.CONDITIONS:
            for seed in d['seeds'][tid]:
                dest=folder/condition/f"seed_{seed['seed_index']}"
                identity={'experiment_id':'sparse_repair_01','cohort':'development_screen','task_id':tid,
                    'condition':condition,**seed,'declaration_sha256':'frozen',
                    'history_sha256':s.sha(pp) if condition=='P' else task['history_sha256'],
                    'original_source_history_sha256':task['history_sha256'],'mapper_sha256':'mapper',
                    'implementation_sha256':'same_generation_code','teacher_answer_prefix_supplied':False}
                answer={**identity,'answer_text':'```python\nprint(1)\n```','answer_ids':[10,11],'rng_final':[1]}
                put(dest/'answer.json',answer);put(dest/'sampler/answer_record.json',answer)
                resume={'state':'complete'};resume['resume_sha256']=s.digest(resume)
                put(dest/'sampler/resume.json',resume)
                put(dest/'working_set.json',{'semantic_trajectory_union_complete':True})
                for name in ['sampler/identity.json','sampler/complete.json','sampler/completion_timing.json']:
                    put(dest/name,{'synthetic':True})
                members={str(p.relative_to(dest)):s.sha(p) for p in dest.rglob('*.json')}
                put(dest/'complete.json',{'identity':identity,'files':members})
                files.append({'path':str((dest/'complete.json').relative_to(folder)),'sha256':s.sha(dest/'complete.json')})
        put(folder/'task_complete.json',{'task_id':tid,'declaration_sha256':'frozen',
            'implementation_sha256':'same_generation_code','prompt_only_template_sha256':s.sha(pp),
            'completed_records':42,'expected_records':42,'backing_cache_fingerprints_unchanged':True,'files':files})
    return c,top


def test_full_public_closure_and_changed_missing_extra_rejected(tmp_path):
    c,top=make_public(tmp_path)
    closure=s.generation_closure(c,'out')
    assert len(closure['answers'])==504 and closure['all_screen_generation_complete']
    assert closure['closure_sha256']==s.digest({k:v for k,v in closure.items() if k!='closure_sha256'})
    p=top/closure['answers'][0]['path'];old=p.read_text();p.write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):s.generation_closure(c,'out')
    p.write_text(old)
    missing=top/'screen/tasks/task__11/task_complete.json';old=missing.read_text();missing.unlink()
    with pytest.raises(FileNotFoundError):s.generation_closure(c,'out')
    missing.write_text(old)
    extra=top/'screen/tasks/task__0/UNDECLARED/seed_0/answer.json';put(extra,{})
    with pytest.raises(ValueError,match='extra draws'):s.generation_closure(c,'out')


def test_private_access_barrier_precedes_open(monkeypatch,tmp_path):
    def forbidden(*a,**kw):raise AssertionError('private byte access attempted')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    with pytest.raises(ValueError,match='sealed before private'):
        s.private_after_closure({'top':tmp_path},tmp_path/'never-read.json')


def test_private_exact_hash_and_full40_inventory_are_enforced(tmp_path,monkeypatch):
    # Small synthetic JSON, not the actual private corpus.
    tests={f't{i}':{'fn_name':None,'tests':[{'input':'1','output':'1'}]} for i in range(40)}
    path=tmp_path/'synthetic-tests.json';put(path,tests)
    c={'top':tmp_path/'public','closure':{'all_screen_generation_complete':True},
       'private_inventory':list(tests),'task_ids':['t0'],'policy':s.base.validate_policy(s.base.read(
           s.ROOT/'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/calibration/frozen_policy.json'))}
    with pytest.raises(ValueError,match='hash differs'):s.private_after_closure(c,path)
    monkeypatch.setattr(s,'PRIVATE_SHA',s.sha(path))
    assert set(s.private_after_closure(c,path))=={'t0'}
    c['private_inventory']=list(tests)[:-1]
    with pytest.raises(ValueError,match='inventory differs'):s.private_after_closure(c,path)


def score_context(tmp):
    raw={'answer_text':'```python\nprint(1)\n```','task_id':'x','condition':'R_10','seed_index':0,'answer_seed':7}
    put(tmp/'answer.json',raw)
    identity={'scorer_version':'unit-test-only'}
    spec={'fn_name':None,'tests':[{'input':'','output':'1'}]}
    contract={k:raw[k] for k in ('task_id','condition','seed_index','answer_seed')}
    r={**contract,'record_id':'record','contract':contract,'contract_sha256':s.digest(contract),
       'answer_path':'answer.json','answer_sha256':s.sha(tmp/'answer.json'),
       'code_sha256':s.base.code_hash(s.base.extract(raw['answer_text']))}
    c={'top':tmp,'plan_sha256':'p','plan_identity_sha256':'pidentity','scorer_identity':identity,'policy':{},
       'plan':{'declaration_sha256':'d','generation_closure_identity_sha256':'g',
               'generation_closure_file_sha256':'gf','scorer_identity_sha256':'s',
               'selected_private_spec_sha256':{'x':s.digest(spec)}}}
    return c,r,{'x':spec},identity


def test_atomic_score_idempotence_preserves_original_and_extraction(tmp_path,monkeypatch):
    c,r,tests,identity=score_context(tmp_path);calls=[]
    def mock_score(code,spec,**kwargs):
        calls.append(code);return {'passed':False,'missing':False,'category':'assertion_failure','scorer_identity':identity}
    monkeypatch.setattr(s.base,'score',mock_score)
    first=s.score_record(c,r,tests,1,'attempt',lambda:None)
    second=s.score_record(c,r,tests,1,'attempt2',lambda:None)
    assert first==second and len(calls)==1
    assert first['original_answer_text']=='```python\nprint(1)\n```'
    assert first['code']=='print(1)' and first['score_v2']['category']=='assertion_failure'


def test_started_uncommitted_score_is_missing_not_candidate_rerun(tmp_path,monkeypatch):
    c,r,tests,identity=score_context(tmp_path)
    folder=tmp_path/s.SCORING/'scores'/r['record_id']
    put(folder/'started.json',{'binding':s.score_binding(c,r)})
    monkeypatch.setattr(s.base,'score',lambda *a,**kw:pytest.fail('candidate must not rerun'))
    result=s.score_record(c,r,tests,1,'recovery',lambda:None)
    assert result['score_v2']['passed'] is None and result['score_v2']['missing'] is True
    assert result['score_v2']['reason']=='worker_interrupted_before_score_commit'


def test_actual_public_frozen_scorer_identity_without_private_reads():
    p='configs/coding_pilot_v1/sparse_repair_01/declaration.json'
    c=s.declaration_context(s.ROOT,p,s.sha(s.ROOT/p))
    assert c['policy']['cpu_seconds']==12 and c['policy']['wall_seconds']==46
    assert c['scorer_identity']['scorer_version']=='gearshift_scorer_v2_20260919_01'
    assert len(c['task_ids'])==12 and len(c['private_inventory'])==40


def test_eight_core_and_path_guards():
    with pytest.raises(ValueError,match='Exactly eight'):s.check_cpus([1,2],{})
    with pytest.raises(ValueError):s.path_inside('/tmp','../escape')
