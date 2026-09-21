"""Synthetic secondary adapter checks; no model, private benchmark or execution."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from gearshift.coding_control import digest,sha,write
from scripts import coding_confirmation_secondary_score as s

ROOT=Path(__file__).resolve().parents[1]


def fixture(tmp_path,monkeypatch,count=2):
    top=tmp_path/'results';top.mkdir();tids=['atcoder/task_'+str(i) for i in range(count)]
    policy=s.read(ROOT/'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/calibration/frozen_policy.json')
    identity=s.base.scorer_identity(policy)
    private=tmp_path/'protected/tests.json';specs={t:{'fn_name':None,'tests':[{'input':'','output':'1'}]} for t in tids};write(private,specs)
    private_identity={'task_ids':tids,'private_file_task_ids':tids,'private_tests_sha256':sha(private)}
    analysis=s.read(ROOT/'configs/coding_pilot_v1/confirmation_01/analysis_plan.json');analysis['task_count']=count;write(tmp_path/'analysis.json',analysis)
    seeds={'tasks':{t:{'answers':[{'seed_index':i,'answer_seed':100+i,'stream':'answer_'+str(i)} for i in range(3)]} for t in tids}}
    d={'experiment_id':'synthetic_secondary','task_ids':tids,'task_count':count,'primary_training_seed':20260918,
        'analysis_path':'analysis.json','inputs':{'analysis.json':sha(tmp_path/'analysis.json')}}
    cp={arm:{'step':1024,'mapper_sha256':letter*64} for arm,letter in [('FIXED','b'),('ROTATING','c')]}
    implementation={}
    for name in s.IMPLEMENTATION:
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);implementation[name]=sha(dest)
    sd={'training_seed':20260919,'conditions':s.CONDITIONS,'task_ids':tids,'answer_count':12*count,'implementation':implementation,'replication_analysis':s.generation.analysis_binding(d)}
    write(tmp_path/'declaration.json',d);write(tmp_path/'secondary_declaration.json',sd)
    ds=sha(tmp_path/'declaration.json');ss=sha(tmp_path/'secondary_declaration.json')
    gp={'experiment_id':d['experiment_id'],'result_root':'results','declaration_path':'declaration.json','declaration_sha256':ds,
        'secondary_declaration_path':'secondary_declaration.json','secondary_declaration_sha256':ss}
    write(tmp_path/'dispatch.json',gp)
    primary={'closed':True};primary['closure_sha256']=digest(primary);write(top/'primary/generation_closure.json',primary)
    files=[];answers=[];histories=[]
    def include(p):files.append({'path':str(p.relative_to(top)),'sha256':sha(p),'bytes':p.stat().st_size})
    include(top/'primary/generation_closure.json')
    for tid in tids:
        hp=top/'primary/tasks'/tid.replace('/','__')/'large_history/source_history.json'
        h={'task_id':tid,'reasoning_ids':[1,2],'reasoning_seconds':2.,'natural_boundary':True,'reasoning_capped':False,'early_eos':False,
            'sampler_timing':{'complete':True,'initial_prefill_seconds':.1,'reconstruction_seconds':0.}}
        write(hp,h);include(hp);histories.append({'task_id':tid,'sha256':sha(hp),'reused_from_primary':True})
        for cond in s.CONDITIONS:
            for seed in seeds['tasks'][tid]['answers']:
                contract={'cohort':'secondary','declaration_sha256':ds,'task_id':tid,'condition':cond,**seed,
                    'checkpoint_sha256':cp[cond.split('_')[0]]['mapper_sha256'],'history_sha256':sha(hp)}
                raw={**contract,'answer_text':'print(1)','answer_ids':[1,151645],'answer_ended_eos':True,'answer_capped':False,
                    'reasoning_model':'source','reasoning_seconds':2.,'source_cost_divisor_for_single_output':1,'answer_seconds':1.,
                    'native_prefill_seconds':.1,'mapping_seconds':.1,'splice_seconds':.1,'cache_clone_seconds':.1,
                    'bridge_seconds':.1,'first_answer_token_seconds':.2,'source_cache_reconstruction_seconds':0.,
                    'single_output_inference_seconds':3.4,'inference_timing_complete':True,
                    'sampler_timing':{'complete':True,'reconstruction_seconds':0.}}
                path=top/'secondary/tasks'/tid.replace('/','__')/cond/('seed_'+str(seed['seed_index']))/'answer.json';write(path,raw);include(path)
                answers.append({'contract':contract,'contract_sha256':digest(contract),'path':str(path.relative_to(top)),'answer_sha256':sha(path)})
    closure={'cohort':'secondary','training_seed':20260919,'task_ids':tids,'conditions':s.CONDITIONS,'answers':answers,'answer_count':12*count,
        'files':files,'histories':histories,'checkpoints':cp,'all_secondary_generation_complete':True,
        'primary_generation_closure_sha256':primary['closure_sha256'],'primary_generation_closure_file_sha256':sha(top/'primary/generation_closure.json')}
    closure['closure_sha256']=digest(closure);write(top/'secondary/generation_closure.json',closure)
    c={'repo_root':tmp_path,'top':top,'plan':gp,'declaration':d,'declaration_sha256':ds,'secondary_declaration':sd,
        'secondary_declaration_sha256':ss,'secondary_checkpoints':cp,'seeds':seeds,'visible':{t:{'prompt':'public task'} for t in tids}}
    def public(repo,path,want,verify_weights=False):
        assert not verify_weights and path=='dispatch.json' and sha(tmp_path/path)==want
        return copy.deepcopy(c)
    def verify(context,seal=False):
        assert seal is False
        for f in closure['files']:
            if sha(top/f['path'])!=f['sha256']:raise ValueError('Synthetic public generation file changed')
        return copy.deepcopy(closure)
    monkeypatch.setattr(s.generation,'public_context',public);monkeypatch.setattr(s.generation,'generation_closure',verify)
    monkeypatch.setattr(s.base,'declared_scorer',lambda *a:(policy,identity,private_identity))
    return c,private,closure


def prepared(tmp_path,monkeypatch):
    _,private,_=fixture(tmp_path,monkeypatch)
    _,p=s.prepare(tmp_path,'dispatch.json',sha(tmp_path/'dispatch.json'),private)
    return s.validate_plan(tmp_path,str(p.relative_to(tmp_path))),private


def result(c,passed=False):
    return {'passed':passed,'missing':passed is None,'category':'infrastructure_failure' if passed is None else 'pass' if passed else 'cpu_timeout',
        'scorer_identity':c['scorer_identity'],'receipts':[]}


def complete(c,private,monkeypatch):
    calls=[]
    def fake(*a,**kw):calls.append(a);return result(c)
    monkeypatch.setattr(s.base,'score',fake)
    tests=s.private_after_closure(c,private)
    for r in c['plan']['records']:s.score_record(c,r,tests,3,'synthetic',lambda:None)
    return calls


@pytest.mark.parametrize('change',['incomplete','stale_seal','changed_raw','missing_implementation','wrong_order','wrong_seed'])
def test_entire_secondary_seal_and_identity_precede_private_access(tmp_path,monkeypatch,change):
    c,private,closure=fixture(tmp_path,monkeypatch)
    if change=='incomplete':monkeypatch.setattr(s.generation,'generation_closure',lambda *a,**kw:None)
    elif change=='stale_seal':write(c['top']/'secondary/generation_closure.json',{'wrong':True})
    elif change=='changed_raw':(c['top']/closure['answers'][0]['path']).write_text('{}')
    elif change=='missing_implementation':c['secondary_declaration']['implementation'].pop(next(iter(s.IMPLEMENTATION)))
    else:
        if change=='wrong_order':closure['answers']=list(reversed(closure['answers']))
        else:closure['answers'][0]['contract']['answer_seed']+=1;closure['answers'][0]['contract_sha256']=digest(closure['answers'][0]['contract'])
        closure['closure_sha256']=digest({k:v for k,v in closure.items() if k!='closure_sha256'});write(c['top']/'secondary/generation_closure.json',closure)
    read_bytes=Path.read_bytes
    def watched(p):
        if p==private:pytest.fail('Private tests read before complete secondary generation verification')
        return read_bytes(p)
    monkeypatch.setattr(Path,'read_bytes',watched)
    with pytest.raises((ValueError,KeyError)):s.prepare(tmp_path,'dispatch.json',sha(tmp_path/'dispatch.json'),private)


def test_secondary_transactions_never_change_primary_or_reroll_committed_failure(tmp_path,monkeypatch):
    c,private=prepared(tmp_path,monkeypatch);before={str(p):p.read_bytes() for p in (c['top']/'primary').rglob('*') if p.is_file()}
    calls=complete(c,private,monkeypatch);r=c['plan']['records'][0]
    saved=s.score_record(c,r,{},4,'new_attempt',lambda:None)
    assert saved['score_v2']['passed'] is False and len(calls)==24
    assert before=={str(p):p.read_bytes() for p in (c['top']/'primary').rglob('*') if p.is_file()}
    assert not (c['top']/'primary/scoring').exists()


def test_interrupted_secondary_score_is_missing_without_candidate_retry(tmp_path,monkeypatch):
    c,private=prepared(tmp_path,monkeypatch);r=c['plan']['records'][0]
    def interrupt(*a,**kw):raise s.base.ScoringInterrupted()
    monkeypatch.setattr(s.base,'score',interrupt)
    with pytest.raises(s.base.ScoringInterrupted):s.score_record(c,r,s.private_after_closure(c,private),3,'first',lambda:None)
    monkeypatch.setattr(s.base,'score',lambda *a,**kw:pytest.fail('Interrupted candidate rerun'))
    assert s.score_record(c,r,{},4,'recover',lambda:None)['score_v2']['passed'] is None


@pytest.mark.parametrize('field,value',[('training_seed',20260918),('cohort','primary'),('secondary_declaration_sha256','0'*64),('primary_generation_closure_file_sha256','0'*64),('policy',{}),('conditions',['A','B'])])
def test_plan_tampering_rejected_before_private_access(tmp_path,monkeypatch,field,value):
    c,_=prepared(tmp_path,monkeypatch);p=copy.deepcopy(c['plan']);p[field]=value;write(c['plan_path'],p)
    with pytest.raises(ValueError):s.validate_plan(tmp_path,str(c['plan_path'].relative_to(tmp_path)))


def test_final_manifest_keeps_all_draws_and_explicit_missing(tmp_path,monkeypatch):
    c,private=prepared(tmp_path,monkeypatch);complete(c,private,monkeypatch)
    r=c['plan']['records'][0];p=c['top']/s.SCORING/'scores'/r['record_id']/'score.json';receipt=s.read(p);receipt['score_v2']=result(c,None);write(p,receipt)
    manifest=s.finalize(tmp_path,str(c['plan_path'].relative_to(tmp_path)))
    assert manifest['committed_answers']==24 and manifest['missing_answers']==1 and manifest['training_seed']==20260919
    assert manifest['files'][0]['passed'] is None
    p.unlink()
    with pytest.raises(ValueError):s.finalize(tmp_path,str(c['plan_path'].relative_to(tmp_path)))


def test_private_tests_cannot_be_secondary_public_artifacts(tmp_path,monkeypatch):
    c,private=prepared(tmp_path,monkeypatch);p=c['top']/'secondary/private.json';shutil.copyfile(private,p)
    with pytest.raises(ValueError,match='public artifacts'):s.private_after_closure(c,p)


def actual_closure_fixture(tmp_path,monkeypatch):
    """Real primary+secondary sampler/closure validators, mocked candidate scorer."""
    import importlib.util
    spec=importlib.util.spec_from_file_location('actual_secondary_generation_fixture',ROOT/'tests/test_coding_confirmation_secondary_generate.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    c=helpers.secondary_fixture(tmp_path,count=1);d=c['declaration'];d['primary_training_seed']=20260918
    analysis=s.read(ROOT/'configs/coding_pilot_v1/confirmation_01/analysis_plan.json');analysis['task_count']=1;write(tmp_path/'analysis.json',analysis)
    d.update(analysis_path='analysis.json',inputs={'analysis.json':sha(tmp_path/'analysis.json')})
    implementation={}
    for name in s.IMPLEMENTATION:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p);implementation[name]=sha(p)
    sd={'training_seed':20260919,'conditions':s.CONDITIONS,'task_ids':d['task_ids'],'answer_count':12,
        'implementation':implementation,'replication_analysis':s.generation.analysis_binding(d)}
    write(tmp_path/'secondary.json',sd)
    # Existing fake generation transactions use their original frozen fixture SHA.
    c.update(secondary_declaration=sd)
    gp={'result_root':'results','declaration_path':'declaration.json','declaration_sha256':c['declaration_sha256'],
        'secondary_declaration_path':'secondary.json','secondary_declaration_sha256':c['secondary_declaration_sha256']}
    c['plan']=gp;write(tmp_path/'dispatch.json',gp)
    for cp in c['secondary_checkpoints'].values():cp['step']=1024
    for path in (c['top']/'secondary/tasks').glob('*/*/seed_*/answer.json'):
        raw=s.read(path);raw.update(reasoning_model='source',source_cost_divisor_for_single_output=1,reasoning_seconds=2.,answer_seconds=1.,
            native_prefill_seconds=.1,mapping_seconds=.1,splice_seconds=.1,cache_clone_seconds=.1,bridge_seconds=.1,first_answer_token_seconds=.2,
            source_cache_reconstruction_seconds=0.,single_output_inference_seconds=3.4,inference_timing_complete=True)
        write(path,raw);(path.parent/'draw_complete.json').unlink()
        s.generation.primary.verify_draw(path.parent,s.read(path.parent/'draw_identity.json'))
    s.generation.generation_closure(c)
    policy=s.read(ROOT/'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/calibration/frozen_policy.json')
    private=tmp_path/'protected/tests.json';write(private,{t:{'fn_name':None,'tests':[{'input':'','output':'1'}]} for t in d['task_ids']})
    pi={'task_ids':d['task_ids'],'private_file_task_ids':d['task_ids'],'private_tests_sha256':sha(private)}
    monkeypatch.setattr(s.generation,'public_context',lambda *a,**kw:dict(c))
    monkeypatch.setattr(s.base,'declared_scorer',lambda *a:(policy,s.base.scorer_identity(policy),pi))
    _,path=s.prepare(tmp_path,'dispatch.json',sha(tmp_path/'dispatch.json'),private)
    return s.validate_plan(tmp_path,str(path.relative_to(tmp_path))),private


def test_real_primary_secondary_generation_closures_reach_scoring_and_private_free_report(tmp_path,monkeypatch):
    from scripts import coding_confirmation_secondary_report as report
    c,private=actual_closure_fixture(tmp_path,monkeypatch);complete(c,private,monkeypatch)
    path=str(c['plan_path'].relative_to(tmp_path));manifest=s.finalize(tmp_path,path);private.unlink()
    assert manifest['committed_answers']==12
    result=report.report(path,repo=tmp_path)
    assert result['secondary_answers']==12 and result['secondary_missing_answers']==0
    assert len(result['history_diagnostics'])==1 and len(result['setup_measurements'])==1
    assert (c['top']/'secondary/report/setup_measurements.csv').is_file()
    # Fixture records predate overhead markers; their costs remain unmeasured.
    assert not (c['top']/'secondary/report/sampler_overheads.csv').exists()


def test_real_primary_control_mutation_blocks_secondary_before_private_read(tmp_path,monkeypatch):
    c,private=actual_closure_fixture(tmp_path,monkeypatch)
    control=next((c['top']/'primary/tasks').glob('*/controls/*.json'));value=s.read(control);value['whole_native_tensor_exact']=False;write(control,value)
    original=Path.read_bytes
    def watched(p):
        if p==private:pytest.fail('Read private bytes after primary-control mutation')
        return original(p)
    monkeypatch.setattr(Path,'read_bytes',watched)
    with pytest.raises(ValueError):s.prepare(tmp_path,'dispatch.json',sha(tmp_path/'dispatch.json'),private)
