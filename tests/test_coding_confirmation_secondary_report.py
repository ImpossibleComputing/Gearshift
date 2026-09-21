"""Second-seed statistics and compact export; every outcome is synthetic."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gearshift.coding_control import digest,sha,write
from scripts import coding_confirmation_secondary_report as r
from scripts import coding_confirmation_secondary_score as s

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('secondary_scoring_fixture',ROOT/'tests/test_coding_confirmation_secondary_score.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


def analysis(n):
    value=s.read(ROOT/'configs/coding_pilot_v1/confirmation_01/analysis_plan.json');value['task_count']=n;return value


def rows(tasks):
    return [{'task_id':t,'condition':c,'seed_index':seed,'passed':False,'missing':False,'category':'test_assertion',
        'cohort':'secondary','training_seed':20260919} for t in tasks for c in r.CONDITIONS for seed in range(3)]


def references(tasks):
    return [{'task_id':t,'condition':c,'seed_index':seed,'passed':True,'missing':False,'category':'pass','cohort':'primary'}
        for t in tasks for c in r.REFERENCE_CONDITIONS for seed in range(3)]


def test_secondary_mean_and_joint_indices_exactly_match_primary_recipe():
    tasks=['z','a','q'];values=rows(tasks)
    for v in values:v['passed']=v['condition']=='ROTATING_M' and v['seed_index']<=tasks.index(v['task_id'])
    summary,_,_,indices=r.statistics(values,tasks,analysis(3))
    primary_rows=[]
    for t in tasks:
        for condition in r.primary.CONDITIONS:
            for seed in range(3):
                match=next((v for v in values if (v['task_id'],v['condition'],v['seed_index'])==(t,condition,seed)),None)
                primary_rows.append(match or {'task_id':t,'condition':condition,'seed_index':seed,'passed':False,'missing':False})
    original,_,_,primary_indices=r.primary.cluster_statistics(primary_rows,tasks,analysis(3))
    assert np.array_equal(indices,primary_indices)
    assert summary['bootstrap_indices_sha256']==original['bootstrap_indices_sha256']==digest(indices.tolist())
    delta=summary['contrasts']['ROTATING_M-FIXED_M'];assert delta['difference']==pytest.approx(2/3)
    assert delta['ci95']==original['contrasts']['ROTATING_M-FIXED_M']['ci95']
    assert summary['training_seed']==20260919 and summary['primary_mapper_scores_pooled_or_selected'] is False
    assert len(summary['conditions'])==4 and len(summary['contrasts'])==3 and summary['native_reference_draws_reused']==0


def test_missing_secondary_outcomes_keep_full_denominators_and_pair_extremes():
    values=rows(['a','b'])
    for v in values:
        if v['task_id']=='a' and v['condition'] in ('ROTATING_M','FIXED_M') and v['seed_index']==0:v.update(passed=None,missing=True,category='infrastructure_failure')
    summary,_,_,_=r.statistics(values,['a','b'],analysis(2));delta=summary['contrasts']['ROTATING_M-FIXED_M']
    assert delta['difference'] is None and delta['ci95'] is None and delta['draws_per_condition']==6
    assert delta['possible_mean_bounds']==pytest.approx([-1/6,1/6]) and delta['unresolved_tasks']==1
    assert summary['secondary_answers']==24 and summary['secondary_missing_answers']==2


def test_native_reference_draws_are_separate_and_never_recounted_as_replication():
    values=rows(['a','b']);refs=references(['a','b'])
    summary,_,_,_=r.statistics(values,['a','b'],analysis(2),refs)
    assert summary['secondary_answers']==24 and summary['native_reference_draws_reused']==24
    assert set(summary['conditions'])==set(r.CONDITIONS) and set(summary['primary_native_references'])==set(r.REFERENCE_CONDITIONS)
    assert len(summary['contrasts'])==7 and summary['contrasts']['ROTATING_H-D']['difference']==-1
    assert summary['contrasts']['ROTATING_H-D']['reference_reused_from_primary'] is True
    assert all(x['source']=='reused_primary_native_reference' for x in summary['primary_native_references'].values())


@pytest.mark.parametrize('change',['duplicate','missing','wrong_seed','wrong_training_seed','primary_mapped','secondary_reference','partial_references','numeric_bool','unmarked_null'])
def test_invalid_seed_pooling_selection_or_coverage_is_rejected(change):
    values=rows(['a']);refs=references(['a'])
    if change=='duplicate':values.append(dict(values[0]))
    elif change=='missing':values.pop()
    elif change=='wrong_seed':values[0]['seed_index']=4
    elif change=='wrong_training_seed':values[0]['training_seed']=20260918
    elif change=='primary_mapped':values[0]['cohort']='primary'
    elif change=='secondary_reference':refs[0]['condition']='ROTATING_M'
    elif change=='partial_references':refs.pop()
    elif change=='numeric_bool':values[0]['passed']=1
    else:values[0]['passed']=None
    with pytest.raises(ValueError):r.statistics(values,['a'],analysis(1),refs)


def test_zero_interval_endpoint_stays_exact_zero_not_positive_significance():
    tasks=['a','b'];values=rows(tasks)
    for v in values:v['passed']=v['task_id']=='a' and v['condition']=='ROTATING_H'
    summary,_,_,_=r.statistics(values,tasks,analysis(2))
    assert summary['contrasts']['ROTATING_H-FIXED_H']['ci95'][0]==0.


def test_compact_secondary_report_needs_no_private_tests_or_primary_scores(tmp_path,monkeypatch):
    c,private=fixtures.prepared(tmp_path,monkeypatch);fixtures.complete(c,private,monkeypatch)
    relative=str(c['plan_path'].relative_to(tmp_path));s.finalize(tmp_path,relative);private.unlink()
    before={str(p):p.read_bytes() for p in (c['top']/'primary').rglob('*') if p.is_file()}
    summary=r.report(relative,repo=tmp_path)
    assert summary['secondary_answers']==24 and summary['secondary_missing_answers']==0
    assert summary['reference_status'].startswith('Not supplied')
    assert before=={str(p):p.read_bytes() for p in (c['top']/'primary').rglob('*') if p.is_file()}
    text=(c['top']/'secondary/report/CONFIRMATION_SECONDARY_RESULTS.md').read_text()
    assert 'does not pool' in text and 'unadjusted 95%' in text and 'not exhaustive' in text
    inventory=s.read(c['top']/'secondary/report/FILE_MANIFEST.json')
    assert all(sha(c['top']/'secondary/report'/f['path'])==f['sha256'] for f in inventory['files'])


def test_secondary_report_refuses_primary_output_directory(tmp_path,monkeypatch):
    c,private=fixtures.prepared(tmp_path,monkeypatch);fixtures.complete(c,private,monkeypatch)
    relative=str(c['plan_path'].relative_to(tmp_path));s.finalize(tmp_path,relative)
    with pytest.raises(ValueError,match='overwrite primary'):r.report(relative,repo=tmp_path,output=c['top']/'primary/report')


@pytest.mark.parametrize('change',['task_order','closure','scorer','declaration'])
def test_native_reference_identity_mismatch_is_rejected(tmp_path,monkeypatch,change):
    c,_=fixtures.prepared(tmp_path,monkeypatch);q=c['plan'];p={'declaration_sha256':q['declaration_sha256'],'task_ids':q['task_ids'],
        'generation_closure_identity_sha256':q['primary_generation_closure_identity_sha256'],
        'generation_closure_file_sha256':q['primary_generation_closure_file_sha256'],'scorer_identity_sha256':q['scorer_identity_sha256']}
    key={'task_order':'task_ids','closure':'generation_closure_file_sha256','scorer':'scorer_identity_sha256','declaration':'declaration_sha256'}[change]
    p[key]=list(reversed(p[key])) if change=='task_order' else '0'*64
    monkeypatch.setattr(r.primary,'load_records',lambda *a:{'plan':p,'rows':[],'bindings':{}})
    with pytest.raises(ValueError):r.primary_references(c,'primary_plan.json',tmp_path)
