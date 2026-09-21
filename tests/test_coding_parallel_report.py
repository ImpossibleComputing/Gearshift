import importlib.util
import json
from pathlib import Path
import zipfile
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/coding_parallel_report.py'
    spec=importlib.util.spec_from_file_location('coding_report_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def rows():
    return [{'task_id':f't{i}','pass':{'A':i<140,'B':i<100,'C':i<130,'D':i<145}} for i in range(200)]


def test_paired_bootstrap_preserves_task_pairing_and_declared_draw_count():
    m=module();data=rows();r=m.paired_statistics(data)
    assert r['bootstrap']['resamples']==10000 and r['task_count']==200
    assert r['passes']=={'A':140,'B':100,'C':130,'D':145}
    assert r['contrasts']['C-B']['difference']==pytest.approx(.15)
    assert r['contrasts']['C-B']['rescues']==30 and r['contrasts']['C-B']['regressions']==0
    assert r['contrasts']['C-B']['ci95'][0]>0
    assert r['gap_closed']['reported'] and r['gap_closed']['value']==pytest.approx(.75)
    assert m.paired_statistics(data)==r
    with pytest.raises(ValueError,match='Duplicate'):m.paired_statistics(data+[data[0]])


def test_uncertain_gap_suppressed_and_degenerate_interval_labeled():
    m=module();data=[{'task_id':f't{i}','pass':dict.fromkeys('ABCD',i<10)} for i in range(20)]
    r=m.paired_statistics(data,draws=1000)
    assert r['contrasts']['C-B']['ci95']==[0,0] and r['contrasts']['C-B']['degenerate_interval']
    assert not r['gap_closed']['reported'] and 'value' not in r['gap_closed']


def test_useful_latency_excludes_fast_failed_capped_and_incomplete_answers():
    m=module()
    def output(seconds,passed=True,capped=False,ended=True):return {'source_inclusive_seconds':seconds,
        'score':{'passed':passed},'answer_capped':capped,'answer_ended_eos':ended}
    data=[{'outputs':{a:output(10 if a=='C' else 20) for a in 'ABCD'}},
          {'outputs':{a:output(.01,passed=a!='C') for a in 'ABCD'}},
          {'outputs':{a:output(.01,capped=a=='C') for a in 'ABCD'}},
          {'outputs':{a:output(.01,ended=a!='C') for a in 'ABCD'}}]
    r=m.latency_summary(data)
    assert r['paired']['C-D']['task_count']==1
    assert r['paired']['C-D']['ratio_of_mean_source_inclusive_seconds']==.5
    assert r['arms']['C']['successful_complete_outputs']==1


def fixture_repo(tmp_path,count=200,complete=True):
    m=module();repo=tmp_path
    m.write(repo/'configs/coding_pilot_v1/pilot.json',{'mapper':{'second_seed_confirmation_ids':[
        {'platform':'p','question_id':str(i)} for i in range(40)]}})
    m.write(repo/'data/coding_pilot_v1/visible/confirmation.json',[{'task_id':f'p/{i}'} for i in range(200)])
    m.write(repo/'selection.json',{'confirmation_used_for_selection':False,'seed':20260915,'checkpoint_sha256':'checkpoint','objective':'natural_handoff_boundary'})
    for name in m.ORIGINAL_TRAJECTORIES:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{"original_trajectory":true}\n')
    (repo/'README.md').write_text('Original README\n');(repo/'RESULTS.md').write_text('Original scores\n');(repo/'requirements.lock.txt').write_text('frozen dependency\n')
    root=repo/'results/coding_pilot_v1/headline/shard';identity={'experiment':'fixed'};m.write(root/'identity.json',identity)
    m.write(root/'native_gate.json',{'passed':True,'identity_sha256':m.digest(identity)})
    tasks={}
    for i in range(count):
        tid=f'p/{i}';folder=root/'tasks'/tid.replace('/','__')
        history={'task_id':tid,'reasoning_capped':False,'prompt_ids':[1],'reasoning_ids':[2,3],
            'prefix_ids':[1,2],'bridge_ids':[3]}
        m.write(folder/'source_history.json',history);m.write(folder/'small_history.json',history)
        passes={}
        for a,limit in [('A',140),('B',100),('C',130),('D',145)]:
            passed=i<limit;passes[a]=passed
            m.write(folder/(a+'.json'),{'task_id':tid,'condition':m.CONDITIONS[a],
                'score':{'passed':passed,'category':'pass' if passed else 'test_assertion','trusted_checker_complete':True,'executed_tests':1,'total_tests':1},
                'answer_ended_eos':True,'answer_capped':False,'source_inclusive_seconds':10 if a=='C' else 20,
                'historical_receiver_prefill_tokens':0,'checkpoint_sha256':'checkpoint','answer_text':'saved output'})
        receipt={'task_id':tid,'identity_sha256':m.digest(identity),'source_capped':False,'small_capped':False,
            'pass':passes,'files':{p.name:m.sha(p) for p in folder.glob('*.json')}}
        m.write(folder/'complete.json',receipt);tasks[tid]=m.sha(folder/'complete.json')
    if complete:m.write(root/'complete.json',{'stage':'confirmation','identity_sha256':m.digest(identity),
        'selection_sha256':m.sha(repo/'selection.json'),'task_count':count,'tasks':tasks})
    else:m.write(root/'candidate_inputs.json',{'selection_sha256':m.sha(repo/'selection.json')})
    request={'pipeline_id':'test','confirmation_roots':[str(root.relative_to(repo))],'second_seed_roots':[],
        'training_roots':[],'selection_path':'selection.json','measurements_complete':False}
    return m,repo,root,request


def test_incomplete_cohort_reports_counts_without_positive_inference(tmp_path):
    m,repo,root,request=fixture_repo(tmp_path,count=3)
    summary,actual,_=m.analyze(repo,request)
    assert summary['coverage']['observed']==3 and summary['primary_inference'] is None
    assert not summary['study_complete'] and summary['publication_recommendation']=='publish as partial/negative result'
    assert len(actual)==3 and 'No confidence interval' in m.markdown(summary)


def test_incomplete_worker_retains_verified_task_observations_without_completion_claim(tmp_path):
    m,repo,root,request=fixture_repo(tmp_path,count=2,complete=False)
    summary,_,_=m.analyze(repo,request)
    assert summary['coverage']['observed']==2
    assert summary['coverage']['incomplete_worker_roots'] and not summary['headline_confirmation_complete']


def test_tampered_output_cannot_enter_statistical_analysis(tmp_path):
    m,repo,root,request=fixture_repo(tmp_path,count=2)
    (root/'tasks/p__0/C.json').write_text('{"score":{"passed":true}}')
    summary,_,_=m.analyze(repo,request)
    assert summary['coverage']['observed']==1 and summary['integrity_issues']
    assert summary['publication_recommendation']=='fix a blocking correctness issue before publication'


def test_passing_score_without_trusted_test_completion_is_rejected(tmp_path):
    m,repo,root,request=fixture_repo(tmp_path,count=1)
    path=root/'tasks/p__0/C.json';obj=m.read(path);obj['score']['trusted_checker_complete']=False;path.write_text(json.dumps(obj))
    receipt_path=path.parent/'complete.json';receipt=m.read(receipt_path);receipt['files']['C.json']=m.sha(path);receipt_path.write_text(json.dumps(receipt))
    done=m.read(root/'complete.json');done['tasks']['p/0']=m.sha(receipt_path);(root/'complete.json').write_text(json.dumps(done))
    summary,_,_=m.analyze(repo,request)
    assert summary['coverage']['observed']==0 and 'trusted-test' in summary['integrity_issues'][0]['error']


def test_bundle_contains_original_and_new_trajectories_and_excludes_heavy_private_files(tmp_path):
    m,repo,root,request=fixture_repo(tmp_path,count=1)
    for name in ['.git/config','results/coding_pilot_v1/model.safetensors','results/coding_pilot_v1/mapper.pt',
                 'data/coding_pilot_v1/private/confirmation.json','results/coding_pilot_v1/cache_lru/cache.pt',
                 'results/coding_pilot_v1/.env','evidence/credentials.json','evidence/pod.json','evidence/connection.json','results/coding_pilot_v1/pod_backup/stale.json']:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('must not export')
    outside=tmp_path.parent/'outside-private.txt';outside.write_text('not allowed')
    (repo/'results/link.txt').symlink_to(outside)
    m.write(repo/'evidence/innocent_name.json',{'resource':{'env':{'PUBLIC_KEY':'sensitive'}}})
    output=repo/'results/coding_pilot_v1/review';output.mkdir();(output/'CODING_RESULTS.md').write_text('partial')
    result=m.build_bundle(repo,repo/'gearshift.zip',output)
    assert result['source_trajectories_included']
    with zipfile.ZipFile(result['path']) as archive:
        names=archive.namelist()
        for original in m.ORIGINAL_TRAJECTORIES:assert 'gearshift/'+original in names
        assert 'gearshift/results/coding_pilot_v1/headline/shard/tasks/p__0/source_history.json' in names
        assert 'gearshift/requirements.lock.txt' in names
        assert not any('/private/' in n or '/.git/' in n or n.endswith(('.pt','.safetensors','.env')) or any(x in n for x in ['credentials.json','pod.json','connection.json','innocent_name.json']) or '/pod_backup/' in n or n.endswith('/link.txt') for n in names)
        manifest=json.loads(archive.read('gearshift/REVIEW_BUNDLE_MANIFEST.json'))
        for name,value in manifest['files'].items():
            import hashlib
            assert hashlib.sha256(archive.read('gearshift/'+name)).hexdigest()==value['sha256']


def test_partial_export_is_immutable_repeatable_and_detects_later_tampering(tmp_path):
    m,repo,root,request=fixture_repo(tmp_path,count=1)
    before=(repo/'RESULTS.md').read_bytes();output=repo/'results/coding_pilot_v1/review'
    receipt=m.export(repo,request,output,repo/'gearshift.zip')
    assert receipt['export_complete'] and not receipt['study_complete']
    assert m.export(repo,request,output,repo/'gearshift.zip')==receipt
    assert (repo/'RESULTS.md').read_bytes()==before
    (output/'CODING_RESULTS.md').write_text('tampered')
    with pytest.raises(ValueError,match='report changed'):m.verify_export(output/'EXPORT.json')


def test_full_confirmation_and_reserved_second_seed_keep_task_denominators(tmp_path):
    m,repo,root,request=fixture_repo(tmp_path,count=200)
    second=repo/'results/coding_pilot_v1/seed2/shard';identity={'experiment':'second'}
    m.write(second/'identity.json',identity);m.write(second/'native_gate.json',{'passed':True,'identity_sha256':m.digest(identity)})
    tasks={}
    for i in range(40):
        tid=f'p/{i}';folder=second/'tasks'/tid.replace('/','__');original=root/'tasks'/tid.replace('/','__')
        m.write(folder/'source_history.json',m.read(original/'source_history.json'))
        output=m.read(original/'C.json');output['score']['passed']=i<20;output['score']['category']='pass' if i<20 else 'test_assertion'
        output['checkpoint_sha256']='second-checkpoint';m.write(folder/'C.json',output)
        receipt={'task_id':tid,'identity_sha256':m.digest(identity),'source_capped':False,'pass':{},
            'candidates':{'second':{'passed':i<20}},'files':{p.name:m.sha(p) for p in folder.glob('*.json')}}
        m.write(folder/'complete.json',receipt);tasks[tid]=m.sha(folder/'complete.json')
    m.write(second/'complete.json',{'stage':'second_seed','identity_sha256':m.digest(identity),
        'selection_sha256':m.sha(repo/'selection.json'),'task_count':40,'tasks':tasks})
    request['second_seed_roots']=[str(second.relative_to(repo))];request['measurements_complete']=True
    for objective,seed in [('ordinary_continuation',20260915),('natural_handoff_boundary',20260915),('natural_handoff_boundary',20260916)]:
        tr=repo/f'results/coding_pilot_v1/train/{objective}-{seed}';ident={'objective':objective,'seed':seed}
        m.write(tr/'identity.json',ident);m.write(tr/'complete.json',{'identity_sha256':m.digest(ident),'objective':objective,'seed':seed,
            'steps':256,'stop_reason':'time_cap','converged':False,'capped_unconverged':True,'wall_seconds':10800,
            'training_predictions':8192,'nominees':[]})
        request['training_roots'].append(str(tr.relative_to(repo)))
    proof=repo/'final_proof.json';record={'passed':True,'gate':'completed_bounded_execution','headline_task_count':200,'second_seed_task_count':40,
        'inputs':{str((second/'complete.json').relative_to(repo)):m.sha(second/'complete.json')}}
    m.write(proof,record);request['completion_proof']={'path':str(proof.relative_to(repo)),'sha256':m.sha(proof),'record':record}
    summary,actual,_=m.analyze(repo,request)
    assert summary['study_complete'] and summary['coverage']['observed']==200
    assert summary['primary_inference']['task_count']==200 and len(actual)==200
    seed=summary['second_seed']['inference'];assert seed['task_count']==40
    assert seed['headline_C_passes']==40 and seed['second_seed_C_passes']==20
    assert seed['second_minus_headline']['difference']==-.5
    assert '512-token chunks' in summary['training']['caveat']
    assert all(r['capped_unconverged'] for r in summary['training']['runs'])


def test_state_discovery_uses_completed_mixed_run_even_when_artifact_map_is_stale(tmp_path):
    m=module();repo=tmp_path;state=repo/'state.json'
    m.write(state,{'pipeline_id':'pipeline1','completed_runs':{'confirmation_with_second_seed':'mixed'},'artifact_roots':{},
        'scientific_stop':{'reason':'Second seed has no eligible checkpoint'}})
    m.write(repo/'evidence/coding_pilot_v1/control/parallel/mixed/plan.json',{'workers':[
        {'worker_id':'headline01','role':'confirmation'},{'worker_id':'replicate','role':'second_seed_training'}]})
    request=m.request_from_state(repo,state)
    assert request['confirmation_roots']==['results/coding_pilot_v1/mixed/headline01']
    assert request['training_roots']==['results/coding_pilot_v1/mixed/replicate']
    assert request['scientific_stop']['reason'].startswith('Second seed')
