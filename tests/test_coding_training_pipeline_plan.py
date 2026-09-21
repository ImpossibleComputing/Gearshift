import importlib.util,json,tarfile,types
from pathlib import Path
import pytest
from gearshift.coding_control import write,sha,digest
from gearshift.coding_snapshot import capture


def load():
    spec=importlib.util.spec_from_file_location('pipeline_plan',Path(__file__).parents[1]/'scripts/coding_pipeline_plan.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_copy_new_never_replaces_existing_evidence(tmp_path):
    m=load();a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(b'original')
    m.copy_new(a,b);m.copy_new(a,b);a.write_bytes(b'changed')
    with pytest.raises(ValueError,match='replace'):m.copy_new(a,b)
    assert b.read_bytes()==b'original'


def imported_fixture(tmp_path,monkeypatch):
    m=load();repo=tmp_path/'repository';repo.mkdir();monkeypatch.setattr(m,'ROOT',repo)
    remote=tmp_path/'remote';run='history-test';worker='w1';result=f'results/coding_pilot_v1/{run}/{worker}'
    evidence=f'evidence/coding_pilot_v1/parallel/{run}/{worker}'
    plan={'schema':1,'run_id':run,'stage':'histories','workers':[{'worker_id':worker}]};identity=digest(plan)
    write(remote/result/'identity.json',{'sample':'fixed'})
    feature=remote/result/'tasks/task/paired_features.pt';feature.parent.mkdir(parents=True);feature.write_bytes(b'paired-features')
    member=str(feature.relative_to(remote));write(remote/result/'feature_manifest.json',{'files':{member:sha(feature)}})
    write(remote/evidence/'worker_status.json',{'state':'complete','stage_identity':identity,'worker_id':worker})
    stage=tmp_path/'snapshot';stage.mkdir();capture(remote,stage)
    backup=repo/f'evidence/coding_pilot_v1/parallel_backups/{run}/{worker}';backup.mkdir(parents=True)
    archive=backup/'final.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(stage.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(stage))
    saved_feature=backup/'mapper_checkpoints'/member;saved_feature.parent.mkdir(parents=True);saved_feature.write_bytes(feature.read_bytes())
    control=repo/f'evidence/coding_pilot_v1/control/parallel/{run}'
    write(control/'plan.json',plan);write(control/'complete.json',{'passed':True,'stage_identity':identity,'workers':[{'worker_id':worker,'passed':True}]})
    write(control/worker/'backup_verified.json',{'verified':True,'worker_confirmed_absent':True,'stage_identity':identity,'sha256':sha(archive),
        'mapper_checkpoints':{member:{'sha256':sha(feature)}}})
    return m,repo,run,result,saved_feature,member


def test_import_checks_compact_and_external_hashes_preserves_paths(tmp_path,monkeypatch):
    m,repo,run,result,saved,member=imported_fixture(tmp_path,monkeypatch)
    plan,roots=m.import_run(run)
    assert roots==[result] and (repo/member).read_bytes()==b'paired-features'
    m.import_run(run)
    saved.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='backup differs'):m.import_run(run)


def test_import_process_complete_does_not_assert_scientific_pass(tmp_path,monkeypatch):
    m,repo,run,result,saved,member=imported_fixture(tmp_path,monkeypatch)
    # Import is transport validation only; no invented baseline or memory proof.
    m.import_run(run)
    assert not (repo/result/'baseline_gate.json').exists()
    assert not (repo/result/'memory_gate.json').exists()


def baseline_fixture(tmp_path,monkeypatch,passed=True):
    m=load();repo=tmp_path/'repository';repo.mkdir();monkeypatch.setattr(m,'ROOT',repo);monkeypatch.chdir(repo)
    baseline_rel='results/coding_pilot_v1/recovered/worker';baseline=repo/baseline_rel
    write(baseline/'baseline_gate.json',{'passed':passed,'cap_gate':passed})
    for index in range(40):
        folder=baseline/f'tasks/task{index}'
        write(folder/'complete.json',{'task_id':f'p/{index}'})
        write(folder/'source_history.json',{'reasoning_seconds':10})
        write(folder/'small_history.json',{'reasoning_seconds':5})
        for arm in ['A','B','D']:write(folder/(arm+'.json'),{'answer_seconds':2})
    approval=repo/'evidence/coding_pilot_v1/control/parallel_500h_approved.json';write(approval,{'approved':True})
    for split,count in [('training',128),('validation',32),('development',40)]:
        write(repo/f'data/coding_pilot_v1/visible/{split}.json',[{'task_id':split+str(i)} for i in range(count)])
    write(repo/'data/coding_pilot_v1/identity.json',{})
    script=repo/'scripts/coding_memory_worker.py';script.parent.mkdir();script.write_text('# worker fixture')
    image='runpod/pytorch@sha256:'+'1'*64
    monkeypatch.setattr(m,'import_run',lambda run:({'image_digest':image},[baseline_rel]))
    monkeypatch.setattr(m,'load_script',lambda name:types.SimpleNamespace(check_baselines=lambda path:{'baseline_identity_file_sha256':'verified'}))
    return m,repo,{'schema':1,'pipeline_id':'pilot','initial_recovery_run':'recovered','region_candidates':['CA-MTL-3','US-WA-1']}


def test_memory_plan_requires_actual_baseline_and_is_hash_bound(tmp_path,monkeypatch):
    m,repo,state=baseline_fixture(tmp_path,monkeypatch)
    plan=m.build('memory',state,repo/'memory-plan.json')
    assert plan['workers'][0]['command']==['scripts/coding_memory_worker.py']
    assert plan['workers'][0]['worker_fields']['baseline_root']=='results/coding_pilot_v1/recovered/worker'
    assert set(plan['gates'])=={'baseline'}
    assert not any('/private/' in p for p in plan['files'])
    assert plan['workers'][0]['region_candidates']==['CA-MTL-3','US-WA-1']
    from gearshift.coding_parallel import validate_plan
    proof=repo/plan['gates']['baseline']['path'];proof.write_text('{}')
    with pytest.raises(ValueError):validate_plan(plan,repo,plan['approval_sha256'])


def test_failed_baseline_is_scientific_stop_not_dispatch(tmp_path,monkeypatch):
    m,repo,state=baseline_fixture(tmp_path,monkeypatch,passed=False)
    with pytest.raises(m.ScientificStop,match='baseline/cap'):m.build('memory',state,repo/'must-not-exist.json')
    assert not (repo/'must-not-exist.json').exists()


def test_missing_memory_completion_never_builds_history_plan(tmp_path,monkeypatch):
    m,repo,state=baseline_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='memory stage'):m.build('histories',state,repo/'must-not-exist.json')
    assert not (repo/'must-not-exist.json').exists()


def test_evaluation_final_cohort_rejects_missing_or_duplicate_tasks(tmp_path,monkeypatch):
    m=load();monkeypatch.setattr(m,'ROOT',tmp_path);root=tmp_path/'results/test';identity={'mock':True}
    write(root/'identity.json',identity);write(root/'native_gate.json',{'passed':True,'identity_sha256':digest(identity)})
    folder=root/'tasks/p__t';write(folder/'C.json',{'score':{'passed':True}})
    write(folder/'complete.json',{'task_id':'p/t','identity_sha256':digest(identity),'pass':{'A':True,'B':False,'C':True,'D':True},'files':{'C.json':sha(folder/'C.json')}})
    write(root/'complete.json',{'identity_sha256':digest(identity),'stage':'confirmation','selection_sha256':'selection',
        'tasks':{'p/t':sha(folder/'complete.json')}})
    files,rows=m.validate_evaluation_roots(['results/test'],['p/t'],'confirmation','selection')
    assert len(rows)==1
    with pytest.raises(ValueError,match='incomplete'):m.validate_evaluation_roots(['results/test'],['p/t','p/other'],'confirmation','selection')
    with pytest.raises(ValueError,match='Duplicate'):m.validate_evaluation_roots(['results/test','results/test'],['p/t'],'confirmation','selection')


def test_mock_full_builder_training_development_and_mixed_isolation(tmp_path,monkeypatch):
    import torch
    from gearshift.coding_training import nominations
    m,repo,state=baseline_fixture(tmp_path,monkeypatch)
    # All local paths below are synthetic proof fixtures, never model runs.
    files=['gearshift/core.py','gearshift/coding_inference.py','gearshift/coding_gradients.py','gearshift/coding_training.py']
    for name in files:
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixed numerical fixture')
    for name in ['coding_history_worker.py','coding_training_worker.py','coding_evaluation_worker.py']:(repo/'scripts'/name).write_text('# fixture worker')
    protocol=repo/'configs/coding_pilot_v1/training_protocol_v1.json';write(protocol,{'fixed':True})
    write(repo/'configs/coding_pilot_v1/pilot.json',{'mapper':{'second_seed_confirmation_ids':[{'platform':'confirmation','question_id':str(i)} for i in range(40)]}})
    confirmation=[{'task_id':'confirmation/'+str(i)} for i in range(200)]
    write(repo/'data/coding_pilot_v1/visible/confirmation.json',confirmation)
    write(repo/'data/coding_pilot_v1/private/confirmation.json',{'hidden':True})
    write(repo/'data/coding_pilot_v1/private/development.json',{'hidden':True})
    roots={'cap_recovery':['results/coding_pilot_v1/recovered/worker'],'memory':['results/coding_pilot_v1/memory/worker'],
        'histories':['results/coding_pilot_v1/histories/worker'],'initialization':['results/coding_pilot_v1/init/worker'],
        'training':['results/coding_pilot_v1/train/ordinary','results/coding_pilot_v1/train/boundary'],
        'development':['results/coding_pilot_v1/development/worker']}
    raw='results/coding_pilot_v1/memory_v2_both_gpu';write(repo/raw/'memory_gate.json',{'passed':True,'training_ready':True,'actual_training_runtime':True,
        'objectives':{name:{'gradient_predictions':32} for name in ['ordinary_continuation','natural_handoff_boundary']}})
    write(repo/raw/'identity.json',{'baseline_identity_file_sha256':'verified','training_protocol_sha256':sha(protocol),
        'implementation':{name:sha(repo/name) for name in files}})
    write(repo/roots['memory'][0]/'memory_gate.json',{'passed':True,'training_ready':True,'selected_probe_root':raw,'probe_gate_sha256':sha(repo/raw/'memory_gate.json')})
    histories=[{'task_id':'training'+str(i),'split':'training'} for i in range(128)]+[{'task_id':'validation'+str(i),'split':'validation'} for i in range(32)]
    history_receipts=[{'task_id':o['task_id'],'source_root':roots['histories'][0]} for o in histories]
    monkeypatch.setattr(m,'verify_history_roots',lambda root,names:(histories,[],history_receipts))
    def completed(name,identity,**extra):
        root=repo/name;write(root/'identity.json',identity);write(root/'native_gate.json',{'passed':True,'identity_sha256':digest(identity)})
        write(root/'complete.json',{'identity_sha256':digest(identity),**extra});return root
    completed(roots['histories'][0],{'histories':True})
    initroot=repo/roots['initialization'][0];initroot.mkdir(parents=True)
    init=initroot/'mapper_initialization.pt';torch.save({'ridge':.01,'training_task_count':128,'fresh_source_specific':True,'history_receipts_sha256':digest(history_receipts)},init)
    completed(roots['initialization'][0],{'init':True},initialization_sha256=sha(init))
    descriptors=[]
    for name,objective in zip(roots['training'],['ordinary_continuation','natural_handoff_boundary']):
        root=repo/name;run=root/'run';identity={'seed':20260915,'objective':objective,'initialization_sha256':sha(init)}
        write(run/'identity.json',identity);curve=[]
        for i,step in enumerate([256,384]):
            checkpoint=run/f'mapper_step_{step:04d}.pt';checkpoint.write_bytes((objective+str(step)).encode())
            curve.append({'step':step,'validation_kl':.2-i*.1,'checkpoint':{'path':checkpoint.name,'sha256':sha(checkpoint)}})
        nominees=nominations(curve);write(run/'validation_curve.json',curve)
        write(run/'complete.json',{'identity_sha256':digest(identity),'objective':objective,'seed':20260915,'nominees':nominees})
        completed(name,{'outer':objective},run_complete_sha256=sha(run/'complete.json'))
        for i,n in enumerate(nominees):
            descriptors.append({'candidate_id':f'{objective}-s20260915-n{i+1}','training_root':str(run.relative_to(repo)),
                'training_complete_sha256':sha(run/'complete.json'),'checkpoint':str((run/n['checkpoint']['path']).relative_to(repo)),
                'checkpoint_sha256':n['checkpoint']['sha256'],'objective':objective,'seed':20260915,'step':n['step'],'initialization_sha256':sha(init)})
    devroot=repo/roots['development'][0];devid={'development':True};receipts={}
    for i in range(40):
        tid='development'+str(i);folder=devroot/'tasks'/tid;write(folder/'C.json',{'raw':'saved'})
        write(folder/'complete.json',{'task_id':tid,'identity_sha256':digest(devid),'files':{'C.json':sha(folder/'C.json')},
            'candidates':{c['candidate_id']:{'passed':i<20+j,'source_inclusive_seconds':10.,'checkpoint_sha256':c['checkpoint_sha256']} for j,c in enumerate(descriptors)}})
        receipts[tid]=sha(folder/'complete.json')
    completed(roots['development'][0],devid,stage='development_candidates',tasks=receipts,candidates=descriptors)
    image='runpod/pytorch@sha256:'+'1'*64
    monkeypatch.setattr(m,'import_run',lambda run:({'image_digest':image},roots[run]))
    real_eval=importlib.util.spec_from_file_location('fixture_eval',Path(__file__).parents[1]/'scripts/coding_evaluation_worker.py')
    ev=importlib.util.module_from_spec(real_eval);real_eval.loader.exec_module(ev)
    monkeypatch.setattr(m,'load_script',lambda name:ev if name=='coding_evaluation_worker.py' else types.SimpleNamespace(check_baselines=lambda path:{'baseline_identity_file_sha256':'verified'}))
    state['initial_recovery_run']='cap_recovery';state['completed_runs']={s:s for s in roots}
    retained_copies,retained_receipt=retained_fixture(m,repo,state)
    state['stage_limits_seconds']={'histories':{'estimated_seconds':1,'maximum_seconds':18000}}
    history_plan=m.build('histories',state,repo/'history-plan.json')
    assert history_plan['workers'][0]['estimated_seconds']==1440
    assert history_plan['forecast']['frozen_reference_estimate_seconds']==1
    assert history_plan['forecast']['completed_baseline_estimate_seconds']==1440
    assert state['stage_limits_seconds']['histories']['estimated_seconds']==1440
    assert state['frozen_stage_reference_estimates']['histories']==1
    state['stage_limits_seconds']['histories']['maximum_seconds']=1000
    with pytest.raises(m.ScientificStop,match='forecast'):m.build('histories',state,repo/'too-short.json')
    train=m.build('training',state,repo/'training-plan.json')
    assert len(train['workers'])==2 and all(w['maximum_seconds']==10800 for w in train['workers'])
    dev=m.build('development',state,repo/'development-plan.json')
    assert len(dev['workers'])==8 and all(len(w['task_ids'])==5 for w in dev['workers'])
    mixed=m.build('confirmation_with_second_seed',state,repo/'mixed-plan.json')
    assert len(mixed['workers'])==8
    assert state['stage_limits_seconds']['confirmation_with_second_seed']['estimated_seconds']==10800
    assert mixed['forecast']['critical_path_estimate_seconds']==10800
    for built in [history_plan,train,dev,mixed]:
        assert built['retained_storage']==state['retained_storage']
        assert all(name in built['files'] for name in retained_copies)
    trainer=next(w for w in mixed['workers'] if w['role']=='second_seed_training')
    assert trainer['task_ids']==[] and trainer['worker_fields']['seed']==20260916
    assert not any('/private/' in name or 'confirmation.json' in name for name in trainer['input_files'])
    headline=[w for w in mixed['workers'] if w['role']=='confirmation']
    assert sum(len(w['task_ids']) for w in headline)==200
    assert all('data/coding_pilot_v1/private/confirmation.json' in w['input_files'] for w in headline)
    # A hard-capped training run with no eligible validation checkpoint stops.
    run=repo/roots['training'][0]/'run';done=json.loads((run/'complete.json').read_text());done['nominees']=[]
    write(run/'complete.json',done);write(run/'validation_curve.json',[])
    outer=json.loads((run.parent/'complete.json').read_text());outer['run_complete_sha256']=sha(run/'complete.json');write(run.parent/'complete.json',outer)
    with pytest.raises(m.ScientificStop,match='without an eligible'):m.build('development',state,repo/'no-plan.json')


def retained_fixture(m,repo,state):
    copies=['evidence/recovery/first.tar.gz','evidence/recovery/second.tar.gz']
    for name in copies:
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'verified recovery archive')
    receipt=repo/'evidence/recovery/receipt.json'
    write(receipt,{'volume_id':'retained1','both_copies_verified':True,'copies':copies,'archive_sha256':sha(repo/copies[0])})
    state['retained_storage']=[{'volume_id':'retained1','name':m.PREFIX+'old-storage',
        'recovery_receipt':str(receipt.relative_to(repo)),'sha256':sha(receipt)}]
    return copies,receipt


def test_retained_storage_is_bound_and_rejects_changed_or_duplicate_proofs(tmp_path,monkeypatch):
    m,repo,state=baseline_fixture(tmp_path,monkeypatch)
    copies,receipt=retained_fixture(m,repo,state)
    plan=m.build('memory',state,repo/'memory-plan.json')
    assert plan['retained_storage']==state['retained_storage']
    assert all(plan['files'][name]==sha(repo/name) for name in [*copies,str(receipt.relative_to(repo))])
    (repo/copies[1]).write_bytes(b'corruption')
    with pytest.raises(ValueError,match='backup changed'):m.retained_storage_inputs(state)
    (repo/copies[1]).write_bytes((repo/copies[0]).read_bytes())
    state['retained_storage']*=2
    with pytest.raises(ValueError,match='identity'):m.retained_storage_inputs(state)


def test_retained_storage_cannot_substitute_single_or_unbound_backup(tmp_path,monkeypatch):
    m,repo,state=baseline_fixture(tmp_path,monkeypatch)
    copies,receipt=retained_fixture(m,repo,state)
    proof=json.loads(receipt.read_text());proof['copies']=[copies[0],copies[0]];write(receipt,proof)
    with pytest.raises(ValueError,match='proof changed'):m.retained_storage_inputs(state)
    state['retained_storage'][0]['sha256']=sha(receipt)
    with pytest.raises(ValueError,match='two verified'):m.retained_storage_inputs(state)
