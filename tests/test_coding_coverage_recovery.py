import copy,json
import pytest
from gearshift.coding_control import write,sha
from gearshift.coding_coverage import validate_recovery


def fixture(root):
    rel='results/coding_pilot_v1/coverage_interrupted/experiment';source=root/rel
    for name in ['training_failure.json','validation_curve.json','training_identity.json','optimizer_reset.json',
                 'frozen_schedule_prefix.json','frozen_endpoint.json','answer_span_audit.json','planned_exposure.json']:
        write(source/name,{'record':name})
    rows=[{'step':i,'normalization_per_arm':32,'arms':{a:{'gradient_predictions':32,'cache_gradients_finite_nonzero':True} for a in ['FIXED','ROTATING']}} for i in range(1,301)]
    write(source/'training_steps.json',rows)
    for step in [0,128,256]:
        pair={}
        for a in ['FIXED','ROTATING']:
            p=source/a/f'mapper_step_{step:04d}.pt';p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(f'{a}/{step}'.encode())
            pair[a]={'path':str(p.relative_to(root)),'sha256':sha(p),'bytes':p.stat().st_size}
        write(source/'paired_checkpoints'/f'step_{step:04d}.json',{'step':step,'arms':pair,'paired_commit':True})
    receipt={'source_result_root':rel,'training_resumed':False,'quality_scores_consulted':False,'target_updates':1024,
             'primary_common_checkpoint':256,'completed_logged_updates':300,
             'inputs':{str(p.relative_to(root)):sha(p) for p in source.rglob('*') if p.is_file()}}
    return receipt,source


def test_recovery_uses_latest_pair_without_quality_selection(tmp_path):
    receipt,source=fixture(tmp_path)
    assert validate_recovery(tmp_path,receipt)['step']==256
    receipt['primary_common_checkpoint']=128
    with pytest.raises(ValueError,match='latest durable'):validate_recovery(tmp_path,receipt)


@pytest.mark.parametrize('field',['training_resumed','quality_scores_consulted'])
def test_recovery_rejects_new_training_or_score_selection(tmp_path,field):
    receipt,_=fixture(tmp_path);receipt[field]=True
    with pytest.raises(ValueError):validate_recovery(tmp_path,receipt)


def test_recovery_rejects_changed_weights_and_unbound_records(tmp_path):
    receipt,source=fixture(tmp_path);p=source/'FIXED/mapper_step_0256.pt';old=p.read_bytes();p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):validate_recovery(tmp_path,receipt)
    p.write_bytes(old);del receipt['inputs'][str((source/'training_steps.json').relative_to(tmp_path))]
    with pytest.raises(ValueError,match='Unbound'):validate_recovery(tmp_path,receipt)


def test_recovery_rejects_incomplete_pair_or_unmatched_update(tmp_path):
    receipt,source=fixture(tmp_path);p=source/'paired_checkpoints/step_0256.json';row=json.loads(p.read_text());row['paired_commit']=False;write(p,row)
    receipt['inputs'][str(p.relative_to(tmp_path))]=sha(p)
    with pytest.raises(ValueError,match='Incomplete paired'):validate_recovery(tmp_path,receipt)
    row['paired_commit']=True;write(p,row);receipt['inputs'][str(p.relative_to(tmp_path))]=sha(p)
    p=source/'training_steps.json';rows=json.loads(p.read_text());rows[-1]['arms']['ROTATING']['gradient_predictions']=31;write(p,rows)
    receipt['inputs'][str(p.relative_to(tmp_path))]=sha(p)
    with pytest.raises(ValueError,match='gradient checks'):validate_recovery(tmp_path,receipt)


def test_recovery_rejects_prior_answers(tmp_path):
    receipt,source=fixture(tmp_path);write(source/'validation/task/FIXED_M/seed_0/answer.json',{'score':True})
    with pytest.raises(ValueError,match='prior quality generation'):validate_recovery(tmp_path,receipt)


@pytest.mark.parametrize('generation_fails',[False,True])
def test_recovered_evaluation_finishes_every_generation_before_loading_tests(tmp_path,monkeypatch,generation_fails):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import scripts.coding_coverage_experiment as experiment
    import gearshift.coding_sandbox as sandbox
    monkeypatch.setattr(experiment,'ROOT',tmp_path)
    validation=[{'task_id':f'validation/{i}'} for i in range(21)]
    training=[{'task_id':f'training/{i}'} for i in range(4)]
    tids=[x['task_id'] for x in validation+training]
    d={'answer_seeds_path':'seeds.json','seen_checkpoint':'seen.pt','seen_checkpoint_sha256':'seen',
       'validation_task_ids':[x['task_id'] for x in validation],'seen_training_task_ids':[x['task_id'] for x in training]}
    write(tmp_path/'seeds.json',{'validation':{},'seen_training':{}})
    write(tmp_path/'data/coding_pilot_v1/visible/coverage_generalization.json',[{'task_id':t} for t in tids])
    write(tmp_path/'data/coding_pilot_v1/private/coverage_generalization.json',{t:{} for t in tids})
    write(tmp_path/'evidence/coding_pilot_v1/sandbox_gate.json',{'passed':True})
    events=[];read=experiment.read
    def checked_read(p):
        if p == tmp_path/'data/coding_pilot_v1/private/coverage_generalization.json':
            assert events==['validation','seen_training'];events.append('private_loaded')
        return read(p)
    monkeypatch.setattr(experiment,'read',checked_read)
    def generate(c,source,b,mapper,histories,visible,seeds,checkpoints,scope):
        if generation_fails:raise RuntimeError('interrupted generation')
        labels=['START_M','START_H','FIXED_M','FIXED_H','ROTATING_M','ROTATING_H','D','P'] if scope=='validation' else ['START_M','SEEN_FIT_M','D']
        for h in histories:
            for label in labels:
                for seed in range(3 if scope=='validation' else 5):
                    write(c['root']/scope/h['task_id'].replace('/','_')/label/f'seed_{seed}'/'answer.json',
                          {'task_id':h['task_id'],'condition':label,'answer_text':'```python\npass\n```'})
        events.append(scope)
    monkeypatch.setattr(experiment,'generate_suite',generate)
    monkeypatch.setattr(experiment,'finish_worker',lambda *a,**kw:events.append('complete'))
    monkeypatch.setattr(sandbox,'score',lambda *a,**kw:{'category':'passed','passed':True})
    c={'root':tmp_path/'results/recovery','guard':lambda:None,'publish':lambda **kw:None,'identity':{}}
    args=(c,d,training,validation,None,None,None,{'START':{'path':'start.pt','sha256':'start'}},128,1024,'engineering_incomplete_last_common_checkpoint',0)
    if generation_fails:
        with pytest.raises(RuntimeError,match='interrupted generation'):experiment.evaluate_and_score(*args)
        assert events==[]
    else:
        experiment.evaluate_and_score(*args)
        assert events==['validation','seen_training','private_loaded','complete']
        assert read(c['root']/'complete.json')['fresh_scored_answers']==564
