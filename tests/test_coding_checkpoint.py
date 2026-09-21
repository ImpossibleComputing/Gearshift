import json
import pytest
from gearshift.coding_control import write,sha,digest
from gearshift.coding_reuse import REQUIRED,NUMERICAL_FILES
from gearshift.coding_checkpoint import validate_checkpoint


@pytest.fixture
def saved(tmp_path):
    root=tmp_path/'repo';parent=root/'results/parent'
    cfg=root/'configs/coding_pilot_v1/pilot.json';write(cfg,{})
    protocol=root/'configs/coding_pilot_v1/control_protocol_v2.json';write(protocol,{'v':2})
    write(root/'data/coding_pilot_v1/identity.json',{'fixed':True})
    write(root/'data/coding_pilot_v1/visible/development.json',[{'task_id':f'task/{i}'} for i in range(40)])
    for name in NUMERICAL_FILES:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('numerical code')
    identity={'config_sha256':sha(cfg),'control_protocol_sha256':sha(protocol),'data_identity':{'fixed':True},
        'implementation':{n:sha(root/n) for n in NUMERICAL_FILES}}
    write(parent/'identity.json',identity);write(parent/'native_gate.json',{'passed':True,'identity_sha256':digest(identity)})
    for i in range(6):
        folder=parent/'tasks'/f'task__{i}'
        for name in REQUIRED:write(folder/name,{'task_id':f'task/{i}','score':{'passed':True}})
        write(folder/'complete.json',{'task_id':f'task/{i}','identity_sha256':digest(identity),
            'files':{p.name:sha(p) for p in folder.iterdir()},'pass':{'A':True,'B':True,'D':True},
            'source_capped':False,'small_capped':False})
    write(parent/'progress.json',{'completed_tasks':6,'total_tasks':40,'passes':{'A':6,'B':6,'D':6},
        'source_cap_count':0,'small_cap_count':0,'observed_baseline_seconds_including_parent':600,
        'remaining_baseline_forecast_seconds':3400})
    return root,parent


def test_checkpoint_binds_membership_measured_runtime_and_original_transactions(saved):
    root,parent=saved;c=validate_checkpoint(parent,root)
    assert c['task_ids']==[f'task/{i}' for i in range(6)] and c['observed_seconds']==600
    assert c['complete_sha256']['task/0']==sha(parent/'tasks/task__0/complete.json')


@pytest.mark.parametrize('change',[{'completed_tasks':5},{'passes':{'A':0,'B':6,'D':6}},
    {'source_cap_count':1},{'remaining_baseline_forecast_seconds':1}])
def test_inconsistent_progress_is_rejected(saved,change):
    root,parent=saved;p=parent/'progress.json';write(p,{**json.loads(p.read_text()),**change})
    with pytest.raises(ValueError):validate_checkpoint(parent,root)


def test_unfinished_or_extra_task_cannot_be_silently_discarded(saved):
    root,parent=saved;write(parent/'tasks/task__6/partial.json',{'token_ids':[1,2]})
    with pytest.raises(ValueError,match='partial task'):validate_checkpoint(parent,root)


def test_changed_completed_measurement_rejected(saved):
    root,parent=saved;write(parent/'tasks/task__0/A.json',{'task_id':'task/0','score':{'passed':False}})
    with pytest.raises(ValueError,match='file changed'):validate_checkpoint(parent,root)
