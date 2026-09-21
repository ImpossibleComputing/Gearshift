import json
from pathlib import Path
import pytest
from gearshift.coding_control import write,sha,digest
from gearshift.coding_reuse import verify_completed,reuse_completed,compatible_parent,REQUIRED,NUMERICAL_FILES

def parent(tmp_path):
    root=tmp_path/'source';identity={'run':'parent'}
    for name in REQUIRED:write(root/name,{'task_id':'t','score':{'passed':True},'verbatim_answer':'x = 1\n\n'})
    write(root/'complete.json',{'task_id':'t','identity_sha256':digest(identity),'pass':{'A':True,'B':True,'D':True},
        'files':{p.name:sha(p) for p in root.iterdir()}})
    return root,identity

def test_reuse_preserves_original_bytes_and_measurement_identity(tmp_path):
    root,identity=parent(tmp_path);before={p.name:p.read_bytes() for p in root.iterdir()};new={'run':'continuation'}
    row=reuse_completed(root,tmp_path/'dest',identity,new,'t','results/parent/tasks/t')
    assert row['identity_sha256']==digest(new) and row['measurement_identity_sha256']==digest(identity)
    for name,data in before.items():
        assert (root/name).read_bytes()==data
        destname=row['reused_from']['preserved_complete_file'] if name=='complete.json' else name
        assert (tmp_path/'dest'/destname).read_bytes()==data
    verify_completed(tmp_path/'dest',new,'t')

def test_tampered_score_or_mixed_identity_cannot_be_reused(tmp_path):
    root,identity=parent(tmp_path);write(root/'A.json',{'task_id':'t','score':{'passed':False}})
    with pytest.raises(ValueError,match='file changed'):reuse_completed(root,tmp_path/'dest',identity,{},'t','parent')
    assert not (tmp_path/'dest').exists()
    with pytest.raises(ValueError,match='identity'):verify_completed(root,{'run':'different'},'t')

def test_reuse_cannot_overwrite_a_completed_destination(tmp_path):
    root,identity=parent(tmp_path);dest=tmp_path/'dest';dest.mkdir();(dest/'keep').write_text('original')
    with pytest.raises(ValueError,match='never overwrite'):reuse_completed(root,dest,identity,{},'t','parent')
    assert (dest/'keep').read_text()=='original'

def test_parent_reuse_checks_generation_code_and_protocol(tmp_path):
    root=tmp_path/'repo';p=root/'results/parent'
    cfg=root/'configs/coding_pilot_v1/pilot.json';write(cfg,{})
    protocol=root/'configs/coding_pilot_v1/control_protocol_v2.json';write(protocol,{'v':2})
    write(root/'data/coding_pilot_v1/identity.json',{'data':'fixed'})
    for name in NUMERICAL_FILES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original code')
    identity={'config_sha256':sha(cfg),'control_protocol_sha256':sha(protocol),'data_identity':{'data':'fixed'},
        'implementation':{name:sha(root/name) for name in NUMERICAL_FILES}}
    write(p/'identity.json',identity);write(p/'native_gate.json',{'passed':True,'identity_sha256':digest(identity)})
    assert compatible_parent(p,root)==identity
    (root/NUMERICAL_FILES[0]).write_text('changed code')
    with pytest.raises(ValueError,match='Numerical'):compatible_parent(p,root)
