"""Carry completed, verified measurements between resource sessions without redrawing.

The new envelope records its own identity and the exact original transaction.
Every original output byte and measurement identity remains available unchanged.
"""
import json,shutil,tempfile,os
from pathlib import Path
from gearshift.coding_control import sha,digest,write
NUMERICAL_FILES=('gearshift/core.py','gearshift/coding_inference.py','gearshift/coding_sandbox.py','scripts/coding_sandbox_child.py')
REQUIRED=('source_history.json','small_history.json','A.json','B.json','D.json')

def compatible_parent(parent_root,current_root):
    parent_root=Path(parent_root);current_root=Path(current_root)
    identity=json.loads((parent_root/'identity.json').read_text())
    if identity['config_sha256']!=sha(current_root/'configs/coding_pilot_v1/pilot.json'):raise ValueError('Parent configuration differs')
    if identity['data_identity']!=json.loads((current_root/'data/coding_pilot_v1/identity.json').read_text()):raise ValueError('Parent data identity differs')
    if identity['control_protocol_sha256']!=sha(current_root/'configs/coding_pilot_v1/control_protocol_v2.json'):raise ValueError('Parent protocol differs')
    for name in NUMERICAL_FILES:
        if identity['implementation'].get(name)!=sha(current_root/name):raise ValueError('Numerical generation/scoring code differs: '+name)
    native=json.loads((parent_root/'native_gate.json').read_text())
    if not native['passed'] or native['identity_sha256']!=digest(identity):raise ValueError('Parent native gate identity differs')
    return identity

def verify_completed(folder,parent_identity,task_id):
    folder=Path(folder);row=json.loads((folder/'complete.json').read_text())
    if row['task_id']!=task_id or row['identity_sha256']!=digest(parent_identity):raise ValueError('Completed task identity differs')
    if not set(REQUIRED).issubset(row['files']):raise ValueError('Completed task is missing outputs')
    for name,want in row['files'].items():
        if Path(name).name!=name or (folder/name).is_symlink():raise ValueError('Invalid transaction member')
        if sha(folder/name)!=want:raise ValueError('Completed task file changed: '+name)
    for name in REQUIRED:
        result=json.loads((folder/name).read_text())
        if result['task_id']!=task_id:raise ValueError('Output task identity differs')
        if name in ['A.json','B.json','D.json'] and result['score']['passed']!=row['pass'][name[0]]:raise ValueError('Committed score differs')
    return row

def reuse_completed(parent_folder,destination,parent_identity,new_identity,task_id,parent_logical_path):
    parent_folder=Path(parent_folder);destination=Path(destination)
    row=verify_completed(parent_folder,parent_identity,task_id)
    if destination.exists():raise ValueError('Destination exists; never overwrite a task transaction')
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='reuse-',dir=destination.parent))
    try:
        for name in row['files']:shutil.copy2(parent_folder/name,stage/name)
        source_sha=sha(parent_folder/'complete.json');original_name='upstream_complete_'+source_sha[:16]+'.json'
        shutil.copy2(parent_folder/'complete.json',stage/original_name)
        new=dict(row)
        new['identity_sha256']=digest(new_identity)
        new['measurement_identity_sha256']=row.get('measurement_identity_sha256',row['identity_sha256'])
        new['reused_from']={'logical_path':str(parent_logical_path),'parent_identity_sha256':digest(parent_identity),
            'complete_file_sha256':source_sha,'preserved_complete_file':original_name,'no_new_generation_or_scoring':True}
        new['files']={p.name:sha(p) for p in stage.iterdir() if p.is_file()}
        write(stage/'complete.json',new)
        # Verify copied bytes again before making the new envelope visible.
        verify_completed(stage,new_identity,task_id)
        os.replace(stage,destination)
        return new
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
