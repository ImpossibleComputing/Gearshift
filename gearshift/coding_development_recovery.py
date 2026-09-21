"""Complete only a capacity-missing shard; never redraw completed task outcomes."""
import json
from pathlib import Path
from .coding_control import sha,digest,PREFIX
from .coding_capacity_retry import confirmed_no_capacity

ALLOWED_CONTROL_CHANGES={'gearshift/coding_capacity_retry.py','gearshift/coding_parallel.py','scripts/coding_recovery_plan.py'}

def verify_coverage(root,proof,remaining,plan=None):
    root=Path(root)
    def read(rel):
        from .coding_parallel import file_in
        p=file_in(root,rel)
        if 'inputs' in proof and proof['inputs'].get(rel)!=sha(p):raise ValueError('Recovery proof does not bind its evidence')
        return json.loads(p.read_text())
    tasks=[x['task_id'] for x in read(proof['dataset_path'])]
    parent=read(proof['parent_plan']);done=read(proof['parent_completion']);failure=read(proof['failure'])
    if len(tasks)!=40 or len(set(tasks))!=40 or parent['task_ids']!=tasks:raise ValueError('Original forty-task cohort changed')
    if parent['stage']!='exploratory_development' or done.get('passed') is not False or done['stage_identity']!=digest(parent):raise ValueError('Parent is not a completed partial development attempt')
    if not confirmed_no_capacity(failure) or done['errors']!=[failure['error']]:raise ValueError('Only this single confirmed capacity failure may be recovered')
    failed=proof['failed_worker_id'];workers={w['worker_id']:w for w in parent['workers']}
    if failed not in workers or workers[failed]['task_ids']!=remaining:raise ValueError('Recovery must retain exactly the missing original shard')
    if {w['worker_id'] for w in done['workers']}!=set(workers)-{failed} or not all(w['passed'] for w in done['workers']):raise ValueError('Healthy sibling completion differs')
    ledger=read(proof['ledger_snapshot']);prefix=PREFIX+parent['run_id']+'-'+failed
    if any(r.get('kind')=='pod' and r.get('name','').startswith(prefix) for r in ledger['resources']):raise ValueError('Capacity shard already received a GPU')
    if plan is not None:
        if len(plan['workers'])!=1 or plan['image_digest']!=parent['image_digest']:raise ValueError('Recovery uses one unchanged worker image')
        new=plan['workers'][0];old=workers[failed]
        if new['command']!=old['command'] or new['worker_fields']!=old['worker_fields']:raise ValueError('Recovery changed scientific worker inputs')
    if any(r.get('name','').startswith(prefix) and 'absent_epoch' not in r for r in ledger['resources']):raise ValueError('Failed shard storage not removed')
    selected=sha(root/proof['selection']);preserved=[]
    for name in proof['preserved_worker_roots']:
        path=root/name;identity=read(name+'/identity.json');complete=read(name+'/complete.json');native=read(name+'/native_gate.json')
        wid=identity['worker_id']
        if wid==failed or wid not in workers or identity['stage_identity']!=digest(parent) or complete['identity_sha256']!=digest(identity):raise ValueError('Parent worker identity differs')
        if not native['passed'] or native['identity_sha256']!=digest(identity):raise ValueError('Parent numerical controls did not pass')
        if complete['selection_sha256']!=selected or set(complete['tasks'])!=set(workers[wid]['task_ids']):raise ValueError('Checkpoint selection or preserved shard differs')
        for tid,expected in complete['tasks'].items():
            task=path/'tasks'/tid.replace('/','__');receipt=read(str(task.relative_to(root))+'/complete.json')
            if sha(task/'complete.json')!=expected or receipt['task_id']!=tid or receipt['identity_sha256']!=digest(identity):raise ValueError('Preserved task transaction differs')
            for rel,expected_file in receipt['files'].items():
                if Path(rel).name!=rel or sha(task/rel)!=expected_file:raise ValueError('Preserved raw output changed')
            if tid in preserved:raise ValueError('Repeated preserved task')
            preserved.append(tid)
    if set(preserved)&set(remaining) or set(preserved)|set(remaining)!=set(tasks) or len(preserved)+len(remaining)!=40:raise ValueError('Recovery is not the exact complement of preserved outcomes')
    if sorted(preserved)!=proof['preserved_task_ids']:raise ValueError('Preserved coverage receipt differs')
    for rel,expected in parent['files'].items():
        if rel not in ALLOWED_CONTROL_CHANGES and sha(root/rel)!=expected:raise ValueError('Scientific input changed during capacity recovery: '+rel)
    return preserved
