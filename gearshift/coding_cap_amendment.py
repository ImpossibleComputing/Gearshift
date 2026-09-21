"""The single declared reasoning-cap amendment; preserve draws and prior outcomes."""
import json
from pathlib import Path
from gearshift.coding_control import digest,sha
from gearshift.coding_reuse import compatible_parent,verify_completed
from gearshift.coding_resume import load_parent

OLD_CAP=16384
NEW_CAP=24576


def validate_parent(parent_root,repo):
    parent_root,repo=Path(parent_root),Path(repo)
    identity=compatible_parent(parent_root,repo)
    if identity.get('reasoning_cap')!=OLD_CAP or identity.get('cap_amendment_sha256'):
        raise ValueError('Only one cap amendment from the original reasoning budget is permitted')
    cfg=json.loads((repo/'configs/coding_pilot_v1/pilot.json').read_text())
    if cfg['protocol']['development_cap_gate']['one_allowed_reasoning_cap_increase']!=NEW_CAP:
        raise ValueError('Amendment differs from the approved scientific protocol')
    tasks=json.loads((repo/'data/coding_pilot_v1/visible/development.json').read_text())
    ids=[t['task_id'] for t in tasks]
    if len(ids)!=40 or len(set(ids))!=40:raise ValueError('The fixed forty-task cohort is required')
    folders={p.name for p in (parent_root/'tasks').iterdir()}
    if folders!={tid.replace('/','__') for tid in ids} or len(list((parent_root/'tasks').glob('*/complete.json')))!=40:
        raise ValueError('Finish and preserve every original development task before amendment')
    rows=[]
    for tid in ids:
        folder=parent_root/'tasks'/tid.replace('/','__')
        row=verify_completed(folder,identity,tid)
        for kind in ['source','small']:
            history=json.loads((folder/(kind+'_history.json')).read_text())
            if history['reasoning_capped']!=row[kind+'_capped']:
                raise ValueError('Cap labels differ from committed histories')
            if history['reasoning_capped'] and (len(history['reasoning_ids'])!=OLD_CAP or history['natural_boundary'] or history['early_eos']):
                raise ValueError('Capped stream does not contain the complete original reasoning allowance')
        rows.append(row)
    cap_counts={kind:sum(r[kind+'_capped'] for r in rows) for kind in ['source','small']}
    if max(cap_counts.values())<=4:raise ValueError('Cap amendment requires exceeding ten percent of the forty-task cohort')
    gate=json.loads((parent_root/'baseline_gate.json').read_text())
    counts={arm:sum(r['pass'][arm] for r in rows) for arm in ['A','B','D']}
    if gate.get('cap_gate') is not False or gate.get('protocol_cap_amendment_needed') is not True or gate.get('counts')!=counts:
        raise ValueError('Parent gate differs from complete transactions')
    return {'parent_identity':identity,'task_ids':ids,'cap_counts':cap_counts,'counts':counts,
        'repeat_task_ids':[r['task_id'] for r in rows if r['source_capped'] or r['small_capped']],
        'reuse_task_ids':[r['task_id'] for r in rows if not r['source_capped'] and not r['small_capped']],
        'parent_identity_file_sha256':sha(parent_root/'identity.json'),'parent_gate_sha256':sha(parent_root/'baseline_gate.json')}


def replay_constraints(parent_root,parent_identity,task_id):
    folder=Path(parent_root)/'tasks'/task_id.replace('/','__')
    row=verify_completed(folder,parent_identity,task_id)
    prefixes,complete=load_parent(parent_root,task_id)
    extended=[];changed_answers=[]
    for kind,segment,answers in [('source','source_reasoning',['answer_A','answer_D']),('small','small_reasoning',['answer_B'])]:
        if row[kind+'_capped']:
            if len(prefixes[segment])!=OLD_CAP:raise ValueError('Incomplete capped reasoning cannot be extended')
            # Preserve every original reasoning token; only its ending may extend.
            complete.pop(segment);extended.append(segment)
            for answer in answers:
                prefixes.pop(answer,None);complete.pop(answer,None);changed_answers.append(answer)
    return prefixes,complete,{'extended_reasoning_segments':extended,'answers_with_changed_conditioning':changed_answers,
        'parent_complete_sha256':sha(folder/'complete.json'),'old_reasoning_cap':OLD_CAP,'new_reasoning_cap':NEW_CAP,
        'same_original_sampling_streams':True,'all_saved_reasoning_prefixes_must_match':True}
