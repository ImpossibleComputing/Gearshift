"""Strict scope and deterministic selection for the post-publication diagnostics."""
import json
from pathlib import Path
from .coding_control import sha

DECLARATION='configs/coding_pilot_v1/post_progress01/declaration.json'
OWNER='configs/coding_pilot_v1/post_progress01/OWNER_INSTRUCTION.txt'

def validate_diagnostic_plan(plan,root):
    from .coding_parallel import verify_gate,confirmation_input
    root=Path(root);proof=verify_gate(root,'post_progress_declaration',plan['gates']['post_progress_declaration'])
    d=json.loads((root/DECLARATION).read_text())
    if proof['inputs'].get(DECLARATION)!=sha(root/DECLARATION) or proof['inputs'].get(OWNER)!=sha(root/OWNER):raise ValueError('Unbound diagnostic instruction')
    if plan.get('scope_amendment')!='post_progress01' or plan.get('scientific_scope_unchanged') is not False:raise ValueError('New diagnostic identity required')
    if any(confirmation_input(p) for p in plan['files']):raise ValueError('Reserved inputs forbidden')
    if len(plan['workers'])!=1:raise ValueError('One bounded worker per diagnostic stage')
    if sum(w['maximum_seconds'] for w in plan['workers'])/3600>d['diagnostic_max_additional_gpu_hours']:raise ValueError('Diagnostic allowance exceeded')
    expected=[]
    if plan['stage']=='post_progress_hybrid':
        expected=[x['task_id'] for x in json.loads((root/'data/coding_pilot_v1/visible/development.json').read_text())]
        if len(expected)!=40:raise ValueError('Forty inspected tasks required')
    elif plan['stage'] in ('post_progress_numerical','post_progress_memorization'):
        expected=[x['task_id'] for x in d['four_training_histories']]
        if len(expected)!=4 or len(set(expected))!=4:raise ValueError('Four predeclared training histories required')
    if plan['task_ids']!=expected:raise ValueError('Diagnostic cohort changed')
    if plan['stage']=='post_progress_numerical' and any('/private/' in p for p in plan['files']):raise ValueError('Numerical checks have no scorer inputs')
    for name in ('numerical_paths','prompt_experiment'):
        if name in plan['gates']:
            gate=verify_gate(root,name,plan['gates'][name])
            if name=='numerical_paths' and gate.get('path_comparison_passed') is not True:raise ValueError('Unresolved numerical discrepancy')
    for w in plan['workers']:
        if w['worker_fields'].get('declaration_sha256')!=sha(root/DECLARATION):raise ValueError('Worker declaration changed')

def selected_histories(repo,declaration):
    from .coding_training import read,validate_history
    rows=[]
    for row in declaration['four_training_histories']:
        folder=Path(repo)/row['folder']
        for name,key in [('source_history.json','source_history_sha256'),('teacher_answer.json','teacher_answer_sha256')]:
            if sha(folder/name)!=row[key]:raise ValueError('Saved training history changed')
        rows.append(validate_history({'task_id':row['task_id'],'split':'training','source_history':read(folder/'source_history.json'),'teacher_answer':read(folder/'teacher_answer.json')}))
    return rows

def splice(native,mapped,prompt_length):
    """Use exact original absolute suffix positions; never rerotate a cropped key."""
    from .core import CacheInjector
    if any(x.shape[-2]!=prompt_length for pair in native for x in pair):raise ValueError('Native prompt boundary differs')
    return CacheInjector.splice_prefix(native,mapped)
