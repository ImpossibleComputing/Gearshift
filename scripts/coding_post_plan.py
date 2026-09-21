#!/usr/bin/env python3
"""Hash-bound one-worker diagnostic allocations; never allocates on import."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import bind,sha
from gearshift.coding_parallel import validate_plan
from gearshift.coding_post_progress import DECLARATION,OWNER
from coding_pipeline_plan import retained_storage_inputs
ROOT=Path(__file__).resolve().parents[1]
BASE='results/coding_pilot_v1/cap_recovery_parallel_06/worker_00'
APPROVAL='evidence/coding_pilot_v1/control/parallel_500h_approved.json'
AMEND='configs/coding_pilot_v1/recovery_20260917/amendment.json'

def build(mode,run_id,numerical_root=None,hybrid_root=None):
    d=json.loads((ROOT/DECLARATION).read_text());files=set();gates={}
    def add(rel):
        p=ROOT/rel
        if not p.is_file() or p.is_symlink():raise ValueError('Missing/unsafe '+str(rel))
        files.add(p)
    def proof(name,paths,**metadata):
        for p in paths:add(p)
        path=ROOT/'evidence/coding_pilot_v1/post_progress01'/('proof_'+name+'_'+run_id+'.json')
        bind(path,{'gate':name,'passed':True,'inputs':{str(p):sha(ROOT/p) for p in paths},**metadata});files.add(path)
        gates[name]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
    for pattern in ('gearshift/*.py','scripts/coding_*.py','configs/coding_pilot_v1/reference/**/*.json'):
        for p in ROOT.glob(pattern):files.add(p)
    for rel in ('configs/coding_pilot_v1/pilot.json','configs/coding_pilot_v1/training_protocol_v1.json','configs/coding_pilot_v1/cap_amendment_v1.json','data/coding_pilot_v1/identity.json',APPROVAL,AMEND,DECLARATION,OWNER,'configs/coding_pilot_v1/recovery_20260917/OWNER_INSTRUCTION.txt',d['selected_checkpoint'],d['corpus_manifest']):add(rel)
    for p in (ROOT/BASE).rglob('*.json'):files.add(p)
    proof('baseline',[BASE+'/baseline_gate.json',BASE+'/identity.json'])
    proof('amendment',[AMEND,'configs/coding_pilot_v1/recovery_20260917/OWNER_INSTRUCTION.txt'])
    proof('post_progress_declaration',[DECLARATION,OWNER,'evidence/coding_pilot_v1/post_progress01/publication_tag_verification.json'])
    if mode!='numerical':
        nr=Path(numerical_root);x=json.loads((ROOT/nr/'numerical_comparison.json').read_text())
        if not x['complete'] or not x['passed']:raise ValueError('Numerical paths require investigation')
        review_path='evidence/coding_pilot_v1/post_progress01/numerical_interpretation.json'
        review=json.loads((ROOT/review_path).read_text())
        if review.get('comparison_sha256')!=sha(ROOT/nr/'numerical_comparison.json') or review.get('material_discrepancies_resolved') is not True:raise ValueError('Numerical interpretation required before subsequent stages')
        proof('numerical_paths',[str(nr/'numerical_comparison.json'),str(nr/'complete.json'),str(nr/'identity.json'),review_path],path_comparison_passed=True)
    if mode=='memorization':
        hr=Path(hybrid_root);x=json.loads((ROOT/hr/'complete.json').read_text())
        if len(x['tasks'])!=40:raise ValueError('Prompt comparison must complete first')
        proof('prompt_experiment',[str(hr/'complete.json'),str(hr/'identity.json')])
    train=json.loads((ROOT/'data/coding_pilot_v1/visible/training.json').read_text())
    chosen={x['task_id'] for x in d['four_training_histories']};lookup={r['task_id']:r for r in train}
    selected=[lookup[r['task_id']] for r in d['four_training_histories']]
    cohort_file='data/coding_pilot_v1/visible/post_progress01_four_training.json';bind(ROOT/cohort_file,selected)
    for row in d['four_training_histories']:
        for name in ('source_history.json','teacher_answer.json','complete.json'):add(row['folder']+'/'+name)
    if mode=='hybrid':
        cohort_file='data/coding_pilot_v1/visible/development.json';selected=json.loads((ROOT/cohort_file).read_text());add('data/coding_pilot_v1/private/development.json')
        for p in ROOT.glob('results/coding_pilot_v1/recovery_dev_20260917_*/dev*/tasks/*/*.json'):files.add(p)
    elif mode=='memorization':
        # Only these four scorer records are exported; scorer inputs stay outside optimizer.
        private=json.loads((ROOT/'data/coding_pilot_v1/private/training.json').read_text())
        small='data/coding_pilot_v1/private/post_progress01_four_training.json';bind(ROOT/small,{tid:private[tid] for tid in chosen});add(small)
    elif mode!='numerical':raise ValueError('Unknown diagnostic stage')
    add(cohort_file)
    old=json.loads((ROOT/'configs/coding_pilot_v1/parallel_history_recovery_01.json').read_text());refs,retained=retained_storage_inputs({'retained_storage':old['retained_storage']});files.update(retained)
    hours={'numerical':3,'hybrid':3,'memorization':6}[mode]
    worker={'worker_id':mode,'command':['scripts/coding_post_'+mode+'.py'],'task_ids':[r['task_id'] for r in selected],'region':'US-CO-1','region_candidates':['US-CO-1','AP-JP-1'],'estimated_seconds':hours*2400,'maximum_seconds':hours*3600,'worker_fields':{'approval_path':APPROVAL,'baseline_root':BASE,'amendment_sha256':sha(ROOT/AMEND),'declaration_sha256':sha(ROOT/DECLARATION),'diagnostic_mode':mode}}
    plan={'schema':1,'run_id':run_id,'stage':'post_progress_'+mode,'approval_sha256':sha(ROOT/APPROVAL),'image_digest':old['image_digest'],'gates':gates,'task_ids':worker['task_ids'],'cohort_files':[cohort_file],'workers':[worker],'retained_storage':refs,'preserve_original_measurements':True,'scientific_scope_unchanged':False,'scope_amendment':'post_progress01','files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},'forecast':{'maximum_gpu_hours':hours,'method':'Bounded engineering reservation; not completion forecast'}}
    # Additional diagnostic cap is enforced against the reconciled cumulative ledger.
    current=json.loads((ROOT/'evidence/coding_pilot_v1/control/watchdog_status.json').read_text());start=d['budget_start']
    if current['gpu_hours']-start['gpu_hours']+hours>d['diagnostic_max_additional_gpu_hours'] or current['upper_usd']-start['upper_usd']+hours*5.56>d['diagnostic_max_additional_usd']:raise ValueError('Post-progress diagnostic sub-budget exhausted')
    path=ROOT/'configs/coding_pilot_v1/post_progress01'/(run_id+'.json');bind(path,plan)
    identity=validate_plan(plan,ROOT,sha(ROOT/APPROVAL));print(json.dumps({'path':str(path),'stage_identity':identity}));return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('run_id');p.add_argument('--numerical-root');p.add_argument('--hybrid-root');a=p.parse_args();build(a.mode,a.run_id,a.numerical_root,a.hybrid_root)
