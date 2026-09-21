#!/usr/bin/env python3
"""Immutable one-GPU paired coverage plans; budget reconciliation before spend."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import bind,sha
from gearshift.coding_coverage import DECLARATION,OWNER,CONFIG,EVIDENCE
from gearshift.coding_parallel import validate_plan
from coding_pipeline_plan import retained_storage_inputs
from coding_cloud_guard import tick
ROOT=Path(__file__).resolve().parents[1]
BASE='results/coding_pilot_v1/cap_recovery_parallel_06/worker_00'
APPROVAL='evidence/coding_pilot_v1/control/parallel_500h_approved.json'
AMEND='configs/coding_pilot_v1/recovery_20260917/amendment.json'


def build(mode,run_id,endpoint_path=None,recovery_path=None):
    d=json.loads((ROOT/DECLARATION).read_text());files=set();gates={}
    def add(rel):
        p=ROOT/rel
        if not p.is_file() or p.is_symlink():raise ValueError('Missing/unsafe '+str(rel))
        files.add(p)
    def proof(name,paths,**metadata):
        for p in paths:add(p)
        path=ROOT/EVIDENCE/('proof_'+name+'_'+run_id+'.json')
        bind(path,{'gate':name,'passed':True,'inputs':{str(p):sha(ROOT/p) for p in paths},**metadata});files.add(path)
        gates[name]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
    for pattern in ('gearshift/*.py','scripts/coding_*.py','configs/coding_pilot_v1/reference/**/*.json'):
        files.update(ROOT.glob(pattern))
    for rel in ('configs/coding_pilot_v1/pilot.json','configs/coding_pilot_v1/training_protocol_v1.json','configs/coding_pilot_v1/cap_amendment_v1.json','data/coding_pilot_v1/identity.json',APPROVAL,AMEND,DECLARATION,OWNER,'configs/coding_pilot_v1/recovery_20260917/OWNER_INSTRUCTION.txt',d['selected_checkpoint'],d['corpus_manifest'],d['schedules_path'],d['panels_path'],d['answer_seeds_path']):add(rel)
    for p in (ROOT/BASE).rglob('*.json'):files.add(p)
    manifest=json.loads((ROOT/d['corpus_manifest']).read_text())
    for rel in manifest['files']:
        if not rel.endswith('.pt'):add(rel)
    proof('baseline',[BASE+'/baseline_gate.json',BASE+'/identity.json'])
    proof('amendment',[AMEND,'configs/coding_pilot_v1/recovery_20260917/OWNER_INSTRUCTION.txt'])
    proof('coverage_declaration',[DECLARATION,OWNER,EVIDENCE+'/publication_tag_verification.json',d['schedules_path'],d['panels_path'],d['answer_seeds_path']])
    nr='results/coding_pilot_v1/post_progress01_numerical_20260917_03/numerical'
    proof('numerical_paths',[nr+'/numerical_comparison.json',nr+'/complete.json',nr+'/identity.json','evidence/coding_pilot_v1/post_progress01/numerical_interpretation.json'],path_comparison_passed=True)
    cohort='data/coding_pilot_v1/visible/coverage_generalization.json';add(cohort)
    hours=2 if mode=='preflight' else (10 if mode=='evaluation_recovery' else 24)
    fields={'approval_path':APPROVAL,'baseline_root':BASE,'amendment_sha256':sha(ROOT/AMEND),'declaration_sha256':sha(ROOT/DECLARATION),'coverage_mode':mode}
    if mode in ['experiment','evaluation_recovery']:
        if not endpoint_path:raise ValueError('Measured frozen endpoint required')
        ep=json.loads((ROOT/endpoint_path).read_text())
        proof('coverage_endpoint',[endpoint_path,ep['preflight_result_path']])
        fields.update(endpoint_path=endpoint_path,endpoint_sha256=sha(ROOT/endpoint_path))
        add(d['seen_checkpoint'])
        validation=json.loads((ROOT/'data/coding_pilot_v1/private/validation.json').read_text())
        training=json.loads((ROOT/'data/coding_pilot_v1/private/training.json').read_text())
        private='data/coding_pilot_v1/private/coverage_generalization.json'
        bind(ROOT/private,{**{tid:validation[tid] for tid in d['validation_task_ids']},**{tid:training[tid] for tid in d['seen_training_task_ids']}});add(private)
        if mode=='evaluation_recovery':
            from gearshift.coding_coverage import validate_recovery
            if not recovery_path:raise ValueError('Frozen recovery decision required')
            recovery=json.loads((ROOT/recovery_path).read_text());validate_recovery(ROOT,recovery)
            for rel in recovery['inputs']:add(rel)
            proof('coverage_recovery',[recovery_path,*recovery['inputs']])
            fields.update(recovery_path=recovery_path,recovery_sha256=sha(ROOT/recovery_path))
    elif mode!='preflight':raise ValueError('Unknown coverage mode')
    old=json.loads((ROOT/'configs/coding_pilot_v1/parallel_history_recovery_01.json').read_text())
    retained_refs=list(old['retained_storage'])
    if mode=='evaluation_recovery':retained_refs.extend(recovery.get('retained_storage',[]))
    refs,retained=retained_storage_inputs({'retained_storage':retained_refs});files.update(retained)
    worker={'worker_id':mode,'command':['scripts/coding_coverage_worker.py'],'task_ids':d['training_task_ids']+d['validation_task_ids'],
        'region':'US-CO-1','region_candidates':['US-CO-1','AP-JP-1'],'estimated_seconds':hours*2400,'maximum_seconds':hours*3600,'worker_fields':fields}
    tick();current=json.loads((ROOT/'evidence/coding_pilot_v1/control/watchdog_status.json').read_text());start=d['budget_start']
    if current['stop'] or current['gpu_hours']-start['gpu_hours']+hours>32 or current['upper_usd']-start['upper_usd']+hours*5.56>200:
        raise ValueError('Coverage sub-budget or watchdog blocks allocation')
    plan={'schema':1,'run_id':run_id,'stage':'coverage_'+mode,'approval_sha256':sha(ROOT/APPROVAL),'image_digest':old['image_digest'],
        'gates':gates,'task_ids':worker['task_ids'],'cohort_files':[cohort],'workers':[worker],'retained_storage':refs,
        'preserve_original_measurements':True,'scientific_scope_unchanged':False,'scope_amendment':'coverage_generalization_v1',
        'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
        'forecast':{'maximum_gpu_hours':hours,'budget_reconciliation_epoch':current['epoch'],'cumulative_upper_usd':current['upper_usd'],'cumulative_gpu_hours':current['gpu_hours'],
                    'additional_spend_reservation':hours*5.56,'method':'Bounded worker lifetime, including setup, training, validation, generation and export.'}}
    path=ROOT/CONFIG/(run_id+'.json');bind(path,plan)
    identity=validate_plan(plan,ROOT,sha(ROOT/APPROVAL));print(json.dumps({'path':str(path),'stage_identity':identity}));return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('run_id');p.add_argument('--endpoint-path');p.add_argument('--recovery-path');a=p.parse_args();build(a.mode,a.run_id,a.endpoint_path,a.recovery_path)
