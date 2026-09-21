#!/usr/bin/env python3
"""Construct separate owner-amended dispatch plans; never allocate here."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha,bind
from gearshift.coding_parallel import validate_plan,task_shards
from coding_memory_preflight import check_baselines
from coding_pipeline_plan import retained_storage_inputs
ROOT=Path(__file__).resolve().parents[1]
BASE='results/coding_pilot_v1/cap_recovery_parallel_06/worker_00'
AMEND='configs/coding_pilot_v1/recovery_20260917/amendment.json'
APPROVAL='evidence/coding_pilot_v1/control/parallel_500h_approved.json'
CORPUS='results/coding_pilot_v1/partial_corpus_20260917_v1'

def build(mode,run_id,recovery_parent=None):
    old=json.loads((ROOT/'configs/coding_pilot_v1/parallel_history_recovery_01.json').read_text())
    proofroot=ROOT/'evidence/coding_pilot_v1/recovery_20260917/proofs';files=set();gates={}
    def add(p):
        p=Path(p);p=p if p.is_absolute() else ROOT/p
        if not p.is_file() or p.is_symlink():raise ValueError('Missing/unsafe '+str(p))
        files.add(p)
    def proof(name,paths,**metadata):
        for p in paths:add(p)
        obj={'gate':name,'passed':True,'inputs':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files) if str(p.relative_to(ROOT)) in paths},**metadata}
        path=proofroot/(name+'_'+run_id+'.json');bind(path,obj);add(path);gates[name]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
    for pattern in ['gearshift/*.py','scripts/coding_*.py','configs/coding_pilot_v1/reference/**/*.json']:
        for p in ROOT.glob(pattern):
            if p.name not in {'coding_recovery_pipeline.py','coding_recovery_report.py','coding_recovery_bundle.py','coding_recovery_capacity.py'}:add(p)
    for name in ['configs/coding_pilot_v1/pilot.json','configs/coding_pilot_v1/training_protocol_v1.json','configs/coding_pilot_v1/cap_amendment_v1.json','data/coding_pilot_v1/identity.json',APPROVAL,AMEND,'configs/coding_pilot_v1/recovery_20260917/OWNER_INSTRUCTION.txt']:
        add(name)
    for p in (ROOT/BASE).rglob('*.json'):add(p)
    check=check_baselines(ROOT/BASE)
    proof('baseline',[BASE+'/baseline_gate.json',BASE+'/identity.json'],**check)
    proof('amendment',[AMEND,'configs/coding_pilot_v1/recovery_20260917/OWNER_INSTRUCTION.txt'],owner_instruction_sha256=json.loads((ROOT/AMEND).read_text())['instruction_sha256'])
    refs,retained=retained_storage_inputs({'retained_storage':old['retained_storage']});files.update(retained)
    common={'approval_path':APPROVAL,'baseline_root':BASE,'amendment_sha256':sha(ROOT/AMEND)}
    if mode=='probe':
        stage='recovery_probe';cohort_file='data/coding_pilot_v1/visible/recovery_probe_20260917.json';tid='leetcode/2893'
        tasks=[r for s in ['training','validation'] for r in json.loads((ROOT/f'data/coding_pilot_v1/visible/{s}.json').read_text()) if r['task_id']==tid]
        assert len(tasks)==1;bind(ROOT/cohort_file,tasks);add(cohort_file)
        saved='results/coding_pilot_v1/history_recovery_20260916_01/history07/tasks/leetcode__2893/inflight_source_reasoning.json';add(saved)
        for p in [Path(saved).parent.parent.parent/'identity.json',Path(saved).parent.parent.parent/'failure.json']:add(p)
        workers=[{'worker_id':'probe','command':['scripts/coding_recovery_worker.py'],'task_ids':[tid],
                  'region':'US-TX-3','region_candidates':['US-TX-3','US-GA-2','CA-MTL-3'],
                  'estimated_seconds':3600,'maximum_seconds':9000,
                  'worker_fields':{**common,'saved_prefix':saved,'saved_prefix_sha256':sha(ROOT/saved),'cohort_file':cohort_file}}]
        cohort=[tid];cohort_files=[cohort_file]
    elif mode=='train':
        stage='exploratory_training';schedule_path='configs/coding_pilot_v1/recovery_20260917/training_schedule_v1.json';add(schedule_path)
        manifest=json.loads((ROOT/CORPUS/'corpus_manifest.json').read_text())
        for rel in manifest['files']:
            if not rel.endswith('.pt'):add(rel)
        for rel in [CORPUS+'/corpus_manifest.json',CORPUS+'/initialization_identity.json',CORPUS+'/initialization_complete.json',CORPUS+'/mapper_initialization.pt']:add(rel)
        for split in ['training','validation']:add('data/coding_pilot_v1/visible/'+split+'.json')
        proof('partial_corpus',[CORPUS+'/corpus_manifest.json',CORPUS+'/initialization_complete.json',CORPUS+'/mapper_initialization.pt',schedule_path],training_count=104,validation_count=21)
        workers=[{'worker_id':'train','command':['scripts/coding_recovery_train.py'],'task_ids':[],
                  'region':'US-GA-2','region_candidates':['US-GA-2','CA-MTL-3','US-TX-3'],
                  'estimated_seconds':7200,'maximum_seconds':10800,
                  'worker_fields':{**common,'corpus_manifest':CORPUS+'/corpus_manifest.json','initialization':CORPUS+'/mapper_initialization.pt',
                    'initialization_sha256':sha(ROOT/CORPUS/'mapper_initialization.pt'),'schedule_path':schedule_path,'schedule_sha256':sha(ROOT/schedule_path)}}]
        cohort=[];cohort_files=[]
    elif mode.startswith('dev:'):
        stage='exploratory_development';trainroot=mode.split(':',1)[1]
        selection_path=trainroot+'/selection.json';selection=json.loads((ROOT/selection_path).read_text())
        if selection['development_used_for_selection'] or selection['confirmation_used_for_selection']:raise ValueError('Checkpoint must be selected before outcomes')
        for rel in [selection_path,trainroot+'/training_identity.json',trainroot+'/complete.json',trainroot+'/validation_curve.json',selection['checkpoint']['path'],CORPUS+'/mapper_initialization.pt',CORPUS+'/corpus_manifest.json']:add(rel)
        if sha(ROOT/selection['checkpoint']['path'])!=selection['checkpoint']['sha256']:raise ValueError('Selected checkpoint changed')
        proof('exploratory_checkpoint',[selection_path,trainroot+'/complete.json',selection['checkpoint']['path']],validation_only=True)
        cohort_file='data/coding_pilot_v1/visible/development.json';add(cohort_file);add('data/coding_pilot_v1/private/development.json')
        cohort=[r['task_id'] for r in json.loads((ROOT/cohort_file).read_text())];cohort_files=[cohort_file];workers=[]
        if recovery_parent:
            from gearshift.coding_development_recovery import verify_coverage
            parentroot='evidence/coding_pilot_v1/control/parallel/'+recovery_parent
            parent=json.loads((ROOT/parentroot/'plan.json').read_text())
            done=json.loads((ROOT/parentroot/'complete.json').read_text())
            succeeded={w['worker_id'] for w in done['workers'] if w['passed']}
            failed=[w for w in parent['workers'] if w['worker_id'] not in succeeded]
            if len(failed)!=1:raise ValueError('Exactly one capacity-missing shard required')
            missing=failed[0];remaining=missing['task_ids']
            roots=[f'results/coding_pilot_v1/{recovery_parent}/{w["worker_id"]}' for w in parent['workers'] if w['worker_id'] in succeeded]
            ledger_path='evidence/coding_pilot_v1/recovery_20260917/'+run_id+'_parent_ledger.json'
            bind(ROOT/ledger_path,json.loads((ROOT/'evidence/coding_pilot_v1/control/ledger.json').read_text()))
            coverage={'dataset_path':cohort_file,'parent_plan':parentroot+'/plan.json','parent_completion':parentroot+'/complete.json',
                'failure':parentroot+'/'+missing['worker_id']+'/error.json','failed_worker_id':missing['worker_id'],
                'ledger_snapshot':ledger_path,'selection':selection_path,'preserved_worker_roots':roots,
                'preserved_task_ids':sorted(t for w in parent['workers'] if w['worker_id'] in succeeded for t in w['task_ids'])}
            verify_coverage(ROOT,coverage,remaining)
            proofpaths=[coverage[k] for k in ['dataset_path','parent_plan','parent_completion','failure','ledger_snapshot','selection']]
            for rel in roots:
                proofpaths += [str(p.relative_to(ROOT)) for p in (ROOT/rel).rglob('*.json')]
            proof('recovery_coverage',proofpaths,**coverage)
            subset_path='data/coding_pilot_v1/visible/'+run_id+'_remaining.json'
            tasks={r['task_id']:r for r in json.loads((ROOT/cohort_file).read_text())}
            bind(ROOT/subset_path,[tasks[t] for t in remaining]);add(subset_path)
            cohort=remaining;cohort_files=[subset_path]
        for i,ids in enumerate(task_shards(cohort,1 if recovery_parent else 4)):
            regions=[['US-GA-2','CA-MTL-3','US-TX-3'],['CA-MTL-3','US-GA-2','EU-FR-1'],['US-GA-2','EU-FR-1','CA-MTL-4'],['CA-MTL-4','CA-MTL-3','US-GA-2']][i]
            workers.append({'worker_id':f'dev{i+1:02d}','command':['scripts/coding_recovery_eval.py'],'task_ids':ids,
                'region':regions[0],'region_candidates':regions,'estimated_seconds':3600,'maximum_seconds':7200,
                'worker_fields':{**common,'selection':selection_path,'selection_sha256':sha(ROOT/selection_path),'initialization':CORPUS+'/mapper_initialization.pt'}})
        if recovery_parent:
            # Preserve command, sampler/checkpoint fields and per-worker bounds.
            workers=[{**missing,'region':'US-GA-2','region_candidates':['US-GA-2','CA-MTL-3','EU-FR-1']}]
    else:raise ValueError('Mode not implemented')
    plan={'schema':1,'run_id':run_id,'stage':stage,'approval_sha256':sha(ROOT/APPROVAL),'image_digest':old['image_digest'],
          'gates':gates,'task_ids':cohort,'cohort_files':cohort_files,'workers':workers,'retained_storage':refs,
          'preserve_original_measurements':True,'scientific_scope_unchanged':False,'scope_amendment':'recovery_20260917_partial_v1',
          'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
          'forecast':{'method':'Engineering reservation only, not a completion promise','maximum_gpu_hours':sum(w['maximum_seconds'] for w in workers)/3600}}
    dest=ROOT/'configs/coding_pilot_v1/recovery_20260917'/(run_id+'.json');bind(dest,plan)
    identity=validate_plan(plan,ROOT,sha(ROOT/APPROVAL));print(json.dumps({'path':str(dest),'stage_identity':identity,'files':len(files)}))
    return dest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('run_id');p.add_argument('--recovery-parent');a=p.parse_args();build(a.mode,a.run_id,a.recovery_parent)
