#!/usr/bin/env python3
"""Uniform immutable-output rescoring and predeclared diagnostic repetitions.

No generation, extraction changes, program repair, or best-run selection. A
crashed score transaction is missing coverage; it is never silently rerun.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import digest,sha,write
from gearshift.coding_sandbox import extract
from gearshift.coding_sandbox_v2 import score,scorer_identity,validate_policy


def read(path):return json.loads(Path(path).read_text())
def secure_path(root,path):
    result=(Path(root)/path).resolve()
    if not result.is_relative_to(Path(root).resolve()):raise ValueError('Manifest path leaves evidence root')
    return result

def original_records(evaluation_root,expected=1512):
    root=Path(evaluation_root);manifest=read(root/'scored_answer_manifest.json')
    if manifest.get('hidden_tests_loaded_after_all_generation') is not True:raise ValueError('Original generation closure is unverified')
    files=manifest['files']
    if len(files)!=expected or len({r['answer_path'] for r in files})!=expected:raise ValueError('Expected exactly one original receipt for every saved answer')
    records=[]
    for item in sorted(files,key=lambda r:r['answer_path']):
        ap=secure_path(root,item['answer_path']);sp=secure_path(root,item['path'])
        if sha(ap)!=item['answer_sha256'] or sha(sp)!=item['sha256']:raise ValueError('Original answer/score changed')
        raw=read(ap);old=read(sp);code=extract(raw['answer_text'])
        if old['code']!=code or old['answer_sha256']!=item['answer_sha256']:raise ValueError('Original code extraction identity differs')
        records.append({**item,'record_id':hashlib.sha256(item['answer_path'].encode()).hexdigest(),
            'task_id':raw['task_id'],'condition':raw['condition'],'form':raw['form'],'step':raw['step'],
            'seed_index':raw['seed_index'],'answer_seed':raw['answer_seed'],
            'code_sha256':hashlib.sha256(code.encode()).hexdigest(),'code':code,'old_score':old['score']})
    return manifest,records


def freeze_plan(evaluation_root,private_path,policy_path,output,expected=1512):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);target=out/'rescore_plan.json'
    p=validate_policy(read(policy_path))
    if p.get('policy_status')!='frozen':raise ValueError('Resource policy must be calibrated and frozen first')
    manifest,records=original_records(evaluation_root,expected)
    # Hashing/loading private data occurs only on the dedicated CPU scorer.
    private_sha=sha(private_path);private=read(private_path)
    for r in records:
        spec=private.get(r['task_id'])
        if spec is None:raise ValueError('Missing original task tests')
        if r['old_score'].get('total_tests',len(spec['tests']))!=len(spec['tests']):raise ValueError('Original test count differs')
        for receipt in r['old_score'].get('receipts',[]):
            test=spec['tests'][receipt['test_index']]
            if receipt.get('input_sha256')!=hashlib.sha256(test['input'].encode()).hexdigest() or receipt.get('expected_sha256')!=hashlib.sha256(test['output'].encode()).hexdigest():raise ValueError('Original scored test bytes differ')
    diagnostic=[]
    for r in records:
        timing=(r['task_id']=='atcoder/abc339_d' and r['condition']=='D') or (r['task_id']=='atcoder/abc335_c' and r['condition']=='P')
        output=r['task_id']=='atcoder/abc344_e' and any('File too large' in t.get('stderr','') for t in r['old_score'].get('receipts',[]))
        if timing or output:diagnostic.append({'record_id':r['record_id'],'reason':'fixed_timing_audit_subset' if timing else 'nine_audited_output_ceiling_failures'})
    plan={'schema_version':1,'kind':'uniform_saved_v2_rescoring','expected_answers':expected,
        'original_manifest_sha256':sha(Path(evaluation_root)/'scored_answer_manifest.json'),
        'original_generation_closure_sha256':manifest['generation_closure_sha256'],
        'private_tests_sha256':private_sha,'task_spec_sha256':{tid:digest(private[tid]) for tid in sorted({r['task_id'] for r in records})},'original_executed_test_hashes_verified':True,'policy':p,'scorer_identity':scorer_identity(p),
        'pipeline_sources':{name:sha(Path(__file__).resolve().parents[1]/name) for name in ('scripts/coding_scorer_repair_rescore.py','scripts/coding_scorer_repair_report.py','scripts/coding_scorer_repair_calibrate.py','scripts/coding_scorer_repair_execute.py')},
        'records':[{k:v for k,v in r.items() if k not in ('code','old_score')} for r in records],
        'diagnostic_subset':diagnostic,'diagnostic_repetitions':3,
        'diagnostic_policy':'All three fixed repeats retained; no favorable repeat selected. Uniform rescore is separate and uses one score transaction.',
        'interrupted_transaction_policy':'Committed start without committed score becomes missing worker_interrupted_before_score_commit. No automatic candidate rerun.',
        'retry_policy':'Only verified sandbox launch/confinement/capture failures get one retry. Candidate CPU/wall/resource/algorithm failures never retry.',
        'analysis':{'task_clusters':21,'seeds_per_task_condition':3,'joint_bootstrap_resamples':10000,'bootstrap_seed':20260918,'interval':'paired task-cluster percentile 95%; exploratory unadjusted'}}
    if target.exists():
        if read(target)!=plan:raise ValueError('Frozen rescore plan differs')
    else:write(target,plan)
    return plan


def validate_frozen(output,evaluation_root,private_path):
    out=Path(output);plan=read(out/'rescore_plan.json');p=plan['policy']
    if scorer_identity(p)!=plan['scorer_identity']:raise ValueError('Frozen scorer source or policy drift')
    for name,expected_sha in plan['pipeline_sources'].items():
        if sha(Path(__file__).resolve().parents[1]/name)!=expected_sha:raise ValueError('Frozen scoring pipeline source drift')
    if sha(Path(evaluation_root)/'scored_answer_manifest.json')!=plan['original_manifest_sha256']:raise ValueError('Original score manifest drift')
    if sha(private_path)!=plan['private_tests_sha256']:raise ValueError('Frozen private-test hash differs')
    return plan


def score_record(record,plan,root,out,private,cpu_id,diagnostic_rep=None):
    rawpath=secure_path(root,record['answer_path']);oldpath=secure_path(root,record['path'])
    if sha(rawpath)!=record['answer_sha256'] or sha(oldpath)!=record['sha256']:raise ValueError('Immutable original artifact drift')
    raw=read(rawpath);old=read(oldpath);code=extract(raw['answer_text'])
    if code!=old['code'] or hashlib.sha256(code.encode()).hexdigest()!=record['code_sha256']:raise ValueError('Candidate extraction drift')
    suffix='' if diagnostic_rep is None else f'_repeat_{diagnostic_rep}'
    directory=Path(out)/('scores' if diagnostic_rep is None else 'diagnostics')/(record['record_id']+suffix)
    directory.mkdir(parents=True,exist_ok=True);target=directory/'score_v2.json'
    binding={'record_id':record['record_id'],'plan_sha256':digest(plan),'answer_sha256':record['answer_sha256'],
        'original_score_sha256':record['sha256'],'code_sha256':record['code_sha256'],'diagnostic_repetition':diagnostic_rep,
        'scorer_identity':plan['scorer_identity']}
    with (directory/'transaction.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.exists():
            saved=read(target)
            if saved['binding']!=binding:raise ValueError('Completed scoring identity collision')
            return saved
        started=directory/'started.json'
        if started.exists():
            previous=read(started)
            if previous['binding']!=binding:raise ValueError('Interrupted scoring identity collision')
            scored={'passed':None,'missing':True,'category':'infrastructure_failure','reason':'worker_interrupted_before_score_commit',
                'scorer_identity':plan['scorer_identity'],'trusted_checker_complete':False,'executed_tests':None}
        else:
            write(started,{'binding':binding,'hostname':platform.node(),'pid':os.getpid(),'cpu_id':cpu_id,'epoch':time.time()})
            spec=private.get(record['task_id'])
            if spec is None:scored={'passed':None,'missing':True,'category':'infrastructure_failure','reason':'missing_task_tests','scorer_identity':plan['scorer_identity']}
            else:scored=score(code,spec,policy=plan['policy'],cpu_id=cpu_id)
        result={'binding':binding,'task_id':record['task_id'],'condition':record['condition'],'form':record['form'],
            'step':record['step'],'seed_index':record['seed_index'],'answer_seed':record['answer_seed'],
            'original_answer_path':record['answer_path'],'original_score_path':record['path'],
            'original_score':old['score'],'score_v2':scored,'finished_epoch':time.time(),
            'candidate_program_unchanged':True,'original_artifacts_unchanged':True}
        write(target,result);return result


def run_shard(evaluation_root,private_path,output,shard_index,shard_count,cpu_id,diagnostics=False):
    plan=validate_frozen(output,evaluation_root,private_path)
    allowed=plan['policy']['physical_cpu_ids']
    if not 0<=shard_index<shard_count<=plan['policy']['maximum_concurrent_candidates'] or cpu_id not in allowed:raise ValueError('Invalid frozen CPU allocation')
    if cpu_id!=allowed[shard_index]:raise ValueError('Shard must use its preassigned physical core')
    os.sched_setaffinity(0,{cpu_id});private=read(private_path)
    ids={r['record_id'] for r in plan['diagnostic_subset']}
    records=[r for r in plan['records'] if not diagnostics or r['record_id'] in ids]
    owned=records[shard_index::shard_count];count=0
    for record in owned:
        for rep in range(plan['diagnostic_repetitions']) if diagnostics else [None]:
            score_record(record,plan,evaluation_root,output,private,cpu_id,rep);count+=1
            write(Path(output)/('diagnostic_workers' if diagnostics else 'workers')/f'shard_{shard_index:02d}.json',
                {'shard_index':shard_index,'shard_count':shard_count,'cpu_id':cpu_id,'completed':count,'expected':len(owned)*(3 if diagnostics else 1),'epoch':time.time(),'plan_sha256':digest(plan)})
    print(json.dumps({'shard':shard_index,'completed':count,'diagnostics':diagnostics}),flush=True)


def finalize(output):
    out=Path(output);plan=read(out/'rescore_plan.json');files=[]
    for record in plan['records']:
        path=out/'scores'/record['record_id']/'score_v2.json'
        if not path.exists():raise ValueError('Rescore coverage incomplete: '+record['record_id'])
        value=read(path)
        if value['binding']['plan_sha256']!=digest(plan) or value['binding']['record_id']!=record['record_id']:raise ValueError('Rescore binding differs')
        files.append({'path':str(path.relative_to(out)),'sha256':sha(path),'record_id':record['record_id'],'passed':value['score_v2']['passed'],'missing':value['score_v2'].get('missing',False)})
    manifest={'schema_version':1,'expected':plan['expected_answers'],'committed':len(files),'missing':sum(r['missing'] for r in files),
        'plan_sha256':digest(plan),'scorer_identity':plan['scorer_identity'],'files':files,'uniform_policy':True,'candidate_programs_unchanged':True}
    target=out/'rescored_answer_manifest.json'
    if target.exists() and read(target)!=manifest:raise ValueError('Completed rescore manifest differs')
    write(target,manifest);return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','diagnostics','finalize']);p.add_argument('--evaluation-root');p.add_argument('--private-tests');p.add_argument('--policy');p.add_argument('--output',required=True);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);p.add_argument('--cpu-id',type=int);a=p.parse_args()
    if a.action=='prepare':r=freeze_plan(a.evaluation_root,a.private_tests,a.policy,a.output);print(json.dumps({'prepared':len(r['records']),'diagnostic_records':len(r['diagnostic_subset']),'plan_sha256':digest(r)}))
    elif a.action=='finalize':r=finalize(a.output);print(json.dumps({'committed':r['committed'],'missing':r['missing']}))
    else:run_shard(a.evaluation_root,a.private_tests,a.output,a.shard_index,a.shard_count,a.cpu_id,a.action=='diagnostics')
if __name__=='__main__':main()
