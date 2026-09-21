#!/usr/bin/env python3
"""Second-training-seed scoring; frozen primary and scorer sources stay unchanged."""
import argparse
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gearshift.coding_control import bind,digest,sha,write
from scripts import coding_confirmation_score as base
from scripts import coding_confirmation_secondary_generate as generation

SCORING='secondary/scoring'
CONDITIONS=['FIXED_M','ROTATING_M','FIXED_H','ROTATING_H']
TRAINING_SEED=20260919
IMPLEMENTATION={'scripts/coding_confirmation_secondary_score.py','scripts/coding_confirmation_secondary_report.py'}
read=base.read
scoped=base.scoped
output_path=base.output_path


def public_context(repo,generation_plan_path,generation_plan_sha256):
    c=generation.public_context(repo,generation_plan_path,generation_plan_sha256,verify_weights=False)
    saved=read(scoped(c['top'],'secondary/generation_closure.json'))
    actual=generation.generation_closure(c,seal=False)
    if actual is None or saved!=actual or saved.get('all_secondary_generation_complete') is not True:raise ValueError('Secondary generation is incomplete or its seal differs')
    if saved.get('closure_sha256')!=digest({k:v for k,v in saved.items() if k!='closure_sha256'}):raise ValueError('Secondary closure identity differs')
    d=c['declaration'];sd=c['secondary_declaration']
    if sd['training_seed']!=TRAINING_SEED or sd['conditions']!=CONDITIONS or sd['task_ids']!=d['task_ids'] or sd['answer_count']!=12*len(d['task_ids']):raise ValueError('Secondary seed or population differs')
    if sd.get('replication_analysis')!=generation.analysis_binding(d):raise ValueError('Secondary analysis rule differs from the frozen addendum')
    if not IMPLEMENTATION<=sd['implementation'].keys():raise ValueError('Secondary scoring/report sources are not frozen')
    for name in IMPLEMENTATION:
        if sha(scoped(repo,name))!=sd['implementation'][name] or sha(ROOT/name)!=sd['implementation'][name]:raise ValueError('Loaded secondary scoring/report source differs')
    policy,identity,private_identity=base.declared_scorer(repo,d)
    c.update(dispatch=c['plan'],dispatch_path=generation_plan_path,dispatch_sha256=generation_plan_sha256,
        closure=saved,closure_file_sha256=sha(c['top']/'secondary/generation_closure.json'),
        policy=policy,scorer_identity=identity,private_identity=private_identity)
    records(c)  # Exact public draw coverage is established before private access.
    return c


def private_after_closure(c,private_path):
    if Path(private_path).resolve().is_relative_to((c['top']/'secondary').resolve()):raise ValueError('Private tests cannot be secondary public artifacts')
    return base.private_after_closure(c,private_path)


def records(c):
    result=[]
    for item in c['closure']['answers']:
        contract=item['contract'];raw=read(scoped(c['top'],item['path']))
        if contract['cohort']!='secondary' or contract['declaration_sha256']!=c['declaration_sha256']:raise ValueError('Secondary answer cohort or declaration differs')
        if sha(c['top']/item['path'])!=item['answer_sha256'] or item['contract_sha256']!=digest(contract):raise ValueError('Closed secondary answer hash differs')
        result.append({'record_id':digest(contract),'contract':contract,'contract_sha256':digest(contract),'answer_path':item['path'],
            'answer_sha256':item['answer_sha256'],'code_sha256':base.code_hash(base.extract(raw['answer_text'])),
            **{k:contract[k] for k in ('task_id','condition','seed_index','answer_seed')}})
    wanted=[(t,cnd,s['seed_index'],s['answer_seed']) for t in c['declaration']['task_ids'] for cnd in CONDITIONS for s in c['seeds']['tasks'][t]['answers']]
    if [(r['task_id'],r['condition'],r['seed_index'],r['answer_seed']) for r in result]!=wanted or len({r['record_id'] for r in result})!=len(wanted):raise ValueError('Secondary exact task/condition/seed population differs')
    return result


def plan_value(c,spec_hashes):
    d=c['declaration'];closure=c['closure']
    return {'schema_version':1,'experiment_id':d['experiment_id'],'cohort':'secondary','training_seed':TRAINING_SEED,
        'result_root':c['dispatch']['result_root'],'declaration_path':c['dispatch']['declaration_path'],'declaration_sha256':c['declaration_sha256'],
        'secondary_declaration_path':c['dispatch']['secondary_declaration_path'],'secondary_declaration_sha256':c['secondary_declaration_sha256'],
        'generation_plan_path':c['dispatch_path'],'generation_plan_sha256':c['dispatch_sha256'],
        'generation_closure_path':'secondary/generation_closure.json','generation_closure_identity_sha256':closure['closure_sha256'],
        'generation_closure_file_sha256':c['closure_file_sha256'],
        'primary_generation_closure_identity_sha256':closure['primary_generation_closure_sha256'],
        'primary_generation_closure_file_sha256':closure['primary_generation_closure_file_sha256'],
        'scorer_identity':c['scorer_identity'],'scorer_identity_sha256':digest(c['scorer_identity']),
        'policy':c['policy'],'policy_identity_sha256':digest(c['policy']),'private_test_identity':c['private_identity'],
        'selected_private_spec_sha256':spec_hashes,'implementation_hashes':c['secondary_declaration']['implementation'],
        'checkpoints':c['secondary_checkpoints'],'task_ids':d['task_ids'],'conditions':CONDITIONS,'expected_answers':12*len(d['task_ids']),
        'records':records(c),'manifest_path':SCORING+'/scored_answer_manifest.json',
        'hidden_tests_loaded_after_primary_and_secondary_generation':True,
        'retry_policy':base.RETRY_POLICY,'interrupted_transaction_policy':base.INTERRUPTED_POLICY}


def prepare(repo,generation_plan_path,generation_plan_sha256,private_path):
    c=public_context(repo,generation_plan_path,generation_plan_sha256)
    tests=private_after_closure(c,private_path)
    plan=plan_value(c,{t:digest(tests[t]) for t in c['declaration']['task_ids']})
    path=output_path(c['top'],SCORING+'/scoring_plan.json');bind(path,plan);return plan,path


def validate_plan(repo,plan_path):
    path=scoped(repo,plan_path);p=read(path)
    c=public_context(repo,p['generation_plan_path'],p['generation_plan_sha256'])
    spec_hashes=p['selected_private_spec_sha256']
    if set(spec_hashes)!=set(c['declaration']['task_ids']) or any(not isinstance(v,str) or not base.re.fullmatch('[0-9a-f]{64}',v) for v in spec_hashes.values()):raise ValueError('Private task-spec identity coverage differs')
    if path!=output_path(c['top'],SCORING+'/scoring_plan.json') or p!=plan_value(c,spec_hashes):raise ValueError('Frozen secondary scoring plan changed')
    c.update(plan=p,plan_path=path,plan_sha256=sha(path),plan_identity_sha256=digest(p));return c


def score_binding(c,r):
    p=c['plan']
    return {'scoring_plan_sha256':c['plan_sha256'],'scoring_plan_identity_sha256':c['plan_identity_sha256'],
        **{k:p[k] for k in ('declaration_sha256','secondary_declaration_sha256','training_seed','generation_closure_identity_sha256',
            'generation_closure_file_sha256','primary_generation_closure_identity_sha256','primary_generation_closure_file_sha256','scorer_identity_sha256')},
        **{k:r[k] for k in ('record_id','answer_sha256','contract_sha256','code_sha256')}}


def validate_receipt(c,r,saved):
    raw=read(scoped(c['top'],r['answer_path']))
    if (saved.get('binding')!=score_binding(c,r) or saved.get('contract')!=r['contract'] or
        saved.get('code')!=base.extract(raw['answer_text']) or saved.get('hidden_tests_loaded_after_primary_and_secondary_generation') is not True or
        any(saved.get(k)!=r[k] for k in ('task_id','condition','seed_index','answer_seed','answer_path','answer_sha256'))):raise ValueError('Secondary score transaction binding differs')
    base.validate_result(saved['score_v2'],c['scorer_identity'])


def score_record(c,r,tests,cpu_id,attempt,guard):
    path=scoped(c['top'],r['answer_path'])
    if sha(path)!=r['answer_sha256']:raise ValueError('Raw secondary answer changed')
    code=base.extract(read(path)['answer_text'])
    if base.code_hash(code)!=r['code_sha256']:raise ValueError('Secondary extraction changed')
    folder=output_path(c['top'],SCORING+'/scores/'+r['record_id']);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'transaction.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);target=folder/'score.json';binding=score_binding(c,r)
        if target.exists():
            saved=read(target);validate_receipt(c,r,saved);return saved
        start=folder/'started.json'
        if start.exists():
            if read(start)['binding']!=binding:raise ValueError('Interrupted secondary transaction changed identity')
            result={'passed':None,'missing':True,'category':'infrastructure_failure','reason':'worker_interrupted_before_score_commit',
                'trusted_checker_complete':False,'executed_tests':None,'scorer_identity':c['scorer_identity']}
        else:
            guard();write(start,{'binding':binding,'attempt':attempt,'cpu_id':cpu_id,'pid':os.getpid(),'epoch':time.time()})
            spec=tests.get(r['task_id'])
            if not spec or digest(spec)!=c['plan']['selected_private_spec_sha256'][r['task_id']]:raise ValueError('Selected frozen test specification unavailable or changed')
            result=base.score(code,spec,guard=guard,policy=c['policy'],cpu_id=cpu_id)
        base.validate_result(result,c['scorer_identity'])
        if sha(path)!=r['answer_sha256']:raise ValueError('Raw secondary answer changed during scoring')
        saved={'binding':binding,'contract':r['contract'],'code':code,'score_v2':result,
            **{k:r[k] for k in ('task_id','condition','seed_index','answer_seed','answer_path','answer_sha256')},
            'attempt':attempt,'cpu_id':cpu_id,'completed_epoch':time.time(),'hidden_tests_loaded_after_primary_and_secondary_generation':True}
        write(target,saved);return saved


def build_manifest(c):
    p=c['plan'];files=[]
    for r in p['records']:
        relative=SCORING+'/scores/'+r['record_id']+'/score.json';path=scoped(c['top'],relative);saved=read(path);validate_receipt(c,r,saved)
        files.append({'path':relative,'sha256':sha(path),**{k:r[k] for k in ('answer_path','answer_sha256','task_id','condition','seed_index','answer_seed')},
            'passed':saved['score_v2']['passed'],'missing':saved['score_v2']['missing']})
    if {f.name for f in (c['top']/SCORING/'scores').glob('*') if f.is_dir()}!={r['record_id'] for r in p['records']}:raise ValueError('Undeclared secondary score transaction')
    return {'schema_version':1,'experiment_id':p['experiment_id'],'cohort':'secondary','training_seed':TRAINING_SEED,
        **{k:p[k] for k in ('declaration_sha256','secondary_declaration_sha256','generation_closure_identity_sha256','generation_closure_file_sha256',
            'primary_generation_closure_identity_sha256','primary_generation_closure_file_sha256','scorer_identity_sha256','task_ids','conditions','expected_answers')},
        'scoring_plan_path':str(c['plan_path'].relative_to(c['repo_root'])),'scoring_plan_sha256':c['plan_sha256'],
        'scoring_plan_identity_sha256':c['plan_identity_sha256'],'committed_answers':len(files),'missing_answers':sum(r['missing'] for r in files),
        'files':files,'hidden_tests_loaded_after_primary_and_secondary_generation':True,'raw_answers_unchanged':True}


def finalize(repo,plan_path):
    c=validate_plan(repo,plan_path);manifest=build_manifest(c);bind(output_path(c['top'],c['plan']['manifest_path']),manifest);return manifest


def worker(c,rs,tests,cpu,attempt,guard,status):
    signal.signal(signal.SIGTERM,base.interruption);signal.signal(signal.SIGINT,base.interruption)
    try:
        os.sched_setaffinity(0,{cpu})
        for i,r in enumerate(rs):
            score_record(c,r,tests,cpu,attempt,guard);write(status,{'completed':i+1,'expected':len(rs),'cpu_id':cpu,'epoch':time.time()})
    except BaseException as exc:
        write(status,{'failed':True,'exception_type':type(exc).__name__,'epoch':time.time()});raise SystemExit(1)


def run(repo,plan_path,private_path,lease_path,lease_sha256,preflight_path,preflight_sha256,cpu_ids):
    c=validate_plan(repo,plan_path);lease,guard=base.allocation(repo,lease_path,lease_sha256,c['top'])
    if lease['experiment_id']!=c['declaration']['experiment_id']:raise ValueError('CPU allocation belongs to another experiment')
    receipt=base.verify_preflight(c,lease,lease_sha256,scoped(repo,preflight_path),preflight_sha256,cpu_ids)
    # The exact primary execution lock also excludes simultaneous primary scoring
    # and capacity preflight on this host; the result namespaces remain separate.
    lockpath=output_path(c['top'],base.SCORING+'/execution.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);removed=base.cleanup_orphans(lease)
        tests=private_after_closure(c,private_path)
        if {t:digest(tests[t]) for t in c['plan']['task_ids']}!=c['plan']['selected_private_spec_sha256']:raise ValueError('Selected private test identities changed')
        attempt=lease['pod_id']+'_'+str(time.time_ns());root=output_path(c['top'],SCORING+'/executions/'+attempt)
        write(root/'identity.json',{'attempt':attempt,'training_seed':TRAINING_SEED,'scoring_plan_sha256':c['plan_sha256'],
            'lease_path':lease_path,'lease_sha256':lease_sha256,'preflight_path':preflight_path,'preflight_sha256':preflight_sha256,
            'cpu_environment':receipt['environment'],'orphan_cleanup_pids':removed})
        processes=[];handlers={sig:signal.signal(sig,base.interruption) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            ctx=multiprocessing.get_context('fork')
            for i,cpu in enumerate(cpu_ids):
                p=ctx.Process(target=worker,args=(c,c['plan']['records'][i::len(cpu_ids)],tests,cpu,attempt,guard,root/f'worker_{i:02d}.json'));p.start();processes.append(p)
            while any(p.is_alive() for p in processes):guard();time.sleep(1)
            if any(p.exitcode for p in processes):raise RuntimeError('Secondary score worker failed; started transactions remain missing on recovery')
        except BaseException as exc:
            write(root/'failed.json',{'exception_type':type(exc).__name__,'epoch':time.time()});raise
        finally:
            for p in processes:
                if p.is_alive():p.terminate()
            for p in processes:
                p.join(timeout=5)
                if p.is_alive():p.kill();p.join()
            try:write(root/'cleanup.json',{'orphan_cleanup_pids':base.cleanup_orphans(lease),'epoch':time.time()})
            finally:
                for sig,handler in handlers.items():signal.signal(sig,handler)
        result=finalize(repo,plan_path);write(root/'complete.json',{'committed_answers':result['committed_answers'],'missing_answers':result['missing_answers'],'epoch':time.time()});return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','finalize']);p.add_argument('--repo',default=str(ROOT))
    for name in ('generation-plan','generation-plan-sha256','plan','private-tests','lease','lease-sha256','preflight','preflight-sha256','cpu-ids'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.action=='prepare':value,path=prepare(a.repo,a.generation_plan,a.generation_plan_sha256,a.private_tests);result={'scoring_plan_path':str(path),'expected_answers':value['expected_answers']}
    elif a.action=='finalize':result=finalize(a.repo,a.plan)
    else:result=run(a.repo,a.plan,a.private_tests,a.lease,a.lease_sha256,a.preflight,a.preflight_sha256,[int(v) for v in a.cpu_ids.split(',')])
    print(json.dumps({'committed_answers':result.get('committed_answers'),'missing_answers':result.get('missing_answers'),**({k:v for k,v in result.items() if k in ('scoring_plan_path','expected_answers')} if a.action=='prepare' else {})}))
if __name__=='__main__':main()
