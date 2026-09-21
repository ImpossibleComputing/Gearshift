#!/usr/bin/env python3
"""Frozen repaired scorer for the sealed 12 x 14 x 3 sparse development screen.

Public-reference preflight may precede generation. Private test bytes are opened
only after all 504 public generation transactions have been reverified and sealed.
No candidate repair, new extraction policy, outcome retry, or provider operation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, digest, sha, write
from scripts import coding_confirmation_score as base

SCORING = 'screen/scoring'
CLOSURE = 'screen/generation_closure.json'
PRIVATE_SHA = '51843183c4b58f74bf995c656fda0f7e788f4913c3d9dc6d7947d22efe1089ae'
CONDITIONS = ('D','H','P','N_5','M_5','R_5','N_10','M_10','R_10','N_25','M_25','R_25','R_random','R_recent')
IMPLEMENTATION = ('scripts/sparse_repair_score.py','scripts/coding_confirmation_score.py',
    'scripts/coding_scorer_repair_calibrate.py','gearshift/coding_control.py',
    'gearshift/coding_confirmation_lease.py','gearshift/coding_sandbox.py',
    'gearshift/coding_sandbox_v2.py','scripts/coding_sandbox_child_v2.py')
read = base.read
path_inside = base.output_path


def declaration_context(repo, declaration_path, declaration_sha):
    repo = Path(repo).resolve()
    path = path_inside(repo, declaration_path)
    if sha(path) != declaration_sha:
        raise ValueError('Frozen sparse declaration changed')
    d = read(path)
    tids = [r['task_id'] for r in d['screen_tasks']]
    population = [r['task_id'] for r in d['population']]
    if (d['experiment_id'] != 'sparse_repair_01' or len(tids) != 12 or len(set(tids)) != 12
        or len(population) != 40 or len(set(population)) != 40 or not set(tids) <= set(population)
        or [c['name'] for c in d['conditions']] != list(CONDITIONS) or d['screen_answer_records'] != 504
        or d['inputs']['private_development_input_sha256'] != PRIVATE_SHA):
        raise ValueError('Sparse cohort, conditions, or private identity changed')
    frozen = d['scorer']
    for field in ('frozen_policy','identity_receipt','calibration_receipt'):
        item = frozen[field]
        if sha(path_inside(repo, item['path'])) != item['sha256']:
            raise ValueError('Frozen scorer reference changed: ' + field)
    policy = base.validate_policy(read(path_inside(repo, frozen['frozen_policy']['path'])))
    identity = base.scorer_identity(policy)
    if (policy.get('policy_status') != 'frozen' or policy['cpu_seconds'] != 12 or policy['wall_seconds'] != 46
        or policy['maximum_concurrent_candidates'] != 8 or frozen['version'] != identity['scorer_version']
        or read(path_inside(repo, frozen['identity_receipt']['path'])) != identity):
        raise ValueError('Exact repaired scorer/policy differs')
    for item in frozen['core_implementation']:
        if identity['files'].get(item['path']) != item['sha256'] or sha(path_inside(repo,item['path'])) != item['sha256']:
            raise ValueError('Core scorer source differs from frozen bytes')
    return {'repo_root':repo,'declaration':d,'declaration_path':str(declaration_path),
            'declaration_sha256':declaration_sha,'policy':policy,'scorer_identity':identity,
            'task_ids':tids,'private_inventory':population}


def generation_closure(c, result_root):
    """Compute closure from public bytes only, checking every descendant receipt."""
    top = path_inside(c['repo_root'], result_root)
    d = c['declaration']; answers=[]; task_receipts=[]; implementations=set(); paths=set()
    for task in d['screen_tasks']:
        tid = task['task_id']; folder = 'screen/tasks/' + tid.replace('/','__')
        tp = path_inside(top, folder+'/task_complete.json'); tr = read(tp)
        if (tr['task_id'] != tid or tr['declaration_sha256'] != c['declaration_sha256'] or tr['completed_records'] != 42
            or tr['expected_records'] != 42 or tr.get('backing_cache_fingerprints_unchanged') is not True):
            raise ValueError('Incomplete or mismatched task generation receipt')
        template_path=path_inside(top,folder+'/prompt_only_template.json')
        template=read(template_path)
        if (sha(template_path)!=tr['prompt_only_template_sha256'] or template.get('task_id')!=tid
            or template.get('declaration_sha256')!=c['declaration_sha256'] or template.get('enable_thinking') is not False
            or '<think>\n\n</think>' not in template.get('rendered_template','')
            or template.get('prefix_ids',[])+template.get('bridge_ids',[])!=template.get('prompt_ids')):
            raise ValueError('Pinned P non-thinking template changed')
        implementations.add(tr['implementation_sha256'])
        listed = {r['path']:r['sha256'] for r in tr['files']}
        expected_paths = {f'{name}/seed_{seed["seed_index"]}/complete.json' for name in CONDITIONS for seed in d['seeds'][tid]}
        if len(tr['files']) != 42 or set(listed) != expected_paths:
            raise ValueError('Task draw closure differs from exactly 14 x 3')
        seeds = d['seeds'][tid]
        if len(seeds) != 3 or [s['seed_index'] for s in seeds] != [0,1,2]:
            raise ValueError('Three predeclared answer draws required')
        for condition in CONDITIONS:
            for seed in seeds:
                rel = f'{folder}/{condition}/seed_{seed["seed_index"]}'
                done_path = path_inside(top, rel+'/complete.json'); done = read(done_path)
                if sha(done_path) != listed[f'{condition}/seed_{seed["seed_index"]}/complete.json']:
                    raise ValueError('Task-bound draw receipt changed')
                contract = {'experiment_id':'sparse_repair_01','cohort':'development_screen','task_id':tid,
                    'condition':condition,**seed,'declaration_sha256':c['declaration_sha256'],
                    'history_sha256':tr['prompt_only_template_sha256'] if condition=='P' else task['history_sha256'],
                    'original_source_history_sha256':task['history_sha256'],'mapper_sha256':d['mapper']['sha256'],
                    'implementation_sha256':tr['implementation_sha256'],'teacher_answer_prefix_supplied':False}
                if done['identity'] != contract:
                    raise ValueError('Raw draw identity differs from frozen cohort')
                required = {'answer.json','working_set.json','sampler/identity.json','sampler/answer_record.json',
                            'sampler/resume.json','sampler/complete.json','sampler/completion_timing.json'}
                if not required <= done['files'].keys():
                    raise ValueError('Draw missing immutable generation transaction members')
                for name, expected in done['files'].items():
                    if sha(path_inside(top,rel+'/'+name)) != expected:
                        raise ValueError('Raw generation artifact changed: '+rel+'/'+name)
                raw_path=rel+'/answer.json'; raw=read(path_inside(top,raw_path))
                if any(raw.get(k) != v for k,v in contract.items()) or not isinstance(raw.get('answer_text'),str):
                    raise ValueError('Answer raw content/contract mismatch')
                sampler = read(path_inside(top,rel+'/sampler/answer_record.json'))
                if any(raw.get(k) != sampler.get(k) for k in ('task_id','answer_seed','answer_ids','answer_text','rng_final')):
                    raise ValueError('Materialized answer differs from committed sampler')
                resume = read(path_inside(top,rel+'/sampler/resume.json'))
                if resume.get('state') != 'complete' or resume.get('resume_sha256') != digest({k:v for k,v in resume.items() if k!='resume_sha256'}):
                    raise ValueError('Sampler transaction lacks valid completion')
                work=read(path_inside(top,rel+'/working_set.json'))
                if work.get('semantic_trajectory_union_complete') is not True:
                    raise ValueError('Working-set coverage is not complete')
                answers.append({'path':raw_path,'answer_sha256':sha(path_inside(top,raw_path)),
                                'contract':contract,'contract_sha256':digest(contract),
                                'complete_path':rel+'/complete.json','complete_sha256':sha(done_path)})
                paths.add(raw_path)
        task_receipts.append({'path':folder+'/task_complete.json','sha256':sha(tp)})
    discovered={str(p.relative_to(top)) for p in (top/'screen/tasks').glob('*/*/seed_*/answer.json')}
    if len(answers)!=504 or discovered!=paths or len(implementations)!=1:
        raise ValueError('Screen is incomplete, contains extra draws, or mixed generation implementations')
    closure={'schema':1,'experiment_id':'sparse_repair_01','cohort':'development_screen',
             'declaration_path':c['declaration_path'],'declaration_sha256':c['declaration_sha256'],
             'result_root':str(result_root),'all_screen_generation_complete':True,'expected_answers':504,
             'generation_implementation_sha256':next(iter(implementations)),
             'task_receipts':task_receipts,'answers':answers,'private_tests_loaded':False}
    closure['closure_sha256']=digest(closure)
    return closure


def seal(repo,declaration_path,declaration_sha,result_root):
    c=declaration_context(repo,declaration_path,declaration_sha)
    closure=generation_closure(c,result_root)
    bind(path_inside(c['repo_root'],str(Path(result_root)/CLOSURE)),closure)
    return closure


def public_context(repo,declaration_path,declaration_sha,result_root):
    c=declaration_context(repo,declaration_path,declaration_sha)
    c['top']=path_inside(c['repo_root'],result_root)
    saved=read(path_inside(c['top'],CLOSURE))
    actual=generation_closure(c,result_root)
    if saved!=actual:
        raise ValueError('Saved screen generation seal differs from actual immutable artifacts')
    c.update(closure=saved,closure_file_sha256=sha(path_inside(c['top'],CLOSURE)),result_root=str(result_root))
    return c


def private_after_closure(c,private_path):
    # A sealed public_context is mandatory; preflight never calls this function.
    if c.get('closure',{}).get('all_screen_generation_complete') is not True:
        raise ValueError('All generation must be sealed before private-test access')
    path=Path(private_path).resolve()
    if path.is_relative_to(c['top']):
        raise ValueError('Private tests cannot live in public experiment output')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PRIVATE_SHA:
        raise ValueError('Exact 40-development private input hash differs')
    tests=json.loads(raw)
    if not isinstance(tests,dict) or set(tests)!=set(c['private_inventory']):
        raise ValueError('Private 40-development task inventory differs')
    selected={tid:tests[tid] for tid in c['task_ids']}
    for tid,spec in selected.items():
        if not isinstance(spec,dict) or not isinstance(spec.get('tests'),list) or not spec['tests'] or 'fn_name' not in spec:
            raise ValueError('Malformed selected test specification')
        for test in spec['tests']:
            if not isinstance(test.get('input'),str) or not isinstance(test.get('output'),str):
                raise ValueError('Malformed trusted test interface')
            base.output_allowance(c['policy'],test['output'],spec['fn_name'])
    return selected


def records(c):
    out=[]
    for item in c['closure']['answers']:
        contract=item['contract'];raw=read(path_inside(c['top'],item['path']))
        out.append({'record_id':digest(contract),'contract':contract,'contract_sha256':digest(contract),
            'answer_path':item['path'],'answer_sha256':item['answer_sha256'],
            'code_sha256':base.code_hash(base.extract(raw['answer_text'])),
            **{k:contract[k] for k in ('task_id','condition','seed_index','answer_seed')}})
    return out


def plan_value(c,spec_hashes):
    for name in IMPLEMENTATION:
        if sha(path_inside(c['repo_root'],name))!=sha(ROOT/name):
            raise ValueError('Loaded scoring source differs from declared repository')
    return {'schema':1,'experiment_id':'sparse_repair_01','cohort':'development_screen',
        'result_root':c['result_root'],'declaration_path':c['declaration_path'],'declaration_sha256':c['declaration_sha256'],
        'generation_closure_identity_sha256':c['closure']['closure_sha256'],
        'generation_closure_file_sha256':c['closure_file_sha256'],
        'scorer_identity':c['scorer_identity'],'scorer_identity_sha256':digest(c['scorer_identity']),
        'policy':c['policy'],'private_development_input_sha256':PRIVATE_SHA,
        'selected_private_spec_sha256':spec_hashes,'task_ids':c['task_ids'],'conditions':list(CONDITIONS),
        'expected_answers':504,'records':records(c),
        'implementation_hashes':{p:sha(path_inside(c['repo_root'],p)) for p in IMPLEMENTATION},
        'hidden_tests_loaded_after_complete_screen_generation':True,
        'retry_policy':base.RETRY_POLICY,'interrupted_transaction_policy':base.INTERRUPTED_POLICY}


def prepare(repo,declaration_path,declaration_sha,result_root,private_path):
    c=public_context(repo,declaration_path,declaration_sha,result_root)
    tests=private_after_closure(c,private_path)
    plan=plan_value(c,{t:digest(tests[t]) for t in c['task_ids']})
    path=path_inside(c['top'],SCORING+'/scoring_plan.json');bind(path,plan)
    return plan,path


def validate_plan(repo,plan_path):
    path=path_inside(repo,plan_path);p=read(path)
    c=public_context(repo,p['declaration_path'],p['declaration_sha256'],p['result_root'])
    hashes=p['selected_private_spec_sha256']
    if set(hashes)!=set(c['task_ids']) or any(not base.re.fullmatch('[0-9a-f]{64}',v) for v in hashes.values()):
        raise ValueError('Selected private specification identity coverage differs')
    if p!=plan_value(c,hashes) or path!=path_inside(c['top'],SCORING+'/scoring_plan.json'):
        raise ValueError('Scoring plan or loaded implementation changed')
    c.update(plan=p,plan_path=path,plan_sha256=sha(path),plan_identity_sha256=digest(p))
    return c


def score_binding(c,r):
    return {'scoring_plan_sha256':c['plan_sha256'],'scoring_plan_identity_sha256':c['plan_identity_sha256'],
        **{k:c['plan'][k] for k in ('declaration_sha256','generation_closure_identity_sha256',
            'generation_closure_file_sha256','scorer_identity_sha256')},
        **{k:r[k] for k in ('record_id','answer_sha256','contract_sha256','code_sha256')}}


def validate_receipt(c,r,saved):
    raw=read(path_inside(c['top'],r['answer_path']))
    if (saved.get('binding')!=score_binding(c,r) or saved.get('contract')!=r['contract']
        or saved.get('original_answer_text')!=raw['answer_text'] or saved.get('code')!=base.extract(raw['answer_text'])
        or saved.get('hidden_tests_loaded_after_complete_screen_generation') is not True
        or any(saved.get(k)!=r[k] for k in ('task_id','condition','seed_index','answer_seed','answer_path','answer_sha256'))):
        raise ValueError('Score receipt/raw candidate binding differs')
    base.validate_result(saved['score_v2'],c['scorer_identity'])


def score_record(c,r,tests,cpu,attempt,guard):
    rawpath=path_inside(c['top'],r['answer_path']);raw=read(rawpath);code=base.extract(raw['answer_text'])
    if sha(rawpath)!=r['answer_sha256'] or base.code_hash(code)!=r['code_sha256']:
        raise ValueError('Raw candidate or unchanged extraction changed')
    folder=path_inside(c['top'],SCORING+'/scores/'+r['record_id']);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'transaction.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);target=folder/'score.json';binding=score_binding(c,r)
        if target.exists():
            saved=read(target);validate_receipt(c,r,saved);return saved
        start=folder/'started.json'
        if start.exists():
            if read(start)['binding']!=binding:
                raise ValueError('Interrupted score identity changed')
            result={'passed':None,'missing':True,'category':'infrastructure_failure',
                    'reason':'worker_interrupted_before_score_commit','executed_tests':None,
                    'trusted_checker_complete':False,'scorer_identity':c['scorer_identity']}
        else:
            guard()
            spec=tests.get(r['task_id'])
            if not spec or digest(spec)!=c['plan']['selected_private_spec_sha256'][r['task_id']]:
                raise ValueError('Frozen selected test specification changed')
            write(start,{'binding':binding,'attempt':attempt,'cpu_id':cpu,'pid':os.getpid(),'epoch':time.time()})
            result=base.score(code,spec,guard=guard,policy=c['policy'],cpu_id=cpu)
        base.validate_result(result,c['scorer_identity'])
        if sha(rawpath)!=r['answer_sha256']:
            raise ValueError('Raw answer changed during scoring')
        saved={'binding':binding,'contract':r['contract'],'original_answer_text':raw['answer_text'],'code':code,
            'score_v2':result,**{k:r[k] for k in ('task_id','condition','seed_index','answer_seed','answer_path','answer_sha256')},
            'attempt':attempt,'cpu_id':cpu,'completed_epoch':time.time(),
            'hidden_tests_loaded_after_complete_screen_generation':True}
        write(target,saved);return saved


def finalize(repo,plan_path):
    c=validate_plan(repo,plan_path);files=[]
    for r in c['plan']['records']:
        relative=SCORING+'/scores/'+r['record_id']+'/score.json';path=path_inside(c['top'],relative)
        saved=read(path);validate_receipt(c,r,saved)
        files.append({'path':relative,'sha256':sha(path),**{k:r[k] for k in ('task_id','condition','seed_index','answer_seed','answer_path','answer_sha256')},
                      'passed':saved['score_v2']['passed'],'missing':saved['score_v2']['missing'],'category':saved['score_v2'].get('category')})
    if {p.name for p in (c['top']/SCORING/'scores').iterdir() if p.is_dir()}!={r['record_id'] for r in c['plan']['records']}:
        raise ValueError('Unexpected score transaction population')
    manifest={'experiment_id':'sparse_repair_01','cohort':'development_screen','expected_answers':504,
        'committed_answers':len(files),'missing_answers':sum(r['missing'] for r in files),
        'scoring_plan_path':str(c['plan_path'].relative_to(c['repo_root'])),'scoring_plan_sha256':c['plan_sha256'],
        'generation_closure_sha256':c['closure']['closure_sha256'],'raw_answers_unchanged':True,
        'hidden_tests_loaded_after_complete_screen_generation':True,'files':files}
    bind(path_inside(c['top'],SCORING+'/scored_answer_manifest.json'),manifest);return manifest


def check_cpus(cpus,policy):
    if len(cpus)!=8:
        raise ValueError('Exactly eight distinct physical CPU cores required')
    return base.cpu_environment(cpus,policy)


def preflight(repo,declaration_path,declaration_sha,lease_path,lease_sha,cpus,output):
    c=declaration_context(repo,declaration_path,declaration_sha)
    lease=base.load_lease(path_inside(repo,lease_path),lease_sha)
    lease,guard=base.allocation(repo,lease_path,lease_sha,lease['allowed_result_root'])
    if lease['experiment_id']!='sparse_repair_01':raise ValueError('Wrong CPU allocation experiment')
    environment=check_cpus(cpus,c['policy'])
    if not Path('/opt/gearshift-sandbox/READY').is_file():raise ValueError('Isolated Linux sandbox is not prepared')
    folder=path_inside(lease['allowed_result_root'],SCORING);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);removed=base.cleanup_orphans(lease)
        cases=list(base.calibration.cases());records_=[]
        def one(case,cpu,rep):
            name,key,inp,expected=case
            result=base.score(base.calibration.REFERENCES[key],{'fn_name':None,'tests':[{'input':inp,'output':expected}]},
                              guard=guard,policy=c['policy'],cpu_id=cpu)
            return {'case':name,'reference':key,'reference_sha256':base.code_hash(base.calibration.REFERENCES[key]),
                'input_sha256':base.code_hash(inp),'expected_sha256':base.code_hash(expected),
                'repetition':rep,'cpu_id':cpu,'score':result}
        for rep in range(3):
            assignments=[cases[i%len(cases)] for i in range(max(len(cpus),len(cases)))]
            with ThreadPoolExecutor(max_workers=8) as pool:
                for offset in range(0,len(assignments),8):
                    futures=[pool.submit(one,case,cpus[i],rep) for i,case in enumerate(assignments[offset:offset+8])]
                    records_.extend(f.result() for f in futures)
        receipt={'schema_version':1,'purpose':'confirmation_cpu_frozen_policy_preflight','experiment_id':'sparse_repair_01',
            'declaration_sha256':declaration_sha,'allocation_lease_sha256':lease_sha,'pod_id':lease['pod_id'],
            'policy_identity_sha256':digest(c['policy']),'scorer_identity_sha256':digest(c['scorer_identity']),
            'environment':environment,'calibration_implementation_sha256':sha(ROOT/'scripts/coding_scorer_repair_calibrate.py'),
            'records':records_,'policy_tuned':False,'private_tests_loaded':False,'orphan_cleanup_pids':removed,'created_epoch':time.time()}
        receipt['passed']=all(r['score']['passed'] is True and r['score']['receipts'][0]['cpu_seconds']<=c['policy']['cpu_seconds']/3 for r in records_)
        path=path_inside(repo,output);bind(path,receipt)
        base.verify_preflight(c,lease,lease_sha,path,sha(path),cpus)
        return receipt


def worker(c,rs,tests,cpu,attempt,guard,status):
    signal.signal(signal.SIGTERM,base.interruption);signal.signal(signal.SIGINT,base.interruption)
    try:
        os.sched_setaffinity(0,{cpu})
        for i,r in enumerate(rs):
            score_record(c,r,tests,cpu,attempt,guard)
            write(status,{'completed':i+1,'expected':len(rs),'cpu_id':cpu,'epoch':time.time()})
    except BaseException as exc:
        write(status,{'failed':True,'exception_type':type(exc).__name__,'epoch':time.time()});raise SystemExit(1)


def run(repo,plan_path,private_path,lease_path,lease_sha,preflight_path,preflight_sha,cpus):
    c=validate_plan(repo,plan_path);check_cpus(cpus,c['policy'])
    lease,guard=base.allocation(repo,lease_path,lease_sha,c['top'])
    if lease['experiment_id']!='sparse_repair_01':raise ValueError('Wrong CPU allocation experiment')
    receipt=base.verify_preflight(c,lease,lease_sha,path_inside(repo,preflight_path),preflight_sha,cpus)
    folder=path_inside(c['top'],SCORING);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);removed=base.cleanup_orphans(lease)
        tests=private_after_closure(c,private_path)
        if {tid:digest(tests[tid]) for tid in c['task_ids']}!=c['plan']['selected_private_spec_sha256']:
            raise ValueError('Selected private specs changed')
        attempt=lease['pod_id']+'_'+str(time.time_ns());out=folder/'executions'/attempt
        write(out/'identity.json',{'scoring_plan_sha256':c['plan_sha256'],'lease_sha256':lease_sha,
            'preflight_sha256':preflight_sha,'cpu_environment':receipt['environment'],'orphan_cleanup_pids':removed})
        processes=[];handlers={sig:signal.signal(sig,base.interruption) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            ctx=multiprocessing.get_context('fork')
            for i,cpu in enumerate(cpus):
                p=ctx.Process(target=worker,args=(c,c['plan']['records'][i::8],tests,cpu,attempt,guard,out/f'worker_{i:02d}.json'))
                p.start();processes.append(p)
            while any(p.is_alive() for p in processes):guard();time.sleep(1)
            if any(p.exitcode for p in processes):raise RuntimeError('Scoring worker failed; interrupted transactions become missing, not candidate retries')
        finally:
            for p in processes:
                if p.is_alive():p.terminate()
            for p in processes:
                p.join(timeout=5)
                if p.is_alive():p.kill();p.join()
            try:write(out/'cleanup.json',{'orphan_cleanup_pids':base.cleanup_orphans(lease)})
            finally:
                for sig,handler in handlers.items():signal.signal(sig,handler)
        result=finalize(repo,plan_path);write(out/'complete.json',{'committed_answers':result['committed_answers'],'missing_answers':result['missing_answers']})
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','seal','prepare','run','finalize']);p.add_argument('--repo',default=str(ROOT))
    for name in ('declaration','declaration-sha256','result-root','private-tests','plan','lease','lease-sha256','preflight','preflight-sha256','cpu-ids','output'):
        p.add_argument('--'+name)
    a=p.parse_args();cpus=[int(v) for v in a.cpu_ids.split(',')] if a.cpu_ids else []
    if a.action=='preflight':
        value=preflight(a.repo,a.declaration,a.declaration_sha256,a.lease,a.lease_sha256,cpus,a.output)
        result={'passed':value['passed'],'private_tests_loaded':False}
    elif a.action=='seal':
        value=seal(a.repo,a.declaration,a.declaration_sha256,a.result_root)
        result={'closure_sha256':value['closure_sha256'],'answers':len(value['answers'])}
    elif a.action=='prepare':
        value,path=prepare(a.repo,a.declaration,a.declaration_sha256,a.result_root,a.private_tests)
        result={'scoring_plan_path':str(path),'expected_answers':value['expected_answers']}
    elif a.action=='finalize':
        value=finalize(a.repo,a.plan);result={k:value[k] for k in ('committed_answers','missing_answers')}
    else:
        value=run(a.repo,a.plan,a.private_tests,a.lease,a.lease_sha256,a.preflight,a.preflight_sha256,cpus)
        result={k:value[k] for k in ('committed_answers','missing_answers')}
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
