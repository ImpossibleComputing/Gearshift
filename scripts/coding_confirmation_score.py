#!/usr/bin/env python3
"""Primary confirmation scoring behind the complete public-generation barrier.

Only the unchanged frozen versioned scorer executes candidates. Public closure,
implementation and identity verification happens before private bytes are read.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gearshift.coding_control import bind,digest,sha,write
from gearshift.coding_confirmation_lease import load_lease,control_root
from gearshift.coding_sandbox import extract
from gearshift.coding_sandbox_v2 import score,scorer_identity,validate_policy,output_allowance
from scripts import coding_confirmation_generate as generation
from scripts import coding_scorer_repair_calibrate as calibration

SCORING='primary/scoring'
RETRY_POLICY='Use unchanged frozen scorer: one verified infrastructure retry; no candidate timeout/resource/quality retries.'
INTERRUPTED_POLICY='A committed start without committed score becomes explicit missing coverage, never a candidate reroll.'


class ScoringInterrupted(BaseException):pass


def interruption(signum,frame):raise ScoringInterrupted('Scoring supervisor or worker interrupted')
REQUIRED_IMPLEMENTATION={
    'scripts/coding_confirmation_score.py','scripts/coding_confirmation_report.py','scripts/coding_scorer_repair_calibrate.py',
    'gearshift/coding_sandbox.py','gearshift/coding_sandbox_v2.py','scripts/coding_sandbox_child_v2.py',
    *generation.REQUIRED_IMPLEMENTATION}


def read(path):return json.loads(Path(path).read_text())
def scoped(root,relative):return generation.scoped_file(root,relative)
def code_hash(code):return hashlib.sha256(code.encode()).hexdigest()


def output_path(root,relative):
    root=Path(root).resolve();relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:raise ValueError('Output must remain inside result root')
    target=root/relative;cursor=target
    while cursor!=root:
        if cursor.is_symlink():raise ValueError('Output path cannot traverse a symlink')
        cursor=cursor.parent
    return target


def declared_scorer(repo,d):
    policy=validate_policy(read(scoped(repo,d['scorer']['policy_path'])))
    if policy.get('policy_status')!='frozen':raise ValueError('Scoring resource policy is not frozen')
    identity=scorer_identity(policy)
    if d['scorer'].get('version')!=identity['scorer_version']:raise ValueError('Declared scorer version differs')
    hashes={**d['inputs'],**d['implementation']}
    for name,expected in identity['files'].items():
        if hashes.get(name)!=expected or d['scorer']['implementation_hashes'].get(name)!=expected:raise ValueError('Frozen scorer implementation differs')
    for name in REQUIRED_IMPLEMENTATION:
        if hashes.get(name)!=sha(scoped(repo,name)) or sha(ROOT/name)!=hashes[name]:raise ValueError('Loaded scoring/verification source is not frozen: '+name)
    private_identity_path=d['scorer']['private_test_identity_path']
    if private_identity_path not in d['inputs']:raise ValueError('Private-test identity receipt is not publicly bound')
    private_identity=read(scoped(repo,private_identity_path))
    selected=d['task_ids'];full=private_identity.get('private_file_task_ids',selected)
    if private_identity.get('task_ids')!=selected or private_identity.get('task_count')!=len(selected):raise ValueError('Private-test selected scope differs')
    if not isinstance(full,list) or len(set(full))!=len(full) or not set(selected)<=set(full):raise ValueError('Private-file task inventory is invalid')
    if private_identity.get('private_tests_sha256')!=d['scorer'].get('private_tests_sha256') or not re.fullmatch('[0-9a-f]{64}',private_identity['private_tests_sha256']):raise ValueError('Private-test hash identity differs')
    gate_path=private_identity['historical_gate_path']
    if gate_path not in d['inputs'] or d['inputs'][gate_path]!=private_identity['historical_gate_sha256']:raise ValueError('Historical private-test identity is not bound')
    gate=read(scoped(repo,gate_path))
    original=[r['task_id'] for r in gate['rows'] if r['split']=='confirmation']
    if gate.get('passed') is not True or len(set(original))!=len(original) or set(original)!=set(full) or gate['identity']['private_files']['confirmation']!=private_identity['private_tests_sha256']:raise ValueError('Historical private-file hash or inventory differs')
    if private_identity.get('status')!='FROZEN' or private_identity.get('experiment_id')!=d['experiment_id'] or private_identity.get('private_test_values_included') is not False or private_identity.get('private_file_task_count')!=len(full):raise ValueError('Private-test public receipt is not frozen or scoped')
    return policy,identity,private_identity


def public_context(repo,generation_plan_path,generation_plan_sha256):
    """No private path or private bytes are accessed in this function."""
    repo=Path(repo).resolve();path=scoped(repo,generation_plan_path)
    if sha(path)!=generation_plan_sha256:raise ValueError('Frozen generation plan changed')
    gp=read(path);d=generation.validate_declaration(repo,gp['declaration_path'],gp['declaration_sha256'])
    if gp['experiment_id']!=d['experiment_id'] or not d.get('source_commit') or gp.get('code_commit')!=d['source_commit']:raise ValueError('Generation plan/declaration source identity differs')
    top=output_path(repo,gp['result_root'])
    policy,identity,private_identity=declared_scorer(repo,d)
    seeds=read(scoped(repo,d['seeds_path']));rows=read(scoped(repo,d['visible_path']))
    if len({r['task_id'] for r in rows})!=len(rows):raise ValueError('Duplicate visible task records')
    visible={r['task_id']:r for r in rows if r['task_id'] in d['task_ids']}
    if set(visible)!=set(d['task_ids']):raise ValueError('Visible task coverage differs')
    c={'repo_root':repo,'top':top,'declaration':d,'declaration_sha256':gp['declaration_sha256'],'seeds':seeds,'visible':visible}
    closure_path=scoped(top,'primary/generation_closure.json');saved=read(closure_path)
    actual=generation.generation_closure(c,seal=False)
    if actual is None or actual!=saved:raise ValueError('Primary generation seal is incomplete or differs from current raw artifacts')
    payload={k:v for k,v in saved.items() if k!='closure_sha256'}
    if saved['closure_sha256']!=digest(payload) or not saved['all_primary_generation_complete']:raise ValueError('Generation closure checksum/completion differs')
    c.update(generation_plan=gp,generation_plan_path=str(Path(generation_plan_path)),generation_plan_sha256=generation_plan_sha256,
        closure=saved,closure_file_sha256=sha(closure_path),policy=policy,scorer_identity=identity,private_identity=private_identity)
    return c


def private_after_closure(c,private_path):
    # The caller obtains c only after recomputing and matching the entire seal.
    path=Path(private_path)
    if path.resolve().is_relative_to((c['top']/'primary').resolve()):raise ValueError('Private tests cannot live inside public primary generation artifacts')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=c['private_identity']['private_tests_sha256']:raise ValueError('Private-test bytes differ from frozen identity')
    tests=json.loads(raw)
    inventory=c['private_identity'].get('private_file_task_ids',c['declaration']['task_ids'])
    if not isinstance(tests,dict) or set(tests)!=set(inventory):raise ValueError('Private-file key inventory differs from historical declaration')
    # Unselected reserve tasks never enter a worker's scoring dictionary.
    selected={tid:tests[tid] for tid in c['declaration']['task_ids']}
    for tid,spec in selected.items():
        if not isinstance(spec,dict) or not isinstance(spec.get('tests'),list) or not spec['tests'] or (spec.get('fn_name') is not None and not isinstance(spec['fn_name'],str)):raise ValueError('Invalid selected private task specification: '+tid)
        for test in spec['tests']:
            if not isinstance(test,dict) or not isinstance(test.get('input'),str) or not isinstance(test.get('output'),str):raise ValueError('Invalid selected private test interface: '+tid)
            output_allowance(c['policy'],test['output'],spec.get('fn_name'))
    return selected


def prepare(repo,generation_plan_path,generation_plan_sha256,private_path):
    c=public_context(repo,generation_plan_path,generation_plan_sha256)
    tests=private_after_closure(c,private_path);d=c['declaration'];records=[]
    for item in c['closure']['answers']:
        raw=read(scoped(c['top'],item['path']));contract=item['contract']
        records.append({'record_id':digest(contract),'contract':contract,'contract_sha256':item['contract_sha256'],
            'answer_path':item['path'],'answer_sha256':item['answer_sha256'],'code_sha256':code_hash(extract(raw['answer_text'])),
            'task_id':contract['task_id'],'condition':contract['condition'],'seed_index':contract['seed_index'],'answer_seed':contract['answer_seed']})
    expected=[(tid,condition,seed['seed_index']) for tid in d['task_ids'] for condition in generation.CONDITIONS for seed in c['seeds']['tasks'][tid]['answers']]
    if [(r['task_id'],r['condition'],r['seed_index']) for r in records]!=expected or len({r['record_id'] for r in records})!=len(expected):raise ValueError('Exact task/condition/seed ordering or coverage differs')
    plan={'schema_version':1,'experiment_id':d['experiment_id'],'cohort':'primary','result_root':c['generation_plan']['result_root'],
        'declaration_path':c['generation_plan']['declaration_path'],'declaration_sha256':c['declaration_sha256'],
        'generation_plan_path':c['generation_plan_path'],'generation_plan_sha256':generation_plan_sha256,
        'generation_closure_path':'primary/generation_closure.json','generation_closure_sha256':c['closure']['closure_sha256'],
        'generation_closure_identity_sha256':c['closure']['closure_sha256'],'generation_closure_file_sha256':c['closure_file_sha256'],
        'scorer_identity':c['scorer_identity'],'scorer_identity_sha256':digest(c['scorer_identity']),
        'policy':c['policy'],'policy_identity_sha256':digest(c['policy']),
        'implementation_hashes':{name:sha(scoped(repo,name)) for name in sorted(REQUIRED_IMPLEMENTATION)},
        'private_test_identity':c['private_identity'],'selected_private_spec_sha256':{tid:digest(tests[tid]) for tid in d['task_ids']},
        'task_ids':d['task_ids'],'conditions':generation.CONDITIONS,'expected_answers':len(expected),'records':records,
        'manifest_path':SCORING+'/scored_answer_manifest.json','hidden_tests_loaded_after_primary_generation':True,
        'retry_policy':RETRY_POLICY,
        'interrupted_transaction_policy':INTERRUPTED_POLICY}
    target=output_path(c['top'],SCORING+'/scoring_plan.json');bind(target,plan)
    return plan,target


def validate_plan(repo,plan_path):
    path=scoped(repo,plan_path);plan=read(path)
    c=public_context(repo,plan['generation_plan_path'],plan['generation_plan_sha256'])
    if path!=output_path(c['top'],SCORING+'/scoring_plan.json'):raise ValueError('Unexpected scoring plan location')
    required={'schema_version':1,'experiment_id':c['declaration']['experiment_id'],'cohort':'primary',
        'result_root':c['generation_plan']['result_root'],'declaration_path':c['generation_plan']['declaration_path'],
        'declaration_sha256':c['declaration_sha256'],'generation_closure_path':'primary/generation_closure.json',
        'generation_closure_sha256':c['closure']['closure_sha256'],'scorer_identity_sha256':digest(c['scorer_identity']),
        'policy_identity_sha256':digest(c['policy']),'manifest_path':SCORING+'/scored_answer_manifest.json',
        'hidden_tests_loaded_after_primary_generation':True,'retry_policy':RETRY_POLICY,'interrupted_transaction_policy':INTERRUPTED_POLICY}
    if any(plan.get(k)!=v for k,v in required.items()) or set(plan['implementation_hashes'])!=REQUIRED_IMPLEMENTATION:raise ValueError('Frozen scoring plan identity or implementation coverage differs')
    spec_hashes=plan['selected_private_spec_sha256']
    if set(spec_hashes)!=set(c['declaration']['task_ids']) or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in spec_hashes.values()):raise ValueError('Selected private specification identity coverage differs')
    if plan['scorer_identity']!=c['scorer_identity'] or plan['policy']!=c['policy'] or plan['private_test_identity']!=c['private_identity']:raise ValueError('Scoring identity drift')
    if plan['generation_closure_identity_sha256']!=c['closure']['closure_sha256'] or plan['generation_closure_file_sha256']!=c['closure_file_sha256']:raise ValueError('Scoring plan points to a different generation seal')
    for name,expected in plan['implementation_hashes'].items():
        if sha(scoped(repo,name))!=expected or sha(ROOT/name)!=expected:raise ValueError('Loaded scoring implementation drift')
    # Reconstruct the complete public record binding without touching tests.
    expected=[]
    for item in c['closure']['answers']:
        contract=item['contract'];raw=read(scoped(c['top'],item['path']))
        expected.append({'record_id':digest(contract),'contract':contract,'contract_sha256':item['contract_sha256'],
            'answer_path':item['path'],'answer_sha256':item['answer_sha256'],'code_sha256':code_hash(extract(raw['answer_text'])),
            'task_id':contract['task_id'],'condition':contract['condition'],'seed_index':contract['seed_index'],'answer_seed':contract['answer_seed']})
    if plan['records']!=expected or plan['expected_answers']!=len(expected) or plan['task_ids']!=c['declaration']['task_ids'] or plan['conditions']!=generation.CONDITIONS:raise ValueError('Scoring plan population drift')
    c.update(plan=plan,plan_path=path,plan_sha256=sha(path),plan_identity_sha256=digest(plan))
    return c


def score_binding(c,record):
    p=c['plan']
    return {'scoring_plan_identity_sha256':c['plan_identity_sha256'],'scoring_plan_sha256':c['plan_sha256'],
        'declaration_sha256':p['declaration_sha256'],'generation_closure_identity_sha256':p['generation_closure_identity_sha256'],
        'generation_closure_file_sha256':p['generation_closure_file_sha256'],'scorer_identity_sha256':p['scorer_identity_sha256'],
        **{key:record[key] for key in ('record_id','answer_sha256','contract_sha256','code_sha256')}}


def validate_result(result,identity):
    if result.get('passed') is None:
        if result.get('missing') is not True or result.get('category')!='infrastructure_failure':raise ValueError('Null score lacks explicit infrastructure missingness')
    elif type(result['passed']) is not bool or result.get('missing') is not False:raise ValueError('Score outcome is not a Boolean or explicit missing')
    if result.get('scorer_identity')!=identity:raise ValueError('Score result scorer identity differs')


def score_record(c,record,tests,cpu_id,attempt,guard):
    path=scoped(c['top'],record['answer_path'])
    if sha(path)!=record['answer_sha256']:raise ValueError('Raw answer changed before scoring')
    raw=read(path);code=extract(raw['answer_text'])
    if code_hash(code)!=record['code_sha256']:raise ValueError('Unchanged extraction binding differs')
    directory=output_path(c['top'],SCORING+'/scores/'+record['record_id']);directory.mkdir(parents=True,exist_ok=True)
    target=directory/'score.json';binding=score_binding(c,record)
    with (directory/'transaction.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.exists():
            saved=read(target)
            if saved['binding']!=binding or saved['code']!=code or saved.get('contract')!=record['contract'] or saved.get('hidden_tests_loaded_after_primary_generation') is not True:raise ValueError('Existing scoring transaction has another identity')
            validate_result(saved['score_v2'],c['scorer_identity']);return saved
        started=directory/'started.json'
        if started.exists():
            if read(started)['binding']!=binding:raise ValueError('Interrupted transaction identity differs')
            result={'passed':None,'missing':True,'category':'infrastructure_failure','reason':'worker_interrupted_before_score_commit',
                    'executed_tests':None,'trusted_checker_complete':False,'scorer_identity':c['scorer_identity']}
        else:
            guard();write(started,{'binding':binding,'attempt':attempt,'cpu_id':cpu_id,'pid':os.getpid(),'epoch':time.time()})
            spec=tests.get(record['task_id'])
            if not spec:result={'passed':None,'missing':True,'category':'infrastructure_failure','reason':'missing_selected_task_tests','executed_tests':0,'scorer_identity':c['scorer_identity']}
            else:result=score(code,spec,guard=guard,policy=c['policy'],cpu_id=cpu_id)
        validate_result(result,c['scorer_identity'])
        if sha(path)!=record['answer_sha256']:raise ValueError('Raw answer changed during scoring')
        receipt={'binding':binding,'contract':record['contract'],'code':code,'score_v2':result,
            **{k:record[k] for k in ('task_id','condition','seed_index','answer_seed','answer_path','answer_sha256')},
            'attempt':attempt,'cpu_id':cpu_id,'completed_epoch':time.time(),'hidden_tests_loaded_after_primary_generation':True}
        write(target,receipt);return receipt


def finalize(repo,plan_path):
    c=validate_plan(repo,plan_path);plan=c['plan'];files=[]
    for record in plan['records']:
        relative=SCORING+'/scores/'+record['record_id']+'/score.json';path=scoped(c['top'],relative);receipt=read(path)
        if receipt['binding']!=score_binding(c,record) or receipt.get('contract')!=record['contract'] or receipt.get('answer_path')!=record['answer_path']:raise ValueError('Score receipt public binding differs')
        raw=read(scoped(c['top'],record['answer_path']))
        if receipt.get('code')!=extract(raw['answer_text']) or any(receipt.get(k)!=record[k] for k in ('task_id','condition','seed_index','answer_seed','answer_sha256')):raise ValueError('Scored code/population differs')
        if receipt.get('hidden_tests_loaded_after_primary_generation') is not True:raise ValueError('Score lacks generation-barrier receipt')
        validate_result(receipt['score_v2'],c['scorer_identity'])
        files.append({'path':relative,'sha256':sha(path),**{k:record[k] for k in ('answer_path','answer_sha256','task_id','condition','seed_index','answer_seed')},
            'missing':receipt['score_v2']['missing'],'passed':receipt['score_v2']['passed']})
    actual={p.name for p in output_path(c['top'],SCORING+'/scores').glob('*') if p.is_dir()}
    if actual!={r['record_id'] for r in plan['records']}:raise ValueError('Unexpected scoring transaction population')
    manifest={'schema_version':1,'experiment_id':plan['experiment_id'],'cohort':'primary','declaration_sha256':plan['declaration_sha256'],
        'generation_closure_sha256':plan['generation_closure_identity_sha256'],'generation_closure_identity_sha256':plan['generation_closure_identity_sha256'],
        'generation_closure_file_sha256':plan['generation_closure_file_sha256'],'scorer_identity_sha256':plan['scorer_identity_sha256'],
        'scoring_plan_path':str(c['plan_path'].relative_to(Path(repo).resolve())),'scoring_plan_sha256':c['plan_sha256'],'scoring_plan_identity_sha256':c['plan_identity_sha256'],
        'task_ids':plan['task_ids'],'conditions':plan['conditions'],'expected_answers':plan['expected_answers'],'committed_answers':len(files),
        'missing_answers':sum(r['missing'] for r in files),'files':files,'hidden_tests_loaded_after_primary_generation':True,'raw_answers_unchanged':True}
    bind(output_path(c['top'],plan['manifest_path']),manifest);return manifest


def cpu_environment(cpu_ids,policy):
    if sys.platform!='linux' or os.geteuid()!=0:raise ValueError('Dedicated Linux CPU scorer required')
    allowed=sorted(os.sched_getaffinity(0));cpus=list(cpu_ids)
    if not cpus or len(set(cpus))!=len(cpus) or len(cpus)>policy['maximum_concurrent_candidates'] or not set(cpus)<=set(allowed):raise ValueError('CPU allocation exceeds frozen concurrency or allowed cores')
    cores=[]
    for cpu in cpus:
        base=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology');cores.append([(base/x).read_text().strip() for x in ('physical_package_id','core_id')])
    if len({tuple(x) for x in cores})!=len(cores):raise ValueError('Scoring CPU assignments share physical cores')
    cg=Path('/sys/fs/cgroup');memory=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:')))*1024
    for name in ('memory.max','memory/memory.limit_in_bytes'):
        if (cg/name).exists() and (value:=(cg/name).read_text().strip())!='max':memory=min(memory,int(value))
    quota=None
    if (cg/'cpu.max').exists():
        values=(cg/'cpu.max').read_text().split();quota=None if values[0]=='max' else int(values[0])/int(values[1])
    elif (cg/'cpu,cpuacct/cpu.cfs_quota_us').exists():
        q=int((cg/'cpu,cpuacct/cpu.cfs_quota_us').read_text());quota=None if q<0 else q/int((cg/'cpu,cpuacct/cpu.cfs_period_us').read_text())
    if (quota is not None and quota<len(cpus)) or memory<len(cpus)*(policy['address_space_bytes']+1024**3):raise ValueError('CPU quota or RAM would oversubscribe scoring')
    return {'hostname':platform.node(),'platform':platform.platform(),'candidate_python_sha256':sha('/usr/bin/python3'),
        'candidate_python_version':subprocess.check_output(['/usr/bin/python3','--version'],text=True).strip(),
        'allowed_cpus':allowed,'cpu_ids':cpus,'physical_cores':cores,'memory_limit_bytes':memory,'cpu_quota_cores':quota}


def allocation(repo,lease_path,lease_sha256,top):
    lease=load_lease(scoped(repo,lease_path),lease_sha256)
    if lease['gpu_count']!=0 or Path(lease['allowed_result_root']).resolve()!=Path(top).resolve() or os.environ.get('RUNPOD_POD_ID')!=lease['pod_id']:raise ValueError('Scoring requires its own dedicated CPU allocation')
    def guard():
        if time.time()>=lease['deadline_epoch']-120 or (control_root(lease)/'lease_guard/STOP').exists():raise TimeoutError('Scoring allocation expired or stopped')
    guard();return lease,guard


def cleanup_orphans(lease,proc_root=Path('/proc')):
    """Kill only orphaned confined children of this reserved CPU allocation.

    pidfds prevent signalling a reused PID. Live shard children are untouched.
    The unchanged sandbox forbids child creation, so one process is the unit.
    """
    if lease['gpu_count']!=0 or os.environ.get('RUNPOD_POD_ID')!=lease['pod_id']:raise ValueError('Orphan cleanup requires the reserved private CPU allocation')
    child=str(ROOT/'scripts/coding_sandbox_child_v2.py').encode();removed=[]
    for folder in Path(proc_root).iterdir():
        if not folder.name.isdigit():continue
        try:
            args=(folder/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if len(args)!=5 or args[:3]!=[b'/usr/bin/python3',b'-I',child] or not args[3].endswith(b'/payload.json') or args[-1]!=b'':continue
        pid=int(folder.name);fd=None
        try:
            fd=os.pidfd_open(pid,0)
            # Re-read after opening the stable kernel process handle.
            current=(folder/'cmdline').read_bytes().split(b'\0')
            status=dict(line.split(':',1) for line in (folder/'status').read_text().splitlines() if ':' in line)
            if current!=args or int(status['PPid'])!=1 or os.getpgid(pid)!=pid:continue
            signal.pidfd_send_signal(fd,signal.SIGKILL,None,0);removed.append(pid)
        except (FileNotFoundError,ProcessLookupError):pass
        finally:
            if fd is not None:os.close(fd)
    return removed


def preflight(repo,declaration_path,declaration_sha256,lease_path,lease_sha256,cpu_ids,output):
    repo=Path(repo).resolve();d=generation.validate_declaration(repo,declaration_path,declaration_sha256);policy,identity,_=declared_scorer(repo,d)
    lease=load_lease(scoped(repo,lease_path),lease_sha256);lease,guard=allocation(repo,lease_path,lease_sha256,lease['allowed_result_root'])
    if lease['experiment_id']!=d['experiment_id']:raise ValueError('CPU allocation belongs to another experiment')
    environment=cpu_environment(cpu_ids,policy)
    if not Path('/opt/gearshift-sandbox/READY').is_file():raise ValueError('Isolated scorer standard library must already be prepared')
    folder=output_path(lease['allowed_result_root'],SCORING);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        removed=cleanup_orphans(lease)
        cases=list(calibration.cases());records=[]
        def one(case,cpu,rep):
            name,key,inp,expected=case
            result=score(calibration.REFERENCES[key],{'fn_name':None,'tests':[{'input':inp,'output':expected}]},guard=guard,policy=policy,cpu_id=cpu)
            return {'case':name,'reference':key,'reference_sha256':code_hash(calibration.REFERENCES[key]),'input_sha256':code_hash(inp),
                'expected_sha256':code_hash(expected),'repetition':rep,'cpu_id':cpu,'score':result}
        for rep in range(3):
            assignments=[cases[i%len(cases)] for i in range(max(len(cpu_ids),len(cases)))]
            with ThreadPoolExecutor(max_workers=len(cpu_ids)) as pool:
                for offset in range(0,len(assignments),len(cpu_ids)):
                    futures=[pool.submit(one,case,cpu_ids[i],rep) for i,case in enumerate(assignments[offset:offset+len(cpu_ids)])]
                    records.extend(f.result() for f in futures)
        receipt={'schema_version':1,'purpose':'confirmation_cpu_frozen_policy_preflight','experiment_id':d['experiment_id'],
            'declaration_sha256':declaration_sha256,'allocation_lease_sha256':lease_sha256,'pod_id':lease['pod_id'],
            'policy_identity_sha256':digest(policy),'scorer_identity_sha256':digest(identity),'environment':environment,
            'calibration_implementation_sha256':sha(ROOT/'scripts/coding_scorer_repair_calibrate.py'),'records':records,
            'policy_tuned':False,'private_tests_loaded':False,'orphan_cleanup_pids':removed,'created_epoch':time.time()}
        receipt['passed']=all(r['score']['passed'] is True and r['score']['receipts'][0]['cpu_seconds']<=policy['cpu_seconds']/3 for r in records)
        bind(output_path(repo,output),receipt)
        if not receipt['passed']:raise ValueError('Host fails comparable frozen reference capacity; no policy tuning or candidate scoring allowed')
        return receipt


def verify_preflight(c,lease,lease_sha256,path,expected_sha,cpu_ids):
    if sha(path)!=expected_sha:raise ValueError('CPU preflight receipt hash differs')
    p=read(path)
    expected={'purpose':'confirmation_cpu_frozen_policy_preflight','experiment_id':c['declaration']['experiment_id'],
        'declaration_sha256':c['declaration_sha256'],'allocation_lease_sha256':lease_sha256,'pod_id':lease['pod_id'],
        'policy_identity_sha256':digest(c['policy']),'scorer_identity_sha256':digest(c['scorer_identity']),
        'environment':cpu_environment(cpu_ids,c['policy']),'calibration_implementation_sha256':sha(ROOT/'scripts/coding_scorer_repair_calibrate.py'),
        'passed':True,'policy_tuned':False,'private_tests_loaded':False}
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('CPU preflight environment, identity or unchanged policy differs')
    cases={name:(key,code_hash(inp),code_hash(out)) for name,key,inp,out in calibration.cases()};seen=set();cpus=set()
    names=list(cases);expected_order=[(names[i%len(names)],rep,cpu_ids[i%len(cpu_ids)]) for rep in range(3) for i in range(max(len(cpu_ids),len(names)))]
    if [(r['case'],r['repetition'],r['cpu_id']) for r in p['records']]!=expected_order:raise ValueError('Capacity reference assignments differ from the fixed preflight')
    for r in p['records']:
        if r['case'] not in cases or r['repetition'] not in range(3) or r['cpu_id'] not in cpu_ids:raise ValueError('Undeclared capacity reference record')
        key,inp,out=cases[r['case']]
        if (r['reference'],r['reference_sha256'],r['input_sha256'],r['expected_sha256'])!=(key,code_hash(calibration.REFERENCES[key]),inp,out):raise ValueError('Public reference identity differs')
        validate_result(r['score'],c['scorer_identity']);receipts=r['score'].get('receipts',[])
        if r['score']['passed'] is not True or len(receipts)!=1 or receipts[0].get('cpu_seconds') is None or type(receipts[0]['cpu_seconds']) not in (int,float) or not math.isfinite(receipts[0]['cpu_seconds']) or not 0<=receipts[0]['cpu_seconds']<=c['policy']['cpu_seconds']/3:raise ValueError('Reference capacity exceeds frozen headroom')
        seen.add((r['case'],r['repetition']));cpus.add(r['cpu_id'])
    if seen!={(name,rep) for name in cases for rep in range(3)} or cpus!=set(cpu_ids):raise ValueError('Capacity preflight population incomplete')
    return p


def _worker(c,records,tests,cpu_id,attempt,guard,status):
    signal.signal(signal.SIGTERM,interruption);signal.signal(signal.SIGINT,interruption)
    try:
        os.sched_setaffinity(0,{cpu_id})
        for index,record in enumerate(records):
            score_record(c,record,tests,cpu_id,attempt,guard)
            write(status,{'completed':index+1,'expected':len(records),'cpu_id':cpu_id,'pid':os.getpid(),'epoch':time.time()})
    except BaseException as exc:
        write(status,{'failed':True,'exception_type':type(exc).__name__,'cpu_id':cpu_id,'pid':os.getpid(),'epoch':time.time()})
        raise SystemExit(1)


def run(repo,plan_path,private_path,lease_path,lease_sha256,preflight_path,preflight_sha256,cpu_ids):
    c=validate_plan(repo,plan_path)
    lease,guard=allocation(repo,lease_path,lease_sha256,c['top'])
    if lease['experiment_id']!=c['declaration']['experiment_id']:raise ValueError('Allocation experiment differs')
    preflight_receipt=verify_preflight(c,lease,lease_sha256,scoped(repo,preflight_path),preflight_sha256,cpu_ids)
    folder=output_path(c['top'],SCORING);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        removed=cleanup_orphans(lease)
        tests=private_after_closure(c,private_path)
        if {tid:digest(tests[tid]) for tid in c['plan']['task_ids']}!=c['plan']['selected_private_spec_sha256']:raise ValueError('Selected private specifications drift')
        attempt=lease['pod_id']+'_'+str(time.time_ns());attempt_root=folder/'executions'/attempt
        write(attempt_root/'identity.json',{'attempt':attempt,'scoring_plan_sha256':c['plan_sha256'],'lease_path':lease_path,
            'lease_sha256':lease_sha256,'preflight_path':preflight_path,'preflight_sha256':preflight_sha256,'cpu_environment':preflight_receipt['environment'],'orphan_cleanup_pids':removed})
        processes=[];handlers={sig:signal.signal(sig,interruption) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            ctx=multiprocessing.get_context('fork')
            for i,cpu in enumerate(cpu_ids):
                proc=ctx.Process(target=_worker,args=(c,c['plan']['records'][i::len(cpu_ids)],tests,cpu,attempt,guard,attempt_root/f'worker_{i:02d}.json'));proc.start();processes.append(proc)
            while any(p.is_alive() for p in processes):guard();time.sleep(1)
            codes=[p.exitcode for p in processes]
            if any(codes):raise RuntimeError('Scoring worker failed; preserve interrupted transactions as missing before bounded recovery: '+str(codes))
        except BaseException as exc:
            write(attempt_root/'failed.json',{'exception_type':type(exc).__name__,'epoch':time.time(),'recovery':'Preserve started-but-uncommitted transactions as missing; never rerun their candidates.'})
            raise
        finally:
            for p in processes:
                if p.is_alive():p.terminate()
            for p in processes:
                p.join(timeout=5)
                if p.is_alive():p.kill();p.join()
            try:
                removed=cleanup_orphans(lease)
                write(attempt_root/'cleanup.json',{'orphan_cleanup_pids':removed,'epoch':time.time()})
            finally:
                for sig,handler in handlers.items():signal.signal(sig,handler)
        result=finalize(repo,plan_path);write(attempt_root/'complete.json',{'committed_answers':result['committed_answers'],'missing_answers':result['missing_answers'],'epoch':time.time()});return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','run','finalize']);p.add_argument('--repo',default=str(ROOT));p.add_argument('--generation-plan');p.add_argument('--generation-plan-sha256');p.add_argument('--plan');p.add_argument('--private-tests');p.add_argument('--declaration');p.add_argument('--declaration-sha256');p.add_argument('--lease');p.add_argument('--lease-sha256');p.add_argument('--preflight');p.add_argument('--preflight-sha256');p.add_argument('--cpu-ids');p.add_argument('--output');a=p.parse_args();cpus=[int(x) for x in a.cpu_ids.split(',')] if a.cpu_ids else []
    if a.action=='prepare':value,path=prepare(a.repo,a.generation_plan,a.generation_plan_sha256,a.private_tests);result={'scoring_plan_path':str(path),'expected_answers':value['expected_answers']}
    elif a.action=='preflight':value=preflight(a.repo,a.declaration,a.declaration_sha256,a.lease,a.lease_sha256,cpus,a.output);result={'passed':value['passed']}
    elif a.action=='finalize':value=finalize(a.repo,a.plan);result={'committed_answers':value['committed_answers'],'missing_answers':value['missing_answers']}
    else:value=run(a.repo,a.plan,a.private_tests,a.lease,a.lease_sha256,a.preflight,a.preflight_sha256,cpus);result={'committed_answers':value['committed_answers'],'missing_answers':value['missing_answers']}
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
