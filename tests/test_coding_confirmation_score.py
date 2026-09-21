"""Synthetic seal/private-barrier/transaction tests; never executes candidates."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from gearshift.coding_control import digest, sha, write
from scripts import coding_confirmation_score as s
from scripts import coding_confirmation_report as reporter

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('generation_fixture', ROOT/'tests/test_coding_confirmation_generate.py')
gfixture = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gfixture)


def fixture(tmp_path, count=1, *, private_mutation=None):
    """Complete fake sampler transactions with the real production validators."""
    c = gfixture.context(tmp_path, count); d = copy.deepcopy(c['declaration'])
    tids = d['task_ids']; reserve = 'atcoder/untouched_reserve'
    specs = {t: {'fn_name': None, 'tests': [{'input': '', 'output': '1\n'}]} for t in [*tids, reserve]}
    if private_mutation: private_mutation(specs)
    private_path = tmp_path/'protected/private.json'; write(private_path, specs)
    policy = s.read(ROOT/'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/calibration/frozen_policy.json')
    identity = s.scorer_identity(policy)
    analysis = s.read(ROOT/'configs/coding_pilot_v1/confirmation_01/analysis_plan.draft.json')
    analysis.update(status='FROZEN', task_count=count)
    gate = {'passed': True, 'rows': [{'task_id': t, 'split': 'confirmation'} for t in [*tids, reserve]],
            'identity': {'private_files': {'confirmation': sha(private_path)}}}
    inputs = {'seeds.json': c['seeds'], 'visible.json': list(c['visible'].values()), 'analysis.json': analysis,
        'configs/coding_pilot_v1/pilot.json': {'models': {role: {**spec, 'dtype': 'bfloat16', 'frozen': True} for role, spec in s.generation.MODELS.items()}},
        'policy.json': policy, 'identity.json': identity, 'historical_gate.json': gate,
        'proof.json': {'files': [{'sha256': cp['mapper_sha256']} for cp in d['primary_checkpoints'].values()]}}
    for name, value in inputs.items(): write(tmp_path/name, value)
    public_identity = {'experiment_id': d['experiment_id'], 'status': 'FROZEN', 'task_ids': tids,
        'task_count': len(tids), 'private_file_task_ids': [*tids, reserve], 'private_file_task_count': len(tids)+1,
        'private_tests_sha256': sha(private_path), 'private_file_rewritten': False, 'private_test_values_included': False,
        'historical_gate_path': 'historical_gate.json', 'historical_gate_sha256': sha(tmp_path/'historical_gate.json')}
    inputs['private_identity.json'] = public_identity; write(tmp_path/'private_identity.json', public_identity)
    for name in s.REQUIRED_IMPLEMENTATION:
        dest = tmp_path/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dest)
    for cp in d['primary_checkpoints'].values(): cp['actual_checkpoint_byte_proof'] = 'proof.json'
    d.update(status='FROZEN', generation_authorized_by_this_file=True, source_commit='synthetic_commit', publication_commit='synthetic_publication', primary_training_seed=20260918,
        task_scope_resolution='synthetic_explicit_scope', seeds_path='seeds.json', visible_path='visible.json', analysis_path='analysis.json',
        inputs={p: sha(tmp_path/p) for p in inputs}, implementation={p: sha(tmp_path/p) for p in s.REQUIRED_IMPLEMENTATION},
        scorer={'version': identity['scorer_version'], 'policy_path': 'policy.json', 'policy_sha256': sha(tmp_path/'policy.json'),
            'policy_identity_sha256': digest(policy), 'identity_path': 'identity.json', 'identity_file_sha256': sha(tmp_path/'identity.json'),
            'implementation_hashes': {p: sha(tmp_path/p) for p in s.REQUIRED_IMPLEMENTATION},
            'private_tests_sha256': sha(private_path), 'private_test_identity_path': 'private_identity.json'})
    write(tmp_path/'declaration.json', d); c.update(declaration=d, declaration_sha256=sha(tmp_path/'declaration.json'))
    gp = {'experiment_id': d['experiment_id'], 'code_commit': d['source_commit'], 'result_root': 'results',
          'declaration_path': 'declaration.json', 'declaration_sha256': c['declaration_sha256']}
    write(tmp_path/'generation_plan.json', gp)
    setup_path=c['status_root']/'model_setup.json'
    setup=s.read(setup_path);setup['declaration_sha256']=c['declaration_sha256'];write(setup_path,setup)
    gfixture.complete_fixture(c)
    # Add legitimate runner-stage timings before committing a synthetic seal.
    for answer in (c['top']/'primary/tasks').glob('*/*/seed_*/answer.json'):
        raw = s.read(answer); condition = raw['condition']; reasoning = 0. if condition=='P' else 2.
        raw.update(source_cost_divisor_for_single_output=1, reasoning_model=None if condition=='P' else 'receiver' if condition=='B' else 'source',
            reasoning_seconds=reasoning, native_prefill_seconds=.1, mapping_seconds=0., splice_seconds=0., cache_clone_seconds=.1,
            source_cache_reconstruction_seconds=0., inference_timing_complete=True, single_output_inference_seconds=reasoning+1.2)
        write(answer, raw); (answer.parent/'draw_complete.json').unlink()
        s.generation.verify_draw(answer.parent, s.read(answer.parent/'draw_identity.json'))
    s.generation.generation_closure(c)
    return c, private_path


def prepared(tmp_path):
    c, private = fixture(tmp_path)
    plan, path = s.prepare(tmp_path, 'generation_plan.json', sha(tmp_path/'generation_plan.json'), private)
    return s.validate_plan(tmp_path, str(path.relative_to(tmp_path))), private


def outcome(c, passed=False, *, cpu=1.):
    return {'passed': passed, 'missing': passed is None, 'category': 'infrastructure_failure' if passed is None else 'pass' if passed else 'test_assertion',
        'scorer_identity': c['scorer_identity'], 'receipts': [{'cpu_seconds': cpu, 'attempts': []}], 'executed_tests': 1}


def fake_scores(c, monkeypatch):
    calls = []
    def fake(code, spec, **kwargs):
        calls.append((code, copy.deepcopy(spec), kwargs)); return outcome(c)
    monkeypatch.setattr(s, 'score', fake)
    return calls


def complete_scores(c, private, monkeypatch):
    calls = fake_scores(c, monkeypatch); tests = s.private_after_closure(c, private)
    for record in c['plan']['records']: s.score_record(c, record, tests, 3, 'synthetic', lambda: None)
    return calls


def test_prepare_binds_full_seal_and_only_selected_private_tasks(tmp_path):
    c, private = prepared(tmp_path); p = c['plan']
    assert len(p['records']) == 24 and list(p['selected_private_spec_sha256']) == p['task_ids']
    assert p['generation_closure_identity_sha256'] != p['generation_closure_file_sha256']
    assert set(s.private_after_closure(c, private)) == set(p['task_ids'])
    assert len(p['private_test_identity']['private_file_task_ids']) == 2
    before = c['plan_path'].read_bytes()
    s.prepare(tmp_path, 'generation_plan.json', sha(tmp_path/'generation_plan.json'), private)
    assert c['plan_path'].read_bytes() == before


@pytest.mark.parametrize('change', ['draft', 'missing_job', 'answer', 'history', 'checkpoint', 'implementation', 'seal', 'seed'])
def test_generation_defects_fail_before_private_read(tmp_path, monkeypatch, change):
    c, private = fixture(tmp_path)
    if change=='draft':
        d=s.read(tmp_path/'declaration.json'); d['status']='DRAFT'; write(tmp_path/'declaration.json',d)
    elif change=='missing_job': next((c['top']/'primary/jobs').glob('*/complete.json')).unlink()
    elif change=='implementation': (tmp_path/'gearshift/core.py').write_text('changed')
    elif change=='seal':
        p=c['top']/'primary/generation_closure.json'; d=s.read(p); d['answer_count']-=1; write(p,d)
    elif change=='history': next((c['top']/'primary/tasks').glob('*/large_history/source_history.json')).write_text('{}')
    elif change=='seed':
        p=tmp_path/'seeds.json'; d=s.read(p); d['tasks'][c['declaration']['task_ids'][0]]['answers'][0]['answer_seed']+=1; write(p,d)
    else:
        p=next((c['top']/'primary/tasks').glob('*/*/seed_*/answer.json')); d=s.read(p)
        d['answer_text' if change=='answer' else 'checkpoint_sha256']='changed'; write(p,d)
    read_bytes=Path.read_bytes
    def watched(p):
        if p==private: pytest.fail('Private file read before valid generation closure')
        return read_bytes(p)
    monkeypatch.setattr(Path,'read_bytes',watched)
    with pytest.raises((ValueError,FileNotFoundError)): s.prepare(tmp_path,'generation_plan.json',sha(tmp_path/'generation_plan.json'),private)


@pytest.mark.parametrize('field,value', [('declaration_sha256','0'*64),('experiment_id','other'),('policy_identity_sha256','0'*64),('retry_policy','rerun_until_pass'),('generation_closure_sha256','0'*64),('manifest_path','elsewhere'),('hidden_tests_loaded_after_primary_generation',False)])
def test_plan_public_identity_tampering_rejected(tmp_path,field,value):
    c,_=prepared(tmp_path); p=c['plan']; p[field]=value; write(c['plan_path'],p)
    with pytest.raises(ValueError): s.validate_plan(tmp_path,str(c['plan_path'].relative_to(tmp_path)))


def test_private_bytes_change_rejected_and_unselected_invalid_spec_never_read(tmp_path):
    c,private=fixture(tmp_path,private_mutation=lambda x:x.update({'atcoder/untouched_reserve':{'not_a_test':'must remain untouched'}}))
    _,path=s.prepare(tmp_path,'generation_plan.json',sha(tmp_path/'generation_plan.json'),private)
    c=s.validate_plan(tmp_path,str(path.relative_to(tmp_path)))
    private.write_text(private.read_text()+' ')
    with pytest.raises(ValueError,match='bytes differ'): s.private_after_closure(c,private)


def test_oversized_trusted_expected_output_aborts_before_candidates(tmp_path,monkeypatch):
    c,private=fixture(tmp_path,private_mutation=lambda x:x['atcoder/task_0']['tests'][0].update(output='x'*(17*1024**2)))
    monkeypatch.setattr(s,'score',lambda *a,**kw:pytest.fail('Candidate execution during input validation'))
    with pytest.raises(ValueError,match='frozen supported allowance'):
        s.prepare(tmp_path,'generation_plan.json',sha(tmp_path/'generation_plan.json'),private)


def test_committed_failures_are_reused_without_reroll_or_raw_changes(tmp_path,monkeypatch):
    c,private=prepared(tmp_path); tests=s.private_after_closure(c,private); calls=fake_scores(c,monkeypatch)
    record=c['plan']['records'][0]; raw=(c['top']/record['answer_path']).read_bytes()
    first=s.score_record(c,record,tests,3,'first',lambda:None)
    second=s.score_record(c,record,tests,4,'replacement_host',lambda:None)
    assert first==second and len(calls)==1 and first['score_v2']['passed'] is False
    assert (c['top']/record['answer_path']).read_bytes()==raw
    assert calls[0][0]==s.extract(s.read(c['top']/record['answer_path'])['answer_text'])
    assert calls[0][2]['policy']==c['policy']


def test_interrupted_transaction_is_missing_and_never_reruns_candidate(tmp_path,monkeypatch):
    c,private=prepared(tmp_path); record=c['plan']['records'][0]
    def interrupt(*a,**kw): raise s.ScoringInterrupted()
    monkeypatch.setattr(s,'score',interrupt)
    with pytest.raises(s.ScoringInterrupted): s.score_record(c,record,s.private_after_closure(c,private),3,'first',lambda:None)
    monkeypatch.setattr(s,'score',lambda *a,**kw:pytest.fail('Interrupted candidate was rerun'))
    receipt=s.score_record(c,record,{},4,'recover',lambda:None)
    assert receipt['score_v2']['passed'] is None and receipt['score_v2']['missing'] is True
    assert receipt['score_v2']['reason']=='worker_interrupted_before_score_commit'


def test_finalization_requires_all_scores_and_retains_missing(tmp_path,monkeypatch):
    c,private=prepared(tmp_path); calls=complete_scores(c,private,monkeypatch)
    last=c['plan']['records'][-1]; path=c['top']/s.SCORING/'scores'/last['record_id']/'score.json'
    old=path.read_bytes(); path.unlink()
    with pytest.raises((ValueError,FileNotFoundError)): s.finalize(tmp_path,str(c['plan_path'].relative_to(tmp_path)))
    path.write_bytes(old); receipt=s.read(path); receipt['score_v2']=outcome(c,None); write(path,receipt)
    manifest=s.finalize(tmp_path,str(c['plan_path'].relative_to(tmp_path)))
    assert len(calls)==24 and manifest['missing_answers']==1 and len(manifest['files'])==24
    assert [(r['task_id'],r['condition'],r['seed_index']) for r in manifest['files']]==[(r['task_id'],r['condition'],r['seed_index']) for r in c['plan']['records']]
    assert manifest['files'][-1]['passed'] is None


@pytest.mark.parametrize('change',['binding','scorer','code','barrier','extra_transaction'])
def test_bad_score_receipt_cannot_finalize(tmp_path,monkeypatch,change):
    c,private=prepared(tmp_path);complete_scores(c,private,monkeypatch)
    record=c['plan']['records'][0];path=c['top']/s.SCORING/'scores'/record['record_id']/'score.json';receipt=s.read(path)
    if change=='binding':receipt['binding']['answer_sha256']='0'*64
    elif change=='scorer':receipt['score_v2']['scorer_identity']={}
    elif change=='code':receipt['code']+='\n# changed'
    elif change=='barrier':receipt['hidden_tests_loaded_after_primary_generation']=False
    else:(path.parent.parent/'unexpected').mkdir()
    write(path,receipt)
    with pytest.raises(ValueError):s.finalize(tmp_path,str(c['plan_path'].relative_to(tmp_path)))


def test_real_seal_to_score_to_report_works_without_private_tests_or_models(tmp_path,monkeypatch):
    c,private=prepared(tmp_path);complete_scores(c,private,monkeypatch)
    relative=str(c['plan_path'].relative_to(tmp_path));s.finalize(tmp_path,relative)
    private.unlink()
    result=reporter.report(relative,repo=tmp_path)
    assert result['task_clusters']==1 and result['answers']==24


def capacity_fixture(c,tmp_path,monkeypatch):
    lease={'pod_id':'synthetic_cpu','experiment_id':c['declaration']['experiment_id']}
    environment={'cpu_ids':[3,7],'physical_cores':[[0,3],[0,7]],'test_environment':True}
    monkeypatch.setattr(s,'cpu_environment',lambda cpus,policy:environment)
    cases=list(s.calibration.cases());records=[]
    for rep in range(3):
        for i,(name,key,inp,out) in enumerate(cases):
            records.append({'case':name,'reference':key,'reference_sha256':s.code_hash(s.calibration.REFERENCES[key]),
                'input_sha256':s.code_hash(inp),'expected_sha256':s.code_hash(out),'repetition':rep,'cpu_id':[3,7][i%2],
                'score':outcome(c,True)})
    p={'purpose':'confirmation_cpu_frozen_policy_preflight','experiment_id':c['declaration']['experiment_id'],
        'declaration_sha256':c['declaration_sha256'],'allocation_lease_sha256':'l'*64,'pod_id':lease['pod_id'],
        'policy_identity_sha256':digest(c['policy']),'scorer_identity_sha256':digest(c['scorer_identity']),
        'environment':environment,'calibration_implementation_sha256':sha(ROOT/'scripts/coding_scorer_repair_calibrate.py'),
        'passed':True,'policy_tuned':False,'private_tests_loaded':False,'records':records}
    path=tmp_path/'cpu_preflight.json';write(path,p)
    return lease,path,p


@pytest.mark.parametrize('change',[None,'other_host','tuned','missing_reference','changed_reference','slow','nan','negative','private_read'])
def test_cpu_preflight_enforces_same_host_complete_fixed_references_and_headroom(tmp_path,monkeypatch,change):
    c,_=prepared(tmp_path);lease,path,p=capacity_fixture(c,tmp_path,monkeypatch)
    if change=='other_host':p['environment']={'other':True}
    elif change=='tuned':p['policy_tuned']=True
    elif change=='missing_reference':p['records'].pop()
    elif change=='changed_reference':p['records'][0]['reference_sha256']='0'*64
    elif change in ('slow','nan','negative'):p['records'][0]['score']['receipts'][0]['cpu_seconds']={'slow':5.,'nan':float('nan'),'negative':-1.}[change]
    elif change=='private_read':p['private_tests_loaded']=True
    path.write_text(json.dumps(p))  # Also exercise rejection of hostile non-finite JSON.
    if change:
        with pytest.raises(ValueError):s.verify_preflight(c,lease,'l'*64,path,sha(path),[3,7])
    else:assert s.verify_preflight(c,lease,'l'*64,path,sha(path),[3,7])==p


def test_run_rejects_noncomparable_cpu_before_private_access(tmp_path,monkeypatch):
    c,_=prepared(tmp_path);lease,path,p=capacity_fixture(c,tmp_path,monkeypatch);p['environment']={'drift':True};write(path,p)
    monkeypatch.setattr(s,'allocation',lambda *a:(lease,lambda:None))
    monkeypatch.setattr(s,'private_after_closure',lambda *a:pytest.fail('Hidden tests loaded before comparable CPU verified'))
    with pytest.raises(ValueError,match='preflight environment'):
        s.run(tmp_path,str(c['plan_path'].relative_to(tmp_path)),'not_read','unused','l'*64,'cpu_preflight.json',sha(path),[3,7])


@pytest.mark.parametrize('failure', ['smt', 'quota', 'memory', 'too_many', 'unavailable'])
def test_cpu_environment_rejects_oversubscription(monkeypatch,failure):
    cpus=[2,4]; values={'/proc/meminfo':'MemTotal: 67108864 kB\n',
        '/sys/fs/cgroup/cpu.max':'200000 100000', '/sys/fs/cgroup/memory.max':str(64*1024**3)}
    for cpu in cpus:
        values[f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id']='0'
        values[f'/sys/devices/system/cpu/cpu{cpu}/topology/core_id']=str(cpu)
    if failure=='smt':values['/sys/devices/system/cpu/cpu4/topology/core_id']='2'
    elif failure=='quota':values['/sys/fs/cgroup/cpu.max']='100000 100000'
    elif failure=='memory':values['/sys/fs/cgroup/memory.max']=str(9*1024**3)
    elif failure=='too_many':cpus=list(range(9))
    else:cpus=[2,5]
    monkeypatch.setattr(s.sys,'platform','linux');monkeypatch.setattr(s.os,'geteuid',lambda:0)
    monkeypatch.setattr(s.os,'sched_getaffinity',lambda pid:{2,4},raising=False)
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**kw:values[str(p)])
    monkeypatch.setattr(Path,'exists',lambda p:str(p) in values)
    policy={'maximum_concurrent_candidates':8,'address_space_bytes':4*1024**3}
    with pytest.raises(ValueError):s.cpu_environment(cpus,policy)


def test_worker_interrupt_records_only_error_type_and_does_not_continue(tmp_path,monkeypatch):
    c,private=prepared(tmp_path);calls=[]
    monkeypatch.setattr(s.os,'sched_setaffinity',lambda *a:None,raising=False)
    monkeypatch.setattr(s.signal,'signal',lambda *a:None)
    def interrupt(*args):calls.append(args[1]['record_id']);raise s.ScoringInterrupted('sensitive diagnostic must not escape')
    monkeypatch.setattr(s,'score_record',interrupt)
    status=tmp_path/'worker_status.json'
    with pytest.raises(SystemExit):s._worker(c,c['plan']['records'],{},3,'attempt',lambda:None,status)
    receipt=s.read(status)
    assert len(calls)==1 and receipt['exception_type']=='ScoringInterrupted' and receipt['failed'] is True
    assert 'sensitive' not in status.read_text()


def test_candidate_timeout_does_not_get_an_outer_retry(tmp_path,monkeypatch):
    c,private=prepared(tmp_path);calls=[]
    def timeout(*a,**kw):
        calls.append(1); result=outcome(c,False);result['category']='cpu_timeout';return result
    monkeypatch.setattr(s,'score',timeout);record=c['plan']['records'][0]
    result=s.score_record(c,record,s.private_after_closure(c,private),3,'attempt',lambda:None)
    assert result['score_v2']['category']=='cpu_timeout' and calls==[1]


def test_declared_private_inventory_mismatch_is_rejected(tmp_path):
    c,private=prepared(tmp_path)
    # Match bytes but lie about the declared full-file key inventory.
    c['private_identity']['private_file_task_ids']=c['declaration']['task_ids']
    with pytest.raises(ValueError,match='key inventory'):s.private_after_closure(c,private)


def test_orphan_cleanup_is_exact_allocation_parent_and_process_identity_scoped(tmp_path,monkeypatch):
    root=tmp_path/'proc';root.mkdir();child=str(s.ROOT/'scripts/coding_sandbox_child_v2.py')
    for pid,ppid,script in [(21,1,child),(22,919,child),(23,1,'/unrelated/child.py'),(24,1,child)]:
        folder=root/str(pid);folder.mkdir()
        (folder/'cmdline').write_bytes(b'\0'.join(x.encode() for x in ['/usr/bin/python3','-I',script,'/tmp/x/payload.json','']))
        (folder/'status').write_text(f'Name: python3\nPPid: {ppid}\n')
    calls=[];monkeypatch.setenv('RUNPOD_POD_ID','reserved_cpu')
    monkeypatch.setattr(s.os,'pidfd_open',lambda pid,flags:pid+100,raising=False)
    monkeypatch.setattr(s.os,'getpgid',lambda pid:pid if pid!=24 else 2)
    monkeypatch.setattr(s.os,'close',lambda fd:None)
    monkeypatch.setattr(s.signal,'pidfd_send_signal',lambda fd,sig,info,flags:calls.append(fd),raising=False)
    assert s.cleanup_orphans({'gpu_count':0,'pod_id':'reserved_cpu'},root)==[21]
    assert calls==[121]  # Live scoring child, unrelated command and wrong session remain untouched.
    with pytest.raises(ValueError):s.cleanup_orphans({'gpu_count':0,'pod_id':'another_cpu'},root)
    with pytest.raises(ValueError):s.cleanup_orphans({'gpu_count':1,'pod_id':'reserved_cpu'},root)


def test_orphan_cleanup_rechecks_command_after_stable_pidfd_open(tmp_path,monkeypatch):
    root=tmp_path/'proc';folder=root/'31';folder.mkdir(parents=True)
    command=b'\0'.join(x.encode() for x in ['/usr/bin/python3','-I',str(s.ROOT/'scripts/coding_sandbox_child_v2.py'),'/tmp/x/payload.json',''])
    (folder/'cmdline').write_bytes(command);(folder/'status').write_text('PPid: 1\n')
    monkeypatch.setenv('RUNPOD_POD_ID','reserved_cpu')
    def open_fd(*args):(folder/'cmdline').write_bytes(b'/unrelated\0');return 131
    monkeypatch.setattr(s.os,'pidfd_open',open_fd,raising=False)
    monkeypatch.setattr(s.os,'close',lambda fd:None)
    monkeypatch.setattr(s.signal,'pidfd_send_signal',lambda *a:pytest.fail('Signalled a changed process identity'),raising=False)
    assert s.cleanup_orphans({'gpu_count':0,'pod_id':'reserved_cpu'},root)==[]
