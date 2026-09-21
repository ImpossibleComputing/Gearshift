import importlib.util,json
from pathlib import Path
import pytest
from gearshift.coding_control import write,sha,digest
from gearshift.coding_reuse import NUMERICAL_FILES

def fixture(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/coding_memory_preflight.py'
    spec=importlib.util.spec_from_file_location('memory_gate_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'ROOT',tmp_path)
    write(tmp_path/'configs/coding_pilot_v1/pilot.json',{'test_configuration':True})
    write(tmp_path/'data/coding_pilot_v1/visible/development.json',[{'task_id':f't{i}'} for i in range(40)])
    protocol=tmp_path/'configs/coding_pilot_v1/control_protocol_v2.json';write(protocol,{'v':2})
    write(tmp_path/'data/coding_pilot_v1/identity.json',{'fixed':True})
    for name in NUMERICAL_FILES:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original numerical code')
    base=tmp_path/'baseline';identity={'config_sha256':sha(tmp_path/'configs/coding_pilot_v1/pilot.json'),
        'control_protocol_sha256':sha(protocol),'data_identity':{'fixed':True},
        'implementation':{name:sha(tmp_path/name) for name in NUMERICAL_FILES}}
    write(base/'identity.json',identity);write(base/'native_gate.json',{'passed':True,'identity_sha256':digest(identity)})
    write(base/'baseline_gate.json',{'passed':True,'cap_gate':True,'counts':{'A':16,'B':8,'D':12}})
    for i in range(40):
        folder=base/f'tasks/t{i}';passes={'A':i<16,'B':i<8,'D':i<12}
        for name in ['source_history','small_history','A','B','D']:
            write(folder/(name+'.json'),{'task_id':f't{i}','score':{'passed':passes.get(name,False)},'reasoning_capped':False})
        write(folder/'complete.json',{'task_id':f't{i}','identity_sha256':digest(identity),'files':{p.name:sha(p) for p in folder.glob('*.json')},
            'pass':passes,'source_capped':False,'small_capped':False})
    return m,base

def test_gate_requires_fixed_cohort_and_matching_transactions(tmp_path,monkeypatch):
    m,base=fixture(tmp_path,monkeypatch);assert m.check_baselines(base)['task_count']==40
    p=base/'tasks/t0/complete.json';row=json.loads(p.read_text());row['task_id']='different_task';write(p,row)
    with pytest.raises(ValueError,match='identity|membership'):m.check_baselines(base)

def test_gate_recomputes_outcomes_instead_of_trusting_pass_flag(tmp_path,monkeypatch):
    m,base=fixture(tmp_path,monkeypatch)
    p=base/'tasks/t0/complete.json';row=json.loads(p.read_text());row['pass']['A']=False;write(p,row)
    with pytest.raises(ValueError,match='score|operational'):m.check_baselines(base)

def test_gate_rejects_mixed_experiment_identity(tmp_path,monkeypatch):
    m,base=fixture(tmp_path,monkeypatch)
    p=base/'tasks/t0/complete.json';row=json.loads(p.read_text());row['identity_sha256']='another_run';write(p,row)
    with pytest.raises(ValueError,match='identity'):m.check_baselines(base)


def test_gate_rejects_changed_generation_and_false_cap_flags(tmp_path,monkeypatch):
    m,base=fixture(tmp_path,monkeypatch)
    p=base/'tasks/t0/source_history.json';h=json.loads(p.read_text());h['reasoning_capped']=True;write(p,h)
    c=base/'tasks/t0/complete.json';row=json.loads(c.read_text());row['files'][p.name]=sha(p);write(c,row)
    with pytest.raises(ValueError,match='cap flags'):m.check_baselines(base)
    (tmp_path/NUMERICAL_FILES[0]).write_text('different numerical code')
    with pytest.raises(ValueError,match='Numerical'):m.check_baselines(base)


def test_memory_deadline_requires_bound_replacement_approval_and_remaining_usage(tmp_path,monkeypatch):
    m,_=fixture(tmp_path,monkeypatch)
    approval=tmp_path/'approval.json';allocation=tmp_path/'allocation.json'
    write(approval,{'approved':True,'scope':'coding_pilot_v1_experiment','gpu_hours':100,'usd_cap':1000,
        'cleanup_reserve_usd':20,'owner_instruction':'Approved','limits_apply_cumulatively':True,
        'supersedes_stage_budget_limits':True,'one_gpu_only':True,'gpu_price_ceiling_usd':5.5,
        'scientific_scope_unchanged':True})
    a={'approval_sha256':sha(approval),'cap_usd':1000,'preflight_deadline_epoch':4600,
       'remaining_authorized_seconds':3600,'started_epoch':1000,'prior_usage':{'gpu_hours':99}}
    write(allocation,a);assert m.approved_deadline(allocation,approval,1000)[1]==4480
    with pytest.raises(ValueError,match='expired'):m.approved_deadline(allocation,approval,4600)
    write(allocation,{**a,'remaining_authorized_seconds':7200})
    with pytest.raises(ValueError,match='cumulative'):m.approved_deadline(allocation,approval,1000)
    write(allocation,{**a,'approval_sha256':'stale'})
    with pytest.raises(ValueError,match='bind'):m.approved_deadline(allocation,approval,1000)


def load_memory_module():
    path=Path(__file__).resolve().parents[1]/'scripts/coding_memory_preflight.py'
    spec=importlib.util.spec_from_file_location('memory_runtime_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_stress_case_uses_full_context_and_exact_prediction_windows():
    from gearshift.coding_training import boundary_case
    m=load_memory_module();obj=m.stress_case([1,2,3])
    positions=[j for a in [0,32,128,512] for j in range(a,a+8)]
    ids,pos=boundary_case(obj,positions)
    assert len(obj['source_history']['prefix_ids'])==40440
    assert len(ids)==520 and len(pos)==32 and pos==positions
    assert ids[0]==151668 and ids[1:]==obj['teacher_answer']['answer_ids'][:519]
    assert len(obj['source_history']['prefix_ids'])+len(ids)==40960


def test_monitor_errors_cannot_be_hidden_by_a_stale_minimum():
    from types import SimpleNamespace
    m=load_memory_module()
    monitor=m.Monitor(SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda:(20*1024**3,100*1024**3))))
    monitor.sample();monitor.errors.append(RuntimeError('sampler died'))
    with pytest.raises(RuntimeError,match='sampler died'):monitor.sample()
    with pytest.raises(RuntimeError,match='sampler died'):monitor.stop()


def test_monitor_final_sample_failure_prevents_success():
    from types import SimpleNamespace
    m=load_memory_module();calls=[]
    def sample():
        calls.append(True)
        if len(calls)>1:raise RuntimeError('final sample failed')
        return 20*1024**3,100*1024**3
    monitor=m.Monitor(SimpleNamespace(cuda=SimpleNamespace(mem_get_info=sample)))
    monitor.sample()
    with pytest.raises(RuntimeError,match='final sample failed'):monitor.stop()


def tiny_runtime(objective):
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from gearshift.coding_training import TrainingRuntime
    from gearshift.core import ModelBackend,CacheExtractor
    m=load_memory_module();torch.manual_seed(19)
    config=Qwen3Config(vocab_size=96,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=1024)
    config._attn_implementation='sdpa'
    def backend():
        b=object.__new__(ModelBackend);b.name='tiny';b.device='cpu';b.dtype=torch.float32;b.input_token_count=0
        b.model=Qwen3ForCausalLM(config).eval().requires_grad_(False);b.config=b.model.config
        return b
    source,receiver=backend(),backend();obj=m.stress_case([1,2,3,4,5],maximum_total_context=528,boundary_token=6)
    with torch.no_grad():
        so=source.forward(obj['source_history']['prefix_ids']);sp=CacheExtractor.tensors(so.past_key_values)
        to=receiver.forward(obj['source_history']['prefix_ids']);tp=CacheExtractor.tensors(to.past_key_values)
    class TinyMapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights=torch.nn.ParameterList([torch.nn.Parameter(torch.eye(16)+.01*torch.randn(16,16)) for _ in range(4)])
            self.biases=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(16)) for _ in range(4)])
        def forward(self,pairs):
            return tuple(tuple(CacheExtractor.unflatten(CacheExtractor.flatten(t)@self.weights[2*i+k]+self.biases[2*i+k],2,8)
                for k,t in enumerate(pair)) for i,pair in enumerate(pairs))
    fixed=m.FixedPair(obj['source_history']['prefix_ids'],sp,tp)
    runtime=TrainingRuntime(source,receiver,TinyMapper(),fixed)
    return m,runtime,obj,sp,tp


@pytest.mark.parametrize('objective',['natural_handoff_boundary','ordinary_continuation'])
def test_probe_exercises_real_runtime_gradients_optimizer_and_validation(objective):
    import torch
    m,runtime,obj,sp,tp=tiny_runtime(objective)
    original=[t.clone() for pair in (*sp,*tp) for t in pair]
    records=[]
    result=m.exercise_objective(runtime,obj,objective,lambda stage,**values:records.append(stage))
    assert result['gradient_predictions']==32 and result['mapper_gradient_tensors']==8
    assert len(result['cache_gradients'])==(16 if objective=='ordinary_continuation' else 4)
    assert result['optimizer_parameters_updated'] and result['cache_gradients_finite_nonzero']
    assert result['validation']['rows'][0]['predictions']==32
    assert objective+'_backward' in records and objective+'_optimizer_step' in records
    assert all(p.grad is None for b in [runtime.source,runtime.receiver] for p in b.model.parameters())
    for old,t in zip(original,[t for pair in (*sp,*tp) for t in pair]):torch.testing.assert_close(old,t,atol=0,rtol=0)


def test_fixed_stress_pair_refuses_gradient_bearing_or_wrong_prefix_cache():
    import torch
    m=load_memory_module();tensor=torch.ones((1,1,2,2),requires_grad=True)
    with pytest.raises(ValueError,match='CPU tensors only'):m.FixedPair([1,2],((tensor,tensor),),((tensor,tensor),))
    pair=((tensor.detach(),tensor.detach()),);fixed=m.FixedPair([1,2],pair,pair)
    with pytest.raises(ValueError,match='prefix changed'):fixed.get({'history':'changed'},lambda:None)


def test_parallel_memory_deadline_binds_both_approval_receipts(tmp_path,monkeypatch):
    from gearshift.coding_control import PARALLEL_PARENT_SHA256
    from test_coding_control import experiment_approval,parallel_approval
    m=load_memory_module();old=tmp_path/'old.json';approval=tmp_path/'new.json';allocation=tmp_path/'allocation.json'
    write(old,experiment_approval());write(approval,parallel_approval())
    original_sha=m.sha
    monkeypatch.setattr(m,'sha',lambda p:PARALLEL_PARENT_SHA256 if Path(p)==old else original_sha(p))
    a={'approval_sha256':sha(approval),'cap_usd':1000,'preflight_deadline_epoch':4600,
       'remaining_authorized_seconds':3600,'started_epoch':1000,'prior_usage':{'gpu_hours':499}}
    write(allocation,a)
    assert m.approved_deadline(allocation,approval,1000,previous_approval_path=old)[1]==4480
    changed=json.loads(approval.read_text());changed['previous_experiment_approval_sha256']='changed';write(approval,changed)
    write(allocation,{**a,'approval_sha256':sha(approval)})
    with pytest.raises(ValueError,match='bind'):m.approved_deadline(allocation,approval,1000,previous_approval_path=old)
