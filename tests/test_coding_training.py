import copy,json,time,types
from pathlib import Path
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from gearshift.core import ModelBackend,CacheExtractor
from gearshift.coding_inference import Backend
from gearshift.coding_training import (make_schedule,schedule_audit,fit_ridge,DiskLRU,TrainingRuntime,
    StopState,nominations,choose_development_candidate,train_bounded,atomic_tensor,boundary_case,paired_samples)


def obj(tid,n):
    return {'task_id':tid,'source_history':{'prefix_ids':[1,2,3,4],'bridge_ids':[5]},
        'teacher_answer':{'answer_ids':[6+i%20 for i in range(n)]}}


def test_exact_short_answer_accumulation_is_shared_no_padding_or_shift():
    data=[obj('short',3),obj('medium',36),obj('long',520)]
    schedule=make_schedule(data,20260915,20)
    assert schedule==make_schedule(data,20260915,20)
    many=[obj(str(i),520) for i in range(16)]
    assert make_schedule(many,20260915,20)!=make_schedule(many,20260916,20)
    touched=set()
    for update in schedule:
        assert update['predictions']==32
        pairs=[]
        for anchor in [0,32,128,512]:
            segments=[s for s in update['segments'] if s['anchor']==anchor]
            assert sum(len(s['positions']) for s in segments)==8
            for s in segments:
                touched.add(s['task_id'])
                for p in s['positions']:
                    assert anchor<=p<anchor+8
                    assert p<len(next(o for o in data if o['task_id']==s['task_id'])['teacher_answer']['answer_ids'])
                    pairs.append((s['task_id'],p))
        assert len(pairs)==len(set(pairs))
    assert touched=={'short','medium','long'}
    audit=schedule_audit(data,schedule)
    assert audit['missing_window_positions']['512']==16
    with pytest.raises(ValueError,match='window'):make_schedule([obj('short',20)],1,1)


def test_ridge_matches_direct_regularized_solution_without_penalizing_bias(tmp_path):
    generator=torch.Generator().manual_seed(2)
    mapper=types.SimpleNamespace(weights=[torch.nn.Parameter(torch.zeros(3,2))],biases=[torch.nn.Parameter(torch.zeros(2))])
    x=torch.randn(64,3,generator=generator);y=x@torch.randn(3,2,generator=generator)+torch.tensor([4.,-7.])
    paths=[]
    for i in range(2):
        p=tmp_path/f'{i}.pt';torch.save({'source':[x[i*32:(i+1)*32]],'target':[y[i*32:(i+1)*32]],'split':'training'},p);paths.append(p)
    report=fit_ridge(mapper,paths)
    aug=torch.cat((x.double(),torch.ones(64,1)),1);penalty=torch.diag(torch.tensor([.01,.01,.01,0.],dtype=torch.float64))
    expected=torch.linalg.solve(aug.T@aug+penalty,aug.T@y.double())
    torch.testing.assert_close(mapper.weights[0],expected[:-1].float());torch.testing.assert_close(mapper.biases[0],expected[-1].float())
    assert report[0]['positions']==64
    bad=tmp_path/'bad.pt';torch.save({'source':[x],'target':[y],'split':'validation'},bad)
    with pytest.raises(ValueError,match='Validation'):fit_ridge(mapper,[bad])


def test_lru_evicts_and_never_keeps_oversized_cache(tmp_path):
    cache=DiskLRU(tmp_path,12000)
    assert cache.get('a',lambda:{'x':torch.zeros(1000)})['x'].numel()==1000
    cache.get('b',lambda:{'x':torch.zeros(1000)});cache.get('c',lambda:{'x':torch.zeros(1000)})
    assert sum(p.stat().st_size for p in tmp_path.glob('*.pt'))<=12000
    assert cache.events['evictions']>=1
    cache.get('big',lambda:{'x':torch.zeros(20000)})
    assert cache.events['uncached_oversized']==1


class TinyBackend(ModelBackend):
    def __init__(self,seed):
        torch.manual_seed(seed)
        config=Qwen3Config(vocab_size=40,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=1024)
        config._attn_implementation='sdpa'
        self.model=Qwen3ForCausalLM(config).eval().requires_grad_(False);self.config=config
        self.device='cpu';self.dtype=torch.float32;self.name=f'toy-{seed}';self.input_token_count=0
    prefill_chunked=Backend.prefill_chunked


class TinyMapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=torch.nn.ParameterList([torch.nn.Parameter(torch.eye(8)*.8) for _ in range(4)])
        self.biases=torch.nn.ParameterList([torch.nn.Parameter(torch.ones(8)*.01) for _ in range(4)])
    def forward(self,pairs):
        return tuple(tuple(CacheExtractor.unflatten(CacheExtractor.flatten(t)@self.weights[2*i+k]+self.biases[2*i+k],1,8)
            for k,t in enumerate(p)) for i,p in enumerate(pairs))


def test_full_gradient_both_objectives_freezes_models_preserves_prefix(tmp_path):
    torch.set_num_threads(2)
    source=TinyBackend(4);target=TinyBackend(9);initial=TinyMapper().state_dict();history=obj('full',520)
    item=make_schedule([history],1,1)[0]
    deltas=[]
    for objective in ['ordinary_continuation','natural_handoff_boundary']:
        mapper=TinyMapper();mapper.load_state_dict(initial)
        runtime=TrainingRuntime(source,target,mapper,DiskLRU(tmp_path/objective,1024**2))
        sp,tp=runtime.pair(history);snapshot=[[x.clone() for x in p] for p in sp]
        optimizer=torch.optim.AdamW(mapper.parameters(),lr=1e-5,weight_decay=0)
        result=runtime.train_update(item,{'full':history},objective,optimizer)
        assert result['gradient_predictions']==32 and result['cache_gradients_finite_nonzero']
        assert result['gradient_norm_before_clipping']>0
        assert all(p.grad is None for b in [source,target] for p in b.model.parameters())
        for p,s in zip(sp,snapshot):
            for x,y in zip(p,s):torch.testing.assert_close(x,y,rtol=0,atol=0)
        deltas.append(torch.cat([(mapper.state_dict()[k]-initial[k]).flatten() for k in initial]))
        validation=runtime.validate([history]);assert validation['rows'][0]['predictions']==32
        assert validation['validation_kl']>=0
    assert not torch.equal(deltas[0],deltas[1])


def test_selection_and_plateau_are_bounded():
    stop=StopState();stop.observe(1.)
    for x in [.998,.997,.996,.9951]:stop.observe(x)
    assert not stop.plateau(128) and stop.plateau(256)
    stop.observe(.9949);assert stop.stale==0
    curve=[{'step':n,'validation_kl':v,'checkpoint':{'path':str(n)}} for n,v in [(128,.01),(256,.2),(384,.1),(512,.1)]]
    assert [r['step'] for r in nominations(curve)]==[384,512]
    rows=[dict(candidate_id='a',passes=20,source_inclusive_seconds=100,step=384,task_count=40),
          dict(candidate_id='b',passes=20,source_inclusive_seconds=100,step=256,task_count=40)]
    assert choose_development_candidate(rows)['candidate_id']=='b'
    rows[1]['task_count']=39
    with pytest.raises(ValueError):choose_development_candidate(rows)


def test_mocked_bounded_worker_retains_capped_noneligible_checkpoint(tmp_path):
    class Runtime:
        def __init__(self):self.mapper=TinyMapper();self.guard=lambda:None;self.calls=[]
        def train_update(self,item,lookup,objective,optimizer):
            self.guard();self.calls.append(item);return {'gradient_predictions':32,'kl':1.}
        def validate(self,rows):return {'validation_kl':1.,'rows':[]}
    runtime=Runtime();data=[obj('t',520)]
    result=train_bounded(runtime,data,data,'natural_handoff_boundary',20260915,tmp_path,time.time()+100,
        {'initialization_sha256':'fixed'},max_updates=2,validation_every=1)
    assert result['steps']==2 and result['capped_unconverged'] and result['no_eligible_checkpoint']
    assert result['training_predictions']==64
    checkpoint=torch.load(tmp_path/'mapper_step_0002.pt',weights_only=True)
    assert checkpoint['optimizer_resume_supported'] is False
    with pytest.raises(ValueError,match='silently restart'):train_bounded(runtime,data,data,'natural_handoff_boundary',1,tmp_path,time.time()+100,{'initialization_sha256':'fixed'})


def test_time_ceiling_wins_even_before_minimum_updates(tmp_path):
    class Runtime:
        mapper=TinyMapper()
        guard=lambda self:None
        def train_update(self,*args):raise AssertionError('Expired run cannot dispatch')
    result=train_bounded(Runtime(),[obj('t',520)],[obj('t',520)],'ordinary_continuation',1,tmp_path,9.,
        {'initialization_sha256':'fixed'},clock=lambda:10.)
    assert result['steps']==0 and result['stop_reason']=='time_cap' and result['no_eligible_checkpoint']


def test_paired_feature_sampling_unrotates_using_original_absolute_positions():
    source=TinyBackend(4);target=TinyBackend(9);ids=[1+i%20 for i in range(80)]
    with torch.no_grad():
        sp=CacheExtractor.tensors(source.prefill_chunked(ids).past_key_values)
        tp=CacheExtractor.tensors(target.prefill_chunked(ids).past_key_values)
        features=paired_samples(source,target,sp,tp,positions=8)
        expected=CacheExtractor.flatten(source.rope(sp[0][0],inverse=True))[features['positions']]
    torch.testing.assert_close(features['source'][0],expected)
    assert features['positions'][0]==0 and features['positions'][-1]==79
    assert features['source_layers']==[0,1]


def test_complete_mock_history_worker_preserves_source_streams_and_separate_feature_manifest(tmp_path,monkeypatch):
    import importlib.util,threading
    from gearshift.coding_control import write,digest,sha
    import gearshift.coding_inference as inference
    spec=importlib.util.spec_from_file_location('history_worker',Path(__file__).parents[1]/'scripts/coding_history_worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    task={'task_id':'training/t','prompt_ids':[1,2]}
    write(tmp_path/'data/coding_pilot_v1/visible/training.json',[task])
    write(tmp_path/'data/coding_pilot_v1/visible/validation.json',[])
    baseline=tmp_path/'baseline';write(baseline/'identity.json',{'reasoning_cap':24576})
    source=TinyBackend(2);target=TinyBackend(3)
    calls=[]
    def reason(backend,prompt,tid,stream,cap,callback):
        calls.append(('reason',tid,stream,cap));callback([151668])
        h={'task_id':tid,'prompt_ids':prompt,'reasoning_ids':[151668],'prefix_ids':prompt,'bridge_ids':[151668],
            'natural_boundary':True,'reasoning_capped':False,'early_eos':False,'prefix_cache_length':len(prompt)}
        with torch.no_grad():cache=backend.prefill_chunked(prompt).past_key_values
        return h,cache
    def answer(backend,h,cache,tid,stream,cap,callback):
        calls.append(('answer',tid,stream,cap));callback([8,151645])
        return {'task_id':tid,'answer_ids':[8,151645],'answer_text':'x','answer_capped':False}
    monkeypatch.setattr(inference,'reason',reason);monkeypatch.setattr(inference,'answer',answer)
    thread=threading.Thread(target=lambda:None);thread.start();published=[]
    root=tmp_path/'results/coding_pilot_v1/test/worker';root.mkdir(parents=True)
    identity={'tag':'mock'}
    context={'spec':{'stage':'histories','task_ids':['training/t'],'baseline_root':str(baseline)},'root':root,
        'guard':lambda:None,'publish':lambda **kw:published.append(kw),'identity':identity,'halt':threading.Event(),'thread':thread}
    complete=module.run(context,(source,target))
    assert complete['task_count']==1 and published[-1]['state']=='complete'
    assert calls==[('reason','training/t','source_reasoning',24576),('answer','training/t','answer_large',4096)]
    folder=root/'tasks/training__t';receipt=json.loads((folder/'complete.json').read_text())
    assert 'paired_features.pt' not in receipt['files']
    assert receipt['feature_files']['paired_features.pt']==sha(folder/'paired_features.pt')
    feature_manifest=json.loads((root/'feature_manifest.json').read_text())
    assert len(feature_manifest['files'])==1
    assert complete['tasks']['training/t']==sha(folder/'complete.json')


def test_partial_accumulation_window_keeps_declared_handoff_anchor(tmp_path):
    from gearshift.coding_training import ordinary_anchor
    assert ordinary_anchor([4,5,6,7])==0
    assert ordinary_anchor([515,516])==512
    with pytest.raises(ValueError):ordinary_anchor([7,32])
    runtime=TrainingRuntime(TinyBackend(4),TinyBackend(9),TinyMapper(),DiskLRU(tmp_path,1024**2))
    history=obj('partial',520);sp,tp=runtime.pair(history)
    esp,etp=runtime.extended_pair(history,[4,5],sp,tp)
    assert esp[0][0].shape[-2]==4 and etp[0][0].shape[-2]==4


def test_worker_rejects_old_or_diagnostic_memory_gate(tmp_path):
    import time
    from gearshift.coding_control import write,sha
    from gearshift.coding_training import worker_context,finish_worker
    repo=tmp_path/'repo';script=repo/'scripts/coding_memory_preflight.py';script.parent.mkdir(parents=True)
    script.write_text("def check_baselines(path):\n    return {'baseline_identity_file_sha256':'base'}\n")
    files=['gearshift/core.py','gearshift/coding_inference.py','gearshift/coding_gradients.py','gearshift/coding_training.py',
        'scripts/coding_history_worker.py','scripts/coding_training_worker.py']
    for f in files:
        p=repo/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# test fixture\n')
    protocol=repo/'configs/coding_pilot_v1/training_protocol_v1.json';write(protocol,{'fixture':True})
    write(repo/'configs/coding_pilot_v1/pilot.json',{})
    write(repo/'data/coding_pilot_v1/identity.json',{})
    memory=repo/'memory/memory_gate.json'
    write(memory.parent/'identity.json',{'baseline_identity_file_sha256':'base','maximum_total_context':40960,
        'training_protocol_sha256':sha(protocol),'implementation':{f:sha(repo/f) for f in files}})
    approval=repo/'approval.json';write(approval,{'approved':True})
    allocation=repo/'allocation.json';write(allocation,{'approval_sha256':sha(approval),'deadline_epoch':time.time()+1000})
    spec=repo/'worker/spec.json';write(spec,{'result_root':str(repo/'results/coding_pilot_v1/test'),'baseline_root':str(repo/'baseline'),
        'memory_gate':str(memory),'approval_path':str(approval),'allocation_path':str(allocation),'deadline_epoch':time.time()+500,
        'stage_identity':'stage','worker_id':'worker'})
    write(memory,{'passed':True,'gradient_predictions':32})
    with pytest.raises(ValueError,match='training-runtime'):worker_context(repo,spec)
    good={'passed':True,'training_ready':True,'actual_training_runtime':True,
        'objectives':{name:{'gradient_predictions':32} for name in ['ordinary_continuation','natural_handoff_boundary']}}
    write(memory,{**good,'source_weights_on_cpu':True})
    with pytest.raises(ValueError,match='offload'):worker_context(repo,spec)
    write(memory,good);context=worker_context(repo,spec);finish_worker(context,'complete')


def test_cpu_owned_extension_matches_prior_cache_and_update_exactly(tmp_path):
    class PriorRuntime(TrainingRuntime):
        def extended_pair_from_cpu(self,obj,positions,source_cpu,target_cpu):
            sp=tuple(tuple(t.to(self.receiver.device) for t in p) for p in source_cpu)
            tp=tuple(tuple(t.to(self.receiver.device) for t in p) for p in target_cpu)
            return self.extended_pair(obj,positions,sp,tp)
    source,target=TinyBackend(4),TinyBackend(9)
    history=obj('equivalence',520);initial=TinyMapper().state_dict()
    fresh=TrainingRuntime(source,target,TinyMapper(),DiskLRU(tmp_path/'fresh',1024**2))
    old=PriorRuntime(source,target,TinyMapper(),DiskLRU(tmp_path/'old',1024**2))
    sp,tp=fresh.pair(history,device='cpu')
    before=tuple(tuple(t.clone() for t in p) for p in (*sp,*tp))
    for anchor in (0,32,128,512):
        positions=list(range(anchor,anchor+8))
        expected=old.extended_pair_from_cpu(history,positions,sp,tp)
        actual=fresh.extended_pair_from_cpu(history,positions,sp,tp)
        for ap,ep in zip(actual,expected):
            for aa,ee in zip(ap,ep):
                for a,e in zip(aa,ee):torch.testing.assert_close(a,e,rtol=0,atol=0)
    for actual,expected in zip((*sp,*tp),before):
        for a,e in zip(actual,expected):torch.testing.assert_close(a,e,rtol=0,atol=0)
    item=make_schedule([history],1,1)[0];results=[];states=[]
    for runtime in (old,fresh):
        runtime.mapper.load_state_dict(initial)
        optimizer=torch.optim.AdamW(runtime.mapper.parameters(),lr=1e-5,weight_decay=0)
        results.append(runtime.train_update(item,{'equivalence':history},'ordinary_continuation',optimizer))
        states.append({k:v.clone() for k,v in runtime.mapper.state_dict().items()})
    assert results[0]==results[1]
    for k in states[0]:torch.testing.assert_close(states[0][k],states[1][k],rtol=0,atol=0)


def test_owned_extension_releases_replaced_device_tensors(monkeypatch):
    import weakref
    from types import SimpleNamespace
    history=obj('lifetime',520)
    sp=tuple((torch.zeros(1,1,4,8),torch.zeros(1,1,4,8)) for _ in range(2))
    tp=tuple((torch.zeros(1,1,4,8),torch.zeros(1,1,4,8)) for _ in range(2))
    base_ids={id(t) for pair in (*sp,*tp) for t in pair};original_to=torch.Tensor.to
    transferred=[]
    def device_copy(t,*args,**kwargs):
        result=original_to(t,*args,**kwargs)
        if id(t) in base_ids:
            result=result.clone();transferred.append(weakref.ref(result))
        return result
    monkeypatch.setattr(torch.Tensor,'to',device_copy)
    retained=[]
    def forward(ids,cache):
        for layer in range(len(cache.layers)):
            cache.update(torch.zeros(1,1,len(ids),8),torch.zeros(1,1,len(ids),8),layer)
        retained.append(sum(ref() is not None for ref in transferred))
        return SimpleNamespace(past_key_values=cache)
    backend=SimpleNamespace(device='cpu',forward=forward)
    runtime=TrainingRuntime(backend,backend,None,None)
    # Reproduce the original ownership pattern without allocating a real GPU.
    device_sp=tuple(tuple(t.to('cpu') for t in p) for p in sp)
    device_tp=tuple(tuple(t.to('cpu') for t in p) for p in tp)
    runtime.extended_pair(history,[32,33],device_sp,device_tp)
    assert retained==[8,8]
    del device_sp,device_tp
    retained.clear();transferred.clear()
    runtime.extended_pair_from_cpu(history,[32,33],sp,tp)
    assert retained==[0,0]


def test_owned_extension_rejects_gradient_bearing_base_cache():
    runtime=TrainingRuntime(TinyBackend(4),TinyBackend(9),TinyMapper(),None)
    bad=((torch.zeros(1,1,4,8,requires_grad=True),torch.zeros(1,1,4,8)),)*2
    with pytest.raises(ValueError,match='detached CPU'):
        runtime.extended_pair_from_cpu(obj('invalid',520),[32],bad,bad)
