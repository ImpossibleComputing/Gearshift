"""Frozen coding-history training with exact windows and bounded, auditable state.

No private tests are imported here. The two objectives use identical source answer
positions; only the point where the historical cache is mapped differs.
"""
from __future__ import annotations
import gc, json, math, os, random, tempfile, time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import torch
from .coding_control import bind, digest, sha, write
from .coding_gradients import continuation
from .core import CacheExtractor, CacheInjector

OFFSETS=(0,32,128,512)
OBJECTIVES=('ordinary_continuation','natural_handoff_boundary')


def read(path):
    return json.loads(Path(path).read_text())


def atomic_tensor(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('Immutable tensor destination already exists')
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temporary.open('wb') as f:
            torch.save(obj,f);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha(path)


def safe_id(task_id):
    if not isinstance(task_id,str) or not task_id or any(x in task_id for x in ['..','\\','\x00']):
        raise ValueError('Invalid task identifier')
    return task_id.replace('/','__')


def valid_positions(answer,anchor):
    """Include the EOS prediction, never a position after the saved answer."""
    return list(range(anchor,min(anchor+8,len(answer))))


def validate_history(obj):
    h=obj['source_history'];a=obj['teacher_answer']
    if h['task_id']!=obj['task_id'] or a['task_id']!=obj['task_id']:raise ValueError('History task differs')
    if not a['answer_ids']:raise ValueError('A saved answer must contain a prediction')
    if h['bridge_ids']!=[151668]:raise ValueError('Only the declared natural closing-think bridge is permitted')
    expected=h['prompt_ids']+(h['reasoning_ids'][:-1] if h['natural_boundary'] else h['reasoning_ids'])
    if h['prefix_ids']!=expected or h['prefix_cache_length']!=len(expected):raise ValueError('Historical prefix differs')
    if h['natural_boundary'] and h['reasoning_ids'][-1]!=151668:raise ValueError('Natural boundary missing')
    if len(expected)+1+len(a['answer_ids'])>40960:raise ValueError('Complete history exceeds declared context')
    eos=[i for i,t in enumerate(a['answer_ids']) if t in [151643,151645]]
    if eos and eos!=[len(a['answer_ids'])-1]:raise ValueError('Saved answer includes post-EOS tokens')
    return obj


def make_schedule(histories,seed,max_updates=2048,offsets=OFFSETS):
    """Every update contains exactly 8 real predictions per declared window.

    Each bucket walks the same seeded permutation of all training histories. A
    short history contributes its available positions, then the next fixed
    history supplies the remainder. No anchor is shifted and no task is dropped.
    """
    byid={o['task_id']:o for o in histories}
    if len(byid)!=len(histories) or not byid:raise ValueError('Duplicate or empty training membership')
    order=sorted(byid);random.Random(seed).shuffle(order)
    availability={anchor:{tid:valid_positions(byid[tid]['teacher_answer']['answer_ids'],anchor) for tid in order} for anchor in offsets}
    for anchor,rows in availability.items():
        if sum(map(len,rows.values()))<8:raise ValueError(f'Fewer than eight distinct actual positions exist at window {anchor}')
    cursors={anchor:[0,0] for anchor in offsets};schedule=[]
    for step in range(1,max_updates+1):
        segments=OrderedDict()
        for anchor in offsets:
            taken=0;seen=set()
            while taken<8:
                index,within=cursors[anchor];tid=order[index%len(order)];positions=availability[anchor][tid]
                if within>=len(positions):cursors[anchor]=[index+1,0];continue
                position=positions[within];cursors[anchor]=[index,within+1]
                if (tid,position) in seen:continue
                seen.add((tid,position));segments.setdefault((tid,anchor),[]).append(position);taken+=1
        parts=[{'task_id':tid,'anchor':anchor,'positions':positions} for (tid,anchor),positions in segments.items()]
        if sum(len(p['positions']) for p in parts)!=8*len(offsets):raise AssertionError('Gradient-bearing count changed')
        schedule.append({'step':step,'segments':parts,'predictions':8*len(offsets)})
    return schedule


def schedule_audit(histories,schedule):
    return {'updates':len(schedule),'predictions_per_update':32,'schedule_sha256':digest(schedule),
        'training_task_ids':sorted(o['task_id'] for o in histories),
        'missing_window_positions':{str(a):sum(8-len(valid_positions(o['teacher_answer']['answer_ids'],a)) for o in histories) for a in OFFSETS},
        'examples_per_update':[len({p['task_id'] for p in s['segments']}) for s in schedule],
        'policy':'Exact positions; deterministic accumulation over additional histories; no shifted anchors, duplicate positions, padding or post-EOS tokens.'}


def boundary_case(obj,positions):
    if not positions or min(positions)<0 or max(positions)>=len(obj['teacher_answer']['answer_ids']):raise ValueError('Invalid actual answer position')
    inputs=obj['source_history']['bridge_ids']+obj['teacher_answer']['answer_ids'][:-1]
    return inputs[:max(positions)+1],list(positions)


def ordinary_anchor(positions):
    matches=[a for a in OFFSETS if all(a<=p<a+8 for p in positions)]
    if len(matches)!=1:raise ValueError('Ordinary segments must stay within one declared window')
    return matches[0]


def paired_samples(source,receiver,sp,tp,positions=64):
    """Unrotate at actual absolute positions before sampling fresh fit features."""
    n=sp[0][0].shape[-2]
    if tp[0][0].shape[-2]!=n or n<1:raise ValueError('Paired prefix lengths differ')
    idx=torch.linspace(0,n-1,min(positions,n),dtype=torch.float64).round().long().unique()
    alignment=[round(i*(len(sp)-1)/(len(tp)-1)) if len(tp)>1 else 0 for i in range(len(tp))]
    xs=[];ys=[]
    for layer,si in enumerate(alignment):
        for kind in range(2):
            x=sp[si][kind];y=tp[layer][kind]
            local=idx.to(x.device)
            x=x.index_select(-2,local);y=y.index_select(-2,local.to(y.device))
            if kind==0:
                x=source.rope(x,positions=local,inverse=True)
                y=receiver.rope(y,positions=local.to(y.device),inverse=True)
            xs.append(CacheExtractor.flatten(x).float().cpu());ys.append(CacheExtractor.flatten(y).float().cpu())
    return {'source':xs,'target':ys,'positions':idx.tolist(),'prefix_tokens':n,'source_layers':alignment,
        'key_representation':'inverse source/receiver RoPE at original absolute positions','split':'training'}


def fit_ridge(mapper,feature_paths,ridge=.01,guard=lambda:None,progress=lambda **kw:None):
    """One streaming pass into FP64 sufficient statistics; bias unpenalized.

    All72 parameter-pair statistics consume about1.2GiB for the approved geometry.
    No full cache or corpus feature tensor is retained in memory.
    """
    if ridge!=.01:raise ValueError('The declared ridge coefficient is0.01')
    reports=[];device=mapper.weights[0].device
    statistics=[]
    for weight in mapper.weights:
        d,out=weight.shape
        statistics.append((torch.zeros((d+1,d+1),dtype=torch.float64,device=device),
            torch.zeros((d+1,out),dtype=torch.float64,device=device)))
    count=0
    for index,path in enumerate(feature_paths):
        guard();obj=torch.load(path,map_location='cpu',weights_only=True)
        if obj['split']!='training':raise ValueError('Validation features cannot fit initialization')
        lengths=set()
        for layer,(weight,(xx,xy)) in enumerate(zip(mapper.weights,statistics)):
            d,out=weight.shape
            x=obj['source'][layer].to(device,dtype=torch.float64);y=obj['target'][layer].to(device,dtype=torch.float64)
            if not 1<=x.shape[0]<=64 or x.shape[0]!=y.shape[0] or x.shape[1]!=d or y.shape[1]!=out:raise ValueError('Paired feature geometry differs')
            lengths.add(len(x));x=torch.cat((x,torch.ones((len(x),1),dtype=torch.float64,device=device)),1)
            xx.addmm_(x.T,x);xy.addmm_(x.T,y);del x,y
        if len(lengths)!=1:raise ValueError('Paired layers have different sampled positions')
        count+=lengths.pop();del obj
        progress(stage='ridge_accumulation',completed_histories=index+1,total_histories=len(feature_paths))
    if not count:raise ValueError('No training fit positions')
    for layer,(weight,bias,(xx,xy)) in enumerate(zip(mapper.weights,mapper.biases,statistics)):
        guard();d,out=weight.shape
        penalty=torch.eye(d+1,dtype=torch.float64,device=device)*ridge;penalty[-1,-1]=0
        solution=torch.linalg.solve(xx+penalty,xy)
        if not torch.isfinite(solution).all():raise FloatingPointError('Nonfinite ridge initialization')
        with torch.no_grad():weight.copy_(solution[:-1].float());bias.copy_(solution[-1].float())
        residual=float(((xx+penalty)@solution-xy).abs().max())
        reports.append({'parameter_pair':layer,'positions':count,'normal_equation_max_residual':residual})
        progress(stage='ridge_fit',completed_parameter_pairs=layer+1,total_parameter_pairs=len(mapper.weights))
        statistics[layer]=None;del xx,xy,penalty,solution
    return reports


class DiskLRU:
    """Reproducible full-cache working set bounded to at most 32 GiB on disk."""
    def __init__(self,path,max_bytes=32*1024**3):
        if max_bytes>32*1024**3 or max_bytes<1:raise ValueError('Cache LRU exceeds declared bound')
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.max_bytes=max_bytes
        self.events={'hits':0,'misses':0,'evictions':0,'uncached_oversized':0}
    def get(self,key,compute):
        filename=self.path/(digest(key)+'.pt')
        if filename.exists():
            result=torch.load(filename,map_location='cpu',weights_only=True);os.utime(filename,None);self.events['hits']+=1;return result
        self.events['misses']+=1;result=compute()
        # Every tensor is detached and offloaded. No gradients survive in the LRU.
        temporary=self.path/('pending-'+str(os.getpid())+'.tmp')
        with temporary.open('wb') as f:torch.save(result,f)
        size=temporary.stat().st_size
        if size>self.max_bytes:
            temporary.unlink();self.events['uncached_oversized']+=1;return result
        files=sorted(self.path.glob('*.pt'),key=lambda p:p.stat().st_mtime_ns)
        used=sum(p.stat().st_size for p in files)
        for old in files:
            if used+size<=self.max_bytes:break
            used-=old.stat().st_size;old.unlink();self.events['evictions']+=1
        os.replace(temporary,filename)
        if sum(p.stat().st_size for p in self.path.glob('*.pt'))>self.max_bytes:raise AssertionError('LRU bound exceeded')
        return result


class TrainingRuntime:
    def __init__(self,source,receiver,mapper,lru,guard=lambda:None):
        self.source,self.receiver,self.mapper,self.lru,self.guard=source,receiver,mapper,lru,guard
        self.cache_gradient_checks=[]
    def pair(self,obj,device=None):
        self.guard();ids=obj['source_history']['prefix_ids']
        def compute():
            with torch.no_grad():
                so=self.source.prefill_chunked(ids);sp=CacheExtractor.tensors(so.past_key_values,device='cpu');del so
                to=self.receiver.prefill_chunked(ids);tp=CacheExtractor.tensors(to.past_key_values,device='cpu');del to
            return {'source':sp,'target':tp}
        pair=self.lru.get({'history':digest(ids),'models':[self.source.name,self.receiver.name],
            'recompute':'complete-prefix-in-512-token-chunks-v1'},compute)
        device=self.receiver.device if device is None else device
        return tuple(tuple(t.to(device) for t in p) for p in pair['source']),tuple(tuple(t.to(device) for t in p) for p in pair['target'])
    def extended_pair(self,obj,positions,sp,tp):
        ids,pos=boundary_case(obj,positions);start=ordinary_anchor(positions)
        if start:
            with torch.no_grad():
                so=self.source.forward(ids[:start],CacheInjector.create(sp,clone=False));sp=CacheExtractor.tensors(so.past_key_values);del so
                to=self.receiver.forward(ids[:start],CacheInjector.create(tp,clone=False));tp=CacheExtractor.tensors(to.past_key_values);del to
        return sp,tp
    def extended_pair_from_cpu(self,obj,positions,source_cpu,target_cpu):
        """Transfer prefix ownership to each mutable cache before extending it.

        The CPU originals remain immutable. No second GPU tuple keeps all old
        layer tensors alive while the model replaces them with extended tensors.
        Forward tokens and call segmentation are identical to extended_pair.
        """
        if any(t.device.type!='cpu' or t.requires_grad for pair in (*source_cpu,*target_cpu) for t in pair):
            raise ValueError('Ordinary window loading requires detached CPU base caches')
        ids,_=boundary_case(obj,positions);start=ordinary_anchor(positions)
        def materialize(backend,pairs):
            cache=CacheInjector.create((tuple(t.to(backend.device) for t in pair) for pair in pairs),clone=False)
            if start:
                with torch.no_grad():
                    out=backend.forward(ids[:start],cache)
                cache=out.past_key_values
            return CacheExtractor.tensors(cache)
        return materialize(self.source,source_cpu),materialize(self.receiver,target_cpu)

    def kl(self,obj,positions,objective,sp,tp,require_grad=True,already_extended=False):
        self.guard();ids,pos=boundary_case(obj,positions)
        if objective=='ordinary_continuation':
            start=ordinary_anchor(positions)
            if not already_extended:sp,tp=self.extended_pair(obj,positions,sp,tp)
            ids=ids[start:];pos=[p-start for p in pos]
        elif objective!='natural_handoff_boundary':raise ValueError('Unknown objective')
        with torch.no_grad():teacher=continuation(self.receiver.model,tp,ids,positions=pos,checkpoint_layers=False).float().log_softmax(-1)
        mapped=self.mapper(sp)
        hooks=[];gradient_checks=[]
        if require_grad:
            for layer,pair in enumerate(mapped):
                for kind,tensor in enumerate(pair):
                    def hook(g,layer=layer,kind=kind):
                        gradient_checks.append({'layer':layer,'kind':kind,'finite':bool(torch.isfinite(g).all()),'absolute_sum':float(g.abs().sum(dtype=torch.float32))})
                        return g
                    hooks.append(tensor.register_hook(hook))
            self.cache_gradient_checks.append(gradient_checks)
        candidate=continuation(self.receiver.model,mapped,ids,positions=pos,checkpoint_layers=require_grad).float().log_softmax(-1)
        return (teacher.exp()*(teacher-candidate)).sum(-1)
    def train_update(self,item,lookup,objective,optimizer):
        optimizer.zero_grad(set_to_none=True);total=0.;grouped=OrderedDict();self.cache_gradient_checks=[]
        for segment in item['segments']:grouped.setdefault(segment['task_id'],[]).append(segment)
        for tid,segments in grouped.items():
            obj=lookup[tid]
            if objective=='ordinary_continuation':
                # CPU base pairs survive between windows. GPU originals are
                # replaced before mapping, so backward retains only one full
                # source cache and one native receiver cache.
                base_sp,base_tp=self.pair(obj,device='cpu')
                for part in segments:
                    self.guard()
                    sp,tp=self.extended_pair_from_cpu(obj,part['positions'],base_sp,base_tp)
                    losses=self.kl(obj,part['positions'],objective,sp,tp,already_extended=True)
                    loss=losses.sum()/item['predictions'];total+=float(loss.detach());loss.backward()
                    del loss,losses,sp,tp
                del base_sp,base_tp
            else:
                sp,tp=self.pair(obj)
                positions=[p for part in segments for p in part['positions']]
                losses=self.kl(obj,positions,objective,sp,tp)
                loss=losses.sum()/item['predictions'];total+=float(loss.detach());loss.backward()
                del loss,losses,sp,tp
            gc.collect()
        checks=[r for part in self.cache_gradient_checks for r in part]
        if not checks or not all(r['finite'] and r['absolute_sum']>0 for r in checks):raise RuntimeError('Missing, zero or nonfinite cache gradient')
        if any(p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum()>0 for p in self.mapper.parameters()):raise RuntimeError('Missing, zero or nonfinite mapper gradient')
        if any(p.grad is not None for b in [self.source,self.receiver] for p in b.model.parameters()):raise RuntimeError('Frozen language model received gradients')
        norm=float(torch.nn.utils.clip_grad_norm_(self.mapper.parameters(),1,error_if_nonfinite=True))
        self.guard();optimizer.step();optimizer.zero_grad(set_to_none=True)
        return {'kl':total,'gradient_norm_before_clipping':norm,'gradient_predictions':item['predictions'],
            'examples':len(grouped),'cache_gradient_tensors_checked':len(checks),'cache_gradients_finite_nonzero':True}
    @torch.no_grad()
    def validate(self,histories):
        rows=[]
        for obj in histories:
            self.guard();positions=[p for a in OFFSETS for p in valid_positions(obj['teacher_answer']['answer_ids'],a)]
            sp,tp=self.pair(obj)
            losses=self.kl(obj,positions,'natural_handoff_boundary',sp,tp,require_grad=False)
            rows.append({'task_id':obj['task_id'],'predictions':len(positions),'positions':positions,'kl':float(losses.mean())})
            del sp,tp,losses;gc.collect()
        return {'validation_kl':sum(r['kl'] for r in rows)/len(rows),'rows':rows,
            'aggregation':'Equal task mean; only existing tokens at declared offsets; shared full-boundary objective.'}


@dataclass
class StopState:
    best_significant:float=math.inf
    stale:int=0
    def observe(self,kl):
        if not math.isfinite(kl) or kl < -1e-5:raise ValueError('Invalid validation KL')
        if kl<self.best_significant*(1-.005):self.best_significant=kl;self.stale=0
        else:self.stale+=1
    def plateau(self,step):return step>=256 and self.stale>=4


def nominations(curve):
    eligible=[r for r in curve if r['step']>=256 and r.get('checkpoint')]
    return sorted(eligible,key=lambda r:(r['validation_kl'],r['step']))[:2]


def choose_development_candidate(rows):
    if not 1<=len(rows)<=4:raise ValueError('Selection requires one to four frozen validation nominees')
    if len({r['candidate_id'] for r in rows})!=len(rows):raise ValueError('Candidate duplicate')
    for r in rows:
        if r['task_count']!=40 or not math.isfinite(r['source_inclusive_seconds']) or r['source_inclusive_seconds']<0:raise ValueError('Incomplete candidate development evaluation')
        if not 0<=r['passes']<=40:raise ValueError('Invalid pass count')
    return min(rows,key=lambda r:(-r['passes'],r['source_inclusive_seconds'],r['step'],r['candidate_id']))


def train_bounded(runtime,histories,validation,objective,seed,root,deadline_epoch,identity,
                  max_updates=2048,wall_seconds=10800,validation_every=128,clock=time.time,checkpoint_callback=lambda:None):
    """Run one objective/seed. A time cap can stop before the minimum updates."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'progress.json').exists() or (root/'complete.json').exists():raise ValueError('Partial training cannot silently restart')
    bind(root/'identity.json',identity)
    schedule=make_schedule(histories,seed,max_updates);write(root/'schedule.json',schedule)
    write(root/'schedule_audit.json',schedule_audit(histories,schedule))
    optimizer=torch.optim.AdamW(runtime.mapper.parameters(),lr=1e-5,weight_decay=0)
    lookup={o['task_id']:o for o in histories};started=clock();deadline=min(deadline_epoch,started+wall_seconds)
    curve=[];steps=[];state=StopState();stop='max_updates';last_step=0
    original_guard=runtime.guard
    def guard():
        original_guard()
        if clock()>=deadline:raise TimeoutError('Declared training time ceiling')
    runtime.guard=guard
    def checkpoint(step,validation_result=None):
        filename=f'mapper_step_{step:04d}.pt';path=root/filename
        atomic_tensor(path,{'state_dict':{k:v.detach().cpu() for k,v in runtime.mapper.state_dict().items()},
            'optimizer_resume_supported':False,'step':step,'seed':seed,'objective':objective,'identity_sha256':digest(identity)})
        descriptor={'path':filename,'sha256':sha(path),'bytes':path.stat().st_size}
        checkpoint_callback()
        if validation_result is not None:
            row={'step':step,**validation_result,'checkpoint':descriptor};curve.append(row);state.observe(row['validation_kl'])
            write(root/'validation_curve.json',curve)
        return descriptor
    try:
        for item in schedule:
            guard();step=item['step'];before=clock()
            row=runtime.train_update(item,lookup,objective,optimizer);last_step=step
            steps.append({'step':step,**row,'wall_seconds':clock()-before});write(root/'training_steps.json',steps)
            write(root/'progress.json',{'state':'running','step':step,'wall_seconds':clock()-started,'deadline_epoch':deadline})
            if step%validation_every==0 or step==max_updates:
                result=runtime.validate(validation);checkpoint(step,result)
                if state.plateau(step):stop='validation_plateau';break
    except TimeoutError:
        stop='time_cap'
    finally:
        runtime.guard=original_guard
    final_checkpoint=root/f'mapper_step_{last_step:04d}.pt'
    if last_step and not final_checkpoint.exists():checkpoint(last_step)
    complete={'identity_sha256':digest(identity),'objective':objective,'seed':seed,'steps':last_step,
        'stop_reason':stop,'converged':stop=='validation_plateau','capped_unconverged':stop in ['time_cap','max_updates'],
        'wall_seconds':clock()-started,'nominees':nominations(curve),'training_predictions':sum(r['gradient_predictions'] for r in steps),
        'no_eligible_checkpoint':not nominations(curve),'initialization_sha256':identity['initialization_sha256']}
    write(root/'complete.json',complete);write(root/'progress.json',{'state':'complete',**complete})
    return complete


def worker_context(repo,spec_path=None):
    """Fail closed on changed baseline, memory, approval, allocation or code."""
    import importlib.util, platform, sys, threading, transformers
    repo=Path(repo)
    spec_path=Path(spec_path or os.environ['GEARSHIFT_WORKER_SPEC']);spec=read(spec_path)
    status_root=Path(os.environ.get('GEARSHIFT_WORKER_ROOT',str(spec_path.parent)))
    result_root=Path(spec['result_root'])
    if not result_root.is_absolute():result_root=repo/result_root
    result_root.resolve().relative_to((repo/'results/coding_pilot_v1').resolve())
    if result_root.exists() and any(result_root.iterdir()):raise ValueError('Worker result namespace is not empty; no silent retry')
    module_spec=importlib.util.spec_from_file_location('coding_memory_gate_checks',repo/'scripts/coding_memory_preflight.py')
    gate_module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(gate_module)
    baseline=gate_module.check_baselines(spec['baseline_root'])
    memory_path=Path(spec['memory_gate']);memory=read(memory_path);memory_identity=read(memory_path.parent/'identity.json')
    if not (memory.get('passed') and memory.get('training_ready') and memory.get('actual_training_runtime')):raise ValueError('Actual training-runtime memory gate required')
    objectives=memory.get('objectives',{})
    if set(objectives)!=set(OBJECTIVES) or any(r.get('gradient_predictions')!=32 for r in objectives.values()):raise ValueError('Both actual32-prediction objectives must pass memory verification')
    if memory_identity.get('training_protocol_sha256')!=sha(repo/'configs/coding_pilot_v1/training_protocol_v1.json'):raise ValueError('Memory training protocol differs')
    if memory.get('source_weights_on_cpu') or memory_identity.get('source_weight_cpu_offload'):raise ValueError('Training source-weight offload is not implemented; both-GPU memory gate required')
    if memory_identity['baseline_identity_file_sha256']!=baseline['baseline_identity_file_sha256'] or memory_identity['maximum_total_context']!=40960:
        raise ValueError('Memory gate is not bound to this baseline/full context')
    for name in ['gearshift/core.py','gearshift/coding_inference.py','gearshift/coding_gradients.py','gearshift/coding_training.py']:
        if memory_identity['implementation'][name]!=sha(repo/name):raise ValueError('Memory numerical implementation changed')
    approval_path=Path(spec['approval_path']);allocation_path=Path(spec['allocation_path'])
    allocation=read(allocation_path);approval=read(approval_path)
    if not approval.get('approved') or allocation['approval_sha256']!=sha(approval_path):raise ValueError('Worker allocation/approval binding differs')
    deadline=float(spec['deadline_epoch'])
    if not math.isfinite(deadline) or deadline<=time.time()+120:raise ValueError('Worker deadline already expired')
    if allocation.get('deadline_epoch') is not None and deadline>allocation['deadline_epoch']:raise ValueError('Worker exceeds allocation deadline')
    current={'state':'running','stage':'validated_prerequisites','stage_identity':spec['stage_identity'],'worker_id':spec['worker_id']}
    halt=threading.Event();lock=threading.Lock();error=[]
    def publish(**values):
        with lock:
            current.update(values);write(status_root/'worker_status.json',{**current,'heartbeat_epoch':time.time()})
            write(status_root/'progress.json',{**current,'heartbeat_epoch':time.time()})
    def heartbeat():
        while not halt.wait(10):
            try:publish()
            except BaseException as exc:error.append(exc);return
    def guard():
        if error:raise RuntimeError('Worker heartbeat persistence failed')
        if time.time()>=deadline-120 or (repo/'evidence/coding_pilot_v1/STOP').exists() or (status_root/'STOP').exists():raise TimeoutError('Worker allocation deadline or stop marker')
        if torch.cuda.is_available():
            free,total=torch.cuda.mem_get_info()
            if torch.cuda.max_memory_allocated()>=.85*total or free<10*1024**3:raise RuntimeError('Actual worker memory/headroom gate failed')
    cfg=read(repo/'configs/coding_pilot_v1/pilot.json');protocol_path=repo/'configs/coding_pilot_v1/training_protocol_v1.json'
    impl={str(p.relative_to(repo)):sha(p) for p in [repo/'gearshift/coding_training.py',repo/'gearshift/coding_gradients.py',repo/'gearshift/coding_inference.py',repo/'gearshift/core.py',repo/'scripts/coding_training_worker.py',repo/'scripts/coding_history_worker.py']}
    evaluation_path=repo/'scripts/coding_evaluation_worker.py'
    if evaluation_path.exists():impl['scripts/coding_evaluation_worker.py']=sha(evaluation_path)
    identity={'schema':1,'worker_spec_sha256':sha(spec_path),'stage_identity':spec['stage_identity'],'worker_id':spec['worker_id'],
        'config_sha256':sha(repo/'configs/coding_pilot_v1/pilot.json'),'training_protocol_sha256':sha(protocol_path),
        'baseline':baseline,'memory_gate_sha256':sha(memory_path),'memory_identity_sha256':sha(memory_path.parent/'identity.json'),
        'approval_sha256':sha(approval_path),'allocation_sha256':sha(allocation_path),'implementation':impl,
        'data_identity':read(repo/'data/coding_pilot_v1/identity.json'),
        'runtime':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'platform':platform.platform(),
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    result_root.mkdir(parents=True,exist_ok=True);bind(result_root/'identity.json',identity)
    publish();thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    return {'spec':spec,'config':cfg,'root':result_root,'status_root':status_root,'identity':identity,'guard':guard,'publish':publish,'halt':halt,'thread':thread}


def initialize_backends(context):
    from .coding_inference import Backend
    from .coding_control_calibration import matched_native_controls
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats();c=context
    c['guard']();c['publish'](stage='load_source');source=Backend(c['config']['models']['source'])
    c['publish'](stage='load_receiver');receiver=Backend(c['config']['models']['receiver']);c['guard']()
    if source.tokenizer.backend_tokenizer.to_str()!=receiver.tokenizer.backend_tokenizer.to_str():raise ValueError('Pinned tokenizer mismatch')
    for role,backend in [('source',source),('receiver',receiver)]:
        c['publish'](stage=role+'_native_controls');c['guard']()
        matched_native_controls(backend,c['root']/'controls',role,max_prefix=32768);c['guard']()
    write(c['root']/'native_gate.json',{'passed':True,'identity_sha256':digest(c['identity'])})
    return source,receiver


def verify_history_roots(repo,roots):
    repo=Path(repo);expected={}
    for split in ['training','validation']:
        for row in read(repo/f'data/coding_pilot_v1/visible/{split}.json'):expected[row['task_id']]=(split,row)
    histories=[];feature_paths=[];receipts=[]
    for root in map(Path,roots):
        manifest=read(root/'identity.json');native=read(root/'native_gate.json');done=read(root/'complete.json')
        if not native['passed'] or native['identity_sha256']!=digest(manifest) or done['identity_sha256']!=digest(manifest):raise ValueError('History worker identity/native gate differs')
        if manifest['config_sha256']!=sha(repo/'configs/coding_pilot_v1/pilot.json') or manifest['training_protocol_sha256']!=sha(repo/'configs/coding_pilot_v1/training_protocol_v1.json'):raise ValueError('History protocol changed')
        for name in ['gearshift/core.py','gearshift/coding_inference.py','scripts/coding_history_worker.py','gearshift/coding_training.py']:
            if manifest['implementation'][name]!=sha(repo/name):raise ValueError('History numerical implementation changed')
        for tid,receipt_sha in done['tasks'].items():
            folder=root/'tasks'/safe_id(tid);receipt=read(folder/'complete.json')
            if sha(folder/'complete.json')!=receipt_sha or receipt['task_id']!=tid or receipt['identity_sha256']!=digest(manifest):raise ValueError('History transaction changed')
            for name,want in receipt['files'].items():
                if Path(name).name!=name or sha(folder/name)!=want:raise ValueError('History artifact changed')
            split,task=expected[tid]
            obj={'task_id':tid,'split':split,'source_history':read(folder/'source_history.json'),'teacher_answer':read(folder/'teacher_answer.json')}
            if obj['source_history']['prompt_ids']!=task['prompt_ids']:raise ValueError('Visible task changed')
            validate_history(obj);histories.append(obj)
            if split=='training':
                if receipt.get('feature_files',{}).get('paired_features.pt')!=sha(folder/'paired_features.pt'):raise ValueError('Paired initialization feature changed')
                feature_paths.append(folder/'paired_features.pt')
            receipts.append({'task_id':tid,'receipt_sha256':receipt_sha,'source_root':str(root)})
    if len(histories)!=len(expected) or {o['task_id'] for o in histories}!=set(expected):raise ValueError('Exactly the fixed 128 training and 32 validation histories are required')
    return sorted(histories,key=lambda o:o['task_id']),feature_paths,receipts


def checkpoint_manifest(repo,root):
    repo,root=Path(repo),Path(root)
    files={str(p.relative_to(repo)):sha(p) for p in sorted(root.glob('mapper*.pt'))}
    if any((repo/p).stat().st_size>1024**3 for p in files) or sum((repo/p).stat().st_size for p in files)>8*1024**3:raise ValueError('Mapper backup bound exceeded')
    write(root/'checkpoint_manifest.json',{'files':files,'policy':'Immutable mapper-only checkpoints; optimizer resume is not supported.'})


def finish_worker(context,state,**values):
    context['publish'](state=state,**values);context['halt'].set();context['thread'].join(timeout=15)
