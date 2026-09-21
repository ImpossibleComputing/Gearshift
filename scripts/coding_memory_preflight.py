#!/usr/bin/env python3
"""Gated full-context gradient stress test; prepared separately from quality runs.

Synthetic repetitions of one training prompt exercise the declared maximum shape.
They are not a source reasoning trajectory, quality result, or fitted mapper.
"""
import argparse,gc,json,math,os,platform,sys,threading,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha,bind,digest
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/coding_pilot_v1'

def check_baselines(path):
    from gearshift.coding_reuse import compatible_parent,verify_completed
    path=Path(path)
    gate=json.loads((path/'baseline_gate.json').read_text())
    if not gate.get('passed') or not gate.get('cap_gate'):raise ValueError('Full development protocol/baseline gate is required')
    rows=list((path/'tasks').glob('*/complete.json'))
    if len(rows)!=40:raise ValueError('Forty committed development cases required')
    identity=compatible_parent(path,ROOT)
    expected={x['task_id'] for x in json.loads((ROOT/'data/coding_pilot_v1/visible/development.json').read_text())}
    actual=[]
    for p in rows:
        row=json.loads(p.read_text())
        verify_completed(p.parent,identity,row['task_id'])
        for kind in ['source','small']:
            history=json.loads((p.parent/(kind+'_history.json')).read_text())
            if history['reasoning_capped']!=row[kind+'_capped']:
                raise ValueError('Baseline cap flags differ from saved histories')
        actual.append(row)
    if {r['task_id'] for r in actual}!=expected or len(expected)!=40:raise ValueError('Development membership changed')
    counts={a:sum(r['pass'][a] for r in actual) for a in ['A','B','D']}
    if counts!=gate['counts'] or not (counts['A']>=12 and 4<=counts['B']<=34 and counts['A']-counts['B']>=4 and counts['D']-counts['B']>=2):raise ValueError('Baseline operational gate failed')
    if any(sum(r[k] for r in actual)>4 for k in ['source_capped','small_capped']):raise ValueError('Development cap protocol gate failed')
    return {'baseline_gate_sha256':sha(path/'baseline_gate.json'),'baseline_identity_file_sha256':sha(path/'identity.json'),'task_count':40}


def approved_deadline(allocation_path,approval_path,now=None,previous_approval_path=None):
    from gearshift.coding_control import decision
    now=time.time() if now is None else now
    allocation=json.loads(Path(allocation_path).read_text())
    approval=json.loads(Path(approval_path).read_text());approval['receipt_sha256']=sha(approval_path)
    if approval.get('parallel_execution_approved'):
        old_path=Path(previous_approval_path) if previous_approval_path else ROOT/'evidence/coding_pilot_v1/experiment_100h_approved.json'
        old=json.loads(old_path.read_text());old['receipt_sha256']=sha(old_path)
        d=decision({'approved_experiment_extension':old,'approved_parallel_extension':approval},now)
    else:
        d=decision({'approved_experiment_extension':approval},now)
    if d['stop'] or allocation.get('approval_sha256')!=sha(approval_path):
        raise ValueError('Allocation must bind the affirmative experiment approval')
    if allocation.get('cap_usd')!=d['hard_total_cap_usd']:
        raise ValueError('Allocation spending ceiling differs')
    deadline=allocation.get('preflight_deadline_epoch')
    allowance=allocation.get('remaining_authorized_seconds')
    prior=allocation.get('prior_usage',{})
    if (type(deadline) not in (int,float) or not math.isfinite(deadline)
        or type(allowance) not in (int,float) or not math.isfinite(allowance) or allowance<=0
        or allowance>(d['hard_total_gpu_hours']-prior.get('gpu_hours',d['hard_total_gpu_hours']))*3600+.01
        or deadline>allocation.get('started_epoch',0)+allowance+.01 or deadline<=now+120):
        raise ValueError('Allocation deadline is expired or exceeds cumulative approval')
    return allocation,deadline-120

class Monitor:
    def __init__(self,torch):
        self.torch=torch;self.minimum_free=None;self.running=False;self.errors=[]
        self.thread=None;self.lock=threading.Lock()
    def check(self):
        if self.errors:raise RuntimeError('GPU memory monitor failed: '+str(self.errors[0])) from self.errors[0]
    def sample(self):
        self.check()
        free,total=self.torch.cuda.mem_get_info()
        with self.lock:self.minimum_free=free if self.minimum_free is None else min(self.minimum_free,free)
    def start(self):
        self.sample();self.running=True
        def loop():
            try:
                while self.running:self.sample();time.sleep(.05)
            except BaseException as exc:self.errors.append(exc);self.running=False
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def stop(self):
        self.running=False
        if self.thread is not None:self.thread.join()
        self.check();self.sample()


def stress_case(base,maximum_total_context=40960,boundary_token=151668):
    """520 consumed inputs end at context limit; offset519 predicts the next token."""
    if not base or maximum_total_context<=520:raise ValueError('Stress prefix is empty')
    n=maximum_total_context-520
    prefix=(base*(n//len(base)+1))[:n]
    answer=(base*(520//len(base)+1))[:520]
    return {'task_id':'synthetic_full_context_diagnostic','split':'training',
        'source_history':{'prefix_ids':prefix,'bridge_ids':[boundary_token]},
        'teacher_answer':{'answer_ids':answer}}


class FixedPair:
    """A real precomputed CPU cache hit through the production pair-loading path."""
    def __init__(self,prefix,source,target):
        if any(t.device.type!='cpu' or t.requires_grad for pair in (*source,*target) for t in pair):
            raise ValueError('The stress cache must retain CPU tensors only')
        self.history=digest(prefix);self.value={'source':source,'target':target}
    def get(self,key,compute):
        if key['history']!=self.history:raise ValueError('Synthetic stress cache prefix changed')
        return self.value


def exercise_objective(runtime,obj,objective,record=lambda *a,**k:None):
    """Exercise production training/backward/AdamW/validation without quality data."""
    import torch
    from gearshift.coding_training import OFFSETS
    before=[p.detach().cpu().clone() for p in runtime.mapper.parameters()]
    checks={}
    class CheckedAdamW(torch.optim.AdamW):
        def step(self,closure=None):
            gradients=[p.grad for p in runtime.mapper.parameters()]
            if not gradients or not all(g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum())>0 for g in gradients):
                raise RuntimeError('Missing, zero or nonfinite mapper gradients at optimizer step')
            if any(p.grad is not None for b in [runtime.source,runtime.receiver] for p in b.model.parameters()):
                raise RuntimeError('Frozen language-model gradients were allocated')
            checks['mapper_gradient_tensors']=len(gradients)
            record(objective+'_backward',mapper_gradient_tensors=len(gradients))
            result=super().step(closure)
            if len(self.state)!=len(gradients):raise RuntimeError('Incomplete AdamW optimizer state')
            if not all(bool(torch.isfinite(p).all()) for p in runtime.mapper.parameters()):
                raise RuntimeError('Nonfinite mapper parameters after optimizer step')
            record(objective+'_optimizer_step',optimizer_state_parameters=len(self.state))
            return result
    optimizer=CheckedAdamW(runtime.mapper.parameters(),lr=1e-5,weight_decay=0)
    item={'step':1,'predictions':32,'segments':[
        {'task_id':obj['task_id'],'anchor':a,'positions':list(range(a,a+8))} for a in OFFSETS]}
    started=time.monotonic()
    result=runtime.train_update(item,{obj['task_id']:obj},objective,optimizer)
    cache_checks=[c for group in runtime.cache_gradient_checks for c in group]
    expected_pairs={(layer,kind) for layer in range(len(runtime.receiver.model.model.layers)) for kind in range(2)}
    groups=4 if objective=='ordinary_continuation' else 1
    complete_groups=(len(runtime.cache_gradient_checks)==groups and all(len(group)==len(expected_pairs)
        and {(c['layer'],c['kind']) for c in group}==expected_pairs for group in runtime.cache_gradient_checks))
    if not complete_groups or not all(c['finite'] and c['absolute_sum']>0 for c in cache_checks):
        raise RuntimeError('Missing, zero or nonfinite mapped-cache gradients')
    if result['gradient_predictions']!=32:raise RuntimeError('Stress gradient prediction count changed')
    if not any(not torch.equal(old,p.detach().cpu()) for old,p in zip(before,runtime.mapper.parameters())):
        raise RuntimeError('The optimizer did not change the mapper')
    record(objective+'_training_step',**result,step_seconds=time.monotonic()-started)
    validation=runtime.validate([obj]);record(objective+'_validation',validation_kl=validation['validation_kl'])
    if not math.isfinite(validation['validation_kl']):raise RuntimeError('Nonfinite validation KL')
    result.update(checks,cache_gradients=cache_checks,validation=validation,
        step_seconds=time.monotonic()-started,optimizer_parameters_updated=True)
    del optimizer,before;gc.collect()
    return result


def main():
    import torch
    from gearshift.coding_inference import Backend,memory_record
    from gearshift.coding_gradients import AffineMapper
    from gearshift.coding_training import TrainingRuntime
    from gearshift.core import CacheExtractor
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-root',required=True,type=Path)
    parser.add_argument('--allocation',required=True,type=Path)
    parser.add_argument('--approval',required=True,type=Path)
    parser.add_argument('--previous-approval',type=Path)
    parser.add_argument('--source-on-cpu',action='store_true',help='Boundary-only offload diagnostic; cannot authorize training')
    parser.add_argument('--output-root',type=Path,help='Fresh probe namespace under results/coding_pilot_v1')
    args=parser.parse_args();baseline=check_baselines(args.baseline_root)
    cfg=json.loads((ROOT/'configs/coding_pilot_v1/pilot.json').read_text())
    allocation,deadline=approved_deadline(args.allocation,args.approval,previous_approval_path=args.previous_approval)
    def guard():
        if time.time()>=deadline or (E/'STOP').exists():raise TimeoutError('Preflight allowance exhausted')
    guard();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    root=ROOT/'results/coding_pilot_v1'/('memory_v2_source_cpu_diagnostic' if args.source_on_cpu else 'memory_v2_both_gpu')
    if args.output_root is not None:
        root=(ROOT/args.output_root).resolve()
        if not root.is_relative_to((ROOT/'results/coding_pilot_v1').resolve()):
            raise ValueError('Memory output must stay in its experiment namespace')
    if root.exists():raise ValueError('Memory attempt already exists; preserve its outcome')
    root.mkdir(parents=True)
    objectives=['natural_handoff_boundary'] if args.source_on_cpu else ['natural_handoff_boundary','ordinary_continuation']
    bind(root/'identity.json',{'config_sha256':sha(ROOT/'configs/coding_pilot_v1/pilot.json'),**baseline,
        'implementation':{f:sha(ROOT/f) for f in ['gearshift/core.py','gearshift/coding_inference.py','gearshift/coding_gradients.py',
            'gearshift/coding_training.py','scripts/coding_memory_preflight.py']},
        'training_protocol_sha256':sha(ROOT/'configs/coding_pilot_v1/training_protocol_v1.json'),
        'synthetic_stress_probe':True,'source_weight_cpu_offload':args.source_on_cpu,'maximum_total_context':40960,
        'objectives':objectives,'actual_training_runtime':True,'training_ready_possible':not args.source_on_cpu,
        'gradient_prediction_offsets':[j for i in [0,32,128,512] for j in range(i,i+8)],'mapper_initialization':'seeded random diagnostic only; not a trained checkpoint',
        'allocation_sha256':sha(args.allocation),'approval_sha256':sha(args.approval),
        'runtime':{'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'image_digest':allocation['image_digest']}})
    records=[];monitor=Monitor(torch);outcomes={};monitor_stopped=False
    def record(stage,**extra):
        torch.cuda.synchronize();monitor.sample();r={'stage':stage,'epoch':time.time(),**memory_record(),**extra}
        r['minimum_observed_free_bytes']=monitor.minimum_free
        r['passed']=r['passed'] and monitor.minimum_free>=10*1024**3
        records.append(r);write(root/'progress.json',{'records':records});guard()
        if not r['passed']:raise RuntimeError('Full-context memory/headroom gate failed at '+stage)
    active_objective={'name':None}
    def runtime_guard():
        guard();monitor.check();free,total=torch.cuda.mem_get_info()
        peak=torch.cuda.max_memory_allocated()
        if peak>=.85*total or free<10*1024**3:
            # Persist the failing sample before raising; a prior successful
            # record cannot explain which of the unchanged limits was crossed.
            records.append({'stage':'runtime_guard_failure','objective':active_objective['name'],
                'epoch':time.time(),'allocated':torch.cuda.memory_allocated(),
                'reserved':torch.cuda.memory_reserved(),'peak_allocated':peak,
                'free':free,'total':total,'passed':False,
                'minimum_observed_free_bytes':monitor.minimum_free,
                'peak_fraction_limit':.85,'minimum_free_limit_bytes':10*1024**3,
                'peak_limit_exceeded':peak>=.85*total,'free_limit_exceeded':free<10*1024**3})
            write(root/'progress.json',{'records':records})
            raise RuntimeError('Actual training path memory/headroom gate failed')
    torch.cuda.reset_peak_memory_stats()
    try:
        monitor.start()
        source=Backend(cfg['models']['source']);receiver=Backend(cfg['models']['receiver']);record('models_loaded')
        from gearshift.coding_control_calibration import matched_native_controls
        for kind,backend in [('source',source),('receiver',receiver)]:
            guard();matched_native_controls(backend,root/'controls',kind,max_prefix=32768);guard()
        write(root/'native_gate.json',{'passed':True,'memory_identity_file_sha256':sha(root/'identity.json')})
        # Native controls reset allocator statistics internally; start a fresh
        # cumulative peak spanning cache preparation and BOTH actual objectives.
        torch.cuda.reset_peak_memory_stats();record('gradient_probe_start')
        task=json.loads((ROOT/'data/coding_pilot_v1/visible/training.json').read_text())[0]
        obj=stress_case(task['prompt_ids']);prefix=obj['source_history']['prefix_ids']
        with torch.no_grad():
            out=source.prefill_chunked(prefix);pairs=CacheExtractor.tensors(out.past_key_values,device='cpu')
        del out;gc.collect();torch.cuda.empty_cache();record('full_source_prefix',prefix_tokens=len(prefix),continuation_tokens=520)
        if args.source_on_cpu:
            before=time.monotonic();source.model.to('cpu');torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
            record('source_weights_offloaded',transfer_seconds=time.monotonic()-before)
        with torch.no_grad():
            out=receiver.prefill_chunked(prefix);native=CacheExtractor.tensors(out.past_key_values,device='cpu')
        del out;gc.collect();torch.cuda.empty_cache();record('full_native_prefix',prefix_tokens=len(prefix))
        fixed=FixedPair(prefix,pairs,native)
        for objective in objectives:
            active_objective['name']=objective
            mapper=AffineMapper(source,receiver)
            if sum(p.numel() for p in mapper.parameters())!=75571200:raise ValueError('Mapper parameter count mismatch')
            runtime=TrainingRuntime(source,receiver,mapper,fixed,runtime_guard)
            outcomes[objective]=exercise_objective(runtime,obj,objective,record)
            del runtime,mapper;gc.collect();torch.cuda.empty_cache();record(objective+'_released')
        # A failed sampling thread or final CUDA sample must never leave a pass.
        monitor.stop();monitor_stopped=True;record('final_verified_memory')
        ready=not args.source_on_cpu and set(outcomes)=={'natural_handoff_boundary','ordinary_continuation'}
        write(root/'memory_gate.json',{'passed':ready,'training_ready':ready,'diagnostic_passed':True,
            'synthetic_stress_probe':True,'source_weights_on_cpu':args.source_on_cpu,'actual_training_runtime':True,
            'prefix_tokens':len(prefix),'continuation_tokens':520,'gradient_predictions':32,'objectives':outcomes,'records':records,
            'limitation':'Two maximum-shape production training steps only; real-history throughput and convergence remain unverified.'
                if ready else 'CPU source offload is a boundary-only diagnostic and does not authorize training.'})
    except BaseException as exc:
        write(root/'memory_gate.json',{'passed':False,'training_ready':False,'error':str(exc),'exception':type(exc).__name__,
            'objectives':outcomes,'records':records})
        raise
    finally:
        if not monitor_stopped:
            try:monitor.stop()
            except BaseException as exc:
                # Keep the original experiment failure while retaining the
                # independent monitor failure as evidence; neither becomes pass.
                write(root/'monitor_failure.json',{'error':str(exc),'exception':type(exc).__name__})

if __name__=='__main__':main()
