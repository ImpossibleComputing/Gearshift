"""Explicitly amended recovery: numerical controls stay strict; memory is observed.

Historical inference/training functions and their evidence remain unchanged.
"""
import contextlib,gc,json,math,os,platform,resource,signal,sys,threading,time
from pathlib import Path
import torch
from .coding_control import bind,digest,sha,write,seed_for
from .coding_training import read

AMENDMENT='configs/coding_pilot_v1/recovery_20260917/amendment.json'

def memory_warnings(row):
    warnings=[]
    if row['peak_allocated']>=.85*row['total']:warnings.append('historical_peak_at_least_85_percent')
    if row['free']<10*1024**3:warnings.append('driver_free_below_10GiB')
    return warnings

class Telemetry:
    def __init__(self,root,cuda=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.cuda=cuda or torch.cuda;self.current={'task_id':None,'sequence_length':None,'stage':'startup'}
        self.scope='process_start_or_external_controls';self.last=0
    def sample(self,force=True,**context):
        self.current.update(context)
        if not force and time.monotonic()-self.last<10:return None
        self.last=time.monotonic();r={'timestamp_epoch':time.time(),'pid':os.getpid(),'device':'cuda:0',**self.current,'peak_reset_scope':self.scope}
        if self.cuda.is_available():
            free,total=self.cuda.mem_get_info();stats=self.cuda.memory_stats()
            r.update(allocated=self.cuda.memory_allocated(),reserved=self.cuda.memory_reserved(),peak_allocated=self.cuda.max_memory_allocated(),
                     peak_reserved=self.cuda.max_memory_reserved(),free=free,total=total,
                     allocator={k:stats.get(k) for k in ['active_bytes.all.current','inactive_split_bytes.all.current','num_alloc_retries','num_ooms']})
            r['warnings']=memory_warnings(r)
        r['host_maxrss_native_units']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        r['host_maxrss_units']='bytes' if sys.platform=='darwin' else 'KiB'
        if Path('/proc/meminfo').exists():r['host_meminfo']='\n'.join(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:','SwapFree:')))
        r['memory_policy']='warning_only; actual allocation failures and resource deadlines remain fatal'
        with (self.root/'memory_telemetry.jsonl').open('a') as f:f.write(json.dumps(r,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        write(self.root/'memory_latest.json',r);return r
    def reset(self,scope):
        self.sample(stage='before_peak_reset')
        if self.cuda.is_available():self.cuda.reset_peak_memory_stats()
        self.scope=scope;self.sample(stage='after_peak_reset')
    def failure(self,exc):
        try:self.sample(stage='exception',exception=type(exc).__name__,error=str(exc))
        except Exception as nested:write(self.root/'telemetry_failure.json',{'error':str(nested)})
        if self.cuda.is_available():
            try:(self.root/'allocator_failure.txt').write_text(self.cuda.memory_summary(abbreviated=False))
            except Exception:pass


def worker_context(repo,spec_path=None):
    import transformers
    from .coding_parallel import verify_gate
    repo=Path(repo);spec_path=Path(spec_path or os.environ['GEARSHIFT_WORKER_SPEC']);spec=read(spec_path)
    if spec['stage'] not in ['recovery_probe','exploratory_training','exploratory_development','post_progress_numerical','post_progress_hybrid','post_progress_memorization','coverage_preflight','coverage_experiment','coverage_evaluation_recovery']:raise ValueError('Not an amended worker stage')
    amendment=read(repo/AMENDMENT)
    if amendment['amendment_id']!='recovery_20260917_partial_v1' or spec['amendment_sha256']!=sha(repo/AMENDMENT):raise ValueError('Amendment mismatch')
    for name,ref in spec['gates'].items():verify_gate(repo,name,ref)
    approval=read(repo/spec['approval_path']);allocation=read(repo/spec['allocation_path'])
    if not approval['approved'] or allocation['approval_sha256']!=sha(repo/spec['approval_path']):raise ValueError('Approval/allocation differs')
    deadline=spec['deadline_epoch']
    if not math.isfinite(deadline) or deadline<=time.time()+120 or deadline>allocation['deadline_epoch']:raise ValueError('Worker deadline invalid')
    root=repo/spec['result_root'];root.resolve().relative_to((repo/'results/coding_pilot_v1').resolve())
    if root.exists() and any(root.iterdir()):raise ValueError('No silent worker retry')
    root.mkdir(parents=True);status_root=repo/spec['worker_root'];status_root.mkdir(exist_ok=True,parents=True)
    telemetry=Telemetry(root);current={'state':'running','stage':'validated_amendment','worker_id':spec['worker_id'],'stage_identity':spec['stage_identity']}
    halt=threading.Event();lock=threading.Lock();errors=[]
    def publish(**values):
        with lock:
            current.update(values);write(status_root/'worker_status.json',{**current,'heartbeat_epoch':time.time()});write(root/'progress.json',{**current,'heartbeat_epoch':time.time()})
    def pulse():
        while not halt.wait(10):
            try:publish()
            except BaseException as exc:errors.append(exc);return
    def guard():
        telemetry.sample(force=False)
        if errors:raise RuntimeError('Heartbeat persistence failed')
        if time.time()>=deadline-120 or (repo/'evidence/coding_pilot_v1/STOP').exists() or (status_root/'STOP').exists():raise TimeoutError('Allocation deadline or stop marker')
    impl={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'gearshift').glob('coding_*.py'))}
    impl.update({str(p.relative_to(repo)):sha(p) for p in sorted((repo/'scripts').glob('coding_recovery*.py'))})
    impl.update({str(p.relative_to(repo)):sha(p) for p in sorted((repo/'scripts').glob('coding_post_*.py'))})
    impl.update({str(p.relative_to(repo)):sha(p) for p in sorted((repo/'scripts').glob('coding_coverage*.py'))})
    impl['gearshift/core.py']=sha(repo/'gearshift/core.py')
    identity={'amendment_sha256':sha(repo/AMENDMENT),'worker_spec_sha256':sha(spec_path),'stage_identity':spec['stage_identity'],'worker_id':spec['worker_id'],
        'config_sha256':sha(repo/'configs/coding_pilot_v1/pilot.json'),'implementation':impl,'allocation_sha256':sha(repo/spec['allocation_path']),
        'runtime':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},
        'legacy_memory_probe':'Historical evidence only; no claim that its fatal thresholds passed in the amended run.'}
    bind(root/'identity.json',identity);publish();thread=threading.Thread(target=pulse,daemon=True);thread.start()
    def stop(signum,frame):raise TimeoutError('Received controlled termination signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    return dict(spec=spec,root=root,status_root=status_root,identity=identity,config=read(repo/'configs/coding_pilot_v1/pilot.json'),guard=guard,publish=publish,halt=halt,thread=thread,telemetry=telemetry)


def initialize_backends(c):
    """Same BF16 inference and cache tolerances, explicitly warning-only memory."""
    from .coding_inference import Backend
    from . import coding_control_calibration as controls
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    t=c['telemetry'];t.reset('model_loading');c['guard']()
    c['publish'](stage='load_source');source=Backend(c['config']['models']['source']);t.sample(stage='source_loaded')
    c['publish'](stage='load_receiver');receiver=Backend(c['config']['models']['receiver']);t.sample(stage='both_models_loaded')
    if source.tokenizer.backend_tokenizer.to_str()!=receiver.tokenizer.backend_tokenizer.to_str():raise ValueError('Tokenizer mismatch')
    old=controls.memory_record
    def observed_memory():
        row=old();row['legacy_memory_threshold_passed']=row['passed'];row['passed']=True;row['policy']='amended_memory_warning_only'
        t.sample(stage='native_controls_memory',peak_reset_note='controls reset peaks before each prefix length; see control record')
        return row
    controls.memory_record=observed_memory
    try:
        for role,b in [('source',source),('receiver',receiver)]:
            c['guard']();c['publish'](stage=role+'_native_controls');controls.matched_native_controls(b,c['root']/'controls',role,max_prefix=32768)
    finally:controls.memory_record=old
    gc.collect();torch.cuda.empty_cache();t.reset('after_native_controls_before_experiment')
    write(c['root']/'native_gate.json',{'passed':True,'identity_sha256':digest(c['identity']),'memory_policy':'warning_only','numerical_tolerances_unchanged':True})
    return source,receiver


@torch.no_grad()
def reason_instrumented(backend,prompt_ids,task_id,stream,cap,folder,telemetry,guard,publish,saved_prefix=None,closing=151668):
    """Replay identical original calls while checking saved tokens; abort on drift.

    No cache tensors are checkpointed. RNG state + all tokens + call segmentation
    are saved before guard checks and on exceptions. Resume requires replaying and
    validating the numerical path; a token/RNG checkpoint alone is not a KV dump.
    """
    from .coding_inference import generator,sample,sync
    rng=generator(task_id,stream,backend.device);initial=rng.get_state().cpu().tolist();tokens=[];cache=None;phase='before_prefill'
    started=time.monotonic();folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    saved_prefix=saved_prefix or []
    def save(state='running'):
        write(folder/'resume.json',{'task_id':task_id,'state':state,'phase':phase,'prompt_ids':prompt_ids,'tokens':tokens,
            'rng_state':rng.get_state().cpu().tolist(),'rng_initial':initial,'seed':seed_for(task_id,0,stream),'stream':stream,'cap':cap,
            'cache_sequence_length':cache.get_seq_length() if cache is not None else None,'saved_prefix_length':len(saved_prefix),
            'saved_prefix_checked_tokens':min(len(tokens),len(saved_prefix)),'cache_checkpointed':False,
            'rebuild_schedule':'prompt in512-token chunks; reasoning single tokens; emitted closing token remains uncached'})
    try:
        save();guard();telemetry.sample(task_id=task_id,sequence_length=len(prompt_ids),stage='source_prefill')
        sync();out=backend.prefill_chunked(prompt_ids);cache=out.past_key_values;logits=out.logits;natural=False;early_eos=False
        for i in range(cap):
            phase='sampling';token=sample(logits,rng);tokens.append(token)
            if i<len(saved_prefix) and token!=saved_prefix[i]:
                write(folder/'replay_divergence.json',{'index':i,'saved_token':saved_prefix[i],'actual_token':token,'exact_recovery':False})
                raise RuntimeError('Saved source trajectory diverged; preserve distinct attempt')
            if token==closing:natural=True;break
            if token in backend.eos:early_eos=True;break
            phase='forward_pending';out=backend.forward([token],cache);cache=out.past_key_values;logits=out.logits;phase='forward_complete'
            if i%64==0:
                save();telemetry.sample(stage='source_reasoning',sequence_length=cache.get_seq_length());publish(stage='source_reasoning',task_id=task_id,generated_tokens=len(tokens));guard()
        if len(tokens)<len(saved_prefix):raise RuntimeError('Source terminated before saved prefix')
        if natural:prefix=prompt_ids+tokens[:-1]
        else:
            if early_eos:out=backend.forward([tokens[-1]],cache);cache=out.past_key_values
            prefix=prompt_ids+tokens
        assert cache.get_seq_length()==len(prefix)
        sync();phase='reasoning_complete';save('complete');telemetry.sample(stage=phase,sequence_length=len(prefix))
        record={'task_id':task_id,'prompt_ids':prompt_ids,'reasoning_ids':tokens,'prefix_ids':prefix,'bridge_ids':[closing],
            'natural_boundary':natural,'reasoning_capped':not natural and not early_eos,'early_eos':early_eos,'prefix_cache_length':len(prefix),
            'seed':seed_for(task_id,0,stream),'rng_initial':initial,'rng_after_reasoning':rng.get_state().cpu().tolist(),
            'reasoning_seconds':time.monotonic()-started,'reasoning_text':backend.tokenizer.decode(tokens,skip_special_tokens=False)}
        write(folder/'source_history.json',record);write(folder/'replay_receipt.json',{'saved_prefix_tokens':len(saved_prefix),'saved_prefix_exact':tokens[:len(saved_prefix)]==saved_prefix,
            'original_prefix_rng_available':False,'rng_recovered_by_identical_seed_and_sampling_calls':True,'new_attempt_identity':True,'continuation_after_saved_prefix_previously_unobserved':True})
        return record,cache
    except BaseException as exc:
        try:save('aborted')
        finally:telemetry.failure(exc)
        raise
