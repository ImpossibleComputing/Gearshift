#!/usr/bin/env python3
"""Recover the saved final development case after a provider-status interruption."""
import gc,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha,bind,digest
ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'evidence/coding_pilot_v1'; R=ROOT/'results/coding_pilot_v1/preflight_cap_v1_recovery1'
SPEC={}

def status(record):
    write(E/'worker_status.json',record)
    if SPEC:
        normalized={'preflight_baselines_finished':'complete','stopped':'failed'}.get(record['state'],record['state'])
        write(Path(SPEC['worker_status_path']),{**record,'state':normalized,'stage_identity':SPEC['stage_identity'],
            'worker_id':SPEC['worker_id'],'heartbeat_epoch':time.time()})


def main():
    global R,SPEC
    SPEC=json.loads(Path(os.environ['GEARSHIFT_WORKER_SPEC']).read_text()) if os.environ.get('GEARSHIFT_WORKER_SPEC') else {}
    if SPEC:R=ROOT/SPEC['result_root']
    import torch,transformers
    from gearshift.coding_inference import Backend,reason,answer,native_controls,memory_record,sync
    from gearshift.coding_control_calibration import matched_native_controls
    from gearshift.coding_resume import load_parent,verify_prefix,verify_segment
    from gearshift.coding_reuse import compatible_parent,reuse_completed
    from gearshift.coding_sandbox import extract,score
    os.chdir(ROOT)
    if R.exists():raise ValueError('Cap-amendment attempt already exists; preserve outcomes rather than retry')
    R.mkdir(parents=True)
    cfg=json.loads((ROOT/'configs/coding_pilot_v1/pilot.json').read_text())
    allocation_path=ROOT/SPEC.get('allocation_path','evidence/coding_pilot_v1/allocation_cap_recovery1.json')
    allocation=json.loads(allocation_path.read_text())
    deadline=allocation.get('preflight_deadline_epoch',SPEC.get('deadline_epoch',0))-120 # reserve final sync/cleanup
    def guard():
        if time.time()>deadline:raise TimeoutError('Authorized worker deadline reached')
        if (E/'STOP').exists():raise InterruptedError('Controller requested a checkpointed stop')
    def progress(**kw):
        guard();status({'state':'running','heartbeat_epoch':time.time(),**kw})
    sandbox=json.loads((E/'sandbox_gate.json').read_text())
    assert sandbox['passed'] and len(sandbox.get('canonical_development',{}))==2
    lock={'torch':torch.__version__,'transformers':transformers.__version__,'python':sys.version,'platform':platform.platform(),
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'image_digest':allocation['image_digest'],
        'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)}
    write(E/'runtime_lock.json',lock)
    implementation={str(p.relative_to(ROOT)):sha(p) for pattern in ['gearshift/coding_*.py','scripts/coding_*.py'] for p in ROOT.glob(pattern)}
    implementation['gearshift/core.py']=sha(ROOT/'gearshift/core.py')
    original_root=ROOT/'results/coding_pilot_v1/preflight_v5'
    parent_root=ROOT/'results/coding_pilot_v1/preflight_cap_v1'
    from gearshift.coding_cap_amendment import NEW_CAP
    from gearshift.coding_cap_recovery import validate_recovery,recovery_constraints
    checkpoint=validate_recovery(parent_root,original_root,ROOT)
    declaration=ROOT/'configs/coding_pilot_v1/cap_amendment_v1.json'
    if allocation.get('cap_amendment_sha256',SPEC.get('worker_fields',{}).get('cap_amendment_sha256'))!=sha(declaration):raise ValueError('Cap amendment declaration differs')
    parent=checkpoint['parent_identity']
    if allocation['approval_sha256']!=sha(E/'parallel_500h_approved.json'):
        raise ValueError('Allocation authorization identity differs')
    if allocation.get('parent_checkpoint',SPEC.get('worker_fields',{}).get('parent_checkpoint'))!= {k:v for k,v in checkpoint.items() if k not in ['parent_identity','original_identity']}:
        raise ValueError('Allocated checkpoint differs from verified parent')
    for f in ['gearshift/coding_inference.py','gearshift/coding_sandbox.py','scripts/coding_sandbox_child.py']:
        assert parent['implementation'][f]==implementation[f],('Numerical generation or scoring implementation changed',f)
    assert parent['config_sha256']==sha(ROOT/'configs/coding_pilot_v1/pilot.json')
    assert parent['data_identity']==json.loads((ROOT/'data/coding_pilot_v1/identity.json').read_text())
    ident={'config_sha256':sha(ROOT/'configs/coding_pilot_v1/pilot.json'),'data_identity':json.loads((ROOT/'data/coding_pilot_v1/identity.json').read_text()),
        'runtime_sha256':digest(lock),'implementation':implementation,'reasoning_cap':NEW_CAP,'cap_amendment_sha256':sha(declaration),'cap_amendment_number':1,'answer_cap':4096,'arms':['A','B','D'],
        'control_protocol_sha256':sha(ROOT/'configs/coding_pilot_v1/control_protocol_v2.json'),
        'sandbox_gate_sha256':sha(E/'sandbox_gate.json'),'parent_identity_sha256':sha(parent_root/'identity.json'),
        'recovery':'Carry completed measurements verbatim with explicit original measurement identities; no new draw or rescoring. Any incomplete saved segment must reproduce exactly.',
        'budget_extension_sha256':sha(E/'parallel_500h_approved.json'),
        'allocation_sha256':sha(allocation_path),'stage_identity':SPEC.get('stage_identity'),'interrupted_cap_recovery':True,'original_identity_sha256':sha(original_root/'identity.json')}
    identity=bind(R/'identity.json',ident)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    # Deterministic controls are separate from sampled quality runs.
    torch.use_deterministic_algorithms(True)
    progress(stage='load_source');source=Backend(cfg['models']['source'])
    progress(stage='load_receiver');receiver=Backend(cfg['models']['receiver'])
    assert source.tokenizer.backend_tokenizer.to_str()==receiver.tokenizer.backend_tokenizer.to_str()
    for text in ['print("café λ")','<think>\n</think>','x = "\\n"\n','def f(a: list[int]):\n return a[::-1]']:
        assert source.tokenizer.encode(text)==receiver.tokenizer.encode(text)
    progress(stage='source_native_controls');matched_native_controls(source,R/'controls','source',max_prefix=32768)
    progress(stage='receiver_native_controls');matched_native_controls(receiver,R/'controls','receiver',max_prefix=32768)
    write(R/'native_gate.json',{'passed':True,'identity_sha256':identity})
    tasks=json.loads((ROOT/'data/coding_pilot_v1/visible/development.json').read_text())
    private=json.loads((ROOT/'data/coding_pilot_v1/private/development.json').read_text())
    completed=[];start=time.monotonic();generated=0;reused=0
    repeat_ids=set(checkpoint['remaining_task_ids'])
    parent_progress=json.loads((parent_root/'progress.json').read_text())
    parent_average=parent_progress['amendment_generation_wall_seconds']/parent_progress['regenerated_tasks']
    def save_progress():
        elapsed=time.monotonic()-start
        forecast=(elapsed/generated if generated else parent_average*1.5)*(len(repeat_ids)-generated)
        summary={'completed_tasks':len(completed),'total_tasks':40,
            'passes':{arm:sum(r['pass'][arm] for r in completed) for arm in ['A','B','D']},
            'preflight_remaining_seconds':deadline-time.time(),'amendment_generation_wall_seconds':elapsed,
            'reused_parent_tasks':reused,'regenerated_tasks':generated,'total_affected_tasks':len(repeat_ids),
            'remaining_baseline_forecast_seconds':forecast,'forecast_exceeds_preflight':forecast>deadline-time.time(),
            'source_cap_count':sum(r['source_capped'] for r in completed),'small_cap_count':sum(r['small_capped'] for r in completed),
            'reasoning_cap_both_models':NEW_CAP,'amendment_number':1,'infrastructure_recovery':True}
        write(R/'progress.json',summary);print(json.dumps(summary),flush=True)
        if generated and summary['forecast_exceeds_preflight']:
            write(E/'ALERT.json',{'reason':'Amendment forecast exceeds cumulative approved allowance','summary':summary,'approval_required_to_extend':True})
            raise TimeoutError('Amendment forecast exceeds ceiling; preserve full fixed cohort membership')
    for index,task in enumerate(tasks):
        tid=task['task_id'];folder=R/'tasks'/tid.replace('/','__')
        upstream=parent_root/'tasks'/tid.replace('/','__')
        if tid not in repeat_ids:
            row=reuse_completed(upstream,folder,parent,ident,tid,str(upstream.relative_to(ROOT)))
            completed.append(row);reused+=1;save_progress();continue
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():
            row=json.loads((folder/'complete.json').read_text());assert row['identity_sha256']==identity;completed.append(row);continue
        progress(stage='development',task_index=index,task_id=tid,arm='A')
        previous,previous_complete,lineage=recovery_constraints(original_root,checkpoint['original_identity'],parent_root,tid)
        write(folder/'cap_amendment_lineage.json',{'parent_identity_sha256':sha(parent_root/'identity.json'),
            'saved_segment_lengths':{k:len(v) for k,v in previous.items()},'completed_segments':previous_complete,**lineage})
        segment='source_reasoning'
        def cb(tokens):
            verify_prefix(tokens,previous.get(segment,[]))
            write(folder/f'partial_{segment}.json',{'task_id':tid,'segment':segment,'token_ids':tokens,'epoch':time.time(),'identity_sha256':identity})
            progress(stage='development',task_index=index,task_id=tid,segment=segment,generated_tokens=len(tokens))
        torch.cuda.reset_peak_memory_stats()
        history,cache=reason(source,task['prompt_ids'],tid,'source_reasoning',NEW_CAP,callback=cb)
        verify_segment(history['reasoning_ids'],'source_reasoning',previous,previous_complete)
        write(folder/'source_history.json',history)
        segment='answer_A';a=answer(source,history,cache,tid,'answer_large',4096,callback=cb);del cache
        verify_segment(a['answer_ids'],segment,previous,previous_complete)
        a.update(condition='large_only',source_reasoning_seconds=history['reasoning_seconds'],historical_receiver_prefill_tokens=0)
        a['code']=extract(a['answer_text']);write(folder/'A.json',a)
        a['score']=score(a['code'],private[tid],guard);write(folder/'A.json',a)
        progress(stage='development',task_index=index,task_id=tid,arm='D')
        sync();before=time.monotonic();native=receiver.prefill_chunked(history['prefix_ids']);sync();prefill_seconds=time.monotonic()-before
        segment='answer_D';d=answer(receiver,history,native.past_key_values,tid,'answer_small',4096,callback=cb);del native
        verify_segment(d['answer_ids'],segment,previous,previous_complete)
        d.update(condition='text_handoff',source_reasoning_seconds=history['reasoning_seconds'],native_prefill_seconds=prefill_seconds,historical_receiver_prefill_tokens=len(history['prefix_ids']))
        d['code']=extract(d['answer_text']);write(folder/'D.json',d)
        d['score']=score(d['code'],private[tid],guard);write(folder/'D.json',d)
        progress(stage='development',task_index=index,task_id=tid,arm='B')
        segment='small_reasoning';small_history,cache=reason(receiver,task['prompt_ids'],tid,'small_reasoning',NEW_CAP,callback=cb)
        verify_segment(small_history['reasoning_ids'],segment,previous,previous_complete)
        write(folder/'small_history.json',small_history)
        segment='answer_B';b=answer(receiver,small_history,cache,tid,'answer_small',4096,callback=cb);del cache
        verify_segment(b['answer_ids'],segment,previous,previous_complete)
        b.update(condition='small_only',own_reasoning_seconds=small_history['reasoning_seconds'],historical_receiver_prefill_tokens=0)
        b['code']=extract(b['answer_text']);write(folder/'B.json',b)
        b['score']=score(b['code'],private[tid],guard);write(folder/'B.json',b)
        gc.collect();torch.cuda.empty_cache();mem=memory_record()
        row={'task_id':tid,'identity_sha256':identity,'pass':{'A':a['score']['passed'],'B':b['score']['passed'],'D':d['score']['passed']},
            'source_capped':history['reasoning_capped'],'small_capped':small_history['reasoning_capped'],
            'source_early_eos':history['early_eos'],'small_early_eos':small_history['early_eos'],
            'memory':mem,'files':{p.name:sha(p) for p in folder.glob('*.json') if p.name!='complete.json'}}
        write(folder/'complete.json',row);completed.append(row)
        generated+=1;save_progress()
        if not mem['passed']:raise RuntimeError('Generation memory headroom gate failed')
    counts={arm:sum(r['pass'][arm] for r in completed) for arm in ['A','B','D']}
    rules={'large_minimum':counts['A']>=12,'small_range':4<=counts['B']<=34,'large_small_gap':counts['A']-counts['B']>=4,'text_small_gap':counts['D']-counts['B']>=2}
    cap_gate=all(sum(r[k] for r in completed)<=4 for k in ['source_capped','small_capped'])
    write(R/'baseline_gate.json',{'passed':all(rules.values()) and cap_gate,'rules':rules,'counts':counts,'cap_gate':cap_gate,
        'remaining_required_before_training':['Actual full-prefix mapper gradient memory test','Matched initialization and training implementation'],
        'protocol_cap_amendment_needed':False,'cap_amendment_exhausted':not cap_gate,'cap_amendment_performed':True,'confirmation_not_run':True})
    status({'state':'preflight_baselines_finished','finished_epoch':time.time(),'baseline_gate_passed':all(rules.values()) and cap_gate,
        'training_authorized_only_after_remaining_memory_and_protocol_gates':True})

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status({'state':'stopped','finished_epoch':time.time(),'error':str(exc),'exception':type(exc).__name__})
        traceback.print_exc();sys.exit(1)
