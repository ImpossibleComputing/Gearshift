#!/usr/bin/env python3
"""Bounded model protocol and fixed-development A/B/D gate. No mapper training."""
import gc,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha,bind,digest
ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'evidence/coding_pilot_v1'; R=ROOT/'results/coding_pilot_v1/preflight_v3'

def main():
    import torch,transformers
    from gearshift.coding_inference import Backend,reason,answer,native_controls,memory_record,sync
    from gearshift.coding_control_calibration import matched_native_controls
    from gearshift.coding_resume import load_parent,verify_prefix,verify_segment
    from gearshift.coding_sandbox import extract,score
    os.chdir(ROOT);R.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/coding_pilot_v1/pilot.json').read_text())
    allocation=json.loads((E/'allocation.json').read_text())
    deadline=allocation['preflight_deadline_epoch']-120 # reserve final sync/cleanup
    def guard():
        if time.time()>deadline or (E/'STOP').exists():raise TimeoutError('Preflight ceiling: checkpoint and stop')
    def progress(**kw):
        guard();write(E/'worker_status.json',{'state':'running','heartbeat_epoch':time.time(),**kw})
    sandbox=json.loads((E/'sandbox_gate.json').read_text())
    assert sandbox['passed'] and len(sandbox.get('canonical_development',{}))==2
    lock={'torch':torch.__version__,'transformers':transformers.__version__,'python':sys.version,'platform':platform.platform(),
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'image_digest':allocation['image_digest'],
        'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)}
    write(E/'runtime_lock.json',lock)
    implementation={str(p.relative_to(ROOT)):sha(p) for pattern in ['gearshift/coding_*.py','scripts/coding_*.py'] for p in ROOT.glob(pattern)}
    implementation['gearshift/core.py']=sha(ROOT/'gearshift/core.py')
    parent_root=ROOT/'results/coding_pilot_v1/preflight_v2'
    parent=json.loads((parent_root/'identity.json').read_text())
    for f in ['gearshift/coding_inference.py','gearshift/coding_sandbox.py','scripts/coding_sandbox_child.py']:
        assert parent['implementation'][f]==implementation[f],('Numerical generation or scoring implementation changed',f)
    assert parent['config_sha256']==sha(ROOT/'configs/coding_pilot_v1/pilot.json')
    assert parent['data_identity']==json.loads((ROOT/'data/coding_pilot_v1/identity.json').read_text())
    ident={'config_sha256':sha(ROOT/'configs/coding_pilot_v1/pilot.json'),'data_identity':json.loads((ROOT/'data/coding_pilot_v1/identity.json').read_text()),
        'runtime_sha256':digest(lock),'implementation':implementation,'reasoning_cap':16384,'answer_cap':4096,'arms':['A','B','D'],
        'control_protocol_sha256':sha(ROOT/'configs/coding_pilot_v1/control_protocol_v2.json'),
        'sandbox_gate_sha256':sha(E/'sandbox_gate.json'),'parent_identity_sha256':sha(parent_root/'identity.json'),
        'recovery':'Recompute the identical seeded streams and verify every saved token prefix. Preserve all earlier evidence; abort on any divergence, without replacing a sample.'}
    identity=bind(R/'identity.json',ident)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    # Deterministic controls are separate from sampled quality runs.
    torch.use_deterministic_algorithms(True)
    progress(stage='load_source');source=Backend(cfg['models']['source'])
    progress(stage='load_receiver');receiver=Backend(cfg['models']['receiver'])
    assert source.tokenizer.backend_tokenizer.to_str()==receiver.tokenizer.backend_tokenizer.to_str()
    for text in ['print("café λ")','<think>\n</think>','x = "\\n"\n','def f(a: list[int]):\n return a[::-1]']:
        assert source.tokenizer.encode(text)==receiver.tokenizer.encode(text)
    progress(stage='source_native_controls');matched_native_controls(source,R/'controls','source')
    progress(stage='receiver_native_controls');matched_native_controls(receiver,R/'controls','receiver')
    write(R/'native_gate.json',{'passed':True,'identity_sha256':identity})
    tasks=json.loads((ROOT/'data/coding_pilot_v1/visible/development.json').read_text())
    private=json.loads((ROOT/'data/coding_pilot_v1/private/development.json').read_text())
    completed=[];start=time.monotonic()
    for index,task in enumerate(tasks):
        tid=task['task_id'];folder=R/'tasks'/tid.replace('/','__');folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():
            row=json.loads((folder/'complete.json').read_text());assert row['identity_sha256']==identity;completed.append(row);continue
        progress(stage='development',task_index=index,task_id=tid,arm='A')
        previous,previous_complete=load_parent(parent_root,tid)
        if previous:write(folder/'recovery_lineage.json',{'parent_identity_sha256':sha(parent_root/'identity.json'),
            'saved_segment_lengths':{k:len(v) for k,v in previous.items()},'completed_segments':previous_complete})
        segment='source_reasoning'
        def cb(tokens):
            verify_prefix(tokens,previous.get(segment,[]))
            write(folder/f'partial_{segment}.json',{'task_id':tid,'segment':segment,'token_ids':tokens,'epoch':time.time(),'identity_sha256':identity})
            progress(stage='development',task_index=index,task_id=tid,segment=segment,generated_tokens=len(tokens))
        torch.cuda.reset_peak_memory_stats()
        history,cache=reason(source,task['prompt_ids'],tid,'source_reasoning',16384,callback=cb)
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
        segment='small_reasoning';small_history,cache=reason(receiver,task['prompt_ids'],tid,'small_reasoning',16384,callback=cb)
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
        elapsed=time.monotonic()-start;forecast=elapsed/len(completed)*(40-len(completed))
        summary={'completed_tasks':len(completed),'total_tasks':40,'passes':{arm:sum(r['pass'][arm] for r in completed) for arm in ['A','B','D']},
            'preflight_remaining_seconds':deadline-time.time(),'remaining_baseline_forecast_seconds':forecast,
            'forecast_exceeds_preflight':forecast>deadline-time.time(),'source_cap_count':sum(r['source_capped'] for r in completed),'small_cap_count':sum(r['small_capped'] for r in completed)}
        write(R/'progress.json',summary);print(json.dumps(summary),flush=True)
        if not mem['passed']:raise RuntimeError('Generation memory headroom gate failed')
        if len(completed)>=5 and summary['forecast_exceeds_preflight']:
            write(E/'ALERT.json',{'reason':'Measured forecast exceeds preflight ceiling','summary':summary,'approval_required_to_extend':True})
            raise TimeoutError('Measured preflight forecast exceeds ceiling; stop without shrinking cohort')
    counts={arm:sum(r['pass'][arm] for r in completed) for arm in ['A','B','D']}
    rules={'large_minimum':counts['A']>=12,'small_range':4<=counts['B']<=34,'large_small_gap':counts['A']-counts['B']>=4,'text_small_gap':counts['D']-counts['B']>=2}
    cap_gate=all(sum(r[k] for r in completed)<=4 for k in ['source_capped','small_capped'])
    write(R/'baseline_gate.json',{'passed':all(rules.values()) and cap_gate,'rules':rules,'counts':counts,'cap_gate':cap_gate,
        'remaining_required_before_training':['Actual full-prefix mapper gradient memory test','Matched initialization and training implementation'],
        'protocol_cap_amendment_needed':not cap_gate,'confirmation_not_run':True})
    write(E/'worker_status.json',{'state':'preflight_baselines_finished','finished_epoch':time.time(),'baseline_gate_passed':all(rules.values()) and cap_gate,
        'training_authorized_only_after_remaining_memory_and_protocol_gates':True})

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(E/'worker_status.json',{'state':'stopped','finished_epoch':time.time(),'error':str(exc),'exception':type(exc).__name__})
        traceback.print_exc();sys.exit(1)
