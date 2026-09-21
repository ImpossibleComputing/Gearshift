#!/usr/bin/env python3
"""Analyze committed exploratory development transactions; never generate or score."""
import argparse,collections,csv,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha,digest
from scripts.coding_parallel_report import paired_statistics
ROOT=Path(__file__).resolve().parents[1]

def repeated_fourgrams(ids):
    grams=[tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3))]
    return (len(grams)-len(set(grams)))/len(grams) if grams else 0.

def report(run_id,training_root):
    run_ids=run_id.split(',') if isinstance(run_id,str) else list(run_id)
    if len(run_ids)!=len(set(run_ids)):raise ValueError('Repeated development attempt')
    records=[];raw=[];seen=set()
    selection_path=ROOT/training_root/'selection.json'
    expected_selection=json.loads(selection_path.read_text()) if selection_path.exists() else None
    if len(run_ids)>1 and expected_selection is None:raise ValueError('Merged attempts require frozen validation selection')
    wanted=[t['task_id'] for t in json.loads((ROOT/'data/coding_pilot_v1/visible/development.json').read_text())]
    roots=[root for rid in run_ids for root in sorted((ROOT/'results/coding_pilot_v1'/rid).glob('*'))]
    for root in roots:
        if not (root/'identity.json').exists():continue
        identity=json.loads((root/'identity.json').read_text())
        if expected_selection and (root/'complete.json').exists():
            done=json.loads((root/'complete.json').read_text())
            if done['selection_sha256']!=sha(selection_path):raise ValueError('Mixed checkpoint selection')
        for path in sorted(root.glob('tasks/*/complete.json')):
            r=json.loads(path.read_text());tid=r['task_id']
            if tid in seen or tid not in wanted or r['identity_sha256']!=digest(identity):raise ValueError('Duplicate/foreign transaction')
            for name,h in r['files'].items():
                if Path(name).name!=name or sha(path.parent/name)!=h:raise ValueError('Committed output changed')
            seen.add(tid);records.append(r)
            for label,name in [('A','A_original'),('B','B_original'),('D_original','D_original'),('D','D_matched'),('C','C'),('C_initial','C_initial')]:
                o=json.loads((path.parent/(name+'.json')).read_text());answer=o['answer_seconds'];bridge=o['bridge_seconds']
                if expected_selection and label in ('C','C_initial'):
                    selected_hash=expected_selection['checkpoint']['sha256'] if label=='C' else expected_selection['initialization_sha256']
                    if o['checkpoint_sha256']!=selected_hash:raise ValueError('Mixed mapped checkpoint bytes')
                source=o.get('source_reasoning_seconds',0.);own=o.get('own_reasoning_seconds',0.);mapping=o.get('mapping_seconds',0.);prefill=o.get('native_prefill_seconds',0.)
                raw.append({'task_id':tid,'run_id':root.parent.name,'worker_id':root.name,'arm':label,'passed':o['score']['passed'],'category':o['score']['category'],
                    'answer_tokens':len(o['answer_ids']),'eos':o['answer_ended_eos'],'capped':o['answer_capped'],
                    'repeated_token_4gram_fraction':repeated_fourgrams(o['answer_ids']),
                    'source_reasoning_seconds':source,'own_reasoning_seconds':own,'mapping_seconds':mapping,'native_prefill_seconds':prefill,'bridge_seconds':bridge,
                    'generation_excluding_bridge_seconds':max(0.,answer-bridge),'answer_seconds':answer,
                    'source_inclusive_estimate_seconds':source+own+mapping+prefill+answer,
                    'total_model_execution_estimate_seconds':source+own+mapping+prefill+answer,
                    'cache_reconstruction_seconds':o.get('source_cache_reconstruction_seconds',0.),
                    'execution_after_saved_source_history_seconds':o.get('source_cache_reconstruction_seconds',0.)+mapping+prefill+answer,
                    'raw_path':str((path.parent/(name+'.json')).relative_to(ROOT)),'raw_sha256':sha(path.parent/(name+'.json'))})
    summary={'created_epoch':time.time(),'run_ids':run_ids,'training_root':training_root,'coverage':{'expected':40,'committed_complete':len(records),'missing_task_ids':sorted(set(wanted)-seen)},
        'exploratory':True,'confirmation_used':False,'arms':{},'paired':None,'initial_paired':None,
        'D_exact_original_tokens':sum(r['D_tokens_match_original'] for r in records),
        'timing_scope':'Source time is archived original prompt prefill plus reasoning. B has no shared source; its own_reasoning_seconds is included in both total_model_execution_estimate_seconds and the backward-compatible source_inclusive_estimate_seconds. C reconstructs source KV offline in 512-token chunks. Reconstruction is separately reported, not silently called live-online speedup. Answer timers include checkpoints/telemetry; A/B and originalD are historical. Current C and matched D share instrumentation. Task component totals exclude model setup, numerical controls, cache-object assembly, sandbox scoring and archive overhead; allocated-resource charges include those stages.',
        'repetition_definition':'Fraction of token4gram occurrences beyond their first occurrence, denominator all token4gram occurrences. Descriptive; code boilerplate can repeat legitimately.'}
    for arm in ['A','B','C','D','D_original','C_initial']:
        rr=[r for r in raw if r['arm']==arm]
        summary['arms'][arm]={'n':len(rr),'passes':sum(r['passed'] for r in rr),'pass_rate':sum(r['passed'] for r in rr)/len(rr) if rr else None,
            'categories':dict(collections.Counter(r['category'] for r in rr)),'eos':sum(r['eos'] for r in rr),'caps':sum(r['capped'] for r in rr),
            'means':{k:statistics.fmean(r[k] for r in rr) if rr else None for k in ['answer_tokens','repeated_token_4gram_fraction','source_reasoning_seconds','own_reasoning_seconds','total_model_execution_estimate_seconds','mapping_seconds','native_prefill_seconds','bridge_seconds','generation_excluding_bridge_seconds','answer_seconds','source_inclusive_estimate_seconds','cache_reconstruction_seconds','execution_after_saved_source_history_seconds']}}
    if len(records)==40:
        for key,carm in [('paired','C'),('initial_paired','C_initial')]:
            pairs=[{'task_id':r['task_id'],'pass':{'A':r['pass']['A'],'B':r['pass']['B'],'C':r['pass'][carm],'D':r['pass']['D_matched']}} for r in records]
            summary[key]=paired_statistics(pairs)
    out=ROOT/'results/coding_pilot_v1/recovery_review_20260917';out.mkdir(exist_ok=True)
    write(out/'development_summary.json',summary)
    with (out/'development_tasks.csv').open('w',newline='') as f:
        if raw:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    selection=json.loads((ROOT/training_root/'selection.json').read_text()) if (ROOT/training_root/'selection.json').exists() else None
    budget=json.loads((ROOT/'evidence/coding_pilot_v1/control/watchdog_status.json').read_text())
    lines=['# Gearshift recovery results','', 'This is the separately identified September 17 exploratory recovery. Original pilot, follow-up, phase2 and larger-model baseline records remain unchanged. The complete progress article is `publication/GEARSHIFT_PROGRESS_01.md` and can be reviewed independently. Nothing has been published or pushed.','',
        f'Committed development coverage: **{len(records)}/40**. Infrastructure-incomplete tasks are listed in `development_summary.json` and are not scored as wrong model answers. Reserved 200-task confirmation and 40-task second-seed sets remain untouched.','',
        '| Arm | Hidden-test passes | Pass rate | EOS / answer cap |','|---|---:|---:|---:|']
    for arm,a in summary['arms'].items():lines.append(f"|{arm}|{a['passes']}/{a['n']}|{100*a['pass_rate']:.1f}%|{a['eos']} / {a['caps']}|" if a['n'] else f'|{arm}|0/0|—|0 / 0|')
    lines+=['', 'Development attempts: '+', '.join('`'+x+'`' for x in run_ids)+'. Tasks are merged only when disjoint, with identical selected mapper bytes; earlier committed outcomes are preserved.', '', 'D is the matched rerun; D_original preserves the saved comparison. A/B are the saved large-only/independent-small results. C is the validation-selected trained mapper; C_initial is the affine initializer. The initializer was evaluated alongside the selected checkpoint after training; its development outcomes did not select the training schedule or checkpoint.',
            f"Matched D reproduced the original answer tokens on {summary['D_exact_original_tokens']}/{len(records)} committed cases. Source token histories were copied exactly; source caches were rebuilt with the full-prefix training schedule."]
    lines+=['', '| Arm | Test assertion | Runtime error | Syntax error | Code timeout | Mean excess token 4-grams |', '|---|---:|---:|---:|---:|---:|']
    for arm,a in summary['arms'].items():
        if not a['n']:continue
        counts=[a['categories'].get(k,0) for k in ['test_assertion','runtime_error','syntax','timeout']]
        lines.append('|'+arm+'|'+'|'.join(str(n) for n in counts)+f"|{100*a['means']['repeated_token_4gram_fraction']:.2f}%|")
    lines+=['', 'Code timeouts are sandbox execution outcomes. Unavailable infrastructure is excluded from scoring and reported as missing coverage. Repetition is descriptive; legitimate code boilerplate can repeat.']
    if summary['paired']:
        lines+=['','| Paired contrast | Difference (pp) | Gains / losses | 95% paired bootstrap interval (pp) |','|---|---:|---:|---:|']
        for key in ['C-B','C-D','C-A']:
            x=summary['paired']['contrasts'][key];lines.append(f"|{key}|{x['difference']*100:+.1f}|{x['rescues']} / {x['regressions']}|[{x['ci95'][0]*100:+.1f}, {x['ci95'][1]*100:+.1f}]|")
        lines+=['','Intervals use 10,000 paired task resamples. They are exploratory, unadjusted for multiple comparisons, and cannot establish equivalence. The development set was already inspected and is not fresh confirmation.']
    if len(records)==40:
        lines+=['','Mean task timings in seconds for the current instrumented handoffs:', '',
            '| Arm | Archived source prompt + reasoning | Mapping | Receiver prefill | Bridge | Answer after bridge | Source-inclusive estimate | Offline source-cache reconstruction |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        columns=['source_reasoning_seconds','mapping_seconds','native_prefill_seconds','bridge_seconds','generation_excluding_bridge_seconds','source_inclusive_estimate_seconds','cache_reconstruction_seconds']
        for arm in ['C','C_initial','D']:
            mean=summary['arms'][arm]['means'];lines.append('|'+arm+'|'+ '|'.join(f'{mean[k]:.3f}' for k in columns)+'|')
        lines+=['','The source-inclusive estimate substitutes archived source execution for offline cache reconstruction; it is not the wall time of a live online handoff. Reconstruction is measured separately. Unequal answer quality, lengths and termination prevent treating a time reduction alone as an equal-quality speedup.']
    if len(records)==40:
        lines += ['', f"The independent smaller model B additionally spent a mean {summary['arms']['B']['means']['own_reasoning_seconds']:.3f} seconds on its own archived prompt/reasoning, for {summary['arms']['B']['means']['total_model_execution_estimate_seconds']:.3f} seconds including its answer. Its reasoning is not free or omitted from the total."]
    if selection:lines+=['',f"Selected checkpoint: step {selection['step']}, validation KL {selection['validation_kl']:.6f}, SHA-256 `{selection['checkpoint']['sha256']}`. Selection used all 21 fixed validation histories and no development outcomes. The corpus contains 104/128 training and 21/32 validation histories; completion selection may underrepresent long histories. Convergence is untested."]
    lines+=['','Per-task raw paths/hashes, parse/execution categories, repeated-token measurements and every timing component are in `results/coding_pilot_v1/recovery_review_20260917/development_tasks.csv` and `development_summary.json`.',
        '',summary['timing_scope'],'',f"Cumulative conservative usage at this report: ${budget['upper_usd']:.2f}, {budget['gpu_hours']:.3f} GPU-hours, against the unchanged $1,000/500-hour limits. Active GPU count: {budget['active_gpu_count']}; active resources: {budget['active_resources']}. This is the task ledger estimate, not an invoice. The preserved original 250 GB failure volume remains approximately $0.04/hour.",
        '','Revised code/configs, test logs, telemetry, memory-failure/recovery evidence, exact corpus and checkpoint identities are included in the rolling review ZIP. Source reasoning trajectories are included; model weights, feature/cache/checkpoint tensors, environments and credentials are excluded with hash/location/regeneration inventories. See `REPRODUCE_RECOVERY.md`.']
    lines+=['','## Engineering recovery and actual training']
    for proot in sorted((ROOT/'results/coding_pilot_v1').glob('recovery_probe_20260917_*/*')):
        telemetry=proot/'memory_telemetry.jsonl'
        if not telemetry.exists():continue
        samples=[json.loads(line) for line in telemetry.read_text().splitlines() if line.strip()]
        measured=[x for x in samples if 'allocated' in x]
        done=json.loads((proot/'complete.json').read_text()) if (proot/'complete.json').exists() else None
        failure=json.loads((proot/'failure.json').read_text()) if (proot/'failure.json').exists() else None
        memory={'root':str(proot.relative_to(ROOT)),'sample_count':len(measured),'warnings':dict(collections.Counter(w for x in measured for w in x.get('warnings',[]))),
            'maximum_observed_allocated':max((x['allocated'] for x in measured),default=None),'maximum_observed_reserved':max((x['reserved'] for x in measured),default=None),
            'minimum_observed_driver_free':min((x['free'] for x in measured),default=None),'completion':done,'failure':failure}
        generation=[x for x in measured if x.get('stage')=='source_reasoning']
        if generation:
            memory['source_generation']={
                'maximum_observed_allocated_GiB':max(x['allocated'] for x in generation)/2**30,
                'maximum_observed_reserved_GiB':max(x['reserved'] for x in generation)/2**30,
                'minimum_observed_free_GiB':min(x['free'] for x in generation)/2**30,
                'maximum_allocator_retries':max(x['allocator'].get('num_alloc_retries',0) for x in generation),
                'maximum_allocator_ooms':max(x['allocator'].get('num_ooms',0) for x in generation)}
        write(out/(proot.parent.name+'_memory_summary.json'),memory)
        from scripts.plot_recovery_memory import plot
        figure=plot(proot,out/(proot.parent.name+'_memory.png'))
        if figure:lines+=['',f'![Single-trajectory memory diagnostic]({figure.relative_to(ROOT)})','']
        lines.append(f"- `{proot.parent.name}`: {len(measured)} memory samples; warning counts {memory['warnings']}; completed={done is not None}; allocation readings and reset scopes retained in raw telemetry.")
        if done:lines.append(f"  Saved prefix matched exactly through {done['saved_prefix_tokens']} tokens; new source reasoning reached {done['reasoning_tokens']} tokens; paired extraction completed={done['paired_extraction_completed']}. This is engineering evidence, not a code-quality result.")
        if generation:
            g=memory['source_generation']
            lines.append(f"  During source generation, live allocation reached {g['maximum_observed_allocated_GiB']:.2f} GiB and reservation {g['maximum_observed_reserved_GiB']:.2f} GiB; minimum sampled driver free memory was {g['minimum_observed_free_GiB']:.3f} GiB. Allocator counters reached {g['maximum_allocator_retries']} allocation retries and {g['maximum_allocator_ooms']} out-of-memory failures. Counters and periodic observations are not a complete per-allocation trace.")
        if failure:lines.append(f"  Preserved failure: {failure['exception']}: {failure['error']}.")
    lines+=['', 'The amended lifecycle also clears released allocator blocks and resets peaks once after native controls. Exact token replay does not mean identical allocator state. The old incident still lacks its numeric readings; successful current recovery cannot retrospectively prove its cause. The recovery uses warnings for the old memory thresholds, strict numerical controls, durable token/RNG checkpoints, and bounded network setup after one observed TLS handshake stall.','']
    tr=ROOT/training_root/'complete.json'
    if tr.exists():
        actual=json.loads(tr.read_text())
        schedule=json.loads((tr.parent/'schedule.json').read_text())
        used=sorted({part['task_id'] for item in schedule if item['step']<=actual['steps'] for part in item['segments']})
        write(out/'functional_training_summary.json',{'completion':actual,'selection':selection,'unique_histories_in_committed_updates':len(used),'task_ids_in_committed_updates':used,'affine_initialization_histories':104,'available_training_histories':104,'partial_interrupted_updates_included':False})
        lines.append(f"Committed functional updates used {len(used)} distinct histories from the fixed 104-history training corpus. The affine initializer used all 104. Partial interrupted updates, if any, are excluded from committed prediction counts.")
        lines.append(f"Actual functional run: {actual['steps']} full updates, {actual['training_predictions']} committed gradient-bearing predictions, {actual['wall_seconds']:.1f} seconds, stop={actual['stop_reason']}; objective natural_handoff_boundary. Fresh affine initialization used 6,656 sampled positions from 104 training histories. All completed histories were natural source endings; training reasoning max 20,079 tokens and validation max 15,690, while the four interrupted originals already exceeded 20,000. This survivor bias is material and explicitly preserved.")
    lines+=['The coding regression suite passed 237 tests before initial dispatch and 254 after the capacity-completion repair. Relevant exact test logs are in `evidence/coding_pilot_v1/recovery_20260917/`. Tests are not counted as experimental results.']
    (ROOT/'RECOVERY_RESULTS.md').write_text('\n'.join(lines)+'\n');return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_id');p.add_argument('training_root');a=p.parse_args();print(json.dumps(report(a.run_id,a.training_root)['coverage']))
