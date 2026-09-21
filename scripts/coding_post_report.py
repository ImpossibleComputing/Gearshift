#!/usr/bin/env python3
"""Compact-record reporting only; never runs models, generated code or tests."""
import ast,collections,csv,json,statistics,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gearshift.coding_control import sha,write
from scripts.coding_recovery_report import repeated_fourgrams
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence/coding_pilot_v1/post_progress01'
OUT=ROOT/'results/coding_pilot_v1/post_progress01_report'

def paired(rows,labels=('M','D','P')):
    if len(rows)!=40:raise ValueError('Paired results require all40 completed cases')
    rng=np.random.default_rng(20260917);indices=rng.integers(0,40,size=(10000,40));result={}
    for other in labels:
        a=np.array([r['H'] for r in rows],dtype=float);b=np.array([r[other] for r in rows],dtype=float);delta=a-b
        result['H-'+other]={'difference':float(delta.mean()),'ci95':np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),'gains':int(((a==1)&(b==0)).sum()),'losses':int(((a==0)&(b==1)).sum())}
    return {'contrasts':result,'method':'10000 joint paired task bootstrap resamples; seed20260917; percentile95%; exploratory, unadjusted. Nonsignificance is not equivalence.'}

def report():
    E.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    if (ROOT/'.git').exists():
        tag=git('rev-parse','gearshift-progress-01^{}')
        if tag!='64725974fa55459350d1c9d09037bab64d0c5ec6':raise ValueError('Publication tag moved')
        change={'tag':'gearshift-progress-01','tag_commit':tag,'tag_object':git('rev-parse','gearshift-progress-01'),'branch':git('branch','--show-current'),'head':git('rev-parse','HEAD'),'subsequent_commits':git('log','--format=%H %s','gearshift-progress-01..HEAD').splitlines(),'pushed':False}
        write(E/'current_git_verification.json',change)
        (E/'change_summary.txt').write_text(git('diff','--stat','gearshift-progress-01','--','gearshift','scripts','tests','configs/coding_pilot_v1/post_progress01','publication','POST_PROGRESS_01_RESULTS.md','REPRODUCE_POST_PROGRESS_01.md')+'\n')
        costs=json.loads((ROOT/'evidence/coding_pilot_v1/control/watchdog_status.json').read_text());write(E/'review_cost_snapshot.json',costs)
    else:
        change=json.loads((E/'current_git_verification.json').read_text());tag=change['tag_commit']
        if tag!='64725974fa55459350d1c9d09037bab64d0c5ec6':raise ValueError('Saved publication receipt differs')
        costs=json.loads((E/'review_cost_snapshot.json').read_text())
    lines=['# Post-Progress 01 diagnostic results','', 'This separate, exploratory diagnostic pass does not revise the frozen article or historical results. The editorial master remains unchanged. This file reports actual records; prepared code is not counted as an executed experiment.','',f'Publication tag: `gearshift-progress-01` → `{tag}`. Local only; not pushed or published. Branch: `'+change['branch']+'`. Full subsequent commit identities are in `evidence/coding_pilot_v1/post_progress01/current_git_verification.json`.','',
        '## Frozen result and source audit','', 'Progress 01: A=34/40, B=27/40, M=3/40, D=34/40, initial mapper=0/40. Selected update 96 had 3,072 supervised prediction positions. Validation kept improving; no plateau or trained-checkpoint ranking failure was established.', '',
        'The all-40 visible-prompt/code audit confirms 13 of 16 runtime failures directly cite missing explicitly requested entrypoints. A fourteenth omits its interface but fails earlier on input parsing. The owner-named tree example ignores the requested vertex weights and prints the maximum unweighted distance sum, whereas the prompt asks for the minimum weighted sum. This is a selected explanatory example, not a prevalence estimate. Exact visible prompt/code and hashes are preserved. No output was repaired before scoring.','',
        '## Actual-model numerical path comparison','']
    numerical=[]
    for p in sorted((ROOT/'results/coding_pilot_v1').glob('post_progress01_numerical_*/*/numerical_comparison.json')):
        d=json.loads(p.read_text());numerical.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'complete':d['complete'],'passed':d['passed'],'rows':[{k:r[k] for k in ['task_id','cache_kind','prefix_tokens','cross_path','manual_vs_default_deployed','batch_vs_sequential','gradients','cache_unchanged','clone_no_alias','passed']} for r in d['rows']]})
        lines += [f'- `{p.parent.parent.name}`: {len(d["rows"])} native/mapped history checks, complete={d["complete"]}, strict path comparison passed={d["passed"]}.']
        for r in d['rows']:
            lines += [f'  {r["task_id"]}, {r["cache_kind"]}, {r["prefix_tokens"]} history tokens: manual/deployed maximum logit difference {r["manual_vs_default_deployed"]["max_abs"]:.8g}; gradient difference {r["gradients"]["cross_path_max_abs"]:.8g}; deployment batch/single-token KL {r["batch_vs_sequential"]["kl"]:.8g}.']
    if not numerical:lines += ['No completed actual-model numerical comparison has been imported yet. Later training must wait for this check and investigation of material discrepancies.']
    write(OUT/'numerical_summary.json',numerical)
    interpretation=E/'numerical_interpretation.json'
    if interpretation.exists():
        review=json.loads(interpretation.read_text());lines+=['',review['interpretation'],'',f"Maximum checked per-position batch-versus-token KL: {review['batch_vs_token_maximum_per_position_kl']:.8g}; all {review['tested_answer_positions_across_cache_kinds']} checked top-ranked predictions match. This is measured sensitivity, not a statement that sampled trajectories must match."]
    lines += ['', 'Gradients cover the final two first-layer value-cache positions and four early continuation tokens, not all keys, layers or mapper parameters. Repeated-path controls set the comparison envelope. Batch versus one-token deployment rounding is recorded separately and is not used to excuse a manual-path discrepancy.','', '## Frozen prompt-preservation comparison','']
    tasks={r['task_id']:r for r in json.loads((ROOT/'data/coding_pilot_v1/visible/development.json').read_text())};raw=[];pairs=[];seen=set();hybrid_roots=[]
    for root in sorted((ROOT/'results/coding_pilot_v1').glob('post_progress01_hybrid_*/hybrid')):
        if not (root/'identity.json').exists():continue
        hybrid_roots.append(str(root.relative_to(ROOT)))
        for p in sorted(root.glob('tasks/*/complete.json')):
            transaction=json.loads(p.read_text());tid=transaction['task_id']
            if tid in seen:raise ValueError('Duplicate task in prompt comparison; select a declared attempt explicitly')
            if any(sha(p.parent/n)!=h for n,h in transaction['files'].items()):raise ValueError('Outcome transaction changed')
            seen.add(tid);pair={'task_id':tid}
            for arm in ['M','H','D','P','A_original','B_original']:
                f=p.parent/(arm+'.json');r=json.loads(f.read_text());wanted=__import__('re').findall(r'^\s*def (\w+)\(',tasks[tid]['prompt'],__import__('re').M)
                try:tree=ast.parse(r['code']);functions=[n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))];classes=[n.name for n in ast.walk(tree) if isinstance(n,ast.ClassDef)];syntax=True
                except SyntaxError:functions=[];classes=[];syntax=False
                pair[arm]=bool(r['score']['passed']);source=r.get('source_reasoning_seconds',0);own=r.get('own_reasoning_seconds',0);mapping=r.get('mapping_seconds',0);prefill=r.get('native_prefill_seconds',0);splice=r.get('splice_seconds',0)
                raw.append({'task_id':tid,'arm':arm,'passed':r['score']['passed'],'category':r['score']['category'],'missing_requested_entrypoint':bool(wanted) and (not set(wanted)<=set(functions) or 'Solution' not in classes),'syntax_valid':syntax,'answer_tokens':len(r['answer_ids']),'eos':r['answer_ended_eos'],'capped':r['answer_capped'],'repeated_token_4gram_fraction':repeated_fourgrams(r['answer_ids']),'source_reasoning_seconds':source,'own_reasoning_seconds':own,'mapping_seconds':mapping,'splice_seconds':splice,'native_prefill_seconds':prefill,'historical_receiver_prefill_tokens':r.get('historical_receiver_prefill_tokens',0),'answer_seconds':r['answer_seconds'],'bridge_seconds':r['bridge_seconds'],'source_cache_reconstruction_seconds':r.get('source_cache_reconstruction_seconds',0),'source_inclusive_estimate_seconds':source+own+mapping+prefill+splice+r['answer_seconds'],'raw':str(f.relative_to(ROOT)),'sha256':sha(f)})
            pairs.append(pair)
    hybrid={'completed_tasks':len(seen),'expected':40,'roots':hybrid_roots,'arms':{},'paired':paired(pairs) if len(pairs)==40 else None}
    for arm in ['M','H','D','P','A_original','B_original']:
        rr=[r for r in raw if r['arm']==arm]
        if rr:hybrid['arms'][arm]={'n':len(rr),'passes':sum(r['passed'] for r in rr),'categories':dict(collections.Counter(r['category'] for r in rr)),'missing_requested_entrypoints':sum(r['missing_requested_entrypoint'] for r in rr),'eos':sum(r['eos'] for r in rr),'caps':sum(r['capped'] for r in rr),'means':{k:statistics.fmean(r[k] for r in rr) for k in ['answer_tokens','repeated_token_4gram_fraction','source_reasoning_seconds','own_reasoning_seconds','mapping_seconds','splice_seconds','native_prefill_seconds','historical_receiver_prefill_tokens','answer_seconds','bridge_seconds','source_cache_reconstruction_seconds','source_inclusive_estimate_seconds']}}
    write(OUT/'hybrid_summary.json',hybrid)
    controls=[]
    for name in hybrid_roots:
        f=ROOT/name/'native_native_splice_controls.json'
        if f.exists():
            rr=json.loads(f.read_text());controls.append({'path':str(f.relative_to(ROOT)),'sha256':sha(f),'n':len(rr),'all_whole_history_splices_exact':all(r['native_native_tensor_exact'] and r['bridge_logits']['max_abs']==0 for r in rr),'partial_prefill_prefix_max_abs':max((r['actual_partial_prefill_prefix_max_abs'] for r in rr),default=0),'partial_prefill_bridge_max_abs':max((r['actual_partial_prefill_splice_bridge_logits']['max_abs'] for r in rr),default=0),'partial_prefill_bridge_max_kl':max((r['actual_partial_prefill_splice_bridge_logits']['kl'] for r in rr),default=0),'partial_prefill_bridge_top1_all_equal':all(r['actual_partial_prefill_splice_bridge_logits']['top1_equal'] for r in rr)})
    write(OUT/'native_splice_summary.json',controls)
    if raw:
        with (OUT/'hybrid_tasks.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
        lines+=['| Arm | Passes | Missing requested entrypoint | EOS / cap |','|---|---:|---:|---:|']
        for arm,r in hybrid['arms'].items():lines.append(f'|{arm}|{r["passes"]}/{r["n"]}|{r["missing_requested_entrypoints"]}|{r["eos"]}/{r["caps"]}|')
    else:lines+=['No scored prompt-preservation outcomes imported yet.']
    if hybrid['paired']:
        lines+=['','| Contrast | Difference, pp | 95% paired interval, pp | Gains / losses |','|---|---:|---:|---:|']
        for name,r in hybrid['paired']['contrasts'].items():lines.append(f'|{name}|{100*r["difference"]:+.1f}|[{100*r["ci95"][0]:+.1f}, {100*r["ci95"][1]:+.1f}]|{r["gains"]}/{r["losses"]}|')
        lines+=['',hybrid['paired']['method']]
        lines+=['','| Arm | Prompt/history prefill tokens | Native prefill, s | Mapping + splice, s | Answer, s | Source-inclusive estimate, s |','|---|---:|---:|---:|---:|---:|']
        for arm in ['M','H','D','P']:
            m=hybrid['arms'][arm]['means'];lines.append(f'|{arm}|{m["historical_receiver_prefill_tokens"]:.1f}|{m["native_prefill_seconds"]:.3f}|{m["mapping_seconds"]+m["splice_seconds"]:.3f}|{m["answer_seconds"]:.2f}|{m["source_inclusive_estimate_seconds"]:.2f}|')
        lines+=['','Times are per-task arithmetic means on the matched worker. Answer timing includes the bridge and periodic telemetry/checkpoint overhead. Source-inclusive estimates reuse each exact saved source-reasoning duration for M/H/D; P has no source generation. Reconstructing the source cache for this diagnostic is recorded separately in the CSV and excluded from the estimate of a handoff with that cache already resident. H saves native prefill work relative to D but has lower correctness and longer answers: this establishes no equal-quality speedup.','']
        c=hybrid['paired']['contrasts'];hm=c['H-M']['difference'];hp=c['H-P']['difference']
        if hm>0:lines+=['', 'Retaining the native prompt improved the observed pass rate relative to pure mapping. This supports prompt-state preservation as a useful intervention in this fixed development sample, without establishing a unique cause of the original failures.']
        else:lines+=['', 'The hybrid did not improve the observed aggregate pass rate over pure mapping. That weakens this particular repair, but does not rule out prompt-related corruption carried within the translated suffix.']
        if hp<=0:lines+=['The hybrid did not exceed prompt-only answering in observed aggregate passes, so this comparison provides no positive aggregate evidence that its transferred reasoning improves over that control. Uncertainty still prevents treating a nonsignificant difference as equivalence.']
        else:lines+=['The hybrid exceeded prompt-only answering in observed aggregate passes; the paired interval above determines how uncertain that difference is. A positive point estimate alone is not proof of transferable reasoning or generalization. The optional wrong-suffix control was not part of the primary comparison.']
    lines+=['','H retains the complete original native prompt cache and inserts the mapped suffix at its original absolute positions. H includes native prompt prefill and mapping/splicing costs. M replays the same original draw for matched timing and reuses its old score only after exact token equality. Fresh D must also reproduce its original answer tokens exactly; changed draws block baseline reuse. P uses the pinned non-thinking template with no separate thinking stage requested; independently thinking B is separate. Native/native cache-surgery controls and partial-prefill rounding diagnostics are preserved. New answer states continue to depend on historical cache. H exceeding M alone does not establish a benefit from transferred reasoning; H versus P is essential. H failure would not exclude prompt corruption because mapped suffix states also depend on source prompt representations.','']
    if controls:
        lines+=['Native/native whole-history cache splicing is exact in the recorded controls: '+str(all(r['all_whole_history_splices_exact'] for r in controls))+'. Separately prefilling only the native prompt changes execution shape; its native/native splice has maximum checked bridge-logit difference '+format(max(r['partial_prefill_bridge_max_abs'] for r in controls),'.8g')+' and maximum bridge KL '+format(max(r['partial_prefill_bridge_max_kl'] for r in controls),'.8g')+'. These bridge measurements do not establish equality of later sampled continuations. Full metrics and source hashes are in native_splice_summary.json.','']
    semantic_path=E/'semantic_case_audit.json'
    if semantic_path.exists():
        audit=json.loads(semantic_path.read_text());lines+=['### Selected semantic examples','',audit['scope'],'']
        for row in audit['rows']:
            lines.append('- '+row['task_id']+': '+row['summary']+' M: '+row['M']+' H: '+row['H']+' D: '+row['D']+' P: '+row['P'])
        lines+=['','Exact visible prompts, raw-output hashes and stopping/fence records are in evidence/coding_pilot_v1/post_progress01/semantic_case_audit.json. The prompt-only formula-fence failure illustrates dependence on output formatting and the fixed extractor; no post-hoc repair score replaces any primary score.','']
    lines+=['## Four seen-training-history optimization','']
    audit_path=E/'pretraining_answer_span_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text());lines+=['The source-answer span audit is fixed before new outputs: '+', '.join(r['task_id'] for r in audit['summary'])+'. The old fixed windows miss every function signature and return line in the three histories containing them; two longer answers have no code tokens in those windows. Actual selected96 exposure is counted separately from the saved update schedule. This documents limited direct supervision, not proof that explanation/prose is useless or that it caused task failures.','']
    memorization=[]
    for root in sorted((ROOT/'results/coding_pilot_v1').glob('post_progress01_memorization_*/memorization')):
        if not (root/'identity.json').exists():continue
        done=json.loads((root/'complete.json').read_text()) if (root/'complete.json').exists() else None
        curve=json.loads((root/'seen_training_curve.json').read_text()) if (root/'seen_training_curve.json').exists() else []
        scores=[]
        for p in sorted(root.glob('evaluations/step_*/*/*_seen.json')):
            r=json.loads(p.read_text());scores.append({'step':r['step'],'task_id':r['task_id'],'arm':r['condition'],'passed':r.get('score',{}).get('passed'),'category':r.get('score',{}).get('category'),'raw':str(p.relative_to(ROOT)),'sha256':sha(p)})
        entry={'root':str(root.relative_to(ROOT)),'completion':done,'curve':[{'step':r['step'],'mean_task_kl':r['mean_task_kl'],'gradient_predictions_so_far':r['gradient_predictions_so_far'],'unique_task_positions_so_far':r['unique_task_positions_so_far'],'per_case_kl':{v['task_id']:v['mean_kl'] for v in r['rows']}} for r in curve],'scores':scores};memorization.append(entry)
        lines.append(f'- `{root.parent.name}`: '+(f'{done["steps"]} completed updates, {done["scored_positions"]} scored positions, {done["unique_task_positions"]} unique task/position pairs, stop={done["stop_reason"]}.' if done else 'Incomplete; only preserved checkpoint records reported.'))
        native=[x for x in scores if x['arm']=='D_seen' and x['passed'] is not None]
        if native:lines.append(f'  Native replay on the same seen cases: {sum(x["passed"] for x in native)}/{len(native)} passes. This is the receiver-quality reference, not an unseen evaluation.')
        for r in entry['curve']:
            ss=[s for s in scores if s['step']==r['step'] and s['arm']=='M_seen' and s['passed'] is not None];lines.append(f'  Update {r["step"]}: dense seen KL {r["mean_task_kl"]:.6g}; free-running hidden-test passes '+(f'{sum(s["passed"] for s in ss)}/{len(ss)}' if ss else 'not yet scored')+'.')
        if entry['curve']:
            first,last=entry['curve'][0],entry['curve'][-1]
            def outcome(tid,step,arm):
                rr=[x for x in scores if x['task_id']==tid and x['step']==step and x['arm']==arm]
                return 'not scored' if not rr or rr[0]['passed'] is None else ('pass' if rr[0]['passed'] else rr[0]['category'])
            lines+=['','| Seen training case | Native D | Initial M | Final M | Initial dense KL | Final dense KL |','|---|---|---|---|---:|---:|']
            for tid,loss in last['per_case_kl'].items():lines.append(f'|{tid}|{outcome(tid,0,"D_seen")}|{outcome(tid,0,"M_seen")}|{outcome(tid,last["step"],"M_seen")}|{first["per_case_kl"][tid]:.6g}|{loss:.6g}|')
            step_path=root/'training_steps.json'
            if step_path.exists():
                updates=json.loads(step_path.read_text());norms=[x['gradient_norm_before_clipping'] for x in updates]
                if norms:lines+=['',f'All {len(updates)} committed updates report finite, nonzero cache gradients: {all(x["cache_gradients_finite_nonzero"] for x in updates)}. Pre-clipping mapper gradient norm ranges from {min(norms):.6g} to {max(norms):.6g}. Per-case losses, exact positions and norms remain in training_steps.json.']
            lines+=['','This deliberate four-history fit changes both supervision coverage and concentration/repetition on a tiny seen cohort. It is not an ablation isolating coverage alone, and its loss is not directly comparable to the previous validation average. Improvement supports trainability on these histories; it cannot establish generalization or sufficient mapper capacity for arbitrary histories. Hidden scores are assigned after optimization and do not control updates or stopping.','']
    if not memorization:lines+=['No four-case optimization run imported yet. The predeclared length-ranked histories and source-answer token spans will determine coverage; this is not a generalization test.']
    write(OUT/'memorization_summary.json',memorization)
    failures=[]
    for p in sorted((ROOT/'evidence/coding_pilot_v1/control/parallel').glob('post_progress01_*/*/error.json')):
        r=json.loads(p.read_text());failures.append({'path':str(p.relative_to(ROOT)),'error':r.get('error'),'sha256':sha(p)})
    for p in sorted((ROOT/'results/coding_pilot_v1').glob('post_progress01_*/*/failure.json')):
        r=json.loads(p.read_text());failures.append({'path':str(p.relative_to(ROOT)),'error':r.get('error'),'sha256':sha(p)})
    write(OUT/'failures.json',failures)
    interpretation_path=E/'final_interpretation.json'
    if interpretation_path.exists():
        interpretation=json.loads(interpretation_path.read_text());lines+=['','## Diagnostic conclusions and remaining uncertainty','']
        lines+=['- '+text for text in interpretation['summary']]
    memory=[]
    for p in sorted((ROOT/'results/coding_pilot_v1').glob('post_progress01_*/*/memory_telemetry.jsonl')):
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if rows:memory.append({'run':p.parent.parent.name,'records':len(rows),'max_recorded_peak_allocated_bytes':max(r['peak_allocated'] for r in rows),'max_recorded_peak_reserved_bytes':max(r['peak_reserved'] for r in rows),'minimum_sampled_free_bytes':min(r['free'] for r in rows),'max_allocator_oom_counter':max(r.get('allocator',{}).get('num_ooms',0) for r in rows),'warning_samples':sum(bool(r.get('warnings')) for r in rows),'sha256':sha(p)})
    write(OUT/'memory_summary.json',memory)
    if memory:
        lines+=['','## Memory telemetry','', '| Run | Maximum recorded allocated peak, GiB | Maximum recorded reserved peak, GiB | Allocator OOM counter |','|---|---:|---:|---:|']
        for r in memory:lines.append(f'|{r["run"]}|{r["max_recorded_peak_allocated_bytes"]/2**30:.2f}|{r["max_recorded_peak_reserved_bytes"]/2**30:.2f}|{r["max_allocator_oom_counter"]}|')
        lines+=['','These are maxima of recorded CUDA allocator peaks across scoped resets, not a claim of continuously sampled whole-device usage. The warning-based policy stayed in force. Raw telemetry records scopes, free/reserved memory, timing and warnings.','']
    lines+=['','## Preserved failures, scope and cost','',f'{len(failures)} controller/worker failure records are indexed in `results/coding_pilot_v1/post_progress01_report/failures.json`. Capacity errors before allocation are not scored as model failures. Earlier attempts are never erased or silently rerolled.','',f'Cumulative conservative ledger estimate at export: ${costs["upper_usd"]:.2f}, {costs["gpu_hours"]:.3f} GPU-hours. This is not an invoice. Live-resource state and timestamps are retained in the cost/resource receipts; cumulative ceilings remain $1,000 / 500 GPU-hours. Diagnostic sub-budget: $120 / 20 additional GPU-hours.','', 'Reserved 200 and second-seed 40 confirmation tasks remain untouched. No broad corpus collection, model-size change, subjective judging or architecture search. Hidden-test outcomes are scorer artifacts, not optimizer inputs. No inference or success is claimed for merely prepared code. Prompt corruption, path mismatch, sparse coverage and capacity remain hypotheses until the corresponding checks support an interpretation.','', 'Heavy source checkpoints/features are separately inventoried and backed up outside Git; the publication snapshot backup is `/Users/qeetbastudio/Gearshift-artifacts/progress-01/`. New heavy inventories and backup receipts are included with this pass. Discarded reproducible caches and unrecoverable historical allocator omissions remain disclosed.']
    (ROOT/'POST_PROGRESS_01_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return {'numerical_runs':len(numerical),'hybrid_completed':len(seen),'memorization_runs':len(memorization),'cost_usd':costs['upper_usd']}

if __name__=='__main__':print(json.dumps(report()))
