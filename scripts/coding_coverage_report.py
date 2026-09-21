#!/usr/bin/env python3
"""Regenerate tables/figures from compact evidence, without weights or tests."""
import argparse,ast,collections,csv,json,re,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gearshift.coding_control import write,sha,digest
from gearshift.coding_coverage import DECLARATION,EVIDENCE,PUBLICATION
from scripts.coding_recovery_report import repeated_fourgrams
ROOT=Path(__file__).resolve().parents[1]
OUT='results/coding_pilot_v1/coverage_generalization_report'
LABELS=['START_M','START_H','FIXED_M','FIXED_H','ROTATING_M','ROTATING_H','D','P']
CONTRASTS=[('ROTATING_M','FIXED_M'),('ROTATING_H','FIXED_H'),('FIXED_M','START_M'),('ROTATING_M','START_M'),
    ('FIXED_H','START_H'),('ROTATING_H','START_H')]+[(a,b) for a in ['START_H','FIXED_H','ROTATING_H'] for b in ['D','P']]


def read(p):return json.loads(Path(p).read_text())
def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def clustered(rows,task_ids,labels,seeds,contrasts,bootstrap_seed=20260918):
    lookup={}
    for r in rows:
        key=(r['task_id'],r['condition'],r['seed_index'])
        if key in lookup:raise ValueError('Duplicate task/condition/seed')
        lookup[key]=r
    expected={(t,a,s) for t in task_ids for a in labels for s in range(seeds)}
    if set(lookup)!=expected:raise ValueError('Incomplete or unexpected matched population')
    means=np.array([[sum(lookup[t,a,s]['passed'] for s in range(seeds))/seeds for a in labels] for t in task_ids])
    indices=np.random.default_rng(bootstrap_seed).integers(0,len(task_ids),size=(10000,len(task_ids)))
    boot=means[indices].mean(axis=1)
    result={'task_clusters':len(task_ids),'seeds_per_task':seeds,'task_ids':task_ids,'conditions':{},'contrasts':{},
        'method':'Equal task mean across predeclared seeds; 10000 joint paired task bootstrap resamples, percentile 95% linear interpolation. Exploratory and unadjusted; nonsignificance is not equivalence. No best-of-seeds score.',
        'bootstrap_seed':bootstrap_seed,'bootstrap_indices_sha256':digest(indices.tolist())}
    per_task=[]
    for i,tid in enumerate(task_ids):
        per_task.append({'task_id':tid,**{a:float(means[i,j]) for j,a in enumerate(labels)}})
    for j,a in enumerate(labels):
        rr=[r for r in rows if r['condition']==a]
        result['conditions'][a]={'pass_rate':float(means[:,j].mean()),'ci95':np.quantile(boot[:,j],[.025,.975]).tolist(),
            'passed_draws':sum(r['passed'] for r in rr),'draws':len(rr),'task_clusters':len(task_ids),
            'failure_categories':dict(collections.Counter(r['category'] for r in rr)),
            'entrypoint_failures':sum(r['missing_requested_entrypoint'] for r in rr),'syntax_failures':sum(not r['syntax_valid'] for r in rr),
            'EOS':sum(r['EOS'] for r in rr),'caps':sum(r['capped'] for r in rr),
            'means':{k:statistics.fmean(r[k] for r in rr) for k in ['answer_tokens','repetition_4gram_fraction','answer_seconds','native_prefill_seconds','mapping_seconds','splice_seconds','historical_receiver_prefill_tokens','source_inclusive_estimate_seconds']}}
    for a,b in contrasts:
        ia,ib=labels.index(a),labels.index(b);delta=means[:,ia]-means[:,ib]
        result['contrasts'][a+'-'+b]={'difference':float(delta.mean()),'ci95':np.quantile(boot[:,ia]-boot[:,ib],[.025,.975]).tolist(),
            'tasks_positive':int((delta>0).sum()),'tasks_negative':int((delta<0).sum()),'tasks_tied':int((delta==0).sum()),'per_task_difference':delta.tolist()}
    return result,per_task,indices.tolist()


def program_row(path,visible,seeds,checkpoint_hashes=None):
    r=read(path);tid=r['task_id'];seed=seeds[tid][r['seed_index']]
    if r['answer_seed']!=seed['answer_seed'] or r['stream']!=seed['stream'] or r['teacher_answer_prefix_supplied'] is not False:raise ValueError('Evaluation seed/input contract differs')
    if r['form']!='P' and sha(path.parents[2]/'source_history.json')!=r['source_history_sha256']:raise ValueError('Saved generation history changed')
    if checkpoint_hashes is not None and r['checkpoint_sha256']!=checkpoint_hashes[r['condition']]:raise ValueError('Scored mapper identity differs from common checkpoint')
    wanted=re.findall(r'^\s*(?:async\s+)?def (\w+)\(',visible[tid]['prompt'],re.M)
    expects_solution=bool(re.search(r'\bclass\s+Solution\b',visible[tid]['prompt']))
    try:
        tree=ast.parse(r['code']);syntax=True
        if expects_solution:
            solutions=[n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Solution']
            functions={f.name for n in solutions for f in n.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
            missing=not solutions or not set(wanted)<=functions
        else:
            functions={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            missing=bool(wanted) and not set(wanted)<=functions
    except SyntaxError:syntax=False;missing=bool(wanted) or expects_solution
    return {'task_id':tid,'condition':r['condition'],'seed_index':r['seed_index'],'answer_seed':r['answer_seed'],
        'passed':bool(r['score']['passed']),'category':r['score']['category'],'syntax_valid':syntax,'missing_requested_entrypoint':missing,
        'answer_tokens':len(r['answer_ids']),'EOS':r['answer_ended_eos'],'capped':r['answer_capped'],
        'repetition_4gram_fraction':repeated_fourgrams(r['answer_ids']),
        **{k:r.get(k,0.) for k in ['answer_seconds','native_prefill_seconds','mapping_seconds','splice_seconds','historical_receiver_prefill_tokens','source_inclusive_estimate_seconds']},
        'raw_path':str(path.relative_to(ROOT)),'raw_sha256':sha(path)}


def render(out,summary,curve,coverage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'gearshift-coverage-v1'})
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.svg'),metadata={'Date':None});plt.close(fig)
    if curve:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,key,title in zip(axes,['legacy_mean_task_kl','broad_mean_task_kl'],['Legacy windows','Fixed broad panel']):
            for arm,color in [('FIXED','#405bb7'),('ROTATING','#a33c32')]:
                rows=sorted((r for r in curve if r['arm']==arm),key=lambda r:r['step'])
                ax.plot([r['step'] for r in rows],[r[key] for r in rows],marker='o',label=arm,color=color)
            ax.set(title=title,xlabel='Additional paired updates',ylabel='Mean task KL (21 validation histories)');ax.legend()
        save(fig,'validation_kl')
    if summary:
        fig,ax=plt.subplots(figsize=(10,4));labels=list(summary['conditions']);r=[summary['conditions'][a] for a in labels]
        y=np.array([x['pass_rate']*100 for x in r]);lo=np.array([x['ci95'][0]*100 for x in r]);hi=np.array([x['ci95'][1]*100 for x in r])
        ax.bar(labels,y,color=['#9298a4','#9298a4','#405bb7','#405bb7','#a33c32','#a33c32','#397052','#9c8251'])
        ax.errorbar(labels,y,yerr=np.vstack([y-lo,hi-y]),fmt='none',color='#222',capsize=4)
        ax.set(ylim=(0,105),ylabel='Mean success across three seeds (%)',title='Validation behavior: 21 task clusters, paired bootstrap 95% intervals')
        ax.tick_params(axis='x',rotation=25);save(fig,'validation_behavior')
    if coverage:
        fig,ax=plt.subplots(figsize=(6,4));labels=list(coverage);ax.bar(labels,[coverage[a]['unique_task_positions'] for a in labels],color=['#405bb7','#a33c32'])
        ax.set(ylabel='Distinct supervised task/position pairs',title='Actual coverage at the primary common checkpoint');save(fig,'training_coverage')


def report(result_root):
    root=ROOT/result_root;out=ROOT/OUT;out.mkdir(parents=True,exist_ok=True);d=read(ROOT/DECLARATION)
    visible={x['task_id']:x for x in read(ROOT/'data/coding_pilot_v1/visible/coverage_generalization.json')};seeds=read(ROOT/d['answer_seeds_path'])
    git=read(ROOT/EVIDENCE/'current_git_verification.json');cost=read(ROOT/EVIDENCE/'review_cost_snapshot.json')
    if git['tag_commit']!=PUBLICATION:raise ValueError('Publication tag differs')
    manifest_path=root/'scored_answer_manifest.json'
    if manifest_path.exists():
        manifest=read(manifest_path)
        for item in manifest['files']:
            if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Committed scored answer changed')
        actual={str(p.relative_to(ROOT)) for scope in ['validation','seen_training'] for p in (root/scope).glob('*/*/seed_*/answer.json') if 'score' in read(p)}
        if actual!={r['path'] for r in manifest['files']}:raise ValueError('Scored manifest membership changed')
    elif (root/'complete.json').exists() and 'primary_common_checkpoint' in read(root/'complete.json'):
        raise ValueError('Completed experiment missing scored manifest')
    train=read(root/'training_complete.json') if (root/'training_complete.json').exists() else None
    curve=read(root/'validation_curve.json') if (root/'validation_curve.json').exists() else []
    checkpoint_hashes={'START_M':d['selected_checkpoint_sha256'],'START_H':d['selected_checkpoint_sha256'],
        'SEEN_FIT_M':d['seen_checkpoint_sha256'],'D':None,'P':None}
    if train:
        cp=read(root/'paired_checkpoints'/f"step_{train['primary_common_checkpoint']:04d}.json")
        for arm in ['FIXED','ROTATING']:
            for form in ['M','H']:checkpoint_hashes[arm+'_'+form]=cp['arms'][arm]['sha256']
        if train['primary_predictions_per_arm']!=train['primary_common_checkpoint']*32:raise ValueError('Primary supervision count differs')
    panels=read(ROOT/d['panels_path'])
    for row in curve:
        if len(row['rows'])!=21 or {r['task_id'] for r in row['rows']}!=set(d['validation_task_ids']):raise ValueError('KL panel population differs')
        for task in row['rows']:
            for panel in ['legacy','broad']:
                if task[panel]['positions']!=panels[task['task_id']][panel] or len(task[panel]['per_position_kl'])!=len(task[panel]['positions']):raise ValueError('KL positions differ from frozen panel')
    summary={};seen_summary={};raw_by_scope={};coverage={}
    for scope,ids,labels,n,contrasts in [('validation',d['validation_task_ids'],LABELS,3,CONTRASTS),
        ('seen_training',d['seen_training_task_ids'],['START_M','SEEN_FIT_M','D'],5,[('SEEN_FIT_M','START_M'),('SEEN_FIT_M','D')])]:
        paths=sorted((root/scope).glob('*/*/seed_*/answer.json'))
        rows=[program_row(p,visible,seeds[scope],checkpoint_hashes) for p in paths if 'score' in read(p)];raw_by_scope[scope]=rows;csv_write(out/(scope+'_draws.csv'),rows)
        try:result,task_rows,indices=clustered(rows,ids,labels,n,contrasts,d['analysis']['bootstrap_seed'])
        except ValueError as exc:
            write(out/(scope+'_incomplete.json'),{'error':str(exc),'scored_draws':len(rows),'expected_draws':len(ids)*len(labels)*n});continue
        # Earlier bounded partial exports may have used this report directory.
        # Their archives preserve the incident; a completed report must not
        # retain a stale marker saying that this population is incomplete.
        (out/(scope+'_incomplete.json')).unlink(missing_ok=True)
        write(out/(scope+'_summary.json'),result);write(out/(scope+'_task_bootstrap_indices.json'),indices);csv_write(out/(scope+'_task_means.csv'),task_rows)
        if scope=='validation':summary=result
        else:seen_summary=result
    for arm in ['FIXED','ROTATING']:
        p=root/(arm+'_primary_checkpoint_exposure.json')
        if p.exists():coverage[arm]=read(p)
    kl_rows=[]
    for r in curve:
        for task in r['rows']:
            for panel in ['legacy','broad']:
                kl_rows.append({'arm':r['arm'],'step':r['step'],'task_id':task['task_id'],'panel':panel,'mean_kl':task[panel]['mean_kl'],'predictions':len(task[panel]['positions'])})
    csv_write(out/'validation_kl_by_task.csv',kl_rows)
    failures=[]
    for pattern in ['results/coding_pilot_v1/coverage_*/*/*failure*.json','evidence/coding_pilot_v1/control/parallel/coverage_*/**/*failure*.json']:
        for p in ROOT.glob(pattern):failures.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'record':read(p)})
    write(out/'failures.json',failures)
    resources=read(ROOT/EVIDENCE/'live_resources.json')
    executed={'result_root':result_root,'training':train,'validation':summary or None,'seen_training':seen_summary or None,'coverage':{a:{k:r[k] for k in ['steps','scored_positions','unique_task_positions']} for a,r in coverage.items()},
        'cost':cost,'live_resources':resources,'complete':bool((root/'complete.json').exists() and summary and seen_summary)}
    write(out/'summary.json',executed)
    lines=['# Coverage generalization results','',
        'Separate controlled research. The first article and its original results remain frozen. Prepared code is not counted as executed research.',
        '',f'Publication tag `gearshift-progress-01` remains `{PUBLICATION}`. Research branch: `{git["branch"]}`; current recorded commit: `{git["head"]}`. No push or publication.',
        '', '## Population and comparison','',
        'Exact existing 104 training and 21 validation histories; completion-selected availability can underrepresent long or difficult histories. The 21 validation tasks are excluded from gradients, but previously informed KL selection. They are not untouched confirmation. Reserved 200 and second-seed confirmation inputs remain untouched.',
        '', 'Both arms start from the original selected update-96 mapper (`'+d['selected_checkpoint_sha256']+'`). Identical fresh AdamW states replace unavailable original optimizer states. Models, affine architecture, pinned BF16 runtime, source-written references, sampler, extraction and scoring are unchanged. No teacher-answer tokens prefix the free-running answers.',
        '', 'FIXED retains offsets 0–7, 32–39, 128–135, 512–519. Missing fixed positions are filled from the next history in the original seeded bucket. ROTATING preserves ordered task contributions and early-bucket positions, rotating later 8-token windows through the full saved answer including actual EOS. The single 30-token training answer cycles all windows because it has no later fixed bucket. All paired updates contain exactly 32 actual predictions with identical normalization; no padding or duplicate task/position within an update.',
        '', '## Actual execution','']
    if summary:
        primary=summary['contrasts']['ROTATING_M-FIXED_M']
        start_delta=summary['contrasts']['ROTATING_M-START_M']
        headline=(f"At the evaluated common checkpoint, pure mapped-answer success was {summary['conditions']['ROTATING_M']['pass_rate']:.1%} for ROTATING, "
            f"{summary['conditions']['FIXED_M']['pass_rate']:.1%} for FIXED, and {summary['conditions']['START_M']['pass_rate']:.1%} for the starting mapper. "
            f"The primary ROTATING-minus-FIXED difference was {100*primary['difference']:+.1f} percentage points "
            f"(paired 95% interval {100*primary['ci95'][0]:+.1f} to {100*primary['ci95'][1]:+.1f}). "
            f"ROTATING improved over its starting checkpoint by {100*start_delta['difference']:+.1f} points "
            f"({100*start_delta['ci95'][0]:+.1f} to {100*start_delta['ci95'][1]:+.1f}). "
            "These are exploratory results on previously selected validation tasks; the primary comparison does not establish that broader coverage caused a reliable improvement.")
        lines[2:2]=[headline,'']
    if train:
        lines += [f'Target: {train["target_additional_updates_per_arm"]} additional updates per arm. Completed paired updates: {train["completed_paired_updates"]}. Primary common checkpoint: {train["primary_common_checkpoint"]}; {train["primary_predictions_per_arm"]:,} scored positions per arm. Stop: `{train["stop_reason"]}`.',
            '', 'The common endpoint was frozen from timing/memory preflight before sampled program-quality scores. The primary checkpoint was not selected by loss or code correctness. Full prefixes and intervening continuation tokens were processed. Shared immutable historical cache preparation saves repeated setup; separate continuation/backward and optimizer times remain recorded. Matched supervision does not mean matched computation.']
        if train.get('completed_updates_are_last_backup_lower_bound'):
            lines += ['', 'Training was interrupted by loss of provider DNS connectivity on the controller. Its automatic guard stopped the GPU. The completed-update count is the last verified backup and a lower bound on work performed before termination. Only the last durable paired mapper state is evaluated; later logged updates are not recoverable as weights. Optimizer states were not saved, so training was not resumed. Fresh matched generation and scoring ran separately under the recovery identity; the original 1,024-update schedule was not completed.']
    else:lines += ['No completed paired training result has been imported. This is a partial export.']
    if coverage:
        lines += ['', '| Arm | Scored positions | Unique task/positions |','|---|---:|---:|']
        for arm,r in coverage.items():lines.append(f'|{arm}|{r["scored_positions"]:,}|{r["unique_task_positions"]:,}|')
        lines += ['', 'Exposure at the evaluated checkpoint, using descriptive token/line labels across all complete code fences. These labels did not select training positions or grade answers.',
            '', '| Answer region | FIXED scored / distinct | ROTATING scored / distinct |', '|---|---:|---:|']
        categories=sorted({name for r in coverage.values() for task in r['tasks'].values() for name in task['by_category']})
        for category in categories:
            cells=[]
            for arm in ['FIXED','ROTATING']:
                rows=[t['by_category'].get(category,{'scored':0,'unique':0}) for t in coverage[arm]['tasks'].values()]
                cells.append(f"{sum(r['scored'] for r in rows):,} / {sum(r['unique'] for r in rows):,}")
            lines.append('|'+category.replace('_',' ')+'|'+'|'.join(cells)+'|')
    step_path=root/'training_steps.json'
    if step_path.exists():
        updates=read(step_path)
        if any(r['step']!=i+1 or r['normalization_per_arm']!=32 or set(r['arms'])!={'FIXED','ROTATING'} for i,r in enumerate(updates)):
            raise ValueError('Paired update log has a gap or altered normalization')
        timing={'completed_paired_updates':len(updates),'shared_full_history_cache_seconds':sum(r['shared_cache_preparation_seconds'] for r in updates),
            'paired_update_wall_seconds':sum(r['wall_seconds'] for r in updates),'arms':{}}
        lines += ['',f'The timing table covers all {len(updates)} logged paired updates, including updates beyond the evaluated checkpoint. The raw log preserves timing for each update.']
        lines+=['','| Arm | Continuation forward/backward, seconds | Optimizer, seconds | Finite nonzero cache gradients |','|---|---:|---:|---|']
        for arm in ['FIXED','ROTATING']:
            rr=[r['arms'][arm] for r in updates]
            timing['arms'][arm]={'continuation_forward_backward_seconds':sum(r['continuation_forward_backward_seconds'] for r in rr),
                'optimizer_seconds':sum(r['optimizer_seconds'] for r in rr),'gradient_predictions':sum(r['gradient_predictions'] for r in rr),
                'all_cache_gradients_finite_nonzero':all(r['cache_gradients_finite_nonzero'] for r in rr),
                'gradient_norm_min':min((r['gradient_norm_before_clipping'] for r in rr),default=None),'gradient_norm_max':max((r['gradient_norm_before_clipping'] for r in rr),default=None)}
            t=timing['arms'][arm];lines.append(f'|{arm}|{t["continuation_forward_backward_seconds"]:.1f}|{t["optimizer_seconds"]:.1f}|{t["all_cache_gradients_finite_nonzero"]}|')
        lines+=['',f'Shared full-history cache preparation: {timing["shared_full_history_cache_seconds"]:.1f} seconds. Total paired-update wall time: {timing["paired_update_wall_seconds"]:.1f} seconds. Arm times synchronize the device and include instrumentation; they are stage wall times, not hardware utilization integrals. Actual leased GPU-hours below include setup, controls, validation, generation and export.']
        write(out/'training_timing_and_gradients.json',timing)
    if summary:
        lines += ['', 'Condition key: **M** translates the full historical cache; **H** combines native original-prompt cache with the translated reasoning suffix; **D** natively reads the complete source history; **P** answers from the original prompt using the non-thinking template. START is the original selected mapper; FIXED and ROTATING are the two common-budget training arms.']
        lines += ['', '## Validation behavior','', 'Each rate averages all three predeclared draws within each task, then the 21 tasks. The 63 draws per condition are not 63 independent task clusters. No best-of-three selection. Intervals use 10,000 identical paired task resamples across all contrasts; exploratory, unadjusted 95% percentile intervals.', '', '| Condition | Passed draws / 63 | Mean task success | Task-cluster 95% interval | Missing requested entrypoint | EOS / cap |','|---|---:|---:|---:|---:|---:|']
        for arm,r in summary['conditions'].items():lines.append(f'|{arm}|{r["passed_draws"]}/63|{r["pass_rate"]:.1%}|[{r["ci95"][0]:.1%}, {r["ci95"][1]:.1%}]|{r["entrypoint_failures"]}|{r["EOS"]}/{r["caps"]}|')
        lines += ['', '| Contrast | Difference, percentage points | Paired 95% interval, pp |','|---|---:|---:|']
        for name,r in summary['contrasts'].items():lines.append(f'|{name}|{100*r["difference"]:+.1f}|[{100*r["ci95"][0]:+.1f}, {100*r["ci95"][1]:+.1f}]|')
        r=summary['contrasts']['ROTATING_M-FIXED_M']
        if r['ci95'][0]>0:conclusion='The primary comparison provides exploratory evidence of improved behavior outside these mapper-training histories. The small, previously selected validation population limits the strength and scope of that conclusion.'
        elif r['ci95'][1]<0:conclusion='The primary comparison favors fixed coverage on this validation population; broader supervision did not recover better held-out behavior in this run.'
        else:conclusion='The primary comparison does not establish a reliable held-out benefit from rotating coverage. Its interval includes zero; this is not an equivalence finding.'
        lines += ['',conclusion,'','H exceeding M alone does not establish a transferred-reasoning benefit. H versus prompt-only P and native full replay D are reported separately. These observations may inform a later decision but did not alter this round’s recipe.']
        lines += ['', '| Condition | Mean answer tokens | Answer seconds | Native prefill seconds | Native prefill tokens | Repeated 4-gram fraction |','|---|---:|---:|---:|---:|---:|']
        for arm,r in summary['conditions'].items():
            m=r['means'];lines.append(f'|{arm}|{m["answer_tokens"]:.1f}|{m["answer_seconds"]:.2f}|{m["native_prefill_seconds"]:.3f}|{m["historical_receiver_prefill_tokens"]:.1f}|{m["repetition_4gram_fraction"]:.3f}|')
        lines += ['', 'H includes native original-prompt work. Timings include bridge and instrumentation overhead. Shared reconstruction and cloning are recorded separately; source-inclusive estimates reuse the saved source duration and are not a demonstrated equal-quality speedup. Parsing, runtime, assertion, timeout and other original score categories remain in the JSON/CSV records. Entrypoint compliance checks requested function names and Solution membership; it is not a full semantic interface proof. Repetition is descriptive, including legitimate repeated code.']
    else:lines += ['', 'Validation generation/scoring is incomplete. No final matched comparison is claimed.']
    if curve:
        lines += ['', '## Validation loss','', 'Separate fixed legacy and broad panels on all 21 validation histories. Their numerical magnitudes are not directly interchangeable. Lower teacher-forced KL need not yield a correct sampled program.','', '| Arm | Additional updates | Legacy KL | Broad KL |','|---|---:|---:|---:|']
        for r in sorted(curve,key=lambda x:(x['step'],x['arm'])):lines.append(f'|{r["arm"]}|{r["step"]}|{r["legacy_mean_task_kl"]:.7g}|{r["broad_mean_task_kl"]:.7g}|')
    if seen_summary:
        lines += ['', '## Four seen training cases','', 'Five predeclared draws on each of the same four training cases. No new optimization for this check. These are four task clusters, not 20 independent tasks and not generalization evidence.','', '| Condition | Passed draws / 20 | Mean task success |','|---|---:|---:|']
        for arm,r in seen_summary['conditions'].items():lines.append(f'|{arm}|{r["passed_draws"]}/20|{r["pass_rate"]:.1%}|')
        seen_tasks=read_csv(out/'seen_training_task_means.csv')
        lines += ['', '| Seen task | Starting M | Four-case-fit M | Native D |','|---|---:|---:|---:|']
        for r in seen_tasks:lines.append(f'|{r["task_id"]}|{float(r["START_M"]):.0%}|{float(r["SEEN_FIT_M"]):.0%}|{float(r["D"]):.0%}|')
    memory=[]
    for p in sorted((ROOT/'results/coding_pilot_v1').glob('coverage_*/*/memory_telemetry.jsonl')):
        rr=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        rr=[r for r in rr if 'peak_allocated' in r]
        if rr:memory.append({'run':p.parent.parent.name,'samples':len(rr),'peak_allocated_GiB':max(r['peak_allocated'] for r in rr)/1024**3,
            'peak_reserved_GiB':max(r['peak_reserved'] for r in rr)/1024**3,'allocator_oom_counter':max((r['allocator'].get('num_ooms') or 0) for r in rr)})
    write(out/'memory_summary.json',memory)
    if memory:
        lines+=['','| Run | Maximum recorded allocated peak, GiB | Reserved peak, GiB | Allocator OOM counter |','|---|---:|---:|---:|']
        for m in memory:lines.append(f'|{m["run"]}|{m["peak_allocated_GiB"]:.2f}|{m["peak_reserved_GiB"]:.2f}|{m["allocator_oom_counter"]}|')
        lines+=['','Memory values are maxima of recorded CUDA allocator peaks across the logged reset scopes; they are not continuous whole-device measurements.']
    lines += ['', '## Controls, limitations and resources','',
        'The retained numerical checks distinguish matched-call agreement from batch-versus-token execution sensitivity. Preflight adds late-position matched-path checks. Every evaluation history has exact native/native whole-cache splice checks; actual partial-prompt prefill rounding is recorded separately. All conditions are generated freshly under this round’s identity, so old 40-task scores are not reused. Historical caches are cloned and checked for isolation; absolute suffix positions are preserved.',
        '', 'Memory remains warning-based; no historical-peak/free-memory veto. Actual allocation errors, nonfinite or missing gradients, failed numerical controls, deadlines and budget ceilings still stop work. Private hidden tests are loaded only after optimization and generation and executed only in the existing sandbox. Extraction is unchanged, including the first eligible code block; function names and failed programs are never repaired for primary scoring.',
        '',f'Preserved failure records: {len(failures)}. Cumulative conservative estimate: ${cost["upper_usd"]:.2f}, {cost["gpu_hours"]:.3f} GPU-hours. This round: ${cost["upper_usd"]-d["budget_start"]["upper_usd"]:.2f}, {cost["gpu_hours"]-d["budget_start"]["gpu_hours"]:.3f} GPU-hours. The ledger is an upper estimate, not an invoice; retained storage continues to accrue. Round ceiling: +$200/32 GPU-hours; cumulative ceiling: $1000/500 GPU-hours.',
        '', 'Live resources and timestamp are recorded in `evidence/coding_pilot_v1/coverage_generalization/live_resources.json`. Heavy mapper backups, exact hashes and retrieval paths are in `excluded_heavy.json`; they are outside Git on the Studio disk, not an off-site disaster-recovery guarantee. Large model weights, tensor caches, private tests and environments are excluded from the review ZIP.',
        '', 'Reproduce compact tables and plots with `python scripts/coding_coverage_report.py --result-root '+result_root+'`. No model weights or private tests are needed for this records-only command. See `REPRODUCE_COVERAGE_GENERALIZATION.md` for execution and provenance details.','']
    (ROOT/'COVERAGE_GENERALIZATION_RESULTS.md').write_text('\n'.join(lines))
    render(out,summary,curve,coverage)
    return executed


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result-root',required=True);a=p.parse_args();report(a.result_root)
