#!/usr/bin/env python3
"""Immutable coding-pilot analysis and compact, source-inclusive local review ZIP.

This script reads completed transactions only. It performs no inference, scoring,
resource allocation, checkpoint selection, publishing, or historical-file edits.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
import zipfile

ARMS=('A','B','C','D')
CONDITIONS={'A':'large_only','B':'small_only','C':'gearshift','D':'text_handoff'}
BOOTSTRAPS=10000
BOOTSTRAP_SEED=20260916
ORIGINAL_TRAJECTORIES=('data/qwen3_1.7b_to_0.6b/reasoning_trajectories.jsonl',
                       'data/qwen3_4b_to_0.6b/reasoning_trajectories.jsonl')
EXCLUDED_PARTS={'.git','.venv','venv','env','__pycache__','.pytest_cache','.mypy_cache','.cache',
    '.env','node_modules','private','raw','cache_lru','paired_cache','paired_caches','model_cache',
    'huggingface','hub','pod_backup','parallel_backups','backups','backups_extracted','review_delivery'}
EXTENSIONS={'.py','.json','.jsonl','.csv','.tsv','.md','.txt','.png','.svg','.pdf','.yml','.yaml',
    '.toml','.ini','.patch','.diff','.sh','.lock'}
MAX_FILE_BYTES=32*1024**2
MAX_BUNDLE_BYTES=1024**3


def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('Refusing to overwrite review evidence: '+str(path))
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def inside(repo,value):
    path=Path(value);path=path if path.is_absolute() else repo/path
    path.resolve().relative_to(repo.resolve())
    return path

def percentile(values,p):
    values=sorted(values);x=(len(values)-1)*p;lo=int(x);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)

def paired_statistics(rows,draws=BOOTSTRAPS,seed=BOOTSTRAP_SEED):
    """Resample original tasks jointly across all arms, never seed/arm rows."""
    if not rows:raise ValueError('No complete task pairs')
    if len({r['task_id'] for r in rows})!=len(rows):raise ValueError('Duplicate original task')
    if type(draws) is not int or draws<1:raise ValueError('Invalid bootstrap count')
    vectors=[tuple(int(r['pass'][a]) for a in ARMS) for r in rows];n=len(vectors)
    means={a:sum(v[i] for v in vectors)/n for i,a in enumerate(ARMS)}
    rng=random.Random(seed);samples={k:[] for k in ['C-B','C-D','C-A','A-B']};ratios=[];zero_denominators=0
    for _ in range(draws):
        sums=[0,0,0,0]
        for _ in range(n):
            v=vectors[rng.randrange(n)]
            for i in range(4):sums[i]+=v[i]
        a,b,c,d=sums
        for key,value in [('C-B',c-b),('C-D',c-d),('C-A',c-a),('A-B',a-b)]:samples[key].append(value/n)
        if a==b:zero_denominators+=1
        else:ratios.append((c-b)/(a-b))
    contrasts={}
    for key,s in samples.items():
        left,right=key.split('-');i,j=ARMS.index(left),ARMS.index(right)
        contrasts[key]={'difference':means[left]-means[right],'ci95':[percentile(s,.025),percentile(s,.975)],
            'rescues':sum(v[i]==1 and v[j]==0 for v in vectors),'regressions':sum(v[i]==0 and v[j]==1 for v in vectors),
            'degenerate_interval':min(s)==max(s),'task_count':n}
    gap=means['A']-means['B'];ci=contrasts['A-B']['ci95']
    suppress=abs(gap)<.05-1e-12 or ci[0]<=0<=ci[1]
    gap_closed={'reported':not suppress,'denominator':gap,'denominator_ci95':ci,
        'suppression_reason':'The large–small gap is below five percentage points or its paired interval includes zero.' if suppress else None}
    if not suppress:
        gap_closed.update(value=(means['C']-means['B'])/gap,bootstrap_zero_denominators=zero_denominators,
            ci95=[percentile(ratios,.025),percentile(ratios,.975)] if not zero_denominators else None,
            interval_limitation='Ratio interval unavailable because some bootstrap denominators are zero.' if zero_denominators else None)
    return {'task_count':n,'passes':{a:sum(v[i] for v in vectors) for i,a in enumerate(ARMS)},'pass_rates':means,
        'contrasts':contrasts,'gap_closed':gap_closed,'bootstrap':{'resamples':draws,'seed':seed,'unit':'original task',
        'method':'Joint paired task percentile bootstrap','confidence_level':.95},
        'caution':'A degenerate or nonsignificant interval does not establish population equivalence.'}


def useful_output(row):
    return (row.get('score',{}).get('passed') is True and row.get('answer_ended_eos') is True
        and row.get('answer_capped') is False)

def numeric(value):return type(value) in (int,float) and math.isfinite(value) and value>=0

def latency_summary(rows):
    result={'policy':'Only hidden-test-passing, EOS-ended, uncapped answers qualify; paired comparisons use the same successful tasks.',
        'scope':'Recorded generation stage sums include source reasoning. They are not whole service wall time or dollar savings.',
        'arms':{},'paired':{}}
    for arm in ARMS:
        good=[r['outputs'][arm] for r in rows if useful_output(r['outputs'][arm])]
        fields={}
        for name in ['source_inclusive_seconds','source_reasoning_seconds','own_reasoning_seconds','mapping_seconds',
                     'native_prefill_seconds','first_answer_token_seconds','answer_seconds','bridge_seconds']:
            values=[r[name] for r in good if numeric(r.get(name))]
            fields[name]={'n':len(values),'mean':statistics.fmean(values),'median':statistics.median(values)} if values else {'n':0,'mean':None,'median':None}
        result['arms'][arm]={'successful_complete_outputs':len(good),'measures':fields}
    for comparator in ['B','D','A']:
        pairs=[(r['outputs']['C'],r['outputs'][comparator]) for r in rows
            if useful_output(r['outputs']['C']) and useful_output(r['outputs'][comparator])
            and numeric(r['outputs']['C'].get('source_inclusive_seconds')) and numeric(r['outputs'][comparator].get('source_inclusive_seconds'))]
        c=[x['source_inclusive_seconds'] for x,y in pairs];other=[y['source_inclusive_seconds'] for x,y in pairs]
        result['paired']['C-'+comparator]={'task_count':len(pairs),'mean_seconds_difference':statistics.fmean([x-y for x,y in zip(c,other)]) if pairs else None,
            'ratio_of_mean_source_inclusive_seconds':sum(c)/sum(other) if pairs and sum(other)>0 else None}
    return result


def strata(rows):
    groups={}
    selectors={'all':lambda r:True,'source_capped':lambda r:r['source_capped'],
        'source_uncapped':lambda r:not r['source_capped'],'small_capped':lambda r:r['small_capped'],
        'neither_reasoner_capped':lambda r:not r['source_capped'] and not r['small_capped']}
    for label,predicate in selectors.items():
        selected=[r for r in rows if predicate(r)]
        groups[label]={'tasks':len(selected),'passes':{a:sum(r['pass'][a] for r in selected) for a in ARMS}}
    failures={}
    for arm in ARMS:
        outputs=[r['outputs'][arm] for r in rows]
        failures[arm]={'score_categories':dict(Counter(o['score'].get('category','unknown') for o in outputs)),
            'answer_capped':sum(o.get('answer_capped') is True for o in outputs),
            'answer_ended_eos':sum(o.get('answer_ended_eos') is True for o in outputs),
            'termination_unknown':sum(type(o.get('answer_capped')) is not bool or type(o.get('answer_ended_eos')) is not bool for o in outputs),
            'repetition':'Not independently measured; no category inferred from pass/fail.'}
    return {'groups':groups,'failure_and_termination':failures}


def load_evaluation(repo,roots,expected,stage,selection_sha,checkpoint_sha=None):
    rows={};issues=[];sources={};duplicates=set();incomplete_roots=[]
    for name in roots:
        root=inside(repo,name)
        try:
            identity=read(root/'identity.json');native=read(root/'native_gate.json')
            ident=digest(identity)
            if (root/'complete.json').exists():done=read(root/'complete.json')
            else:
                candidates=read(root/'candidate_inputs.json')
                done={'stage':stage,'identity_sha256':ident,'selection_sha256':candidates['selection_sha256'],
                    'tasks':{read(p)['task_id']:sha(p) for p in (root/'tasks').glob('*/complete.json')}}
                done['task_count']=len(done['tasks']);incomplete_roots.append(str(name))
            if done.get('stage')!=stage or done.get('identity_sha256')!=ident:raise ValueError('Stage completion identity differs')
            if native.get('passed') is not True or native.get('identity_sha256')!=ident:raise ValueError('Native controls do not bind this worker')
            if not selection_sha or done.get('selection_sha256')!=selection_sha:raise ValueError('Frozen selection differs or is missing')
            if done.get('task_count')!=len(done['tasks']):raise ValueError('Worker task count differs')
            sources[str(root.relative_to(repo))]=sha(root/'complete.json') if (root/'complete.json').exists() else None
        except (OSError,ValueError,KeyError,TypeError) as exc:
            issues.append({'root':str(name),'error':str(exc)});continue
        for tid,want in done['tasks'].items():
            try:
                if tid not in expected:raise ValueError('Task is outside frozen membership')
                folder=root/'tasks'/tid.replace('/','__');receipt=read(folder/'complete.json')
                if receipt['task_id']!=tid or receipt['identity_sha256']!=ident or sha(folder/'complete.json')!=want:raise ValueError('Task transaction differs')
                required={'source_history.json','C.json'} | ({'A.json','B.json','D.json','small_history.json'} if stage=='confirmation' else set())
                if not required<=set(receipt['files']):raise ValueError('Required output is missing from the committed hash inventory')
                for filename,value in receipt['files'].items():
                    if Path(filename).name!=filename or sha(folder/filename)!=value:raise ValueError('Task output hash differs')
                history=read(folder/'source_history.json')
                if history['task_id']!=tid or history['reasoning_capped']!=receipt['source_capped']:raise ValueError('Source history/cap identity differs')
                arms=ARMS if stage=='confirmation' else ('C',)
                output={a:read(folder/(a+'.json')) for a in arms}
                passes={}
                for arm,o in output.items():
                    if o['task_id']!=tid or o['condition']!=CONDITIONS[arm] or type(o['score']['passed']) is not bool:raise ValueError('Arm identity or score is malformed')
                    passes[arm]=o['score']['passed']
                    if passes[arm] and not (o['score'].get('trusted_checker_complete') is True
                        and type(o['score'].get('total_tests')) is int and o['score']['total_tests']>0
                        and o['score'].get('executed_tests')==o['score']['total_tests']):
                        raise ValueError('A passing answer lacks complete trusted-test evidence')
                    if o['score'].get('category') in ('missing_tests','sandbox_unavailable'):raise ValueError('Trusted tests were unavailable')
                if stage=='confirmation':
                    if receipt['pass']!=passes:raise ValueError('Saved pass counts differ from individual arm scores')
                    small=read(folder/'small_history.json')
                    if small['task_id']!=tid or small['reasoning_capped']!=receipt['small_capped']:raise ValueError('Small history/cap identity differs')
                elif len(receipt['candidates'])!=1 or next(iter(receipt['candidates'].values()))['passed']!=passes['C']:
                    raise ValueError('Second-seed candidate score differs')
                if checkpoint_sha and output['C'].get('checkpoint_sha256')!=checkpoint_sha:raise ValueError('Headline checkpoint differs from frozen selection')
                if output['C'].get('historical_receiver_prefill_tokens')!=0:raise ValueError('Mapped-cache arm performed historical receiver prefill')
                if tid in rows or tid in duplicates:
                    rows.pop(tid,None);duplicates.add(tid);raise ValueError('Duplicate original-task observation')
                rows[tid]={'task_id':tid,'pass':passes,'outputs':output,'source_history':history,
                    'source_capped':receipt['source_capped'],'small_capped':receipt.get('small_capped',False),
                    'source_root':str(root.relative_to(repo)),'transaction_sha256':want}
            except (OSError,ValueError,KeyError,TypeError) as exc:issues.append({'root':str(name),'task_id':tid,'error':str(exc)})
    complete=set(rows)==set(expected) and not issues and not incomplete_roots
    return [rows[k] for k in sorted(rows)],{'complete':complete,'observed':len(rows),'expected':len(expected),
        'missing_task_ids':sorted(set(expected)-set(rows)),'issues':issues,'incomplete_worker_roots':incomplete_roots,'source_completions':sources}


def training_summary(repo,roots):
    rows=[];issues=[]
    for name in roots:
        root=inside(repo,name);root=root/'run' if (root/'run').is_dir() else root
        try:
            identity=read(root/'identity.json');done=read(root/'complete.json')
            if done['identity_sha256']!=digest(identity):raise ValueError('Training completion identity differs')
            curve=read(root/'validation_curve.json') if (root/'validation_curve.json').exists() else []
            rows.append({'root':str(root.relative_to(repo)),'objective':done['objective'],'seed':done['seed'],
                'steps':done['steps'],'stop_reason':done['stop_reason'],'converged':done['converged'],
                'capped_unconverged':done['capped_unconverged'],'wall_seconds':done['wall_seconds'],
                'training_predictions':done['training_predictions'],'nominees':done['nominees'],'validation_curve':curve,
                'identity_sha256':digest(identity),'completion_sha256':sha(root/'complete.json')})
        except (OSError,ValueError,KeyError,TypeError) as exc:issues.append({'root':str(name),'error':str(exc)})
    return {'runs':rows,'issues':issues,'caveat':'Training rebuilds the entire saved prefix in 512-token chunks. This preserves token identities and absolute positions, but BF16 arithmetic can differ from the original one-token source generation. No exact-logit equivalence between those segmentations is claimed.'}


def request_from_state(repo,path):
    state=read(path);pid=state['pipeline_id']
    if not isinstance(pid,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',pid):raise ValueError('Invalid pipeline identifier')
    request={'pipeline_id':pid,'pipeline_state':str(path.relative_to(repo)),
        'confirmation_roots':[],'second_seed_roots':[],'training_roots':[],
        'baseline_root':state.get('baseline_root'),'scientific_stop':state.get('scientific_stop'),
        'measurements_complete':state.get('measurements_complete',False),'completed_runs':state.get('completed_runs',{})}
    for stage,run in request['completed_runs'].items():
        if not isinstance(run,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',run):raise ValueError('Invalid completed run identifier')
        plan=read(repo/'evidence/coding_pilot_v1/control/parallel'/run/'plan.json')
        for worker in plan['workers']:
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',worker['worker_id']):raise ValueError('Invalid worker identifier')
            root=f"results/coding_pilot_v1/{run}/{worker['worker_id']}"
            if stage=='confirmation_with_second_seed':
                request['confirmation_roots' if worker.get('role')=='confirmation' else 'training_roots'].append(root)
            elif stage=='seed_sensitivity':request['second_seed_roots'].append(root)
            elif stage=='training':request['training_roots'].append(root)
            elif stage=='cap_recovery' and not request['baseline_root']:request['baseline_root']=root
    selected=repo/'evidence/coding_pilot_v1/pipeline'/pid/'selected_recipe.json'
    request['selection_path']=str(selected.relative_to(repo)) if selected.exists() else None
    if state.get('completion_proof'):
        proof=inside(repo,state['completion_proof']);request['completion_proof']={'path':str(proof.relative_to(repo)),'sha256':sha(proof),'record':read(proof)}
    return request


def execution_wall_times(repo,request):
    stages=[]
    for stage,run in request.get('completed_runs',{}).items():
        directory=repo/'evidence/coding_pilot_v1/control/parallel'/run
        if not (directory/'dispatch.json').exists() or not (directory/'complete.json').exists():continue
        start=read(directory/'dispatch.json').get('epoch');end=read(directory/'complete.json').get('epoch')
        stages.append({'stage':stage,'run_id':run,'started_epoch':start,'finished_epoch':end,
            'wall_seconds':end-start if numeric(start) and numeric(end) and end>=start else None})
    return {'stages':stages,'scope':'Studio dispatch through worker completion and cleanup, including preparation/backup overhead. Concurrent workers count once in stage wall time; GPU-hours remain separately cumulative.'}


def analyze(repo,request):
    repo=Path(repo).resolve();cfg=read(repo/'configs/coding_pilot_v1/pilot.json')
    members=read(repo/'data/coding_pilot_v1/visible/confirmation.json');expected={r['task_id'] for r in members}
    if len(members)!=200 or len(expected)!=200:raise ValueError('Expected exactly 200 unique frozen confirmation tasks')
    reserved={r['platform']+'/'+r['question_id'] for r in cfg['mapper']['second_seed_confirmation_ids']}
    if len(reserved)!=40 or not reserved<=expected:raise ValueError('Expected exactly 40 reserved second-seed tasks')
    selection=None;selection_sha=None;selection_issue=None
    if request.get('selection_path'):
        try:
            path=inside(repo,request['selection_path']);selection=read(path)
            if selection.get('confirmation_used_for_selection') is not False or selection.get('seed')!=20260915:
                raise ValueError('Headline selection is not frozen from first-seed development')
            selection_sha=sha(path)
        except (OSError,ValueError,KeyError,TypeError) as exc:selection_issue=str(exc)
    rows,coverage=load_evaluation(repo,request.get('confirmation_roots',[]),expected,'confirmation',selection_sha,selection.get('checkpoint_sha256') if selection else None)
    seed_rows,seed_coverage=load_evaluation(repo,request.get('second_seed_roots',[]),reserved,'second_seed',selection_sha)
    primary=paired_statistics(rows) if coverage['complete'] else None
    seed_summary={'coverage':seed_coverage,'inference':None}
    if coverage['complete'] and seed_coverage['complete']:
        headline={r['task_id']:r for r in rows};paired=[]
        for row in seed_rows:
            original=headline[row['task_id']]
            for key in ['prompt_ids','reasoning_ids','prefix_ids','bridge_ids']:
                if row['source_history'].get(key)!=original['source_history'].get(key):
                    seed_coverage['complete']=False;seed_coverage['issues'].append({'task_id':row['task_id'],'error':'Second seed source history differs'})
            # A/B/D remain the original shared task outcomes; only C is replicated.
            paired.append({'task_id':row['task_id'],'pass':{'A':original['pass']['C'],'B':original['pass']['C'],
                'C':row['pass']['C'],'D':original['pass']['C']}})
        if seed_coverage['complete']:
            stat=paired_statistics(paired)
            seed_summary['inference']={'task_count':40,'headline_C_passes':sum(r['pass']['A'] for r in paired),
                'second_seed_C_passes':sum(r['pass']['C'] for r in paired),'second_minus_headline':stat['contrasts']['C-A'],
                'bootstrap':stat['bootstrap'],'policy':'Reserved 40 original tasks; a sensitivity check, not 40 additional independent headline tasks or a basis for selecting a better seed.'}
    training=training_summary(repo,request.get('training_roots',[]))
    baseline=None
    if request.get('baseline_root'):
        root=inside(repo,request['baseline_root'])
        if (root/'baseline_gate.json').exists():baseline={'root':str(root.relative_to(repo)),
            'baseline_gate_sha256':sha(root/'baseline_gate.json'),'gate':read(root/'baseline_gate.json')}
    costs=None;costpath=repo/'evidence/coding_pilot_v1/control/watchdog_status.json'
    if costpath.exists():
        raw=read(costpath);costs={k:raw.get(k) for k in ['epoch','state','upper_usd','gpu_hours','active_resources',
            'hard_total_cap_usd','hard_total_gpu_hours','cleanup_dispatch_ceiling_usd']}
    issues=coverage['issues']+seed_coverage['issues']+training['issues']+([{'selection':selection_issue}] if selection_issue else [])
    completion_proof_valid=False
    if request.get('completion_proof'):
        try:
            proof=request['completion_proof'];path=inside(repo,proof['path']);actual=read(path)
            if sha(path)!=proof['sha256'] or actual!=proof['record'] or actual.get('passed') is not True:
                raise ValueError('Final bounded execution proof changed')
            if actual.get('headline_task_count')!=200 or actual.get('second_seed_task_count')!=40:
                raise ValueError('Final proof cohort counts differ')
            for name,want in actual.get('inputs',{}).items():
                if sha(inside(repo,name))!=want:raise ValueError('Final proof input changed')
            completion_proof_valid=True
        except (OSError,ValueError,KeyError,TypeError) as exc:issues.append({'completion_proof':str(exc)})
    recipes={(r['objective'],r['seed']) for r in training['runs']}
    expected_recipes={('ordinary_continuation',20260915),('natural_handoff_boundary',20260915)}
    if selection:expected_recipes.add((selection['objective'],20260916))
    training_complete=expected_recipes<=recipes and len(expected_recipes)==3
    complete=bool(primary and seed_coverage['complete'] and training_complete and completion_proof_valid
        and request.get('measurements_complete') and not issues)
    latency=latency_summary(rows) if rows else None
    verdict='publish as partial/negative result'
    if issues:verdict='fix a blocking correctness issue before publication'
    elif complete and primary['contrasts']['C-B']['ci95'][0]>0:
        pair=latency['paired']['C-D']
        if pair['task_count'] and pair['ratio_of_mean_source_inclusive_seconds']<1:
            verdict='publish as promising optimization with caveats'
    summary={'schema':1,'namespace':'coding_pilot_v1','created_epoch':time.time(),'request':request,
        'study_complete':complete,'completion_proof_verified':completion_proof_valid,'training_coverage_complete':training_complete,'headline_confirmation_complete':coverage['complete'],'coverage':coverage,
        'observed_descriptive_passes':{a:sum(r['pass'][a] for r in rows) for a in ARMS},
        'primary_inference':primary,'second_seed':seed_summary,'latency':latency,'strata':strata(rows) if rows else None,
        'training':training,'execution_wall_times':execution_wall_times(repo,request),'baseline':baseline,'selection':selection,'selection_sha256':selection_sha,'costs':costs,
        'publication_recommendation':verdict,'integrity_issues':issues,
        'limits':['Development observations and incomplete confirmation subsets do not establish transfer benefit.',
            'Public benchmark/model pretraining exposure is unknown; the fixed release is not a contamination guarantee.',
            'Text handoff is a diagnostic reference, not a theoretical upper bound.',
            'Quality uses one sampled output per task/arm; there is no repair, best-of-many or seed selection.',
            'Recorded stage sums do not include every orchestration, loading, copying or validation cost.',
            'No public publishing, repository push or owner license choice is performed by this export.']}
    return summary,rows,seed_rows


def markdown(summary):
    s=summary;lines=['# Coding pilot results','',
        'This report is a separate analysis. Original pilot and phase-2 reports and measurements remain unchanged.','',
        '**Status:** '+('Bounded measurements and second-seed confirmation complete.' if s['study_complete'] else 'Partial bounded result; the full study is not complete.'),'']
    if s['request'].get('scientific_stop'):lines+=['The pipeline stopped at a declared gate: '+json.dumps(s['request']['scientific_stop'],ensure_ascii=False), '']
    coverage=s['coverage'];lines+=['## Observations','',f"Verified headline coverage: **{coverage['observed']}/200 original tasks**. Verified second-seed coverage: **{s['second_seed']['coverage']['observed']}/40 reserved tasks**.",'']
    if s['primary_inference']:
        p=s['primary_inference'];lines+=['| Arm | Passes | Pass@1 |','|---|---:|---:|']
        for a in ARMS:lines.append(f"| {a} — {CONDITIONS[a]} | {p['passes'][a]}/200 | {p['pass_rates'][a]:.1%} |")
        lines+=['','All contrasts use 10,000 paired resamples of the same original tasks.','',
            '| Contrast | Difference, percentage points | 95% paired interval | Rescues / regressions |',
            '|---|---:|---:|---:|']
        for key in ['C-B','C-D','C-A']:
            r=p['contrasts'][key];lines.append(f"| {key} | {r['difference']*100:+.1f} | [{r['ci95'][0]*100:+.1f}, {r['ci95'][1]*100:+.1f}] | {r['rescues']} / {r['regressions']} |")
        gap=p['gap_closed'];lines+=['']
        if gap['reported']:lines+=[f"Secondary gap closed: {gap['value']:.3f}. The denominator is {gap['denominator']:.1%}; its interval excludes zero. This ratio is not a retained-performance or equivalence guarantee."]
        else:lines+=['Secondary gap closed is suppressed: '+gap['suppression_reason']]
        if any(r['degenerate_interval'] for r in p['contrasts'].values()):lines+=['At least one bootstrap interval is degenerate. This does not demonstrate population equivalence.']
        lines+=['']
    else:
        lines+=['Only descriptive counts are available: '+', '.join(f"{a}={s['observed_descriptive_passes'][a]}/{coverage['observed']}" for a in ARMS)+'.',
            'No confidence interval, gap-closed ratio, or positive transfer conclusion is drawn from this incomplete cohort.','']
    if s['baseline']:lines+=['Development gate evidence is preserved in `summary.json`. It is separate from headline confirmation.','']
    if s['second_seed']['inference']:
        r=s['second_seed']['inference'];d=r['second_minus_headline'];lines+=[f"On the reserved 40 tasks, headline C passed {r['headline_C_passes']} and second-seed C passed {r['second_seed_C_passes']}. The paired difference is {d['difference']*100:+.1f} points, interval [{d['ci95'][0]*100:+.1f}, {d['ci95'][1]*100:+.1f}]. These are the same original tasks, not extra independent samples.",'']
    lines+=['## Training and numerical scope','']
    if s['training']['runs']:
        lines+=['| Objective | Seed | Updates | Stop | Converged |','|---|---:|---:|---|---|']
        for r in s['training']['runs']:lines.append(f"| {r['objective']} | {r['seed']} | {r['steps']} | {r['stop_reason']} | {r['converged']} |")
        lines+=['','A capped, unconverged run does not establish a mapper-capacity ceiling.','']
    else:lines+=['No completed mapper training run was supplied to this analysis.','']
    lines+=[s['training']['caveat'],'','## Timing and costs','']
    if s['latency']:
        lines+=[s['latency']['policy'],s['latency']['scope'],'',
            '| Paired complete successes | Tasks | C minus comparator, seconds | C/comparator mean-time ratio |',
            '|---|---:|---:|---:|']
        for key,r in s['latency']['paired'].items():
            d='unavailable' if r['mean_seconds_difference'] is None else f"{r['mean_seconds_difference']:+.3f}"
            ratio='unavailable' if r['ratio_of_mean_source_inclusive_seconds'] is None else f"{r['ratio_of_mean_source_inclusive_seconds']:.3f}"
            lines.append(f"| {key} | {r['task_count']} | {d} | {ratio} |")
        lines+=['','Failed, capped or incomplete short answers are excluded from useful-speed comparisons. Timing subset selection is explicit; it does not establish full-cohort service speed.','']
    else:lines+=['No eligible confirmation timing comparison has been executed.','']
    if s['execution_wall_times']['stages']:
        lines+=['Actual stage wall-clock durations (including setup and backup): '+', '.join(
            r['stage']+'='+('unknown' if r['wall_seconds'] is None else f"{r['wall_seconds']/3600:.2f} h")
            for r in s['execution_wall_times']['stages'])+'.',s['execution_wall_times']['scope'],'']
    if s['costs']:lines+=[f"Recorded cumulative conservative spending: ${s['costs'].get('upper_usd')}; aggregate GPU-hours: {s['costs'].get('gpu_hours')}. The receipt timestamp and cleanup status are in `summary.json`; this is a cost upper estimate, not an invoice.",'']
    lines+=['## Inferences and unresolved work','',f"Recommendation: **{s['publication_recommendation']}**.",'']
    if not s['study_complete']:lines+=['The remaining required gates, complete confirmation, or replication have not all been established. The report preserves the bounded result without treating unexecuted work as successful.','']
    if s['integrity_issues']:lines+=['Some supplied records failed integrity validation. See `summary.json`; excluded records must be resolved before scientific publication.','']
    lines += ['- '+value for value in s['limits']]
    lines+=['','Source trajectories, raw compact records, code and hash inventories accompany this report. Heavy tensors are excluded; their existing hash/identity receipts are preserved where available. An explicit project license still requires the owner’s choice if no root LICENSE exists.','']
    return '\n'.join(lines)


def write_csv(output,rows):
    with (output/'task_outcomes.csv').open('w',newline='') as f:
        columns=['task_id',*ARMS,'source_capped','small_capped',*[a+'_source_inclusive_seconds' for a in ARMS],'source_root','transaction_sha256']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in rows:writer.writerow({'task_id':r['task_id'],**r['pass'],'source_capped':r['source_capped'],
            'small_capped':r['small_capped'],**{a+'_source_inclusive_seconds':r['outputs'][a].get('source_inclusive_seconds') for a in ARMS},
            'source_root':r['source_root'],'transaction_sha256':r['transaction_sha256']})


def plot_results(output,summary):
    if not summary['primary_inference'] and not summary['training']['runs']:return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=[]
    if summary['primary_inference']:
        p=summary['primary_inference'];keys=['C-B','C-D','C-A'];means=[100*p['contrasts'][k]['difference'] for k in keys]
        low=[m-100*p['contrasts'][k]['ci95'][0] for k,m in zip(keys,means)]
        high=[100*p['contrasts'][k]['ci95'][1]-m for k,m in zip(keys,means)]
        fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(keys,means,yerr=[low,high],fmt='o',capsize=5,color='#2369a2')
        ax.axhline(0,color='#999',linewidth=1);ax.set_ylabel('Pass@1 difference (percentage points)')
        ax.set_title('200 fixed tasks · paired task bootstrap, 95% intervals');fig.tight_layout()
        path=output/'paired_contrasts.png';fig.savefig(path,dpi=160);plt.close(fig);paths.append(path.name)
    curves=[r for r in summary['training']['runs'] if r['validation_curve']]
    if curves:
        fig,ax=plt.subplots(figsize=(7,4))
        for r in curves:ax.plot([x['step'] for x in r['validation_curve']],[x['validation_kl'] for x in r['validation_curve']],marker='.',label=f"{r['objective']} / {r['seed']}")
        ax.set_xlabel('Optimizer updates');ax.set_ylabel('Shared boundary validation KL');ax.legend(fontsize=8);fig.tight_layout()
        path=output/'learning_curves.png';fig.savefig(path,dpi=160);plt.close(fig);paths.append(path.name)
    return paths


def bundle_candidates(repo,excluded_output=None):
    roots=['gearshift','configs','tests','scripts','results','plots','evidence','gearshift_review','gearshift_phase2','.github','third_party']
    paths=set()
    for root in roots:
        directory=repo/root
        if not directory.is_dir():continue
        for base,dirs,files in os.walk(directory):
            dirs[:]=[d for d in dirs if d not in EXCLUDED_PARTS and not d.startswith(('resume_parent_','cap_control_parent_','recovery_parent_'))
                and not Path(base,d).is_symlink()]
            for name in files:paths.add(Path(base,name))
    paths.update(p for p in repo.iterdir() if p.is_file())
    paths.update(repo/p for p in ORIGINAL_TRAJECTORIES if (repo/p).is_file())
    visible=repo/'data/coding_pilot_v1/visible'
    if visible.exists():paths.update(visible.glob('*.json'))
    for name in ['identity.json','draft_membership.json']:
        path=repo/'data/coding_pilot_v1'/name
        if path.exists():paths.add(path)
    result=[];omissions=[]
    for path in sorted(paths):
        rel=path.relative_to(repo);parts=set(rel.parts)
        if path.is_symlink() or EXCLUDED_PARTS & parts or any(part.startswith('.env') for part in rel.parts):continue
        if excluded_output and path.resolve()==excluded_output.resolve():continue
        if path.name=='EXPORT.json' or str(rel) in ['REVIEW_BUNDLE_MANIFEST.json','REVIEW_README.md']:continue
        if path.suffix not in EXTENSIONS and path.name not in ['.gitignore','LICENSE','LICENSE.md','LICENSE.txt','uv.lock']:
            if path.suffix in ['.pt','.safetensors','.bin','.npy','.npz','.pkl','.pickle','.zip','.gz']:
                omissions.append({'path':str(rel),'bytes':path.stat().st_size,'reason':'Reproducible tensor, weight, cache or archive; existing hash receipts are included where available.'})
            continue
        if path.name.lower() in ['pod.json','connection.json','connections.json','provider_response.json','pod_create.json','pod_response.json']:
            omissions.append({'path':str(rel),'reason':'Raw provider/connection receipt excluded; sanitized allocation proofs are retained.'});continue
        if any(word in path.name.lower() for word in ['credentials','api_key','apikey','id_rsa','id_ed25519']):
            omissions.append({'path':str(rel),'reason':'Credential-bearing filename excluded.'});continue
        size=path.stat().st_size
        if size>MAX_FILE_BYTES:
            omissions.append({'path':str(rel),'bytes':size,'reason':'Above compact evidence file limit; no bytes copied.'});continue
        result.append(path)
    return result,omissions


def contains_credentials(value):
    forbidden={'apikey','api_key','authorization','access_token','refresh_token','client_secret','private_key',
        'ssh_private_key','password','public_key','env'}
    if isinstance(value,dict):
        return any(str(k).lower() in forbidden or contains_credentials(v) for k,v in value.items())
    if isinstance(value,list):return any(contains_credentials(v) for v in value)
    return False


def stable_bytes(path):
    for _ in range(3):
        before=path.stat();payload=path.read_bytes();after=path.stat()
        if before.st_size==after.st_size==len(payload) and before.st_mtime_ns==after.st_mtime_ns:return payload
    raise RuntimeError('File changed while review snapshot was read: '+str(path))


def build_bundle(repo,bundle_path,report_output):
    repo=Path(repo).resolve();bundle_path=Path(bundle_path).resolve()
    if bundle_path.exists():raise ValueError('Review ZIP already exists; preserve its identity')
    candidates,omissions=bundle_candidates(repo,bundle_path);files={};total=0
    temporary=bundle_path.with_name(bundle_path.name+'.tmp');bundle_path.parent.mkdir(parents=True,exist_ok=True)
    if temporary.exists():raise ValueError('Unfinished ZIP exists; inspect before retry')
    try:
        with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path in candidates:
                try:payload=stable_bytes(path)
                except (OSError,RuntimeError) as exc:
                    omissions.append({'path':str(path.relative_to(repo)),'reason':'Unstable/unreadable live file: '+str(exc)});continue
                if path.suffix=='.json':
                    try:credential_material=contains_credentials(json.loads(payload))
                    except (ValueError,UnicodeDecodeError):credential_material=False
                    if credential_material:
                        omissions.append({'path':str(path.relative_to(repo)),'reason':'Credential/environment fields excluded; no content copied.'});continue
                total+=len(payload)
                if total>MAX_BUNDLE_BYTES:raise ValueError('Compact bundle exceeds 1 GiB uncompressed; narrow redundant evidence explicitly')
                name=str(path.relative_to(repo));files[name]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
                archive.writestr('gearshift/'+name,payload)
            missing=[p for p in ORIGINAL_TRAJECTORIES if p not in files]
            manifest={'schema':1,'created_epoch':time.time(),'files':files,'uncompressed_bytes':total,'omissions':omissions,
                'missing_required_original_trajectories':missing,'source_trajectories_included':not missing,
                'report_directory':str(report_output.relative_to(repo)),
                'excluded_categories':['Git internals','Virtual environments','Credentials','Model weights','Mapper/feature/KV tensors',
                    'Private hidden tests','Heavy archives and duplicate extracted backups','Model caches','Reproducible intermediate artifacts'],
                'regeneration':'Use preserved run identities, pinned configs, lockfile and pipeline plans. Scientific reruns require fresh namespaces; saved outputs are never relabeled. Heavy checkpoints remain local with their existing hash receipts.',
                'license':'Root project LICENSE is present.' if any((repo/p).exists() for p in ['LICENSE','LICENSE.md','LICENSE.txt']) else 'Project license remains an owner decision; third-party notices are preserved.',
                'publishing_authorized':False}
            archive.writestr('gearshift/REVIEW_BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
            archive.writestr('gearshift/REVIEW_README.md',
                '# Coding pilot review bundle\n\nStart with `'+str(report_output.relative_to(repo))+'/CODING_RESULTS.md`, then `summary.json` and `task_outcomes.csv`. '
                'Inspect the preserved source histories, outputs, native/memory controls and transaction hashes before making claims. '
                'Original pilot and follow-up reports remain separate and unchanged. '
                'The bundle is a local review artifact; no publication or license selection occurred.\n\n'
                'Heavy tensors and private hidden tests are excluded. Existing checkpoint hashes and reconstruction code are retained. '
                'All source files in this snapshot are inventoried in `REVIEW_BUNDLE_MANIFEST.json`; archive-only README and manifest are outside its self-referential file list.\n')
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:raise ValueError('ZIP integrity check failed')
            expected={'gearshift/'+name for name in files}|{'gearshift/REVIEW_BUNDLE_MANIFEST.json','gearshift/REVIEW_README.md'}
            if set(archive.namelist())!=expected:raise ValueError('ZIP membership differs')
            for name,item in files.items():
                if hashlib.sha256(archive.read('gearshift/'+name)).hexdigest()!=item['sha256']:raise ValueError('Archived evidence hash differs')
        os.replace(temporary,bundle_path)
    except BaseException:
        # The incomplete temporary is intentionally retained for diagnosis.
        raise
    return {'path':str(bundle_path),'bytes':bundle_path.stat().st_size,'sha256':sha(bundle_path),
        'files':len(files)+2,'source_trajectories_included':not missing,'missing_required_original_trajectories':missing,
        'omitted_files':len(omissions)}


def verify_export(path):
    receipt=read(path)
    if receipt.get('export_complete') is not True:raise ValueError('Review export is incomplete')
    bundle=Path(receipt['bundle']['path'])
    if sha(bundle)!=receipt['bundle']['sha256']:raise ValueError('Review ZIP changed after verification')
    for name,value in receipt['report_files'].items():
        if sha(Path(path).parent/name)!=value:raise ValueError('Final report changed after verification')
    return receipt


def export(repo,request,output,bundle):
    repo=Path(repo).resolve();output=inside(repo,output);bundle=inside(repo,bundle)
    if (output/'EXPORT.json').exists():
        result=verify_export(output/'EXPORT.json')
        if result['request_sha256']!=digest(request):raise ValueError('Cannot reuse export for another request')
        return result
    if output.exists() and any(output.iterdir()):raise ValueError('Partial review output exists; inspect before creating a new immutable export')
    summary,rows,seed_rows=analyze(repo,request);output.mkdir(parents=True,exist_ok=True)
    write(output/'summary.json',summary);write(output/'request.json',request)
    (output/'CODING_RESULTS.md').write_text(markdown(summary))
    write_csv(output,rows)
    write(output/'second_seed_outcomes.json',[{'task_id':r['task_id'],'passed':r['pass']['C'],
        'source_root':r['source_root'],'transaction_sha256':r['transaction_sha256']} for r in seed_rows])
    plots=plot_results(output,summary)
    try:
        revision=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
        status=subprocess.run(['git','status','--short'],cwd=repo,capture_output=True,text=True,check=True).stdout
    except (OSError,subprocess.CalledProcessError):revision=None;status='Unavailable'
    write(output/'provenance.json',{'created_epoch':summary['created_epoch'],'git_commit':revision,'working_tree_status':status,
        'analysis_script_sha256':sha(Path(__file__)),'configuration_sha256':sha(repo/'configs/coding_pilot_v1/pilot.json'),
        'request_sha256':digest(request),'bootstrap_resamples':BOOTSTRAPS,'bootstrap_seed':BOOTSTRAP_SEED,
        'plots':plots,'historical_results_modified':False})
    bundle_info=build_bundle(repo,bundle,output)
    report_files={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    receipt={'schema':1,'export_complete':bundle_info['source_trajectories_included'],
        'study_complete':summary['study_complete'],'headline_confirmation_complete':summary['headline_confirmation_complete'],
        'request_sha256':digest(request),'bundle':bundle_info,'report_directory':str(output),'report_files':report_files,
        'publication_recommendation':summary['publication_recommendation'],'created_epoch':time.time()}
    write(output/'EXPORT.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    source=parser.add_mutually_exclusive_group(required=True);source.add_argument('--state',type=Path);source.add_argument('--request',type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--bundle',required=True,type=Path)
    args=parser.parse_args();repo=args.repo.resolve()
    request=request_from_state(repo,inside(repo,args.state)) if args.state else read(inside(repo,args.request))
    receipt=export(repo,request,args.output,args.bundle);print(json.dumps(receipt,indent=2))
    if not receipt['export_complete']:raise SystemExit('Review ZIP is missing required original source trajectories; see EXPORT.json')

if __name__=='__main__':main()
