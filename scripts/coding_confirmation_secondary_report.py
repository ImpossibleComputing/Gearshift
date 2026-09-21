#!/usr/bin/env python3
"""Separate second-training-seed report from closed compact receipts only."""
import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gearshift.coding_control import digest,sha,write
from scripts import coding_confirmation_report as primary
from scripts import coding_confirmation_secondary_score as scoring

CONDITIONS=scoring.CONDITIONS
REFERENCE_CONDITIONS=['A','B','D','P']
MAPPED_CONTRASTS=[['ROTATING_M','FIXED_M'],['ROTATING_H','FIXED_H'],['ROTATING_H','ROTATING_M']]
REFERENCE_CONTRASTS=[['ROTATING_H',c] for c in ['D','P','B','A']]


def statistics(rows,task_ids,analysis,references=None):
    """Never pool mapper seeds; references contain only primary native baselines."""
    primary.validate_analysis(analysis,task_ids)
    references=[] if references is None else references
    names=CONDITIONS+(REFERENCE_CONDITIONS if references else [])
    if any(r['condition'] not in CONDITIONS or r.get('cohort')!='secondary' or r.get('training_seed')!=20260919 for r in rows) or any(r['condition'] not in REFERENCE_CONDITIONS or r.get('cohort')!='primary' for r in references):raise ValueError('Training seed strata or native-only reference scope differs')
    lookup={}
    for r in [*rows,*references]:
        key=r['task_id'],r['condition'],r['seed_index']
        if key in lookup or type(r.get('missing')) is not bool or r['missing'] is not (r.get('passed') is None) or (r['passed'] is not None and type(r['passed']) is not bool):raise ValueError('Duplicated or ambiguous secondary/reference outcome')
        lookup[key]=r
    if set(lookup)!={(t,c,s) for t in task_ids for c in names for s in range(3)}:raise ValueError('Exact secondary/reference task-condition-seed coverage differs')
    low=np.zeros((len(task_ids),len(names)),dtype=np.int64);high=low.copy();task_rows=[]
    for i,t in enumerate(task_ids):
        for j,name in enumerate(names):
            values=[lookup[t,name,s]['passed'] for s in range(3)];n=sum(v is True for v in values);m=sum(v is None for v in values)
            low[i,j]=n;high[i,j]=n+m
            task_rows.append({'task_id':t,'condition':name,'source':'secondary_training_seed' if name in CONDITIONS else 'reused_primary_native_reference',
                'draws':3,'passed':n,'missing':m,'pass_rate':None if m else n/3,'lower':n/3,'upper':(n+m)/3})
    indices=np.random.Generator(np.random.PCG64(20260919)).integers(0,len(task_ids),(10000,len(task_ids)))
    blo,bhi=low[indices].sum(axis=1),high[indices].sum(axis=1);denominator=3*len(task_ids)
    quant=lambda values:np.quantile(values,[.025,.975],method='linear').tolist()
    conditions={}
    for j,name in enumerate(names):
        selected=[lookup[t,name,s] for t in task_ids for s in range(3)];missing=sum(r['missing'] for r in selected)
        conditions[name]={'source':'secondary_training_seed' if name in CONDITIONS else 'reused_primary_native_reference',
            'draws':denominator,'passed_draws':int(low[:,j].sum()),'missing_draws':missing,'pass_rate':None if missing else float(low[:,j].sum()/denominator),
            'ci95':None if missing else quant(blo[:,j]/denominator),'possible_mean_bounds':[float(low[:,j].sum()/denominator),float(high[:,j].sum()/denominator)],
            'sensitivity_lower_ci95':quant(blo[:,j]/denominator),'sensitivity_upper_ci95':quant(bhi[:,j]/denominator),
            'categories':dict(collections.Counter(r.get('category','unspecified') for r in selected))}
    contrasts={};gains=[]
    for a,b in MAPPED_CONTRASTS+(REFERENCE_CONTRASTS if references else []):
        i,j=names.index(a),names.index(b);lo,hi=low[:,i]-high[:,j],high[:,i]-low[:,j]
        complete=not conditions[a]['missing_draws'] and not conditions[b]['missing_draws']
        bootlo,boothi=(blo[:,i]-bhi[:,j])/denominator,(bhi[:,i]-blo[:,j])/denominator
        contrasts[a+'-'+b]={'positive':a,'negative':b,'role':'secondary_seed_replication' if [a,b]==MAPPED_CONTRASTS[0] else 'secondary_seed_exploratory',
            'reference_reused_from_primary':b in REFERENCE_CONDITIONS,'tasks':len(task_ids),'draws_per_condition':denominator,
            'difference':float(lo.sum()/denominator) if complete else None,'ci95':quant(bootlo) if complete else None,
            'point_estimate_complete':complete,'possible_mean_bounds':[float(lo.sum()/denominator),float(hi.sum()/denominator)],
            'sensitivity_lower_ci95':quant(bootlo),'sensitivity_upper_ci95':quant(boothi),
            'task_gains':int((lo>0).sum()),'task_losses':int((hi<0).sum()),'task_ties':int(((lo==0)&(hi==0)).sum()),
            'unresolved_tasks':int(((lo<=0)&(hi>=0)&(lo!=hi)).sum()),'multiplicity':'Exploratory unadjusted95% task-cluster intervals; no additional primary claim.'}
        for k,t in enumerate(task_ids):gains.append({'contrast':a+'-'+b,'task_id':t,'difference_lower':float(lo[k]/3),'difference_upper':float(hi[k]/3)})
    return {'cohort':'secondary','training_seed':20260919,'task_ids':task_ids,'task_clusters':len(task_ids),'answer_draws_per_task_condition':3,
        'secondary_answers':len(rows),'secondary_missing_answers':sum(r['missing'] for r in rows),'native_reference_draws_reused':len(references),
        'joint_resamples':10000,'bootstrap_seed':20260919,'rng':'numpy.random.Generator(numpy.random.PCG64(20260919))',
        'bootstrap_indices_sha256':digest(indices.tolist()),'numpy_version':np.__version__,
        'conditions':{k:v for k,v in conditions.items() if k in CONDITIONS},'primary_native_references':{k:v for k,v in conditions.items() if k in REFERENCE_CONDITIONS},
        'contrasts':contrasts,'primary_mapper_scores_pooled_or_selected':False,
        'reference_status':'All four declared primary native controls reused' if references else 'Not supplied; native-reference contrasts remain unreported',
        'missingness_policy':'Keep all tasks and3 fixed draws; null is infrastructure missing, never failure or dropped. Extreme0/1 sensitivity is not observed-data inference.',
        'interpretation':'Limited replication conditional on the common original96 initializer and104-history corpus. Training seeds remain separate; no best-seed selection, pooling, equivalence claim, or universal generalization claim.'},task_rows,gains,indices


def load_secondary(plan_path,repo=ROOT):
    c=scoring.validate_plan(repo,plan_path);p=c['plan'];expected=scoring.build_manifest(c)
    manifest_path=scoring.scoped(c['top'],p['manifest_path']);manifest=scoring.read(manifest_path)
    if manifest!=expected:raise ValueError('Final secondary score manifest differs')
    d=c['declaration'];analysis=primary.checked(repo,d['analysis_path'],d['inputs'][d['analysis_path']]);primary.validate_analysis(analysis,d['task_ids'])
    files={f['path']:f for f in c['closure']['files']};histories={};history_overhead={};history_rows=[];overhead_rows=[]
    for tid in d['task_ids']:
        path='primary/tasks/'+tid.replace('/','__')+'/large_history/source_history.json'
        if path not in files:raise ValueError('Reused primary history is not secondary-sealed')
        h=primary.checked(c['top'],path,files[path]['sha256']);histories[tid]=h
        history_overhead[tid]=primary.overhead_receipt(c['top'],files,Path(path).parent,h)
        history_rows.append({'task_id':tid,'reused_from_primary':True,'source_history_path':path,'source_history_sha256':files[path]['sha256'],
            'reasoning_tokens':len(h['reasoning_ids']),'reasoning_seconds':h.get('reasoning_seconds'),
            'natural_boundary':h['natural_boundary'],'reasoning_capped':h['reasoning_capped'],'early_eos':h['early_eos']})
    rows=[]
    for item in manifest['files']:
        r=primary.checked(c['top'],item['path'],item['sha256']);raw=primary.checked(c['top'],item['answer_path'],item['answer_sha256']);tid=r['task_id']
        sampler=Path(item['answer_path']).parent/'sampler';answer_overhead=None
        if raw.get('sampler_timing',{}).get('overhead') is not None:
            rel=str(sampler/'answer_record.json')
            if rel not in files:raise ValueError('Secondary sampler overhead is not closed')
            record=primary.checked(c['top'],rel,files[rel]['sha256']);answer_overhead=primary.overhead_receipt(c['top'],files,sampler,record)
            overhead_rows.append({'task_id':tid,'condition':r['condition'],'seed_index':r['seed_index'],**answer_overhead})
        rows.append({**{k:r[k] for k in ('task_id','condition','seed_index','answer_seed')},'training_seed':20260919,'cohort':'secondary',
            'passed':r['score_v2']['passed'],'missing':r['score_v2']['missing'],'category':r['score_v2']['category'],
            **primary.interface_diagnostics(r['code'],c['visible'][tid]['prompt']),
            'answer_tokens':len(raw['answer_ids']),'answer_ended_eos':raw['answer_ended_eos'],'answer_capped':raw['answer_capped'],
            'repeated_fourgram_fraction':primary.repeated_fourgrams(raw['answer_ids']),
            **primary.timing_record(raw,histories[tid],r['score_v2']),
            **primary.overhead_for_draw(answer_overhead,history_overhead[tid],False),
            'answer_path':item['answer_path'],'answer_sha256':item['answer_sha256'],'score_path':item['path'],'score_sha256':item['sha256']})
    setup=[]
    for relative,item in files.items():
        path=Path(relative)
        if path.name!='model_setup.json' and path.parent.name!='mapper_loads':continue
        value=primary.checked(c['top'],relative,item['sha256'])
        if value.get('declaration_sha256')!=p['declaration_sha256'] or value.get('charged_to_single_output_inference') is not False:raise ValueError('Secondary setup receipt identity differs')
        loading=value.get('model_loading_seconds',{})
        setup.append({'kind':'worker_models' if path.name=='model_setup.json' else 'mapper_checkpoint','path':relative,'sha256':item['sha256'],
            'source_loading_seconds':primary.seconds(loading.get('source')),'receiver_loading_seconds':primary.seconds(loading.get('receiver')),
            'mapper_initialization_seconds':primary.seconds(value.get('mapper_initialization_seconds')),'checkpoint_load_seconds':primary.seconds(value.get('checkpoint_load_seconds'))})
    return c,manifest,rows,analysis,history_rows,overhead_rows,setup


def primary_references(c,plan_path,repo):
    if plan_path is None:return [],None
    loaded=primary.load_records(plan_path,repo);p=loaded['plan'];q=c['plan']
    if (p['declaration_sha256']!=q['declaration_sha256'] or p['task_ids']!=q['task_ids'] or
        p['generation_closure_identity_sha256']!=q['primary_generation_closure_identity_sha256'] or
        p['generation_closure_file_sha256']!=q['primary_generation_closure_file_sha256'] or p['scorer_identity_sha256']!=q['scorer_identity_sha256']):raise ValueError('Primary native references have different tasks, histories or scoring policy')
    return [{**r,'cohort':'primary'} for r in loaded['rows'] if r['condition'] in REFERENCE_CONDITIONS],loaded['bindings']


def report(plan_path,repo=ROOT,primary_plan=None,output=None):
    c,manifest,rows,analysis,histories,overheads,setup=load_secondary(plan_path,repo)
    references,reference_binding=primary_references(c,primary_plan,repo)
    summary,task_rows,gains,indices=statistics(rows,c['plan']['task_ids'],analysis,references)
    summary.update(experiment_id=c['plan']['experiment_id'],primary_training_seed=c['declaration']['primary_training_seed'],
        declaration_sha256=c['declaration_sha256'],secondary_declaration_sha256=c['secondary_declaration_sha256'],
        checkpoints=c['secondary_checkpoints'],replication_analysis=c['secondary_declaration']['replication_analysis'],scoring_plan_sha256=c['plan_sha256'],
        scored_manifest_sha256=sha(c['top']/c['plan']['manifest_path']),primary_reference_binding=reference_binding,
        diagnostics={k:v for k,v in primary.diagnostics_summary(rows).items() if k in CONDITIONS},
        setup_measurements=setup,history_diagnostics=histories,
        timing_policy='Full reused primary source history is charged to each single-answer stage-sum estimate. H pays native prompt prefill. Cache reconstruction/model loading/checkpoint I/O remain separate; failed-answer shortening is not useful speedup. Fleet accounting does not charge shared historical generation twice.')
    out=scoring.output_path(c['top'],'secondary/report') if output is None else Path(output).resolve()
    if out.is_relative_to((c['top']/'primary').resolve()):raise ValueError('Secondary report cannot overwrite primary artifacts')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'summary.json',summary);write(out/'joint_bootstrap_indices.json',indices.tolist())
    primary.csv_write(out/'per_draw.csv',rows);primary.csv_write(out/'per_task.csv',task_rows);primary.csv_write(out/'gains_losses.csv',gains)
    if overheads:primary.csv_write(out/'sampler_overheads.csv',overheads)
    if setup:primary.csv_write(out/'setup_measurements.csv',setup)
    lines=['# Confirmation: separate second training seed','',
        f"Training seed `{20260919}`; {summary['task_clusters']} shared frozen tasks ×4 mapped/hybrid conditions ×3 answer draws = {len(rows)} secondary answers; {summary['secondary_missing_answers']} missing.",'',
        f"Primary declaration `{c['declaration_sha256']}`; secondary addendum `{c['secondary_declaration_sha256']}`. The primary report and original training seed remain independent. This report does not pool the two training seeds or select the better seed/checkpoint.",'',
        '| Secondary condition | Passed / draws | Missing | Mean success |','|---|---:|---:|---:|']
    for name,value in summary['conditions'].items():
        rate='incomplete' if value['pass_rate'] is None else f"{100*value['pass_rate']:.2f}%"
        lines.append(f"|{name}|{value['passed_draws']}/{value['draws']}|{value['missing_draws']}|{rate}|")
    lines+=['','All intervals below are exploratory, unadjusted95% paired task-cluster intervals. The same10,000 PCG64 task-index resamples and task order as the primary analysis are used; each task averages all three fixed draws. Intervals touching/crossing zero are inconclusive, not equivalence.','',
        '| Contrast | Difference, percentage points | 95% interval | Source of negative condition |','|---|---:|---|---|']
    for name,value in summary['contrasts'].items():
        delta='incomplete' if value['difference'] is None else f"{100*value['difference']:+.2f}"
        ci='missingness sensitivities in summary.json' if value['ci95'] is None else f"[{100*value['ci95'][0]:+.2f}, {100*value['ci95'][1]:+.2f}]"
        lines.append(f"|{name}|{delta}|{ci}|{'Reused primary native reference' if value['reference_reused_from_primary'] else 'Same secondary training seed'}|")
    lines+=['',summary['reference_status']+'. Native A/B/D/P records, when shown, are reused primary measurements, not new secondary generation or additional replication draws. Primary mapped outcomes are never pooled into this analysis.','',
        'Missing infrastructure outcomes remain explicit with full-denominator extreme0/1 sensitivities. No complete-case or best-of-three substitution is made. Two paired training seeds provide limited replication conditional on one initializer and corpus; this is not exhaustive run-to-run uncertainty.','',summary['timing_policy'],'',
        'A result against P does not isolate dependence on the particular transferred reasoning. No mechanism or architecture claim follows from this secondary comparison.','',
        'Frozen secondary mapper endpoints:','']
    for arm,cp in c['secondary_checkpoints'].items():lines.append(f"- {arm}: step1024; mapper `{cp['mapper_sha256']}`.")
    lines+=['',f"Joint task-resample matrix identity: `{summary['bootstrap_indices_sha256']}`.",'',
        'Regeneration uses the compact generation/scoring manifests, raw answers, histories and receipts only. It does not load model weights, read private tests, execute candidates or modify the primary report.']
    text='\n'.join(lines)+'\n'
    for a,b in {'×4':'× 4','×3':'× 3','unadjusted95%':'unadjusted 95%','same10,000':'same 10,000','extreme0/1':'extreme 0/1','step1024':'step 1024'}.items():text=text.replace(a,b)
    (out/'CONFIRMATION_SECONDARY_RESULTS.md').write_text(text)
    write(out/'FILE_MANIFEST.json',{'cohort':'secondary','training_seed':20260919,'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()]})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--repo',default=str(ROOT));p.add_argument('--primary-plan');p.add_argument('--output');a=p.parse_args()
    value=report(a.plan,a.repo,a.primary_plan,a.output);print(json.dumps({'secondary_answers':value['secondary_answers'],'missing':value['secondary_missing_answers']}))
