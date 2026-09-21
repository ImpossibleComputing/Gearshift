#!/usr/bin/env python3
"""Regenerate phase-two tables and figures using compact records only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gearshift.phase2_io import read,write,digest,artifact,validate_transaction
from gearshift.phase2_reporting import csv_write,paired_cluster


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def stratum(stage,task_id):
    if stage in ['long_outputs','branching']:return task_id.rsplit('_',1)[-1]
    return 'primary'


def objective_pairs(conditions):
    """Fixed contrasts, including each seed's matched training-objective arms."""
    mapped=[c for c in conditions if c=='M' or c.startswith('M/')]
    references=[c for c in conditions if c in ['C','C/newturn','B','B/newturn','B/native','S','S_think','T','P','H','H/selected','M/frozen','M/initial']]
    pairs={(a,b) for a in mapped for b in references if a!=b}
    for boundary in conditions:
        if boundary.startswith('M/boundary/'):
            ordinary=boundary.replace('M/boundary/','M/ordinary/',1)
            if ordinary in conditions:pairs.add((boundary,ordinary))
    return pairs


def collect_stage(root):
    mf=read(root/'manifest.json');identity=mf['identity']
    if digest(identity)!=mf['identity_sha256']:raise ValueError('Stage manifest changed')
    tasks={t['task_id']:t for t in identity['tasks']};raw={};trajectories={}
    for p in sorted((root/'questions').glob('*.json')):
        obj=read(p);tid=obj['task_id'];conditions=identity['conditions']
        if not isinstance(conditions,list):conditions=conditions[tid]
        validate_transaction(obj,mf['identity_sha256'],tid,conditions)
        trajectories[tid]=obj['trajectory']
        for r in obj['rows']:raw[tid,r['condition']]=r
    score_path=root/'objective_scores.json';scored=read(score_path)['rows'] if score_path.exists() else []
    for r in scored:
        if (r['task_id'],r['condition']) not in raw:raise ValueError('Score without canonical output')
    return raw,scored,dict(planned=len(tasks),completed=len(trajectories),state='complete' if (root/'complete.json').exists() else 'partial'),trajectories


def objective_tables(root,dest):
    raw,scored,progress,trajectories=collect_stage(root);groups=defaultdict(list)
    for r in scored:groups[r['family'],stratum(root.name,r['task_id']),r['condition']].append(r)
    summary=[]
    for (family,subset,condition),g in sorted(groups.items()):
        outputs=[raw[r['task_id'],condition] for r in g];metric=g[0]['score'].get('primary_metric')
        metric={'hidden_test_pass':'correct'}.get(metric,metric)
        values=[r['score'].get(metric) for r in g] if metric else []
        values=[v for v in values if v is not None]
        entry=dict(stage=root.name,family=family,stratum=subset,condition=condition,n=len(g),scored_n=len(values),metric=metric,
            mean_score=float(np.mean(values)) if values else None,
            mean_output_tokens=float(np.mean([r['answer_tokens'] for r in outputs])),
            mean_words=float(np.mean([r['score']['words'] for r in g])),
            mean_repeated_4grams=float(np.mean([r['score']['repeated_4grams'] for r in g])),
            mean_repeated_4gram_fraction=float(np.mean([r['score']['repeated_4grams']/max(1,r['score']['words']-3) for r in g])),
            eos=sum(r['answer_stop_reason']=='eos' for r in outputs),caps=sum(r['answer_stop_reason']=='cap' for r in outputs),
            mean_total_ms=float(np.mean([r['total_wall_ms'] for r in outputs])),
            median_total_ms=float(np.median([r['total_wall_ms'] for r in outputs])),
            mean_handoff_ms=float(np.mean([r['handoff_wall_ms'] for r in outputs])),
            mean_source_ms=float(np.mean([r['source_wall_ms'] for r in outputs])),
            mean_mapping_ms=float(np.mean([r['mapping_ms'] for r in outputs])),
            mean_prefill_ms=float(np.mean([r['prefill_ms'] for r in outputs])),
            mean_bridge_ms=float(np.mean([r['bridge_ms'] for r in outputs])),
            mean_generation_ms=float(np.mean([r['answer_generation_ms'] for r in outputs])),
            mean_summary_ms=float(np.mean([r.get('summary_wall_ms',0) for r in outputs])),
            mean_first_logit_ms=float(np.mean([r['first_answer_logit_ms'] for r in outputs])))
        first=[r['first_answer_token_ms'] for r in outputs if r.get('first_answer_token_ms') is not None]
        entry['mean_first_answer_token_ms']=float(np.mean(first)) if first else None
        if all('amortized_output_wall_ms' in r for r in outputs):
            entry['mean_amortized_output_ms']=float(np.mean([r['amortized_output_wall_ms'] for r in outputs]))
        for flag in ['correct','format_valid','correct_and_valid','all_fields_correct','field_format_valid','all_fields_correct_and_valid','all_literal_constraints','source_marker_coverage','semantic_field_accuracy']:
            vals=[r['score'][flag] for r in g if r['score'].get(flag) is not None]
            if vals:entry[flag+'_mean']=float(np.mean(vals))
        if family=='code':
            statuses=defaultdict(int)
            for r in g:statuses[r['score']['code']['status']]+=1
            entry['code_statuses']=json.dumps(dict(statuses),sort_keys=True)
        if family=='evidence' and root.name in ['characterization','long_outputs','branching','timing_repeats']:
            entry['task_validity_note']='Undefined pricing unit in characterization parent; see EVIDENCE_AUDIT.md. Frozen scores unchanged.'
        summary.append(entry)
    comparisons=[]
    for family,subset in sorted({(r['family'],stratum(root.name,r['task_id'])) for r in scored}):
        rows=[r for r in scored if r['family']==family and stratum(root.name,r['task_id'])==subset];conditions=sorted({r['condition'] for r in rows})
        pairs=objective_pairs(conditions)
        if root.name=='ablation':
            pairs.update((f'M/{a}/newturn',f'M/{a}/old') for a in ['initial','plaintext','chat'])
            pairs.update([('M/chat/newturn','M/initial/newturn'),('M/chat/newturn','M/plaintext/newturn')])
        metrics={'arithmetic':['correct','format_valid','correct_and_valid'],'code':['correct'],
                 'evidence':['all_fields_correct','field_format_valid','source_marker_coverage'],
                 'writing':['all_literal_constraints','literal_constraint_fraction']}[family]
        for a,b in sorted(pairs):
            for metric in metrics:
                r=paired_cluster(rows,a,b,metric)
                if r:
                    by={c:{x['task_id']:x['score'][metric] for x in rows if x['condition']==c and x['score'].get(metric) is not None} for c in [a,b]}
                    common=set(by[a])&set(by[b])
                    comparisons.append(dict(stage=root.name,family=family,stratum=subset,**r,
                        mean_a=float(np.mean([by[a][i] for i in common])),mean_b=float(np.mean([by[b][i] for i in common])),
                        total_ms_a=float(np.mean([raw[i,a]['total_wall_ms'] for i in common])),total_ms_b=float(np.mean([raw[i,b]['total_wall_ms'] for i in common]))))
    csv_write(dest/f'{root.name}_summary.csv',summary);csv_write(dest/f'{root.name}_paired.csv',comparisons)
    write(dest/f'{root.name}_summary.json',dict(progress=progress,summary=summary,paired=comparisons))
    return summary,comparisons,progress


def judge_tables(root,dest):
    folder=root/'judging';key_path=folder/'condition_key.json'
    if not key_path.exists():return [],[],dict(packets=0,scored=0,pairs_complete=0)
    keys=read(key_path);pairs=defaultdict(list);scored_packets=0
    for key in keys:
        packet=read(folder/'packets'/f'{key["packet_id"]}.json')
        if digest(packet)!=key['packet_sha256']:raise ValueError('Judge packet key mismatch')
        p=folder/'judgments'/key['packet_id']/'result.json'
        if not p.exists():continue
        result=read(p)
        if result['packet_sha256']!=key['packet_sha256']:raise ValueError('Judgment belongs to another packet')
        if result['status']!='scored':continue
        scored_packets+=1;pairs[key['pair_id']].append((key,result['judgment']))
    aggregate=defaultdict(list);individual=[];differences=[];disagreements=[]
    for pair_id,oriented in pairs.items():
        if len(oriented)!=2 or {k['orientation'] for k,j in oriented}!={0,1}:continue
        key=oriented[0][0];left,right=key['left'],key['right'];winners=[];accepted={left:[],right:[]};scores={left:defaultdict(list),right:defaultdict(list)}
        for k,j in oriented:
            winners.append(k['condition_by_label'].get(j['preference'],j['preference']))
            for label,c in k['condition_by_label'].items():
                accepted[c].append(j[label]['acceptable'])
                for d in ['task_fulfillment','correctness_consistency','coverage','clarity_style']:scores[c][d].append(j[label][d])
        status='order_disagreement' if winners[0]!=winners[1] else ('left_preferred' if winners[0]==left else 'right_preferred' if winners[0]==right else winners[0])
        acceptance_disagreement=any(values[0]!=values[1] for values in accepted.values())
        score_disagreement=any(values[0]!=values[1] for dims in scores.values() for values in dims.values())
        item=dict(stage=root.name,pair_id=pair_id,task_id=key['task_id'],cluster_id=key['cluster_id'],family=key['family'],stratum=stratum(root.name,key['task_id']),left=left,right=right,
            outcome=status,left_acceptable=float(np.mean(accepted[left])),right_acceptable=float(np.mean(accepted[right])),orientations=winners,
            preference_order_disagreement=status=='order_disagreement',acceptability_order_disagreement=acceptance_disagreement,dimension_order_disagreement=score_disagreement,
            left_dimensions={d:float(np.mean(v)) for d,v in scores[left].items()},right_dimensions={d:float(np.mean(v)) for d,v in scores[right].items()})
        individual.append(item);aggregate[key['family'],item['stratum'],left,right].append(item)
        if status=='order_disagreement' or acceptance_disagreement or score_disagreement:disagreements.append(item)
    summaries=[]
    for (family,subset,left,right),items in sorted(aggregate.items()):
        counts={s:sum(r['outcome']==s for r in items) for s in ['left_preferred','right_preferred','tie','neither_acceptable','order_disagreement']}
        entry=dict(stage=root.name,family=family,stratum=subset,left=left,right=right,n_pairs=len(items),n_clusters=len({r['cluster_id'] for r in items}),
            **counts,left_acceptable=float(np.mean([r['left_acceptable'] for r in items])),right_acceptable=float(np.mean([r['right_acceptable'] for r in items])))
        entry['acceptability_order_disagreements']=sum(r['acceptability_order_disagreement'] for r in items)
        entry['dimension_order_disagreements']=sum(r['dimension_order_disagreement'] for r in items)
        for side in ['left','right']:
            for dimension in ['task_fulfillment','correctness_consistency','coverage','clarity_style']:
                entry[side+'_'+dimension]=float(np.mean([r[side+'_dimensions'][dimension] for r in items]))
        summaries.append(entry)
        clustered=[]
        for r in items:
            for condition,value in [(left,r['left_acceptable']),(right,r['right_acceptable'])]:
                clustered.append(dict(task_id=r['task_id'],cluster_id=r['cluster_id'],condition=condition,score={'acceptable':value}))
        ci=paired_cluster(clustered,left,right,'acceptable')
        if ci:differences.append(dict(stage=root.name,family=family,stratum=subset,**ci))
    csv_write(dest/f'{root.name}_judge_summary.csv',summaries);csv_write(dest/f'{root.name}_judge_paired.csv',differences)
    write(dest/f'{root.name}_judge_cases.json',individual);write(folder/'order_disagreements.json',disagreements)
    return summaries,differences,dict(packets=len(keys),scored=scored_packets,pairs_complete=len(individual),pending_or_invalid=len(keys)-scored_packets)


def plot_judged(root,dest):
    case_path=dest/f'{root.name}_judge_cases.json'
    if not case_path.exists():return
    cases=read(case_path)
    if not cases:return
    raw,_,_,_=collect_stage(root);groups=defaultdict(list)
    for r in cases:groups[r['family'],r['stratum'],r['left'],r['right']].append(r)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries=[]
    for (family,subset,left,right),rs in groups.items():
        fig,ax=plt.subplots(figsize=(6,4));xy=[]
        time_metric='amortized_output_wall_ms' if root.name=='branching' else 'total_wall_ms'
        time_label='Mean seconds per output, source and prefix paid once across three branches' if root.name=='branching' else 'Mean total wall time (seconds)'
        for side,condition in [('left',left),('right',right)]:
            x=float(np.mean([raw[r['task_id'],condition][time_metric] for r in rs]))/1000
            y=100*float(np.mean([r[side+'_acceptable'] for r in rs]));xy.append((x,y));ax.scatter(x,y,s=60)
            ax.annotate(condition,(x,y),xytext=(5,5),textcoords='offset points',fontsize=9)
            summaries.append(dict(stage=root.name,family=family,stratum=subset,pair_left=left,pair_right=right,condition=condition,
                n_pairs=len(rs),n_parents=len({r['cluster_id'] for r in rs}),mean_total_seconds=x,time_metric=time_metric,acceptable_fraction=y/100,
                timing_population='Same judged cases for both conditions; two orientations averaged within each case'))
        ax.plot(*zip(*xy),linestyle=':',color='gray');ax.set(xlabel=time_label,ylabel='Strict judged acceptability (%)',ylim=(-5,105),
            title=f'{family} / {subset}: {len(rs)} paired cases');ax.grid(alpha=.2)
        caveat='; evidence pricing unit ambiguous: EVIDENCE_AUDIT.md' if family=='evidence' and root.name in ['characterization','long_outputs','branching'] else ''
        fig.suptitle(root.name+'; see paired intervals and order disagreements'+caveat,fontsize=8);fig.tight_layout()
        name=(root.name+'_'+family+'_'+subset+'_'+left+'_vs_'+right).replace('/','-')
        fig.savefig(dest/'plots'/f'{name}_judged_quality_latency.png',dpi=160);plt.close(fig)
    csv_write(dest/f'{root.name}_judged_quality_latency.csv',summaries)


def plot(summary,destination,title):
    summary=[r for r in summary if r['mean_score'] is not None]
    if not summary:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    families=sorted({r['family'] for r in summary});fig,axes=plt.subplots(1,len(families),figsize=(5*len(families),5.5),squeeze=False)
    for ax,family in zip(axes[0],families):
        full_n=max(r['n'] for r in summary if r['family']==family)
        for r in summary:
            if r['family']!=family or r['n']!=full_n:continue
            time_value=r.get('mean_amortized_output_ms',r['mean_total_ms'])
            ax.scatter(time_value/1000,100*r['mean_score'],s=35,label=r['condition'])
        time_label='Mean seconds per output (three branches)' if any('mean_amortized_output_ms' in r for r in summary) else 'Mean total wall time (seconds)'
        ax.set(title=family+f' / shared full set (N={full_n})',xlabel=time_label,ylabel='Objective metric (%)',ylim=(-5,105));ax.grid(alpha=.2)
        if family=='evidence' and any(r.get('task_validity_note') for r in summary if r['family']==family):
            ax.set_title(f'evidence / N={full_n}\nPricing unit ambiguous: see EVIDENCE_AUDIT.md',fontsize=9)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=7)
    fig.suptitle(title+'\nEvidence/writing checks are separate from judged semantic acceptability',fontsize=10)
    fig.tight_layout();fig.savefig(destination,dpi=160);fig.savefig(destination.with_suffix('.svg'));plt.close(fig)


def plot_text_alternatives(comparisons,destination,title):
    metrics={'arithmetic':'correct','code':'correct','evidence':'all_fields_correct','writing':'all_literal_constraints'}
    rows=[r for r in comparisons if r['a']=='M' and r['b'] in ['T','P'] and r['metric']==metrics[r['family']]]
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(rows),1,figsize=(7,3.5*len(rows)),squeeze=False)
    for ax,r in zip(axes.flat,rows):
        for c,side in [(r['a'],'a'),(r['b'],'b')]:
            ax.scatter(r['total_ms_'+side]/1000,100*r['mean_'+side],s=60,label=c)
        ax.set(title=f'{r["family"]} / {r["stratum"]}: {r["n_clusters"]} shared parent cases; M−{r["b"]} {100*r["difference"]:.1f} pp [{100*r["ci_low"]:.1f}, {100*r["ci_high"]:.1f}]',
            xlabel='Mean total wall time (seconds)',ylabel='Objective checks (%)',ylim=(-5,105));ax.legend();ax.grid(alpha=.2)
    fig.suptitle(title+' / matched populations; prose checks are not semantic acceptability',fontsize=10)
    fig.tight_layout();fig.savefig(destination,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/phase2_v1');p.add_argument('--destination');args=p.parse_args()
    root=Path(args.root);dest=Path(args.destination) if args.destination else root/'report';dest.mkdir(parents=True,exist_ok=True)
    plots=dest/'plots';plots.mkdir(exist_ok=True);sections=['# Phase-two compact-record tables','These tables use raw outputs and stored scores only. No model weights, mapper checkpoints or cache tensors are loaded. Missing checkpoints are not verified. Statistical resampling keeps original parent tasks together; no equivalence claim follows from an interval crossing zero.']
    all_summary=[];all_paired=[];all_judges=[];statuses=[]
    for stage in sorted(root.iterdir()):
        if not stage.is_dir() or not (stage/'manifest.json').exists() or not (stage/'questions').is_dir():continue
        summary,paired,progress=objective_tables(stage,dest);judge,jci,jstatus=judge_tables(stage,dest)
        statuses.append(dict(stage=stage.name,**progress,judging=jstatus));all_summary.extend(summary);all_paired.extend(paired);all_judges.extend(judge)
        sections+=['## '+stage.name,f'Raw tasks: {progress["completed"]}/{progress["planned"]}; state: {progress["state"]}. Valid judged orientations: {jstatus["scored"]}/{jstatus["packets"]}.']
        if stage.name in ['characterization','long_outputs','branching','timing_repeats']:
            sections.append('Evidence tasks inherit an undefined pricing unit. Read EVIDENCE_AUDIT.md before interpreting their frozen numeric grades or semantic judgments. Original scores remain unchanged; the separate sensitivity analysis is not a replacement primary result.')
        if stage.name=='timing_repeats':
            sections.append('Descriptive timing subset: two cases per family. The raw table and scatter plot retain each repetition separately, including warmup 0; repetitions are not independent quality samples. Measured-repeat medians and dispersion exclude warmup and are regenerated by phase2_diagnostics_report.py in report/repeated_timings.csv. Different output lengths and pending semantic acceptability prevent a quality-matched speed claim.')
        if summary:
            sections.append(md_table(['Family / stratum','Condition','N','Metric','Score','Mean output tokens','EOS/cap','Mean total seconds'],
                [[r['family']+' / '+r['stratum'],r['condition'],r['n'],r['metric'],f'{100*r["mean_score"]:.1f}%' if r['mean_score'] is not None else 'pending',f'{r["mean_output_tokens"]:.1f}',f'{r["eos"]}/{r["caps"]}',f'{r["mean_total_ms"]/1000:.3f}'] for r in summary]))
            if stage.name!='ablation':
                for subset in sorted({r['stratum'] for r in summary}):
                    title=stage.name+' / '+subset
                    if stage.name=='timing_repeats':title+=' / raw repetitions including warmup 0; not median estimates'
                    plot([r for r in summary if r['stratum']==subset],plots/f'{stage.name}_{subset}_quality_latency.png',title)
        if judge:sections.append(md_table(['Family / stratum','Pair','N','Left wins','Right wins','Ties','Neither','Preference order disagreements','Acceptable left/right'],
            [[r['family']+' / '+r['stratum'],r['left']+' vs '+r['right'],r['n_pairs'],r['left_preferred'],r['right_preferred'],r['tie'],r['neither_acceptable'],r['order_disagreement'],f'{100*r["left_acceptable"]:.1f}% / {100*r["right_acceptable"]:.1f}%'] for r in judge]))
        plot_judged(stage,dest)
        plot_text_alternatives(paired,plots/f'{stage.name}_text_alternatives.png',stage.name)
    memory=root/'memory/counterfactual_scores_v2.json'
    if memory.exists():
        rows=read(memory)['summary'];csv_write(dest/'memory_counterfactual.csv',rows)
        sections+=['## Paired counterfactual memory diagnostic',md_table(['Condition','N variants','Decision correct','All fields correct','Field format valid','Conflicting fields'],
            [[r['condition'],r['n'],r['decision_correct'],r['all_fields_correct'],r['field_format_valid'],r['conflicting']] for r in rows]),
            'Twenty-four variants belong to twelve parent packets. Corrected field scoring accepts consistent repetition semantically and resolves source-ID aliases; field formatting and contradictions remain separate. The original strict scores are retained.']
    curves=[]
    for path in sorted((root/'training').glob('*/seed_*/complete.json')):
        result=read(path)
        for arm,desc in result['arms'].items():curves.append(dict(pair=path.parents[1].name,seed=path.parent.name,arm=arm,steps=result['steps_per_arm'],stop_reason=result['stop_reason'],**{k:v for k,v in desc.items() if k!='checkpoint'}))
    if curves:csv_write(dest/'training_summary.csv',curves);sections+=['## Training',md_table(['Pair','Seed','Objective','Updates','Best step','Validation KL','Stop'],[[r['pair'],r['seed'],r['arm'],r['steps'],r['best_step'],f'{r["validation_kl"]:.5f}',r['stop_reason']] for r in curves])]
    csv_write(dest/'all_objective_summary.csv',all_summary);csv_write(dest/'all_paired_objective.csv',all_paired);csv_write(dest/'all_judge_summary.csv',all_judges)
    write(dest/'status.json',statuses);(dest/'TABLES.md').write_text('\n\n'.join(sections)+'\n')
    print('Regenerated compact-record tables and plots:',dest)


if __name__=='__main__':main()
