#!/usr/bin/env python3
"""Compact-record latency, source-overlap, branching and fidelity diagnostics."""
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gearshift.phase2_io import read,write
from gearshift.phase2_reporting import csv_write,paired_cluster
from phase2_report import collect_stage,stratum


def interval(values):
    x=np.array(values,dtype=float);rng=np.random.default_rng(20260915)
    samples=x[rng.integers(0,len(x),(5000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)))


def main():
    root=Path('results/phase2_v1');dest=root/'report';dest.mkdir(exist_ok=True)
    paired=[];source_rows=[];overlap=[];repeats=[];fanout=[];fidelity=[]
    for stage in root.iterdir():
        if not (stage/'manifest.json').exists() or not (stage/'questions').is_dir():continue
        raw,scored,progress,trs=collect_stage(stage);by_family=defaultdict(list)
        stage_identity=read(stage/'manifest.json')['identity'];tasks={t['task_id']:t for t in stage_identity['tasks']}
        for tid,tr in trs.items():
            task=tasks[tid];by_family[task['family'],stratum(stage.name,tid)].append((tid,tr))
        for (family,subset),items in by_family.items():
            source_rows.append(dict(stage=stage.name,family=family,stratum=subset,n_task_variants=len(items),
                n_parents=len({tasks[tid]['cluster_id'] for tid,tr in items}),
                mean_prompt_tokens=float(np.mean([len(tr['prompt_ids']) for tid,tr in items])),
                mean_reasoning_tokens=float(np.mean([len(tr['source_reasoning_ids']) for tid,tr in items])),
                histories_with_at_most_four_reasoning_tokens=sum(len(tr['source_reasoning_ids'])<=4 for tid,tr in items),
                completed=sum(tr['source_completed'] for tid,tr in items),
                caps=sum(tr['stop_reason']=='cap' for tid,tr in items),
                mean_source_prefill_ms=float(np.mean([tr['source_prefill_ms'] for tid,tr in items])),
                mean_source_reasoning_ms=float(np.mean([tr['source_reasoning_ms'] for tid,tr in items])),
                origin=items[0][1]['source_cache_origin']))
        for (tid,c),r in raw.items():
            history=trs[tid]['source_reasoning_ids'];answer=r['answer_token_ids'];source4={tuple(history[i:i+4]) for i in range(len(history)-3)}
            answer4=[tuple(answer[i:i+4]) for i in range(len(answer)-3)]
            content=list(answer);eos=set(stage_identity['source' if c.split('/')[0]=='B' else 'target']['effective_eos_ids'])
            while content and content[-1] in eos:content.pop()
            present=any(history[i:i+len(content)]==content for i in range(len(history)-len(content)+1)) if content else None
            overlap.append(dict(stage=stage.name,task_id=tid,cluster_id=tasks[tid]['cluster_id'],family=tasks[tid]['family'],stratum=stratum(stage.name,tid),condition=c,
                answer_tokens=len(answer),source_reasoning_tokens=len(history),answer_4gram_fraction_in_source=float(np.mean([v in source4 for v in answer4])) if answer4 else None,
                complete_answer_content_sequence_in_source=present,answer_content_tokens=len(content),
                definition='Lexical token overlap, not proof of reasoning transfer or source-plan correctness'))
        for family,subset in sorted(by_family):
            selected=[r for r in raw.values() if r['family']==family and stratum(stage.name,r['task_id'])==subset]
            conditions={r['condition'] for r in selected};mapped=[c for c in conditions if c=='M' or c.startswith('M/')]
            references=[c for c in conditions if c in ['C','C/newturn','B','B/newturn','B/native','S','S_think','T','P','M/frozen','M/initial']]
            for a in mapped:
                for b in references:
                    if a==b:continue
                    ids={r['task_id'] for r in selected if r['condition']==a}&{r['task_id'] for r in selected if r['condition']==b}
                    if not ids:continue
                    for metric in ['total_wall_ms','handoff_wall_ms','first_answer_token_ms','answer_tokens','amortized_output_wall_ms']:
                        rows=[dict(task_id=r['task_id'],cluster_id=tasks[r['task_id']]['cluster_id'],condition=r['condition'],score={metric:r.get(metric)})
                              for r in selected if r['task_id'] in ids and r['condition'] in [a,b]]
                        ci=paired_cluster(rows,a,b,metric)
                        if ci:
                            means={c:float(np.mean([r['score'][metric] for r in rows if r['condition']==c and r['score'][metric] is not None])) for c in [a,b]}
                            paired.append(dict(stage=stage.name,family=family,stratum=subset,mean_a=means[a],mean_b=means[b],**ci))
        if stage.name=='timing_repeats':
            grouped=defaultdict(list)
            for r in raw.values():
                if not r['warmup']:grouped[r['task_id'],r['condition'].rsplit('/repeat_',1)[0]].append(r)
            for (tid,condition),rs in grouped.items():
                for metric in ['source_wall_ms','mapping_ms','prefill_ms','clone_ms','bridge_ms','first_answer_token_ms','answer_generation_ms','handoff_wall_ms','total_wall_ms','answer_tokens']:
                    v=[r[metric] for r in rs if r.get(metric) is not None]
                    if v:repeats.append(dict(task_id=tid,family=tasks[tid]['family'],condition=condition,metric=metric,repeats=len(v),
                        median=float(np.median(v)),p25=float(np.quantile(v,.25)),p75=float(np.quantile(v,.75)),minimum=min(v),maximum=max(v),
                        source_timing='One live source trajectory, shared across repeats; source generation is not rerun per repetition'))
        for p in (stage/'parents').glob('*.json'):
            obj=read(p)
            for condition,costs in obj['fanout_costs'].items():
                fanout.append(dict(stage=stage.name,parent_task_id=obj['task_id'],condition=condition,**costs,
                    mean_amortized_output_ms=costs['total_ms']/costs['outputs'],
                    mean_one_off_output_ms=costs['source_ms']+costs['prefix_setup_ms']+costs['summary_ms']+float(np.mean(costs['output_wall_ms']))))
            for r in obj['teacher_forced']:
                for w in r['windows']:fidelity.append(dict(stage=stage.name,task_id=r['task_id'],cluster_id=r['cluster_id'],family=tasks[r['task_id']]['family'],condition=r['condition'],**w))
    csv_write(dest/'paired_latency.csv',paired);csv_write(dest/'source_reasoning.csv',source_rows);csv_write(dest/'source_answer_overlap.csv',overlap)
    csv_write(dest/'repeated_timings.csv',repeats);csv_write(dest/'fanout_costs.csv',fanout);csv_write(dest/'teacher_forced_cases.csv',fidelity)
    groups=defaultdict(list)
    for r in fidelity:groups[r['family'],r['condition'],r['start'],r['end']].append(r)
    summaries=[]
    for (family,condition,start,end),rs in groups.items():
        for metric in ['kl','mapped_nll','native_nll','top1']:
            clusters=defaultdict(list)
            for r in rs:clusters[r['cluster_id']].append(r[metric])
            summaries.append(dict(family=family,condition=condition,start=start,end=end,metric=metric,n_parents=len(clusters),
                observed_tokens=sum(r['observed_tokens'] for r in rs),**interval([np.mean(v) for v in clusters.values()])))
    csv_write(dest/'teacher_forced_summary.csv',summaries)
    if summaries:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        families=sorted({r['family'] for r in summaries});fig,axes=plt.subplots(1,len(families),figsize=(6*len(families),4),squeeze=False)
        for ax,family in zip(axes[0],families):
            for condition in ['M','H']:
                rs=sorted([r for r in summaries if r['family']==family and r['condition']==condition and r['metric']=='kl'],key=lambda r:r['start'])
                ax.errorbar([r['start'] for r in rs],[r['mean'] for r in rs],yerr=[[r['mean']-r['ci_low'] for r in rs],[r['ci_high']-r['mean'] for r in rs]],marker='o',label=condition)
            ax.set(title=family,xlabel='First position of teacher-forced window',ylabel='Mean native-to-mapped KL');ax.legend();ax.grid(alpha=.2)
        fig.suptitle('Task-cluster intervals; native continuation is a fidelity reference, not truth');fig.tight_layout();fig.savefig(dest/'plots/teacher_forced_fidelity.png',dpi=160);plt.close(fig)
    training=[]
    for p in sorted((root/'training').glob('*/seed_*/*_curve.json')):
        curve=read(p);pair=p.parents[1].name;seed=p.parent.name;arm=p.name.removesuffix('_curve.json')
        for r in curve:
            for family,value in r['by_domain'].items():training.append(dict(pair=pair,seed=seed,arm=arm,step=r['step'],family=family,validation_kl=value))
    csv_write(dest/'training_curves.csv',training)
    exposures=[]
    for p in sorted((root/'training').glob('*/seed_*/training_steps.json')):
        grouped=defaultdict(list)
        for r in read(p):grouped[r['arm'],r['family'],r['requested_anchor']].append(r)
        for (arm,family,requested),rs in sorted(grouped.items()):
            exposures.append(dict(pair=p.parents[1].name,seed=p.parent.name,arm=arm,family=family,requested_anchor=requested,
                updates=len(rs),gradient_predictions=sum(r['prediction_tokens'] for r in rs),
                mean_actual_anchor=float(np.mean([r['actual_anchor'] for r in rs])),
                predictions_at_or_after_32=sum(r['prediction_tokens'] for r in rs if r['actual_anchor']>=32),
                predictions_at_or_after_128=sum(r['prediction_tokens'] for r in rs if r['actual_anchor']>=128),
                predictions_at_or_after_256=sum(r['prediction_tokens'] for r in rs if r['actual_anchor']>=256),
                mean_history_tokens=float(np.mean([r['history_tokens'] for r in rs])),
                mean_prefix_tokens=float(np.mean([r['prefix_tokens'] for r in rs])),
                note='Short teacher answers fall back to their final available window; requested anchors are matched but actual anchors may differ by protocol.'))
    csv_write(dest/'training_prediction_exposure.csv',exposures)
    seed_summary=[];common=root/'tasks/adaptation_reservation.json'
    if common.exists():
        ids=set(read(common)['all_seed_task_ids']);scores=root/'confirmation_1p7_to_0p6/objective_scores.json'
        if scores.exists():
            grouped=defaultdict(list)
            for r in read(scores)['rows']:
                if r['task_id'] in ids:grouped[r['family'],r['condition']].append(r)
            for (family,condition),rs in sorted(grouped.items()):
                metric={'arithmetic':'correct','code':'correct','evidence':'all_fields_correct','writing':'all_literal_constraints'}[family]
                seed_summary.append(dict(family=family,condition=condition,n=len(rs),metric=metric,mean=float(np.mean([r['score'][metric] for r in rs])),
                    population='Same prespecified 32 confirmation tasks (8 per family) for all seeds and baselines; do not multiply sample size by seeds'))
    csv_write(dest/'seed_variability_common_tasks.csv',seed_summary)
    if training:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for pair in sorted({r['pair'] for r in training}):
            fig,axes=plt.subplots(2,2,figsize=(10,7))
            for ax,family in zip(axes.flat,['arithmetic','code','evidence','writing']):
                for seed in sorted({r['seed'] for r in training if r['pair']==pair}):
                    for arm in ['ordinary','boundary']:
                        rs=[r for r in training if (r['pair'],r['family'],r['seed'],r['arm'])==(pair,family,seed,arm)]
                        if rs:ax.plot([r['step'] for r in rs],[r['validation_kl'] for r in rs],label=arm+'/'+seed.removeprefix('seed_'))
                ax.set(title=family,xlabel='Updates per objective',ylabel='Validation KL');ax.grid(alpha=.2);ax.legend(fontsize=6)
            fig.suptitle(pair+' / native-receiver validation, not final task quality');fig.tight_layout();fig.savefig(dest/'plots'/f'{pair}_training_curves.png',dpi=160);plt.close(fig)
    write(dest/'diagnostics_status.json',dict(paired_latency_rows=len(paired),fidelity_windows=len(fidelity),timing_rows=len(repeats),fanout_rows=len(fanout),
        notes=['All bootstrap units are original tasks or parents.','Reported first-answer-token latency starts at handoff; source and summary production are separately charged in total wall time.',
               'No ablation source-inclusive utility inference: its histories were replayed.','Final semantic quality and teacher-forced fidelity are separate measurements.']))
    print('Regenerated extended compact-record diagnostics')


if __name__=='__main__':main()
