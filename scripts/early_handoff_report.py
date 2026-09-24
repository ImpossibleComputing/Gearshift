#!/usr/bin/env python3
"""Whitelist-only public analysis of fixed-fraction early handoff; never exports raw text/tokens."""
import argparse,collections,csv,json,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gearshift.early_handoff import read,write,sha,reasoning_body,CONDITIONS

def collect(storage,d):
 records=[]
 lengths=read(storage/'control/prepared_inputs.json')['prompt_token_lengths']
 seal=read(storage/'control/generation_seal.json')
 if not seal['all_generation_complete'] or seal['expected_answers']!=840:raise ValueError('Complete generation required')
 for x in seal['files']:
  if sha(storage/x['path'])!=x['sha256']:raise ValueError('Sealed generation changed')
 if not (storage/'control/scoring_complete.json').exists():raise ValueError('Complete score inventory required')
 for task in d['population']['task_ids']:
  source=read(storage/'raw'/task.replace('/','__')/'LARGE_ONLY/reasoning/complete.json')
  for c in CONDITIONS:
   h=read(storage/'raw'/task.replace('/','__')/c/'reasoning/complete.json')
   for draw in range(3):
    folder=storage/'raw'/task.replace('/','__')/c/f'answer_{draw}';a=read(folder/'answer.json');g=read(folder/'complete.json')
    score_path=storage/'scores'/task.replace('/','__')/c/f'{draw}.json';s=read(score_path)
    if s['answer_sha256']!=sha(folder/'answer.json'):raise ValueError('Answer/score identity differs')
    sc=s['score'];large=c=='LARGE_ONLY'
    receiver=0 if large else h['active_seconds']
    total=a['source_prefix_seconds']+receiver+g['active_seconds']+a['bridge_seconds']+a['clone_seconds']+a['transfer_seconds']
    record={'task_id':task,'condition':c,'draw':draw,'fraction_percent':a['fraction_percent'],'source_token_index':a['source_index'],'source_trajectory_sha256':a['source_history_sha256'],'receiver_seed':None if large else h.get('seed'),'answer_seed':g['seed'],'answer_sha256':sha(folder/'answer.json'),'score_sha256':sha(score_path),'passed':sc.get('passed'),'missing':sc.get('missing',False),'failure_category':sc.get('category','unspecified'),'source_reasoning_length':len(reasoning_body(source)),'source_prefix_accounted_seconds':a['source_prefix_seconds'],'receiver_prefill_tokens':0 if large else lengths[task]+a['source_index'],'receiver_prefill_seconds':0 if large else h['prefill_seconds'],'receiver_reasoning_tokens':0 if large or c=='FULL_TEXT' else len(reasoning_body(h)),'receiver_reasoning_seconds':0 if large else h['generation_seconds'],'answer_tokens':len(g['generated_ids']),'answer_seconds':g['active_seconds'],'answer_output_characters':len(a['answer_text']),'bridge_seconds':a['bridge_seconds'],'clone_seconds':a['clone_seconds'],'transfer_seconds':a['transfer_seconds'],'single_output_stage_sum_seconds':total,'source_reasoning_capped':source['capped'],'receiver_reasoning_capped':False if large else h['capped'],'answer_capped':g['capped'],'answer_ended_eos':g['ended_eos'],'recovery_seconds':h.get('recovery_seconds',0)+g.get('recovery_seconds',0)+a['cache_reconstruction_setup_seconds'],'scoring_wall_seconds':s['completed_epoch']-s['started_epoch'],'gpu_kernel_busy_seconds':None,'runpod_dollars':0.}
    # Prompt lengths are safe scalar counts; never emit underlying IDs.
    record['total_generated_tokens']=a['source_index']+record['receiver_reasoning_tokens']+record['answer_tokens']
    record['token_count_scope']='Reasoning bodies exclude closing-think; answer count includes terminal EOS. No pretence these bodies count inserted bridges.'
    records.append(record)
 return records

def analyze(records,task_ids):
 if len(records)!=len(task_ids)*len(CONDITIONS)*3:raise ValueError('Unexpected draw population')
 lookup={(r['task_id'],r['condition'],r['draw']):r for r in records}
 if len(lookup)!=len(records):raise ValueError('Duplicate draws')
 idx=np.random.Generator(np.random.PCG64(20260922)).integers(0,len(task_ids),(10000,len(task_ids)))
 rate={};cost={};summary={};tasks=[]
 for c in CONDITIONS:
  quality=[];timing=[]
  for t in task_ids:
   rows=[lookup[t,c,i] for i in range(3)];missing=sum(x['missing'] for x in rows)
   if any(type(x['passed']) not in (bool,type(None)) or x['missing'] is not (x['passed'] is None) for x in rows):raise ValueError('Ambiguous score/missingness')
   wins=sum(x['passed'] is True for x in rows);q=wins/3;seconds=float(np.mean([x['single_output_stage_sum_seconds'] for x in rows]));quality.append(q);timing.append(seconds)
   tasks.append({'task_id':t,'condition':c,'passed':wins,'draws':3,'missing':missing,'pass_rate':None if missing else q,'quality_lower_bound':q,'quality_upper_bound':(wins+missing)/3,'mean_stage_seconds':seconds})
  rate[c]=np.array(quality);cost[c]=np.array(timing);allrows=[r for r in records if r['condition']==c];missing=sum(r['missing'] for r in allrows)
  summary[c]={'passed':sum(r['passed'] is True for r in allrows),'draws':len(allrows),'missing':missing,'pass_rate':None if missing else float(rate[c].mean()),'quality_lower_bound':float(rate[c].mean()),'quality_upper_bound':float(rate[c].mean()+missing/len(allrows)),'ci95':None if missing else np.quantile(rate[c][idx].mean(axis=1),[.025,.975]).tolist(),'mean_stage_seconds':float(cost[c].mean()),'stage_seconds_ci95':np.quantile(cost[c][idx].mean(axis=1),[.025,.975]).tolist(),'failure_categories':dict(collections.Counter(x['failure_category'] for x in allrows)),'receiver_reasoning_tokens_mean':float(np.mean([x['receiver_reasoning_tokens'] for x in allrows])),'source_reasoning_tokens_mean':float(np.mean([x['source_token_index'] for x in allrows]))}
 contrasts={}
 for c in ['H10','H25','H50','H75']:
  for ref in ['SMALL_ONLY','LARGE_ONLY','FULL_TEXT']:
   q=rate[c]-rate[ref];dt=cost[c]-cost[ref];complete=not summary[c]['missing'] and not summary[ref]['missing']
   contrasts[c+'-'+ref]={'difference':float(q.mean()) if complete else None,'ci95':np.quantile(q[idx].mean(axis=1),[.025,.975]).tolist() if complete else None,'paired_task_gains':int((q>0).sum()) if complete else None,'paired_task_losses':int((q<0).sum()) if complete else None,'paired_task_ties':int((q==0).sum()) if complete else None,'mean_seconds_difference':float(dt.mean()),'seconds_difference_ci95':np.quantile(dt[idx].mean(axis=1),[.025,.975]).tolist(),'latency_ratio':float(cost[c].mean()/cost[ref].mean()),'interval_scope':'Exploratory unadjusted task-cluster intervals; missing outcomes are not treated as observed failures.'}
 frontier=[c for c in CONDITIONS if summary[c]['pass_rate'] is not None and not any(summary[o]['pass_rate'] is not None and summary[o]['pass_rate']>=summary[c]['pass_rate'] and summary[o]['mean_stage_seconds']<=summary[c]['mean_stage_seconds'] and (summary[o]['pass_rate']>summary[c]['pass_rate'] or summary[o]['mean_stage_seconds']<summary[c]['mean_stage_seconds']) for o in CONDITIONS if o!=c)]
 oracle=[]
 if not any(s['missing'] for s in summary.values()):
  for j,t in enumerate(task_ids):
   best=min(CONDITIONS[:-1],key=lambda c:(-rate[c][j],cost[c][j],CONDITIONS.index(c)))
   oracle.append({'task_id':t,'selected_fraction_condition':best,'observed_three_draw_mean_success':float(rate[best][j]),'mean_stage_seconds':float(cost[best][j])})
 return {'conditions':summary,'contrasts':contrasts,'observed_quality_latency_frontier':frontier,'oracle':{'task_choices':oracle,'mean_success':float(np.mean([x['observed_three_draw_mean_success'] for x in oracle])) if oracle else None,'mean_stage_seconds':float(np.mean([x['mean_stage_seconds'] for x in oracle])) if oracle else None,'limitation':'Retrospective selection on the same three draws; optimistic and nondeployable. Fraction timing also requires future source length. Not a learned or validated policy.'},'tasks':len(task_ids),'draws':len(records),'missing':sum(x['missing'] for x in records),'bootstrap_seed':20260922,'bootstrap_resamples':10000,'currency':'Measured new RunPod spend $0; local energy/depreciation unmeasured. Quality/latency frontier is hardware-conditional, not an H200 dollar frontier.'},tasks

def csv_write(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)

def plot_summary(summary,path):
 # Missing outcomes are intervals of possible values, never zero or NaN annotations.
 import matplotlib
 matplotlib.use('Agg')
 from matplotlib import pyplot as plt
 order=list(CONDITIONS[:-1]);fractions=[0,10,25,50,75,100]
 fig,axes=plt.subplots(1,3,figsize=(15,4.5))
 for c,fraction in zip(order,fractions):
  v=summary['conditions'][c];lo=v['quality_lower_bound'];hi=v['quality_upper_bound']
  axes[0].plot([fraction,fraction],[lo,hi],color='tab:blue',linewidth=3)
  axes[0].scatter([fraction,fraction],[lo,hi],color='tab:blue',s=18)
 axes[0].set_xticks(fractions)
 axes[0].set(xlabel='Source reasoning fraction (%)',ylabel='Success: missing-outcome bounds',ylim=(.65,1))
 axes[1].plot(fractions,[summary['conditions'][c]['mean_stage_seconds'] for c in order],'o-')
 axes[1].set_xticks(fractions)
 axes[1].set(xlabel='Source reasoning fraction (%)',ylabel='Reconstructed single-output stage sum (s)')
 for c,v in summary['conditions'].items():
  x=v['mean_stage_seconds'];lo=v['quality_lower_bound'];hi=v['quality_upper_bound']
  axes[2].plot([x,x],[lo,hi],linewidth=3);axes[2].scatter([x,x],[lo,hi],s=18)
  offset={'SMALL_ONLY':(-4,9),'H10':(0,-17),'FULL_TEXT':(-44,9),'LARGE_ONLY':(-40,-17)}.get(c,(0,9))
  axes[2].annotate(c,(x,hi),xytext=offset,textcoords='offset points',fontsize=7)
 axes[2].set(xlabel='Reconstructed stage sum (s)',ylabel='Success: missing-outcome bounds',ylim=(.77,.95),xlim=(450,1740))
 fig.suptitle('M4 Max BF16 development screen — bounds are NOT confidence intervals')
 fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--storage',type=Path,default=ROOT/'data/early_handoff_01');p.add_argument('--input',type=Path,help='Recompute from safe analysis_input.json without raw data');p.add_argument('--output',type=Path,default=ROOT/'results/early_handoff_01');a=p.parse_args()
 if a.input:
  compact=read(a.input);d=compact['declaration'];records=compact['records']
 else:
  d=read(ROOT/'configs/early_handoff_01/declaration.json');records=collect(a.storage,d)
 summary,tasks=analyze(records,d['population']['task_ids']);a.output.mkdir(parents=True,exist_ok=True)
 write(a.output/'per_draw.json',records);csv_write(a.output/'per_draw.csv',records);csv_write(a.output/'per_task.csv',tasks);write(a.output/'summary.json',summary)
 write(a.output/'analysis_input.json',{'declaration':d,'records':records,'scope':'Whitelisted scalar outcomes/costs and identity hashes only; no raw programs, prompts, tokens or tests.'})
 plot_summary(summary,a.output/'frontier.png')
 try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
 except subprocess.CalledProcessError:commit='See MANIFEST.json in the review ZIP (non-Git compact regeneration).'
 lines=['# Early handoff development screen','','## Scope and exact protocol','',f"40 previously inspected development tasks, three fixed final-answer draws per task-condition; {len(records)} planned/recorded draws. One source trajectory and one receiver continuation per fraction-task. Repeated draws are not independent tasks.",'','Local Apple M4 Max, 128 GB; PyTorch 2.14.0 MPS/BF16/SDPA, unquantized Qwen3-32B → Qwen3-8B. Local ARM Linux/Colima uses the unchanged scorer and fixed limits, not the historical x86 machine.','',f"Source revision: `{d['models']['source']['revision']}`. Receiver: `{d['models']['receiver']['revision']}`.",'','Exact ordered population: `configs/early_handoff_01/declaration.json` and `results/early_handoff_01/per_task.csv`. Pinned LiveCodeBench revision and all input identities are in the declaration and preparation receipt. Raw benchmark material is fetched upstream, not redistributed.','','Fractions are 0/10/25/50/75/100% of actual source reasoning-token count (closing-think excluded); floor to the exact token index, never a semantic boundary. Intermediate cuts continue reasoning inside the same original assistant turn. No future source reasoning, source answer, mapper, cache transfer or router is provided. At 100% the receiver answers after complete native text replay. Large-only is separate.','','Fractions depend on eventual source length and are retrospective, not an online stopping policy. Intermediate source time is cumulative measured prefix time from the saved full source execution. It is an accounting estimate, not a separately timed online system. All full source generation is charged to actual study work.','','## Fixed-fraction quality and accounting','','| Condition | Passes / draws | Mean success | Mean stage seconds | Receiver reasoning tokens |','|---|---:|---:|---:|---:|']
 for c,v in summary['conditions'].items():lines.append(f"| {c} | {v['passed']}/{v['draws']} | {v['pass_rate']} | {v['mean_stage_seconds']:.3f} | {v['receiver_reasoning_tokens_mean']:.1f} |")
 lines+=['','![Quality and latency frontier](results/early_handoff_01/frontier.png)','','## Paired task comparisons','','Every intermediate fraction is compared with small-only, large-only and full-text handoff in `summary.json`: mean quality difference, paired 95% task-cluster intervals, gains/losses/ties and latency differences. All 10,000 resamples use the frozen seed 20260922. Intervals are exploratory and unadjusted; overlap with zero does not establish equivalence.','',f"Observed nondominated conditions: {', '.join(summary['observed_quality_latency_frontier'])}. This is a noisy empirical quality/latency frontier, not a demonstrated serving or monetary frontier.",'','## Oracle, failures and limitations','',summary['oracle']['limitation'],f"Retrospective oracle observed mean success: {summary['oracle']['mean_success']}; mean stage seconds: {summary['oracle']['mean_stage_seconds']}.",'','Per-task and per-draw files include all failures, caps, missingness, source indices, receiver compensation, output lengths, seeds, timings and hashes. Missing infrastructure outcomes remain null; they are never silently turned into failures or dropped. No hidden-test content or diagnostics is exported. Source/receiver weights are frozen. Reused development exposure and hardware/runtime changes limit generalization.','',f"Missing outcomes: {summary['missing']}. Infrastructure recovery and numerical preflight receipts are separately retained in ignored storage; any incidents must be summarized before final review.",'','## Costs','',summary['currency'],'Synchronized stage wall times are not measured GPU-kernel busy time; that field is unavailable. Model loading, dependency downloads, recovery and scoring are study overhead, not silently free work. A per-output counterfactual charges the full required reasoning to each final draw rather than amortizing it across the three experimental answers.','','## Reproduction','','See `REPRODUCE_EARLY_HANDOFF.md` for exact preparation, local feasibility, generation, isolated scoring and analysis commands. Safe compact analysis can be recomputed from `analysis_input.json`; raw generation requires pinned upstream dependencies and documented local hardware.','',f'Implementation Git commit used for this report: `{commit}`. The final commit including the report is recorded in the review bundle manifest after committing the report.','','## Decision','','Review the fixed-condition contrasts and hardware-conditional frontier before proposing any separately frozen confirmation. No router, state translation, sparse mechanism, confirmation or benchmark extension has been launched.']
 (a.output/'analysis_draft.md').write_text('\n'.join(lines)+'\n')
 print('Analysis written; final human-readable interpretation/incident audit and safe commit/bundle remain required.')
if __name__=='__main__':main()
