#!/usr/bin/env python3
"""Read-only raw-evidence audit and explicitly post-run missingness sensitivity.

No generation/scoring imports, retries, raw-text exports or policy changes.
Compact mode needs only the public analysis input and retained safe audit receipt.
"""
import argparse, collections, math, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gearshift.early_handoff import (CONDITIONS,FRACTIONS,CLOSING,read,write,sha,digest,
 reasoning_body,handoff,seed,continuation_cap,source_prefix_seconds)
from scripts.early_handoff_report import csv_write


def sensitivity(records, task_ids):
 lookup={(r['task_id'],r['condition'],r['draw']):r for r in records}
 assert len(lookup)==len(task_ids)*21
 bounds={};cost={};tasks=[]
 for c in CONDITIONS:
  vals=[];times=[]
  for task in task_ids:
   rows=[lookup[task,c,i] for i in range(3)]
   lo=sum(r['passed'] is True for r in rows)/3
   hi=lo+sum(r['missing'] for r in rows)/3
   vals.append((lo,hi));times.append(float(np.mean([r['single_output_stage_sum_seconds'] for r in rows])))
   tasks.append({'task_id':task,'condition':c,'quality_lower_bound':lo,'quality_upper_bound':hi})
  bounds[c]=np.array(vals);cost[c]=np.array(times)
 idx=np.random.Generator(np.random.PCG64(20260922)).integers(0,len(task_ids),(10000,len(task_ids)))
 contrasts={}
 for c in ['H10','H25','H50','H75']:
  for ref in ['SMALL_ONLY','LARGE_ONLY','FULL_TEXT']:
   lo=bounds[c][:,0]-bounds[ref][:,1];hi=bounds[c][:,1]-bounds[ref][:,0]
   fixed=np.isclose(lo,hi)
   contrasts[c+'-'+ref]={
    'difference_identification_bounds':[float(lo.mean()),float(hi.mean())],
    'task_bootstrap_sensitivity_envelope':[float(np.quantile(lo[idx].mean(axis=1),.025)),float(np.quantile(hi[idx].mean(axis=1),.975))],
    'definite_task_gains':int((lo>1e-12).sum()),'definite_task_losses':int((hi< -1e-12).sum()),
    'definite_task_ties':int((fixed & np.isclose(lo,0)).sum()),
    'ambiguous_tasks':int((~((lo>1e-12)|(hi< -1e-12)|(fixed & np.isclose(lo,0)))).sum())}
 # This subset is explicitly supplementary, never silently substituted for 40 tasks.
 oracle=[];complete=[]
 for j,task in enumerate(task_ids):
  if any(not np.isclose(bounds[c][j,0],bounds[c][j,1]) for c in CONDITIONS):continue
  complete.append(task);best=min(CONDITIONS[:-1],key=lambda c:(-bounds[c][j,0],cost[c][j],CONDITIONS.index(c)))
  oracle.append({'task_id':task,'condition':best,'quality':float(bounds[best][j,0]),'mean_stage_seconds':float(cost[best][j])})
 olo=np.max(np.stack([bounds[c][:,0] for c in CONDITIONS[:-1]]),axis=0)
 ohi=np.max(np.stack([bounds[c][:,1] for c in CONDITIONS[:-1]]),axis=0)
 means={c:bounds[c].mean(axis=0) for c in CONDITIONS}
 robust=[];possible=[]
 for c in CONDITIONS:
  dominated=any(cost[o].mean()<=cost[c].mean() and means[o][0]>=means[c][1] and (cost[o].mean()<cost[c].mean() or means[o][0]>means[c][1]) for o in CONDITIONS if o!=c)
  if dominated:robust.append(c)
  else:possible.append(c)
 return {'scope':'Post-run descriptive missingness sensitivity, not a replacement primary analysis. All 40 tasks retained in bounds. Bootstrap envelopes combine adverse/best missing-outcome completions with task resampling, are exploratory and unadjusted, and do not establish equivalence.',
  'contrasts':contrasts,'robustly_dominated_under_missingness_bounds':robust,'potential_frontier_not_resolved_by_missingness':possible,
  'frontier_limitation':'Empirical means only; sampling uncertainty is not incorporated in dominance. Reconstructed local latency is not deployable or monetary cost.',
  'oracle':{'scope':'Post-run supplementary oracle; never deployable. Quality-first, latency tie-break; LARGE_ONLY excluded from fraction choices. Same three draws used for selection and evaluation.',
   'all_task_quality_identification_bounds':[float(olo.mean()),float(ohi.mean())],
   'fully_scored_task_count':len(complete),'excluded_from_subset':[t for t in task_ids if t not in complete],
   'fully_scored_subset_quality':float(np.mean([r['quality'] for r in oracle])) if oracle else None,
   'fully_scored_subset_mean_seconds':float(np.mean([r['mean_stage_seconds'] for r in oracle])) if oracle else None,
   'task_choices_on_fully_scored_subset':oracle},'per_task_quality_bounds':tasks}


def audit_raw(storage,d,records):
 seal=read(storage/'control/generation_seal.json');prepared=read(storage/'control/prepared_inputs.json')
 assert seal['declaration_sha256']==prepared['declaration_sha256']==sha(ROOT/'configs/early_handoff_01/declaration.json')
 assert seal['all_generation_complete'] and seal['expected_answers']==840 and len(seal['files'])==1680
 assert len({f['path'] for f in seal['files']})==1680
 for f in seal['files']:
  p=storage/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes']
 assert sha(storage/'inputs/visible.json')==prepared['visible_sha256']
 assert sha(storage/'inputs/private_tests.json')==prepared['private_tests_sha256']
 visible=read(storage/'inputs/visible.json');completion=read(storage/'control/scoring_complete.json')
 for path,h in completion['scorer_identity']['files'].items():assert sha(ROOT/path)==h
 assert completion['policy_file_sha256']==sha(ROOT/d['scoring']['policy_path'])
 expected=set();reason_seconds=collections.Counter();answer_seconds=collections.Counter();tokens=collections.Counter();extras=collections.Counter();caps=collections.Counter();ends=collections.Counter();categories=collections.Counter();missing=[];retry=0;failed_attempts=0;recoveries=0;task_table=[];failed_ids=set();complete_counts=collections.Counter()
 lookup={(r['task_id'],r['condition'],r['draw']):r for r in records}
 for task in d['population']['task_ids']:
  base=storage/'raw'/task.replace('/','__');source=read(base/'LARGE_ONLY/reasoning/complete.json');prompt=visible[task]['prompt_ids']
  assert digest(prompt)==source['input_ids_sha256'];assert digest(source['generated_ids'])==source['generated_ids_sha256']
  task_row={'task_id':task}
  for c in CONDITIONS:
   large=c=='LARGE_ONLY';h=read(base/c/'reasoning/complete.json');fraction=None if large else FRACTIONS[CONDITIONS.index(c)]
   view=None if large else handoff(prompt,source,fraction)
   inp=prompt if large else (view['input_ids'][:-1] if fraction==100 else view['input_ids'])
   assert h['input_ids_sha256']==digest(inp)
   role='source' if large else 'receiver';stream='source_reasoning' if large else 'receiver_reasoning'
   if c!='FULL_TEXT':
    assert h['contract']=={'input_ids_sha256':digest(inp),'cap':24576 if large else continuation_cap(view['source_index']),'seed':seed(task,0,stream),'kind':'reason','model_role':role}
    assert h['seed']==seed(task,0,stream) and h['generated_ids_sha256']==digest(h['generated_ids'])
    recoveries+=len(h.get('recovery_attempts',[]));tokens[role+'_reasoning_emissions']+=len(h['generated_ids'])
   reason_seconds[c]+=h['active_seconds'];caps[c+'_reasoning']+=int(h['capped']);ends[c+'_reasoning_early_eos']+=int(h['ended_eos'])
   complete_counts['reasoning_or_prefill_records']+=1
   answer_prefix=inp+reasoning_body(h)+[CLOSING]
   for draw in range(3):
    folder=base/c/f'answer_{draw}';a=read(folder/'answer.json');g=read(folder/'complete.json');row=lookup[task,c,draw]
    sp=storage/'scores'/task.replace('/','__')/c/f'{draw}.json';s=read(sp);sc=s['score'];expected.add(sp)
    assert s['task_id']==task and s['condition']==c and s['draw']==draw
    assert sha(folder/'answer.json')==s['answer_sha256']==row['answer_sha256'] and sha(sp)==row['score_sha256']
    assert a['answer_record_sha256']==sha(folder/'complete.json') and a['history_sha256']==sha(base/c/'reasoning/complete.json')
    assert a['source_history_sha256']==sha(base/'LARGE_ONLY/reasoning/complete.json')
    assert a['source_index']==(len(reasoning_body(source)) if large else view['source_index'])
    assert a['source_prefix_seconds']==(source['active_seconds'] if large else source_prefix_seconds(source,view['source_index'],fraction))
    assert g['contract']=={'input_ids_sha256':digest(answer_prefix),'cap':4096,'seed':seed(task,draw,'source_answer' if large else 'receiver_answer'),'kind':'answer','model_role':role}
    assert digest(g['generated_ids'])==g['generated_ids_sha256']
    assert row['receiver_seed']==(None if large else h.get('seed'))
    assert row['passed']==sc['passed'] and row['missing']==sc['missing'] and sc['scorer_identity']==completion['scorer_identity']
    assert sc['missing'] is (sc['passed'] is None)
    assert s['started_epoch']>=seal['epoch']
    tokens[role+'_answer_emissions']+=len(g['generated_ids']);answer_seconds[c]+=g['active_seconds']
    for k in ['clone_seconds','bridge_seconds','cache_reconstruction_setup_seconds']:extras[k]+=a[k]
    if draw==0:extras['unique_transfer_seconds']+=a['transfer_seconds']
    recoveries+=len(g.get('recovery_attempts',[]));caps[c+'_answer']+=int(g['capped']);categories[sc['category']]+=1
    complete_counts['answer_records']+=1
    if sc['passed'] is False:failed_ids.add(task)
    for receipt in sc.get('receipts',[]):
     attempts=receipt.get('attempts',[]);retry+=max(0,len(attempts)-1)
     failed_attempts+=sum(bool(at.get('missing')) for at in attempts)
    if sc['missing']:
     terminal=sc['receipts'][-1]
     missing.append({'task_id':task,'condition':c,'draw':draw,'attempt_count':len(terminal['attempts']),
      'return_codes':[at.get('returncode') for at in terminal['attempts']],
      'reasons':[at.get('infrastructure_reason') for at in terminal['attempts']],
      'maximum_rss_kib_max':max(at['maximum_rss_kib'] for at in terminal['attempts'])})
   rows=[lookup[task,c,i] for i in range(3)];task_row[c+'_passes']=sum(r['passed'] is True for r in rows);task_row[c+'_missing']=sum(r['missing'] for r in rows)
  task_table.append(task_row)
 assert expected==set((storage/'scores').rglob('*.json'))
 assert completion['answers']==840 and completion['missing']==len(missing) and completion['passed']==categories['pass']
 worker=read(storage/'control/worker_complete.json');launch=read(storage/'control/launch.json');status=read(storage/'control/pipeline_status.json')
 load={r:read(storage/'control'/f'{r}_loaded.json')['model_load_seconds'] for r in ['source','receiver']}
 elapsed=worker['completed_epoch']-worker['started_epoch'];unique=sum(reason_seconds.values())+sum(answer_seconds.values())+sum(extras.values())
 scoring_start=min(read(p)['started_epoch'] for p in expected);scoring_end=max(read(p)['completed_epoch'] for p in expected)
 kernel=storage/'control/scoring_kernel_audit.log';log=kernel.read_text() if kernel.exists() else ''
 receipt_files=['generation_seal.json','worker_complete.json','scoring_complete.json','prepared_inputs.json','launch.json','pipeline_status.json','source_mps_preflight.json','receiver_mps_preflight.json','local_scorer_preflight.json','model_weights_verified.json','pipeline_analysis.log','scoring_kernel_audit.log']
 receipt={'schema':'gearshift.early_handoff.audit.v1','generation_implementation_commit':launch['implementation_commit'],
  'verified':dict(complete_counts),'scorer_unchanged':True,'all_sealed_hashes_sizes_verified':True,'exact_task_condition_draw_inventory_verified':True,
  'prompt_prefix_seed_contracts_verified':True,'scoring_began_after_seal':True,'generation_recovery_attempts':recoveries,
  'scoring':{'passed':categories['pass'],'missing':len(missing),'observed_failures':840-categories['pass']-len(missing),'categories':dict(categories),'tasks_with_observed_failures':sorted(failed_ids),'infrastructure_retry_count':retry,'missing_execution_attempts':failed_attempts,'final_missing_records':missing,
    'vm_memory_gib':6,'parallel_scoring_workers':2,'per_candidate_address_space_gib':4,
    'kernel_oom_kill_events':log.count('Out of memory: Killed process'),
    'incident':'Guest kernel records OOM kills of scoring Python children. 6 GiB VM with two concurrent candidates and 4 GiB per-candidate limits was underprovisioned. Frozen scorer retains 13 SIGKILL outcomes as missing; no extra retry or reclassification.'},
  'caps':dict(caps),'early_reasoning_eos':dict(ends),
  'actual_study_work':{'generation_wall_seconds':elapsed,'generation_started_epoch':worker['started_epoch'],'generation_completed_epoch':worker['completed_epoch'],
    'unique_reasoning_or_prefill_stage_seconds':dict(reason_seconds),'unique_answer_stage_seconds':dict(answer_seconds),
    'unique_setup_seconds':dict(extras),'model_load_seconds':load,'unique_measured_stage_and_setup_seconds':unique,
    'generation_unassigned_residual_seconds':elapsed-unique-sum(load.values()),
    'actual_unique_generated_emissions':dict(tokens),'scoring_elapsed_seconds':scoring_end-scoring_start,
    'scoring_sum_of_draw_wall_seconds':sum(r['scoring_wall_seconds'] for r in records),
    'initial_pipeline_until_report_failure_seconds':status['stopped_epoch']-status['started_epoch'],
    'cost_scope':'Actual unique work is not the sum of reconstructed per-output hypothetical costs. Residual includes Python, serialization, checkpoint IO, cleanup and scheduling; attribution not separately instrumented. Downloads, preflights and review time are outside generation wall. No kernel-busy measurement or energy/depreciation estimate.',
    'new_runpod_spend_usd':0,'local_energy_depreciation_usd':None,'gpu_kernel_busy_seconds':None},
  'report_incident':{'initial_analysis_failed':True,'reason':'Plot annotated a null pass rate after missing scoring outcomes. Fixed reporting only; no generation or scoring rerun.',
    'metadata_correction':'Initial report exporter mislabeled answer seed as receiver seed. Receiver seed now records the reasoning RNG seed, null for FULL_TEXT. Saved generation seeds unchanged.'},
  'private_receipt_sha256':{f:sha(storage/'control'/f) for f in receipt_files if (storage/'control'/f).exists()},
  'safe_preflights':{r:read(storage/'control'/f'{r}_mps_preflight.json') for r in ['source','receiver']}}
 return receipt,task_table


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'results/early_handoff_01/analysis_input.json');p.add_argument('--storage',type=Path,default=ROOT/'data/early_handoff_01');p.add_argument('--output',type=Path,default=ROOT/'results/early_handoff_01');p.add_argument('--compact',action='store_true');a=p.parse_args()
 compact=read(a.input);a.output.mkdir(parents=True,exist_ok=True)
 write(a.output/'missingness_sensitivity.json',sensitivity(compact['records'],compact['declaration']['population']['task_ids']))
 if not a.compact:
  receipt,tasks=audit_raw(a.storage,compact['declaration'],compact['records']);write(a.output/'audit.json',receipt);csv_write(a.output/'task_outcomes.csv',tasks)
 print('Safe audit/sensitivity complete. No inference or scoring performed.')
if __name__=='__main__':main()
