from pathlib import Path
from collections import Counter
from decimal import Decimal, InvalidOperation
import ast, hashlib, json, math, re, statistics, platform
import numpy as np

ROOT=Path('/mnt/data/gearshift_review/gearshift')
OUT=Path('/mnt/data/gearshift_audit')
# Execute the original pure parsing functions; no inference/library stubs are involved here.
ns={'re':re,'Decimal':Decimal,'InvalidOperation':InvalidOperation}
tree=ast.parse((ROOT/'gearshift/reasoning.py').read_text())
subset=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('numbers','numeric_answer')],type_ignores=[])
exec(compile(subset,str(ROOT/'gearshift/reasoning.py'),'exec'),ns)
parse=ns['numeric_answer']
STRICT=r'\s*(?:\\boxed\{)?[-+]?\$?\d[\d,]*(?:\.\d+)?\}?\.?\s*'
audit={'source_zip_sha256':hashlib.sha256(Path('/mnt/data/gearshift.zip').read_bytes()).hexdigest(),'scope':'Independent arithmetic and record-consistency checks; no model inference rerun.','pairs':{}}
for name in ('qwen3_1.7b_to_0.6b','qwen3_4b_to_0.6b'):
 p=ROOT/'results'/name
 fr=json.loads((p/'functional.json').read_text()); rr=json.loads((p/'reasoning.json').read_text())
 s=json.loads((p/'summary.json').read_text()); mf=json.loads((p/'reasoning_manifest.json').read_text())
 pair={}
 pair['functional_rows']=len(fr); pair['reasoning_rows']=len(rr)
 pair['functional_observations_unique']=len({(r['condition'],r['context_length'],r['example']) for r in fr})==len(fr)
 pair['reasoning_observations_unique']=len({(r['condition'],r['dataset_index']) for r in rr})==len(rr)
 pair['manifest_indices_equal_record_indices']=set(mf['selected_indices'])=={r['dataset_index'] for r in rr}
 pair['numeric_scores_recomputed_correctly']=all((parse(r['answer'])==r['gold'])==r['correct'] for r in rr)
 pair['kv_historical_target_prefill_zero']=all(r['target_prefill_tokens']==0 for r in rr if 'kv' in r['condition'])
 pair['shared_source_trajectory_hashes_match']=all(len({r['shared_source_trajectory_sha256'] for r in rr if r['dataset_index']==i and r['condition']!='A_small_only'})==1 for i in mf['selected_indices'])
 pair['functional']={}
 for cond in sorted({r['condition'] for r in fr}):
  g=[r for r in fr if r['condition']==cond]
  # Equal continuation length in every record: mean NLL across cases is token-weighted.
  nll=sum(sum(r['token_nll']) for r in g)/sum(len(r['token_nll']) for r in g)
  st=next(x for x in s['functional_aggregate'] if x['condition']==cond)
  actual={'first_prediction_kl':statistics.mean(r['kl'] for r in g),'first_prediction_top1':statistics.mean(r['top1_agreement'] for r in g),'ppl':math.exp(nll),'all_64_teacher_forced_top1':statistics.mean(x for r in g for x in r['teacher_forced_top1'])}
  actual['summary_matches']=all(math.isclose(actual[k],st[v],rel_tol=1e-6,abs_tol=1e-8) for k,v in [('first_prediction_kl','kl'),('first_prediction_top1','top1_agreement'),('ppl','perplexity')])
  pair['functional'][cond]=actual
 pair['reasoning']={}
 for cond in sorted({r['condition'] for r in rr}):
  g=[r for r in rr if r['condition']==cond]
  completed=[r for r in g if r['reasoning_completed']];truncated=[r for r in g if not r['reasoning_completed']]
  first_correct=[r for r in g if parse(r['answer'].strip().splitlines()[0] if r['answer'].strip() else '')==r['gold']]
  pair['reasoning'][cond]={'n':len(g),'correct':sum(r['correct'] for r in g),'strict_correct':sum(bool(re.fullmatch(STRICT,r['answer'])) and r['correct'] for r in g),'first_line_exploratory_correct':len(first_correct),'completed_n':len(completed),'completed_correct':sum(r['correct'] for r in completed),'truncated_n':len(truncated),'truncated_correct':sum(r['correct'] for r in truncated),'hit_token_cap':sum(r['answer_hit_token_limit'] for r in g),'mean_stage_sum_ms':statistics.mean(sum(r[k] for k in ('source_prefill_ms','source_reasoning_ms','source_answer_ms','target_prefill_ms','mapper_ms','target_delimiter_ms','target_generation_ms')) for r in g)}
 if (p/'functional_training.json').exists():
  h=json.loads((p/'functional_training.json').read_text())['history']
  pair['validation_curve']=[{'step':x['step'],'kl':x['validation_kl']} for x in h if 'validation_kl' in x]
  r2=json.loads((p/'functional_reconstruction_test.json').read_text())
  pair['cache_r2']={cond:{kind:statistics.mean(r['r2'] for r in r2 if r['condition']==cond and r['kind']==kind) for kind in ('k','v')} for cond in {r['condition'] for r in r2}}
 a=json.loads((p/'stopping_audit.json').read_text())
 pair['stopping_audit_pointwise_correctness_flips']=[{'dataset_index':r['dataset_index'],'condition':r['condition'],'original':r['original_correct'],'replay':r['correct']} for r in a if r['correct']!=r['original_correct']]
 pair['stopping_audit_aggregate_scores_unchanged']=all(sum(r['correct'] for r in a if r['condition']==c)==sum(r['original_correct'] for r in a if r['condition']==c) for c in {r['condition'] for r in a})
 pair['stopping_audit_cap_matches']=all(r['answer_hit_token_limit']==next(x['answer_hit_token_limit'] for x in rr if x['condition']==r['condition'] and x['dataset_index']==r['dataset_index']) for r in a)
 pair['source_trajectories_in_archive']=(ROOT/'data'/name/'reasoning_trajectories.jsonl').exists()
 audit['pairs'][name]=pair
(OUT/'record_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
