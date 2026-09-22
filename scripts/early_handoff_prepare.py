#!/usr/bin/env python3
"""Prepare the frozen 40-task development population from verified official upstream files."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gearshift.early_handoff import sha,digest,write,read
from scripts.fetch_livecodebench import verify_rows
from scripts.coding_prepare import prompt,decode_tests

def main():
 from transformers import AutoTokenizer
 p=argparse.ArgumentParser();p.add_argument('--storage',type=Path,default=ROOT/'data/early_handoff_01');a=p.parse_args()
 d=read(ROOT/'configs/early_handoff_01/declaration.json');m=read(ROOT/'configs/coding_pilot_v1/draft_membership.json')
 tok=AutoTokenizer.from_pretrained(a.storage/'models/receiver',local_files_only=True)
 wanted=set(d['population']['task_ids']);visible={};private={};source_checks=[]
 for name,meta in m['source_files'].items():
  path=ROOT/'data/coding_pilot_v1/raw'/name
  if sha(path)!=meta['sha256'] or path.stat().st_size!=meta['bytes']:raise ValueError('Upstream file mismatch')
  source_checks.append({'file':name,'sha256':meta['sha256'],**verify_rows(path,name,m)})
  with path.open('rb') as f:
   for line in f:
    row=json.loads(line);tid=row['platform']+'/'+str(row['question_id'])
    if tid not in wanted:continue
    if tid in visible:raise ValueError('Duplicate task')
    text=prompt(row);ids=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=True)
    if len(ids)>d['budgets']['prompt_max_tokens']:raise ValueError('Prompt exceeds frozen bound; do not crop or omit')
    visible[tid]={'task_id':tid,'prompt':text,'prompt_ids':ids,'prompt_ids_sha256':digest(ids)}
    tests=json.loads(row['public_test_cases'])+decode_tests(row['private_test_cases'])
    private[tid]={'fn_name':json.loads(row['metadata']).get('func_name'),'tests':tests}
 if set(visible)!=wanted or set(private)!=wanted:raise ValueError('Incomplete population')
 write(a.storage/'inputs/visible.json',visible,immutable=True)
 write(a.storage/'inputs/private_tests.json',private,immutable=True)
 write(a.storage/'control/prepared_inputs.json',{'declaration_sha256':sha(ROOT/'configs/early_handoff_01/declaration.json'),'task_ids':d['population']['task_ids'],'visible_sha256':sha(a.storage/'inputs/visible.json'),'private_tests_sha256':sha(a.storage/'inputs/private_tests.json'),'source_checks':source_checks,'prompt_token_lengths':{k:len(v['prompt_ids']) for k,v in visible.items()},'model_execution_performed':False,'private_tests_used_for_generation':False},immutable=True)
 print('Verified and prepared all 40 development tasks; raw material remains ignored.')
if __name__=='__main__':main()
