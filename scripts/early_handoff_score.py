#!/usr/bin/env python3
"""Local Linux-only scoring after a complete generation seal. No model imports."""
import argparse,concurrent.futures,json,os,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gearshift.early_handoff import read,write,sha,CONDITIONS
from gearshift.coding_sandbox import extract
from gearshift.coding_sandbox_v2 import score,scorer_identity
from scripts.coding_scorer_repair_calibrate import setup_sandbox
POLICY='evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/calibration/frozen_policy.json'

def preflight(storage):
 setup_sandbox()
 result=subprocess.run(['/usr/bin/python3','-m','pytest','-q','-ra','-p','no:cacheprovider','tests/test_coding_sandbox_v2.py'],cwd=ROOT,text=True,capture_output=True)
 receipt={'platform':platform.platform(),'python':sys.version,'uid':os.geteuid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'scorer_identity':scorer_identity(read(ROOT/POLICY)),'historical_policy_file_sha256':sha(ROOT/POLICY),'suite_returncode':result.returncode,'suite_stdout':result.stdout,'suite_stderr':result.stderr,'benchmark_tests_loaded':False,'passed':result.returncode==0,'limits_changed':False,'qualification':'Same limits and checker, but ARM Linux in a local VM is not the historical x86 scoring machine.'}
 write(storage/'control/local_scorer_preflight.json',receipt,immutable=True)
 if result.returncode:raise RuntimeError('Linux scorer preflight failed')
 print(json.dumps({'passed':True,'summary':result.stdout.splitlines()[-1]}))

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','run']);p.add_argument('--storage',type=Path,default=Path('/experiment'));p.add_argument('--workers',type=int,default=2);a=p.parse_args()
 if sys.platform!='linux' or os.geteuid()!=0:raise RuntimeError('Dedicated local Linux root container required')
 if a.stage=='preflight':return preflight(a.storage)
 if not 1<=a.workers<=4:raise ValueError('Bounded local VM parallelism is 1..4')
 setup_sandbox();d=read(ROOT/'configs/early_handoff_01/declaration.json');seal=read(a.storage/'control/generation_seal.json');prepared=read(a.storage/'control/prepared_inputs.json')
 if not seal['all_generation_complete'] or seal['expected_answers']!=840 or seal['declaration_sha256']!=sha(ROOT/'configs/early_handoff_01/declaration.json'):raise ValueError('Missing full generation seal')
 for f in seal['files']:
  path=a.storage/f['path']
  if sha(path)!=f['sha256'] or path.stat().st_size!=f['bytes']:raise ValueError('Generation seal differs')
 if sha(a.storage/'inputs/private_tests.json')!=prepared['private_tests_sha256']:raise ValueError('Pinned tests differ')
 tests=read(a.storage/'inputs/private_tests.json');policy=read(ROOT/POLICY);jobs=[];cpus=sorted(os.sched_getaffinity(0))
 for ti,task in enumerate(d['population']['task_ids']):
  for condition in CONDITIONS:
   for draw in range(3):jobs.append((task,condition,draw))
 def job(x):
  task,condition,draw=x;path=a.storage/'scores'/task.replace('/','__')/condition/f'{draw}.json'
  answer_path=a.storage/'raw'/task.replace('/','__')/condition/f'answer_{draw}/answer.json'
  if path.exists():
   r=read(path)
   if r['answer_sha256']!=sha(answer_path):raise ValueError('Existing scored answer differs')
   return r
  answer=read(answer_path);started=time.time();result=score(extract(answer['answer_text']),tests[task],policy=policy)
  r={'task_id':task,'condition':condition,'draw':draw,'answer_sha256':sha(answer_path),'started_epoch':started,'completed_epoch':time.time(),'score':result}
  write(path,r,immutable=True);return r
 # Quality failures are never retried; scorer-internal verified infrastructure has
 # exactly the original single retry. No quality result is read by generation.
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:records=list(pool.map(job,jobs))
 write(a.storage/'control/scoring_complete.json',{'answers':len(records),'missing':sum(r['score'].get('missing',False) for r in records),'passed':sum(r['score'].get('passed') is True for r in records),'scorer_identity':scorer_identity(policy),'policy_file_sha256':sha(ROOT/POLICY),'epoch':time.time()},immutable=True)
 print('Scoring complete:',len(records),'draws')
if __name__=='__main__':main()
