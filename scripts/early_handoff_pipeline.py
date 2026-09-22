#!/usr/bin/env python3
"""Single-owner local pipeline. No provider API, paid allocation, retries or Git writes."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gearshift.early_handoff import read,write,sha,validate_ledger

def main():
 p=argparse.ArgumentParser();p.add_argument('--storage',type=Path,default=ROOT/'data/early_handoff_01');a=p.parse_args();a.storage=a.storage.resolve();control=a.storage/'control'
 lock=(control/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 validate_ledger(read(control/'budget_ledger.json'))
 for role in ['source','receiver']:
  if not read(control/f'{role}_mps_preflight.json')['passed']:raise RuntimeError('Both local feasibility gates must pass first')
 if not read(control/'local_scorer_preflight.json')['passed']:raise RuntimeError('Scorer gate missing')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Commit and review implementation before generation')
 if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='research/early-handoff-01':raise RuntimeError('Wrong branch')
 child=None;started=time.time();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 def stop(signum,frame):
  if child is not None and child.poll() is None:child.terminate()
  raise InterruptedError('Pipeline interrupted; partial evidence retained')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 state={'owner_pid':os.getpid(),'started_epoch':started,'implementation_commit':commit,'declaration_sha256':sha(ROOT/'configs/early_handoff_01/declaration.json'),'new_remote_resources':[],'runpod_spend_usd':0}
 docker=['docker','--host',f'unix://{Path.home()}/.colima/default/docker.sock','run','--rm','--network','none','--name','gearshift-early-handoff-score','--mount',f'type=bind,src={ROOT},dst=/repo,readonly','--mount',f'type=bind,src={a.storage},dst=/experiment','gearshift-early-scorer:01','python3','scripts/early_handoff_score.py','run']
 stages=[('generation',[sys.executable,'scripts/early_handoff_worker.py','all','--storage',str(a.storage)]),('scoring',docker),('analysis',[sys.executable,'scripts/early_handoff_report.py','--storage',str(a.storage)])]
 try:
  for stage,cmd in stages:
   state.update(stage=stage,stage_started_epoch=time.time(),status='running');write(control/'pipeline_status.json',state)
   with (control/f'pipeline_{stage}.log').open('ab',buffering=0) as log:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    state['child_pid']=child.pid;write(control/'pipeline_status.json',state);code=child.wait()
   if code:raise RuntimeError(f'{stage} failed with exit code {code}; no automatic experimental retry')
  state.update(status='analysis_complete_review_required',completed_epoch=time.time());write(control/'pipeline_status.json',state)
 except BaseException as exc:
  state.update(status='stopped',incident_type=type(exc).__name__,incident=str(exc),stopped_epoch=time.time());write(control/'pipeline_status.json',state);raise
 finally:
  if child is not None and child.poll() is None:
   child.terminate()
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.kill();child.wait()
if __name__=='__main__':main()
