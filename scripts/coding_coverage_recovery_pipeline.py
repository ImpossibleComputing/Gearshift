#!/usr/bin/env python3
"""Studio-owned evaluation recovery followed by backup, cleanup and review."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write
from gearshift.coding_coverage import EVIDENCE
from coding_coverage_pipeline import import_stage,commit_evidence,export,git
from coding_recovery_pipeline import wait_controller_idle
ROOT=Path(__file__).resolve().parents[1];E=ROOT/EVIDENCE


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=json.loads((ROOT/a.plan).read_text());run=plan['run_id']
    if plan['stage']!='coverage_evaluation_recovery':raise ValueError('Evaluation recovery plan required')
    result='results/coding_pilot_v1/'+run+'/evaluation_recovery';status_path=E/'recovery_pipeline_status.json'
    if status_path.exists():raise ValueError('Inspect previous recovery; no duplicate controller')
    def status(**kw):write(status_path,{'epoch':time.time(),'pid':os.getpid(),'run_id':run,'result_root':result,**kw})
    try:
        subprocess.run([sys.executable,'scripts/coding_parallel_session.py','--plan',a.plan,'--validate-only'],cwd=ROOT,check=True)
        status(state='running',stage='evaluation_recovery')
        with (E/'recovery_controller.txt').open('ab') as f:
            child=subprocess.run([sys.executable,'scripts/coding_parallel_session.py','--plan',a.plan],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        wait_controller_idle(ROOT/'evidence/coding_pilot_v1/control/recovery_controller.lock')
        if not (ROOT/'evidence/coding_pilot_v1/control/parallel'/run/'complete.json').exists():
            raise RuntimeError('Controller stopped before dispatch completion; see recovery_controller.txt. No scientific result claimed.')
        done,roots=import_stage(run)
        if child.returncode or not done['passed']:raise RuntimeError('Recovery incomplete; inspect rather than repeat')
        commit_evidence('Preserve last-common checkpoint evaluation and matched controls')
        status(state='running',stage='review_export')
        receipt=export(result)
        status(state='complete',stage='review_verified',receipt=receipt,delivery_commit=git('rev-parse','HEAD'))
    except BaseException as exc:
        status(state='needs_inspection',stage='bounded_stop',error=str(exc),traceback=traceback.format_exc())
        try:export(result)
        except Exception as nested:write(E/'recovery_export_failure.json',{'error':str(nested),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
