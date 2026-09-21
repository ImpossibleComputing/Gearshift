#!/usr/bin/env python3
"""One bounded capacity-only completion after the original stage cleans up."""
import argparse,json,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,bind
from coding_recovery_pipeline import import_diagnostic,run_stage,status
from coding_pipeline_plan import import_run
from coding_recovery_plan import build
from coding_recovery_report import report
from coding_recovery_bundle import build as bundle
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/coding_pilot_v1/recovery_20260917'

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--run-id',required=True);p.add_argument('--training-root',required=True);a=p.parse_args()
    own=E/(a.run_id+'_followon_status.json')
    if own.exists():raise ValueError('One-shot follow-on already exists')
    def update(**kw):write(own,{'epoch':time.time(),**kw})
    update(state='waiting',parent=a.parent,run_id=a.run_id)
    try:
        deadline=time.monotonic()+3*3600
        while True:
            prior=json.loads((E/'pipeline_status.json').read_text())
            done=ROOT/'evidence/coding_pilot_v1/control/parallel'/a.parent/'complete.json'
            if done.exists() and prior['state']!='running':break
            if time.monotonic()>deadline:raise TimeoutError('Original stage has not completed; no duplicate dispatch')
            time.sleep(15)
        bind(E/(a.run_id+'_parent_pipeline_status.json'),prior)
        import_diagnostic(a.parent)
        plan=build('dev:'+a.training_root,a.run_id,recovery_parent=a.parent)
        update(state='running',parent=a.parent,run_id=a.run_id)
        status(state='running',stage='capacity_missing_development_shard',run_id=a.run_id,parent_run_id=a.parent,training_root=a.training_root)
        rc=run_stage(plan)
        if rc:import_diagnostic(a.run_id)
        else:import_run(a.run_id)
        summary=report([a.parent,a.run_id],a.training_root);bundle()
        state='complete' if summary['coverage']['committed_complete']==40 else 'needs_inspection'
        update(state=state,coverage=summary['coverage'])
        status(state=state,stage='review_exported',run_ids=[a.parent,a.run_id],coverage=summary['coverage'],training_root=a.training_root)
    except BaseException as exc:
        update(state='needs_inspection',error=str(exc),traceback=traceback.format_exc())
        raise
if __name__=='__main__':main()
