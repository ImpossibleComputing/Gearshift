#!/usr/bin/env python3
"""Studio-owned bounded follow-on after a separately reviewed numerical gate."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha,digest
from gearshift.coding_snapshot import verify_archive,GLOBAL_RECEIPTS
from coding_pipeline_plan import import_run
from coding_recovery_pipeline import wait_controller_idle
from coding_post_plan import build
from coding_post_report import report
from coding_post_bundle import build as bundle
from plot_post_progress01 import render
from coding_cloud_guard import tick
from coding_parallel_session import own_resources
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/coding_pilot_v1/post_progress01'

def import_stage(run_id):
    import tempfile
    control=ROOT/'evidence/coding_pilot_v1/control/parallel'/run_id
    done=json.loads((control/'complete.json').read_text())
    if done.get('passed'):plan,roots=import_run(run_id)
    else:
        import coding_recovery_pipeline as old
        old.E=E;old.import_diagnostic(run_id);plan=json.loads((control/'plan.json').read_text());roots=[]
    for worker in plan['workers']:
        wid=worker['worker_id'];proof_path=control/wid/'backup_verified.json'
        if not proof_path.exists():continue
        proof=json.loads(proof_path.read_text());archive=ROOT/'evidence/coding_pilot_v1/parallel_backups'/run_id/wid/'final.tar.gz'
        if not proof['verified'] or sha(archive)!=proof['sha256']:raise ValueError('Runtime archive unverified')
        with tempfile.TemporaryDirectory(prefix='post-runtime-') as tmp:
            manifest=verify_archive(archive,tmp)
            if manifest['scope']['stage_identity']!=digest(plan):raise ValueError('Runtime identity differs')
            target=E/'runtime_receipts'/run_id/wid;target.mkdir(parents=True,exist_ok=True);files={}
            for name in GLOBAL_RECEIPTS:
                rel='evidence/coding_pilot_v1/'+name
                if rel not in manifest['files'] or not name.endswith('.json'):continue
                p=Path(tmp)/rel;b=p.read_bytes();dst=target/name
                if dst.exists() and dst.read_bytes()!=b:raise ValueError('Runtime receipt changed')
                dst.write_bytes(b);files[name]={'sha256':sha(dst),'bytes':len(b)}
            write(target/'SOURCE.json',{'run_id':run_id,'worker_id':wid,'stage_identity':digest(plan),'archive_sha256':sha(archive),'files':files})
    return done,roots

def export():
    tick();write(E/'live_resources.json',{'epoch':time.time(),'resources':own_resources()});report();render();return bundle()

def main():
    p=argparse.ArgumentParser();p.add_argument('--numerical-root',required=True);p.add_argument('--hybrid-run',required=True);p.add_argument('--memorization-run',required=True);a=p.parse_args()
    status_path=E/'pipeline_status.json'
    if status_path.exists():raise ValueError('Pipeline identity already exists; inspect without duplicate dispatch')
    def status(**v):write(status_path,{'epoch':time.time(),**v})
    try:
        hybrid_root=None
        for mode,run_id in [('hybrid',a.hybrid_run),('memorization',a.memorization_run)]:
            wait_controller_idle(ROOT/'evidence/coding_pilot_v1/control/recovery_controller.lock')
            tick();plan=build(mode,run_id,a.numerical_root,hybrid_root)
            status(state='running',stage=mode,run_id=run_id,numerical_root=a.numerical_root,hybrid_root=hybrid_root)
            subprocess.run([sys.executable,'scripts/coding_parallel_session.py','--plan',str(plan),'--validate-only'],cwd=ROOT,check=True)
            with (E/(run_id+'_controller.txt')).open('ab') as f:child=subprocess.run([sys.executable,'scripts/coding_parallel_session.py','--plan',str(plan)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            wait_controller_idle(ROOT/'evidence/coding_pilot_v1/control/recovery_controller.lock');done,roots=import_stage(run_id);receipt=export()
            if child.returncode or not done['passed']:raise RuntimeError('Bounded '+mode+' failed; raw failure and partial bundle preserved')
            if mode=='hybrid':hybrid_root=roots[0]
        status(state='complete',stage='review_exported',receipt=receipt,numerical_root=a.numerical_root,hybrid_root=hybrid_root)
    except BaseException as exc:
        status(state='needs_inspection',stage='bounded_stop',error=str(exc),traceback=traceback.format_exc())
        try:export()
        except Exception as export_error:write(E/'export_failure.json',{'error':str(export_error)})
        raise

if __name__=='__main__':main()
