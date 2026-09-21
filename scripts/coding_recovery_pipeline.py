#!/usr/bin/env python3
"""Studio-owned bounded sequence; memory diagnosis is independent of training.

No confirmation, subjective judging, publication or automatic failed-run retry.
"""
import argparse,fcntl,json,shutil,subprocess,sys,tempfile,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha,digest
from gearshift.coding_snapshot import verify_archive
from coding_pipeline_plan import copy_new,import_run
from coding_recovery_plan import build
from coding_recovery_report import report
from coding_recovery_bundle import build as bundle
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/coding_pilot_v1/recovery_20260917'

def status(**row):write(E/'pipeline_status.json',{'epoch':time.time(),**row})

def import_diagnostic(run_id):
    control=ROOT/'evidence/coding_pilot_v1/control/parallel'/run_id
    if not (control/'plan.json').exists():
        notes=[{'imported':False,'reason':'Controller ended before allocation/dispatch registration'}]
        write(E/(run_id+'_import.json'),{'notes':notes});return notes
    plan=json.loads((control/'plan.json').read_text());notes=[]
    for w in plan['workers']:
        wid=w['worker_id'];proofpath=control/wid/'backup_verified.json'
        if not proofpath.exists():notes.append({'worker':wid,'imported':False,'reason':'no allocated worker backup'});continue
        proof=json.loads(proofpath.read_text());base=ROOT/'evidence/coding_pilot_v1/parallel_backups'/run_id/wid
        if not proof['verified'] or not proof['worker_confirmed_absent'] or sha(base/'final.tar.gz')!=proof['sha256']:raise ValueError('Diagnostic backup not verified')
        if len(set(proof['copies']))<2 or any(sha(ROOT/p)!=proof['sha256'] for p in proof['copies']):raise ValueError('Two backup copies required')
        with tempfile.TemporaryDirectory(prefix='recovery-import-') as tmp:
            manifest=verify_archive(base/'final.tar.gz',tmp)
            if manifest['scope']['stage_identity']!=digest(plan):raise ValueError('Diagnostic snapshot identity differs')
            prefixes=[f'results/coding_pilot_v1/{run_id}/{wid}/',f'evidence/coding_pilot_v1/parallel/{run_id}/{wid}/']
            for rel in manifest['files']:
                if any(rel.startswith(s) for s in prefixes):copy_new(Path(tmp)/rel,ROOT/rel)
            for rel,info in proof.get('mapper_checkpoints',{}).items():
                src=base/'mapper_checkpoints'/rel
                if sha(src)!=info['sha256']:raise ValueError('Checkpoint backup changed')
                copy_new(src,ROOT/rel)
        notes.append({'worker':wid,'imported':True,'snapshot_sha256':proof['sha256']})
    write(E/(run_id+'_import.json'),{'notes':notes});return notes

def wait_controller_idle(lock_path,timeout=120.):
    """A completion receipt precedes the previous controller's final ledger tick.

    Wait for its exclusive lock to be released; never unlink a lock or start
    another resource controller concurrently with cleanup.
    """
    deadline=time.monotonic()+timeout
    with Path(lock_path).open('a') as lock:
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(lock,fcntl.LOCK_UN);return
            except BlockingIOError:
                if time.monotonic()>=deadline:raise TimeoutError('Previous resource controller remains active')
                time.sleep(min(.2,max(0.,deadline-time.monotonic())))

def run_stage(plan):
    wait_controller_idle(ROOT/'evidence/coding_pilot_v1/control/recovery_controller.lock')
    log=E/(plan.stem+'_controller.txt')
    with log.open('ab') as f:
        p=subprocess.run([sys.executable,'scripts/coding_parallel_session.py','--plan',str(plan)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    return p.returncode

def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',required=True);p.add_argument('--train-run',required=True);p.add_argument('--dev-run',required=True);a=p.parse_args()
    if (E/'pipeline_status.json').exists():raise ValueError('Pipeline already exists; inspect rather than silently retry')
    try:
        status(state='running',stage='waiting_for_single_probe',run_id=a.probe)
        done=ROOT/'evidence/coding_pilot_v1/control/parallel'/a.probe/'complete.json'
        while not done.exists():
            status(state='running',stage='waiting_for_single_probe',run_id=a.probe);time.sleep(20)
        import_diagnostic(a.probe)
        probe=json.loads(done.read_text());status(state='running',stage='probe_ended_prepare_training',probe_passed=probe['passed'])
        # A source-generation failure is not evidence against the independent
        # full-prefix training path. Both attempts retain separate identities.
        train_id=a.train_run;train_plan=ROOT/'configs/coding_pilot_v1/recovery_20260917'/(train_id+'.json')
        status(state='running',stage='functional_training',run_id=train_id,probe_passed=probe['passed'])
        rc=run_stage(train_plan)
        if rc:
            import_diagnostic(train_id);bundle();raise RuntimeError('Bounded training worker failed; inspect preserved diagnostics')
        _,roots=import_run(train_id);training_root=roots[0]
        status(state='running',stage='freeze_validation_selection',training_root=training_root)
        dev_id=a.dev_run;plan=build('dev:'+training_root,dev_id)
        status(state='running',stage='development',run_id=dev_id,training_root=training_root)
        rc=run_stage(plan)
        if rc:import_diagnostic(dev_id)
        else:import_run(dev_id)
        summary=report(dev_id,training_root);receipt=bundle()
        status(state='complete' if summary['coverage']['committed_complete']==40 else 'needs_inspection',stage='review_exported',coverage=summary['coverage'],training_root=training_root,probe_passed=probe['passed'])
    except BaseException as exc:
        status(state='needs_inspection',stage='bounded_stop',error=str(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
