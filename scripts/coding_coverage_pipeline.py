#!/usr/bin/env python3
"""Studio-owned bounded preflight-to-review execution, independent of laptop."""
import argparse,json,os,subprocess,sys,tempfile,time,traceback,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha,digest,bind
from gearshift.coding_coverage import DECLARATION,CONFIG,EVIDENCE,PUBLICATION,choose_endpoint
from coding_cloud_guard import tick
from coding_parallel_session import own_resources,sanitized_pod
from coding_recovery_pipeline import wait_controller_idle
from coding_coverage_plan import build as build_plan
from coding_coverage_report import report,OUT
from coding_coverage_bundle import build as bundle,permitted
ROOT=Path(__file__).resolve().parents[1];E=ROOT/EVIDENCE


def read(p):return json.loads(Path(p).read_text())
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def commit_evidence(message):
    files=[]
    for pattern in [EVIDENCE+'/**/*',CONFIG+'/**/*','results/coding_pilot_v1/coverage_*/**/*',
        'evidence/coding_pilot_v1/parallel/coverage_*/**/*']:
        for p in ROOT.glob(pattern):
            if p.is_file() and not p.is_symlink() and permitted(str(p.relative_to(ROOT))) and p.name not in ['pipeline_status.json','pipeline_process.json']:
                files.append(str(p.relative_to(ROOT)))
    for p in ['COVERAGE_GENERALIZATION_RESULTS.md','REPRODUCE_COVERAGE_GENERALIZATION.md']:
        if (ROOT/p).exists():files.append(p)
    files=sorted(set(files))
    # Only the newly owned compact namespace; never stage the mutable global
    # control tree, private data, checkpoints, environments or unrelated edits.
    for i in range(0,len(files),300):subprocess.run(['git','add','-f','--',*files[i:i+300]],cwd=ROOT,check=True)
    if files and subprocess.run(['git','diff','--cached','--quiet','--',*files],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-q','-m',message,'--',*files],cwd=ROOT,check=True)
    return git('rev-parse','HEAD')


def refresh_receipts():
    tick();cost=read(ROOT/'evidence/coding_pilot_v1/control/watchdog_status.json');write(E/'review_cost_snapshot.json',cost)
    resources=own_resources();resources['pods']=[sanitized_pod(p) for p in resources['pods']]
    write(E/'live_resources.json',{'epoch':time.time(),'resources':resources})
    if git('rev-parse','gearshift-progress-01^{}')!=PUBLICATION:raise ValueError('Publication tag moved')
    # Publication content, historical summaries and the old review archive must
    # remain unchanged throughout the new research branch.
    protected=['publication','POST_PROGRESS_01_RESULTS.md','RESULTS.md','RECOVERY_RESULTS.md']
    parent=read(ROOT/DECLARATION)['research_parent_commit']
    if git('diff','--name-only',parent,'--',*protected):raise ValueError('Protected prior evidence changed')
    write(E/'current_git_verification.json',{'tag_commit':PUBLICATION,'tag_object':git('rev-parse','gearshift-progress-01'),
        'branch':git('branch','--show-current'),'head':git('rev-parse','HEAD'),'capture_phase':'Completed scientific evidence before report/export commit; archive manifest records delivery commit.',
        'subsequent_commits':git('log','--format=%H %s','gearshift-progress-01..HEAD').splitlines(),'publication_files_unchanged_from_research_parent':True,'pushed':False})
    return cost


def import_stage(run_id):
    import coding_post_pipeline as previous
    previous.E=E
    return previous.import_stage(run_id)


def verify_regeneration(receipt,result_root):
    selected=['COVERAGE_GENERALIZATION_RESULTS.md']
    selected += [str(p.relative_to(ROOT)) for p in (ROOT/OUT).glob('*') if p.suffix in ['.json','.csv','.png','.svg']]
    expected={name:sha(ROOT/name) for name in selected}
    with tempfile.TemporaryDirectory(prefix='coverage-review-check-') as tmp:
        with zipfile.ZipFile(receipt['path']) as z:
            for n in z.namelist():
                if n.startswith('/') or '..' in Path(n).parts:raise ValueError('Unsafe review member')
            z.extractall(tmp)
        extracted=Path(tmp)/'gearshift'
        if any(extracted.rglob('*.pt')) or (extracted/'data/coding_pilot_v1/private').exists() or (extracted/'.git').exists():
            raise ValueError('Compact review contains forbidden execution artifacts')
        subprocess.run([sys.executable,'scripts/coding_coverage_report.py','--result-root',result_root],cwd=extracted,check=True)
        changed=[name for name,h in expected.items() if sha(extracted/name)!=h]
        if changed:raise ValueError('Compact regeneration changed '+str(changed))
    write(E/'compact_regeneration_verification.json',{'passed':True,'source_archive_sha256':receipt['sha256'],
        'result_root':result_root,'verified_outputs':expected,'model_weights_present':False,'private_tests_present':False,'git_present':False,
        'method':'Extract compact archive, run records-only report, compare every generated table/figure/report SHA-256.'})


def export(result_root):
    refresh_receipts();report(result_root)
    receipt=bundle();verify_regeneration(receipt,result_root)
    commit_evidence('Preserve controlled coverage results, uncertainty, and verified review evidence')
    # Repack after the evidence/report commit, recording the actual delivery
    # revision in the manifest without recursive self-referential tracked files.
    receipt=bundle()
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight-run',required=True);p.add_argument('--experiment-run',required=True);a=p.parse_args()
    status_path=E/'pipeline_status.json'
    if status_path.exists():raise ValueError('Pipeline already exists; inspect rather than duplicate')
    def status(**kw):write(status_path,{'epoch':time.time(),'pid':os.getpid(),**kw})
    result_root='results/coding_pilot_v1/'+a.preflight_run+'/preflight'
    try:
        status(state='running',stage='waiting_for_existing_preflight',run_id=a.preflight_run)
        control=ROOT/'evidence/coding_pilot_v1/control/parallel'/a.preflight_run
        deadline=read(ROOT/'evidence/coding_pilot_v1/parallel'/a.preflight_run/'preflight'/'allocation.json')['deadline_epoch']+1800
        while not (control/'complete.json').exists():
            if time.time()>deadline:raise TimeoutError('Preflight controller completion overdue; inspect resource state')
            time.sleep(15)
        wait_controller_idle(ROOT/'evidence/coding_pilot_v1/control/recovery_controller.lock')
        done,roots=import_stage(a.preflight_run)
        if not done['passed']:raise RuntimeError('Preflight failed; no automatic identical retry')
        result_root=roots[0];preflight_path=result_root+'/preflight_result.json';preflight=read(ROOT/preflight_path)
        choice=choose_endpoint(preflight)
        choice.update(preflight_result_path=preflight_path,preflight_result_file_sha256=sha(ROOT/preflight_path),preflight_run=a.preflight_run)
        ep=CONFIG+'/endpoint_'+a.experiment_run+'.json';bind(ROOT/ep,choice)
        write(E/'preflight_decision.json',choice)
        refresh_receipts();commit_evidence('Record measured preflight and freeze common coverage endpoint before quality scores')
        status(state='running',stage='building_experiment',endpoint=choice['endpoint'],run_id=a.experiment_run)
        plan=build_plan('experiment',a.experiment_run,ep)
        commit_evidence('Bind final paired coverage dispatch and evaluation inputs')
        subprocess.run([sys.executable,'scripts/coding_parallel_session.py','--plan',str(plan),'--validate-only'],cwd=ROOT,check=True)
        status(state='running',stage='experiment',endpoint=choice['endpoint'],run_id=a.experiment_run,plan=str(plan.relative_to(ROOT)))
        with (E/'experiment_controller.txt').open('ab') as f:
            child=subprocess.run([sys.executable,'scripts/coding_parallel_session.py','--plan',str(plan)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        wait_controller_idle(ROOT/'evidence/coding_pilot_v1/control/recovery_controller.lock')
        done,roots=import_stage(a.experiment_run);result_root='results/coding_pilot_v1/'+a.experiment_run+'/experiment'
        if child.returncode or not done['passed']:raise RuntimeError('Experiment incomplete; preserve partial evidence and concrete blocker')
        commit_evidence('Preserve completed paired training and matched free-running evidence')
        status(state='running',stage='review_export',run_id=a.experiment_run)
        receipt=export(result_root)
        status(state='complete',stage='review_verified',run_id=a.experiment_run,receipt=receipt,delivery_commit=git('rev-parse','HEAD'))
    except BaseException as exc:
        status(state='needs_inspection',stage='bounded_stop',result_root=result_root,error=str(exc),traceback=traceback.format_exc())
        try:export(result_root)
        except Exception as nested:write(E/'export_failure.json',{'error':str(nested),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
