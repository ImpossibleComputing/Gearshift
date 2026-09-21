#!/usr/bin/env python3
"""Finalize compact results after judging, without inference or paid services."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read,write,artifact
from gearshift.phase2_completion_state import transient_usage_check_failure
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'evidence/phase2/studio_transfer'
RUNTIME=STATE/'finalization_runtime'

def terminal_ready(job,queue):
    if job.get('state')=='running':return False
    if job.get('state')!='complete':raise ValueError('Judge wrapper did not complete successfully')
    if queue.get('state') not in ['all_exported_packets_attempted','stopped_with_pending_packets']:
        raise ValueError('Judge queue has no recognized terminal receipt')
    return not transient_usage_check_failure(queue)

def cloud_absence():
    ledger=read(STATE/'cloud/ledger.json');ids=set(ledger['pods'])
    result=subprocess.run(['/opt/homebrew/bin/runpodctl','pod','list','--all','--output','json'],text=True,capture_output=True,timeout=30,check=True)
    pods=json.loads(result.stdout)
    if not isinstance(pods,list):raise ValueError('Invalid provider listing')
    remaining=[p['id'] for p in pods if p['id'] in ids or str(p.get('name','')).startswith('gearshift-phase2-20260914-')]
    if remaining:raise ValueError('Task cloud pods remain; cleanup required before completion')
    if ledger['allocated_upper_bound_usd_hour']!=0:raise ValueError('Ledger still shows allocated task spend')
    return dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),task_pod_ids=sorted(ids),remaining_task_pods=[],
        conservative_upper_bound_usd=ledger['estimated_upper_bound_usd'],unrelated_pods_untouched=True)

def run(command,index):
    start=time.time();stem=RUNTIME/f'command_{index:02d}'
    with stem.with_suffix('.txt').open('w') as log:
        result=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    receipt=dict(command=command,returncode=result.returncode,seconds=time.time()-start)
    write(stem.with_suffix('.json'),receipt)
    if result.returncode:raise RuntimeError('Finalization command failed: '+str(command))
    return receipt

def main():
    from gearshift.phase2_pause import require_unpaused
    require_unpaused()
    parser=argparse.ArgumentParser();parser.add_argument('--job');parser.add_argument('--check-ready',action='store_true');args=parser.parse_args()
    os.chdir(ROOT)
    if args.job is None:args.job=read(STATE/'active_judge_job.json')['job'] if (STATE/'active_judge_job.json').exists() else 'final_judge_queue_02_after_reset'
    job_path=STATE/'jobs'/args.job/'status.json'
    required=['scripts/phase2_report.py','scripts/phase2_diagnostics_report.py','scripts/phase2_judging_report.py',
        'scripts/phase2_evidence_unit_audit.py','scripts/phase2_review_packets.py','scripts/build_phase2_bundle.py','scripts/phase2_verify_bundle.py',
        'PHASE2_RESULTS.md','REPRODUCE_PHASE2.md','REVIEW_GUIDE.md']
    if not all((ROOT/p).exists() for p in required) or not job_path.exists():raise ValueError('Missing finalization input')
    if args.check_ready:
        print(json.dumps(dict(ready=True,judge_job=read(job_path)['state'],cloud=cloud_absence()),indent=2));return
    RUNTIME.mkdir(parents=True,exist_ok=True)
    lock=(RUNTIME/'finalizer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=dict(pid=os.getpid(),started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),judge_job=args.job)
    subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        while True:
            if (STATE/'active_judge_job.json').exists():
                current_job=read(STATE/'active_judge_job.json')['job']
                if current_job!=args.job:
                    args.job=current_job;job_path=STATE/'jobs'/args.job/'status.json';base['judge_job']=args.job
            if not job_path.exists():
                write(RUNTIME/'status.json',dict(base,state='waiting_for_job_start',heartbeat_epoch=time.time()))
                time.sleep(5);continue
            job=read(job_path);queue=read(ROOT/'results/phase2_v1/judging_queue_status.json')
            if terminal_ready(job,queue):break
            if job.get('state')=='complete' and transient_usage_check_failure(queue):
                write(RUNTIME/'status.json',dict(base,state='waiting_for_service_recovery',heartbeat_epoch=time.time()))
                time.sleep(30);continue
            if time.time()-job.get('heartbeat_epoch',0)>180:raise RuntimeError('Judge wrapper heartbeat is stale')
            os.kill(job['pid'],0)
            write(RUNTIME/'status.json',dict(base,state='waiting_for_healthy_judging',heartbeat_epoch=time.time()))
            time.sleep(30)
        write(RUNTIME/'status.json',dict(base,state='finalizing',heartbeat_epoch=time.time()))
        cleanup=cloud_absence();write(STATE/'final_cloud_cleanup_verified.json',cleanup)
        # Freeze operational ledger files before packaging and committing.
        service=f'gui/{os.getuid()}/com.gearshift.phase2.cloud-watchdog'
        stopped=subprocess.run(['/bin/launchctl','bootout',service],capture_output=True,text=True)
        write(RUNTIME/'watchdog_retired.json',dict(service=service,returncode=stopped.returncode))
        # A reboot may already have removed the agent. Task pod absence above is the cleanup gate.
        # Preserve the launchctl result without failing a completed research delivery for an absent service.
        commands=[['scripts/check_artifacts.py','--require-complete'],['scripts/check_phase2.py','--require-complete'],['-m','pytest','-q'],
            ['scripts/phase2_report.py'],['scripts/phase2_diagnostics_report.py'],['scripts/phase2_judging_report.py'],
            ['scripts/phase2_evidence_unit_audit.py'],['scripts/phase2_review_packets.py','examples'],['scripts/phase2_review_packets.py','export']]
        receipts=[run(cmd,i) for i,cmd in enumerate(commands)]
        coverage=read(ROOT/'results/phase2_v1/report/judging_coverage.json')
        note=('All exported subjective-judgment packets were attempted; valid and invalid attempts are reported separately.' if queue['state']=='all_exported_packets_attempted'
              else 'The subjective-judgment queue ended at its declared usage/service stopping condition. Remaining packets are explicitly unscored; this is a bounded partial judging result.')
        manuscript=ROOT/'PHASE2_RESULTS.md';text=manuscript.read_text()
        text=text.replace('**All planned inference, training and objective scoring are complete. Isolated subjective judging is running.**',
            '**The bounded research run is complete.** '+note)
        manuscript.write_text(text)
        completion=dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scientific_inference_and_objective_scoring='complete',
            judging=coverage['final_totals'],judging_state=coverage['state'],human_labels='pending; not a completion prerequisite',
            evidence_task_caveat='EVIDENCE_AUDIT.md: undefined characterization pricing unit; original grades preserved and separate sensitivity audit included.',
            cloud_cleanup=cleanup,validation_commands=receipts,publication='No publication or push; no project license selected.')
        write(STATE/'review_completion.json',completion)
        (ROOT/'STUDIO_STATUS.md').write_text('# Studio research status\n\nThe bounded research run and cloud cleanup are complete. '+note+'\n\nStart with PHASE2_RESULTS.md, EVIDENCE_AUDIT.md, JUDGING_RESULTS.md and REVIEW_GUIDE.md. The compact archive is gearshift_phase2_review.zip. Its manifest, receipt and fresh-extraction verification are beside it. Human labels are pending in the independent 30-pair packet. The finalization runtime status records the final local commit. Nothing was published or pushed.\n')
        run(['scripts/build_phase2_bundle.py'],len(commands))
        run(['scripts/phase2_verify_bundle.py'],len(commands)+1)
        verification=read(ROOT/'phase2_review_bundle_verification.json')
        if verification['status']!='passed' or artifact(ROOT/'gearshift_phase2_review.zip')!={k:verification[k] for k in ['bytes','sha256']}:raise ValueError('Final ZIP is not the verified ZIP')
        manifest=read(ROOT/'phase2_review_bundle_manifest.json')
        paths=[*manifest['files'],'phase2_review_bundle_manifest.json','phase2_review_bundle_receipt.json','phase2_review_bundle_verification.json']
        ignored_result=subprocess.run(['git','check-ignore','--stdin'],input='\n'.join(paths)+'\n',cwd=ROOT,text=True,capture_output=True)
        if ignored_result.returncode not in [0,1]:raise ValueError('Cannot audit ignored bundle files')
        ignored=set(ignored_result.stdout.splitlines())
        allowed_ignored={'evidence/pilot/snapshot/data/qwen3_1.7b_to_0.6b/reasoning_trajectories.jsonl','evidence/pilot/snapshot/data/qwen3_4b_to_0.6b/reasoning_trajectories.jsonl'}
        if ignored-allowed_ignored:raise ValueError('Unexpected ignored compact record; inspect before staging')
        normal=[p for p in paths if p not in ignored]
        for offset in range(0,len(normal),150):subprocess.run(['git','add','--',*normal[offset:offset+150]],cwd=ROOT,check=True,capture_output=True)
        if ignored:subprocess.run(['git','add','-f','--',*sorted(ignored)],cwd=ROOT,check=True,capture_output=True)
        staged=subprocess.run(['git','diff','--cached','--name-only','-z'],cwd=ROOT,check=True,capture_output=True).stdout.split(b'\0')
        if any(Path(p.decode()).suffix in ['.pt','.safetensors','.zip','.npy'] for p in staged if p):raise ValueError('Heavy artifact staged unexpectedly')
        diff=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT)
        if diff.returncode==1:subprocess.run(['git','commit','-m','Complete phase-two transfer evaluation and review bundle'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        elif diff.returncode!=0:raise ValueError('Cannot inspect staged changes')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        receipt=dict(base,state='complete',completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),commit=commit,
            bundle=read(ROOT/'phase2_review_bundle_receipt.json'),verification=artifact(ROOT/'phase2_review_bundle_verification.json'),
            delivery='Ready on the Studio. A monitoring task can copy the compact ZIP and receipts to the laptop when reachable.')
        write(STATE/'CONTROL_DONE.json',receipt);write(RUNTIME/'status.json',receipt)
    except Exception as exc:
        write(RUNTIME/'status.json',dict(base,state='attention_required',error=repr(exc),failed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        raise

if __name__=='__main__':main()
