#!/usr/bin/env python3
"""Detached bounded CPU orchestration; survives console/SSH disconnection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write,sha
from scripts.coding_scorer_repair_rescore import freeze_plan,finalize
from scripts.coding_scorer_repair_report import report


def execute(evaluation_root,private_tests,policy,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    plan=freeze_plan(evaluation_root,private_tests,policy,out);cpus=plan['policy']['physical_cpu_ids'];workers=plan['policy']['maximum_concurrent_candidates'];started=time.time()
    status=out/'execution_status.json'
    write(status,{'state':'diagnostics','started_epoch':started,'pid':os.getpid(),'planned_records':len(plan['records']),'planned_workers':workers})
    for phase in ('diagnostics','run'):
        processes=[];logs=[]
        try:
            for index in range(workers):
                log=(out/f'{phase}_worker_{index:02d}.log').open('a');logs.append(log)
                cmd=[sys.executable,str(Path(__file__).with_name('coding_scorer_repair_rescore.py')),phase,
                    '--evaluation-root',str(evaluation_root),'--private-tests',str(private_tests),'--output',str(out),
                    '--shard-index',str(index),'--shard-count',str(workers),'--cpu-id',str(cpus[index])]
                processes.append(subprocess.Popen(cmd,stdout=log,stderr=log,stdin=subprocess.DEVNULL))
            while any(p.poll() is None for p in processes):
                write(status,{'state':phase,'started_epoch':started,'epoch':time.time(),'pid':os.getpid(),'workers':[{'pid':p.pid,'returncode':p.poll()} for p in processes]})
                time.sleep(5)
            codes=[p.wait() for p in processes]
            if any(codes):raise RuntimeError(f'{phase} worker exited: {codes}; preserve receipts and missing transactions before bounded recovery')
        finally:
            for log in logs:log.close()
    manifest=finalize(out);summary=report(out)
    write(status,{'state':'complete','started_epoch':started,'completed_epoch':time.time(),'committed':manifest['committed'],'missing':manifest['missing'],'diagnostic_complete':summary['diagnostic_complete'],'manifest_sha256':sha(out/'rescored_answer_manifest.json')})
    print(json.dumps({'state':'complete','committed':manifest['committed'],'missing':manifest['missing']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation-root',required=True);p.add_argument('--private-tests',required=True);p.add_argument('--policy',required=True);p.add_argument('--output',required=True);a=p.parse_args();execute(a.evaluation_root,a.private_tests,a.policy,a.output)
if __name__=='__main__':main()
