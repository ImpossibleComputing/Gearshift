#!/usr/bin/env python3
"""Resume only the exact declared interrupted history cohort in a fresh namespace."""
import argparse,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write
from gearshift.coding_training import worker_context,finish_worker
from gearshift.coding_history_recovery import run_recovered
import coding_history_worker
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker-spec');args=parser.parse_args();context=None
    try:
        context=worker_context(ROOT,args.worker_spec)
        run_recovered(context,ROOT,coding_history_worker)
    except BaseException as exc:
        if context:
            finish_worker(context,'failed',exception=type(exc).__name__,error=str(exc))
            write(context['root']/'failure.json',{'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
