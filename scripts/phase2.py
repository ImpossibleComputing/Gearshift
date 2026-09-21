#!/usr/bin/env python3
"""Local phase-two stage entry point; every stage has a separate immutable identity."""
import argparse
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--config',default='configs/phase2_v1.json')
    p.add_argument('--tasks');p.add_argument('--conditions');p.add_argument('--mapper',default='results/followup_v1/selected_mapper.pt')
    args=p.parse_args();cfg=read(args.config)
    from gearshift.phase2_inference import backends,run_ablation,run_suite
    started=time.perf_counter();utc=datetime.now(timezone.utc).isoformat();failure=None;loaded=None
    try:
        source,target=backends(cfg);loaded=time.perf_counter()-started
        if args.stage=='ablation':run_ablation(cfg,source,target)
        else:run_suite(cfg,source,target,args.stage,args.tasks,args.mapper,read(args.conditions))
    except BaseException as exc:
        failure=repr(exc);raise
    finally:
        root=Path(cfg['output'])/args.stage;attempt=len(list(root.glob('command_timing_*.json')))
        write(root/f'command_timing_{attempt}.json',dict(started_utc=utc,model_load_seconds=loaded,
             command_wall_seconds=time.perf_counter()-started,error=failure))


if __name__=='__main__':main()
