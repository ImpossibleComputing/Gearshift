#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from gearshift.phase2_io import read
from gearshift.phase2_extended import reserve,run_extended,run_timing
from gearshift.phase2_inference import backends
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reserve','long_outputs','branching','timing']);p.add_argument('--config',default='configs/phase2_v1.json');args=p.parse_args();cfg=read(args.config)
    if args.stage=='reserve':reserve(cfg)
    else:
        source,target=backends(cfg)
        if args.stage=='timing':run_timing(cfg,source,target)
        else:run_extended(cfg,source,target,args.stage)
