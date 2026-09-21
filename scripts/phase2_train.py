#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read
from gearshift.phase2_inference import backends
from gearshift.phase2_training import prepare_training,train


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train']);p.add_argument('--config',default='configs/phase2_v1.json')
    p.add_argument('--pair',default='1p7_to_0p6');p.add_argument('--initialization',default='results/followup_v1/selected_mapper.pt')
    args=p.parse_args();cfg=read(args.config);source,target=backends(cfg)
    if args.stage=='prepare':prepare_training(cfg,source,target,args.pair)
    else:train(cfg,source,target,args.initialization,args.pair)


if __name__=='__main__':main()
