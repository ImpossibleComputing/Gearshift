#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_snapshot import stream
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker-spec');args=parser.parse_args()
    stream(Path(__file__).resolve().parents[1],sys.stdout.buffer,worker_spec=args.worker_spec)
