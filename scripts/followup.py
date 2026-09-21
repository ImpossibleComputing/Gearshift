#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.followup import prepare, train_comparison, task_comparison, backends
from gearshift.core import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','task']);p.add_argument('--config',default='configs/followup_v1.json');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());start=time.perf_counter()
    source,target=backends(cfg);load=time.perf_counter()-start
    try:
        {'prepare':prepare,'train':train_comparison,'task':task_comparison}[a.stage](cfg,source,target)
    except Exception:
        import traceback
        failure=Path(cfg['output'])/'failures.jsonl';failure.parent.mkdir(parents=True,exist_ok=True)
        with failure.open('a') as f:f.write(json.dumps(dict(stage=a.stage,traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start))+'\n')
        raise
    timing=Path(cfg['output'])/(a.stage+'_command_timing.json')
    if not timing.exists():save_json(timing,dict(model_load_seconds=load,full_command_seconds=time.perf_counter()-start))


if __name__=='__main__':main()
