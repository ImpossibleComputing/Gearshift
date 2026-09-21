#!/usr/bin/env python3
"""Sequential reproducible baseline runner; no concurrent accelerator stages."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/qwen3_1.7b_to_0.6b.json')
p.add_argument('--refine',action='store_true')
p.add_argument('--stretch',action='store_true')
a=p.parse_args()
stages=['environment','introspect','control','extract','train','evaluate']
if a.refine: stages+=['refine']
if a.stretch: stages+=['stretch']
if a.refine or a.stretch: stages+=['evaluate']
stages+=['reasoning']
linear_saved=False
for stage in stages:
    subprocess.run([sys.executable,'scripts/run.py',stage,'--config',a.config],check=True)
    if stage=='evaluate' and not linear_saved and (a.refine or a.stretch):
        root=Path(json.loads(Path(a.config).read_text())['output'])
        for name in ['functional.json','functional.csv','latency.csv','free_continuations.json']:
            source=root/name
            shutil.copy2(source,source.with_name(source.stem+'_linear'+source.suffix))
        linear_saved=True
