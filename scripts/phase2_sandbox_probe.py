#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_sandbox import probe
from gearshift.phase2_io import write
if __name__=='__main__':
    result=probe();write('evidence/phase2/sandbox_probe.json',result)
    print({k:v for k,v in result.items() if k in ['passed','restrictions']})
    if not result['passed']:raise SystemExit(1)
