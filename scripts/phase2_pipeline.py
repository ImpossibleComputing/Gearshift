#!/usr/bin/env python3
"""Run the locked local scientific groups, stopping at the first infrastructure failure."""
from pathlib import Path
import subprocess
import sys
import datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read,write,artifact


def main():
    root=Path('results/phase2_v1')
    read(root/'final_protocol_lock.json')
    for group in ['characterization','extensions','adaptation','replication']:
        result=subprocess.run([sys.executable,'scripts/phase2_execute.py',group])
        if result.returncode:raise SystemExit(result.returncode)
    write(root/'generation_pipeline_complete.json',dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        groups={g:artifact(Path('evidence/phase2/execution')/g/'commands.json') for g in ['characterization','extensions','adaptation','replication']}))


if __name__=='__main__':main()
