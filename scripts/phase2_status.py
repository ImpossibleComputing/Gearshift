#!/usr/bin/env python3
"""Read current local progress without inspecting candidate answers."""
import json
from pathlib import Path


def main():
    root=Path('results/phase2_v1');parts=[]
    for stage in ['characterization','long_outputs','branching','timing_repeats','confirmation_1p7_to_0p6','confirmation_4b_to_0p6']:
        p=root/stage/'progress.json'
        if p.exists():
            x=json.loads(p.read_text());parts.append(f'{stage}: {x["completed"]}/{x["planned"]} ({x["state"]})')
    for pair in ['1p7_to_0p6','4b_to_0p6']:
        p=root/'training'/pair
        if (p/'preparation/manifest.json').exists():parts.append(f'{pair} training histories: {len(list((p/"preparation/cases").glob("*.json")))}/160')
        for progress in sorted(p.glob('seed_*/progress.json')):
            x=json.loads(progress.read_text());parts.append(f'{pair}/{progress.parent.name}: step {x["step"]} ({x["state"]})')
    if (root/'judging_queue_status.json').exists():parts.append('judging: '+json.loads((root/'judging_queue_status.json').read_text())['state'])
    for stage in ['characterization','long_outputs','branching','confirmation_1p7_to_0p6','confirmation_4b_to_0p6']:
        folder=root/stage/'judging'
        if (folder/'condition_key.json').exists():
            n=len(list((folder/'judgments').glob('*/result.json')));total=len(json.loads((folder/'condition_key.json').read_text()))
            parts.append(f'{stage} judge attempts: {n}/{total}')
    failed=[]
    for group in Path('evidence/phase2/execution').glob('*'):
        latest={}
        for p in sorted(group.glob('*_attempt*.json')):latest[p.name.rsplit('_attempt',1)[0]]=p
        for label,p in latest.items():
            if json.loads(p.read_text())['returncode']!=0:failed.append(str(p))
    if failed:parts.append('INFRASTRUCTURE FAILURE: '+', '.join(failed))
    if (root/'generation_pipeline_complete.json').exists():parts.append('All local scientific execution groups completed')
    print(' | '.join(parts))


if __name__=='__main__':main()
