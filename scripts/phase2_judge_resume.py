#!/usr/bin/env python3
"""Resume byte-identical judging across repository commits, retaining provenance."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read,write,immutable,digest,artifact

def validate_provenance_only_resume(previous,current):
    old=copy.deepcopy(previous);new=copy.deepcopy(current)
    old_sha=old['source'].pop('git_sha')
    new_sha=new['source'].pop('git_sha')
    if digest(old)!=digest(new):raise ValueError('Judging runtime changed beyond repository revision')
    if not old_sha or not new_sha:raise ValueError('Missing source revision provenance')
    for name,expected in current['source']['files'].items():
        if artifact(name)!=expected:raise ValueError('Active judging source bytes changed')
        if artifact(Path(current['source']['snapshot'])/name)!=expected:raise ValueError('Preserved judging source bytes changed')
    return dict(previous_git_sha=old_sha,current_git_sha=new_sha)

def main():
    from gearshift.phase2_pause import require_unpaused
    require_unpaused()
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run']);p.add_argument('--stage',required=True)
    p.add_argument('--scope',choices=['all','primary'],default='all');p.add_argument('--task-ids');p.add_argument('--limit',type=int);args=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import phase2_judge as judge
    original_bind=judge.bind
    def bind_runtime(path,identity):
        path=Path(path)
        if not path.exists():return original_bind(path,identity)
        previous=read(path)
        if digest(previous['identity'])!=previous['identity_sha256']:raise ValueError('Stored runtime manifest changed')
        if digest(identity)==previous['identity_sha256']:return original_bind(path,identity)
        if path.name!='runtime_manifest.json':raise ValueError('Unexpected resume binding')
        revisions=validate_provenance_only_resume(previous['identity'],identity)
        receipt=path.parent/'runtime_resumptions'/f'{time.time_ns()}.json'
        immutable(receipt,dict(reason='Repository revision changed while all bound scientific code, isolation, instructions, CLI, rubric and policy hashes stayed identical.',
            original_runtime_manifest=artifact(path),current_requested_identity=identity,wrapper_source=artifact(Path(__file__)),
            permitted_difference='source.git_sha only; historical manifest remains untouched',**revisions))
        return previous
    judge.bind=bind_runtime
    judge.run(Path('results/phase2_v1')/args.stage/'judging',args.limit,args.scope,args.task_ids)

if __name__=='__main__':main()
