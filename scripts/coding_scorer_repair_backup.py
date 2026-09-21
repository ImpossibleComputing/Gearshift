#!/usr/bin/env python3
"""Compact scorer-repair backup; never includes tests' private inputs or environments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import sha,write

EXPERIMENT='confirmation_01_20260919T094418Z'


def backup(repo,output):
    repo=Path(repo).resolve();output=Path(output).resolve();paths=set()
    result=repo/'results/coding_pilot_v1'/EXPERIMENT/'scorer_repair'
    evidence=repo/'evidence/coding_pilot_v1'/EXPERIMENT/'scorer_repair'
    if json.loads((result/'execution_status.json').read_text())['state']!='complete':raise ValueError('Scoring execution incomplete')
    for directory in (result,evidence):
        paths.update(p for p in directory.rglob('*') if p.is_file() and not p.is_symlink())
    old=repo/'results/coding_pilot_v1/coverage_generalization_v2_20260918T233720Z/evaluation'
    paths.add(old/'scored_answer_manifest.json')
    for item in json.loads((old/'scored_answer_manifest.json').read_text())['files']:
        for key,h in (('path','sha256'),('answer_path','answer_sha256')):
            path=old/item[key]
            if sha(path)!=item[h]:raise ValueError('Original source artifact drift')
            paths.add(path)
    paths.update(repo/n for n in ['gearshift/__init__.py','gearshift/coding_control.py','gearshift/coding_sandbox.py','gearshift/coding_sandbox_v2.py','scripts/coding_sandbox_child.py','scripts/coding_sandbox_child_v2.py','tests/test_coding_sandbox_v2.py','tests/test_coding_scorer_repair.py','configs/coding_pilot_v1/confirmation_01/SCORER_REPAIR_RUNBOOK.md','SCORER_REPAIR_RESULTS.md'])
    paths.update((repo/'scripts').glob('coding_scorer_repair*.py'))
    rows=[]
    for path in sorted(paths):
        relative=path.relative_to(repo)
        if any(x in relative.parts for x in ('private','.git','.venv','__pycache__','.cache')) or path.suffix in ('.pt','.bin','.safetensors','.pyc'):raise ValueError('Excluded artifact in compact backup')
        if path.is_symlink() or not path.is_file():raise ValueError('Missing or linked backup source')
        rows.append({'path':relative.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    manifest={'schema_version':1,'scope':'Completed saved-v2 scorer repair only; no fresh confirmation outputs.','experiment_id':EXPERIMENT,'files':rows,'private_tests_included':False,'weights_or_environments_included':False}
    if output.exists():raise ValueError('Backup destination already exists; do not overwrite')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for r in rows:z.write(repo/r['path'],r['path'])
        z.writestr('SCORER_REPAIR_FILE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output) as z:
        for r in rows:
            content=z.read(r['path'])
            if len(content)!=r['bytes'] or hashlib.sha256(content).hexdigest()!=r['sha256']:raise ValueError('Backup readback mismatch')
        if set(z.namelist())!={r['path'] for r in rows}|{'SCORER_REPAIR_FILE_MANIFEST.json'}:raise ValueError('Unexpected archive member')
    receipt={'archive_path':str(output),'bytes':output.stat().st_size,'sha256':sha(output),'member_files':len(rows)+1,'all_members_readback_verified':True,'private_tests_included':False}
    write(output.with_suffix('.receipt.json'),receipt);return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default=Path(__file__).resolve().parents[1]);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(backup(a.repo,a.output)))
if __name__=='__main__':main()
