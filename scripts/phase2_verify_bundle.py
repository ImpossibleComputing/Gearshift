#!/usr/bin/env python3
"""Extract and validate a compact review ZIP independently of weights and Git."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def comparable_csv(path):
    with path.open() as f:
        rows=list(csv.DictReader(f))
    identifiers=['stage','family','stratum','condition','a','b','metric','pair','seed','arm']
    keyed={tuple((k,r[k]) for k in identifiers if k in r):r for r in rows}
    if len(keyed)!=len(rows):raise ValueError('Non-unique summary identity: '+str(path))
    return keyed


def compare_tables(expected,actual):
    if expected.keys()!=actual.keys():raise ValueError('Regenerated table membership differs')
    for key,left in expected.items():
        right=actual[key]
        if left.keys()!=right.keys():raise ValueError('Regenerated columns differ')
        for column,a in left.items():
            b=right[column]
            if a==b:continue
            try:equal=math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10)
            except (ValueError,TypeError):equal=False
            if not equal:raise ValueError('Regenerated metric differs: '+str((key,column)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',default=str(ROOT/'gearshift_phase2_review.zip'));args=p.parse_args()
    source=Path(args.zip);record=dict(zip=str(source),bytes=source.stat().st_size,sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),commands=[])
    try:
        with tempfile.TemporaryDirectory(prefix='gearshift-review-verify-') as td:
            folder=Path(td)
            with zipfile.ZipFile(source) as archive:
                names=archive.namelist()
                if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP path')
                forbidden={'.git','.venv','venv','env','__pycache__','.cache','node_modules'}
                for name in names:
                    path=Path(name)
                    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='gearshift':raise ValueError('Unsafe ZIP path')
                    if forbidden.intersection(path.parts) or path.suffix in {'.pt','.safetensors','.npy','.npz'}:raise ValueError('Heavy artifact in ZIP')
                if archive.testzip() is not None:raise ValueError('CRC mismatch')
                archive.extractall(folder)
            checkout=folder/'gearshift'
            record['without_weights_or_git']=not (checkout/'.git').exists() and not list(checkout.rglob('*.pt')) and not list(checkout.rglob('*.safetensors'))
            report=checkout/'results/phase2_v1/report'
            expected={name:comparable_csv(report/name) for name in ['all_objective_summary.csv','all_paired_objective.csv','training_summary.csv']}
            commands=[['scripts/check_artifacts.py','--require-complete'],['scripts/check_phase2.py','--require-complete','--bundle'],
                ['-m','pytest','-q'],['scripts/phase2_report.py'],['scripts/phase2_diagnostics_report.py'],
                ['scripts/phase2_judging_report.py'],['scripts/phase2_evidence_unit_audit.py'],['scripts/phase2_review_packets.py','examples'],['scripts/phase2_review_packets.py','export'],
                ['scripts/followup_report.py','--records-only','--destination','results/regenerated_previous_followup','--verify-checkpoint']]
            env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',MPLBACKEND='Agg')
            for command in commands:
                start=time.time();result=subprocess.run([sys.executable,*command],cwd=checkout,env=env,capture_output=True,text=True,timeout=900)
                item=dict(command=command,returncode=result.returncode,seconds=time.time()-start,stdout=result.stdout[-14000:],stderr=result.stderr[-3000:]);record['commands'].append(item)
                if result.returncode:raise ValueError('Extracted-bundle command failed: '+str(command))
            for name,table in expected.items():compare_tables(table,comparable_csv(report/name))
            status=json.loads((checkout/'results/regenerated_previous_followup/summary.json').read_text())['checkpoint_verification']['status']
            if status!='unavailable_not_verified':raise ValueError('Omitted checkpoint incorrectly reported as available')
            record.update(status='passed',summary_tables_regenerated=True,omitted_checkpoint_status=status,archive_members=len(names),finished_epoch=time.time())
    except Exception as exc:
        record.update(status='failed',error=repr(exc),finished_epoch=time.time())
        raise
    finally:
        (ROOT/'phase2_review_bundle_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='commands'},indent=2))


if __name__=='__main__':main()
