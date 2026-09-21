#!/usr/bin/env python3
"""Regenerate public numerical analyses, without benchmark text, candidates or models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import coding_confirmation_report as primary
from scripts import coding_confirmation_secondary_report as secondary

CONF=Path('results/coding_pilot_v1/confirmation_01_20260919T094418Z')
def rows(path):
    def scalar(v):
        if v=='True':return True
        if v=='False':return False
        if v=='':return None
        try:return json.loads(v)
        except (ValueError,TypeError):return v
    with path.open(newline='') as f:return [{k:scalar(v) for k,v in row.items()} for row in csv.DictReader(f)]
def read(p):return json.loads(p.read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def verify_subset(actual,expected,label):
    differences=[k for k in actual if actual[k]!=expected.get(k)]
    if differences:raise ValueError(label+' differs: '+str(differences))
    return {'all_recomputed_fields_equal':True,'fields':list(actual)}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve()
    if out.exists() and any(out.iterdir()):ap.error('Use a new empty output directory')
    if out==ROOT or out.is_relative_to(ROOT/'results'):ap.error('Cannot replace archived inputs')
    out.mkdir(parents=True,exist_ok=True)
    # Verify every retained original evidence byte before interpretation.
    manifest=read(ROOT/'release/source_manifest.json')
    checked=[]
    for r in manifest['files']:
        if not r['path'].startswith('results/'):continue
        p=ROOT/r['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['original_sha256']:raise ValueError('Original evidence differs: '+r['path'])
        checked.append(r['path'])
    analysis=read(ROOT/'configs/coding_pilot_v1/confirmation_01/analysis_plan.json')
    old=read(ROOT/CONF/'primary/report/summary.json');r=rows(ROOT/CONF/'primary/report/per_task_seed.csv')
    stats,task,gains,_=primary.cluster_statistics(r,old['task_ids'],analysis)
    receipt={'retained_evidence_files_byte_identical':checked,'primary':verify_subset(stats,old,'primary')}
    save(out/'primary/statistics.json',stats)
    primary.csv_write(out/'primary/per_task.csv',task);primary.csv_write(out/'primary/task_gains_losses.csv',gains)
    regenerated={**old,**stats,'diagnostics':primary.diagnostics_summary(r)}
    receipt['primary_diagnostics_equal']=regenerated['diagnostics']==old['diagnostics']
    if not receipt['primary_diagnostics_equal']:raise ValueError('Primary diagnostics differ')
    (out/'primary/REPORT.md').write_text(primary.render(regenerated))
    r2=rows(ROOT/CONF/'secondary/report/per_draw.csv')
    refs=[{**x,'cohort':'primary'} for x in r if x['condition'] in secondary.REFERENCE_CONDITIONS]
    old2=read(ROOT/CONF/'secondary/report/summary.json')
    stats2,task2,gains2,_=secondary.statistics(r2,old2['task_ids'],analysis,refs)
    receipt['secondary']=verify_subset(stats2,old2,'secondary')
    save(out/'secondary/statistics.json',stats2)
    primary.csv_write(out/'secondary/per_task.csv',task2);primary.csv_write(out/'secondary/gains_losses.csv',gains2)
    lines=['# Separate training-seed replication (regenerated)','','Same 200 tasks; three draws each. Native references are reused, not fresh observations.','', '| Condition | Passed / draws | Mean success |','|---|---:|---:|']
    for k,v in stats2['conditions'].items():lines.append(f"| {k} | {v['passed_draws']} / {v['draws']} | {v['pass_rate']:.6%} |")
    lines+=['','All paired intervals and contrasts: statistics.json. Original full narrative and timing caveats remain in the retained secondary report.']
    (out/'secondary/REPORT.md').write_text('\n'.join(lines)+'\n')
    for name,script,extra in [('sparse','sparse_repair_report.py',['--decision-review','results/sparse_repair_01/report/decision_review.json']),('economics','sparse_repair_economics.py',[])]:
        folder='report' if name=='sparse' else 'economics'
        reportname='SPARSE_REPAIR_RESULTS.md' if name=='sparse' else 'ECONOMIC_CEILING.md'
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--input',str(ROOT/f'results/sparse_repair_01/{folder}/input_records.json'),'--output',str(out/name),'--report',str(out/reportname),*extra],check=True,stdout=subprocess.DEVNULL,cwd=ROOT)
        summary=read(out/name/'summary.json');original=read(ROOT/f'results/sparse_repair_01/{folder}/summary.json')
        if summary!=original:raise ValueError(name+' summary differs')
        if (out/reportname).read_bytes()!=(ROOT/reportname).read_bytes():raise ValueError(name+' report bytes differ')
        receipt[name]={'entire_summary_equal':True,'report_bytes_identical':True,'report_sha256':hashlib.sha256((out/reportname).read_bytes()).hexdigest()}
    receipt['scope']='Records-only recomputation. No grading/model execution; no independent verification of private raw programs/tests. Frozen byte identities and outcome/timing statistics are preserved.'
    save(out/'verification.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
