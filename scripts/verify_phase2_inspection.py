#!/usr/bin/env python3
"""Offline integrity, blinding and records-only reproducibility checks for inspection ZIP."""
import argparse, hashlib, json, os, subprocess, sys, tempfile, zipfile
from pathlib import Path, PurePosixPath

def artifact(p):
    b=Path(p).read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def verify(archive):
    checks=[]
    with tempfile.TemporaryDirectory(prefix='gearshift-inspection-verify-') as td:
        root=Path(td)/'snapshot';root.mkdir()
        with zipfile.ZipFile(archive) as z:
            names=z.namelist();assert len(names)==len(set(names))
            for name in names:
                p=PurePosixPath(name)
                assert not p.is_absolute() and '..' not in p.parts
                assert not set(p.parts)&{'.git','.venv','venv','__pycache__','.cache','node_modules'}
                assert p.suffix not in {'.pt','.safetensors','.npy','.npz'}
                assert ((z.getinfo(name).external_attr>>16)&0o170000)!=0o120000
            assert z.testzip() is None
            z.extractall(root)
        manifest=read(root/'MANIFEST.json')
        assert set(names)==set(manifest['files'])|{'MANIFEST.json'}
        for name,a in manifest['files'].items():assert artifact(root/name)==a,name
        checks.append('CRC, exact file inventory, all payload hashes, safe paths and heavyweight exclusions')
        repo=root/'04_provenance/repository'
        for name,a in read(root/'04_provenance/scientific_record_hashes.json').items():
            assert artifact(repo/name)==a,name
        checks.append('Paused scientific records match source hashes')
        original=repo/'results/phase2_v1/human_review/blinded';blind=root/'01_blinded/reserved_30'
        assert len(list(blind.glob('human_*.json')))==30
        for p in original.iterdir():
            if p.is_file():assert artifact(p)==artifact(blind/p.name),p.name
        reservation=read(repo/'results/phase2_v1/tasks/human_reservation.json')
        key=read(repo/'results/phase2_v1/human_review/separate_condition_key.json')
        assert len(reservation['selected'])==len(key)==30
        for selection,k in zip(reservation['selected'],key):
            assert all(selection[x]==k[x] for x in ['task_id','family','left','right'])
        for folder in [blind,root/'01_blinded/diagnostic_order_disagreements']:
            for p in folder.glob('*.json'):
                if p.name=='blank_labels.json':continue
                packet=read(p);assert set(packet)=={'packet_id','task','supporting_evidence','rubric','candidates'}
                assert set(packet['candidates'])=={'A','B'}
                assert p.stem==packet['packet_id']
        checks.append('Exact original 30-pair membership, blank form/viewer and verbatim payloads; blinded packet schema contains no condition/provenance fields')
        unblind=read(root/'04_provenance/exported_pair_condition_key.json')
        statuses=read(root/'02_judge_evidence/review_pair_status.json')
        byid={r['review_id']:r for r in statuses}
        counts={'review_pairs':len(unblind),'reserved_both_orders':0,'reserved_single_order':0,'reserved_no_orders':0,'diagnostic_pairs':0}
        for r in unblind:
            relative='reserved_30' if r['kind']=='reserved' else 'diagnostic_order_disagreements'
            packet=read(root/'01_blinded'/relative/(r['review_id']+'.json'))
            k0=next(k for k in r['keys'] if k['orientation']==0)
            original_packet=read(repo/'results/phase2_v1'/r['stage']/'judging/packets'/(k0['packet_id']+'.json'))
            assert {k:v for k,v in packet.items() if k!='packet_id'}=={k:v for k,v in original_packet.items() if k!='packet_id'}
            n=0
            for k in r['keys']:
                source=repo/'results/phase2_v1'/r['stage']/'judging/judgments'/k['packet_id']/'result.json'
                exported=root/'02_judge_evidence/pairs'/r['review_id']/f'order_{k["orientation"]}'
                if source.exists():
                    assert artifact(source)==artifact(exported/'result.json');n+=1
                else:assert read(exported/'PENDING.json')['status']=='pending_unscored'
                p=exported/'anonymous_packet.original.json';assert artifact(p)==artifact(repo/'results/phase2_v1'/r['stage']/'judging/packets'/(k['packet_id']+'.json'))
            if r['kind']=='reserved':counts[{2:'reserved_both_orders',1:'reserved_single_order',0:'reserved_no_orders'}[n]]+=1
            else:counts['diagnostic_pairs']+=1
        assert counts['diagnostic_pairs']<=12
        checks.append('All exported raw judgments and candidate strings match original records; missing judgments remain pending')
        env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
        dest=Path(td)/'regenerated'
        p=subprocess.run([sys.executable,str(repo/'scripts/phase2_inspection_diagnostics.py'),'--repo',str(repo),'--output',str(dest)],capture_output=True,text=True,env=env,timeout=180)
        assert p.returncode==0,p.stderr
        for name in ['coverage_by_stage.csv','coverage_by_stage_family_comparison.csv','acceptability_failure_causes.csv','dimension_score_distributions.csv','material_violations_verbatim.csv','complete_pair_order_diagnostics.jsonl','order_disagreement_summary.csv','code_failure_totals.csv','code_per_output.jsonl','code_per_output.csv','code_paired_outcomes.csv','code_paired_summary.csv','code_decoding_settings.json','diagnostic_pair_selection.json','diagnostic_summary.json']:
            assert artifact(dest/name)==artifact(root/'03_diagnostics'/name),'Regeneration differs: '+name
        originals=list((root/'03_diagnostics/code_outputs').rglob('*'))
        for f in originals:
            if f.is_file():assert artifact(f)==artifact(dest/'code_outputs'/f.relative_to(root/'03_diagnostics/code_outputs'))
        summary=read(dest/'diagnostic_summary.json')
        assert summary['code_outputs']==1148
        assert summary['code_tasks_by_stage']=={'characterization':100,'confirmation_1p7_to_0p6':32,'confirmation_4b_to_0p6':16}
        checks.append('All diagnostic tables and 1148 code-output/execution exports reproduce byte-for-byte without weights, network, candidate execution or judge calls')
        p=subprocess.run([sys.executable,str(repo/'scripts/coding_pilot_plan_check.py')],capture_output=True,text=True,env=env,timeout=60)
        assert p.returncode==0,p.stderr
        assert read(repo/'configs/coding_pilot_v1/pilot.json')['authorization']['execution_allowed'] is False
        checks.append('Pilot metadata, geometry, immutable pins, disjoint draft membership and unarmed authorization validate')
        return dict(status='passed',checks=checks,review_material_counts=counts,coverage=summary['final_coverage'],code_outputs=1148,pilot_executed=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
    result={**verify(a.archive),'archive':str(a.archive.resolve()),**artifact(a.archive)}
    if a.receipt:a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
