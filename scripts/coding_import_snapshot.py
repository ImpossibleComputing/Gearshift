#!/usr/bin/env python3
"""Import immutable completed baseline records from a verified compact snapshot."""
import argparse,json,shutil,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import sha,digest,write
from gearshift.coding_snapshot import verify_archive
from gearshift.coding_reuse import verify_completed
ROOT=Path(__file__).resolve().parents[1]

def copy_new(src,dest):
    if dest.exists():
        if sha(src)!=sha(dest):raise ValueError('Refusing to replace existing evidence: '+str(dest))
    else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);args=parser.parse_args()
    imported=[]
    with tempfile.TemporaryDirectory() as tmp:
        stage=Path(tmp)/'snapshot';manifest=verify_archive(args.archive,stage)
        runtime_files=list((stage/'evidence/coding_pilot_v1').rglob('runtime_lock.json'))
        # Include the separately identified cap amendment as well as original
        # preflight sessions; the amendment retains its upstream transactions.
        for source in sorted((stage/'results/coding_pilot_v1').glob('preflight_*')):
            if not (source/'identity.json').exists():continue
            identity=json.loads((source/'identity.json').read_text());dest=ROOT/'results/coding_pilot_v1'/source.name
            for p in source.glob('*.json'):
                if p.name!='progress.json':copy_new(p,dest/p.name)
            for p in (source/'controls').glob('*.json'):copy_new(p,dest/'controls'/p.name)
            count=0
            for p in sorted(source.glob('tasks/*/complete.json')):
                row=json.loads(p.read_text());verify_completed(p.parent,identity,row['task_id'])
                if 'reused_from' in row:
                    lineage=row['reused_from'];preserved=p.parent/lineage['preserved_complete_file']
                    upstream=stage/lineage['logical_path']/'complete.json'
                    if sha(preserved)!=lineage['complete_file_sha256'] or sha(upstream)!=sha(preserved):raise ValueError('Reuse lineage differs from original transaction')
                for f in p.parent.iterdir():
                    if f.is_file():copy_new(f,dest/'tasks'/p.parent.name/f.name)
                count+=1
            if (source/'progress.json').exists():copy_new(source/'progress.json',dest/f'progress_at_{count}_tasks.json')
            runtime=[p for p in runtime_files if digest(json.loads(p.read_text()))==identity['runtime_sha256']]
            if not runtime:raise ValueError('Runtime lock missing for '+source.name)
            copy_new(runtime[0],ROOT/'evidence/coding_pilot_v1'/source.name/'runtime_lock.json')
            imported.append({'namespace':source.name,'completed_tasks':count,'identity_sha256':digest(identity)})
    receipt={'archive_sha256':sha(args.archive),'snapshot_epoch':manifest['epoch'],'imported':imported}
    dest=ROOT/'evidence/coding_pilot_v1/imports'/(receipt['archive_sha256']+'.json')
    if dest.exists():
        if json.loads(dest.read_text())!=receipt:raise ValueError('Snapshot import identity differs')
    else:write(dest,receipt)
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
