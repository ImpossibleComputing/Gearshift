#!/usr/bin/env python3
"""Standard-library audit of compact phase-two records and source snapshots."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read,digest,artifact,validate_transaction

ROOT=Path(__file__).resolve().parents[1]


def check(path,expected):
    if artifact(path)!={k:expected[k] for k in ['bytes','sha256']}:raise ValueError(f'Artifact mismatch: {path}')


def snapshots(obj):
    if isinstance(obj,dict):
        if 'snapshot' in obj and 'files' in obj:
            for name,expected in obj['files'].items():check(ROOT/obj['snapshot']/name,expected)
        for value in obj.values():snapshots(value)
    elif isinstance(obj,list):
        for value in obj:snapshots(value)


def main():
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');p.add_argument('--bundle',action='store_true');args=p.parse_args()
    root=ROOT/'results/phase2_v1';states=[]
    for path in root.rglob('*manifest.json'):
        obj=read(path)
        if 'identity' in obj:
            if digest(obj['identity'])!=obj['identity_sha256']:raise ValueError(f'Identity changed: {path}')
        snapshots(obj)
    for stage in root.iterdir():
        if not (stage/'manifest.json').exists() or not (stage/'questions').is_dir():continue
        manifest=read(stage/'manifest.json');identity=manifest['identity'];tasks={t['task_id']:t for t in identity['tasks']}
        observed={};raw={}
        for path in (stage/'questions').glob('*.json'):
            obj=read(path);tid=obj['task_id']
            if tid not in tasks or tid in observed:raise ValueError('Unexpected or duplicate task')
            conditions=identity['conditions'];conditions=conditions if isinstance(conditions,list) else conditions[tid]
            validate_transaction(obj,manifest['identity_sha256'],tid,conditions);observed[tid]=path
            for row in obj['rows']:raw[tid,row['condition']]=row
        if (stage/'complete.json').exists():
            if set(observed)!=set(tasks):raise ValueError('False stage completion')
            for name,expected in read(stage/'complete.json')['files'].items():check(stage/name,expected)
        if (stage/'objective_scores.json').exists():
            scoring=read(stage/'objective_scores.json');dest=stage/scoring['scoring_attempt']
            check(dest/'manifest.json',scoring['scoring_manifest']);all_rows=[]
            for tid,path in observed.items():
                score_path=dest/path.name
                if not score_path.exists():continue
                score=read(score_path)
                if score['input_sha256']!=artifact(path)['sha256']:raise ValueError('Score output identity changed')
                for r in score['rows']:
                    if (r['task_id'],r['condition']) not in raw:raise ValueError('Score without raw output')
                all_rows.extend(score['rows'])
            keys=lambda rs:sorted(rs,key=lambda r:(r['task_id'],r['condition']))
            if keys(all_rows)!=keys(scoring['rows']):raise ValueError('Consolidated scores differ from canonical score files')
        states.append((stage.name,len(observed),len(tasks),(stage/'complete.json').exists()))
        keys=stage/'judging/condition_key.json'
        if keys.exists():
            for key in read(keys):
                packet=read(stage/'judging/packets'/f'{key["packet_id"]}.json')
                if digest(packet)!=key['packet_sha256']:raise ValueError('Blind packet changed')
                result=stage/'judging/judgments'/key['packet_id']/'result.json'
                if result.exists() and read(result)['packet_sha256']!=key['packet_sha256']:raise ValueError('Judgment packet changed')
    for path in (root/'training').glob('*/preparation/complete.json'):
        for name,expected in read(path)['cases'].items():check(path.parent/name,expected)
    if args.require_complete:
        required={'ablation','memory','development','characterization','long_outputs','branching','timing_repeats','confirmation_1p7_to_0p6','confirmation_4b_to_0p6'}
        complete={name for name,n,total,done in states if done}
        if required-complete:raise ValueError('Unfinished stages: '+', '.join(sorted(required-complete)))
        for pair in ['1p7_to_0p6','4b_to_0p6']:
            selected=read(root/'training'/pair/'selection.json')
            for seed in selected['seeds']:
                if not (root/'training'/pair/f'seed_{seed}/complete.json').exists():raise ValueError('Unfinished declared training seed')
    if args.bundle:
        for name,expected in read(ROOT/'phase2_review_bundle_manifest.json')['files'].items():check(ROOT/name,expected)
    for name,n,total,done in sorted(states):print(f'{name}: {n}/{total} canonical tasks; '+('complete' if done else 'partial'))
    print('PASS: compact identities, source snapshots, exact conditions, paired histories, token accounting, scores and judge packet hashes')


if __name__=='__main__':main()
