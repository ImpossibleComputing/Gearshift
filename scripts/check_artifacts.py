#!/usr/bin/env python3
"""Standard-library integrity checks; usable on CPU CI and a tensor-free review ZIP."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def check(path,desc):
    assert path.is_file(),f'Missing artifact: {path}'
    assert path.stat().st_size==desc['bytes'] and sha(path)==desc['sha256'],f'Changed artifact: {path}'


def canonical(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');p.add_argument('--heavy',action='store_true');args=p.parse_args()
    manifest=json.loads((ROOT/'evidence/pilot/manifest.json').read_text())
    historical=ROOT/'evidence/phase2/historical_manifest.json'
    if historical.exists():
        for name,desc in json.loads(historical.read_text())['files'].items():check(ROOT/name,desc)
    for name,desc in manifest['files'].items():
        check(ROOT/'evidence/pilot/snapshot'/name,desc)
        if name.startswith(('results/','plots/','configs/','data/')) or name=='RESULTS.md':check(ROOT/name,desc)
    if args.heavy:
        for name,desc in manifest['local_heavy_artifacts'].items():check(ROOT/name,desc)
    root=ROOT/'results/followup_v1'
    for path in root.rglob('*manifest.json'):
        obj=json.loads(path.read_text())
        if 'identity' in obj:assert canonical(obj['identity'])==obj['identity_sha256'],path
    if (root/'task_complete.json').exists():
        mf=json.loads((root/'task_manifest.json').read_text())['identity']
        rows=json.loads((root/'task_records.json').read_text())
        expected={(i,c) for i in mf['ids'] for c in mf['conditions']}
        keys=[(r['dataset_index'],r['condition']) for r in rows]
        assert len(keys)==len(set(keys)) and set(keys)==expected
        trajectories=[json.loads(l) for l in (root/'trajectories/test.jsonl').read_text().splitlines()]
        index={r['dataset_index']:r for r in trajectories}
        assert len(index)==len(trajectories) and set(index)==set(mf['ids'])
        for r in rows:
            tr=index[r['dataset_index']]
            assert r['shared_source_trajectory_sha256']==canonical(tr['prompt_ids']+tr['source_reasoning_ids'])
            assert r['correct_and_valid']==(r['correct'] and r['format_valid'])
            assert r['actual_backend_input_tokens']==r['target_prefill_tokens']+len(r['handoff_input_token_ids'])+r['answer_tokens']
        exp=json.loads((root/'experiment_manifest.json').read_text())['identity']
        assert not set(exp['test_ids'])&set(exp['pilot_excluded_ids'])
        assert not set(exp['train_ids'])&set(exp['validation_ids'])
        for name in ['affine_initialization.json','selection.json']:
            assert (root/name).exists()
        selection=json.loads((root/'selection.json').read_text())
        if args.heavy:check(root/'selected_mapper.pt',selection['selected_artifact'])
    elif args.require_complete:raise AssertionError('Follow-up task is not complete')
    # Every stage's source hash must resolve to an included snapshot.
    for name in ['experiment_manifest.json','training_manifest.json','task_manifest.json']:
        path=root/name
        if not path.exists():continue
        files=json.loads(path.read_text())['identity']['code']
        snapshot=ROOT/'evidence/followup/source_snapshots'/('preparation' if name=='experiment_manifest.json' else canonical(files))
        for name,desc in files.items():check(snapshot/name,desc)
    bundle=ROOT/'review_bundle_manifest.json'
    source_manifest=ROOT/'evidence/followup/source_config_manifest.json'
    def previous_delivery_path(name):
        preserved=ROOT/'evidence/phase2/previous_delivery_source'/name
        return preserved if preserved.exists() else ROOT/name
    if source_manifest.exists():
        for name,desc in json.loads(source_manifest.read_text())['files'].items():check(previous_delivery_path(name),desc)
    if bundle.exists():
        for name,desc in json.loads(bundle.read_text())['files'].items():check(previous_delivery_path(name),desc)
    print('PASS: frozen pilot, identified records, source snapshots, paired trajectories and artifact hashes')


if __name__=='__main__':main()
