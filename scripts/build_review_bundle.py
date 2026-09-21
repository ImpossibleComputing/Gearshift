#!/usr/bin/env python3
"""Explicit small-evidence ZIP, with a complete SHA-256 inventory."""
import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_artifacts import ROOT,sha


def main():
    excluded={'.git','.venv','venv','__pycache__','.pytest_cache','.cache'}
    extensions={'.py','.json','.jsonl','.csv','.md','.txt','.png','.svg','.yml','.yaml','.toml','.ini','.patch','.diff'}
    roots=['gearshift','configs','tests','scripts','results','plots','evidence','gearshift_review','.github','third_party']
    candidates=[p for dirname in roots for p in (ROOT/dirname).rglob('*') if p.is_file()]
    candidates += [p for p in ROOT.iterdir() if p.is_file()]
    candidates += [ROOT/'data'/name/'reasoning_trajectories.jsonl' for name in ['qwen3_1.7b_to_0.6b','qwen3_4b_to_0.6b']]
    files={}
    for p in sorted(set(candidates)):
        rel=p.relative_to(ROOT)
        if excluded&set(rel.parts) or p.is_symlink():continue
        if p.name=='review_bundle_manifest.json':continue
        if p.suffix not in extensions and p.name!='.gitignore':continue
        if p.stat().st_size>16*1024*1024:raise ValueError(f'Unexpected heavy evidence file: {rel}')
        files[str(rel)]={'bytes':p.stat().st_size,'sha256':sha(p)}
    manifest=dict(files=files,excluded='Git internals, virtualenvs, model weights, mapper tensor checkpoints, token/KV tensor caches, old ZIPs, bytecode and reproducible binary artifacts',
        selected_mapper='results/followup_v1/selected_mapper.pt (separate local export; hash/size in selection.json)',
        license_status='Project LICENSE still requires owner choice. No public release authorized.')
    (ROOT/'review_bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    path=ROOT/'gearshift_followup_review.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in [*files,'review_bundle_manifest.json']:z.write(ROOT/name,'gearshift/'+name)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)+1
        for pair in ['qwen3_1.7b_to_0.6b','qwen3_4b_to_0.6b']:
            assert 'gearshift/data/'+pair+'/reasoning_trajectories.jsonl' in z.namelist()
    print(f'{path}: {path.stat().st_size:,} bytes; {len(files)+1} files; SHA-256 {sha(path)}')


if __name__=='__main__':main()
