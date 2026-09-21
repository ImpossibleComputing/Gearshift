#!/usr/bin/env python3
"""Fetch official pinned benchmark files into ignored local storage; no model/code execution."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/coding_pilot_v1/draft_membership.json'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()

def verify_rows(path, filename, manifest):
    wanted = {(r['platform'], str(r['question_id'])): r for rows in manifest['selected'].values()
              for r in rows if r['upstream_file'] == filename}
    found = set(); rows = 0
    with path.open('rb') as f:
        for line in f:
            row = json.loads(line); rows += 1
            key = (row['platform'], str(row['question_id']))
            if key not in wanted: continue
            if key in found: raise ValueError('Duplicate selected task identifier')
            variants = [line, line.rstrip(b'\r\n'), json.dumps(row, sort_keys=True).encode(),
                        json.dumps(row, sort_keys=True, separators=(',', ':')).encode()]
            if wanted[key]['source_row_sha256'] not in {hashlib.sha256(v).hexdigest() for v in variants}:
                raise ValueError('Selected row hash mismatch: ' + '/'.join(key))
            found.add(key)
    if found != set(wanted): raise ValueError('Missing selected task identifiers')
    if rows != manifest['source_files'][filename]['rows']: raise ValueError('Upstream row count mismatch')
    return {'rows': rows, 'selected_identifiers_verified': len(found)}

def fetch_one(filename, manifest, dest):
    meta = manifest['source_files'][filename]
    expected_url = ('https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/'
                    + manifest['dataset_revision'] + '/' + filename)
    if meta['url'] != expected_url: raise ValueError('Unexpected upstream URL')
    path = dest / filename
    if not path.exists():
        partial = path.with_suffix('.partial')
        with urllib.request.urlopen(expected_url, timeout=180) as response, partial.open('wb') as out:
            while chunk := response.read(8 * 1024**2): out.write(chunk)
        if partial.stat().st_size != meta['bytes'] or digest(partial) != meta['sha256']:
            partial.unlink(); raise ValueError('Downloaded file failed integrity check')
        partial.replace(path)
    if path.stat().st_size != meta['bytes'] or digest(path) != meta['sha256']:
        raise ValueError('Cached file failed integrity check')
    return {'file': filename, 'sha256': meta['sha256'], 'bytes': meta['bytes'],
            **verify_rows(path, filename, manifest)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', action='append', help='Only these upstream files; default all six (~4.5 GB)')
    parser.add_argument('--output', type=Path, default=ROOT/'data/coding_pilot_v1/raw')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args(); manifest = json.loads(MANIFEST.read_text())
    names = args.file or list(manifest['source_files'])
    if len(set(names)) != len(names) or not set(names) <= set(manifest['source_files']): parser.error('Unknown/duplicate filename')
    if not 1 <= args.workers <= 6: parser.error('workers must be 1..6')
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        verified = list(pool.map(lambda name: fetch_one(name, manifest, args.output), names))
    print(json.dumps({'dataset': 'livecodebench/code_generation_lite', 'revision': manifest['dataset_revision'],
                      'release': manifest['release'], 'files': verified,
                      'note': 'Local third-party data only. Never commit, bundle or upload these files.'}, indent=2))
if __name__ == '__main__': main()
