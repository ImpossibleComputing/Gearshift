"""Compact, immutable phase-two transactions (no model loading required)."""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
import platform
import importlib.metadata
from pathlib import Path


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def artifact(path):
    p = Path(path); h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''): h.update(block)
    return dict(bytes=p.stat().st_size, sha256=h.hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def runtime_identity():
    versions={}
    for package in ['torch','transformers','numpy','scipy','datasets','tokenizers']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    return dict(python=platform.python_version(),platform=platform.platform(),packages=versions)


def write(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); tmp.replace(p)


def immutable(path, obj):
    p = Path(path)
    if p.exists():
        if digest(read(p)) != digest(obj): raise ValueError(f'Immutable artifact mismatch: {p}')
    else: write(p, obj)
    return obj


def bind(path, identity):
    return immutable(path, dict(identity=identity, identity_sha256=digest(identity)))


def snapshot(names):
    files = {str(p): artifact(p) for p in sorted(map(Path, names))}
    root = Path('evidence/phase2/source_snapshots') / digest(files)
    for name, desc in files.items():
        p = root / name
        if not p.exists(): p.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(name, p)
        if artifact(p) != desc: raise ValueError('Source snapshot changed')
    return dict(files=files, snapshot=str(root), git_sha=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())


def inference_source():
    names = ['gearshift/core.py','gearshift/mapping.py','gearshift/reasoning.py','gearshift/followup.py',
             'gearshift/identity.py','gearshift/data.py','gearshift/phase2_io.py','gearshift/phase2_inference.py',
             'gearshift/phase2_tasks.py','requirements.lock.txt','scripts/phase2.py']
    return snapshot(names)


def validate_transaction(obj, identity_sha, task_id, conditions):
    if obj['identity_sha256'] != identity_sha or obj['task_id'] != task_id:
        raise ValueError('Task identity mismatch')
    if obj['payload_sha256'] != digest({k:v for k,v in obj.items() if k != 'payload_sha256'}):
        raise ValueError('Task payload changed')
    keys = [r['condition'] for r in obj['rows']]
    if len(keys) != len(set(keys)) or set(keys) != set(conditions):
        raise ValueError('Incomplete/duplicate task transaction')
    tr = obj['trajectory']
    if tr['token_sha256'] != digest(tr['prompt_ids'] + tr['source_reasoning_ids']):
        raise ValueError('Source tokens changed')
    for row in obj['rows']:
        if row['shared_source_trajectory_sha256'] != tr['token_sha256']:
            raise ValueError('Unpaired source history')
        if row['answer_tokens'] != len(row['answer_token_ids']): raise ValueError('Answer token count')
        if row['actual_backend_input_tokens'] != row['prefill_tokens'] + row['bridge_tokens'] + row['answer_tokens'] + row.get('small_reasoning_tokens',0):
            raise ValueError('Input token accounting')
        if row['condition'].split('/')[0] == 'M' and row['prefill_tokens'] != 0:
            raise ValueError('Mapped handoff unexpectedly prefills target history')


def checkpoint_status(path, expected):
    if not Path(path).is_file(): return dict(status='unavailable_not_verified', expected=expected)
    actual = artifact(path)
    if actual != expected: raise ValueError('Checkpoint hash mismatch')
    return dict(status='verified', expected=expected, actual=actual)
