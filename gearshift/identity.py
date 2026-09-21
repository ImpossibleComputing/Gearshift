"""Fail-closed identities and append-only scientific record validation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .core import file_sha256, save_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def tokenizer_identity(tokenizer):
    return dict(name=tokenizer.name_or_path,
        revision=tokenizer.init_kwargs.get('_commit_hash', tokenizer.init_kwargs.get('revision')),
        backend_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        chat_template_sha256=digest(tokenizer.chat_template),
        special_tokens=tokenizer.special_tokens_map)


def backend_identity(backend):
    return dict(name=backend.name, revision=backend.config._commit_hash,
        config_sha256=digest(backend.config.to_dict()), dtype=str(backend.dtype),
        device=str(backend.device), tokenizer=tokenizer_identity(backend.tokenizer),
        effective_eos_ids=sorted(backend.eos))


def bind(path, identity, **metadata):
    """Validate before writing. Existing manifests (including annotations) are untouched."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('identity') != identity or old.get('identity_sha256') != digest(identity):
            raise ValueError(f'Experiment identity mismatch at {path}; choose a new run ID/output. Existing evidence is unchanged.')
        return old
    obj = dict(identity=identity, identity_sha256=digest(identity), **metadata)
    save_json(path, obj)
    return obj


def validate_records(rows, question_ids, conditions, *, complete=False):
    expected = {(int(i), c) for i in question_ids for c in conditions}
    actual = [(int(r['dataset_index']), r['condition']) for r in rows]
    if len(set(actual)) != len(actual):
        raise ValueError('Duplicate (question_id, condition) observations')
    if not set(actual) <= expected:
        raise ValueError('Records contain questions/conditions outside the experiment identity')
    missing = expected - set(actual)
    if complete and missing:
        raise ValueError(f'Incomplete experiment: {len(missing)} observations missing')
    return dict(state='partial' if missing else 'complete', expected=len(expected),
                completed=len(actual), missing=len(missing))


def protect_pilot(output):
    manifest = Path(__file__).resolve().parents[1] / 'evidence/pilot/manifest.json'
    if manifest.exists():
        root = manifest.parents[2]
        if Path(output).resolve() in {root / 'results/qwen3_1.7b_to_0.6b', root / 'results/qwen3_4b_to_0.6b'}:
            raise ValueError('This output is the immutable pilot. Use a new output directory.')


def config_identity(cfg):
    return {k: v for k, v in cfg.items() if k != 'output'}


def dataset_identity(dataset, name, revision, split):
    """Record resolved Arrow bytes and reject Datasets' offline 'latest cached' fallback drift."""
    files=getattr(dataset,'cache_files',[])
    if revision and len(revision)==40 and files and any(revision not in f['filename'] for f in files):
        raise ValueError(f'Resolved dataset cache does not match pinned revision {revision}')
    return dict(name=name,revision=revision,split=split,fingerprint=dataset._fingerprint,
                resolved_arrow_files=[artifact(f['filename']) for f in files])


def artifact(path):
    path = Path(path)
    return dict(sha256=file_sha256(path), bytes=path.stat().st_size)


def validate_array(path, descriptor):
    import numpy as np
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Missing cached file: {path}')
    try:
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        valid = (list(array.shape) == descriptor['shape'] and str(array.dtype) == descriptor['dtype']
                 and artifact(path) == {k: descriptor[k] for k in ('sha256', 'bytes')})
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f'Invalid cached file: {path}') from exc
    if not valid:
        raise ValueError(f'Cached shape, dtype, bytes or hash mismatch: {path}')


def array_descriptor(path):
    import numpy as np
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    return dict(shape=list(array.shape), dtype=str(array.dtype), **artifact(path))


def extraction_identity(backend, role, split, token_descriptor):
    c = backend.config
    return dict(schema=2, backend=backend_identity(backend), role=role, split=split,
        tokens=token_descriptor, positions='absolute 0..block_length-1; reset for each block',
        stored_dtype='float16', layers=c.num_hidden_layers,
        features=c.num_key_value_heads * c.head_dim, keys='post-RoPE', layout='positions, flattened heads')


def validate_extraction(directory, identity):
    directory = Path(directory)
    marker = directory / 'complete.json'
    if not marker.exists():
        raise ValueError(f'Incomplete extraction: {directory}')
    obj = json.loads(marker.read_text())
    if obj.get('identity') != identity or obj.get('identity_sha256') != digest(identity):
        raise ValueError('Cache extraction identity mismatch')
    expected = {f'{identity["split"]}_{identity["role"]}_{layer}_{kind}.npy'
                for layer in range(identity['layers']) for kind in ('k', 'v')}
    if set(obj.get('files', {})) != expected:
        raise ValueError('Incomplete cache marker file inventory')
    shape = [int(__import__('math').prod(identity['tokens']['shape'])), identity['features']]
    for name, desc in obj['files'].items():
        if desc['shape'] != shape or desc['dtype'] != identity['stored_dtype']:
            raise ValueError('Cache geometry disagrees with token/model identity')
        validate_array(directory / name, desc)
    return obj
