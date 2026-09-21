from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from .core import CacheExtractor, save_json, timed
from .identity import (digest, tokenizer_identity, bind, array_descriptor, validate_array, dataset_identity,
                       extraction_identity, validate_extraction)


def token_identity(cfg, tokenizer, corpus, split):
    base = dict(schema=2, dataset='Salesforce/wikitext', config='wikitext-2-raw-v1',
        revision=cfg.get('wikitext_revision'), split=split, fingerprint=corpus[split]._fingerprint,
        resolved_dataset=dataset_identity(corpus[split],'Salesforce/wikitext',cfg.get('wikitext_revision'),split),
        tokenizer=tokenizer_identity(tokenizer), seed=cfg['seed'], selection='seeded nonoverlapping blocks per split v2')
    if split == 'test':
        base.update(lengths=cfg['lengths'], continuation_tokens=cfg['continuation_tokens'], count=cfg['eval_examples'])
    else:
        base.update(block_size=cfg['block_size'], count=cfg[split+'_tokens'])
    return base


def run_cache_root(cfg, identities):
    namespace = digest(str(Path(cfg['output']).resolve()))
    return Path(cfg.get('cache_store', 'data/v2')) / 'runs' / namespace / digest(identities)


def link_file(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise ValueError(f'Unexpected cache path already exists: {destination}')
    destination.symlink_to(source)


def selected_token_descriptor(root, split):
    meta=json.loads((Path(root)/'tokens.json').read_text())
    selected=meta['splits'][split]
    path=Path(root)/f'{split}_tokens.npy'
    validate_array(path,selected['array'])
    return dict(selected['array'],selection_identity=digest(selected))


def prepare_tokens(cfg, tokenizer):
    corpus = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', revision=cfg.get('wikitext_revision'))
    identities = {split: token_identity(cfg, tokenizer, corpus, split) for split in ['train','validation','test']}
    root = run_cache_root(cfg, identities)
    store = Path(cfg.get('cache_store', 'data/v2'))
    prepared = {}
    for split, identity in identities.items():
        dest = store / 'tokens' / digest(identity)
        marker = dest / 'complete.json'; path = dest / f'{split}_tokens.npy'
        if marker.exists():
            obj = json.loads(marker.read_text())
            if obj.get('identity') != identity or obj.get('identity_sha256') != digest(identity):
                raise ValueError('Token identity mismatch')
            validate_array(path, obj['array'])
            n=max(cfg['lengths'])+cfg['continuation_tokens']+1 if split=='test' else cfg['block_size']
            count=cfg['eval_examples'] if split=='test' else cfg[split+'_tokens']//n
            indices=obj.get('selected_block_indices',[])
            if (obj['array']['shape']!=[count,n] or obj['array']['dtype']!='int32' or
                len(indices)!=count or len(set(indices))!=count or any(i<0 for i in indices)):
                raise ValueError('Token shape or selected block indices disagree with selection identity')
            if hashlib.sha256(np.load(path,allow_pickle=False).tobytes()).hexdigest()!=obj['selected_token_sha256']:
                raise ValueError('Selected token hash mismatch')
        else:
            if dest.exists() and any(dest.iterdir()):
                raise ValueError(f'Incomplete token selection at {dest}; recover explicitly into a new cache store')
            dest.mkdir(parents=True, exist_ok=True)
            text = '\n\n'.join(x for x in corpus[split]['text'] if x.strip())
            tokens = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
            n = max(cfg['lengths'])+cfg['continuation_tokens']+1 if split == 'test' else cfg['block_size']
            count = cfg['eval_examples'] if split == 'test' else cfg[split+'_tokens']//n
            if count < 1 or count > len(tokens)//n:
                raise ValueError('Requested token budget cannot be sampled without replacement')
            rng = np.random.default_rng(cfg['seed'] + ['train','validation','test'].index(split))
            indices = rng.permutation(len(tokens)//n)[:count]
            blocks = np.stack([tokens[i*n:(i+1)*n] for i in indices])
            np.save(path, blocks)
            obj = bind(marker, identity, array=array_descriptor(path), selected_block_indices=indices.tolist(),
                text_sha256=hashlib.sha256(text.encode()).hexdigest(), selected_token_sha256=hashlib.sha256(blocks.tobytes()).hexdigest())
        prepared[split] = (path, obj)
    # No writes to the run view until every upstream artifact validates.
    root.mkdir(parents=True, exist_ok=True)
    for split, (path, obj) in prepared.items():
        link_file(path, root / f'{split}_tokens.npy')
    meta = bind(root/'tokens.json', identities, splits={split: obj for split, (_, obj) in prepared.items()})
    bind(Path(cfg['output'])/'data_manifest.json', identities, splits=meta['splits'])
    return root


@torch.inference_mode()
def extract_pairs(cfg, source, target, root):
    metadata = {}
    for split in ['train', 'validation']:
        token_desc = selected_token_descriptor(root, split)
        ids = np.load(root / f'{split}_tokens.npy', mmap_mode='r')
        for role, backend in [('source', source), ('target', target)]:
            identity = extraction_identity(backend, role, split, token_desc)
            dest = Path(cfg.get('cache_store', 'data/v2')) / 'extracted' / digest(identity)
            marker = dest / 'complete.json'
            if marker.exists():
                obj = validate_extraction(dest, identity)
                print(f'Validated/reusing {split} {role} caches', flush=True)
            else:
                if dest.exists() and any(dest.iterdir()):
                    raise ValueError(f'Incomplete extraction at {dest}; recover explicitly, marker alone never authorizes reuse')
                dest.mkdir(parents=True, exist_ok=True)
                shape = (ids.size, identity['features'])
                maps = {(layer, kind): np.lib.format.open_memmap(dest/f'{split}_{role}_{layer}_{kind}.npy',
                        mode='w+', dtype=np.float16, shape=shape)
                        for layer in range(identity['layers']) for kind in ['k','v']}
                times = []
                for i, row in enumerate(ids):
                    out, ms = timed(lambda: backend.prefill(row.tolist()), backend.device)
                    for layer, pair in enumerate(CacheExtractor.tensors(out.past_key_values)):
                        for kind, x in zip(['k','v'], pair):
                            maps[layer,kind][i*len(row):(i+1)*len(row)] = CacheExtractor.flatten(x).cpu().numpy()
                    times.append(ms)
                    del out
                    print(f'extract {split} {role}: {i+1}/{len(ids)}', flush=True)
                for m in maps.values(): m.flush()
                del maps
                files = {p.name: array_descriptor(p) for p in dest.glob('*.npy')}
                obj = bind(marker, identity, files=files, prefill_ms=times)
                validate_extraction(dest, identity)
            for name in obj['files']:
                link_file(dest/name, root/name)
            metadata[split+'_'+role] = obj
    bind(root/'cache_manifest.json', {k: v['identity'] for k,v in metadata.items()}, extractions=metadata)
    bind(Path(cfg['output'])/'cache_manifest.json', {k: v['identity'] for k,v in metadata.items()}, extractions=metadata)


def validate_training_caches(cfg, source, target, root):
    for split in ['train','validation']:
        for role, backend in [('source',source),('target',target)]:
            identity = extraction_identity(backend, role, split, selected_token_descriptor(root,split))
            directory = Path(cfg.get('cache_store','data/v2'))/'extracted'/digest(identity)
            obj = validate_extraction(directory, identity)
            for name in obj['files']:
                if (root/name).resolve() != (directory/name).resolve():
                    raise ValueError('Cache view points to another extraction')
    return json.loads((root/'cache_manifest.json').read_text())['identity_sha256']

def read_features(root, split, role, layer, kind, indices=None, backend=None, content=False):
    arr = np.load(root/f'{split}_{role}_{layer}_{kind}.npy', mmap_mode='r')
    if indices is None:
        indices = np.arange(len(arr))
    x = torch.from_numpy(np.array(arr[indices], dtype=np.float32))
    if content:
        block_size = np.load(root/f'{split}_tokens.npy',mmap_mode='r').shape[1]
        pos = torch.from_numpy(np.asarray(indices) % block_size).to(backend.device)
        k = CacheExtractor.unflatten(x.to(backend.device),backend.config.num_key_value_heads,backend.config.head_dim)
        x = CacheExtractor.flatten(backend.rope(k,pos,inverse=True)).cpu()
    return x
