import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gearshift.identity import (bind, digest, validate_records, array_descriptor,
                                extraction_identity, validate_extraction)
from gearshift.data import prepare_tokens, token_identity, run_cache_root
from gearshift.runner import ExperimentRunner


class Tokenizer:
    name_or_path='fixture'
    init_kwargs={'revision':'tokenizer-revision'}
    chat_template='template'
    special_tokens_map={}
    backend_tokenizer=SimpleNamespace(to_str=lambda:'fixture backend')
    def encode(self, text, **kw):
        return list(range(1000)) if text.startswith('training fixture') else [ord(c) for c in text]
    def apply_chat_template(self, messages, **kw): return [1, len(messages[0]['content']), 3]
    def convert_tokens_to_ids(self, token): return 7


class Split(dict):
    _fingerprint='fingerprint-1'


def config(tmp_path):
    return dict(output=str(tmp_path/'out'), cache_store=str(tmp_path/'store'), seed=42,
        source='source', target='target', wikitext_revision='rev-1', gsm8k_revision='gsm-1',
        block_size=16, train_tokens=64, validation_tokens=32, lengths=[8,16],
        continuation_tokens=4, eval_examples=2, gsm_examples=24, reasoning_tokens=8, answer_tokens=4)


@pytest.mark.parametrize('key,value', [('gsm_examples',12),('gsm_examples',48),
    ('gsm8k_revision','changed'),('wikitext_revision','changed'),('lengths',[8,32]),
    ('continuation_tokens',8),('eval_examples',3),('answer_delimiter','new delimiter'),
    ('effective_eos_ids',[1,2])])
def test_runner_rejects_changed_config_without_writes(tmp_path,key,value):
    cfg=config(tmp_path); runner=ExperimentRunner(cfg)
    before={p.name:p.read_bytes() for p in runner.out.iterdir()}
    with pytest.raises(ValueError,match='Configuration differs'):
        ExperimentRunner({**cfg,key:value})
    assert before=={p.name:p.read_bytes() for p in runner.out.iterdir()}


@pytest.mark.parametrize('old,new', [(list(range(24)),list(range(12))),
    (list(range(12)),list(range(24))), (list(range(12)),list(range(1,13)))])
def test_sample_identity_changes_fail_before_write(tmp_path,old,new):
    path=tmp_path/'manifest.json'; bind(path,{'ids':old},audit={'original_eos':[1],'replay':'separate'})
    before=path.read_bytes()
    with pytest.raises(ValueError,match='identity mismatch'): bind(path,{'ids':new})
    assert path.read_bytes()==before


def test_noop_preserves_manifest_and_partial_state(tmp_path):
    p=tmp_path/'manifest.json'; bind(p,{'ids':[1,2]},audit={'live':1,'replay':2})
    before=p.read_bytes(); stamp=p.stat().st_mtime_ns
    bind(p,{'ids':[1,2]},audit={'live':999})
    assert p.read_bytes()==before and p.stat().st_mtime_ns==stamp
    rows=[{'dataset_index':1,'condition':'C'}]
    assert validate_records(rows,[1,2],['C'])['state']=='partial'
    with pytest.raises(ValueError,match='Incomplete'): validate_records(rows,[1,2],['C'],complete=True)
    with pytest.raises(ValueError,match='Duplicate'): validate_records(rows*2,[1,2],['C'])
    with pytest.raises(ValueError,match='outside'): validate_records(rows,[2],['C'])


def test_token_selection_identity_and_shared_training(tmp_path,monkeypatch):
    corpus={s:Split(text=['training fixture text']) for s in ['train','validation','test']}
    monkeypatch.setattr('gearshift.data.load_dataset',lambda *a,**k:corpus)
    cfg=config(tmp_path); tok=Tokenizer()
    first=prepare_tokens(cfg,tok)
    # Identical basenames are separate views; actual content-addressed training may safely be shared.
    second=prepare_tokens({**cfg,'output':str(tmp_path/'different'/'out')},tok)
    assert first!=second
    assert (first/'train_tokens.npy').resolve()==(second/'train_tokens.npy').resolve()
    changed=prepare_tokens({**cfg,'output':str(tmp_path/'eval2'),'eval_examples':3,'lengths':[16,32], 'continuation_tokens':8},tok)
    assert (changed/'train_tokens.npy').resolve()==(first/'train_tokens.npy').resolve()
    assert (changed/'test_tokens.npy').resolve()!=(first/'test_tokens.npy').resolve()
    train_id=token_identity(cfg,tok,corpus,'train')
    assert train_id!=token_identity({**cfg,'wikitext_revision':'r2'},tok,corpus,'train')
    corpus['train']._fingerprint='new'
    assert train_id!=token_identity(cfg,tok,corpus,'train')
    (first/'test_tokens.npy').resolve().unlink()
    with pytest.raises(ValueError,match='Missing cached'): prepare_tokens(cfg,tok)


@pytest.mark.parametrize('damage',['missing','truncated','wrong_shape','wrong_hash','wrong_role','wrong_positions'])
def test_completion_marker_is_not_proof(tmp_path,damage):
    identity=dict(schema=2,role='target',split='train',layers=1,features=4,
                  tokens={'shape':[2,3]},stored_dtype='float16',positions='absolute')
    files={}
    for kind in ['k','v']:
        p=tmp_path/f'train_target_0_{kind}.npy'; np.save(p,np.zeros((6,4),dtype=np.float16))
        files[p.name]=array_descriptor(p)
    bind(tmp_path/'complete.json',identity,files=files)
    validate_extraction(tmp_path,identity)
    if damage=='missing': p.unlink()
    elif damage=='truncated': p.write_bytes(p.read_bytes()[:100])
    elif damage=='wrong_shape': np.save(p,np.zeros((5,4),dtype=np.float16))
    elif damage=='wrong_hash': np.save(p,np.ones((6,4),dtype=np.float16))
    elif damage=='wrong_role': identity['role']='source'
    else: identity['positions']='reset suffix to zero'
    with pytest.raises(ValueError): validate_extraction(tmp_path,identity)


@pytest.mark.parametrize('change',['12','48','ids','fingerprint','eos','delimiter'])
def test_actual_benchmark_resume_entrypoint(tmp_path,monkeypatch,change):
    from gearshift.reasoning import benchmark
    cfg=config(tmp_path); out=Path(cfg['output']); out.mkdir(); (out/'mappers.pt').write_bytes(b'fixture')
    class Dataset:
        _fingerprint='fp'
        def __len__(self): return 100
        def __getitem__(self,i): return {'question':f'question {i}', 'answer':'#### 1'}
    ds=Dataset()
    monkeypatch.setattr('gearshift.reasoning.load_dataset',lambda *a,**kw:ds)
    monkeypatch.setattr('gearshift.reasoning.CacheAdapter.load',lambda *a:SimpleNamespace(maps={}))
    monkeypatch.setattr('gearshift.reasoning.backend_identity',lambda b:dict(eos=sorted(b.eos)))
    backend=SimpleNamespace(tokenizer=Tokenizer(),eos={9})
    # Stop exactly after identity creation, before any model generation.
    monkeypatch.setattr('gearshift.reasoning.think',lambda *a:(_ for _ in ()).throw(RuntimeError('inference sentinel')))
    with pytest.raises(RuntimeError,match='sentinel'): benchmark(cfg,backend,backend)
    manifest=out/'reasoning_manifest.json'; obj=json.loads(manifest.read_text()); obj['audit']={'original':True,'replay':'untouched'}
    manifest.write_text(json.dumps(obj,indent=4))
    ids=obj['identity']['selected_indices']; conditions=obj['identity']['conditions']
    rows=[dict(dataset_index=i,condition=c) for i in ids for c in conditions]
    (out/'reasoning.json').write_text(json.dumps(rows))
    before=manifest.read_bytes()
    benchmark(cfg,backend,backend)  # true no-op; would fail if inference happened
    assert manifest.read_bytes()==before
    if change in ['12','48']: cfg['gsm_examples']=int(change)
    elif change=='ids': cfg['gsm_indices']=list(reversed(ids))
    elif change=='fingerprint': ds._fingerprint='new-fp'
    elif change=='eos': backend.eos={9,10}
    else: cfg['answer_delimiter']='custom delimiter'
    with pytest.raises(ValueError): benchmark(cfg,backend,backend)
    assert manifest.read_bytes()==before
    assert json.loads((out/'reasoning.json').read_text())==rows


def test_offline_latest_cache_must_match_requested_revision(tmp_path):
    from gearshift.identity import dataset_identity
    revision='a'*40; other='b'*40
    directory=tmp_path/revision;directory.mkdir();p=directory/'data.arrow';p.write_bytes(b'fixture')
    ds=SimpleNamespace(_fingerprint='same-fingerprint',cache_files=[{'filename':str(p)}])
    assert dataset_identity(ds,'fixture',revision,'train')['resolved_arrow_files'][0]['bytes']==7
    with pytest.raises(ValueError,match='pinned revision'):dataset_identity(ds,'fixture',other,'train')


def test_selection_revision_is_part_of_extraction_identity():
    # Equal token bytes from differently identified upstream selections are not silently relabeled.
    a={'shape':[2,3],'dtype':'int32','bytes':152,'sha256':'same-token-bytes','selection_identity':'revision-a'}
    b={**a,'selection_identity':'revision-b'}
    assert digest(a)!=digest(b)


def test_mapper_without_provenance_cannot_resume(tmp_path,monkeypatch):
    from gearshift.mapping import train_mappers
    monkeypatch.setattr('gearshift.data.validate_training_caches',lambda *args:'validated-cache')
    (tmp_path/'mappers.pt').write_bytes(b'unidentified fixture')
    with pytest.raises(ValueError,match='no validated training provenance'):
        train_mappers({'output':str(tmp_path)},None,None,tmp_path)
    assert not (tmp_path/'mapper_training_manifest.json').exists()
