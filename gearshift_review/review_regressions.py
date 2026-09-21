"""Run source-level controls without Hugging Face or model downloads.

Original function/class ASTs execute unchanged; only unavailable external I/O or
model-loading endpoints are replaced in explicitly marked resume fixtures.
This does not run model inference, DynamicCache, or the full original test suite.
"""
from pathlib import Path
from decimal import Decimal, InvalidOperation
from types import SimpleNamespace
from unittest.mock import patch
import ast, hashlib, json, math, os, random, re, tempfile
import numpy as np
import pandas as pd
import torch

ROOT=Path('/mnt/data/gearshift_review/gearshift')
OUT=Path('/mnt/data/gearshift_audit')
ns=dict(np=np,pd=pd,torch=torch,json=json,Path=Path,hashlib=hashlib,random=random,re=re,Decimal=Decimal,InvalidOperation=InvalidOperation)

def load_defs(rel,names,space=ns):
    path=ROOT/rel
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected}==set(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),space)

load_defs('gearshift/core.py',['save_json','file_sha256','seed_all','CacheExtractor','rotate_half','distributions','tensor_metrics'])
load_defs('gearshift/reasoning.py',['numbers','numeric_answer','benchmark'])
load_defs('gearshift/data.py',['prepare_tokens'])
load_defs('gearshift/runner.py',['ExperimentRunner'])
load_defs('tests/test_core.py',['test_rotation_inverse_for_different_positions','test_distribution_metrics_known_values','test_constant_mean_baseline_has_zero_r2','test_answer_grading'])
ns['math']=math
results={'scope':__doc__,'original_pure_unit_tests':[],'regressions':[]}
for test in ['test_rotation_inverse_for_different_positions','test_distribution_metrics_known_values','test_constant_mean_baseline_has_zero_r2','test_answer_grading']:
    ns[test](); results['original_pure_unit_tests'].append({'name':test,'result':'PASS'})

cfg=json.loads((ROOT/'configs/qwen3_1.7b_to_0.6b.json').read_text())
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    # Pure config constructor: no model instantiated. Each scenario starts with original settings.
    admitted=[]
    changes={'gsm8k_revision':'different-dataset-revision','wikitext_revision':'different-text-revision','eval_examples':16,'continuation_tokens':128,'lengths':[128,512,1024,2048,8192]}
    for key,value in changes.items():
        out=root/key
        base={**cfg,'output':str(out)}
        ns['ExperimentRunner'](base)
        try: ns['ExperimentRunner']({**base,key:value});admitted.append(key)
        except ValueError: pass
    results['regressions'].append({'name':'configuration_identity_gaps','changed_fields_admitted_without_rejection':admitted})

    # The original prepare_tokens returns early on an existing marker: tokenizer/load_dataset must not execute.
    cwd=os.getcwd()
    try:
        os.chdir(root)
        saved=Path('data/same-name');saved.mkdir(parents=True);(saved/'tokens.json').write_text('{}')
        a=ns['prepare_tokens']({**cfg,'output':'results/experiment-A/same-name'},None)
        b=ns['prepare_tokens']({**cfg,'output':'results/experiment-B/same-name'},None)
        results['regressions'].append({'name':'output_basename_collision','data_roots_equal':a==b,'data_root':str(a),'metadata_validated_before_reuse':False})
    finally: os.chdir(cwd)

    # Resume fixture: original 24-example saved answers, asked to rerun 12. Stub only model/dataset loading.
    out=root/'resume-test';out.mkdir()
    raw=json.loads((ROOT/'results/qwen3_1.7b_to_0.6b/reasoning.json').read_text())
    manifest=json.loads((ROOT/'results/qwen3_1.7b_to_0.6b/reasoning_manifest.json').read_text())
    (out/'mappers.pt').write_bytes(b'fixture hash; this is not a model checkpoint')
    manifest['mapper_sha256']=ns['file_sha256'](out/'mappers.pt')
    (out/'reasoning.json').write_text(json.dumps(raw));(out/'reasoning_manifest.json').write_text(json.dumps(manifest))
    class DatasetFixture:
        _fingerprint='fixture'
        def __len__(self): return 1319
        def __getitem__(self,i): raise AssertionError('No inference should be needed: all requested indices already have records')
    ns['load_dataset']=lambda *a,**k:DatasetFixture()
    ns['CacheAdapter']=SimpleNamespace(load=lambda *a,**k:SimpleNamespace(maps={'functional_k':[]}))
    ns['benchmark']({**cfg,'output':str(out),'gsm_examples':12},SimpleNamespace(),SimpleNamespace(eos={151643,151645}))
    after=json.loads((out/'reasoning_manifest.json').read_text());still=json.loads((out/'reasoning.json').read_text())
    results['regressions'].append({'name':'resume_sample_size_change','requested_questions':12,'manifest_questions':len(after['selected_indices']),'saved_questions':len({r['dataset_index'] for r in still}),'saved_answer_rows':len(still),'rejected':False,'issue_reproduced':len(after['selected_indices'])!=len({r['dataset_index'] for r in still})})

(OUT/'regression_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
