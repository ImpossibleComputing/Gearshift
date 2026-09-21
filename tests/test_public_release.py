"""Synthetic release adapters and upstream-integrity checks; no network or models."""
import hashlib
import json
from pathlib import Path
import pytest
from scripts import fetch_livecodebench as fetch
from scripts import reproduce_public as reproduce

def fixture(tmp_path):
    row={'platform':'synthetic','question_id':'1','question_content':'Synthetic placeholder, not a benchmark problem.'}
    line=json.dumps(row,sort_keys=True).encode()+b'\n'
    p=tmp_path/'test.jsonl';p.write_bytes(line)
    m={'selected':{'synthetic':[{'platform':'synthetic','question_id':'1','upstream_file':p.name,'source_row_sha256':hashlib.sha256(line).hexdigest()}]},'source_files':{p.name:{'rows':1}}}
    return p,m

def test_pinned_rows_identifiers_and_hashes(tmp_path):
    p,m=fixture(tmp_path)
    assert fetch.verify_rows(p,p.name,m)=={'rows':1,'selected_identifiers_verified':1}

def test_mutated_row_rejected(tmp_path):
    p,m=fixture(tmp_path);p.write_text(p.read_text().replace('placeholder','changed'))
    with pytest.raises(ValueError,match='hash mismatch'):fetch.verify_rows(p,p.name,m)

def test_missing_identifier_rejected(tmp_path):
    p,m=fixture(tmp_path);p.write_text(p.read_text().replace('"1"','"2"'))
    with pytest.raises(ValueError,match='Missing selected'):fetch.verify_rows(p,p.name,m)

def test_duplicate_identifier_rejected(tmp_path):
    p,m=fixture(tmp_path);p.write_bytes(p.read_bytes()*2)
    with pytest.raises(ValueError,match='Duplicate'):fetch.verify_rows(p,p.name,m)

def test_unexpected_upstream_url_rejected(tmp_path):
    p,m=fixture(tmp_path);m['dataset_revision']='a'*40;m['source_files'][p.name]['url']='https://example.invalid'
    with pytest.raises(ValueError,match='Unexpected upstream'):fetch.fetch_one(p.name,m,tmp_path)

def test_result_changes_not_silently_accepted():
    with pytest.raises(ValueError,match='differs'):reproduce.verify_subset({'passed':1},{'passed':0},'synthetic')

def test_csv_binary_and_missing_types(tmp_path):
    p=tmp_path/'rows.csv';p.write_text('passed,missing,seed_index,task_id\nTrue,False,0,synthetic/1\n,True,1,synthetic/2\n')
    r=reproduce.rows(p);assert r[0]['passed'] is True and r[1]['passed'] is None
    assert r[0]['seed_index']==0 and r[1]['missing'] is True

def test_frozen_public_fixture():
    p=fetch.ROOT/'results/phase2_v1/datasets/HumanEvalPlus-OriginFmt-v0.1.10.jsonl.gz'
    assert fetch.digest(p)=='daa7661c8189924068069b0872a440b491edb60f8bdf431d5957adc88d18bae5'
