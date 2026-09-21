import json
from pathlib import Path
import zipfile
import pytest
from scripts import sparse_repair_review as review


def fixture_root(tmp_path):
    for name in review.REQUIRED_REPORTS:(tmp_path/name).write_text('Actual results and limitations: test fixture only.\n')
    visible=tmp_path/'data/coding_pilot_v1/visible/development.json';visible.parent.mkdir(parents=True);visible.write_text('[]\n')
    cases=[]
    for i in range(16):
        relative=f'results/old/dev/{i}/source_history.json';p=tmp_path/relative;p.parent.mkdir(parents=True);p.write_text(json.dumps({'task_id':str(i),'prefix_ids':[1,2]}))
        cases.append({'task_id':str(i),'history_path':relative,'history_sha256':review.digest(p.read_bytes())})
    declaration={'calibration_tasks':cases[:4],'screen_tasks':cases[4:],
                 'inputs':{'public_development_inputs':{'sha256':review.digest(visible.read_bytes())}}}
    p=tmp_path/review.DECLARATION;p.parent.mkdir(parents=True);p.write_text(json.dumps(declaration))
    result=tmp_path/review.RESULT;result.mkdir(parents=True);(result/'artifact_audit.json').write_text('{"private_values_read":false}')
    return cases


def test_inventory_contains_only_selected_histories_and_skips_nestedarchives(tmp_path):
    cases=fixture_root(tmp_path)
    old=tmp_path/'results/old/not-selected/source_history.json';old.parent.mkdir(parents=True);old.write_text('DO NOT INCLUDE')
    (tmp_path/review.RESULT/'old_backup.tar.gz').write_bytes(b'not a nested archive')
    (tmp_path/review.RESULT/'runtime.pt').write_bytes(b'not a tensor')
    payload,coverage=review.collect(tmp_path,git_state={'HEAD':'test'})
    assert {x['history_path'] for x in cases}<=payload.keys()
    assert 'results/old/not-selected/source_history.json' not in payload
    assert len(coverage['exclusions'])==2
    assert not any(x.endswith(('.pt','.gz')) for x in payload)


def test_knowncredentials_rejected_but_attention_key_code_allowed():
    review.reject_secrets(b'query_key_scores = native_keys @ query.T; API_KEY=os.environ.get("API_KEY")','code.py')
    for secret in [('rpa_'+'A'*32),('hf_'+'x'*32),('ghp_'+'b'*40),('sk-proj-'+'c'*32)]:
        with pytest.raises(ValueError,match='intentionally not echoed') as e:
            review.reject_secrets(secret.encode(),'evidence.json')
        assert secret not in str(e.value)


def test_symlink_and_changed_saved_history_rejected(tmp_path):
    cases=fixture_root(tmp_path)
    p=tmp_path/cases[0]['history_path'];p.write_text('{}')
    with pytest.raises(ValueError,match='history changed'):review.collect(tmp_path,git_state={})
    p.unlink();p.symlink_to(tmp_path/review.REQUIRED_REPORTS[0])
    with pytest.raises(ValueError,match='Symlink'):review.collect(tmp_path,git_state={})


def test_build_verify_and_preserve_existing_zip(tmp_path,monkeypatch):
    fixture_root(tmp_path);monkeypatch.setattr(review,'git_identities',lambda root:{'publication_tag':review.PUBLICATION_TAG,'publication_commit':review.PUBLICATION_COMMIT,'HEAD':'test'})
    out=tmp_path/'review.zip';receipt=review.build(tmp_path,out)
    assert receipt['verified_temporary_extraction'] is True and receipt['scientific_completion_asserted_by_packaging'] is False
    assert receipt['sha256']==review.digest(out.read_bytes())
    with pytest.raises(FileExistsError):review.build(tmp_path,out)
    with zipfile.ZipFile(out) as z:
        manifest=json.loads(z.read(review.MANIFEST));assert all(x['sha256']==review.digest(z.read(x['path'])) for x in manifest['files'])


def test_archive_rejects_traversal_and_checksum_changes(tmp_path):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr(review.MANIFEST,'{"files":[]}');z.writestr('../escape','x')
    with pytest.raises(ValueError,match='Unsafe'):review.verify_archive(p)
    with zipfile.ZipFile(p,'w') as z:
        z.writestr(review.MANIFEST,json.dumps({'files':[{'path':'file.json','bytes':2,'sha256':'0'*64}]}));z.writestr('file.json','{}')
    with pytest.raises(ValueError,match='checksum differs'):review.verify_archive(p)


def test_missing_reports_prevent_delivery_build(tmp_path):
    fixture_root(tmp_path);(tmp_path/'SPARSE_REPAIR_RESULTS.md').unlink()
    with pytest.raises(ValueError,match='Required reports missing'):review.collect(tmp_path,require_complete_reports=True,git_state={})


def test_local_dependency_closure_reads_ast_without_import_execution(tmp_path):
    (tmp_path/'scripts').mkdir();(tmp_path/'gearshift').mkdir()
    (tmp_path/'scripts/sparse_repair_fake.py').write_text('from gearshift.helper import f\nraise RuntimeError("must never execute")\n')
    (tmp_path/'gearshift/helper.py').write_text('from .leaf import x\ndef f():return x\n')
    (tmp_path/'gearshift/leaf.py').write_text('x=1\n')
    assert review.dependency_closure(tmp_path,['scripts/sparse_repair_fake.py'])==['gearshift/helper.py','gearshift/leaf.py','scripts/sparse_repair_fake.py']


def test_optional_decision_review_commands_are_canonical_and_snapshot_separate(tmp_path):
    fixture_root(tmp_path)
    payload,_=review.collect(tmp_path,git_state={})
    commands=json.loads(payload['REPRODUCE_REPORT_COMMANDS.json'])
    assert commands['optional_decision_review_included'] is False
    assert '--decision-review' not in commands['commands'][1]
    original=b'{"measurements":"unchanged"}\n'
    input_path=tmp_path/review.REPORT_INPUT;input_path.parent.mkdir(parents=True);input_path.write_bytes(original)
    decision={'schema':'gearshift.sparse_repair.decision_review.v1',
        'evidence_input_sha256':review.digest(original),
        'declaration_sha256':review.digest((tmp_path/review.DECLARATION).read_bytes())}
    (tmp_path/review.DECISION_REVIEW).write_text(json.dumps(decision))
    payload,_=review.collect(tmp_path,git_state={})
    commands=json.loads(payload['REPRODUCE_REPORT_COMMANDS.json'])
    assert commands['commands'][1][-2:]==['--decision-review',review.DECISION_REVIEW]
    assert commands['working_directory']=='extracted_archive_root'
    assert commands['optional_decision_review_included'] is True
    assert commands['decision_review_sha256']==review.digest(payload[review.DECISION_REVIEW])
    assert payload[review.REPORT_INPUT]==original
    assert input_path.read_bytes()==original


@pytest.mark.parametrize('field',['evidence_input_sha256','declaration_sha256','schema'])
def test_optional_decision_review_stale_binding_rejected(tmp_path,field):
    fixture_root(tmp_path)
    p=tmp_path/review.REPORT_INPUT;p.parent.mkdir(parents=True);p.write_text('{}')
    decision={'schema':'gearshift.sparse_repair.decision_review.v1',
        'evidence_input_sha256':review.digest(p.read_bytes()),
        'declaration_sha256':review.digest((tmp_path/review.DECLARATION).read_bytes())}
    decision[field]='invalid'
    (tmp_path/review.DECISION_REVIEW).write_text(json.dumps(decision))
    with pytest.raises(ValueError,match='not bound'):
        review.collect(tmp_path,git_state={})


def test_total_guard_allows_complete_evidence_without_relaxing_member_guard(tmp_path,monkeypatch):
    assert review.MAX_TOTAL_BYTES==1024*1024**2
    assert review.MAX_MEMBER_BYTES==128*1024**2
    review.check_total_size(768*1024**2+1)
    review.check_total_size(1024*1024**2)
    with pytest.raises(ValueError,match='1024MiB'):
        review.check_total_size(1024*1024**2+1)
    p=tmp_path/'public.json';p.write_bytes(b'123456789')
    monkeypatch.setattr(review,'MAX_MEMBER_BYTES',8)
    with pytest.raises(ValueError,match='member exceeds'):
        review.read_public(tmp_path,'public.json')


def test_archive_verifier_rejects_total_size_before_loading_manifest(tmp_path,monkeypatch):
    p=tmp_path/'oversize.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr(review.MANIFEST,'not valid JSON but already over total guard')
    monkeypatch.setattr(review,'MAX_TOTAL_BYTES',8)
    with pytest.raises(ValueError,match='1024MiB'):
        review.verify_archive(p)


def test_frozen_private_storage_public_binding_and_provider_receipt_are_included(tmp_path):
    fixture_root(tmp_path)
    receipt='results/sparse_repair_01/resources/private_storage_fallback/created_volume.json'
    p=tmp_path/receipt;p.parent.mkdir(parents=True);p.write_text('{"id":"synthetic-volume"}')
    config=tmp_path/review.STORAGE_BINDING
    config.write_text(json.dumps({'provider_volume_receipt':{'path':receipt,'sha256':review.digest(p.read_bytes())}}))
    payload,_=review.collect(tmp_path,git_state={})
    assert payload[review.STORAGE_BINDING]==config.read_bytes()
    assert payload[receipt]==p.read_bytes()
    p.write_text('{"id":"changed"}')
    with pytest.raises(ValueError,match='provider receipt differs'):
        review.collect(tmp_path,git_state={})
    p.unlink()
    with pytest.raises(ValueError,match='Missing/unscoped'):
        review.collect(tmp_path,git_state={})
