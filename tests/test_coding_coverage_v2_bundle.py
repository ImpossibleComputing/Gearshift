import copy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import coding_coverage_v2_bundle as bundle


def fixture(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; repo.mkdir(); relative = 'results/coding_pilot_v1/v2_fixture'; root = repo/relative
    erel = 'evidence/coding_pilot_v1/v2_fixture'; evidence = repo/erel; evidence.mkdir(parents=True)
    def put(relative_path, value):
        path = repo/relative_path; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(bundle.encoded(value) if isinstance(value,(dict,list)) else value.encode())
        return path
    def info(path): return {'sha256':bundle.sha(path),'bytes':path.stat().st_size}
    public = {}
    def inp(relative_path, value):
        path = put(relative_path,value); public[relative_path] = info(path); return path
    inp('README.md','# Scientific project\n'); inp('requirements.lock.txt','numpy==2.2.6\nmatplotlib==3.11.2\n')
    selected = 'results/coding_pilot_v1/selected/mapper.pt'; start = inp(selected,'starting mapper')
    history = 'results/coding_pilot_v1/history_recovery_20260916_01/worker/task/source_history.json'
    answer = history.replace('source_history','teacher_answer')
    hp = put(history,{'reasoning':'saved source trajectory'}); ap = put(answer,{'answer_ids':[1,2]})
    corpus = {'files':{history:info(hp),answer:info(ap),'paired_cache.pt':{'sha256':'excluded','bytes':999}}}
    corpus_path = 'results/coding_pilot_v1/corpus/corpus_manifest.json'; cp = inp(corpus_path,corpus)
    declaration_path = 'configs/coding_pilot_v1/coverage_generalization_v2/declaration.json'
    declaration = {'selected_checkpoint':selected,'selected_checkpoint_sha256':bundle.sha(start),
                   'corpus_manifest':corpus_path,'corpus_manifest_sha256':bundle.sha(cp)}
    dp = inp(declaration_path,declaration)
    inp('configs/coding_pilot_v1/reference/source/model.safetensors.index.json',{'metadata_only':True})
    score = put(relative+'/evaluation/scored_answer_manifest.json',{'files':[]})
    put(relative+'/execution_complete.json',{'experiment_id':'v2_fixture','full_1024_1024_training_completed':True,
        'all_jobs_generated_and_scored':True,'scored_answer_manifest_sha256':bundle.sha(score)})
    put(relative+'/evaluation/evaluation_plan.json',{'experiment_id':'v2_fixture'})
    summary = {'full_1024_per_arm_completed':True,'primary_step':1024,'fixture_value':4}
    report = relative+'/evaluation/report'
    put(report+'/summary.json',summary); put(report+'/COVERAGE_GENERALIZATION_V2_RESULTS.md','Fixture result: 4\n')
    put(report+'/plot.svg','<svg xmlns="http://www.w3.org/2000/svg"></svg>\n')
    inp('scripts/coding_coverage_v2_report.py', '''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo-root');p.add_argument('--result-root');p.add_argument('--plan');p.add_argument('--output');a=p.parse_args()
r=Path(a.repo_root);assert not list(r.rglob('*.pt'));assert not any(x.is_dir() and x.name=='private' for x in r.rglob('*'))
o=Path(a.output);o.mkdir(parents=True);obj={'full_1024_per_arm_completed':True,'primary_step':1024,'fixture_value':2+2}
(o/'summary.json').write_text(json.dumps(obj,indent=2,sort_keys=True)+'\\n')
(o/'COVERAGE_GENERALIZATION_V2_RESULTS.md').write_text('Fixture result: 4\\n')
(o/'plot.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>\\n')
''')
    for arm in ('FIXED','ROTATING'):
        ar = relative+'/arms/'+arm
        put(ar+'/training_complete.json',{'full_target_completed':True,'completed_updates':1024,'scored_positions':32768})
        put(ar+'/resume_preflight.json',{'passed':True}); checkpoints = []
        for step in (0,128,256,512,768,1024):
            folder = ar+f'/checkpoints/step_{step:04d}'; files = {}
            record = {'step':step}
            for key in ('mapper','full'):
                name = key+'.pt'; path = put(folder+'/'+name,arm+str(step)+key); files[name] = info(path)
                record[key] = {'path':'/workspace/GearshiftV2/'+folder+'/'+name,**files[name]}
            put(folder+'/manifest.json',{'step':step,'arm':arm,'complete_resumable':True,'verified_roundtrip':True,'files':files})
            checkpoints.append(record)
        put(ar+'/checkpoint_manifest.json',{'checkpoints':checkpoints})
    put(relative+'/private/hidden_tests.json',{'secret':'must be excluded'})
    put(relative+'/cache/transient.json',{'data':'reproducible'})
    put(relative+'/model.safetensors','model weights')
    put(relative+'/attempts/failure.json',{'error':'preserved engineering attempt'})
    put(relative+'/memory_telemetry.jsonl',json.dumps({'allocated_bytes':123})+'\n')
    put(erel+'/review_cost_snapshot.json',{'upper_usd':500.25,'gpu_hours':80.})
    put(erel+'/live_resources.json',{'pods':[],'volumes':['durable-volume']})
    put(erel+'/provider_auth.json',{'api_key':'must-not-copy'})
    put('gearshift_previous_review.zip','previous archive bytes')
    plan = {'experiment_id':'v2_fixture','result_root':relative,'execution_role':'primary','code_commit':'frozen-source-commit',
            'input_manifest':public,'declaration_path':declaration_path,'declaration_sha256':bundle.sha(dp)}
    pp = put(erel+'/dispatch_primary.json',plan)
    monkeypatch.setattr(bundle,'git_proof',lambda _: {'publication_commit':bundle.PUBLICATION,'delivery_commit':'delivery-commit','branch':'research/coverage-v2'})
    return repo,root,evidence,pp


def test_bundle_is_compact_verified_reproducible_and_preserves_prior_archive(tmp_path,monkeypatch):
    repo,root,evidence,plan = fixture(tmp_path,monkeypatch)
    before = bundle.sha(repo/'gearshift_previous_review.zip')
    result = bundle.build(repo,root,evidence,execution_plan=plan)
    assert result['compact_regeneration_passed'] and result['full_1024_1024_training_completed']
    path = Path(result['path']); manifest = bundle.verify_archive(path)
    names = set(manifest['files'])
    assert not any(Path(n).suffix in bundle.HEAVY_SUFFIXES or 'private' in Path(n).parts for n in names)
    assert any(n.endswith('source_history.json') for n in names)
    assert any(n.endswith('teacher_answer.json') for n in names)
    assert any(n.endswith('model.safetensors.index.json') for n in names)
    assert any(n.endswith('memory_telemetry.jsonl') for n in names)
    assert any(n.endswith('failure.json') for n in names)
    assert not any(n.endswith('provider_auth.json') for n in names)
    inventory_name = next(n for n in names if n.endswith('excluded_heavy_artifacts.json'))
    with zipfile.ZipFile(path) as archive:
        inventory = json.loads(archive.read(bundle.PREFIX+inventory_name))
        assert len(inventory['files']) == 25
        assert all(r['locally_hash_verified'] for r in inventory['files'])
        assert all('sha256' in r and 'bytes' in r and 'retrieval_path' in r for r in inventory['files'])
    assert bundle.sha(repo/'gearshift_previous_review.zip') == before
    with pytest.raises(ValueError,match='already exists'):
        bundle.build(repo,root,evidence,execution_plan=plan)


@pytest.mark.parametrize('change',['unfinished','missing_checkpoint','changed_public_input','credential'])
def test_bundle_fails_closed_on_incomplete_or_changed_evidence(tmp_path,monkeypatch,change):
    repo,root,evidence,plan = fixture(tmp_path,monkeypatch)
    if change == 'unfinished': (root/'execution_complete.json').unlink()
    elif change == 'missing_checkpoint': (root/'arms/FIXED/checkpoints/step_0128/manifest.json').unlink()
    elif change == 'changed_public_input': (repo/'README.md').write_text('changed after dispatch')
    else: (evidence/'unsafe.json').write_text(json.dumps({'authorization':'actual-secret-value'}))
    with pytest.raises(ValueError):
        bundle.build(repo,root,evidence,execution_plan=plan)
    assert not (repo/'gearshift_coverage_generalization_v2_review.zip').exists()


def test_archive_verifier_rejects_an_unlisted_private_member(tmp_path,monkeypatch):
    repo,root,evidence,plan = fixture(tmp_path,monkeypatch)
    result = bundle.build(repo,root,evidence,execution_plan=plan)
    with zipfile.ZipFile(result['path'],'a') as archive:
        archive.writestr('gearshift/data/private/hidden.json','{}')
    with pytest.raises(ValueError,match='membership'):
        bundle.verify_archive(result['path'])


def test_credential_detector_finds_realistic_tokens_but_not_regex_source():
    bundle.credential_scan('script.py',b"PATTERN = rb'hf_[A-Za-z0-9]{30,}'")
    with pytest.raises(ValueError,match='Credential signature'):
        bundle.credential_scan('log.txt',b'hf_'+b'a'*40)


def test_credential_scan_distinguishes_pinned_tokenizer_vocabulary_from_secrets():
    tokens = {'model':{'vocab':{'password':10,'authorization':11,'api_key':12}}}
    bundle.credential_scan('configs/reference/tokenizer.json',bundle.encoded(tokens))
    tokens['model']['vocab']['password'] = 'real-secret'
    with pytest.raises(ValueError,match='credential field'):
        bundle.credential_scan('configs/reference/tokenizer.json',bundle.encoded(tokens))
    with pytest.raises(ValueError,match='credential field'):
        bundle.credential_scan('configs/reference/tokenizer.json',bundle.encoded({'password':12345}))


def test_credential_scan_accepts_only_the_explicit_scientific_approval_schema():
    authorization = {'execution_allowed':False,'previous_phase_authorization_applies':False,
                     'owner_approval_required':True,'approved_spending_cap_usd':None}
    bundle.credential_scan('pilot.json',bundle.encoded({'authorization':authorization}))
    authorization['api_key'] = 'actual-secret'
    with pytest.raises(ValueError,match='credential field'):
        bundle.credential_scan('pilot.json',bundle.encoded({'authorization':authorization}))
    with pytest.raises(ValueError,match='credential field'):
        bundle.credential_scan('pilot.json',bundle.encoded({'authorization':{'Bearer':'actual-secret'}}))


def test_budget_approval_reference_is_metadata_only_in_monitoring_receipts():
    name = 'evidence/run/monitoring/status_20260919T060048Z.json'
    reference = 'budget/owner_amendment_20260919T032843Z.json'
    value = {'budget':{'authorization':reference,'cumulative_cap_usd':2500}}
    bundle.credential_scan(name,bundle.encoded(value))
    for filename, content in [
        ('pilot.json', value),
        (name, {'authorization':reference}),
        (name, {'budget':{'authorization':'Bearer actual-secret'}}),
        (name, {'budget':{'authorization':'../'+reference}}),
        (name, {'budget':{'authorization':reference+' actual-secret'}}),
        (name, {'budget':{'authorization':reference,'api_key':'actual-secret'}}),
        (name, {'budget':{'authorization':{'record':reference,'api_key':'actual-secret'}}}),
    ]:
        with pytest.raises(ValueError,match='credential field'):
            bundle.credential_scan(filename,bundle.encoded(content))


@pytest.mark.parametrize('mode',['valid','changed','linked'])
def test_executed_input_overlay_is_hash_bound_and_preserves_historical_input(tmp_path,monkeypatch,mode):
    repo,root,evidence,plan = fixture(tmp_path,monkeypatch)
    name = 'configs/coding_pilot_v1/reference/source/model.safetensors.index.json'
    historical = repo/name
    executed = historical.read_bytes()
    historical.write_text('{"historical_version":"preserved"}\n')
    historical_before = historical.read_bytes()
    overlay = evidence/'executed_inputs'
    replacement = overlay/name; replacement.parent.mkdir(parents=True)
    if mode == 'linked':
        target = tmp_path/'outside.json'; target.write_bytes(executed)
        replacement.symlink_to(target)
    else:
        replacement.write_bytes(executed if mode == 'valid' else b'{"changed":true}\n')
    if mode != 'valid':
        with pytest.raises(ValueError):
            bundle.build(repo,root,evidence,execution_plan=plan,input_overlay=overlay)
    else:
        result = bundle.build(repo,root,evidence,execution_plan=plan,input_overlay=overlay)
        manifest = bundle.verify_archive(result['path'])
        assert not any('/executed_inputs/' in member for member in manifest['files'])
        with zipfile.ZipFile(result['path']) as archive:
            assert archive.read(bundle.PREFIX+name) == executed
            proof = next(n for n in manifest['files'] if n.endswith('executed_input_overlay.json'))
            assert json.loads(archive.read(bundle.PREFIX+proof))['all_files_bound_to_frozen_dispatch_manifest']
    assert historical.read_bytes() == historical_before


@pytest.mark.parametrize('mode',['rounding','color_change','many_pixels','vector_change','data_change','dimensions'])
def test_regeneration_requires_exact_data_and_vectors_with_bounded_raster_rounding(tmp_path,mode):
    from PIL import Image
    original=tmp_path/'original'; regenerated=tmp_path/'regenerated'
    original.mkdir();regenerated.mkdir()
    for root in (original,regenerated):
        (root/'summary.json').write_text('{"value":1}\n')
        (root/'plot.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>\n')
        Image.new('RGBA',(100,100),(100,100,100,255)).save(root/'plot.png')
    expected={p.name:{'bytes':p.stat().st_size,'sha256':bundle.sha(p)} for p in original.iterdir()}
    with Image.open(regenerated/'plot.png') as source: changed=source.copy()
    changed.putpixel((0,0),(102 if mode!='color_change' else 103,100,100,255))
    if mode=='many_pixels':
        for x in range(20):changed.putpixel((x,0),(102,100,100,255))
    if mode=='dimensions':changed=changed.resize((101,100))
    changed.save(regenerated/'plot.png')
    if mode=='vector_change':(regenerated/'plot.svg').write_text('<svg>changed</svg>')
    if mode=='data_change':(regenerated/'summary.json').write_text('{"value":2}\n')
    if mode=='rounding':
        _,rounding=bundle.compare_regenerated_report(original,regenerated,expected)
        assert rounding['plot.png']['paired_svg_exact']
        assert rounding['plot.png']['changed_pixels']==1
    else:
        with pytest.raises(ValueError):bundle.compare_regenerated_report(original,regenerated,expected)
