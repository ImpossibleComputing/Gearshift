"""Compact review export tests: synthetic immutable outputs, no candidate execution."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from gearshift.coding_control import digest, sha, write
from scripts import coding_confirmation_review as review
from scripts import coding_confirmation_report as reporter

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('score_fixture',ROOT/'tests/test_coding_confirmation_score.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def completed(tmp_path,monkeypatch):
    original=fixture.gfixture.context
    def context(path,count=1):
        c=original(path,count)
        for arm,cp in c['declaration']['primary_checkpoints'].items():
            name=f'checkpoints/{arm}/manifest.json';write(path/name,{'step':1024,'arm':arm,'files':{'mapper.pt':{'sha256':cp['mapper_sha256']}}})
            cp.update(manifest_path=name,manifest_sha256=sha(path/name))
        return c
    monkeypatch.setattr(fixture.gfixture,'context',context)
    c,private=fixture.prepared(tmp_path);fixture.complete_scores(c,private,monkeypatch)
    plan=str(c['plan_path'].relative_to(tmp_path));fixture.s.finalize(tmp_path,plan)
    private.unlink(); reporter.report(plan,repo=tmp_path)
    # The real report gate is used; only Git proof is synthetic in this temp repo.
    monkeypatch.setattr(review,'git_identity',lambda repo,d:{'source_commit':d['source_commit'],'publication_commit':d['publication_commit']})
    repaired(tmp_path)
    (tmp_path/'uv.lock').write_text('version = 1\n')
    (tmp_path/'README.md').write_text('Synthetic public README\n')
    return plan


def repaired(repo):
    original=repo/'original';root=repo/'results/scorer_repair'
    write(original/'answer.json',{'answer_text':'synthetic candidate','answer_ids':[1,2]})
    write(original/'score.json',{'code':'synthetic candidate','score':{'passed':False,'category':'timeout'}})
    old_closure={'files':[],'kl_files':[],'jobs':[]}
    write(original/'generation_closure.json',old_closure)
    write(original/'scored_answer_manifest.json',{'generation_closure_sha256':digest(old_closure)})
    row={'record_id':'record','answer_path':'answer.json','answer_sha256':sha(original/'answer.json'),
         'path':'score.json','sha256':sha(original/'score.json'),'code_sha256':review.hash_bytes(b'synthetic candidate')}
    plan={'expected_answers':1,'original_manifest_sha256':sha(original/'scored_answer_manifest.json'),
          'original_generation_closure_sha256':digest(old_closure),'records':[row]}
    write(root/'rescore_plan.json',plan)
    corrected={'binding':{'record_id':'record','plan_sha256':digest(plan),'answer_sha256':row['answer_sha256'],
                'original_score_sha256':row['sha256'],'code_sha256':row['code_sha256']},
                'original_score':{'passed':False,'category':'timeout'},'score_v2':{'passed':True,'missing':False}}
    write(root/'scores/record/score_v2.json',corrected)
    manifest={'plan_sha256':digest(plan),'committed':1,'missing':0,
              'files':[{'record_id':'record','path':'scores/record/score_v2.json','sha256':sha(root/'scores/record/score_v2.json')}]}
    write(root/'rescored_answer_manifest.json',manifest)
    summary={'plan_sha256':digest(plan),'manifest_sha256':sha(root/'rescored_answer_manifest.json'),
        'diagnostic_complete':True,'original_scores_preserved':True,'task_clusters':1,'diagnostics':[]}
    write(root/'report/summary.json',summary)
    (root/'report/SCORER_REPAIR_RESULTS.md').write_text('Original: 0; corrected: 1. Synthetic repair.\n')


def test_complete_primary_export_with_incomplete_secondary_and_every_member_verified(tmp_path,monkeypatch):
    plan=completed(tmp_path,monkeypatch)
    write(tmp_path/'results/replication/arms/FIXED/resume_preflight.json',{'passed':True})
    out=tmp_path/'delivery/new'
    result=review.export(plan,out,repo=tmp_path,original_evaluation='original')
    assert result['all_outputs_verified'] and result['every_member_read_back']
    assert result['archive_sha256']==sha(out/review.ZIP_NAME)
    assert result['coverage']['task_count']==1 and result['coverage']['answers_scored']==24
    assert result['coverage']['source_histories']==result['coverage']['independent_small_histories']==1
    assert result['secondary']['primary_delivery_waited_for_secondary'] is False
    assert not result['secondary']['arms']['FIXED']['full_1024_completion_reported']
    with zipfile.ZipFile(out/review.ZIP_NAME) as z:
        names=z.namelist();m=json.loads(z.read(review.MANIFEST))
        assert 'uv.lock' in names and 'README.md' in names
        assert 'original/answer.json' in names and 'original/score.json' in names
        assert 'results/scorer_repair/scores/record/score_v2.json' in names
        assert any(n.endswith('large_history/source_history.json') for n in names)
        assert any(n.endswith('sampler/completion_timing.json') for n in names)
        assert 'results/replication/arms/FIXED/resume_preflight.json' in names
        assert set(names)=={r['path'] for r in m['files']}|{review.MANIFEST}
        for item in m['files']:
            data=z.read(item['path']);assert len(data)==item['bytes'] and review.hash_bytes(data)==item['sha256']
        assert z.read('CONFIRMATION_RESULTS.md')==(out/'CONFIRMATION_RESULTS.md').read_bytes()
    assert 'separate' in (out/'CONFIRMATION_RESULTS.md').read_text().lower()
    # Reading/regenerating a report from the extracted public bundle needs no tests or models.
    extracted=tmp_path/'unpacked'
    with zipfile.ZipFile(out/review.ZIP_NAME) as z:z.extractall(extracted)
    regenerated=reporter.report(plan,repo=extracted)
    assert regenerated['answers']==24


@pytest.mark.parametrize('damage',['seal','scores','report','raw','report_statistics','scorer_old'])
def test_missing_or_changed_required_evidence_never_creates_delivery(tmp_path,monkeypatch,damage):
    plan=completed(tmp_path,monkeypatch)
    if damage=='seal':(tmp_path/'results/primary/generation_closure.json').unlink()
    elif damage=='scores':(tmp_path/'results/primary/scoring/scored_answer_manifest.json').unlink()
    elif damage=='report':(tmp_path/'results/primary/report/report_manifest.json').unlink()
    elif damage=='raw':next((tmp_path/'results/primary/tasks').glob('*/*/seed_*/answer.json')).write_text('{}')
    elif damage=='scorer_old':(tmp_path/'original/score.json').write_text('{}')
    else:
        p=tmp_path/'results/primary/report/summary.json';v=review.read(p);v['contrasts']['ROTATING_M-FIXED_M']['difference']=1.;write(p,v)
        r=tmp_path/'results/primary/report/report_manifest.json';v=review.read(r)
        for i in v['files']:
            if i['path']=='summary.json':i.update(bytes=p.stat().st_size,sha256=sha(p))
        write(r,v)
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):
        review.export(plan,tmp_path/'delivery',repo=tmp_path,original_evaluation='original')
    assert not (tmp_path/'delivery').exists()


def test_existing_export_is_not_overwritten_even_if_empty(tmp_path,monkeypatch):
    out=tmp_path/'existing';out.mkdir()
    monkeypatch.setattr(review,'gate',lambda *a:pytest.fail('Should reject before any evidence work'))
    with pytest.raises(ValueError,match='already exists'):review.export('no_plan.json',out,repo=tmp_path)


@pytest.mark.parametrize('name',['private/tests.json','protected/reference.json','.git/config','models/model.safetensors',
    'data/paired_cache/item.pt','data/hidden_tests.json','reference_solution.json','.venv/pyvenv.cfg','results/tensor.npy','results/tensor.npz','secrets.json','.env',
    '.ssh/id_ed25519','reference_answers/answers.json','cache/item.json'])
def test_heavy_private_and_secret_paths_are_not_read(name,tmp_path,monkeypatch):
    p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('do not read')
    assert not review.permitted(name)
    monkeypatch.setattr(Path,'read_bytes',lambda p:pytest.fail('Prohibited path opened'))
    with pytest.raises(ValueError):review.Collection(tmp_path).add(name)


def test_tree_prunes_private_paths_and_excludes_even_tiny_tensors(tmp_path,monkeypatch):
    for name in ['results/public.json','results/mapper.pt','results/cache/content.json','results/private/tests.json']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    files=review.Collection(tmp_path);files.tree('results')
    assert set(files.files)=={'results/public.json'}


@pytest.mark.parametrize('value',[{'api_key':'not-a-public-key'}, {'nested':{'password':'secret'}},
    {'hidden_tests':[{'input':'secret','output':'secret'}]}, {'reference_solution':'secret'},
    {'authorization':'Bearer secret-value'}])
def test_structural_secrets_or_private_values_fail_closed(value):
    with pytest.raises(ValueError):review.scan('receipt.json',review.encoded(value))


def test_hashes_and_private_input_paths_are_public_metadata():
    review.scan('receipt.json',review.encoded({'private_tests_sha256':'a'*64,'private_test_path':'data/private/tests.json',
        'private_test_values_included':False,'password':None,'authorization':'owner approval'}))


def test_credential_text_signatures_reject_logs_and_raw_artifacts():
    with pytest.raises(ValueError):review.scan('log.txt',('rpa_'+'a'*30).encode())


@pytest.mark.parametrize('path',['../escape.json','/absolute.json','a/../../x.json','a\\b.json','a//b.json'])
def test_path_traversal_rejected(path):
    with pytest.raises(ValueError):review.permitted(path)


def test_links_cannot_smuggle_private_or_outside_artifacts(tmp_path):
    (tmp_path/'outside.json').write_text('{}');(tmp_path/'results').mkdir()
    (tmp_path/'results/receipt.json').symlink_to(tmp_path/'outside.json')
    with pytest.raises(ValueError,match='Linked'):review.Collection(tmp_path).add('results/receipt.json')


def test_injected_secret_aborts_and_removes_staging(tmp_path,monkeypatch):
    plan=completed(tmp_path,monkeypatch)
    write(tmp_path/'results/resource.json',{'api_key':'not-public'})
    with pytest.raises(ValueError,match='Credential'):review.export(plan,tmp_path/'delivery',repo=tmp_path,original_evaluation='original')
    assert not (tmp_path/'delivery').exists() and not list(tmp_path.glob('.confirmation-review-*'))


def test_archive_tampering_or_unlisted_members_rejected(tmp_path):
    path=tmp_path/'test.zip';content=b'public'
    inventory={'files':[{'path':'report.txt','bytes':len(content),'sha256':review.hash_bytes(content)}]}
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('report.txt',content);z.writestr(review.MANIFEST,review.encoded(inventory));z.writestr('extra.json','{}')
    with pytest.raises(ValueError,match='membership'):review.verify_archive(path)
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('report.txt',b'changed');z.writestr(review.MANIFEST,review.encoded(inventory))
    with pytest.raises(ValueError,match='hash/size'):review.verify_archive(path)


def test_supplement_report_is_included_without_overwriting_original(tmp_path,monkeypatch):
    plan=completed(tmp_path,monkeypatch);root=tmp_path/'results/scorer_repair';p=root/'supplement_v1';p.mkdir()
    (p/'SCORER_REPAIR_RESULTS.md').write_text('Expanded original versus corrected results.\n')
    write(p/'source_summary.json',review.read(root/'report/summary.json'))
    write(p/'FILE_MANIFEST.json',{'private_tests_included':False,'files':[
        {'path':f.name,'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(p.iterdir())]})
    before=(root/'report/SCORER_REPAIR_RESULTS.md').read_bytes()
    review.export(plan,tmp_path/'delivery',repo=tmp_path,original_evaluation='original')
    assert (tmp_path/'delivery/SCORER_REPAIR_RESULTS.md').read_text().startswith('Expanded')
    assert (root/'report/SCORER_REPAIR_RESULTS.md').read_bytes()==before


def test_saved_costs_are_timestamped_and_not_a_live_claim(tmp_path):
    prefix='evidence/coding_pilot_v1/experiment';r=tmp_path/prefix/'resources'
    write(r/'older.json',{'observed_at_utc':'2026-09-19T08:00:00Z','this_round_compute_estimate_usd':12.,'conservative_cumulative_estimate_usd':300.})
    write(r/'newer.json',{'observed_at_utc':'2026-09-19T09:00:00Z','this_round_compute_estimate_usd':16.,'conservative_cumulative_estimate_usd':304.})
    write(r/'unknown_time.json',{'this_round_compute_estimate_usd':9999.})
    value=review.saved_resources(review.Collection(tmp_path),prefix)
    assert value['latest_dated_cost_receipt']['saved_values']['conservative_cumulative_estimate_usd']==304.
    assert value['live_provider_check_performed'] is False and value['export_time_is_not_observation_time']


def test_delivery_race_preserves_the_other_directory(tmp_path,monkeypatch):
    plan=completed(tmp_path,monkeypatch);out=tmp_path/'delivery';original=review.verify_archive
    def racing(*args):
        verified=original(*args);out.mkdir();(out/'keep.txt').write_text('other export');return verified
    monkeypatch.setattr(review,'verify_archive',racing)
    with pytest.raises(FileExistsError):review.export(plan,out,repo=tmp_path,original_evaluation='original')
    assert (out/'keep.txt').read_text()=='other export' and not (out/'DELIVERY_RECEIPT.json').exists()
