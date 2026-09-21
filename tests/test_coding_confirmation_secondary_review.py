"""Packaging gates must reject altered conclusions without candidate execution."""
import importlib.util
from pathlib import Path

import pytest

from gearshift.coding_control import sha, write
from scripts import coding_confirmation_secondary_review as review

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('secondary_report_fixture', ROOT / 'tests/test_coding_confirmation_secondary_report.py')
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)


def saved_report(tmp_path, monkeypatch):
    c, private = fixture.fixtures.prepared(tmp_path, monkeypatch)
    fixture.fixtures.complete(c, private, monkeypatch)
    plan = str(c['plan_path'].relative_to(tmp_path))
    fixture.s.finalize(tmp_path, plan)
    private.unlink()
    references = fixture.references(c['plan']['task_ids'])
    monkeypatch.setattr(fixture.r, 'primary_references', lambda *a: (references, {'synthetic': True}))
    monkeypatch.setattr(fixture.s.base, 'score', lambda *a, **k: pytest.fail('Packaging executed a candidate'))
    fixture.r.report(plan, repo=tmp_path, primary_plan='synthetic_primary_plan.json')
    return c, plan


def test_frozen_secondary_report_reproduces_without_private_values_or_candidates(tmp_path, monkeypatch):
    c, plan = saved_report(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in (c['top'] / 'secondary/report').iterdir()}
    _, _, result = review.verify_report(plan, 'synthetic_primary_plan.json', tmp_path)
    assert result['secondary_answers'] == 24 and result['native_reference_draws_reused'] == 24
    assert before == {str(p): p.read_bytes() for p in (c['top'] / 'secondary/report').iterdir()}


@pytest.mark.parametrize('name', ['summary.json', 'per_draw.csv', 'CONFIRMATION_SECONDARY_RESULTS.md'])
def test_recomputed_gate_rejects_tampering_even_with_rehashed_manifest(tmp_path, monkeypatch, name):
    c, plan = saved_report(tmp_path, monkeypatch)
    root = c['top'] / 'secondary/report'; path = root / name
    if name == 'summary.json':
        value = review.public.read(path); value['contrasts']['ROTATING_M-FIXED_M']['difference'] = 1.; write(path, value)
    else:
        path.write_text(path.read_text() + '\nAltered report content\n')
    manifest = review.public.read(root / 'FILE_MANIFEST.json')
    for row in manifest['files']:
        if row['path'] == name: row.update(bytes=path.stat().st_size, sha256=sha(path))
    write(root / 'FILE_MANIFEST.json', manifest)
    with pytest.raises(ValueError, match='differs from frozen'):
        review.verify_report(plan, 'synthetic_primary_plan.json', tmp_path)


def test_existing_secondary_delivery_is_never_overwritten(tmp_path, monkeypatch):
    out = tmp_path / 'existing'; out.mkdir()
    monkeypatch.setattr(review, 'verify_report', lambda *a: pytest.fail('Gate should not run'))
    with pytest.raises(ValueError, match='already exists'):
        review.export('secondary.json', 'primary.json', tmp_path, out, repo=tmp_path)
