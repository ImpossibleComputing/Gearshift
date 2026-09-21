import importlib.util
import io
import json
from pathlib import Path
import urllib.error

import pytest

from scripts import coding_confirmation_regional_dispatch as r

spec = importlib.util.spec_from_file_location('placement_fixture', Path(__file__).with_name('test_coding_confirmation_regional_dispatch.py'))
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def rejected(tmp_path, monkeypatch, status=400):
    area, api = f.setup(tmp_path, monkeypatch)
    api.post_error = urllib.error.HTTPError('provider', status, 'rejected', {}, io.BytesIO(b'{"message":"unclassified"}'))
    with pytest.raises(RuntimeError): r.allocate('failed', 'US-CO-1', 1)
    api.post_error = None
    return area, api


def test_http400_requires_two_delayed_inventories_then_new_names_can_proceed(tmp_path, monkeypatch):
    area, api = rejected(tmp_path, monkeypatch)
    monkeypatch.setattr(r.time, 'time', lambda: 100061.)
    assert not r.reconcile_rejected_pod('failed')['reconciled']
    with pytest.raises(ValueError, match='sixty'): r.reconcile_rejected_pod('failed')
    monkeypatch.setattr(r.time, 'time', lambda: 100122.)
    assert r.reconcile_rejected_pod('failed')['reconciled']
    assert r.history([]) == ([], [])
    assert len([c for c in api.calls if c[1]=='POST']) == 1
    with pytest.raises(ValueError, match='Allocation exists'): r.allocate('failed','US-CO-1',1,volume_id='volume_0')
    r.allocate('next', 'US-CO-1', 1, volume_id='volume_0')
    assert r.read(area/'failed/creation_failure.json')['explicit_capacity_rejection'] is False


@pytest.mark.parametrize('status', [403, 408, 429, 500, 503])
def test_transport_quota_permission_or_server_uncertainty_remains_blocked(tmp_path, monkeypatch, status):
    area, api = rejected(tmp_path, monkeypatch, status)
    monkeypatch.setattr(r.time, 'time', lambda: 100122.)
    with pytest.raises(ValueError, match='uncertainty'): r.reconcile_rejected_pod('failed')
    assert not (area/'failed/rejected_request_reconciliation.json').exists()


@pytest.mark.parametrize('change', ['pod_appears', 'changed_request', 'known_identity', 'tampered_timing'])
def test_reconciled_receipt_cannot_hide_late_creation_or_changed_evidence(tmp_path, monkeypatch, change):
    area, api = rejected(tmp_path, monkeypatch)
    monkeypatch.setattr(r.time, 'time', lambda: 100061.); r.reconcile_rejected_pod('failed')
    monkeypatch.setattr(r.time, 'time', lambda: 100122.); r.reconcile_rejected_pod('failed')
    folder=area/'failed'
    if change=='pod_appears': api.rows=[{'id':'late','name':'gearshift-confirmation-failed'}]
    elif change=='changed_request':
        value=r.read(folder/'create_request.json');value['disk']=99;f.atomic_json(folder/'create_request.json',value)
    elif change=='known_identity':f.atomic_json(folder/'allocated_pod_identity.json',{'pod_id':'late'})
    else:
        value=r.read(folder/'rejected_request_reconciliation.json');value['observations'][1]['epoch']=100062.
        f.atomic_json(folder/'rejected_request_reconciliation.json',value)
    with pytest.raises(ValueError):r.history(api.rows)
