"""Exercise paid-resource cleanup decisions without allocating resources."""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('watchdog', Path(__file__).resolve().parents[1] / 'scripts/phase2_cloud_watchdog.py')
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


def pod(pid='task', rate=3, status='RUNNING'):
    return dict(id=pid, name=w.PREFIX+pid, costPerHr=rate, desiredStatus=status)


def test_discovery_includes_stopped_and_excludes_unrelated():
    rows = [pod(), pod('stopped', status='EXITED'), dict(id='unrelated', name='other', costPerHr=999)]
    ledger, targets, reasons = w.evaluate({}, rows, w.START+60)
    assert set(ledger['pods']) == {'task', 'stopped'}
    assert not targets and not reasons


def test_deadline_deletes_only_task_pods(monkeypatch):
    monkeypatch.setattr(w, 'DEADLINE', w.START + 24 * 3600)
    _, targets, reasons = w.evaluate({}, [pod(), dict(id='other', name='unrelated')], w.DEADLINE)
    assert targets == ['task'] and 'deadline' in reasons


def test_budget_headroom():
    ledger, _, _ = w.evaluate({}, [pod()], w.START)
    ledger['pods']['task']['charged_from'] = w.START - 900 / 3.25 * 3600
    _, targets, reasons = w.evaluate(ledger, [pod()], w.START)
    assert targets == ['task'] and 'budget_with_two_minute_margin' in reasons


def test_rate_limit_and_unknown_rate():
    for rate, reason in [(21, 'allocated_rate'), (None, 'unknown_rate'), (float('nan'), 'unknown_rate')]:
        _, targets, reasons = w.evaluate({}, [pod(rate=rate)], w.START)
        assert targets == ['task'] and reason in reasons


def test_provider_failure_uses_durable_ids():
    ledger, _, _ = w.evaluate({}, [pod()], w.START)
    _, targets, reasons = w.evaluate(ledger, [], w.START+30, False)
    assert targets == ['task'] and reasons == ['provider_listing_failure']


def test_termination_requires_confirmed_absence_and_cost_never_decreases():
    ledger, _, _ = w.evaluate({}, [pod()], w.START+60)
    ledger, _, _ = w.evaluate(ledger, [], w.START+120)
    cost = ledger['estimated_upper_bound_usd']
    later, targets, _ = w.evaluate(ledger, [], w.START+240)
    assert later['estimated_upper_bound_usd'] == cost and not targets


def test_daemon_persists_discovery_before_delete_and_confirms_later(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(w, 'STATE', tmp_path)
    monkeypatch.setattr(w, 'DEADLINE', w.START + 24 * 3600)
    monkeypatch.setattr(w.time, 'time', lambda: w.DEADLINE+1)
    monkeypatch.setattr(w.subprocess, 'Popen', lambda *a, **kw: None)
    monkeypatch.setattr('sys.argv', ['guard', '--once'])
    deleted=[]
    def provider(*args):
        if args[:2] == ('pod', 'list'):
            return [pod(), dict(id='other',name='unrelated')]
        assert args == ('pod', 'delete', 'task')
        assert 'task' in json.loads((tmp_path/'ledger.json').read_text())['pods']
        deleted.append(args[2]); return None
    monkeypatch.setattr(w,'cli',provider)
    w.main()
    assert deleted==['task']
    status=json.loads((tmp_path/'watchdog_status.json').read_text())
    assert status['state']=='shutdown_enforcing' and status['actions'][0]['ok']
    assert json.loads((tmp_path/'ledger.json').read_text())['pods']['task']['terminated_confirmed_at'] is None


def test_daemon_listing_failure_attempts_registered_cleanup(tmp_path, monkeypatch):
    ledger,_,_=w.evaluate({},[pod()],w.START+60)
    w.write(tmp_path/'ledger.json',ledger)
    monkeypatch.setattr(w,'STATE',tmp_path)
    monkeypatch.setattr(w.time,'time',lambda:w.START+120)
    monkeypatch.setattr(w.subprocess,'Popen',lambda *a,**kw:None)
    monkeypatch.setattr('sys.argv',['guard','--once'])
    deleted=[]
    def provider(*args):
        if args[:2]==('pod','list'):raise TimeoutError()
        deleted.append(args);return None
    monkeypatch.setattr(w,'cli',provider)
    w.main()
    assert deleted==[('pod','delete','task')]
