import importlib.util
import json
import pytest
from pathlib import Path
from gearshift.coding_control import PREFIX,write

def test_watchdog_never_deletes_unrelated_resources(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/coding_cloud_guard.py'
    spec=importlib.util.spec_from_file_location('pilot_guard_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'C',tmp_path);clock=[1.0];monkeypatch.setattr(m.time,'time',lambda:clock[0])
    write(tmp_path/'allocation_intent.json',{'started_epoch':0})
    write(tmp_path/'controller_heartbeat.json',{'epoch':1})
    called=[]
    def fake(*args):
        called.append(args)
        if args[1]=='list':
            return [{'id':'ours_'+args[0],'name':PREFIX+'test','costPerHr':4.59},{'id':'foreign','name':'unrelated-work','costPerHr':99}]
        return {}
    monkeypatch.setattr(m,'cli',fake);m.tick()
    assert not any(x[1]=='delete' for x in called)
    clock[0]=4*3600;write(tmp_path/'controller_heartbeat.json',{'epoch':clock[0]});write(tmp_path/'backup_verified.json',{'verified':True,'volume_id':'ours_network-volume'})
    m.tick()
    deletes=[x for x in called if x[1]=='delete']
    assert {x[2] for x in deletes}=={'ours_pod','ours_network-volume'}
    assert (tmp_path/'STOP').exists()

def test_unverified_volume_is_preserved_when_controller_lost(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/coding_cloud_guard.py'
    spec=importlib.util.spec_from_file_location('pilot_guard_test2',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'C',tmp_path);monkeypatch.setattr(m.time,'time',lambda:1000.)
    write(tmp_path/'allocation_intent.json',{'started_epoch':0});called=[]
    def fake(*args):
        called.append(args)
        if args[1]=='list':return [{'id':args[0],'name':PREFIX+'test','costPerHr':4.59}]
        return {}
    monkeypatch.setattr(m,'cli',fake);m.tick()
    assert ('pod','delete','pod') in called
    assert not any(x[:2]==('network-volume','delete') for x in called)

def test_confirmed_cleanup_clears_only_stale_controller_flag(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/coding_cloud_guard.py'
    spec=importlib.util.spec_from_file_location('pilot_guard_test3',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'C',tmp_path);monkeypatch.setattr(m.time,'time',lambda:6*3600.)
    write(tmp_path/'allocation_intent.json',{'started_epoch':0})
    write(tmp_path/'ledger.json',{'stage':'preflight','stage_started_epoch':0,'controller_lost':True,'resources':[
        {'kind':'pod','id':'old','started_epoch':0,'absent_epoch':1800,'upper_rate_usd':5.52}]})
    monkeypatch.setattr(m,'cli',lambda *args:[])
    m.tick();status=json.loads((tmp_path/'watchdog_status.json').read_text())
    assert status['state']=='all_task_resources_absent'
    assert not status['stop']
    assert status['gpu_hours']==.5 and status['upper_usd']==2.76


def load_guard(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/coding_cloud_guard.py'
    spec=importlib.util.spec_from_file_location('parallel_guard',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'C',tmp_path);clock=[1000.];monkeypatch.setattr(m.time,'time',lambda:clock[0])
    write(tmp_path/'allocation_intent.json',{'started_epoch':0})
    return m,clock


def configure_parallel(tmp_path,monkeypatch,m):
    from test_coding_control import experiment_approval,parallel_approval
    from gearshift.coding_control import PARALLEL_PARENT_SHA256
    write(tmp_path/'experiment_extension_approved.json',experiment_approval())
    write(tmp_path/'parallel_500h_approved.json',parallel_approval())
    original=m.sha
    monkeypatch.setattr(m,'sha',lambda p:PARALLEL_PARENT_SHA256 if p.name=='experiment_extension_approved.json' else original(p))


def resources(tmp_path,ids=('a','b')):
    rows={'pod':[],'network-volume':[]}
    for ident in ids:
        owner='controller-'+ident
        for kind in ['pod','volume']:
            r={'kind':kind,'id':kind+ident,'name':PREFIX+kind+ident,'started_epoch':0,
               'upper_rate_usd':5.52 if kind=='pod' else .04,'controller_id':owner}
            if kind=='pod':r.update(gpu_count=1,gpu_type='NVIDIA H200')
            write(tmp_path/'resource_receipts'/f'{kind}{ident}.json',r)
            rows['pod' if kind=='pod' else 'network-volume'].append({**r,'costPerHr':4.59,'gpuCount':1,'gpuTypeId':'NVIDIA H200'})
        write(tmp_path/'controller_heartbeats'/f'{owner}.json',{'controller_id':owner,'epoch':1000})
        write(tmp_path/'backups_verified'/f'volume{ident}.json',{'verified':True,'volume_id':'volume'+ident,'epoch':1000,'sha256':'verified'})
    return rows


def test_parallel_controller_failure_only_stops_its_resources(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    rows=resources(tmp_path)
    write(tmp_path/'controller_heartbeats/controller-a.json',{'controller_id':'controller-a','epoch':0})
    # A fresh global heartbeat must not mask the failed worker controller.
    write(tmp_path/'controller_heartbeat.json',{'epoch':clock[0]})
    called=[]
    def fake(*args):
        called.append(args)
        return rows[args[0]] if args[1]=='list' else {}
    monkeypatch.setattr(m,'cli',fake);m.tick()
    assert {c[2] for c in called if c[1]=='delete'}=={'poda','volumea'}
    assert not (tmp_path/'STOP').exists()
    s=json.loads((tmp_path/'watchdog_status.json').read_text())
    assert not s['stop'] and s['controller_stop_resource_ids']==['poda','volumea']


def test_parallel_budget_stop_is_global_but_never_deletes_unverified_volume(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    rows=resources(tmp_path)
    (tmp_path/'backups_verified/volumeb.json').unlink()
    write(tmp_path/'ledger.json',{'resources':[],'prior_upper_usd':980})
    called=[]
    def fake(*args):
        called.append(args)
        return rows[args[0]] if args[1]=='list' else {}
    monkeypatch.setattr(m,'cli',fake);m.tick()
    assert {c[2] for c in called if c[1]=='delete'}=={'poda','podb','volumea'}
    assert (tmp_path/'STOP').exists()


def test_transient_provider_error_blocks_dispatch_without_killing_healthy_workers(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    rows=resources(tmp_path,('a',));called=[];fail=[False]
    def fake(*args):
        called.append(args)
        if args[1]=='list':
            if fail[0]:raise RuntimeError('temporary provider timeout')
            return rows[args[0]]
        return {}
    monkeypatch.setattr(m,'cli',fake);m.tick();fail[0]=True;clock[0]+=15;m.tick()
    s=json.loads((tmp_path/'watchdog_status.json').read_text())
    assert not s['stop'] and s['dispatch_blocked']
    assert not any(c[1]=='delete' for c in called)
    assert s['gpu_hours']==1015/3600
    fail[0]=False;clock[0]+=15;m.tick()
    s=json.loads((tmp_path/'watchdog_status.json').read_text())
    assert not s['dispatch_blocked'] and not (tmp_path/'STOP').exists()


def test_sustained_provider_failure_terminates_tracked_resources(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    rows=resources(tmp_path,('a',));called=[];fail=[False]
    def fake(*args):
        called.append(args)
        if args[1]=='list':
            if fail[0]:raise RuntimeError('provider offline')
            return rows[args[0]]
        return {}
    monkeypatch.setattr(m,'cli',fake);m.tick();fail[0]=True
    for _ in range(3):clock[0]+=15;m.tick()
    assert {c[2] for c in called if c[1]=='delete'}=={'poda','volumea'}
    s=json.loads((tmp_path/'watchdog_status.json').read_text())
    assert s['stop'] and 'provider_state_unknown' in s['stop_reasons']


def test_provider_grace_requires_fresh_verified_backup(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    rows=resources(tmp_path,('a',));called=[];fail=[False]
    def fake(*args):
        called.append(args)
        if args[1]=='list':
            if fail[0]:raise RuntimeError('provider offline')
            return rows[args[0]]
        return {}
    monkeypatch.setattr(m,'cli',fake);m.tick()
    (tmp_path/'backups_verified/volumea.json').unlink();fail[0]=True;m.tick()
    assert ('pod','delete','poda') in called
    assert ('network-volume','delete','volumea') not in called


def test_provider_grace_ends_at_120_seconds_even_before_third_failure(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    rows=resources(tmp_path,('a',));called=[];fail=[False]
    def fake(*args):
        called.append(args)
        if args[1]=='list':
            if fail[0]:raise RuntimeError('provider offline')
            return rows[args[0]]
        return {}
    monkeypatch.setattr(m,'cli',fake);m.tick();fail[0]=True;m.tick()
    assert not any(c[1]=='delete' for c in called)
    clock[0]+=120;m.tick()
    assert ('pod','delete','poda') in called


def test_live_ack_allows_provider_grace_but_never_volume_deletion(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    rows=resources(tmp_path,('a',));called=[];fail=[False]
    final=tmp_path/'backups_verified/volumea.json'
    write(tmp_path/'backups_ack/volumea.json',json.loads(final.read_text()));final.unlink()
    def fake(*args):
        called.append(args)
        if args[1]=='list':
            if fail[0]:raise RuntimeError('provider offline')
            return rows[args[0]]
        return {}
    monkeypatch.setattr(m,'cli',fake);m.tick();fail[0]=True;m.tick()
    assert not any(c[1]=='delete' for c in called)
    for _ in range(2):clock[0]+=15;m.tick()
    assert ('pod','delete','poda') in called
    assert ('network-volume','delete','volumea') not in called


def test_h200_and_single_gpu_enforced_for_parallel_pods(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    rows=resources(tmp_path,('a',));rows['pod'][0]['gpuTypeId']='NVIDIA A100';called=[]
    def fake(*args):
        called.append(args)
        return rows[args[0]] if args[1]=='list' else {}
    monkeypatch.setattr(m,'cli',fake);m.tick()
    assert ('pod','delete','poda') in called
    assert 'allocation_outside_approved_hardware_or_quote' in json.loads((tmp_path/'watchdog_status.json').read_text())['stop_reasons']


def test_legacy_final_backup_receipt_remains_usable(tmp_path,monkeypatch):
    m,_=load_guard(tmp_path,monkeypatch)
    receipt={'volume_id':'volumea','worker_confirmed_absent':True,'copies':['a.tar.gz','b.tar.gz'],'sha256':'a'*64,'epoch':1000}
    write(tmp_path/'backup_verified.json',receipt)
    assert m.backup_for({'id':'volumea'})==receipt
    assert m.backup_for({'id':'volumeb'})=={}


def test_new_provider_resource_has_bounded_receipt_registration_grace(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    called=[]
    def fake(*args):
        called.append(args)
        if args[:2]==('pod','list'):
            return [{'id':'new','name':PREFIX+'new','costPerHr':4.59,'gpuCount':1,'gpuTypeId':'NVIDIA H200'}]
        return [] if args[1]=='list' else {}
    monkeypatch.setattr(m,'cli',fake);m.tick()
    s=json.loads((tmp_path/'watchdog_status.json').read_text())
    assert not s['stop'] and s['dispatch_blocked']
    assert not any(c[1]=='delete' for c in called)
    clock[0]+=60;m.tick()
    assert ('pod','delete','new') in called
    assert not (tmp_path/'STOP').exists()


def test_receipt_registration_clears_grace_and_enforces_scoped_heartbeat(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    def fake(*args):
        if args[:2]==('pod','list'):return [{'id':'new','name':PREFIX+'new','costPerHr':4.59,'gpuCount':1}]
        return [] if args[1]=='list' else {}
    monkeypatch.setattr(m,'cli',fake);m.tick()
    write(tmp_path/'resource_receipts/new.json',{'kind':'pod','id':'new','name':PREFIX+'new','started_epoch':1000,
          'upper_rate_usd':5.52,'controller_id':'worker-new','gpu_count':1,'gpu_type':'NVIDIA H200'})
    write(tmp_path/'controller_heartbeats/worker-new.json',{'controller_id':'worker-new','epoch':1000})
    clock[0]+=15;m.tick()
    s=json.loads((tmp_path/'watchdog_status.json').read_text())
    assert not s['stop'] and not s['dispatch_blocked']


def test_predispatch_intent_binds_creation_time_before_id_receipt(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    write(tmp_path/'allocation_intents/worker-new.json',{'controller_id':'worker-new','started_epoch':990,
          'resource_names':[PREFIX+'new'],'gpu_count':1,'gpu_type':'NVIDIA H200'})
    write(tmp_path/'controller_heartbeats/worker-new.json',{'controller_id':'worker-new','epoch':1000})
    def fake(*args):
        if args[:2]==('pod','list'):return [{'id':'new','name':PREFIX+'new','costPerHr':4.59,'gpuCount':1}]
        return [] if args[1]=='list' else {}
    monkeypatch.setattr(m,'cli',fake);m.tick()
    s=json.loads((tmp_path/'watchdog_status.json').read_text())
    assert s['gpu_hours']==10/3600 and not s['dispatch_blocked']


def test_exact_provider_h200_sxm_display_alias_is_allowed(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);configure_parallel(tmp_path,monkeypatch,m)
    rows=resources(tmp_path,('a',));pod=rows['pod'][0];pod.pop('gpuTypeId');pod['machine']={'gpuDisplayName':'H200 SXM'}
    called=[]
    def fake(*args):
        called.append(args)
        return rows[args[0]] if args[1]=='list' else {}
    monkeypatch.setattr(m,'cli',fake);m.tick()
    assert not any(c[1]=='delete' for c in called)
    assert not json.loads((tmp_path/'watchdog_status.json').read_text())['stop']
    pod['machine']['gpuDisplayName']='H100 SXM';m.tick()
    assert ('pod','delete','poda') in called


def configure_inspection(tmp_path,monkeypatch,m):
    from test_coding_control import storage_intent
    configure_parallel(tmp_path,monkeypatch,m)
    intent=storage_intent();intent['approval_sha256']=m.sha(tmp_path/'parallel_500h_approved.json')
    intent.update(started_epoch=1000,deadline_epoch=4600)
    write(tmp_path/'storage_inspection_intent.json',intent)
    write(tmp_path/'controller_heartbeats/inspect.json',{'controller_id':'inspect','epoch':1000})
    return intent


def test_guard_allows_only_bound_zero_gpu_storage_pod_and_charges_cpu_rate(tmp_path,monkeypatch):
    m,clock=load_guard(tmp_path,monkeypatch);intent=configure_inspection(tmp_path,monkeypatch,m)
    rows=resources(tmp_path,tuple(str(i) for i in range(8)))
    rows['pod'].append({'id':'cpu','name':intent['resource_name'],'gpuCount':0,'costPerHr':.20,'networkVolumeId':'retained'})
    called=[]
    def fake(*args):
        called.append(args)
        return rows[args[0]] if args[1]=='list' else {}
    monkeypatch.setattr(m,'cli',fake);m.tick();clock[0]+=60;m.tick()
    status=json.loads((tmp_path/'watchdog_status.json').read_text())
    assert not status['stop'] and status['active_gpu_count']==8 and status['active_cpu_storage_inspection_count']==1
    ledger=json.loads((tmp_path/'ledger.json').read_text());cpu=next(r for r in ledger['resources'] if r['id']=='cpu')
    assert cpu['upper_rate_usd']==.50 and cpu['gpu_count']==0
    assert not any(c[1]=='delete' for c in called)
    # Refresh healthy research supervision, expire just the one-hour CPU mount.
    clock[0]=4600
    for i in range(8):write(tmp_path/f'controller_heartbeats/controller-{i}.json',{'controller_id':f'controller-{i}','epoch':4600})
    write(tmp_path/'controller_heartbeats/inspect.json',{'controller_id':'inspect','epoch':4600})
    m.tick();assert {c[2] for c in called if c[1]=='delete'}=={'cpu'}
    assert not (tmp_path/'STOP').exists()


@pytest.mark.parametrize('patch',[{'costPerHr':.51},{'gpuCount':None},{'networkVolumeId':'other'},{'gpuCount':1}])
def test_cpu_inspection_rejects_excess_quote_missing_gpu_count_and_wrong_volume(tmp_path,monkeypatch,patch):
    m,clock=load_guard(tmp_path,monkeypatch);intent=configure_inspection(tmp_path,monkeypatch,m)
    row={'id':'cpu','name':intent['resource_name'],'gpuCount':0,'costPerHr':.20,'networkVolumeId':'retained'}
    row.update(patch)
    called=[]
    def fake(*args):
        called.append(args)
        return [row] if args[:2]==('pod','list') else [] if args[1]=='list' else {}
    monkeypatch.setattr(m,'cli',fake);m.tick()
    assert ('pod','delete','cpu') in called
    assert 'allocation_outside_approved_hardware_or_quote' in json.loads((tmp_path/'watchdog_status.json').read_text())['stop_reasons']
