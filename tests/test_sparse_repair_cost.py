"""Stdlib receipt arithmetic only; never contacts the provider."""
import json
from pathlib import Path
import pytest
from scripts import sparse_repair_cost as cost


def put(path,row):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(row))


def fixture(tmp):
    root=tmp/'run';historical=tmp/'historical.json'
    volumes=[{'id':f'v{i}','size':200 if i<6 else 170,'dataCenter':'R','type':'STANDARD'} for i in range(7)]
    old={'id':'old','name':'unrelated','status':'EXITED','cost':9}
    put(historical,{'conservative_cumulative_estimate_usd':cost.HISTORICAL_BASELINE,
        'hard_ceiling_usd':2500,'soft_target_usd':2000,'protected_network_volumes':volumes,
        'complete_provider_pod_inventory':[old]})
    put(root/'resources/inventory_latest.json',{'epoch':10800,'pods':[old], 'volumes':volumes})
    return root,historical


def allocation(root,name,pid,start=3600,deadline=18000,rate=4,upper=5,closed=None,present=False):
    folder=root/'resources'/name
    put(folder/'lease.json',{'experiment_id':'sparse_repair_01','pod_id':pid,'allocation_epoch':start,
        'deadline_epoch':deadline,'upper_hourly_usd':upper,'gpu_count':1,'total_cap_usd':2500,'cleanup_reserve_usd':40})
    pod={'id':pid,'name':name,'status':'RUNNING','cost':rate}
    put(folder/'pod.json',pod)
    if closed is not None:
        hours=(closed-start)/3600
        put(folder/'closed.json',{'pod_id':pid,'observed_absent_epoch':closed,'hours_upper_to_absence':hours,
            'quoted_compute_usd_upper':hours*rate,'upper_compute_usd':hours*upper,'invoice_final':False})
    if present:
        path=root/'resources/inventory_latest.json';row=json.loads(path.read_text());row['pods'].append(pod);put(path,row)


def test_separates_quote_elapsed_reservations_and_unposted_storage(tmp_path):
    root,h=fixture(tmp_path)
    allocation(root,'closed','c',closed=7200)
    allocation(root,'active','a',present=True)
    value=cost.make_ledger(root,h)
    assert value['incremental_quote_elapsed_compute_estimate_usd']==12 #4*1h +4*2h
    assert value['incremental_compute_reservation_scenario_usd']==25 #5*1h +5*4h
    assert value['remaining_upper_compute_reservation_usd']==10
    assert value['cumulative_estimate_excluding_unposted_storage_usd']==cost.HISTORICAL_BASELINE+12
    assert value['conservative_cumulative_reservation_scenario_usd']==1360+25+20+40
    assert value['inherited_baseline_reserve_allowance_usd']==pytest.approx(17.105879174068)
    assert value['invoice_final'] is False and value['active_owned_pod_ids']==['a']
    assert value['protected_storage']['protected_allocated_gb']==1370
    assert value['protected_storage']['actual_hourly_rate_usd'] is None
    assert value['protected_storage']['unposted_cost_usd'] is None
    assert value['new_compute_assumed_for_historically_exited_unrelated_usd']==0
    assert len(value['historically_exited_unrelated_pods'])==1
    assert value==cost.make_ledger(root,h) #no current-clock input
    assert len(value['historical_source']['sha256'])==64


def test_absent_without_closed_keeps_reservation_and_expired_not_clamped(tmp_path):
    root,h=fixture(tmp_path)
    allocation(root,'absent','x',deadline=18000)
    value=cost.make_ledger(root,h);row=value['per_allocations'][0]
    assert row['resource_status']=='ABSENT_IN_INVENTORY_UNCLOSED_RESERVATION'
    assert row['quote_times_elapsed_compute_estimate_usd']==8
    assert row['compute_reservation_scenario_usd']==20
    allocation(root,'overdue','y',deadline=7200,present=True)
    row=next(r for r in cost.make_ledger(root,h)['per_allocations'] if r['pod_id']=='y')
    assert row['compute_reservation_scenario_usd']==10 #actual2h at upperrate exceeds1hlease
    assert any('exceeds immutable deadline' in w for w in row['warnings'])


def test_unowned_running_flagged_not_added_as_zero_charge(tmp_path):
    root,h=fixture(tmp_path);p=root/'resources/inventory_latest.json';row=json.loads(p.read_text())
    row['pods'].append({'id':'unknown','name':'somebody_else','status':'RUNNING','cost':12})
    put(p,row);value=cost.make_ledger(root,h)
    assert len(value['unowned_running_resources'])==1
    assert value['compute_estimate_scope_complete'] is False
    assert value['incremental_quote_elapsed_compute_estimate_usd']==0
    assert value['unowned_running_resources'][0]['cost']==12


def test_closed_receipt_arithmetic_and_later_presence_rejected(tmp_path):
    root,h=fixture(tmp_path);allocation(root,'bad','bad',closed=7200,present=True)
    with pytest.raises(ValueError,match='still present'):cost.make_ledger(root,h)
    p=root/'resources/inventory_latest.json';row=json.loads(p.read_text());row['pods']=row['pods'][:1];put(p,row)
    p=root/'resources/bad/closed.json';row=json.loads(p.read_text());row['upper_compute_usd']=1;put(p,row)
    with pytest.raises(ValueError,match='arithmetic differs'):cost.make_ledger(root,h)


def test_stale_inventory_nonfinite_quote_and_changed_baseline_rejected(tmp_path):
    root,h=fixture(tmp_path);allocation(root,'future','f',start=10900)
    with pytest.raises(ValueError,match='Inventory predates'):cost.make_ledger(root,h)
    p=root/'resources/future/pod.json';row=json.loads(p.read_text());row['cost']=float('nan');put(p,row)
    with pytest.raises(ValueError,match='finite'):cost.make_ledger(root,h)
    row=json.loads(h.read_text());row['conservative_cumulative_estimate_usd']+=1;put(h,row)
    with pytest.raises(ValueError,match='Historical source baseline'):cost.make_ledger(root,h)


def test_unknown_request_storage_changes_and_atomic_output(tmp_path):
    root,h=fixture(tmp_path)
    put(root/'resources/unknown/reservation.json',{'reservation':{}})
    p=root/'resources/inventory_latest.json';row=json.loads(p.read_text());row['volumes']=row['volumes'][:-1];put(p,row)
    value=cost.make_ledger(root,h)
    assert value['unresolved_allocation_attempts'][0]['compute_cost_assumed_usd'] is None
    assert value['protected_storage']['inventory_missing_protected_ids']==['v6']
    assert not (root/'cost_ledger.json').exists() #make_ledger never writes
    out=root/'test-output.json';cost.write_atomic(out,value)
    assert json.loads(out.read_text())==value and not list(root.glob('*.tmp'))


def test_cli_without_collect_does_not_write_live_ledger(tmp_path,capsys):
    root,h=fixture(tmp_path)
    cost.main(['--root',str(root),'--historical',str(h)])
    assert json.loads(capsys.readouterr().out)['invoice_final'] is False
    assert not (root/'cost_ledger.json').exists()
    cost.main(['--root',str(root),'--historical',str(h),'--collect'])
    assert (root/'cost_ledger.json').exists()


def test_new_unowned_exited_is_not_treated_as_historical_zero_compute(tmp_path):
    root,h=fixture(tmp_path);p=root/'resources/inventory_latest.json';row=json.loads(p.read_text())
    row['pods'].append({'id':'new-exited','name':'unknown-history','status':'EXITED','cost':9})
    put(p,row);v=cost.make_ledger(root,h)
    assert len(v['newly_observed_unowned_exited_resources'])==1
    assert len(v['historically_exited_unrelated_pods'])==1
    assert v['compute_estimate_scope_complete'] is False


def temporary_storage(root,h,*,present=True,receipt=True):
    path=root/'resources/inventory_latest.json';latest=json.loads(path.read_text())
    if present:
        latest['volumes'].append({'id':'kzmo0a5erc','name':'temporary-scoring','size':20,
            'dataCenter':'EUR-IS-1','type':'STANDARD'})
        put(path,latest)
    liability=root/'resources/private_storage_fallback/storage_liability.json'
    if receipt:
        put(liability,{'epoch':10500,'status':'CREATED_NO_COMPUTE_YET',
            'new_volume_id':'kzmo0a5erc','new_volume_region':'EUR-IS-1','new_volume_size_gb':20,
            'protected_original_ids':[v['id'] for v in json.loads(h.read_text())['protected_network_volumes']],
            'all_original_volumes_preserved':True,'additional_storage_actual_rate_usd':None,
            'additional_storage_unposted_cost_usd':None,'actual_storage_charge_assumed_zero':False})
    return liability


def test_temporary_storage_is_additional_20gb_not_an_eighth_protected_volume(tmp_path):
    root,h=fixture(tmp_path);before=cost.make_ledger(root,h)
    liability=temporary_storage(root,h);value=cost.make_ledger(root,h)
    protected=value['protected_storage'];current=value['current_storage_inventory']
    assert protected==before['protected_storage']
    assert protected['protected_volume_count']==7 and protected['protected_allocated_gb']==1370
    assert current['current_total_volume_count']==8 and current['current_total_allocated_gb']==1390
    assert current['current_protected_volume_count']==7 and current['current_protected_allocated_gb']==1370
    assert current['additional_unprotected_volume_count']==1 and current['additional_unprotected_allocated_gb']==20
    extra=current['additional_unprotected_volumes'][0]
    assert extra['id']=='kzmo0a5erc' and extra['original_protected_volume'] is False
    assert extra['temporary_diagnostic_volume_identified_by_receipt'] is True
    assert extra['actual_hourly_rate_usd'] is None and extra['unposted_cost_usd'] is None
    assert extra['actual_storage_charge_assumed_zero'] is False
    assert current['actual_hourly_rate_usd'] is None and current['unposted_cost_usd'] is None
    assert current['temporary_storage_volume_present_in_current_inventory'] is True
    assert any(s['path']==str(liability.resolve()) and len(s['sha256'])==64 for s in value['sources'])
    assert value['cumulative_estimate_excluding_unposted_storage_usd']==before['cumulative_estimate_excluding_unposted_storage_usd']
    assert value['conservative_cumulative_reservation_scenario_usd']==before['conservative_cumulative_reservation_scenario_usd']
    assert value==cost.make_ledger(root,h)
    assert not (root/'cost_ledger.json').exists()


def test_additional_inventory_without_liability_receipt_is_not_ignored_or_claimed_temporary(tmp_path):
    root,h=fixture(tmp_path);temporary_storage(root,h,receipt=False)
    value=cost.make_ledger(root,h);current=value['current_storage_inventory']
    assert current['current_total_allocated_gb']==1390
    assert current['additional_unprotected_volume_count']==1
    assert current['temporary_storage_liability_receipt'] is None
    assert current['additional_unprotected_volumes'][0]['temporary_diagnostic_volume_identified_by_receipt'] is False
    assert current['additional_unprotected_volumes'][0]['unposted_cost_usd'] is None
    assert any('Additional unprotected network volumes' in w for w in value['warnings'])


def test_absent_temporary_volume_keeps_unknown_liability_without_inventing_current_capacity(tmp_path):
    root,h=fixture(tmp_path);temporary_storage(root,h,present=False)
    value=cost.make_ledger(root,h);current=value['current_storage_inventory']
    assert current['current_total_volume_count']==7 and current['current_total_allocated_gb']==1370
    assert current['additional_unprotected_volume_count']==0
    assert current['temporary_storage_volume_present_in_current_inventory'] is False
    assert current['temporary_storage_liability_receipt']['new_volume_id']=='kzmo0a5erc'
    assert current['unposted_cost_usd'] is None and current['actual_storage_charge_assumed_zero'] is False
    assert any('no storage-charge cessation' in w for w in value['warnings'])


def test_temporary_storage_cannot_change_original_protected_identity(tmp_path):
    root,h=fixture(tmp_path);path=temporary_storage(root,h)
    original=json.loads(path.read_text());changed={**original,'protected_original_ids':original['protected_original_ids']+['kzmo0a5erc']}
    put(path,changed)
    with pytest.raises(ValueError,match='protected-volume identity'):cost.make_ledger(root,h)
    changed={**original,'protected_original_ids':['substituted']+original['protected_original_ids'][1:]};put(path,changed)
    with pytest.raises(ValueError,match='protected-volume identity'):cost.make_ledger(root,h)
    put(path,{**original,'new_volume_id':'v0'})
    with pytest.raises(ValueError,match='separate from original protected'):cost.make_ledger(root,h)
    put(path,original);history=json.loads(h.read_text());history['protected_network_volumes'].append({'id':'kzmo0a5erc','size':20})
    put(h,history)
    with pytest.raises(ValueError,match='seven-volume/1370'):cost.make_ledger(root,h)


def test_temporary_size_region_or_receipt_age_discrepancy_is_not_silently_normalized(tmp_path):
    root,h=fixture(tmp_path);path=temporary_storage(root,h)
    row=json.loads(path.read_text());row['new_volume_size_gb']=30;row['new_volume_region']='OTHER';row['epoch']=10900;put(path,row)
    value=cost.make_ledger(root,h)
    assert value['current_storage_inventory']['current_total_allocated_gb']==1390
    assert any('size/region differs' in w for w in value['warnings'])
    assert any('Inventory predates temporary-storage' in w for w in value['warnings'])
    latest=root/'resources/inventory_latest.json';row=json.loads(latest.read_text());row['volumes'][-1]['size']=None;put(latest,row)
    with pytest.raises(ValueError,match='current volume size'):cost.make_ledger(root,h)


def expansion(root,*,update_inventory=True,result=True,result_epoch=9000):
    folder=root/'resources/private_score_final04'
    put(folder/'private_volume_expansion_intent.json',{'epoch':8999,'volume_id':'v6',
        'from_size_gb':170,'to_size_gb':175,'new_incremental_gb':5,
        'reason':'Non-destructive expansion after staging quota; no files deleted.','actual_rate_unknown':True})
    result_path=folder/'private_volume_expansion_result.json'
    if result:put(result_path,{'epoch':result_epoch,'returncode':0,'stdout':json.dumps({'id':'v6','size':175,'dataCenterId':'R'}),'stderr':''})
    if update_inventory:
        path=root/'resources/inventory_latest.json';inventory=json.loads(path.read_text())
        next(v for v in inventory['volumes'] if v['id']=='v6')['size']=175;put(path,inventory)
    return result_path


def test_protected_expansion_preserves_historical1370_and_reports_current1375_plus_extra20(tmp_path):
    root,h=fixture(tmp_path);historical_bytes=h.read_bytes();temporary_storage(root,h);result=expansion(root)
    value=cost.make_ledger(root,h);current=value['current_storage_inventory'];protected=value['protected_storage']
    assert h.read_bytes()==historical_bytes
    assert protected['protected_volume_count']==7 and protected['protected_allocated_gb']==1370
    assert protected['inventory_missing_protected_ids']==[] and protected['inventory_changed_size_ids']==['v6']
    assert protected['protected_sizes_are_historical_baseline_not_current_capacity'] is True
    assert current['current_protected_volume_count']==7 and current['current_protected_allocated_gb']==1375
    assert current['current_total_volume_count']==8 and current['current_total_allocated_gb']==1395
    assert current['additional_unprotected_allocated_gb']==20
    assert current['protected_expansion_gb_in_current_inventory']==5
    change=current['original_protected_volume_size_changes'][0]
    assert change['id']=='v6' and change['change']=='expansion' and change['delta_gb']==5
    assert change['same_original_protected_identity'] is True
    assert change['matching_successful_expansion_result_paths']==[str(result.resolve())]
    receipt=current['protected_volume_expansion_receipts'][0]
    assert receipt['provider_success_result_reported'] and receipt['current_inventory_matches_successful_expansion']
    assert receipt['actual_hourly_rate_usd'] is None and receipt['unposted_cost_usd'] is None
    assert any(s['path']==str(result.resolve()) for s in value['sources'])
    assert any('distinguish expansion from missing/deleted' in w for w in value['warnings'])
    assert value==cost.make_ledger(root,h)
    assert not (root/'cost_ledger.json').exists()


def test_protected_expansion_receipt_does_not_overwrite_older_inventory_totals(tmp_path):
    root,h=fixture(tmp_path);expansion(root,update_inventory=False,result_epoch=11000)
    value=cost.make_ledger(root,h);current=value['current_storage_inventory']
    assert current['current_protected_allocated_gb']==1370
    assert current['current_total_allocated_gb']==1370
    assert current['protected_expansion_gb_in_current_inventory']==0
    receipt=current['protected_volume_expansion_receipts'][0]
    assert receipt['provider_success_result_reported'] is True
    assert receipt['current_inventory_matches_successful_expansion'] is False
    assert receipt['to_size_gb']==175
    assert any('Inventory predates confirmed protected-volume expansion' in w for w in value['warnings'])


def test_protected_expansion_intent_is_not_success_and_mismatched_success_is_rejected(tmp_path):
    root,h=fixture(tmp_path);path=expansion(root,update_inventory=False,result=False)
    value=cost.make_ledger(root,h)
    assert value['current_storage_inventory']['protected_volume_expansion_receipts'][0]['provider_success_result_reported'] is False
    put(path,{'epoch':9000,'returncode':0,'stdout':json.dumps({'id':'another','size':175,'dataCenterId':'R'})})
    with pytest.raises(ValueError,match='provider result differs'):cost.make_ledger(root,h)
    put(path,{'epoch':9000,'returncode':0,'stdout':'not JSON'})
    with pytest.raises(ValueError,match='lacks provider volume JSON'):cost.make_ledger(root,h)
