import json
import pytest
from gearshift.coding_control import bind, decision, seed_for

def ledger():
    return {'stage':'preflight','stage_started_epoch':0,'resources':[
        {'kind':'pod','started_epoch':0,'upper_rate_usd':5.75}]}

def test_preflight_warns_then_stops():
    assert decision(ledger(), 3*3600)['warning'] == ['stage_hours']
    d = decision(ledger(), 4*3600)
    assert d['stop'] and 'stage_hours' in d['stop_reasons']

def test_stopped_resources_charged_until_absent():
    x=ledger(); x['resources'][0]['status']='STOPPED'
    assert decision(x,3600)['upper_usd']==5.75
    x['resources'][0]['absent_epoch']=3600
    assert decision(x,7200)['upper_usd']==5.75

def test_stage_limit_counts_gpu_hours_and_reports_offline_wall_time():
    x=ledger();x['resources'][0]['absent_epoch']=1800
    d=decision(x,6*3600)
    assert d['stage_gpu_hours']==.5 and d['stage_wall_hours']==6
    assert not d['stop']

def test_total_cleanup_reserve_and_unknown_state():
    x=ledger(); x['prior_upper_usd']=279
    assert decision(x,3600)['stop']
    x=ledger();x['provider_state_unknown']=True
    assert decision(x,0)['stop']

def test_identity_and_order_independent_streams(tmp_path):
    p=tmp_path/'identity.json';bind(p,{'revision':'a'})
    with pytest.raises(ValueError):bind(p,{'revision':'b'})
    assert seed_for('task',0,'answer_small')==seed_for('task',0,'answer_small')
    assert seed_for('task',0,'source_reasoning')!=seed_for('task',0,'small_reasoning')

def test_approved_extension_preserves_usage_and_overall_limits():
    from gearshift.coding_control import decision,remaining_preflight_seconds
    ext={'approved':True,'stage':'preflight','gpu_hours':10,'usd_cap':150,'owner_instruction':'Approved','receipt_sha256':'receipt'}
    ledger={'approved_preflight_extension':ext,'resources':[{'kind':'pod','started_epoch':0,'absent_epoch':5*3600,'upper_rate_usd':5.52}]}
    d=decision(ledger,5*3600)
    assert not d['stop'] and d['gpu_hours']==5 and d['upper_usd']==27.6
    assert remaining_preflight_seconds(ledger,5*3600)==5*3600
    ledger['resources'][0]['absent_epoch']=10*3600
    assert 'stage_hours' in decision(ledger,10*3600)['stop_reasons']
    ledger['prior_upper_usd']=280
    assert 'total_cost' in decision(ledger,10*3600)['stop_reasons']

def test_unapproved_or_invalid_extension_cannot_enable_dispatch():
    from gearshift.coding_control import decision
    for ext in [{'approved':False},{'approved':True,'stage':'preflight','gpu_hours':100,'usd_cap':1000,'owner_instruction':'text','receipt_sha256':'x'}]:
        assert 'invalid_budget_approval' in decision({'approved_preflight_extension':ext},0)['stop_reasons']

def test_malformed_approval_fails_closed_and_overall_gpu_cap_still_applies():
    from gearshift.coding_control import decision
    base={'approved':True,'stage':'preflight','gpu_hours':10,'usd_cap':150,'owner_instruction':'Approved','receipt_sha256':'receipt'}
    for extension in [[],{**base,'gpu_hours':'ten'}]:
        assert 'invalid_budget_approval' in decision({'approved_preflight_extension':extension},0)['stop_reasons']
    ledger={'stage':'cleanup','stage_start_gpu_hours':47,'approved_preflight_extension':base,
        'resources':[{'kind':'pod','started_epoch':0,'absent_epoch':48*3600,'upper_rate_usd':1}]}
    d=decision(ledger,48*3600)
    assert d['stage_gpu_hour_ceiling']==4 and 'GPU_hours' in d['stop_reasons']


def experiment_approval():
    return {'approved':True,'scope':'coding_pilot_v1_experiment','gpu_hours':100,'usd_cap':1000,
        'cleanup_reserve_usd':20,'owner_instruction':'Approved replacement budget','receipt_sha256':'bound',
        'limits_apply_cumulatively':True,'supersedes_stage_budget_limits':True,'one_gpu_only':True,
        'gpu_price_ceiling_usd':5.5,'scientific_scope_unchanged':True}


def test_replacement_budget_preserves_all_prior_usage_and_supersedes_stage_caps():
    from gearshift.coding_control import remaining_preflight_seconds
    x={'approved_experiment_extension':experiment_approval(),'resources':[
        {'kind':'pod','started_epoch':0,'absent_epoch':20*3600,'upper_rate_usd':5.52}]}
    d=decision(x,30*3600)
    assert d['upper_usd']==110.4 and d['gpu_hours']==20 and not d['stop']
    assert d['stage_gpu_hour_ceiling']==100 and d['stage_dollar_ceiling']==1000
    assert remaining_preflight_seconds(x,30*3600)==80*3600
    x.update(stage='training',stage_start_gpu_hours=19,stage_start_upper_usd=100)
    d=decision(x,30*3600)
    assert d['stage_gpu_hours']==1 and d['stage_gpu_hour_ceiling']==81


def test_replacement_budget_total_limits_and_cleanup_reserve():
    x={'approved_experiment_extension':experiment_approval(),'resources':[
        {'kind':'pod','started_epoch':0,'absent_epoch':75*3600,'upper_rate_usd':1}]}
    assert 'GPU_hours' in decision(x,0)['warning']
    x['resources'][0]['absent_epoch']=100*3600
    assert 'GPU_hours' in decision(x,0)['stop_reasons']
    d=decision({'approved_experiment_extension':experiment_approval(),'prior_upper_usd':980},0)
    assert d['hard_total_cap_usd']==1000 and 'total_cost' in d['stop_reasons']


def test_missing_or_malformed_replacement_budget_fails_closed():
    assert decision({},0)['hard_total_cap_usd']==300
    for patch in [{'approved':False},{'gpu_hours':101},{'usd_cap':1001},{'gpu_hours':'100'},
                  {'gpu_hours':float('nan')},{'limits_apply_cumulatively':False},{'one_gpu_only':False},
                  {'scientific_scope_unchanged':False},{'receipt_sha256':''}]:
        d=decision({'approved_experiment_extension':{**experiment_approval(),**patch}},0)
        assert d['stop'] and 'invalid_experiment_approval' in d['stop_reasons']
        assert d['hard_total_cap_usd']==300


def parallel_approval():
    from gearshift.coding_control import PARALLEL_PARENT_SHA256
    return {'approved':True,'scope':'coding_pilot_v1_experiment','gpu_hours':500,'usd_cap':1000,
        'cleanup_reserve_usd':20,'owner_instruction':'Increase ceiling to 500 hours and parallelize',
        'receipt_sha256':'new-bound-receipt','previous_experiment_approval_sha256':PARALLEL_PARENT_SHA256,
        'limits_apply_cumulatively':True,'supersedes_stage_budget_limits':True,'max_concurrent_gpus':8,
        'one_gpu_per_pod':True,'allowed_gpu':'NVIDIA H200','gpu_price_ceiling_usd':5.5,
        'parallel_execution_approved':True,'scientific_scope_unchanged':True,'fixed_cohort_and_sampling_unchanged':True}


def parallel_ledger():
    from gearshift.coding_control import PARALLEL_PARENT_SHA256
    return {'approved_experiment_extension':{**experiment_approval(),'receipt_sha256':PARALLEL_PARENT_SHA256},
            'approved_parallel_extension':parallel_approval(),'resources':[]}


def test_parallel_approval_preserves_historical_receipt_and_all_usage():
    x=parallel_ledger()
    old=json.loads(json.dumps(x['approved_experiment_extension']))
    x['resources']=[{'kind':'pod','started_epoch':0,'absent_epoch':100*3600,'gpu_count':1,'upper_rate_usd':1},
        {'kind':'pod','started_epoch':100*3600,'gpu_count':1,'upper_rate_usd':5.52},
        {'kind':'pod','started_epoch':100*3600,'gpu_count':1,'upper_rate_usd':5.52}]
    d=decision(x,110*3600)
    assert d['gpu_hours']==120 and d['upper_usd']==210.4
    assert d['active_gpu_count']==2 and d['active_upper_rate_usd']==11.04
    assert d['hard_total_gpu_hours']==500 and d['hard_total_cap_usd']==1000
    assert d['cleanup_dispatch_ceiling_usd']==980 and not d['stop']
    assert x['approved_experiment_extension']==old


def test_parallel_approval_rejects_scope_or_hardware_or_history_changes():
    for patch in [{'gpu_hours':501},{'usd_cap':1001},{'max_concurrent_gpus':9},{'max_concurrent_gpus':True},
                  {'one_gpu_per_pod':False},{'allowed_gpu':'A100'}, {'parallel_execution_approved':False},
                  {'scientific_scope_unchanged':False},{'fixed_cohort_and_sampling_unchanged':False},
                  {'previous_experiment_approval_sha256':'wrong'},{'receipt_sha256':''}]:
        x=parallel_ledger();x['approved_parallel_extension'].update(patch)
        assert 'invalid_parallel_approval' in decision(x,0)['stop_reasons']
    x=parallel_ledger();x['approved_experiment_extension']['receipt_sha256']='wrong'
    assert 'invalid_parallel_approval' in decision(x,0)['stop_reasons']


def test_total_gpu_hours_counts_gpu_count_and_simultaneous_pods():
    x=parallel_ledger()
    x['resources']=[{'kind':'pod','started_epoch':0,'absent_epoch':100*3600,'gpu_count':4,'upper_rate_usd':1},
                    {'kind':'pod','started_epoch':0,'absent_epoch':100*3600,'upper_rate_usd':1}]
    d=decision(x,100*3600)
    assert d['gpu_hours']==500 and 'GPU_hours' in d['stop_reasons']
    assert d['upper_usd']==200  # A recorded rate is already a whole-pod rate.


def test_fleet_affordability_uses_concurrent_total_and_retained_volume_rate():
    from gearshift.coding_control import remaining_budget_seconds
    x=parallel_ledger();x['prior_upper_usd']=100
    x['resources']=[{'kind':'volume','started_epoch':0,'upper_rate_usd':.40}]
    seconds=remaining_budget_seconds(x,0,dispatch_gpu_count=8)
    assert seconds==pytest.approx(880/(8*5.56+.40)*3600)
    with pytest.raises(ValueError):remaining_budget_seconds(x,0,dispatch_gpu_count=9)
    with pytest.raises(ValueError):remaining_budget_seconds(x,0,dispatch_gpu_count=8,dispatch_upper_rate_usd=5.56)


def test_parallel_concurrency_and_dollar_reserve_enforced_independently():
    x=parallel_ledger();x['resources']=[{'kind':'pod','started_epoch':0,'upper_rate_usd':5.52} for _ in range(9)]
    assert 'concurrent_gpu_limit' in decision(x,0)['stop_reasons']
    x=parallel_ledger();x['prior_upper_usd']=980
    assert 'total_cost' in decision(x,0)['stop_reasons']


def test_provider_grace_never_permits_new_dispatch():
    from gearshift.coding_control import remaining_budget_seconds
    x=parallel_ledger();x.update(provider_state_unknown=True,provider_unknown_grace=True)
    d=decision(x,0)
    assert not d['stop'] and d['dispatch_blocked']
    with pytest.raises(ValueError,match='provider_state_unknown'):remaining_budget_seconds(x,0)


def storage_intent():
    from gearshift.coding_control import PREFIX
    return {'purpose':'storage_inspection','started_epoch':0,'deadline_epoch':3600,
        'controller_id':'inspect','resource_name':PREFIX+'inspect','volume_id':'retained',
        'maximum_cpu_pods':1,'gpu_count':0,'maximum_hourly_usd':.50,'read_only':True,
        'model_inference_allowed':False,'approval_sha256':parallel_approval()['receipt_sha256'],
        'receipt_sha256':'inspection-bound'}


def cpu_resource():
    intent=storage_intent()
    return {'kind':'pod','id':'cpu','name':intent['resource_name'],'started_epoch':0,'upper_rate_usd':.50,
        'gpu_count':0,'controller_id':'inspect','volume_id':'retained','storage_inspection_intent_sha256':intent['receipt_sha256']}


def test_one_bound_cpu_inspection_charges_cost_but_zero_gpu_time_with_eight_h200s():
    x=parallel_ledger();x['approved_storage_inspection']=storage_intent()
    x['resources']=[cpu_resource()]+[{'kind':'pod','started_epoch':0,'gpu_count':1,'upper_rate_usd':5.52} for _ in range(8)]
    d=decision(x,1800)
    assert not d['stop'] and d['gpu_hours']==4 and d['active_gpu_count']==8
    assert d['active_cpu_storage_inspection_count']==1 and d['upper_usd']==22.33
    x['resources'][0]['absent_epoch']=1800
    d=decision(x,4000);assert d['active_cpu_storage_inspection_count']==0 and not d['stop']


def test_cpu_inspection_has_strict_identity_duration_and_no_inference():
    for patch in [{'read_only':False},{'model_inference_allowed':True},{'deadline_epoch':3601},
                  {'maximum_cpu_pods':2},{'maximum_hourly_usd':.51},{'approval_sha256':'wrong'}]:
        x=parallel_ledger();x['approved_storage_inspection']={**storage_intent(),**patch};x['resources']=[cpu_resource()]
        assert 'unauthorized_cpu_storage_inspection' in decision(x,1)['stop_reasons']
    x=parallel_ledger();x['approved_storage_inspection']=storage_intent();x['resources']=[cpu_resource()]
    assert 'storage_inspection_deadline' in decision(x,3600)['stop_reasons']
    x['resources'][0]['storage_inspection_intent_sha256']='changed'
    assert 'unauthorized_cpu_storage_inspection' in decision(x,1)['stop_reasons']
    x['resources']=[cpu_resource(),cpu_resource()]
    assert 'cpu_storage_inspection_count' in decision(x,1)['stop_reasons']


def test_unapproved_zero_gpu_pod_is_never_a_research_hardware_exception():
    x=parallel_ledger();x['resources']=[cpu_resource()]
    d=decision(x,1800)
    assert d['gpu_hours']==0 and d['upper_usd']==.25 and d['stop']
