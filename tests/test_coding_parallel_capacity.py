import copy

import pytest

from gearshift.coding_capacity_retry import NO_INSTANCES, confirmed_no_capacity, prepare_retry, scientific_payload
from gearshift.coding_control import PREFIX, digest


def fixture():
    plan = {'run_id': 'attempt_01', 'stage': 'cap_recovery', 'files': {'code.py': 'frozen'},
        'task_ids': ['unchanged'], 'gates': {'baseline': 'fixed'},
        'workers': [{'worker_id': 'worker_00', 'region': 'CA-MTL-3', 'task_ids': ['unchanged'],
            'worker_fields': {'seed': 123, 'source_history_sha256': 'parent'}, 'command': ['code.py']}]}
    failure = {'state': 'failed', 'error': 'Provider pod create failed: help text\n{"error":"' + NO_INSTANCES + '"}\n'}
    dispatch = {'passed': False, 'stage_identity': digest(plan), 'workers': [], 'errors': [failure['error']]}
    ledger = {'resources': [{'kind': 'volume', 'id': 'volume1', 'name': PREFIX + 'attempt_01-worker_00-storage', 'absent_epoch': 100}]}
    absence = {'verified': True, 'source': 'provider_lists', 'epoch': 110, 'pods': [], 'volumes': []}
    return plan, [failure], dispatch, ledger, absence


def retry(args, **kw):
    return prepare_retry(*args, new_run_id='attempt_02', regions=['US-TX-4'], now=120, **kw)


def test_capacity_retry_changes_only_region_and_run_id():
    args = fixture(); original = copy.deepcopy(args)
    plan, proof = retry(args)
    assert args == original
    assert digest(scientific_payload(plan)) == digest(scientific_payload(args[0]))
    assert plan['workers'][0]['worker_fields'] == args[0]['workers'][0]['worker_fields']
    assert proof['attempt_number'] == 2 and proof['no_gpu_ever_allocated_in_attempt']
    assert proof['new_plan_identity'] == digest(plan)


@pytest.mark.parametrize('mutation', ['ambiguous', 'success', 'allocated_gpu', 'live_volume', 'stale_absence', 'missing_absence', 'changed_identity', 'partial_outcome', 'other_error'])
def test_capacity_preparation_rejects_scientific_or_uncertain_retry(mutation):
    args = fixture(); plan, failures, dispatch, ledger, absence = args
    if mutation == 'ambiguous': failures[0]['error'] = 'Provider pod create failed: request timed out'
    elif mutation == 'success': failures[0]['state'] = 'complete'
    elif mutation == 'allocated_gpu': ledger['resources'].append({'kind': 'pod', 'id': 'pod1', 'name': PREFIX + 'attempt_01-worker_00', 'absent_epoch': 100})
    elif mutation == 'live_volume': absence['volumes'] = [{'id': 'volume1', 'name': PREFIX + 'attempt_01-worker_00-storage'}]
    elif mutation == 'stale_absence': absence['epoch'] = -10
    elif mutation == 'missing_absence': del ledger['resources'][0]['absent_epoch']
    elif mutation == 'changed_identity': plan['files']['code.py'] = 'changed'
    elif mutation == 'partial_outcome': dispatch['workers'] = [{'passed': True}]
    elif mutation == 'other_error': dispatch['errors'].append('Generation failed')
    with pytest.raises(ValueError): retry(args)


def test_no_second_region_change_after_three_total_attempts():
    args = fixture()
    second, proof = retry(args)
    failures = args[1]
    dispatch = {'passed': False, 'stage_identity': digest(second), 'workers': [], 'errors': [failures[0]['error']]}
    third, proof3 = prepare_retry(second, failures, dispatch, {'resources': []}, args[4],
        new_run_id='attempt_03', regions=['US-GA-2'], now=120, prior_proof=proof)
    assert proof3['attempt_number'] == 3
    dispatch['stage_identity'] = digest(third)
    with pytest.raises(ValueError, match='exhausted'):
        prepare_retry(third, failures, dispatch, {'resources': []}, args[4],
            new_run_id='attempt_04', regions=['CA-MTL-3'], now=120, prior_proof=proof3)


def test_modified_lineage_cannot_authorize_scientific_payload_changes():
    args = fixture(); second, proof = retry(args)
    second['workers'][0]['worker_fields']['seed'] = 999
    dispatch = {'passed': False, 'stage_identity': digest(second), 'workers': [], 'errors': [args[1][0]['error']]}
    with pytest.raises(ValueError, match='lineage'):
        prepare_retry(second, args[1], dispatch, {'resources': []}, args[4], new_run_id='attempt_03',
            regions=['US-GA-2'], now=120, prior_proof=proof)


def test_exact_error_required_not_merely_phrase_embedded_in_unknown_failure():
    assert confirmed_no_capacity(fixture()[1][0])
    assert not confirmed_no_capacity({'state': 'failed', 'error': 'Provider pod create failed: ' + NO_INSTANCES})
    assert not confirmed_no_capacity({'state': 'failed', 'error': 'Provider pod create failed: \n{"error":"' + NO_INSTANCES + '","ambiguous":true}'})


def test_short_exact_capacity_response_and_ambiguous_failure():
    import json
    error='failed to create pod: There are no longer any instances available.'
    assert confirmed_no_capacity({'state':'failed','error':'Provider pod create failed: help\n'+json.dumps({'error':error})})
    assert not confirmed_no_capacity({'state':'failed','error':'Provider pod create failed: help\n'+json.dumps({'error':error,'ambiguous':True})})
    assert not confirmed_no_capacity({'state':'failed','error':'Provider pod create failed: timeout after '+error})
