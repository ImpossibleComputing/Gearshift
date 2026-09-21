"""Dollar accounting must retain old live liabilities while reclaiming released capacity."""
import copy

import pytest

from scripts import coding_confirmation_regional_dispatch as s


def lease(pid='pod1'):
    return {'experiment_id': 'confirmation_test', 'pod_id': pid, 'allocation_epoch': 1000.,
        'deadline_epoch': 44200., 'upper_hourly_usd': 5.55, 'gpu_count': 1,
        'baseline_usd': 519., 'total_cap_usd': 2500, 'total_cap_gpu_hours': None,
        'cleanup_reserve_usd': 40, 'other_reserved_usd': 0,
        'allowed_result_root': '/workspace/project/results/test', 'control_relative': 'allocations/' + pid}


def test_more_than_twenty_historical_slots_use_dollars_and_do_not_edit_old_leases():
    leases = [lease('pod' + str(i)) for i in range(24)]; original = copy.deepcopy(leases)
    now = 4600.; absent = {l['pod_id']: {'pod_id': l['pod_id'], 'http_status': 404, 'observed_epoch': now} for l in leases}
    result = s.reconciled_budget(leases, [], absent, {}, now, 11.1, 12)
    assert result['reserved_total_usd'] == pytest.approx(s.base.BASELINE + 24 * 5.55 + 25 + 40 + 12 * 11.1)
    assert not result['historical_gpu_slot_limit_applied'] and leases == original


def test_live_deadlines_remain_fully_funded_and_higher_actual_rate_counts():
    l = lease(); now = 4600.
    result = s.reconciled_budget([l], [{'id': 'pod1', 'cost': 7.}], {}, {}, now, 5.55, 12)
    row = result['old_allocations'][0]
    assert row['accrued_upper_usd'] == 7. and row['remaining_lease_upper_usd'] == 77.
    assert result['reserved_total_usd'] == pytest.approx(s.base.BASELINE + 84 + 25 + 40 + 12 * 5.55)


def test_earlier_confirmed_release_preserves_charge_without_counting_idle_future():
    l = lease(); now = 8200.; absent = {'pod1': {'pod_id': 'pod1', 'http_status': 404, 'observed_epoch': now}}
    closed = {'pod1': {'pod_id': 'pod1', 'observed_absent_epoch': 2800., 'compute_estimate_usd': 2.3}}
    result = s.reconciled_budget([l], [], absent, closed, now, 5.55, 12)
    assert result['old_allocations'][0]['accrued_upper_usd'] == 2.775
    assert result['old_allocations'][0]['remaining_lease_upper_usd'] == 0


@pytest.mark.parametrize('problem', ['missing_absence', 'not404', 'stale_absence', 'wrong_pod', 'bad_cost', 'expired_live', 'closed_live', 'hard_cap'])
def test_uncertainty_or_insufficient_dollars_prevent_new_reservation(problem):
    l = lease(); now = 4600.; pods = []; absent = {'pod1': {'pod_id': 'pod1', 'http_status': 404, 'observed_epoch': now}}; closed = {}; rate = 5.55
    if problem == 'missing_absence': absent = {}
    elif problem == 'not404': absent['pod1']['http_status'] = 500
    elif problem == 'stale_absence': absent['pod1']['observed_epoch'] -= 1
    elif problem == 'wrong_pod': absent['pod1']['pod_id'] = 'other'
    elif problem in ['bad_cost', 'expired_live', 'closed_live']:
        pods = [{'id': 'pod1', 'cost': 5.55}]; absent = {}
        if problem == 'bad_cost': pods[0]['cost'] = None
        elif problem == 'expired_live': now = l['deadline_epoch']
        else: closed['pod1'] = {'pod_id': 'pod1', 'observed_absent_epoch': 2800., 'compute_estimate_usd': 2.3}
    else: rate = 200.
    with pytest.raises(ValueError): s.reconciled_budget([l], pods, absent, closed, now, rate, 12)
