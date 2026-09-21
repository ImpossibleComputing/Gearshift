"""Regression checks for the owner's amended dollar-only authority."""
import pytest
from gearshift.coding_confirmation_lease import verify_lease


def lease():
    return {'experiment_id': 'confirmation_test', 'pod_id': 'pod1',
            'allocation_epoch': 1000., 'deadline_epoch': 4600.,
            'upper_hourly_usd': 5., 'gpu_count': 1,
            'baseline_usd': 500., 'baseline_gpu_hours': 600.,
            'total_cap_usd': 2500, 'total_cap_gpu_hours': None,
            'cleanup_reserve_usd': 40, 'other_reserved_usd': 0,
            'allowed_result_root': '/workspace/project/results/confirmation_test',
            'control_relative': 'allocations/pod1'}


def test_soft_target_not_a_kill_switch_and_old_hour_limit_removed():
    row = lease(); row['baseline_usd'] = 2000
    result = verify_lease(row)
    assert result['soft_target_exceeded'] is True
    assert result['reserved_total_usd'] == 2045
    assert result['reserved_total_gpu_hours'] == 601


def test_all_other_reservations_and_cleanup_count_against_hard_cap():
    row = lease(); row['other_reserved_usd'] = 1955
    with pytest.raises(ValueError, match='hard ceiling'):
        verify_lease(row)
    row['other_reserved_usd'] -= 1
    assert verify_lease(row)['reserved_total_usd'] == 2499


@pytest.mark.parametrize('field,value', [('total_cap_usd', 1000), ('total_cap_gpu_hours', 500)])
def test_obsolete_authority_is_never_silently_reintroduced(field, value):
    row = lease(); row[field] = value
    with pytest.raises(ValueError, match='authority'):
        verify_lease(row)


def test_cpu_allocation_is_guarded_without_invented_gpu_hours():
    row = lease(); row['gpu_count'] = 0; row['upper_hourly_usd'] = .736
    result = verify_lease(row)
    assert result['reserved_total_gpu_hours'] == 600
    assert result['reserved_total_usd'] == 540.736


def test_guard_cannot_release_another_allocation():
    row = lease(); row['control_relative'] = 'allocations/some_other_pod'
    with pytest.raises(ValueError, match='this allocation'):
        verify_lease(row)
