import pytest
from scripts.phase2_finish_review import terminal_ready

def test_finalization_requires_successful_job_and_recognized_queue_end():
    assert not terminal_ready({'state':'running'},{'state':'stopped_with_pending_packets'})
    assert terminal_ready({'state':'complete'},{'state':'all_exported_packets_attempted'})
    assert terminal_ready({'state':'complete'},{'state':'stopped_with_pending_packets'})
    with pytest.raises(ValueError):terminal_ready({'state':'failed'},{'state':'all_exported_packets_attempted'})
    with pytest.raises(ValueError):terminal_ready({'state':'complete'},{'state':'unknown'})


def test_failed_allowance_read_waits_but_verified_usage_stop_finishes():
    queue={'state':'stopped_with_pending_packets','status':{'usage_guard_stop':{'allowed':False,'error':'Read-only usage request failed'}}}
    assert not terminal_ready({'state':'complete'},queue)
    queue['status']['usage_guard_stop']={'allowed':False,'windows':[{'usedPercent':85}]}
    assert terminal_ready({'state':'complete'},queue)


def test_judge_service_denial_is_not_a_transient_allowance_read():
    from gearshift.phase2_completion_state import transient_usage_check_failure
    assert not transient_usage_check_failure({'state':'stopped_with_pending_packets','status':{'service_blocked':True,'usage_guard_stop':{'error':'unavailable'}}})
    assert not transient_usage_check_failure({'state':'all_exported_packets_attempted'})
