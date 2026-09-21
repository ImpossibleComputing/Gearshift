from scripts.phase2_review_recovery import recovery_action

def test_only_verified_reboot_restarts_judging():
    job={'state':'running','started_epoch':100}
    assert recovery_action(job,200,False)=='resume_after_reboot'
    assert recovery_action(job,50,False)=='attention_required'
    assert recovery_action(job,200,True)=='healthy'
    assert recovery_action({'state':'failed','started_epoch':100},200,False)=='attention_required'
    assert recovery_action({'state':'complete','started_epoch':100},200,False)=='ensure_finalizer'
