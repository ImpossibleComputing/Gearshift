"""Completion distinguishes a failed allowance read from a verified resource stop."""
def transient_usage_check_failure(queue):
    status=queue.get('status',{})
    guard=status.get('usage_guard_stop') or {}
    return (queue.get('state')=='stopped_with_pending_packets'
        and bool(guard.get('error')) and not guard.get('windows')
        and not status.get('service_blocked'))
