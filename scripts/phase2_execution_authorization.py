"""Execution authorization, separate from all scientific experiment settings."""
import datetime as dt
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_FILE = ROOT / 'configs/phase2_execution_authorization.json'
LEGACY_SHUTDOWN_UTC = '2026-09-15T08:41:00Z'
LEGACY_END_UTC = '2026-09-15T08:51:00Z'


def execution_authorization(path=AUTHORIZATION_FILE):
    if not path.exists():
        return dict(shutdown_deadline_utc=LEGACY_SHUTDOWN_UTC, authorization_end_utc=LEGACY_END_UTC, authorization_type='original_24_hours')
    data = json.loads(path.read_text())
    if (data.get('schema') != 1 or data.get('authorization_type') != 'owner_removed_time_cutoff'
            or not data.get('owner_message') or not data.get('recorded_utc')
            or data.get('shutdown_deadline_utc', False) is not None
            or data.get('authorization_end_utc', False) is not None
            or data.get('hard_cap_usd') != 1000 or data.get('shutdown_threshold_usd') != 900
            or data.get('max_allocated_hourly_usd') != 20):
        raise ValueError('Invalid owner authorization or changed spending safeguards')
    return data


AUTHORIZATION = execution_authorization()
SHUTDOWN_UTC = AUTHORIZATION['shutdown_deadline_utc']
AUTHORIZATION_END_UTC = AUTHORIZATION['authorization_end_utc']
DEADLINE = math.inf if SHUTDOWN_UTC is None else dt.datetime.fromisoformat(SHUTDOWN_UTC.replace('Z', '+00:00')).timestamp()


def deadline_epoch():
    return None if math.isinf(DEADLINE) else DEADLINE


def remaining_timeout():
    import time
    return None if math.isinf(DEADLINE) else max(.1, DEADLINE - time.time())
