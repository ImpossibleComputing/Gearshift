"""Validate a fixed completed checkpoint before a separately authorized restart."""
import json, math
from pathlib import Path
from gearshift.coding_control import sha
from gearshift.coding_reuse import compatible_parent, verify_completed


def validate_checkpoint(parent_root, root, expected_count=6):
    parent_root, root = Path(parent_root), Path(root)
    identity = compatible_parent(parent_root, root)
    tasks = json.loads((root/'data/coding_pilot_v1/visible/development.json').read_text())
    if len(tasks) != 40 or len({t['task_id'] for t in tasks}) != 40:
        raise ValueError('The fixed forty-task cohort is required')
    wanted = [t['task_id'] for t in tasks[:expected_count]]
    markers = list((parent_root/'tasks').glob('*/complete.json'))
    if len(markers) != expected_count:
        raise ValueError('Unexpected completed checkpoint membership')
    folders = {p.name for p in (parent_root/'tasks').iterdir()}
    if folders != {tid.replace('/', '__') for tid in wanted}:
        raise ValueError('Unexpected or partial task outside the completed checkpoint')
    rows = [verify_completed(parent_root/'tasks'/tid.replace('/', '__'), identity, tid) for tid in wanted]
    progress = json.loads((parent_root/'progress.json').read_text())
    counts = {arm: sum(r['pass'][arm] for r in rows) for arm in ('A','B','D')}
    if (progress['completed_tasks'], progress['total_tasks'], progress['passes']) != (expected_count, 40, counts):
        raise ValueError('Checkpoint progress differs from verified outputs')
    for label, key in [('source_cap_count','source_capped'),('small_cap_count','small_capped')]:
        if progress[label] != sum(r[key] for r in rows):
            raise ValueError('Checkpoint cap counts differ')
    elapsed = progress['observed_baseline_seconds_including_parent']
    forecast = progress['remaining_baseline_forecast_seconds']
    if not math.isfinite(elapsed) or elapsed <= 0 or not math.isclose(forecast, elapsed/expected_count*(40-expected_count), rel_tol=1e-9):
        raise ValueError('Checkpoint forecast is inconsistent')
    return {'parent_identity':identity, 'task_ids':wanted, 'counts':counts,
            'completed_tasks':expected_count, 'observed_seconds':elapsed,
            'remaining_forecast_seconds':forecast,
            'complete_sha256':{tid:sha(parent_root/'tasks'/tid.replace('/','__')/'complete.json') for tid in wanted}}
