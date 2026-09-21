#!/usr/bin/env python3
"""Reserve matching generation capacity on a clean public regional volume.

Placement only: scientific workers require separate verified staging/resume proof.
No failed/ambiguous POST is retried automatically. A known volume is preserved
before validation; protected volumes are never deleted or attached by this helper.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import coding_confirmation_dispatch as base
from gearshift.coding_confirmation_lease import atomic_json, sha256_file, verify_lease

REMOTE = '/workspace/GearshiftConfirmationPrimary'
REGIONS = {'US-CO-1', 'US-GA-2', 'CA-MTL-3', 'EU-FR-1', 'EUR-IS-4', 'EUR-IS-5'}
PROTECTED_VOLUMES = {'lgk3howszi', 'o0ndo35b3f', 'jj2zyi9yrc', 'r4q5tbmdb9'}
SAFE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
CAPACITY_DETAIL = 'There are no longer any instances available with the requested specifications. Please refresh and try again.'


def read(path): return json.loads(Path(path).read_text())
def finite(value): return type(value) in (int, float) and math.isfinite(value) and value >= 0


def inventory():
    rows = []; ids = set(); cursors = set(); cursor = None
    for _ in range(100):
        path = 'pods?includeClusterPods=true&limit=100'
        if cursor: path += '&cursor=' + urllib.parse.quote(cursor, safe='')
        value = base.api(path)
        if not isinstance(value, dict) or not isinstance(value.get('pods'), list) or not isinstance(value.get('pagination'), dict):
            raise ValueError('Ambiguous provider pod inventory')
        for row in value['pods']:
            if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not SAFE_ID.fullmatch(row['id']) or row['id'] in ids:
                raise ValueError('Invalid/duplicate provider pod identity')
            ids.add(row['id']); rows.append(row)
        page = value['pagination']
        if page.get('hasNextPage') is False: return rows
        cursor = page.get('nextCursor')
        if page.get('hasNextPage') is not True or not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise ValueError('Incomplete provider pagination')
        cursors.add(cursor)
    raise ValueError('Provider inventory exceeds bounded pagination')


def capacity_rejected(folder):
    failure = read(folder/'creation_failure.json') if (folder/'creation_failure.json').is_file() else {}
    if failure.get('http_status') == 400 and failure.get('explicit_capacity_rejection') is True: return True
    prior = read(folder/'provider_rejection.json') if (folder/'provider_rejection.json').is_file() else {}
    if prior.get('http_status') == 400:
        try:
            if json.loads(prior.get('body', ''))['detail'] == CAPACITY_DETAIL: return True
        except (ValueError, KeyError, TypeError): pass
    # Existing legacy dispatcher omitted the 400 response body. Accept its
    # explicit historical reconciliation, but still require a fresh full list.
    old = read(folder/'ambiguous_creation.json') if (folder/'ambiguous_creation.json').is_file() else {}
    resolution = read(folder/'reconciliation.json') if (folder/'reconciliation.json').is_file() else {}
    return (old.get('http_status') == 400 and resolution.get('no_pod_created') is True
            and resolution.get('matching_pods') == [] and bool(resolution.get('observed_at_utc')))


def volume_inventory():
    """REST v2's list-network-volumes contract is a full, nonpaginated list."""
    value = base.api('network-volumes')
    if not isinstance(value, dict) or not isinstance(value.get('networkVolumes'), list):
        raise ValueError('Ambiguous full network-volume inventory')
    if 'pagination' in value and value['pagination'].get('hasNextPage') is not False:
        raise ValueError('Network-volume inventory is not complete')
    rows = []; seen = set()
    for item in value['networkVolumes']:
        if (not isinstance(item, dict) or not isinstance(item.get('id'), str) or not SAFE_ID.fullmatch(item['id']) or
            item['id'] in seen or not isinstance(item.get('name'), str) or not isinstance(item.get('dataCenter'), str) or
            not finite(item.get('size')) or item['size'] <= 0):
            raise ValueError('Invalid/duplicate network-volume inventory identity')
        seen.add(item['id']); rows.append({k: item[k] for k in ('id', 'name', 'size', 'dataCenter', 'type') if k in item})
    return sorted(rows, key=lambda x: x['id'])


def failed_volume_identity(folder):
    required = ['volume_request.json', 'volume_creation_failure.json', 'reservation.json']
    if folder.is_symlink() or any((folder/p).is_symlink() for p in required): raise ValueError('Linked volume reconciliation evidence')
    if any((folder/p).exists() for p in ('lease.json', 'create_request.json', 'allocated_pod_identity.json', 'pod.json', 'volume.json')):
        raise ValueError('This request has a known volume or pod path; reconcile its actual resource instead')
    intent, failure, reserve = [read(folder/p) for p in required]
    if (intent.get('action') != 'create' or intent.get('volume_id') is not None or
        intent.get('name') != 'gearshift-confirmation-public-' + folder.name or intent.get('size') != 200 or
        intent.get('region') not in REGIONS or reserve.get('region') != intent['region'] or
        not finite(failure.get('epoch')) or failure.get('reconciliation_required') is not True or
        reserve.get('lease', {}).get('experiment_id') != base.EXPERIMENT):
        raise ValueError('Failed volume request identity is not established')
    return intent, failure, {p: sha256_file(folder/p) for p in required}


def validate_volume_reconciliation(folder, current):
    intent, failure, hashes = failed_volume_identity(folder)
    path = folder/'volume_creation_reconciliation.json'
    if path.is_symlink(): raise ValueError('Linked volume reconciliation receipt')
    value = read(path); observations = value.get('observations', [])
    if (value.get('schema') != 1 or value.get('experiment_id') != base.EXPERIMENT or value.get('allocation') != folder.name or
        value.get('inputs') != hashes or value.get('request') != intent or
        value.get('outcome') != 'no_matching_volume_observed' or value.get('creation_outcome_still_unknown') is not True or
        value.get('other_region_attempt_permitted') is not True or value.get('retry_original_request_permitted') is not False or
        value.get('potential_storage_gb_reserved') != 200 or len(observations) != 2):
        raise ValueError('Volume reconciliation identity/uncertainty differs')
    previous = None
    for observation in observations:
        rows = observation.get('volumes')
        if (observation.get('api_path') != 'network-volumes' or observation.get('full_inventory') is not True or
            not finite(observation.get('observed_epoch')) or observation['observed_epoch'] < failure['epoch'] + 60 or
            not isinstance(rows, list) or observation.get('inventory_sha256') != hashlib.sha256(encoded(rows)).hexdigest() or
            any(row.get('name') == intent['name'] for row in rows)):
            raise ValueError('Saved full volume-absence observation differs')
        if previous is not None and previous != rows: raise ValueError('Saved volume inventories were inconsistent')
        previous = rows
    if any(row['name'] == intent['name'] for row in current):
        raise ValueError('Previously absent volume is now visible; preserve/reconcile it before continuing')
    return value


def encoded(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def rejected_pod_inputs(folder):
    """Only a definite HTTP400 response can use the repeated-absence path."""
    names = ['reservation.json', 'create_request.json', 'creation_failure.json']
    if folder.is_symlink() or any((folder/n).is_symlink() for n in names): raise ValueError('Linked rejected request')
    if any((folder/n).exists() for n in ['lease.json', 'allocated_pod_identity.json', 'pod.json']):
        raise ValueError('Known pod identity must be reconciled directly')
    reserve, request, failure = [read(folder/n) for n in names]
    if (reserve.get('lease', {}).get('experiment_id') != base.EXPERIMENT or
            request.get('name') != 'gearshift-confirmation-' + folder.name or
            failure.get('http_status') != 400 or not finite(failure.get('epoch')) or
            failure.get('reconciliation_required') is not True):
        raise ValueError('Repeated absence cannot resolve transport or server uncertainty')
    return request, failure, {n: sha256_file(folder/n) for n in names}


def validate_rejected_absence(request, failure, observation):
    rows = observation.get('pods')
    if (observation.get('full_inventory') is not True or not finite(observation.get('epoch')) or
            observation['epoch'] < failure['epoch'] + 60 or not isinstance(rows, list) or
            any(not isinstance(p, dict) or not SAFE_ID.fullmatch(p.get('id', '')) for p in rows) or
            len({p['id'] for p in rows}) != len(rows) or
            any(p.get('name') == request['name'] for p in rows)):
        raise ValueError('Rejected pod lacks a complete, delayed absence observation')


def rejected_pod_reconciled(folder):
    path = folder/'rejected_request_reconciliation.json'
    if not path.is_file(): return False
    if path.is_symlink(): raise ValueError('Linked rejection reconciliation')
    request, failure, hashes = rejected_pod_inputs(folder); value = read(path)
    observations = value.get('observations', [])
    if (value.get('inputs') != hashes or value.get('outcome') != 'http400_rejection_repeatedly_absent' or
            value.get('reason_classified_as_capacity') is not False or value.get('retry_original_request_permitted') is not False or
            len(observations) != 2): raise ValueError('Rejected pod reconciliation identity differs')
    for observation in observations: validate_rejected_absence(request, failure, observation)
    if observations[1]['epoch'] < observations[0]['epoch'] + 60:
        raise ValueError('Rejected pod absence observations are not separated by sixty seconds')
    return True


def reconcile_rejected_pod(name):
    """Two separated complete inventories after HTTP400; never replay the POST."""
    if not isinstance(name, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,99}', name):
        raise ValueError('Unsafe rejected allocation name')
    with (base.EVIDENCE/'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        folder = base.EVIDENCE/name; request, failure, hashes = rejected_pod_inputs(folder)
        current = {'epoch': time.time(), 'full_inventory': True, 'pods': [base.safe(p) for p in inventory()]}
        validate_rejected_absence(request, failure, current)
        if rejected_pod_reconciled(folder): return {'reconciled': True, 'new_inventory_still_absent': True}
        first = folder/'first_absence_inventory.json'
        if not first.exists():
            atomic_json(first, current); return {'reconciled': False, 'first_observation_saved': True, 'minimum_wait_seconds': 60}
        if first.is_symlink(): raise ValueError('Linked first absence observation')
        earlier = read(first); validate_rejected_absence(request, failure, earlier)
        if current['epoch'] < earlier['epoch'] + 60: raise ValueError('Wait sixty seconds between complete inventories')
        receipt = {'inputs': hashes, 'outcome': 'http400_rejection_repeatedly_absent',
            'reason_classified_as_capacity': False, 'retry_original_request_permitted': False,
            'observations': [earlier, current]}
        atomic_json(folder/'rejected_request_reconciliation.json', receipt)
        assert rejected_pod_reconciled(folder)
        return {'reconciled': True, 'new_inventory_still_absent': True, 'provider_error_reason_unknown': True}


def reconcile_volume_create(name):
    """Read provider metadata only; preserve unknown outcome and its storage slot."""
    if not isinstance(name, str) or not name.replace('_', '').replace('-', '').isalnum() or len(name) > 100:
        raise ValueError('Invalid allocation name')
    with (base.EVIDENCE/'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        folder = base.EVIDENCE/name; target = folder/'volume_creation_reconciliation.json'
        if target.exists(): raise ValueError('Reconciliation receipt already exists; preserve it and verify live state')
        intent, failure, hashes = failed_volume_identity(folder)
        if time.time() < failure['epoch'] + 60: raise ValueError('Allow at least sixty seconds after the failed create before absence reconciliation')
        observations = []
        for i in range(2):
            if i: time.sleep(1)
            rows = volume_inventory()
            if any(row['name'] == intent['name'] for row in rows):
                raise ValueError('Matching volume exists; retain its identity and reconcile it, do not claim absence')
            observations.append({'api_path': 'network-volumes', 'full_inventory': True, 'observed_epoch': time.time(),
                'volumes': rows, 'inventory_sha256': hashlib.sha256(encoded(rows)).hexdigest()})
        if observations[0]['volumes'] != observations[1]['volumes']: raise ValueError('Provider volume inventories changed; no reconciliation recorded')
        receipt = {'schema': 1, 'experiment_id': base.EXPERIMENT, 'allocation': name, 'inputs': hashes,
            'request': intent, 'observations': observations, 'outcome': 'no_matching_volume_observed',
            'creation_outcome_still_unknown': True, 'original_failure_reason_recovered': False,
            'other_region_attempt_permitted': True, 'retry_original_request_permitted': False,
            'potential_storage_gb_reserved': 200, 'provider_mutations_performed': False,
            'note': 'Two complete later inventories show no matching name. This does not reconstruct the lost provider error or prove that creation never occurred. Retain this region\'s potential storage slot, and recheck current absence before another-region allocation.'}
        atomic_json(target, receipt)
        validate_volume_reconciliation(folder, observations[-1]['volumes'])
        return receipt


def sanitized_cli_failure(exc):
    """Keep structured CLI failure details without copying raw auth-bearing output."""
    streams = {}
    for name in ('stdout', 'stderr'):
        raw = getattr(exc, name, None)
        if raw is None and name == 'stdout': raw = getattr(exc, 'output', None)
        if raw is None: continue
        raw = raw.encode() if isinstance(raw, str) else raw
        item = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        try: value = json.loads(raw)
        except (ValueError, UnicodeError): value = None
        if isinstance(value, dict):
            for key in ('code', 'status', 'error', 'detail', 'title'):
                part = value.get(key)
                if type(part) in (int, float, bool) or part is None:
                    if part is not None and (type(part) is not float or math.isfinite(part)): item[key] = part
                elif isinstance(part, str):
                    part = re.sub(r'(?:rpa_|hf_|sk-)[A-Za-z0-9_-]{12,}', '[REDACTED]', part)
                    part = re.sub(r'(?i)(Bearer\s+)\S+', r'\1[REDACTED]', part)
                    part = re.sub(r'(?i)((?:api[_-]?key|authorization|password|token)\s*[=:]\s*)[^\s,;]+', r'\1[REDACTED]', part)
                    item[key] = part[:2000]
        else: item['unstructured_text_omitted'] = True
        streams[name] = item
    return streams


def history(rows):
    """Count issued leases forever; unresolved allocations block new names."""
    leases = []; proofs = []; names = {p.get('name') for p in rows}; known_ids = set(); volumes = None
    for folder in sorted(base.EVIDENCE.iterdir()):
        if not folder.is_dir(): continue
        if folder.is_symlink(): raise ValueError('Linked allocation evidence')
        path = folder/'lease.json'
        if path.exists():
            if path.is_symlink(): raise ValueError('Linked historical lease')
            lease = read(path); proof = verify_lease(lease)
            if lease['experiment_id'] != base.EXPERIMENT or lease['pod_id'] in known_ids:
                raise ValueError('Historical lease experiment/identity collision')
            leases.append(lease); known_ids.add(lease['pod_id']); proofs.append(proof)
            continue
        if not (folder/'reservation.json').exists(): continue
        request = read(folder/'create_request.json') if (folder/'create_request.json').is_file() else None
        if request is not None:
            if request.get('name') in names: raise ValueError('A previously failed/unknown request now exists at provider')
            if not capacity_rejected(folder) and not rejected_pod_reconciled(folder):
                raise ValueError('Unresolved prior pod creation: ' + folder.name)
        else:
            volume_intent = read(folder/'volume_request.json') if (folder/'volume_request.json').is_file() else {}
            if volume_intent.get('action') == 'get' and (folder/'volume_lookup_failure.json').is_file(): continue
            if (folder/'volume_creation_reconciliation.json').is_file():
                if volumes is None: volumes = volume_inventory()
                validate_volume_reconciliation(folder, volumes)
                continue
            raise ValueError('Unresolved prior reservation or volume creation: ' + folder.name)
    for pod in rows:
        if str(pod.get('name', '')).startswith('gearshift-confirmation-') and pod['id'] not in known_ids:
            raise ValueError('Experiment pod has no reconciled immutable lease')
    return leases, proofs


def public_volumes():
    known = {}
    for p in sorted(base.EVIDENCE.glob('*/volume.json')):
        if p.is_symlink() or p.parent.is_symlink(): raise ValueError('Linked volume evidence')
        value = read(p)
        # Only the original allocation's exact create-name is provenance;
        # receipts copied by later --volume-id attempts cannot invent an origin.
        if value.get('name') != 'gearshift-confirmation-public-' + p.parent.name: continue
        reserve = read(p.parent/'reservation.json')
        region = reserve.get('region')
        if (region not in REGIONS or value.get('dataCenterId') != region or value.get('size') != 200 or
                not isinstance(value.get('id'), str) or not SAFE_ID.fullmatch(value['id']) or value['id'] in PROTECTED_VOLUMES):
            raise ValueError('Original public volume provenance is invalid')
        if value['id'] in known and known[value['id']]['volume'] != value: raise ValueError('Conflicting public volume origins')
        known[value['id']] = {'volume': value, 'path': str(p.relative_to(ROOT)), 'sha256': sha256_file(p)}
    if len(known) > len(REGIONS): raise ValueError('Regional volume storage envelope exhausted')
    return known


def reconciled_budget(leases, rows, absent, closed, now, rate, duration):
    """Conserve dollars across immutable deadlines, not historical GPU slots.

Every old live reservation remains fully funded. Released compute is bounded
through a provider-confirmed absence, and absent leases without an earlier
receipt are conservatively charged through this observation. Storage and
cleanup retain separate allowances. No existing lease or guard is changed.
"""
    live = {p['id']: p for p in rows}
    if any(not finite(v) or v <= 0 for v in (now, rate, duration)):
        raise ValueError('Invalid reconciled allocation bounds')
    if set(closed) & set(live): raise ValueError('Released compute receipt still names a live pod')
    if set(absent) != {l['pod_id'] for l in leases if l['pod_id'] not in live}:
        raise ValueError('Every absent historical pod requires a fresh absence proof')
    accrued = 0.; remaining = 0.; details = []
    for lease in leases:
        verify_lease(lease)
        pid = lease['pod_id']; start = lease['allocation_epoch']; upper = lease['upper_hourly_usd']
        if now < start: raise ValueError('Budget observation predates allocation')
        if pid in live:
            pod = live[pid]
            if not finite(pod.get('cost')) or pod['cost'] <= 0:
                raise ValueError('Live provider compute rate is unavailable')
            if now >= lease['deadline_epoch']:
                raise ValueError('Expired pod remains billable; resolve its guard before expanding')
            # An unexpectedly higher actual rate must still be fully reserved.
            upper = max(upper, pod['cost'])
            end = now; future = (lease['deadline_epoch'] - now) / 3600 * upper
        else:
            proof = absent[pid]
            if proof.get('http_status') != 404 or proof.get('pod_id') != pid or proof.get('observed_epoch') != now:
                raise ValueError('Absent pod proof differs from this reconciliation')
            old = closed.get(pid)
            end = now
            if old is not None:
                if (old.get('pod_id') != pid or not finite(old.get('observed_absent_epoch')) or
                        not start <= old['observed_absent_epoch'] <= now or not finite(old.get('compute_estimate_usd'))):
                    raise ValueError('Historical released-compute receipt differs')
                end = old['observed_absent_epoch']
            future = 0.
        cost = (end - start) / 3600 * upper
        if pid in closed: cost = max(cost, closed[pid]['compute_estimate_usd'])
        accrued += cost; remaining += future
        details.append({'pod_id': pid, 'accrued_upper_usd': cost, 'remaining_lease_upper_usd': future,
                        'upper_hourly_usd': upper, 'charged_through_epoch': end, 'live': pid in live})
    storage = 25.
    baseline = base.BASELINE + accrued
    other = remaining + storage
    reserved = baseline + other + duration * rate + 40
    if reserved >= 2500: raise ValueError('Reconciled new reservation reaches the hard dollar ceiling')
    return {'baseline_usd': baseline, 'other_reserved_usd': other, 'reserved_total_usd': reserved,
        'historical_baseline_usd': base.BASELINE, 'retained_storage_allowance_usd': storage,
        'cleanup_reserve_usd': 40, 'old_allocations': details,
        'existing_leases_unchanged': True, 'historical_gpu_slot_limit_applied': False}


def allocate(name, region, count, *, volume_id=None, reconcile_budget=False):
    if (not isinstance(name, str) or not name.replace('_', '').replace('-', '').isalnum() or len(name) > 100 or
            region not in REGIONS or type(count) is not int or count not in (1, 2, 4, 8)):
        raise ValueError('Invalid bounded regional allocation')
    if volume_id is not None and (not isinstance(volume_id, str) or not SAFE_ID.fullmatch(volume_id) or volume_id in PROTECTED_VOLUMES):
        raise ValueError('Protected or invalid regional volume')
    base.EVIDENCE.mkdir(parents=True, exist_ok=True)
    with (base.EVIDENCE / 'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        folder = base.EVIDENCE / name
        if folder.exists(): raise ValueError('Allocation exists; reconcile instead of duplicating')
        rows = inventory(); leases, proofs = history(rows)
        if not reconcile_budget and sum(x['gpu_count'] for x in leases) + count > 20:
            raise ValueError('Issued allocation envelope exhausted; explicit reconciliation required')
        if any(p.get('name') == 'gearshift-confirmation-' + name for p in rows):
            raise ValueError('Provider already has this named allocation')
        known = public_volumes()
        uncertain_regions = {read(p.parent/'volume_request.json')['region']
            for p in base.EVIDENCE.glob('*/volume_creation_reconciliation.json')}
        if region in uncertain_regions: raise ValueError('This region retains an unresolved possible volume; do not retry its creation')
        if volume_id is not None and volume_id not in known: raise ValueError('Reuse requires recorded original clean public volume provenance')
        same_region = [v for v in known.values() if v['volume']['dataCenterId'] == region]
        if volume_id is None and same_region: raise ValueError('Reuse the existing same-region public volume; no duplicate storage allocation')
        if volume_id is None and len(known) + len(uncertain_regions) >= len(REGIONS): raise ValueError('Regional volume storage envelope exhausted')
        if volume_id is not None and known[volume_id]['volume']['dataCenterId'] != region: raise ValueError('Known volume belongs to another region')
        epoch = time.time(); rate = 5.55 * count; duration = 12
        # The twenty GPU slots are already fully reserved. Preserve any later
        # CPU-replacement reservation high-water mark rather than resetting it.
        high_water = max([base.BASELINE + base.ENVELOPE + 40, *[p['reserved_total_usd'] for p in proofs]])
        baseline = max([base.BASELINE, *[v['baseline_usd'] for v in leases]])
        reconciliation = None
        if reconcile_budget:
            # Read each full live identity/rate and confirm every disappearance
            # independently. Missing/ambiguous provider responses forbid spend.
            rows = [base.safe(base.api('pods/' + p['id'])) for p in rows]
            present = {p['id'] for p in rows}; absent = {}
            for old in leases:
                if old['pod_id'] in present: continue
                try: base.api('pods/' + old['pod_id'])
                except Exception as exc:
                    if getattr(exc, 'code', None) != 404: raise
                else: raise ValueError('Pod appeared after full inventory; reconcile before allocating')
                absent[old['pod_id']] = {'pod_id': old['pod_id'], 'http_status': 404, 'observed_epoch': None}
            epoch = time.time()
            for proof in absent.values(): proof['observed_epoch'] = epoch
            closed = {}
            for path in base.EVIDENCE.glob('*/final_compute_estimate.json'):
                value = read(path)
                if value['pod_id'] in closed: raise ValueError('Duplicate released-compute receipt')
                closed[value['pod_id']] = value
            reconciliation = reconciled_budget(leases, rows, absent, closed, epoch, rate, duration)
            baseline = reconciliation['baseline_usd']; high_water = reconciliation['reserved_total_usd']
            reconciliation.update(observed_epoch=epoch, fresh_absence_proofs=absent,
                provider_inventory=[base.safe(p) for p in rows],
                historical_lease_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in base.EVIDENCE.glob('*/lease.json')})
        lease = {'experiment_id': base.EXPERIMENT, 'pod_id': 'reservation_pending',
            'allocation_epoch': epoch, 'deadline_epoch': epoch + duration * 3600,
            'upper_hourly_usd': rate, 'gpu_count': count,
            'other_reserved_usd': high_water - baseline - 40 - duration * rate,
            'baseline_usd': baseline, 'baseline_gpu_hours': 91.24181750045884,
            'total_cap_usd': 2500, 'total_cap_gpu_hours': None, 'cleanup_reserve_usd': 40,
            'allowed_result_root': REMOTE + '/' + base.RESULT}
        reservation = verify_lease(lease)
        folder.mkdir()
        if reconciliation is not None: atomic_json(folder / 'budget_reconciliation.json', reconciliation)
        atomic_json(folder / 'reservation.json', {'lease': lease, 'reservation': reservation,
            'region': region, 'placement_only_no_numerical_change': True,
            'historical_reservation_high_water_usd': max([high_water, *[p['reserved_total_usd'] for p in proofs]]),
            'reservation_basis': 'fresh_reconciled_liabilities' if reconcile_budget else 'original_twenty_slot_envelope',
            'regional_storage_limit': 'At most one recorded 200GB public volume per allowed region; retained storage remains in the existing storage allowance and subsequent cost reconciliation.'})
        atomic_json(folder / 'pre_create_inventory.json', [base.safe(p) for p in rows if p.get('status') == 'RUNNING'])
        atomic_json(folder / 'volume_request.json', {'action': 'create' if volume_id is None else 'get',
            'volume_id': volume_id, 'name': 'gearshift-confirmation-public-' + name, 'size': 200, 'region': region,
            'origin': known.get(volume_id)})
        try:
            if volume_id is None:
                volume = json.loads(subprocess.check_output([base.CLI, 'network-volume', 'create', '--name',
                    'gearshift-confirmation-public-' + name, '--size', '200', '--data-center-id', region], stderr=subprocess.PIPE))
            else:
                volume = json.loads(subprocess.check_output([base.CLI, 'network-volume', 'get', volume_id], stderr=subprocess.PIPE))
        except Exception as exc:
            atomic_json(folder / ('volume_creation_failure.json' if volume_id is None else 'volume_lookup_failure.json'),
                {'error_type': type(exc).__name__, 'returncode': getattr(exc, 'returncode', None),
                 'reconciliation_required': volume_id is None, 'volume_deletion_attempted': False, 'epoch': time.time(),
                 'sanitized_cli_output': sanitized_cli_failure(exc)})
            raise RuntimeError('Volume operation failed; preserve/reconcile any created volume') from None
        # Record even an unexpected successful response BEFORE validation, so
        # failed placement cannot hide an owned billable volume from cleanup.
        safe_volume = {k: volume[k] for k in ('id', 'name', 'size', 'dataCenterId') if k in volume} if isinstance(volume, dict) else {'invalid_response_type': type(volume).__name__}
        atomic_json(folder / 'volume.json', safe_volume)
        valid = (isinstance(volume, dict) and isinstance(volume.get('id'), str) and SAFE_ID.fullmatch(volume['id']) and
            volume['id'] not in PROTECTED_VOLUMES and volume.get('dataCenterId') == region and volume.get('size') == 200 and
            (volume.get('name') == 'gearshift-confirmation-public-' + name if volume_id is None else safe_volume == known[volume_id]['volume']))
        if not valid:
            atomic_json(folder / 'volume_scope_failure.json', {'reconciliation_required': True, 'volume_preserved': True})
            raise ValueError('Region, size or clean public volume provenance differs')
        request = {'name': 'gearshift-confirmation-' + name, 'image': base.IMAGE, 'disk': 40,
            'cloud': 'SECURE', 'dataCenterIds': [region], 'ports': ['22/tcp'], 'startSsh': True,
            'mounts': {'network': [{'volumeId': volume['id'], 'path': '/workspace'}]},
            'gpu': {'id': 'NVIDIA H200', 'count': count, 'minRamPerGpu': 96, 'minVcpuCountPerGpu': 8}}
        atomic_json(folder / 'create_request.json', request)
        try:
            pod = base.api('pods', 'POST', request)
            if not isinstance(pod, dict) or not isinstance(pod.get('id'), str) or not SAFE_ID.fullmatch(pod['id']):
                raise ValueError('Create response has no valid pod identity')
        except Exception as exc:
            status = getattr(exc, 'code', None); detail = None
            if hasattr(exc, 'read'):
                try: detail = json.loads(exc.read(4096))['detail']
                except (ValueError, KeyError, TypeError): pass
            explicit = status == 400 and detail == CAPACITY_DETAIL
            atomic_json(folder / 'creation_failure.json', {'error_type': type(exc).__name__,
                'http_status': status, 'explicit_capacity_rejection': explicit,
                'provider_message': CAPACITY_DETAIL if explicit else 'Response omitted; reconcile provider inventory.',
                'epoch': time.time(), 'reconciliation_required': True})
            raise RuntimeError('Regional allocation failed; see saved reconciliation requirement') from None
        atomic_json(folder / 'allocated_pod_identity.json', {'pod_id': pod['id'],
            'network_volume_id': volume['id'], 'created_epoch': time.time(), 'guard_arming_required': True})
        safe_pod = base.safe(pod)
        if 'cost' in safe_pod and not finite(safe_pod['cost']): safe_pod['cost'] = None
        atomic_json(folder / 'pod.json', safe_pod)
        actual_rate = pod.get('cost'); gpu = pod.get('gpu')
        if finite(actual_rate) and actual_rate > 0:
            bounded_rate = max(rate, actual_rate * 1.2 + .03 * count)
            # Spend is conserved; a higher observed rate shortens the deadline.
            lease['upper_hourly_usd'] = bounded_rate
            lease['deadline_epoch'] = epoch + duration * rate / bounded_rate * 3600
        if isinstance(gpu, dict) and type(gpu.get('count')) is int and 0 < gpu['count'] <= 8:
            lease['gpu_count'] = gpu['count']  # retain actual issued slots even on scope failure
        lease.update(pod_id=pod['id'], network_volume_id=volume['id'], control_relative='allocations/' + pod['id'])
        reservation = verify_lease(lease); atomic_json(folder / 'lease.json', lease)
        networks = pod.get('mounts', {}).get('network') if isinstance(pod.get('mounts'), dict) else None
        if (not isinstance(gpu, dict) or gpu.get('id') != 'NVIDIA H200' or gpu.get('count') != count or pod.get('cpu') is not None or
            pod.get('dataCenterId') != region or networks != [{'volumeId': volume['id'], 'path': '/workspace'}] or
            not finite(actual_rate) or actual_rate <= 0):
            atomic_json(folder / 'pod_scope_failure.json', {'pod_id': pod['id'], 'reconciliation_required': True,
                'arm_guard_before_remediation': True, 'do_not_stage_scientific_work': True, 'volume_preserved': True})
            raise ValueError('Created pod differs from regional scope; owned pod/lease retained for guarded reconciliation')
        print(json.dumps({'allocation': name, 'pod': base.safe(pod), 'reservation': reservation}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--name', required=True); p.add_argument('--region')
    p.add_argument('--count', type=int); p.add_argument('--volume-id'); p.add_argument('--reconcile-volume-create', action='store_true')
    p.add_argument('--reconcile-budget', action='store_true')
    p.add_argument('--reconcile-rejected-pod', action='store_true')
    a = p.parse_args()
    if a.reconcile_rejected_pod:
        if a.region is not None or a.count is not None or a.volume_id is not None or a.reconcile_budget or a.reconcile_volume_create:
            p.error('reconcile requires only --name')
        print(json.dumps(reconcile_rejected_pod(a.name), indent=2))
    elif a.reconcile_volume_create:
        if a.region is not None or a.count is not None or a.volume_id is not None or a.reconcile_budget: p.error('reconcile requires only --name')
        print(json.dumps(reconcile_volume_create(a.name), indent=2))
    else:
        if a.region is None or a.count is None: p.error('allocation requires --region and --count')
        allocate(a.name, a.region, a.count, volume_id=a.volume_id, reconcile_budget=a.reconcile_budget)
