"""Prepare a capacity-only retry; never allocate or retry a scientific outcome."""
import copy
import json
import re
import time

from gearshift.coding_control import PREFIX, digest
from gearshift.coding_parallel import safe_id

NO_INSTANCES = ('failed to create pod: There are no longer any instances available '
    'with the requested specifications. Please refresh and try again.')


def confirmed_no_capacity(record):
    """Recognize only the exact provider no-instance response, not timeouts."""
    if record.get('state') != 'failed':
        return False
    text = record.get('error', '')
    if not isinstance(text, str) or not text.startswith('Provider pod create failed:'):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        response = json.loads(lines[-1])
    except (ValueError, IndexError):
        return False
    return response in ({'error': NO_INSTANCES}, {'error': 'failed to create pod: There are no longer any instances available.'})


def scientific_payload(plan):
    """Remove only scheduling names/regions; retain every task, input and draw."""
    payload = copy.deepcopy(plan)
    payload.pop('run_id', None)
    for worker in payload['workers']:
        worker.pop('region', None)
    return payload


def prepare_retry(parent_plan, failures, dispatch, ledger, absence, *, new_run_id,
                  regions, prior_proof=None, now=None):
    """Return a new immutable plan and auditable proof; callers persist both.

    ``absence`` is a fresh provider listing receipt with complete ``pods`` and
    ``volumes`` lists. Every historical resource remains in the ledger. A single
    pod ever allocated under this attempt disqualifies this capacity-only path.
    """
    now = time.time() if now is None else now
    run_id = safe_id(parent_plan['run_id']); safe_id(new_run_id)
    if new_run_id == run_id:
        raise ValueError('Capacity retry requires a new immutable run ID')
    if dispatch.get('stage_identity') != digest(parent_plan) or dispatch.get('passed') is not False:
        raise ValueError('Recorded failed dispatch identity required')
    if dispatch.get('workers'):
        raise ValueError('Completed worker output cannot be retried as capacity')
    workers = parent_plan['workers']
    if len(failures) != len(workers) or not all(confirmed_no_capacity(x) for x in failures):
        raise ValueError('Exact recorded no-instance error required for every worker')
    if any(error not in [f['error'] for f in failures] for error in dispatch.get('errors', [])):
        raise ValueError('Dispatch has additional non-capacity failures')
    if len(dispatch.get('errors', [])) != len(failures):
        raise ValueError('Dispatch/failure receipts do not cover the whole attempt')
    if (absence.get('verified') is not True or absence.get('source') != 'provider_lists'
        or not 0 <= now - absence.get('epoch', 0) <= 120
        or not isinstance(absence.get('pods'), list) or not isinstance(absence.get('volumes'), list)):
        raise ValueError('Fresh complete provider-absence evidence required')
    prefix = PREFIX + run_id + '-'
    resources = [r for r in ledger.get('resources', []) if r.get('name', '').startswith(prefix)]
    if any(r.get('kind') == 'pod' for r in resources):
        raise ValueError('A GPU was allocated in this attempt; scientific recovery is required')
    if any('absent_epoch' not in r or r['absent_epoch'] > absence['epoch'] for r in resources):
        raise ValueError('Attempt resources are not confirmed absent')
    observed = absence['pods'] + absence['volumes']
    if any(r.get('name', '').startswith(prefix) for r in observed):
        raise ValueError('Provider still lists an attempt resource')
    if {r['id'] for r in resources} & {r['id'] for r in observed}:
        raise ValueError('Tracked attempt storage is still present')
    if len(regions) != len(workers) or any(not isinstance(r, str) or not re.fullmatch(r'[A-Z0-9-]+', r) for r in regions):
        raise ValueError('One explicit region per worker required')
    if regions == [w['region'] for w in workers]:
        raise ValueError('Capacity-only retry must use a different region')
    if prior_proof is None:
        attempt, first_identity = 2, digest(parent_plan)
    else:
        if (prior_proof.get('new_plan_identity') != digest(parent_plan)
            or prior_proof.get('scientific_payload_identity') != digest(scientific_payload(parent_plan))
            or prior_proof.get('capacity_only') is not True):
            raise ValueError('Capacity lineage does not bind this failed plan')
        attempt = prior_proof['attempt_number'] + 1
        first_identity = prior_proof['first_plan_identity']
    if attempt > 3:
        raise ValueError('Three bounded capacity attempts exhausted')
    new_plan = copy.deepcopy(parent_plan)
    new_plan['run_id'] = new_run_id
    for worker, region in zip(new_plan['workers'], regions):
        worker['region'] = region
    unchanged = digest(scientific_payload(parent_plan))
    if digest(scientific_payload(new_plan)) != unchanged:
        raise ValueError('Scientific payload changed during capacity preparation')
    proof = {'schema': 1, 'epoch': now, 'capacity_only': True, 'passed': True,
        'attempt_number': attempt, 'maximum_capacity_attempts': 3,
        'first_plan_identity': first_identity, 'parent_plan_identity': digest(parent_plan),
        'new_plan_identity': digest(new_plan), 'scientific_payload_identity': unchanged,
        'failure_receipts_identity': digest(failures), 'failed_dispatch_identity': digest(dispatch),
        'ledger_identity': digest(ledger), 'absence_receipt_identity': digest(absence),
        'prior_proof_identity': digest(prior_proof) if prior_proof else None,
        'no_gpu_ever_allocated_in_attempt': True, 'all_attempt_resources_absent': True,
        'changed_fields': ['run_id', 'workers[].region'],
        'prior_resource_ids': [r['id'] for r in resources]}
    return new_plan, proof
