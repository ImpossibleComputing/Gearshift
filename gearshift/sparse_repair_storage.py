"""One frozen CPU-only storage exception; no private bytes or provider operations.

This is operational scope, not scorer/generation policy. The original destination
remains the default. An arbitrary caller-supplied receipt cannot authorize a mount.
"""
import hashlib
import json
from pathlib import Path

ORIGINAL_VOLUME = 'lgk3howszi'
FALLBACK_VOLUME = 'kzmo0a5erc'
FALLBACK_REGION = 'EUR-IS-1'
FALLBACK_PROFILE = 'cpu5g16'
BINDING_PATH = 'configs/coding_pilot_v1/sparse_repair_01/private_scoring_storage_01.json'
BINDING_SHA256 = '6541e265566b3f907ff0aa9dbb203f623b342519c0dc83edb2ecfd780b0c4845'
DECLARATION_PATH = 'configs/coding_pilot_v1/sparse_repair_01/declaration.json'
CLOSURE_PATH = 'results/sparse_repair_01/screen/generation_closure.json'
RESULT_ROOT = '/workspace/GearshiftSparseRepair/results/sparse_repair_01'


def binding_reference():
    return {'path': BINDING_PATH, 'sha256': BINDING_SHA256}


def checked_bytes(repo, relative, expected):
    root = Path(repo).resolve()
    path = Path(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError('Storage evidence path must remain in the public checkout')
    source = root / path
    for cursor in (source, *source.parents):
        if cursor == root:
            break
        if cursor.is_symlink():
            raise ValueError('Storage evidence cannot traverse a symlink')
    if not source.is_file():
        raise ValueError('Storage evidence regular file missing')
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Frozen storage evidence hash differs: ' + relative)
    return raw


def load_binding(repo):
    """Read only exact public identities; never open private/development.json."""
    value = json.loads(checked_bytes(repo, BINDING_PATH, BINDING_SHA256))
    if (value['volume_id'] != FALLBACK_VOLUME or value['region'] != FALLBACK_REGION
            or value['size_gb'] != 20 or value['cpu_profile'] != FALLBACK_PROFILE
            or value['mount_path'] != '/workspace' or value['allowed_result_root'] != RESULT_ROOT
            or value['generation_mount_allowed'] is not False):
        raise ValueError('Wrong narrowly approved private storage identity')
    reference = value['provider_volume_receipt']
    provider = json.loads(checked_bytes(repo, reference['path'], reference['sha256']))
    if (provider['id'] != FALLBACK_VOLUME or provider['dataCenterId'] != FALLBACK_REGION
            or provider['size'] != 20):
        raise ValueError('Provider storage receipt differs')
    checked_bytes(repo, DECLARATION_PATH, value['declaration_sha256'])
    closure = json.loads(checked_bytes(repo, CLOSURE_PATH, value['generation_closure_file_sha256']))
    if (closure['closure_sha256'] != value['generation_closure_sha256']
            or closure['all_screen_generation_complete'] is not True
            or closure['expected_answers'] != 504 or len(closure['answers']) != 504):
        raise ValueError('Storage exception requires the exact sealed 504-answer generation')
    return value


def validate_cpu_lease(repo, lease, plan=None):
    """Default original mount; fallback needs explicit immutable lease + plan binding."""
    if lease.get('experiment_id') != 'sparse_repair_01' or lease.get('gpu_count') != 0:
        raise ValueError('Dedicated matching CPU lease required')
    volume = lease.get('network_volume_id')
    if volume == ORIGINAL_VOLUME:
        if lease.get('private_storage_binding') is not None or (plan is not None and (
                plan.get('private_volume_id', ORIGINAL_VOLUME) != ORIGINAL_VOLUME
                or plan.get('private_storage_binding') is not None)):
            raise ValueError('Original private CPU destination cannot use a fallback binding')
        return None
    if volume != FALLBACK_VOLUME:
        raise ValueError('Unapproved private CPU volume')
    if (lease.get('private_storage_binding') != binding_reference()
            or lease.get('storage_region') != FALLBACK_REGION
            or lease.get('cpu_profile') != FALLBACK_PROFILE
            or lease.get('allowed_result_root') != RESULT_ROOT):
        raise ValueError('Fallback CPU lease requires the exact frozen storage binding')
    if plan is not None and (plan.get('private_volume_id') != FALLBACK_VOLUME
            or plan.get('private_storage_binding') != binding_reference()):
        raise ValueError('Fallback CPU plan requires explicit private volume and frozen binding')
    value = load_binding(repo)
    if plan is not None and (plan.get('declaration_path') != DECLARATION_PATH
            or plan.get('declaration_sha256') != value['declaration_sha256']):
        raise ValueError('Fallback CPU plan declaration differs from frozen storage binding')
    return value
