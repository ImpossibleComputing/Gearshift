"""Complete, immutable local-volume checkpoints for the clean coverage comparison.

A directory becomes visible only after both tensor files were fsynced, safely
reloaded and compared to the state being committed. Network mirrors are readers;
they are not part of this transaction or of the training continuation contract.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch
from .coding_control import digest, sha, write

SCHEMA = 1
IDENTITY_FIELDS = ('experiment_id', 'arm', 'code_commit', 'config_sha256',
                   'schedules_sha256', 'corpus_sha256', 'selected_checkpoint_sha256', 'models')
COMPONENTS = ['mapper', 'optimizer', 'scheduler', 'update', 'schedule_position',
              'python_rng', 'numpy_rng', 'torch_cpu_rng', 'torch_cuda_rng',
              'scaler_explicitly_none_for_bf16', 'checkpoint_identity', 'training_log']


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {'python': random.getstate(), 'numpy': {'algorithm': algorithm,
            'keys': keys.tolist(), 'position': position, 'has_gauss': has_gauss,
            'cached_gaussian': cached_gaussian}, 'torch_cpu': torch.get_rng_state().cpu(),
            'torch_cuda': [s.cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(value):
    cuda = value['torch_cuda']
    current_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda) != current_devices:
        raise ValueError('Checkpoint CUDA RNG device count differs from this process')
    random.setstate(value['python'])
    npstate = value['numpy']
    np.random.set_state((npstate['algorithm'], np.asarray(npstate['keys'], dtype=np.uint32),
                         npstate['position'], npstate['has_gauss'], npstate['cached_gaussian']))
    torch.set_rng_state(value['torch_cpu'])
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def validate_identity(identity):
    if not isinstance(identity, dict) or any(k not in identity for k in IDENTITY_FIELDS):
        raise ValueError('Incomplete checkpoint scientific identity')
    if identity['arm'] not in ('FIXED', 'ROTATING') or not identity['models']:
        raise ValueError('Invalid arm or missing pinned model identities')
    if any(not identity[k] for k in IDENTITY_FIELDS):
        raise ValueError('Empty checkpoint scientific identity')
    digest(identity)


def capture_state(mapper, optimizer, scheduler, step, identity, training_log=(), scaler=None):
    validate_identity(identity)
    if type(step) is not int or step < 0:
        raise ValueError('Invalid completed update count')
    if scaler is not None:
        raise ValueError('This frozen BF16 recipe does not use a gradient scaler')
    if [r['step'] for r in training_log] != list(range(1, step + 1)):
        raise ValueError('Checkpoint training log must cover each completed update exactly once')
    return {'schema_version': SCHEMA, 'state_dict': cpu_copy(mapper.state_dict()),
            'optimizer': cpu_copy(optimizer.state_dict()), 'scheduler': cpu_copy(scheduler.state_dict()),
            'step': step, 'schedule_position': step, 'rng': cpu_copy(rng_state()), 'scaler': None,
            'mixed_precision': 'bfloat16_no_scaler', 'checkpoint_identity': copy.deepcopy(identity),
            'checkpoint_identity_sha256': digest(identity), 'training_log': copy.deepcopy(list(training_log)),
            'optimizer_resume_supported': True}


def state_difference(left, right):
    """An exact nested comparison; float errors are measured only on mismatches."""
    differences = []
    def visit(a, b, path):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                differences.append({'path': path, 'reason': 'tensor shape or dtype differs'}); return
            if not torch.equal(a, b):
                error = float((a.to(torch.float64) - b.to(torch.float64)).abs().max()) if a.numel() else 0.
                differences.append({'path': path, 'reason': 'tensor values differ', 'max_abs': error})
        elif isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                differences.append({'path': path, 'reason': 'dictionary keys differ'}); return
            for k in a: visit(a[k], b[k], f'{path}.{k}')
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                differences.append({'path': path, 'reason': 'sequence type or length differs'}); return
            for i, (x, y) in enumerate(zip(a, b)): visit(x, y, f'{path}[{i}]')
        elif type(a) is not type(b) or a != b:
            differences.append({'path': path, 'reason': 'scalar differs'})
    visit(left, right, 'state')
    return {'exact': not differences, 'differences': differences,
            'maximum_tensor_absolute_difference': max((x.get('max_abs', 0.) for x in differences), default=0.)}


def state_tensor_sha256(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        cpu = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(cpu.dtype).encode()); h.update(str(tuple(cpu.shape)).encode())
        h.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _tensor_file(path, value):
    with path.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    check = state_difference(value, loaded)
    if not check['exact']:
        raise ValueError('Checkpoint roundtrip verification failed: ' + str(check))
    return {'sha256': sha(path), 'bytes': path.stat().st_size}


def save_checkpoint(path, mapper, optimizer, scheduler, step, identity, training_log=(), scaler=None):
    path = Path(path)
    if path.exists():
        raise ValueError('Immutable checkpoint destination already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ('.incomplete-' + path.name + '-' + uuid.uuid4().hex)
    temporary.mkdir()
    payload = capture_state(mapper, optimizer, scheduler, step, identity, training_log, scaler)
    files = {'full.pt': _tensor_file(temporary / 'full.pt', payload)}
    mapper_payload = {'state_dict': payload['state_dict'], 'step': step, 'arm': identity['arm'],
        'checkpoint_identity': copy.deepcopy(identity), 'checkpoint_identity_sha256': digest(identity),
        'optimizer_resume_supported': True, 'full_checkpoint_file': 'full.pt'}
    files['mapper.pt'] = _tensor_file(temporary / 'mapper.pt', mapper_payload)
    manifest = {'schema_version': SCHEMA, 'step': step, 'arm': identity['arm'],
        'checkpoint_identity': copy.deepcopy(identity), 'checkpoint_identity_sha256': digest(identity),
        'schedule_position': step, 'complete_resumable': True, 'verified_roundtrip': True,
        'files': files, 'state_components': COMPONENTS, 'mapper_tensor_sha256': state_tensor_sha256(payload['state_dict']),
        'created_epoch': time.time(), 'training_rows': len(training_log)}
    write(temporary / 'manifest.json', manifest)
    _fsync_directory(temporary)
    # This directory rename is the only publication event. Incomplete attempts
    # remain separately named for failure audit and are never consumed by readers.
    os.rename(temporary, path); _fsync_directory(path.parent)
    return checkpoint_receipt(path, manifest)


def checkpoint_receipt(path, manifest=None):
    path = Path(path).resolve()
    if manifest is None:
        manifest = json.loads((path / 'manifest.json').read_text())
    return {'step': manifest['step'], 'arm': manifest['arm'], 'directory': str(path),
            'manifest_sha256': sha(path / 'manifest.json'),
            'checkpoint_identity_sha256': manifest['checkpoint_identity_sha256'],
            'mapper': {'path': str(path / 'mapper.pt'), **manifest['files']['mapper.pt']},
            'full': {'path': str(path / 'full.pt'), **manifest['files']['full.pt']}}


def verify_checkpoint(path, expected_identity=None):
    path = Path(path)
    if path.name.startswith('.') or path.is_symlink():
        raise ValueError('Uncommitted or linked checkpoint directory')
    manifest = json.loads((path / 'manifest.json').read_text())
    identity = manifest['checkpoint_identity']; validate_identity(identity)
    if manifest.get('schema_version') != SCHEMA or manifest.get('complete_resumable') is not True or manifest.get('verified_roundtrip') is not True:
        raise ValueError('Incomplete checkpoint manifest')
    if manifest.get('checkpoint_identity_sha256') != digest(identity) or (expected_identity is not None and identity != expected_identity):
        raise ValueError('Checkpoint scientific identity mismatch')
    if manifest['arm'] != identity['arm'] or manifest['schedule_position'] != manifest['step']:
        raise ValueError('Checkpoint arm or schedule position differs')
    if set(manifest['files']) != {'full.pt', 'mapper.pt'} or manifest['state_components'] != COMPONENTS:
        raise ValueError('Missing complete checkpoint components')
    for name, receipt in manifest['files'].items():
        file = path / name
        if file.is_symlink() or not file.is_file() or file.stat().st_size != receipt['bytes'] or sha(file) != receipt['sha256']:
            raise ValueError('Checkpoint file hash or size verification failed: ' + name)
    payload = torch.load(path / 'full.pt', map_location='cpu', weights_only=True)
    required = {'state_dict', 'optimizer', 'scheduler', 'step', 'schedule_position', 'rng', 'scaler',
                'mixed_precision', 'checkpoint_identity', 'checkpoint_identity_sha256', 'training_log', 'optimizer_resume_supported', 'schema_version'}
    if set(payload) != required or payload['optimizer_resume_supported'] is not True or payload['scaler'] is not None or payload['mixed_precision'] != 'bfloat16_no_scaler':
        raise ValueError('Full checkpoint is missing resumable state')
    if payload['checkpoint_identity'] != identity or payload['checkpoint_identity_sha256'] != digest(identity):
        raise ValueError('Full state identity mismatch')
    if payload['schema_version'] != SCHEMA or payload['step'] != manifest['step'] or payload['schedule_position'] != payload['step']:
        raise ValueError('Full state schedule position mismatch')
    if set(payload['rng']) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}:
        raise ValueError('Full checkpoint is missing an RNG stream')
    if set(payload['optimizer']) != {'state', 'param_groups'} or not payload['optimizer']['param_groups']:
        raise ValueError('Incomplete optimizer state')
    if payload['step'] > 0 and not payload['optimizer']['state']:
        raise ValueError('Trained checkpoint cannot have fresh optimizer state')
    if payload['scheduler'].get('last_epoch') != payload['step']:
        raise ValueError('Scheduler and schedule position differ')
    if [r['step'] for r in payload['training_log']] != list(range(1, payload['step'] + 1)):
        raise ValueError('Full checkpoint training log is not contiguous')
    if state_tensor_sha256(payload['state_dict']) != manifest['mapper_tensor_sha256']:
        raise ValueError('Mapper tensor identity mismatch')
    mapper = torch.load(path / 'mapper.pt', map_location='cpu', weights_only=True)
    if mapper['checkpoint_identity'] != identity or mapper['step'] != payload['step'] or not state_difference(payload['state_dict'], mapper['state_dict'])['exact']:
        raise ValueError('Evaluation mapper and full checkpoint differ')
    return manifest, payload


def restore_state(payload, mapper, optimizer, scheduler, expected_identity=None, scaler=None):
    if expected_identity is not None and payload['checkpoint_identity'] != expected_identity:
        raise ValueError('Checkpoint scientific identity mismatch')
    if payload.get('optimizer_resume_supported') is not True or payload.get('schedule_position') != payload.get('step'):
        raise ValueError('Checkpoint cannot resume exactly')
    if scaler is not None or payload['scaler'] is not None:
        raise ValueError('Unexpected gradient scaler for the frozen BF16 recipe')
    mapper.load_state_dict(payload['state_dict'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    optimizer.zero_grad(set_to_none=True)
    restore_rng(payload['rng'])
    return payload['step']


def load_checkpoint(path, mapper, optimizer, scheduler, expected_identity):
    manifest, payload = verify_checkpoint(path, expected_identity)
    restore_state(payload, mapper, optimizer, scheduler, expected_identity)
    return manifest, payload


def refresh_index(checkpoint_root, expected_identity):
    checkpoint_root = Path(checkpoint_root)
    rows = []
    for path in sorted(checkpoint_root.glob('step_*')):
        manifest, _ = verify_checkpoint(path, expected_identity)
        if path.name != f"step_{manifest['step']:04d}":
            raise ValueError('Checkpoint directory name and committed update differ')
        rows.append(checkpoint_receipt(path, manifest))
    steps = [r['step'] for r in rows]
    if len(steps) != len(set(steps)) or steps != sorted(steps):
        raise ValueError('Duplicate or unordered immutable checkpoints')
    result = {'checkpoints': rows, 'latest_step': max(steps) if steps else None,
              'checkpoint_identity_sha256': digest(expected_identity),
              'retention': 'All immutable common checkpoints retained; minimum latest two.'}
    write(checkpoint_root.parent / 'checkpoint_manifest.json', result)
    return result
