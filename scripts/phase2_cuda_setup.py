#!/usr/bin/env python3
"""Fetch only pinned public snapshots, verify exact Studio hashes, record CUDA host."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import snapshot_download
from gearshift.phase2_io import read, write, runtime_identity


def main():
    expected = read('evidence/phase2/studio_transfer/model_hashes_studio.json')
    groups = {}
    for key, desc in expected.items():
        owner, model, revision, name = key.split('/')
        groups.setdefault((owner + '/' + model, revision), {})[name] = desc
    checked = {}
    for (model, revision), files in groups.items():
        folder = Path(snapshot_download(model, revision=revision, allow_patterns=list(files), token=False))
        for name, desc in files.items():
            path = folder / name
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    h.update(block)
            actual = dict(bytes=path.stat().st_size, sha256=h.hexdigest())
            assert actual == desc, model + '/' + name
            checked[model + '/' + revision + '/' + name] = actual
    write('evidence/phase2/cuda_execution/model_hashes_verified.json', checked)
    host = runtime_identity()
    host['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
    host['pip_freeze'] = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    write('evidence/phase2/cuda_execution/runtime.json', host)


if __name__ == '__main__':
    main()
