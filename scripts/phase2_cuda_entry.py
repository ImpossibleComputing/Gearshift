#!/usr/bin/env python3
"""Apply the declared CUDA runtime policy before any scientific module loads."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

POLICY = dict(deterministic_algorithms=True, cublas_workspace_config=':4096:8',
              sdpa_backend='math', cudnn_benchmark=False, allow_tf32=False)


def configure():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = POLICY['cublas_workspace_config']
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # One implementation for live/restored layouts and differentiable training.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def main():
    cfg = json.loads(Path('configs/phase2_cuda.json').read_text())
    expected = cfg['cuda_runtime_policy']
    assert expected['settings'] == POLICY
    assert expected['entry_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    configure()
    args = sys.argv[1:]
    assert args
    if args[0] == '-m':
        # Match `python -m`: imports resolve from the working project directory,
        # not just this wrapper's scripts directory.
        sys.path.insert(0, str(Path.cwd()))
        sys.argv = args[1:]
        runpy.run_module(args[1], run_name='__main__', alter_sys=True)
    else:
        sys.argv = args
        sys.path.insert(0, str(Path(args[0]).resolve().parent))
        runpy.run_path(args[0], run_name='__main__')


if __name__ == '__main__':
    main()
