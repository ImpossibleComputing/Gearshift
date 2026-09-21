#!/usr/bin/env python3
"""Thin standalone entrypoint for one secondary training arm on a single H200."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from gearshift.coding_control import write
from scripts.coding_confirmation_replication import run_replication


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--arm', required=True, choices=['FIXED', 'ROTATING'])
    parser.add_argument('--worker-id', required=True)
    parser.add_argument('--resume-checkpoint')
    args = parser.parse_args(argv); c = None
    try:
        # The parent runtime implements the new allocation policy. In particular,
        # this does not import the obsolete v2 fixed-dollar lease/context builder.
        from scripts.coding_confirmation_runtime import build_context
        c = build_context(args.plan, role='replication_train', worker_id=args.worker_id, arm=args.arm)
        if args.resume_checkpoint:
            c['resume_checkpoint'] = args.resume_checkpoint
        result = run_replication(c, args.arm)
        c['publish'](state='complete', stage='replication_training_complete', arm=args.arm,
                     completed_updates=result['completed_updates'])
        return result
    except BaseException as error:
        if c is not None:
            c['telemetry'].failure(error)
            write(Path(c['status_root']) / 'replication_failure.json', {
                'exception': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                'arm': args.arm, 'worker_id': args.worker_id, 'original_v2_artifacts_modified': False})
            c['publish'](state='failed', stage='replication_training_failed', error=str(error), arm=args.arm)
        raise


if __name__ == '__main__': main()
