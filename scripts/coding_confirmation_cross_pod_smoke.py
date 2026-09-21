#!/usr/bin/env python3
"""Derive one cross-pod continuation from an immutable completed synthetic smoke.

No baseline, interruption, prompt, seed, sampler, or benchmark is created here.
The original archive remains intact. Only the previously declared65-token states
are restored into a fresh derived folder; a single resume-only invocation is
allowed. Passing demonstrates these four bounded cases on this pair of pods.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import bind, digest, sha, write
from scripts import coding_confirmation_resume_smoke as smoke

SCRIPT = 'scripts/coding_confirmation_cross_pod_smoke.py'
MANIFEST = 'CROSS_POD_DERIVATION.json'
SOURCE_ARCHIVE = 'provenance/completed_origin_smoke.tar.gz'
SCOPE = ('Observed exact continuation of the original four synthetic paths after the predeclared65-token '
         'checkpoint, on the identified origin/target H200 pods and frozen runtime only. No new baseline, '
         'seed, prompt, benchmark output, training-resume, long-context, arbitrary cut, or universal proof.')


def read(path): return json.loads(Path(path).read_text())
def checksum(data): return hashlib.sha256(data).hexdigest()


def relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe archive or packet path')
    return str(path)


def archive_files(path, prefix):
    """Read only compact ordinary files; never let tar choose filesystem paths."""
    prefix = PurePosixPath(relative(prefix)); files = {}; total = 0
    with tarfile.open(path, 'r:*') as archive:
        for member in archive:
            name = PurePosixPath(relative(member.name))
            if member.isdir(): continue
            if not member.isfile(): raise ValueError('Links and special archive members are forbidden')
            total += member.size
            if member.size > 64 * 1024**2 or total > 256 * 1024**2:
                raise ValueError('Smoke transfer must remain compact; heavy artifacts are forbidden')
            if name.suffix not in ('.json', '.log', '.lock'):
                raise ValueError('Unexpected noncompact artifact in smoke archive')
            if not name.is_relative_to(prefix): continue
            key = str(name.relative_to(prefix))
            if key in files: raise ValueError('Duplicate archive member')
            files[key] = archive.extractfile(member).read()
    if not files or 'plan.json' not in files: raise ValueError('Smoke archive prefix does not contain a plan')
    return files


def valid_runtime(runtime):
    expected = {'torch': '2.8.0+cu128', 'transformers': '4.57.6', 'attention': 'sdpa',
                'dtype': 'bfloat16', 'TF32': False, 'deterministic_algorithms': True}
    return all(runtime.get(k) == v for k, v in expected.items()) and 'H200' in runtime.get('gpu', '')


def original_proof(repo, files):
    """Recompute all original comparisons before deriving any resumed checkpoint."""
    with tempfile.TemporaryDirectory(prefix='gearshift-smoke-source-') as temporary:
        folder = Path(temporary)
        for name, data in files.items():
            path = folder / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        plan_sha = checksum(files['plan.json'])
        plan = smoke.validate_plan(repo, folder, plan_sha)
        if plan['cap'] != 128 or plan['interrupt_after'] != 65:
            raise ValueError('Only the original128-token smoke with its declared65-token cut can be derived')
        if 'SMOKE_PROOF.json' not in files: raise ValueError('Original completed smoke proof is required')
        proof = smoke.verify(repo, folder, plan_sha)
        if not proof['all_cases_passed'] or proof['cross_pod_resume_for_all_cases']:
            raise ValueError('Expected a completed passing same-pod original smoke')
        pods = set()
        for case in smoke.CASES:
            for phase in ['baseline_result', 'interruption', 'resume_result']:
                row = read(folder / 'proof' / case / (phase + '.json'))
                pods.add(row['attempt']['pod_id'])
                if not valid_runtime(row['attempt']['runtime']):
                    raise ValueError('Original smoke is not the pinned H200 numerical runtime')
            cut = read(folder / 'proof' / case / 'interrupted_state.json')
            if (cut['state'] != 'running' or len(cut['tokens']) != 65 or cut['timing']['resume_count'] != 0 or
                    cut['cache_checkpointed'] is not False or cut['resume_sha256'] != digest({k:v for k,v in cut.items() if k != 'resume_sha256'})):
                raise ValueError('Original interruption is not an untouched65-token running checkpoint')
        if len(pods) != 1: raise ValueError('Original proof does not have one unambiguous origin pod')
        return plan, proof, pods.pop()


def derive(repo, source_archive, source_sha, prefix, destination, target_pod):
    repo, source_archive, destination = Path(repo).resolve(), Path(source_archive), Path(destination)
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', target_pod): raise ValueError('Invalid target pod')
    if sha(source_archive) != source_sha: raise ValueError('Original archive hash differs')
    if destination.exists(): raise ValueError('Derived smoke destination must be new; originals are never overwritten')
    files = archive_files(source_archive, prefix)
    plan, proof, origin_pod = original_proof(repo, files)
    if target_pod == origin_pod: raise ValueError('Cross-pod continuation requires a different target pod')
    selected = {'plan.json': ('plan.json', files['plan.json'], True)}
    for name, data in files.items():
        if name.startswith('baseline/') and not name.endswith('.lock'):
            selected[name] = (name, data, True)
    for case in smoke.CASES:
        root = 'proof/' + case + '/'; dest = 'restarted/' + case + '/'
        for name in ['baseline_started.json', 'baseline_result.json', 'interrupt_started.json',
                     'interruption.json', 'interrupted_state.json']:
            selected[root + name] = (root + name, files[root + name], True)
        for name, data in files.items():
            if name.startswith(root + 'interrupt/'):
                selected[name] = (name, data, True)
        checkpoint = json.loads(files[root + 'interrupted_state.json'])
        identity = json.loads(files[dest + 'identity.json'])
        if digest(identity) != checkpoint['identity_sha256']:
            raise ValueError('Archived sampler identity differs from interrupted checkpoint')
        selected[dest + 'identity.json'] = (dest + 'identity.json', files[dest + 'identity.json'], True)
        selected[dest + 'resume.json'] = (root + 'interrupted_state.json', files[root + 'interrupted_state.json'], False)
        attempt_id = checkpoint['timing']['overhead']['attempt_id']
        if Path(attempt_id).name != attempt_id or not attempt_id.endswith('.json'):
            raise ValueError('Interrupted attempt identity is unsafe')
        attempt_path = dest + 'attempts/' + attempt_id
        attempt = json.loads(files[attempt_path])
        if (attempt.get('state') != 'started' or attempt.get('identity_sha256') != checkpoint['identity_sha256'] or
                attempt.get('initial_committed_tokens') != 0):
            raise ValueError('Only the original abruptly interrupted attempt may be retained')
        selected[attempt_path] = (attempt_path, files[attempt_path], True)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        archive_target = destination / SOURCE_ARCHIVE; archive_target.parent.mkdir()
        with source_archive.open('rb') as source, archive_target.open('xb') as target:
            shutil.copyfileobj(source, target); target.flush(); os.fsync(target.fileno())
        if sha(archive_target) != source_sha: raise ValueError('Original archive changed while preserving provenance')
        rows = []
        for name, (origin, data, immutable) in sorted(selected.items()):
            path = destination / name; path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as out: out.write(data); out.flush(); os.fsync(out.fileno())
            rows.append({'path': name, 'source_path': origin, 'bytes': len(data), 'sha256': checksum(data), 'immutable_after_resume': immutable})
        value = {'schema':1, 'purpose':'cross_pod_continuation_of_completed_smoke',
            'origin_pod_id':origin_pod, 'target_pod_id':target_pod,
            'source_archive_path':SOURCE_ARCHIVE, 'source_archive_sha256':source_sha,
            'source_archive_prefix':relative(prefix), 'source_plan_sha256':checksum(files['plan.json']),
            'source_completed_proof_sha256':checksum(files['SMOKE_PROOF.json']),
            'source_file_inventory': [{'path':name,'bytes':len(data),'sha256':checksum(data)} for name,data in sorted(files.items())],
            'helper_path':SCRIPT, 'helper_sha256':sha(repo / SCRIPT), 'harness_sha256':plan['harness_sha256'],
            'files':rows, 'allowed_phase':'resume', 'allowed_cases':smoke.CASES, 'allowed_invocations':1,
            'interrupt_after':65, 'cap':128, 'baseline_recomputed':False, 'new_seed_or_prompt':False,
            'previous_resume_artifacts_active':False, 'original_completed_archive_preserved':True,
            'timing_scope':'Original interrupted timing bytes retained; crash tails stay incomplete. New reconstruction/timing belongs to this derived continuation.',
            'scope':SCOPE}
        bind(destination / MANIFEST, value)
        return {'folder':str(destination), 'derivation_sha256':sha(destination / MANIFEST),
                'plan_sha256':value['source_plan_sha256'], 'target_pod_id':target_pod}
    except BaseException as exc:
        write(destination / 'DERIVATION_FAILED.json', {'error':str(exc),'type':type(exc).__name__})
        raise


def validate_derivation(repo, folder, expected_sha, *, before_resume):
    folder = Path(folder); path = folder / MANIFEST
    if sha(path) != expected_sha: raise ValueError('Cross-pod derivation hash differs')
    d = read(path)
    required = {'schema':1,'purpose':'cross_pod_continuation_of_completed_smoke','allowed_phase':'resume',
        'allowed_cases':smoke.CASES,'allowed_invocations':1,'interrupt_after':65,'cap':128,
        'baseline_recomputed':False,'new_seed_or_prompt':False,'previous_resume_artifacts_active':False,
        'original_completed_archive_preserved':True,'helper_path':SCRIPT,'source_archive_path':SOURCE_ARCHIVE,'scope':SCOPE}
    if any(d.get(k) != v for k,v in required.items()) or sha(Path(repo)/SCRIPT) != d['helper_sha256']:
        raise ValueError('Derived continuation protocol or helper differs')
    if sha(folder / SOURCE_ARCHIVE) != d['source_archive_sha256']: raise ValueError('Preserved original archive changed')
    smoke.validate_plan(repo, folder, d['source_plan_sha256'])
    for row in d['files']:
        if before_resume or row['immutable_after_resume']:
            path = smoke.scoped(folder, relative(row['path']))
            if path.is_symlink() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('Derived original input changed: ' + row['path'])
    if before_resume:
        allowed = {row['path'] for row in d['files']} | {MANIFEST,SOURCE_ARCHIVE}
        actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
        if actual != allowed: raise ValueError('Derived folder contains unapproved prior attempts or outputs')
    return d


def resume_once(repo, folder, derivation_sha, lease_path, lease_sha, timeout=3600):
    repo, folder = Path(repo).resolve(), Path(folder).resolve()
    if repo != ROOT.resolve(): raise ValueError('Run helper inside the staged source it verifies')
    if not 0 < timeout <= 7200: raise ValueError('Cross-pod smoke wall time must be at most two hours')
    d = validate_derivation(repo, folder, derivation_sha, before_resume=True)
    lease, guard = smoke.allocation_guard(folder, lease_path, lease_sha)
    if lease['pod_id'] != d['target_pod_id'] or lease['pod_id'] == d['origin_pod_id']:
        raise ValueError('Reserved target pod differs from immutable derivation')
    attempt = folder / 'cross_pod_execution'; attempt.mkdir(exist_ok=False)
    write(attempt / 'started.json', {'derivation_sha256':derivation_sha,'target_pod_id':lease['pod_id'],
                                   'epoch':time.time(),'retry_allowed':False})
    from scripts.coding_confirmation_replication_supervisor import child_environment
    command = [sys.executable,str(repo / smoke.SCRIPT),'phase','--repo',str(repo),'--folder',str(folder),
        '--plan-sha256',d['source_plan_sha256'],'--phase','resume','--case','all',
        '--lease',str(lease_path),'--lease-sha256',lease_sha]
    child = None
    try:
        guard()
        with (attempt / 'resume.log').open('xb') as log:
            child = subprocess.Popen(command,cwd=repo,env=child_environment(lease,0),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
            write(attempt / 'process.json', {'pid':child.pid,'phase':'resume','case':'all','single_invocation':True})
            code = child.wait(timeout=timeout)
        write(attempt / 'exit.json', {'returncode':code,'epoch':time.time()})
        if code != 0: raise RuntimeError('Single cross-pod continuation failed; no retry or reroll permitted')
        validate_derivation(repo, folder, derivation_sha, before_resume=False)
        proof = smoke.verify(repo,folder,d['source_plan_sha256'])
        if (not proof['all_cases_passed'] or not proof['cross_pod_resume_for_all_cases'] or
                any(read(folder/'proof'/case/'resume_result.json')['attempt']['pod_id'] != d['target_pod_id'] for case in smoke.CASES)):
            raise ValueError('Cross-pod continuation comparisons did not pass on declared target')
        result = {'schema':1,'derivation_sha256':derivation_sha,'original_completed_proof_sha256':d['source_completed_proof_sha256'],
            'derived_smoke_proof_sha256':sha(folder/'SMOKE_PROOF.json'),'origin_pod_id':d['origin_pod_id'],
            'target_pod_id':d['target_pod_id'],'all_four_cases_passed':True,'cross_pod_resume_for_all_cases':True,
            'scope':SCOPE,'baseline_recomputed':False,'new_seed_or_prompt':False}
        bind(folder/'CROSS_POD_PROOF.json',result)
        return result
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
        write(attempt/'failure.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
                                    'retry_allowed':False,'scope':SCOPE})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('derive'); p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--source-archive',type=Path,required=True); p.add_argument('--source-archive-sha256',required=True)
    p.add_argument('--archive-prefix',required=True); p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--target-pod-id',required=True)
    p=sub.add_parser('resume'); p.add_argument('--repo',type=Path,default=ROOT); p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--derivation-sha256',required=True); p.add_argument('--lease',type=Path,required=True)
    p.add_argument('--lease-sha256',required=True); p.add_argument('--timeout-seconds',type=int,default=3600)
    a=parser.parse_args()
    if a.command=='derive': result=derive(a.repo,a.source_archive,a.source_archive_sha256,a.archive_prefix,a.destination,a.target_pod_id)
    else: result=resume_once(a.repo,a.folder,a.derivation_sha256,a.lease,a.lease_sha256,a.timeout_seconds)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
