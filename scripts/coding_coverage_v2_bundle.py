#!/usr/bin/env python3
"""Verified compact review delivery, independent of the running scientific jobs.

The builder never changes historical archives or frozen experiment inputs. It
requires completed execution (or an explicit engineering-failure receipt), checks
the public input identities, verifies archive membership, and regenerates the
report from an extracted archive with no model/checkpoint/private-test files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time
import zipfile

PUBLICATION = '64725974fa55459350d1c9d09037bab64d0c5ec6'
PREFIX = 'gearshift/'
TEXT = {'.py','.json','.jsonl','.csv','.tsv','.txt','.log','.md','.yaml','.yml','.toml','.ini'}
ALLOWED_SUFFIXES = TEXT | {'.png','.svg','.pdf'}
HEAVY_SUFFIXES = {'.pt','.pth','.bin','.safetensors','.npy','.npz','.pkl','.pickle','.ckpt','.onnx','.gguf'}
DENIED_PARTS = {'.git','.venv','.pilot-venv','venv','virtualenv','__pycache__','.pytest_cache',
                'private','credentials','secrets','.ssh','.aws','.cache','cache','caches','model_cache',
                'paired_cache','paired_caches','tensor_cache','node_modules'}
SECRET_PATTERNS = [rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA|ENCRYPTED) PRIVATE KEY-----',
                   rb'rpa_[A-Za-z0-9]{25,}', rb'hf_[A-Za-z0-9]{30,}', rb'sk-[A-Za-z0-9_-]{35,}',
                   rb'(?i)authorization\s*[:=]\s*["\x27]?Bearer\s+[A-Za-z0-9_.-]{20,}']
SECRET_KEYS = {'api_key','apikey','runpod_api_key','hf_token','openai_api_key','authorization',
               'access_token','refresh_token','auth_token','password','client_secret'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def encoded(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe archive/input path: '+str(name))
    return p


def permitted(name):
    p = safe_relative(name); lower = [part.lower() for part in p.parts]
    if DENIED_PARTS.intersection(lower) or p.suffix.lower() not in ALLOWED_SUFFIXES:
        return False
    if p.name.startswith('.') and p.name not in {'.gitignore'}:
        return False
    if p.name.lower().endswith(('.tmp','.lock','.env','.pem','.key')):
        return False
    if p.name.lower() in {'provider_auth.json','provider_auth.txt','runpod_credentials.json','credentials.json','secrets.json'}:
        return False
    return True


def scoped(repo, name, *, require_file=True):
    relative = safe_relative(str(name)); path = repo/Path(*relative.parts)
    path.resolve().relative_to(repo)
    for part in [path, *path.parents]:
        if part == repo:
            break
        if part.is_symlink():
            raise ValueError('Linked input/evidence is not permitted: '+str(name))
    if require_file and not path.is_file():
        raise ValueError('Missing required compact evidence: '+str(name))
    return path


def credential_scan(name, content):
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, content):
            raise ValueError('Credential signature in review member: '+name)
    if Path(name).suffix.lower() not in {'.json','.jsonl'}:
        return
    try:
        values = [json.loads(line) for line in content.decode().splitlines() if line.strip()] if name.endswith('.jsonl') else [json.loads(content)]
    except (ValueError, UnicodeError):
        raise ValueError('Malformed JSON evidence: '+name)
    def check(value, trail=()):
        if isinstance(value, dict):
            # Token text is a vocabulary key, not an authentication field.
            vocabulary = (Path(name).name == 'tokenizer.json' and trail == ('model','vocab')
                          and all(isinstance(k,str) and type(v) is int for k,v in value.items()))
            if vocabulary:
                return
            for key, part in value.items():
                approval_flags = {'execution_allowed','previous_phase_authorization_applies','owner_approval_required'}
                spending_authorization = (
                    key == 'authorization' and not trail and isinstance(part,dict)
                    and set(part) == approval_flags | {'approved_spending_cap_usd'}
                    and all(type(part[k]) is bool for k in approval_flags)
                    and (part['approved_spending_cap_usd'] is None
                         or type(part['approved_spending_cap_usd']) in (int,float)))
                # Monitoring receipts link to a public, timestamped approval
                # record. Only this exact path shape/location is metadata;
                # arbitrary text, bearer values and nested credentials fail.
                budget_approval_reference = (
                    key == 'authorization' and trail == ('budget',)
                    and Path(name).parent.name == 'monitoring'
                    and re.fullmatch(r'status_[0-9]{8}T[0-9]{6}Z\.json', Path(name).name)
                    and isinstance(part,str)
                    and re.fullmatch(r'budget/owner_amendment_[0-9]{8}T[0-9]{6}Z\.json',part))
                if key.lower() in SECRET_KEYS and not spending_authorization and not budget_approval_reference and part not in (None, '', False, '[REDACTED]', '<redacted>', 'REDACTED'):
                    raise ValueError('Unredacted credential field in review member: '+name)
                check(part,trail+(key,))
        elif isinstance(value, list):
            for index,part in enumerate(value):
                check(part,trail+(index,))
    for value in values:
        check(value)


def git_proof(repo):
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args], text=True).strip()
    tag = git('rev-parse','gearshift-progress-01^{commit}')
    if tag != PUBLICATION:
        raise ValueError('Frozen publication tag changed')
    return {'publication_tag': 'gearshift-progress-01', 'publication_commit': tag,
            'delivery_commit': git('rev-parse','HEAD'), 'branch': git('branch','--show-current'),
            'worktree_status': git('status','--porcelain'), 'repository_modified_by_builder': False,
            'publication_or_push_performed': False}


def find_execution_plan(repo, result_relative, evidence_root, explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = repo/path
        path.resolve().relative_to(repo)
        candidates = [path]
    else:
        candidates = sorted(set((repo/'configs/coding_pilot_v1/coverage_generalization_v2').rglob('*.json')) |
                            set(evidence_root.rglob('*.json')))
    found = []
    for path in candidates:
        if path.is_symlink() or path.stat().st_size > 16*1024**2:
            continue
        obj = read(path)
        if isinstance(obj, dict) and obj.get('result_root') == result_relative and 'input_manifest' in obj and obj.get('execution_role','primary') == 'primary':
            found.append((path,obj))
    if len(found) != 1:
        raise ValueError('Supply --execution-plan; expected exactly one frozen primary input plan')
    return found[0]


def execution_gate(root, engineering_failure=None):
    complete_path = root/'execution_complete.json'
    if engineering_failure is None:
        if not complete_path.is_file():
            raise ValueError('Execution is not complete; no final review bundle')
        complete = read(complete_path)
        if complete.get('full_1024_1024_training_completed') is not True or complete.get('all_jobs_generated_and_scored') is not True:
            raise ValueError('Full clean1024/1024 execution receipt is not passed')
        score = root/'evaluation/scored_answer_manifest.json'
        if complete['scored_answer_manifest_sha256'] != sha(score):
            raise ValueError('Execution score-manifest hash differs')
        for arm in ('FIXED','ROTATING'):
            r = read(root/'arms'/arm/'training_complete.json')
            if r.get('full_target_completed') is not True or r.get('completed_updates') != 1024 or r.get('scored_positions') != 32768:
                raise ValueError('Both complete1024-update training receipts are required')
            if read(root/'arms'/arm/'resume_preflight.json').get('passed') is not True:
                raise ValueError('Missing passed actual-model checkpoint resume test')
            expected = {f'step_{step:04d}' for step in (0,128,256,512,768,1024)}
            actual = {p.parent.name for p in (root/'arms'/arm/'checkpoints').glob('step_*/manifest.json')}
            if actual != expected:
                raise ValueError('All six durable checkpoint manifests are required per arm')
        return complete
    failure = read(engineering_failure)
    if not isinstance(failure, dict) or not failure.get('reason') or failure.get('checkpoint_selection_by_quality', False):
        raise ValueError('Explicit engineering failure needs a reason and no quality-based selection')
    return {'full_1024_1024_training_completed': False, 'engineering_failure': failure,
            'complete_receipt_present': complete_path.exists()}


def heavy_inventory(repo, root, plan, declaration, backup_manifest=None):
    entries = []
    starting = declaration['selected_checkpoint']; start_record = plan['input_manifest'].get(starting)
    if not start_record or start_record['sha256'] != declaration['selected_checkpoint_sha256']:
        raise ValueError('Starting mapper is absent from the frozen input inventory')
    index_paths = {}
    for arm in ('FIXED','ROTATING'):
        index = read(root/'arms'/arm/'checkpoint_manifest.json')
        for r in index['checkpoints']:
            for key in ('full','mapper'):
                item = r[key]; index_paths[arm,r['step'],key+'.pt'] = item
    entries.append({'path': starting, **start_record, 'kind': 'original_selected96_mapper',
                    'retrieval_path': starting, 'retrieval_path_kind': 'relative_to_staged_runtime_checkout'})
    for arm in ('FIXED','ROTATING'):
        for folder in sorted((root/'arms'/arm/'checkpoints').glob('step_*')):
            manifest = read(folder/'manifest.json'); step = manifest['step']
            if manifest.get('arm') != arm or manifest.get('complete_resumable') is not True or manifest.get('verified_roundtrip') is not True:
                raise ValueError('Checkpoint is not a verified complete resumable state')
            if set(manifest['files']) != {'full.pt','mapper.pt'}:
                raise ValueError('Unexpected checkpoint file inventory')
            for name, record in manifest['files'].items():
                index = index_paths[arm,step,name]
                if index['sha256'] != record['sha256'] or index['bytes'] != record['bytes']:
                    raise ValueError('Checkpoint index and atomic manifest disagree')
                path = folder/name; rel = str(path.relative_to(repo))
                entries.append({'path': rel, **record, 'kind': 'full_resumable_checkpoint' if name == 'full.pt' else 'mapper_checkpoint',
                    'arm': arm, 'step': step, 'retrieval_path': index['path'], 'retrieval_path_kind': 'recorded_runtime_path',
                    'manifest_path': str((folder/'manifest.json').relative_to(repo)), 'manifest_sha256': sha(folder/'manifest.json')})
    for entry in entries:
        local = scoped(repo, entry['path'], require_file=False)
        entry['present_in_delivery_checkout'] = local.is_file()
        entry['locally_hash_verified'] = False
        if local.is_file():
            if local.stat().st_size != entry['bytes'] or sha(local) != entry['sha256']:
                raise ValueError('Locally present heavy artifact changed: '+entry['path'])
            entry['locally_hash_verified'] = True
            entry['local_retrieval_path'] = str(local)
    return {'files': entries, 'backup_manifest': read(backup_manifest) if backup_manifest else None,
            'checkpoint_inventory_source': 'Atomic verified checkpoint manifests and matching checkpoint index; every locally present file is rehashed.',
            'off_pod_backup_claim': 'Only the supplied backup manifest can establish external copies; absence from this checkout is not a verified backup.',
            'model_weights': 'Excluded. Recreate exact frozen revisions and verify saved source/receiver shard hash receipts.',
            'private_tests': 'Excluded. Authorized local scoring inputs contain only the21 declared validation tasks; their hash remains in the frozen plan.',
            'environments_and_caches': 'Excluded. Rebuild using pinned bootstrap source, container digest and package receipts.',
            'network_volume_deletion_authorized': False}


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(not n.startswith(PREFIX) for n in names):
            raise ValueError('Duplicate or unexpected archive root')
        manifest_name = PREFIX+'REVIEW_BUNDLE_MANIFEST.json'
        manifest = json.loads(archive.read(manifest_name))
        expected = {PREFIX+n for n in manifest['files']} | {manifest_name}
        if set(names) != expected:
            raise ValueError('Archive membership differs from size/hash manifest')
        for name in names:
            rel = name.removeprefix(PREFIX)
            if not permitted(rel):
                raise ValueError('Excluded artifact in archive: '+rel)
            content = archive.read(name); credential_scan(rel,content)
            if name != manifest_name:
                info = manifest['files'][rel]
                if len(content) != info['bytes'] or hashlib.sha256(content).hexdigest() != info['sha256']:
                    raise ValueError('Archive member size/hash differs: '+rel)
        return manifest


def compare_regenerated_report(original, regenerated, expected):
    actual = {p.name:{'sha256':sha(p),'bytes':p.stat().st_size}
              for p in regenerated.iterdir() if p.is_file()}
    if set(actual) != set(expected):
        raise ValueError('Compact report regeneration membership differs')
    rounding = {}
    for name in sorted(actual):
        if actual[name] == expected[name]:
            continue
        vector = str(Path(name).with_suffix('.svg'))
        if not name.endswith('.png') or vector not in actual or actual[vector] != expected[vector]:
            raise ValueError('Compact report regeneration differs: '+name)
        # The exact vector plot binds the content. ARM/x86 rasterization may
        # round a few antialiased edge pixels by at most two 8-bit levels.
        from PIL import Image
        import numpy as np
        with Image.open(original/name) as source, Image.open(regenerated/name) as target:
            a = np.array(source.convert('RGBA'),dtype=np.int16)
            b = np.array(target.convert('RGBA'),dtype=np.int16)
        if a.shape != b.shape:
            raise ValueError('Regenerated PNG dimensions differ: '+name)
        difference = np.abs(a-b)
        changed = int(np.any(difference,axis=2).sum()); pixels = a.shape[0]*a.shape[1]
        maximum = int(difference.max())
        if maximum > 2 or changed > pixels*0.001:
            raise ValueError('Regenerated PNG exceeds platform rounding envelope: '+name)
        rounding[name] = {'paired_svg_exact':True,'maximum_channel_difference':maximum,
                          'changed_pixels':changed,'total_pixels':pixels,
                          'maximum_allowed_channel_difference':2,'maximum_allowed_changed_fraction':0.001,
                          'original':expected[name],'regenerated':actual[name]}
    return actual, rounding


def regenerate_from_archive(path, result_relative, report_relative, expected, engineering_failure_relative=None):
    """Extract only the manifest-verified compact ZIP and compare every report artifact."""
    verify_archive(path)
    with tempfile.TemporaryDirectory(prefix='gearshift-v2-review-regeneration-') as tmp:
        staging = Path(tmp)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(staging)
        repo = staging/'gearshift'; out = staging/'regenerated'
        if any(p.suffix.lower() in HEAVY_SUFFIXES for p in repo.rglob('*') if p.is_file()) or any(p.is_dir() and p.name == 'private' for p in repo.rglob('*')):
            raise ValueError('Compact regeneration tree contains heavyweight/private inputs')
        command = [sys.executable, str(repo/'scripts/coding_coverage_v2_report.py'), '--repo-root',str(repo),
                   '--result-root',result_relative+'/evaluation','--plan',result_relative+'/evaluation/evaluation_plan.json', '--output',str(out)]
        if engineering_failure_relative:
            failure = read(repo/engineering_failure_relative)
            command += ['--primary-step',str(failure['primary_common_checkpoint']), '--engineering-failure',str(repo/engineering_failure_relative)]
        env = {k:v for k,v in os.environ.items() if not any(secret in k.upper() for secret in ('TOKEN','SECRET','PASSWORD','API_KEY','CREDENTIAL'))}
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MPLBACKEND='Agg')
        completed = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=300)
        if completed.returncode:
            raise RuntimeError('Compact report regeneration failed: '+completed.stderr[-6000:])
        actual, rounding = compare_regenerated_report(repo/report_relative,out,expected)
        return {'passed': True, 'files': actual, 'model_or_mapper_weights_present': False, 'private_tests_present': False,
                'generated_code_executed': False, 'all_report_bytes_exact':not rounding,
                'data_markdown_and_svg_bytes_exact':True,'png_platform_rounding':rounding,
                'method': 'Extract verified archive and rerun offline. Require exact JSON/CSV/SVG/Markdown bytes. PNG differences require an exact paired SVG and at most two 8-bit channel levels in at most 0.1% of pixels; record every exception.',
                'original_report_path': report_relative}


def build(repo_root, result_root, evidence_root, output=None, execution_plan=None,
          engineering_failure=None, heavy_backup_manifest=None, verify_regeneration=True,
          input_overlay=None):
    repo = Path(repo_root).resolve()
    def inside(value):
        p = Path(value); p = p if p.is_absolute() else repo/p
        p = p.resolve(); p.relative_to(repo); return p
    root = inside(result_root); evidence = inside(evidence_root); rel = str(root.relative_to(repo))
    out = Path(output).resolve() if output else repo/'gearshift_coverage_generalization_v2_review.zip'
    if out.exists() or out.with_suffix(out.suffix+'.receipt.json').exists():
        raise ValueError('Review output already exists; preserve it and choose a new output path')
    if not root.is_dir() or not evidence.is_dir():
        raise ValueError('Missing experiment result/evidence directory')
    failure_path = inside(engineering_failure) if engineering_failure else None
    backup_path = inside(heavy_backup_manifest) if heavy_backup_manifest else None
    completed = execution_gate(root, failure_path)
    plan_path, plan = find_execution_plan(repo, rel, evidence, execution_plan)
    overlay = inside(input_overlay) if input_overlay else None
    if overlay is not None and not overlay.is_dir():
        raise ValueError('Missing executed-input overlay directory')
    overlay_used = {}
    evaluation_plan = read(root/'evaluation/evaluation_plan.json')
    if evaluation_plan['experiment_id'] != plan['experiment_id']:
        raise ValueError('Execution/evaluation scientific identity differs')
    if completed.get('experiment_id',plan['experiment_id']) != plan['experiment_id']:
        raise ValueError('Completed run identity differs')
    declaration_path = scoped(repo, plan['declaration_path']); declaration = read(declaration_path)
    if sha(declaration_path) != plan['declaration_sha256']:
        raise ValueError('Frozen declaration differs')
    report_root = root/'evaluation/report'; summary = read(report_root/'summary.json')
    if not failure_path and (summary.get('full_1024_per_arm_completed') is not True or summary.get('primary_step') != 1024):
        raise ValueError('Final report does not describe the full clean primary run')
    proof = git_proof(repo); inventory = heavy_inventory(repo,root,plan,declaration,backup_path)
    selected = {}; generated = {}; excluded = []
    def add(path, relative=None, expected=None, required=False):
        path = Path(path); name = relative or str(path.relative_to(repo))
        if not permitted(name):
            if required:
                raise ValueError('Required compact input is excluded: '+name)
            excluded.append(name); return
        frozen = plan['input_manifest'].get(name)
        if frozen is not None:
            expected = frozen
            if overlay is not None:
                candidate = overlay/Path(*safe_relative(name).parts)
                if candidate.exists() or candidate.is_symlink():
                    path = candidate
                    overlay_used[name] = {'source':str(candidate.relative_to(repo)), **frozen}
        checked = scoped(repo,str(path.relative_to(repo)))
        if expected and (checked.stat().st_size != expected['bytes'] or sha(checked) != expected['sha256']):
            raise ValueError('Executed public input changed: '+name)
        if name in selected and selected[name] != checked:
            raise ValueError('Conflicting archive member')
        selected[name] = checked
    # The immutable deployment manifest is authoritative, not a broad repo walk.
    for name, expected in plan['input_manifest'].items():
        if Path(name).suffix.lower() in HEAVY_SUFFIXES:
            excluded.append(name); continue
        add(repo/name, expected=expected, required=True)
    corpus_path = scoped(repo,declaration['corpus_manifest']); corpus = read(corpus_path)
    if sha(corpus_path) != declaration['corpus_manifest_sha256']:
        raise ValueError('Frozen corpus manifest changed')
    add(corpus_path, required=True)
    for name, expected in corpus['files'].items():
        if name.endswith('.pt'):
            continue
        add(repo/name, expected=expected, required=True)
    for base in (root,evidence):
        for p in sorted(base.rglob('*')):
            if overlay is not None and (p == overlay or overlay in p.parents):
                continue
            if p.is_symlink():
                raise ValueError('Linked result/evidence input: '+str(p))
            if p.is_file():
                add(p)
    add(plan_path,required=True)
    for p in sorted(declaration_path.parent.rglob('*')):
        if p.is_file():
            add(p)
    for name in ('README.md','requirements.txt','requirements.lock.txt','pytest.ini','THIRD_PARTY_NOTICES.md',
                 'scripts/coding_coverage_v2_bundle.py','tests/test_coding_coverage_v2_bundle.py',
                 'scripts/coding_coverage_v2_optional_helpers.py','tests/test_coding_coverage_v2_optional_helpers.py',
                 'COVERAGE_V2_HANDOFF.md'):
        if (repo/name).is_file():
            add(repo/name,required=True)
    for p in repo.glob('REPRODUCE_COVERAGE_GENERALIZATION_V2*.md'):
        add(p,required=True)
    add(report_root/'COVERAGE_GENERALIZATION_V2_RESULTS.md','COVERAGE_GENERALIZATION_V2_RESULTS.md',required=True)
    metadata = str(evidence.relative_to(repo))+'/review_bundle'
    generated[metadata+'/git_publication_proof.json'] = encoded(proof)
    if overlay_used:
        generated[metadata+'/executed_input_overlay.json'] = encoded({
            'files':overlay_used, 'all_files_bound_to_frozen_dispatch_manifest':True,
            'historical_checkout_inputs_modified':False})
    generated[metadata+'/excluded_heavy_artifacts.json'] = encoded(inventory)
    if backup_path:
        add(backup_path,required=True)
    expected_report = {p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in report_root.iterdir() if p.is_file()}
    failure_rel = str(failure_path.relative_to(repo)) if failure_path else None
    if failure_path:
        add(failure_path,required=True)
    report_lock = str(evidence.relative_to(repo))+'/packaging/review_requirements.lock.txt'
    report_install = ['python -m pip install -r '+report_lock] if (repo/report_lock).is_file() else []
    repro = ['# Reproduce the compact coverage-v2 report','',
        'Extract this archive and open its `gearshift` directory. The report uses NumPy and Matplotlib; exact runtime package receipts and the lockfile are included. No model download, mapper weights, private tests or generated-code execution are required.', '',
        '```sh', *report_install, 'python scripts/coding_coverage_v2_report.py \\',
        '  --result-root '+rel+'/evaluation \\',
        '  --plan '+rel+'/evaluation/evaluation_plan.json \\',
        '  --output regenerated_report'+(' \\' if failure_path else '')]
    if failure_path:
        repro += ['  --primary-step '+str(read(failure_path)['primary_common_checkpoint'])+' \\', '  --engineering-failure '+failure_rel]
    repro += ['```','', 'Compare the regenerated files with the saved report inventory and regeneration receipt. The task bootstrap resamples and per-draw outcomes are included as JSON/CSV. Tables, statistics, Markdown and vector SVGs must reproduce exactly. The receipt separately records any bounded Mac/Linux PNG edge-rounding differences; original archive bytes retain exact SHA-256 verification.', '',
        'Heavy artifacts are excluded. Their exact hashes, sizes and recorded retrieval paths are in `'+metadata+'/excluded_heavy_artifacts.json`. This inventory does not authorize deleting any durable volume.', '',
        'The first publication tag and earlier archives are unchanged. This is an independent research review bundle.']
    generated['REPRODUCE_COVERAGE_GENERALIZATION_V2_REVIEW.md'] = ('\n'.join(repro)+'\n').encode()
    generated['REVIEW_GUIDE_V2.md'] = ('# Coverage v2 review guide\n\n'
        'Start with [the results](COVERAGE_GENERALIZATION_V2_RESULTS.md).\n\n'
        '- [Reproduce the tables and figures](REPRODUCE_COVERAGE_GENERALIZATION_V2_REVIEW.md).\n'
        '- Execution plan: `'+str(plan_path.relative_to(repo))+'`.\n'
        '- Raw answers, source trajectories, training and scoring receipts: `'+rel+'`.\n'
        '- Resource, cost and failure evidence: `'+str(evidence.relative_to(repo))+'`.\n'
        '- Excluded heavy artifact hashes and retrieval paths: `'+metadata+'/excluded_heavy_artifacts.json`.\n'
        '- Every included file is listed in `REVIEW_BUNDLE_MANIFEST.json` with its size and SHA-256.\n').encode()
    manifest_base = {'schema_version':1, 'experiment_id':plan['experiment_id'], 'delivery_git':proof,
        'execution_code_commit':plan['code_commit'], 'result_root':rel, 'evidence_root':str(evidence.relative_to(repo)),
        'full_1024_1024_training_completed':completed.get('full_1024_1024_training_completed',False),
        'engineering_failure':completed.get('engineering_failure'), 'source_reasoning_trajectories_included':True,
        'start_here':'COVERAGE_GENERALIZATION_V2_RESULTS.md','heavy_inventory':metadata+'/excluded_heavy_artifacts.json',
        'excluded':sorted(set(excluded)), 'excluded_classes':['private hidden tests','credentials','model and mapper weights','tensor caches','environments','Git internals','nested archives'],
        'manifest_self_hash':'This manifest excludes its own hash; the external receipt contains the final archive SHA-256.'}
    def pack(path):
        records = {}
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in sorted(set(selected)|set(generated)):
                if name in selected and name in generated:
                    raise ValueError('Generated metadata collides with original evidence: '+name)
                content = generated[name] if name in generated else selected[name].read_bytes()
                credential_scan(name,content)
                if not permitted(name):
                    raise ValueError('Forbidden output member: '+name)
                archive.writestr(PREFIX+name,content)
                records[name] = {'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
            manifest = {**manifest_base,'files':records,'uncompressed_bytes':sum(v['bytes'] for v in records.values())}
            archive.writestr(PREFIX+'REVIEW_BUNDLE_MANIFEST.json',encoded(manifest))
        verify_archive(path)
        return manifest
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gearshift-v2-bundle-') as tmp:
        provisional = Path(tmp)/'provisional.zip'; pack(provisional)
        if verify_regeneration:
            regenerated = regenerate_from_archive(provisional,rel,str(report_root.relative_to(repo)),expected_report,failure_rel)
        else:
            regenerated = {'passed':False,'executed':False,'reason':'Caller explicitly disabled archive regeneration.'}
        generated[metadata+'/compact_regeneration.json'] = encoded(regenerated)
        final = Path(tmp)/'final.zip'; manifest = pack(final)
        # Copy into a new file so an interrupted export cannot replace any archive.
        with final.open('rb') as source, out.open('xb') as target:
            for block in iter(lambda: source.read(8*1024**2),b''):
                target.write(block)
            target.flush(); os.fsync(target.fileno())
    verify_archive(out)
    receipt = {'path':str(out),'sha256':sha(out),'bytes':out.stat().st_size,'files':len(manifest['files']),
        'all_entry_hashes_verified':True,'credential_signature_scan_passed':True,'compact_regeneration_passed':regenerated['passed'],
        'delivery_commit':proof['delivery_commit'],'branch':proof['branch'],'created_epoch':time.time(),
        'publication_commit':PUBLICATION,'full_1024_1024_training_completed':manifest['full_1024_1024_training_completed']}
    with out.with_suffix(out.suffix+'.receipt.json').open('x') as f:
        json.dump(receipt,f,indent=2);f.write('\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--repo-root',required=True);p.add_argument('--result-root',required=True);p.add_argument('--evidence-root',required=True)
    p.add_argument('--output');p.add_argument('--execution-plan');p.add_argument('--engineering-failure');p.add_argument('--heavy-backup-manifest')
    p.add_argument('--input-overlay',help='Repository-contained exact executed inputs, archived at their original paths without editing historical checkout inputs')
    p.add_argument('--skip-regeneration',action='store_true');a=p.parse_args()
    print(json.dumps(build(a.repo_root,a.result_root,a.evidence_root,a.output,a.execution_plan,a.engineering_failure,
                           a.heavy_backup_manifest,not a.skip_regeneration,a.input_overlay),indent=2))
