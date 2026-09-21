#!/usr/bin/env python3
"""Create a new compact primary review export; never runs models or candidates.

Example (from the checkout or extracted bundle):
  python scripts/coding_confirmation_review.py --plan results/.../primary/scoring/scoring_plan.json --delivery /absolute/new-export
Reproduce tables independently with coding_confirmation_report.py --plan <same>.
The optional second training seed never gates this primary delivery. Saved billing
and backup receipts remain timestamped observations, not live verification.
"""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_control import digest, sha
from scripts import coding_confirmation_report as reporter

ZIP_NAME = 'gearshift_confirmation_review.zip'
MANIFEST = 'CONFIRMATION_FILE_MANIFEST.json'
TEXT = {'.py', '.json', '.jsonl', '.csv', '.tsv', '.txt', '.log', '.md', '.yaml', '.yml', '.toml', '.ini', '.lock'}
SUFFIXES = TEXT | {'.png', '.svg', '.pdf'}
DENIED_PARTS = {'.git', '.venv', '.pilot-venv', 'venv', 'virtualenv', '__pycache__', '.pytest_cache',
    'private', 'protected', 'hidden_tests', 'private_tests', 'reference_answers', 'reference_solutions',
    'credentials', 'secrets', '.ssh', '.aws', '.runpod', '.cache', 'cache', 'caches', 'model_cache',
    'paired_cache', 'paired_caches', 'tensor_cache', 'node_modules', 'weights', 'model_weights'}
SECRET_KEYS = {'api_key', 'apikey', 'runpod_api_key', 'hf_token', 'openai_api_key', 'access_token',
    'refresh_token', 'auth_token', 'password', 'client_secret', 'private_key', 'bearer_token'}
SIGNATURES = [rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA|ENCRYPTED) PRIVATE KEY-----',
    rb'rpa_[A-Za-z0-9]{25,}', rb'hf_[A-Za-z0-9]{30,}', rb'sk-[A-Za-z0-9_-]{35,}',
    rb'(?i)authorization\s*[:=]\s*["\x27]?Bearer\s+[A-Za-z0-9_.-]{20,}']
MAX_FILE_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 8 * 1024**3
ORIGINAL_EVALUATION = 'results/coding_pilot_v1/coverage_generalization_v2_20260918T233720Z/evaluation'


def encoded(value): return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
def hash_bytes(value): return hashlib.sha256(value).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def relative(name):
    name = str(name); p = PurePosixPath(name)
    if not name or '\\' in name or '\x00' in name or p.is_absolute() or '..' in p.parts or p.as_posix() != name:
        raise ValueError('Unsafe review path: ' + name)
    return p


def permitted(name):
    p = relative(name); lower = [v.lower() for v in p.parts]; leaf = lower[-1]
    if set(lower) & DENIED_PARTS or any(v.startswith('.env') for v in lower): return False
    if any(v.startswith('.') and v != '.gitignore' for v in lower): return False
    if leaf in {'private.json','private_tests.json','hidden_tests.json','reference_answers.json','reference_solution.json','reference_solutions.json'}: return False
    if leaf.startswith(('id_rsa', 'id_ed25519')) or leaf in {'credentials.json', 'secrets.json', 'provider_auth.json'}: return False
    if leaf.endswith(('.tmp', '.pem', '.key')): return False
    if p.suffix == '.lock' and leaf not in {'uv.lock', 'poetry.lock', 'requirements.lock'}: return False
    # The only numerical archive is the small, report-bound task resample matrix.
    if p.suffix == '.npz': return p.parts[-3:] == ('primary', 'report', 'bootstrap_indices.npz')
    return p.suffix.lower() in SUFFIXES or leaf in {'readme', 'license', '.gitignore'}


def public_path(repo, name):
    p = relative(name)
    if not permitted(name): raise ValueError('Non-public or heavyweight input cannot be exported: ' + str(name))
    return reporter.scoped(repo, str(p))


def scan(name, content):
    if any(re.search(pattern, content) for pattern in SIGNATURES):
        raise ValueError('Credential signature in review input: ' + name)
    if Path(name).suffix not in {'.json', '.jsonl'}: return
    values = [json.loads(line) for line in content.decode().splitlines() if line.strip()] if name.endswith('.jsonl') else [json.loads(content)]
    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in SECRET_KEYS and item not in (None, '', False, '[REDACTED]', '<redacted>', 'REDACTED'):
                    raise ValueError('Credential field in review input: ' + name)
                # One preserved declaration uses this exact policy sentence as a label.
                policy_label = key == 'hidden_tests' and item == 'Loaded only for isolated scoring after all optimization and free generation; never optimizer/model inputs.'
                if key.lower() in {'private_tests', 'hidden_tests', 'reference_solution', 'reference_solutions', 'reference_answers'} and not policy_label and item not in (None, '', False, [], {}):
                    raise ValueError('Private test/reference values in review input: ' + name)
                if key.lower() == 'authorization' and isinstance(item, str) and not item.lower().startswith(('owner', 'user', 'explicit', 'direct', 'approved', 'existing')):
                    raise ValueError('Unrecognized authorization field in review input: ' + name)
                inspect(item)
        elif isinstance(value, list):
            for item in value: inspect(item)
    for value in values: inspect(value)


class Collection:
    def __init__(self, repo): self.repo, self.files, self.excluded = Path(repo).resolve(), {}, []

    def add(self, name, expected=None):
        name = str(relative(name)); path = public_path(self.repo, name)
        if not path.is_file(): raise ValueError('Required compact evidence is missing: ' + name)
        if path.stat().st_size > MAX_FILE_BYTES: raise ValueError('Compact evidence exceeds size limit: ' + name)
        if name in self.files and expected and self.files[name] not in (None, expected):
            raise ValueError('Conflicting evidence identities: ' + name)
        self.files[name] = expected or self.files.get(name)
        return path

    def tree(self, name):
        root = reporter.scoped(self.repo, str(relative(name)))
        if not root.exists(): return
        # Prune before opening; never follows links, environments, private roots or weights.
        for folder, dirs, files in os.walk(root, followlinks=False):
            for dirname in list(dirs):
                p = Path(folder) / dirname
                if p.is_symlink(): raise ValueError('Linked review input: ' + str(p.relative_to(self.repo)))
                if dirname.lower() in DENIED_PARTS or dirname.startswith('.'): dirs.remove(dirname)
            for leaf in sorted(files):
                p = Path(folder) / leaf; rel = str(p.relative_to(self.repo))
                if permitted(rel): self.add(rel)
                else: self.excluded.append(rel)

    def bytes(self, name):
        path = public_path(self.repo, name); before = path.stat(); content = path.read_bytes(); after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('Evidence changed during export; retry a new snapshot: ' + name)
        if len(content) > MAX_FILE_BYTES or self.files[name] and hash_bytes(content) != self.files[name]:
            raise ValueError('Review evidence hash/size changed: ' + name)
        scan(name, content)
        return content


def gate(repo, plan_name):
    """Reject private/linked paths before the report loader touches any bytes."""
    files = Collection(repo); plan = read(files.add(plan_name))
    d = read(files.add(plan['declaration_path'], plan['declaration_sha256']))
    root_name = str(relative(plan['result_root'])); root = reporter.scoped(files.repo, root_name)
    for name, expected in {**d['inputs'], **d['implementation'], **d['scorer']['implementation_hashes']}.items(): files.add(name, expected)
    closure_name = root_name + '/' + plan['generation_closure_path']
    closure = read(files.add(closure_name, plan['generation_closure_file_sha256']))
    for item in closure['files']: files.add(root_name + '/' + item['path'], item['sha256'])
    manifest_name = root_name + '/' + plan['manifest_path']; manifest = read(files.add(manifest_name))
    for item in manifest['files']: files.add(root_name + '/' + item['path'], item['sha256'])
    files.add(plan['generation_plan_path'], plan['generation_plan_sha256'])
    # Verify all declaration dependencies before invoking an analysis reader.
    for name in list(files.files): files.bytes(name)
    loaded = reporter.load_records(plan_name, files.repo)
    report_root = root / 'primary/report'; rm_name = root_name + '/primary/report/report_manifest.json'
    rm = read(files.add(rm_name)); summary = read(files.add(root_name + '/primary/report/summary.json'))
    if (rm.get('experiment_id') != d['experiment_id'] or rm.get('cohort') != 'primary' or
        rm.get('source_bindings') != loaded['bindings'] or rm.get('generation_or_candidate_execution_performed') is not False or
        rm.get('report_script_sha256') != d['implementation']['scripts/coding_confirmation_report.py'] or
        summary.get('bindings') != loaded['bindings'] or summary.get('experiment_id') != d['experiment_id'] or
        summary.get('cohort') != 'primary' or summary.get('training_seed') != d['primary_training_seed']):
        raise ValueError('Completed report does not bind the frozen primary evidence')
    names = set()
    for item in rm['files']:
        if len(relative(item['path']).parts) != 1 or item['path'] in names: raise ValueError('Invalid report manifest membership')
        names.add(item['path']); p = files.add(str((report_root/item['path']).relative_to(files.repo)), item['sha256'])
        if p.stat().st_size != item['bytes']: raise ValueError('Report file size changed')
    required = {'summary.json','REPORT.md','per_task_seed.csv','per_task.csv','task_gains_losses.csv','histories.csv','conditions.csv','bootstrap_indices.npz'}
    if not required <= names: raise ValueError('Completed report is missing required compact outputs')
    calculated, _, _, indices = reporter.cluster_statistics(loaded['rows'], d['task_ids'], loaded['analysis'])
    if any(summary.get(k) != v for k, v in calculated.items()): raise ValueError('Report statistics differ from immutable scored records')
    if summary.get('answers') != len(loaded['rows']) or summary.get('missing_draws') != sum(r['missing'] for r in loaded['rows']):
        raise ValueError('Report coverage differs')
    if summary['identities'] != {'source_commit':d['source_commit'], 'publication_commit':d['publication_commit'],
        'primary_checkpoints':{arm:{k:cp[k] for k in ['step','mapper_sha256']} for arm,cp in d['primary_checkpoints'].items()}}:
        raise ValueError('Report checkpoint/source identity differs')
    if summary.get('diagnostics') != reporter.diagnostics_summary(loaded['rows']) or summary.get('history_diagnostics') != loaded['history_rows']:
        raise ValueError('Report diagnostics differ from compact records')
    if (summary['setup_measurements']['records'] != loaded['setup_rows'] or
        summary['sampler_overhead_measurements']['records'] != loaded['overhead_rows']):
        raise ValueError('Report setup/timing records differ')
    if summary.get('bootstrap_archive_sha256') != sha(report_root/'bootstrap_indices.npz'):
        raise ValueError('Report bootstrap archive binding differs')
    with reporter.np.load(report_root/'bootstrap_indices.npz', allow_pickle=False) as saved:
        if saved.files != ['indices'] or not reporter.np.array_equal(saved['indices'], indices):
            raise ValueError('Saved bootstrap matrix differs from frozen joint resamples')
    if (report_root/'REPORT.md').read_text() != reporter.render(summary): raise ValueError('Report prose differs from saved numerical summary')
    for name in files.files: files.bytes(name)
    return files, loaded, summary, manifest, closure


def add_repair(files, result_name, original_name):
    root_name = result_name + '/scorer_repair'; root = files.repo/root_name
    plan = read(files.add(root_name+'/rescore_plan.json')); manifest = read(files.add(root_name+'/rescored_answer_manifest.json'))
    summary = read(files.add(root_name+'/report/summary.json'))
    files.add(root_name+'/report/SCORER_REPAIR_RESULTS.md')
    if (manifest['plan_sha256'] != digest(plan) or manifest['committed'] != plan['expected_answers'] or
        summary['plan_sha256'] != digest(plan) or summary['manifest_sha256'] != sha(root/'rescored_answer_manifest.json') or
        summary.get('diagnostic_complete') is not True or summary.get('original_scores_preserved') is not True):
        raise ValueError('Scorer repair is not complete and bound')
    original = read(files.add(original_name+'/scored_answer_manifest.json', plan['original_manifest_sha256']))
    old_closure = read(files.add(original_name+'/generation_closure.json'))
    if digest(old_closure) != original['generation_closure_sha256']: raise ValueError('Original generation closure changed')
    for item in old_closure.get('kl_files', []): files.add(original_name+'/'+item['path'],item['sha256'])
    for item in old_closure.get('jobs', []):
        jobroot=original_name+'/jobs/'+str(relative(item['job_id']))
        complete=read(files.add(jobroot+'/generation_complete.json',item['completion_sha256']))
        files.add(jobroot+'/job.json')
        for control in complete.get('control_files', []): files.add(jobroot+'/'+control['path'],control['sha256'])
        for raw in complete.get('files', []):
            files.add(jobroot+'/'+str(Path(raw['path']).parent/'complete.json'),raw['receipt_sha256'])
            files.add(jobroot+'/'+str(Path(raw['path']).parent/'draw_identity.json'))
    if original['generation_closure_sha256'] != plan['original_generation_closure_sha256']:
        raise ValueError('Original scorer generation identity differs')
    expected = {r['record_id']:r for r in plan['records']}
    if len(expected) != plan['expected_answers'] or len(manifest['files']) != len(expected): raise ValueError('Scorer-repair coverage differs')
    seen = set()
    for item in manifest['files']:
        rid=item['record_id']; row=expected[rid]
        if rid in seen: raise ValueError('Duplicate scorer-repair record')
        seen.add(rid)
        ap=files.add(original_name+'/'+row['answer_path'], row['answer_sha256']); files.add(original_name+'/'+row['path'], row['sha256'])
        raw=read(ap)
        if raw.get('source_history_sha256'):
            files.add(str((ap.parents[2]/'source_history.json').relative_to(files.repo)),raw['source_history_sha256'])
        corrected=read(files.add(root_name+'/'+item['path'], item['sha256'])); b=corrected['binding']
        if any(b.get(k)!=v for k,v in {'record_id':rid,'plan_sha256':digest(plan),'answer_sha256':row['answer_sha256'],
            'original_score_sha256':row['sha256'],'code_sha256':row['code_sha256']}.items()): raise ValueError('Scorer-repair receipt binding differs')
        old=read(files.repo/original_name/row['path'])
        if corrected['original_score'] != old['score']: raise ValueError('Original score was not preserved')
    for item in summary['diagnostics']:
        for rep in item['repetitions']: files.add(root_name+'/'+rep['path'], rep['sha256'])
    files.tree(root_name)
    repair_report = root_name+'/report/SCORER_REPAIR_RESULTS.md'
    supplement = root/'supplement_v1/FILE_MANIFEST.json'
    if supplement.is_file():
        sm = read(files.add(str(supplement.relative_to(files.repo))))
        if sm.get('private_tests_included') is not False: raise ValueError('Unsafe scorer supplement')
        seen = set()
        for item in sm['files']:
            if len(relative(item['path']).parts) != 1 or item['path'] in seen: raise ValueError('Invalid supplement inventory')
            seen.add(item['path']); p = files.add(root_name+'/supplement_v1/'+item['path'],item['sha256'])
            if p.stat().st_size != item['bytes']: raise ValueError('Supplement file size changed')
        if 'SCORER_REPAIR_RESULTS.md' not in seen: raise ValueError('Supplement report absent')
        if read(supplement.parent/'source_summary.json') != summary: raise ValueError('Supplement source summary differs')
        repair_report = root_name+'/supplement_v1/SCORER_REPAIR_RESULTS.md'
    return {'report_path':repair_report, 'answers':plan['expected_answers'], 'missing':manifest['missing'], 'task_clusters':summary['task_clusters'],
        'diagnostic_repeat_count':sum(len(x['repetitions']) for x in summary['diagnostics']),
        'saved_validation_not_fresh_confirmation':True, 'summary_path':root_name+'/report/summary.json',
        'original_manifest_path':original_name+'/scored_answer_manifest.json'}


def git_identity(repo, declaration):
    def git(*args): return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    tag = git('rev-parse','gearshift-progress-01^{commit}')
    if tag != declaration['publication_commit']: raise ValueError('Preserved publication tag differs')
    verified = {}
    for name, expected in declaration['implementation'].items():
        content = subprocess.check_output(['git','-C',str(repo),'show',declaration['source_commit']+':'+name])
        if hash_bytes(content) != expected: raise ValueError('Scientific commit does not contain the frozen source: '+name)
        verified[name] = expected
    return {'frozen_source_commit_files_verified':verified,'source_commit':declaration['source_commit'],'publication_commit':tag,
        'publication_tag':'gearshift-progress-01','delivery_checkout_commit':git('rev-parse','HEAD'),
        'worktree_status':git('status','--porcelain'), 'live_provider_verification_performed':False,
        'git_modified_push_or_publish_performed':False}


def secondary_snapshot(root, d):
    arms={}
    for arm in ('FIXED','ROTATING'):
        folder=root/'replication/arms'/arm
        candidates=[folder/'checkpoints/step_1024/export/training_complete.json',
                    folder/'training_complete.json']
        present=[p for p in candidates if p.is_file()]
        p=present[0] if present else candidates[0]
        value=read(p) if present else None
        if any(read(other)!=value for other in present[1:]):
            raise ValueError('Conflicting secondary training completion receipts: '+arm)
        arms[arm]={'training_completion_receipt_present':value is not None,
            'full_1024_completion_reported':bool(value and value.get('full_target_completed') is True and value.get('completed_updates')==1024),
            'receipt_path':str(p.relative_to(root)) if value else None}
    # Deliberately do not infer validated secondary analysis from training completion.
    return {'training_seed':d.get('secondary_training_seed'), 'arms':arms,
        'status':'Separate secondary work; available metadata snapshot only, no secondary result claim in this primary export.',
        'primary_delivery_waited_for_secondary':False,
        'secondary_generation_seal_present':(root/'secondary/generation_closure.json').is_file(),
        'secondary_scored_manifest_present':(root/'secondary/scoring/scored_answer_manifest.json').is_file()}


def saved_resources(files, evidence_name):
    """Extract only dated, saved accounting; no network or claim of live status."""
    root=files.repo/evidence_name/'resources'; candidates=[]; inventories=[]
    if root.is_dir():
        for p in sorted(root.glob('*.json')):
            value=read(p)
            if not isinstance(value,dict): continue
            item={'path':str(p.relative_to(files.repo)),'sha256':sha(p),
                  'observed_at_utc':value.get('observed_at_utc')}
            if any(k in value for k in ('this_round_compute_estimate_usd','conservative_cumulative_estimate_usd')):
                item['saved_values']={k:value[k] for k in ('this_round_compute_estimate_usd','conservative_cumulative_estimate_usd',
                    'new_storage_unfinalized','soft_target_usd','hard_ceiling_usd') if k in value}
                candidates.append(item)
            if 'inventory' in p.name or 'live_snapshot' in p.name: inventories.append(item)
    dated=[x for x in candidates if x['observed_at_utc']]
    return {'latest_dated_cost_receipt':max(dated,key=lambda x:x['observed_at_utc']) if dated else None,
        'all_cost_receipts':candidates,'resource_inventory_receipts':inventories,
        'live_provider_check_performed':False,'export_time_is_not_observation_time':True}


def verify_archive(path, expected_manifest_sha256=None):
    """Hash every member after compression, including the non-self-referential inventory."""
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or MANIFEST not in names: raise ValueError('Duplicate/missing archive member')
        inventory=z.read(MANIFEST)
        if expected_manifest_sha256 and hash_bytes(inventory)!=expected_manifest_sha256: raise ValueError('Inventory changed in ZIP')
        m=json.loads(inventory); entries=m['files']
        if len(entries)!=len({r['path'] for r in entries}) or set(names)!={r['path'] for r in entries}|{MANIFEST}:
            raise ValueError('Archive membership differs from inventory')
        for item in entries:
            name=item['path']
            if not permitted(name): raise ValueError('Prohibited archive member: '+name)
            content=z.read(name)
            if len(content)!=item['bytes'] or hash_bytes(content)!=item['sha256']: raise ValueError('Archive member hash/size differs: '+name)
            scan(name,content)
        return {'archive_sha256':sha(path), 'archive_bytes':Path(path).stat().st_size,
            'inventory_sha256':hash_bytes(inventory), 'inventory_bytes':len(inventory),
            'verified_members':len(names), 'every_member_read_back':True}


def export(plan_path, delivery, *, repo=ROOT, original_evaluation=ORIGINAL_EVALUATION):
    repo=Path(repo).resolve(); out=Path(delivery).absolute()
    if out.exists() or out.is_symlink(): raise ValueError('Delivery already exists; use a new directory')
    files,loaded,summary,manifest,closure=gate(repo,plan_path)
    root=loaded['root']; d=loaded['declaration']; result_name=str(root.relative_to(repo))
    evidence_name='evidence/coding_pilot_v1/'+d['experiment_id']
    # Include only public source/configuration trees, never a whole-repository walk.
    for name in ('gearshift','scripts','tests','configs',str(Path(loaded['plan']['declaration_path']).parent),evidence_name,result_name):
        if name=='.': continue
        files.tree(name)
    for name in ('README.md','RESULTS.md','pyproject.toml','uv.lock','requirements.lock','requirements.txt','LICENSE','.gitignore'):
        if (repo/name).is_file(): files.add(name)
    for cp in d['primary_checkpoints'].values():
        files.add(cp['manifest_path'], cp['manifest_sha256'])
        files.add(cp['actual_checkpoint_byte_proof'],d['inputs'][cp['actual_checkpoint_byte_proof']])
    corpus=d.get('secondary_training',{}).get('corpus_manifest')
    if corpus: files.add(corpus,d['secondary_training']['corpus_manifest_sha256'])
    repair=add_repair(files,result_name,str(relative(original_evaluation)))
    git=git_identity(repo,d); secondary=secondary_snapshot(root,d)
    now=datetime.now(timezone.utc).isoformat()
    coverage={'task_count':len(d['task_ids']), 'task_ids':d['task_ids'], 'conditions':d['primary_conditions'],
        'seed_indices':[0,1,2], 'answers_generated':closure['answer_count'], 'answers_scored':manifest['committed_answers'],
        'explicit_missing_outcomes':summary['missing_draws'], 'source_histories':sum(r['role']=='source' for r in loaded['history_rows']),
        'independent_small_histories':sum(r['role']=='small' for r in loaded['history_rows']),
        'endpoint':1024,'primary_training_seed':d['primary_training_seed']}
    metadata={'schema_version':1,'experiment_id':d['experiment_id'],'export_utc':now,
        'primary_status':'Generation and score transactions closed; explicit missing outcomes retained.',
        'coverage':coverage,'bindings':loaded['bindings'],'git':git,'secondary':secondary,'scorer_repair':repair,
        'packager_sha256':sha(Path(__file__)), 'saved_resource_receipts_only':True,
        'saved_resource_accounting':saved_resources(files,evidence_name),
        'resource_status_note':'Saved receipts have their own observation times; export time is not a fresh provider check. Use included resources/* billing/reconciliation/inventory receipts for saved costs and remaining resources.',
        'heavy_backup_note':'Weights/checkpoints are excluded. Frozen checkpoint manifests and actual-byte proof receipts retain hashes and retrieval paths. No heavy bytes or external backups are rechecked during packaging.',
        'private_tests_or_reference_answers_included':False,'model_or_tensor_files_included':False,
        'candidate_execution_performed':False,'excluded_paths':sorted(set(files.excluded))}
    extra='\n\n## Review delivery\n\n'+f"Exported {now}; endpoint 1,024; {coverage['task_count']} tasks × 8 conditions × 3 answer seeds = {coverage['answers_scored']} committed primary scores ({coverage['explicit_missing_outcomes']} explicit missing).\n\n"+secondary['status']+' Primary delivery did not wait for it. See REVIEW_DELIVERY.json for per-arm saved progress.\n\nScorer repair, original versus corrected validation conclusions, and all retained repetitions are in SCORER_REPAIR_RESULTS.md and the scorer_repair/report records. These are earlier examined validation tasks, not the fresh primary population.\n\n'+metadata['resource_status_note']+'\n\n'+metadata['heavy_backup_note']+'\n'
    generated={'CONFIRMATION_RESULTS.md':(reporter.render(summary)+extra).encode(),
        'SCORER_REPAIR_RESULTS.md':files.bytes(repair['report_path']),
        'REVIEW_DELIVERY.json':encoded(metadata),
        'REPRODUCE.md':('''# Reproduce compact reports\n\nUse the included pyproject/lockfile to install analysis dependencies. No weights, private tests, or candidate execution are needed for these report commands. Run from the extracted archive root:\n\n```sh\npython scripts/coding_confirmation_report.py --plan '''+str(plan_path)+'''\npython scripts/coding_scorer_repair_report.py --root '''+result_name+'''/scorer_repair\n```\n\nThe primary report writes only its primary/report namespace. Preserve the delivery ZIP as the original export. The saved bootstrap matrix and frozen analysis specify identical task-cluster resamples. Raw generation/scoring records are immutable inputs. The optional second training seed remains a separate analysis.\n\nCONFIRMATION_FILE_MANIFEST.json inventories every other ZIP member. The external delivery receipt gives the manifest's own SHA-256/size and the ZIP SHA-256/size, avoiding circular self-hashes. Every ZIP member was read back and verified.\n''').encode()}
    # Build and verify privately; the completion receipt is published last.
    # mkdir reservation rejects an existing/racing delivery rather than replacing it.
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.confirmation-review-',dir=out.parent))
    try:
        inventory=[];total=0; archive=stage/ZIP_NAME
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for name in sorted(set(files.files)|set(generated)):
                if name in files.files and name in generated: raise ValueError('Generated delivery name conflicts with input')
                content=generated[name] if name in generated else files.bytes(name); total+=len(content)
                if total>MAX_TOTAL_BYTES: raise ValueError('Compact export exceeded total size limit')
                scan(name,content);z.writestr(name,content)
                inventory.append({'path':name,'bytes':len(content),'sha256':hash_bytes(content)})
            inv=encoded({'schema_version':1,'experiment_id':d['experiment_id'],'export_utc':now,
                'inventory_self_hash':'Recorded in external DELIVERY_RECEIPT.json; avoids circular hash.',
                'files':inventory,'uncompressed_bytes_excluding_inventory':total})
            z.writestr(MANIFEST,inv)
        verification=verify_archive(archive,hash_bytes(inv))
        for name,content in generated.items(): (stage/name).write_bytes(content)
        (stage/MANIFEST).write_bytes(inv)
        receipt={**metadata,**verification,'zip_path':str(out/ZIP_NAME),
            'results_path':str(out/'CONFIRMATION_RESULTS.md'),'repair_results_path':str(out/'SCORER_REPAIR_RESULTS.md'),
            'archive_member_count':len(inventory)+1,'all_outputs_verified':True}
        (stage/'DELIVERY_RECEIPT.json').write_bytes(encoded(receipt))
        # mkdir reservation prevents replacing even an empty concurrently created export.
        out.mkdir()
        for p in sorted(stage.iterdir()):
            if p.name != 'DELIVERY_RECEIPT.json': os.replace(p,out/p.name)
        os.replace(stage/'DELIVERY_RECEIPT.json',out/'DELIVERY_RECEIPT.json')
        stage.rmdir()
        return receipt
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',required=True);ap.add_argument('--delivery',required=True)
    ap.add_argument('--original-evaluation',default=ORIGINAL_EVALUATION)
    args=ap.parse_args()
    print(json.dumps(export(args.plan,args.delivery,original_evaluation=args.original_evaluation),indent=2))
