#!/usr/bin/env python3
"""Explicitly allowlisted compact sparse-repair review ZIP and byte verification.

Default action inventories only. --build is an explicit packaging action, not
model generation, candidate execution, a provider operation, or publication.
Secrets, private tests, tensor files and nested archives are never packaged.
"""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RESULT=Path('results/sparse_repair_01')
DECLARATION=Path('configs/coding_pilot_v1/sparse_repair_01/declaration.json')
PUBLICATION_TAG='gearshift-progress-01'
PUBLICATION_COMMIT='64725974fa55459350d1c9d09037bab64d0c5ec6'
MANIFEST='SPARSE_REPAIR_FILE_MANIFEST.json'
MAX_MEMBER_BYTES=128*1024**2
MAX_TOTAL_BYTES=1024*1024**2
STORAGE_BINDING='configs/coding_pilot_v1/sparse_repair_01/private_scoring_storage_01.json'
DECISION_REVIEW=(RESULT/'report/decision_review.json').as_posix()
REPORT_INPUT=(RESULT/'report/input_records.json').as_posix()
REQUIRED_REPORTS=('SPARSE_REPAIR_RESULTS.md','ECONOMIC_CEILING.md','MAC_STUDIO_HANDOFF_STATUS.md','README_REPRODUCE_SPARSE_REPAIR.md')
STATIC_FILES=(
 'gearshift_studio_handoff/CODEX_STUDIO_SPARSE_REPAIR_01.md','gearshift_studio_handoff/IDEAS.md','gearshift_studio_handoff/README.md',
 'MAC_STUDIO_HANDOFF.md','requirements.txt','requirements.lock.txt','pytest.ini',
 'configs/coding_pilot_v1/confirmation_01/protocol.json','configs/coding_pilot_v1/pilot.json',
 'configs/coding_pilot_v1/cap_amendment_v1.json','configs/coding_pilot_v1/draft_membership.json',
 'configs/coding_pilot_v1/confirmation_01/private_test_identity.json',
 'configs/coding_pilot_v1/confirmation_01/OWNER_AMENDMENT_200_TASKS.md',
 'configs/coding_pilot_v1/reference/official_metadata.json',
 STORAGE_BINDING,
 'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/calibration/frozen_policy.json',
 'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/calibration/scorer_identity.json',
 'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/calibration/calibration_receipt.json',
 'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/resources/primary_pair01/model_weights_verified.json',
 'evidence/coding_pilot_v1/coverage_generalization_v2_20260918T233720Z/backups/external_heavy_backup_manifest.json',
 'data/coding_pilot_v1/visible/development.json',
)
# No full original result tree, public model tensors, nested ZIP or private values.
PUBLIC_EXTENSIONS={'.json','.jsonl','.md','.txt','.csv','.tsv','.xml','.log','.plist','.py','.toml','.yaml','.yml','.png','.svg'}
BINARY_EXTENSIONS={'.pt','.pth','.bin','.safetensors','.npy','.npz','.pkl','.pickle','.onnx'}
ARCHIVE_EXTENSIONS={'.zip','.tar','.gz','.bz2','.xz','.7z','.rar','.tgz'}
SKIP_DIRS={'.git','.venv','.pilot-venv','__pycache__','.pytest_cache','.cache','private','secrets','release_backups','uploads','staging'}
SECRET_PATTERNS=(
 ('private-key',rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
 ('runpod-token',rb'\brpa_[A-Za-z0-9_-]{24,}\b'),
 ('github-token',rb'\bgh[pousr]_[A-Za-z0-9]{20,}\b'),
 ('github-fine-grained-token',rb'\bgithub_pat_[A-Za-z0-9_]{24,}\b'),
 ('huggingface-token',rb'\bhf_[A-Za-z0-9]{24,}\b'),
 ('openai-token',rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b'),
 ('aws-access-key',rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 ('google-api-key',rb'\bAIza[0-9A-Za-z_-]{35}\b'),
 ('literal-bearer-token',rb'\bBearer [A-Za-z0-9._~-]{28,}'),
 ('literal-sensitive-field',rb'["\'](?:RUNPOD_API_KEY|OPENAI_API_KEY|HF_TOKEN|AWS_SECRET_ACCESS_KEY|refresh_token)["\']\s*:\s*["\'][A-Za-z0-9_+./=-]{24,}["\']'),
)


def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def json_bytes(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def check_total_size(total):
    if total>MAX_TOTAL_BYTES:raise ValueError('Compact review exceeds 1024MiB before compression')


def safe_member(name):
    path=Path(name)
    if not isinstance(name,str) or '\\' in name or path.is_absolute() or not path.parts or '..' in path.parts or name.endswith('/'):
        raise ValueError('Unsafe archive member path')
    if any(x in SKIP_DIRS for x in path.parts):raise ValueError('Excluded/private archive member path')
    if path.suffix.lower() in BINARY_EXTENSIONS|ARCHIVE_EXTENSIONS:
        raise ValueError('Tensor or nested archive forbidden')
    return path.as_posix()


def reject_secrets(raw,path):
    for label,pattern in SECRET_PATTERNS:
        if re.search(pattern,raw):raise ValueError(f'Possible {label} in {path}; credential value intentionally not echoed')


def read_public(root,relative):
    name=safe_member(str(relative));root=Path(root).resolve();p=root/name
    for cursor in (p,*p.parents):
        if cursor==root:break
        if cursor.is_symlink():raise ValueError('Symlink rejected: '+name)
    if not p.resolve().is_relative_to(root) or not p.is_file():raise ValueError('Missing/unscoped public input: '+name)
    if p.stat().st_size>MAX_MEMBER_BYTES:raise ValueError('Compact member exceeds128MiB: '+name)
    raw=p.read_bytes();reject_secrets(raw,name)
    return raw


def public_tree(root,relative,excluded):
    base=Path(root)/relative
    if not base.exists():return []
    rows=[]
    for folder,dirs,files in os.walk(base,followlinks=False):
        folder=Path(folder)
        for name in list(dirs):
            p=folder/name;rel=p.relative_to(root).as_posix()
            if p.is_symlink():raise ValueError('Symlink rejected: '+rel)
            if name in SKIP_DIRS or name.startswith('.'):
                excluded.append({'path':rel,'reason':'excluded directory'});dirs.remove(name)
        for name in files:
            p=folder/name;rel=p.relative_to(root).as_posix()
            if p.is_symlink():raise ValueError('Symlink rejected: '+rel)
            suffix=p.suffix.lower()
            if name.endswith(('.lock','.tmp','.DS_Store')) or name.startswith('.'):
                excluded.append({'path':rel,'reason':'transient/hidden file'});continue
            if suffix in BINARY_EXTENSIONS|ARCHIVE_EXTENSIONS or suffix not in PUBLIC_EXTENSIONS:
                excluded.append({'path':rel,'reason':'tensor/archive/unknown extension not allowed'});continue
            # Avoid recursively packaging a previous review's own receipts/inventory.
            if name in {'review_receipt.json','review_inventory.json'}:
                excluded.append({'path':rel,'reason':'prior packaging output'});continue
            rows.append(rel)
    return sorted(rows)


def dependency_closure(root,seeds):
    """Only explicit local Python imports under scripts/gearshift; no execution."""
    root=Path(root);pending=list(seeds);done=set()
    while pending:
        relative=pending.pop()
        if relative in done or not (root/relative).is_file():continue
        if Path(relative).parts[0] not in ('scripts','gearshift','tests'):raise ValueError('Dependency escapes source allowlist')
        done.add(relative)
        tree=ast.parse(read_public(root,relative),filename=relative)
        parts=Path(relative).with_suffix('').parts
        for node in ast.walk(tree):
            candidates=[]
            if isinstance(node,ast.Import):candidates=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                module=node.module or ''
                if node.level:
                    parent=list(parts[:-1]);base=parent[:len(parent)-node.level+1]
                    module='.'.join(base+([module] if module else []))
                candidates=[module]+[module+'.'+a.name for a in node.names]
            for module in candidates:
                if module.split('.')[0] not in ('scripts','gearshift'):continue
                path=module.replace('.','/')+'.py'
                if (root/path).is_file() and path not in done:pending.append(path)
                init=module.replace('.','/')+'/__init__.py'
                if (root/init).is_file() and init not in done:pending.append(init)
    return sorted(done)


def git_identities(root):
    def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    publication=git('rev-parse',PUBLICATION_TAG+'^{commit}')
    if publication!=PUBLICATION_COMMIT:raise ValueError('Immutable publication tag changed')
    return {'captured_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'branch':git('branch','--show-current'),
        'HEAD':git('rev-parse','HEAD'),'publication_tag':PUBLICATION_TAG,'publication_commit':publication,
        'publication_tag_object':git('rev-parse',PUBLICATION_TAG),
        'tracked_worktree_changes':git('status','--short','--untracked-files=no').splitlines(),
        'interpretation':'Each included working-tree byte is bound by the ZIP manifest; HEAD alone is not a proof of uncommitted source identity.'}


def portable_report_commands(payload):
    """Canonical extraction-root argv; never execute reports during packaging."""
    diagnostic=['python3','scripts/sparse_repair_report.py','--input',REPORT_INPUT,
                '--output','results/sparse_repair_01/report','--report','SPARSE_REPAIR_RESULTS.md']
    present=DECISION_REVIEW in payload
    if present:
        review=json.loads(payload[DECISION_REVIEW])
        if (REPORT_INPUT not in payload
            or review.get('schema')!='gearshift.sparse_repair.decision_review.v1'
            or review.get('evidence_input_sha256')!=digest(payload[REPORT_INPUT])
            or review.get('declaration_sha256')!=digest(payload[DECLARATION.as_posix()])):
            raise ValueError('Optional decision review is not bound to the packaged numerical snapshot/declaration')
        diagnostic += ['--decision-review',DECISION_REVIEW]
    return {'schema':'gearshift.sparse_repair.portable_report_commands.v1',
        'working_directory':'extracted_archive_root','commands':[
            ['python3','scripts/sparse_repair_economics.py','--input',
             'results/sparse_repair_01/economics/input_records.json','--output',
             'results/sparse_repair_01/economics','--report','ECONOMIC_CEILING.md'],diagnostic],
        'optional_decision_review_included':present,
        'decision_review_sha256':digest(payload[DECISION_REVIEW]) if present else None,
        'note':'Report CLI validates all five analyst decisions against computed measurements. '
               'The optional review stays separate from immutable numerical input_records.json. '
               'These commands do not execute models or candidates; use canonical relative paths for byte-identical Markdown.'}


def collect(root,require_complete_reports=False,git_state=None):
    root=Path(root).resolve();excluded=[];selected=set();missing=[]
    for name in REQUIRED_REPORTS+STATIC_FILES:
        if (root/name).exists():selected.add(name)
        elif name in REQUIRED_REPORTS:missing.append(name)
    if require_complete_reports and missing:raise ValueError('Required reports missing: '+', '.join(missing))
    if not (root/DECLARATION).is_file():raise ValueError('Frozen declaration required')
    d=read(root/DECLARATION);selected.add(DECLARATION.as_posix())
    tasks=d['calibration_tasks']+d['screen_tasks']
    if len(d['calibration_tasks'])!=4 or len(d['screen_tasks'])!=12 or len({x['task_id'] for x in tasks})!=16:
        raise ValueError('Review must preserve exactly4calibration+12screen task identities')
    for row in tasks:
        raw=read_public(root,row['history_path'])
        if digest(raw)!=row['history_sha256']:raise ValueError('Selected original history changed')
        selected.add(row['history_path'])
    # Include the complete original public40-task inputs so pinned template/hashes
    # remain reusable; only the selected16 saved source histories enter the ZIP.
    visible=root/'data/coding_pilot_v1/visible/development.json'
    if not visible.is_file() or digest(visible.read_bytes())!=d['inputs']['public_development_inputs']['sha256']:
        raise ValueError('Exact public development input bytes missing/changed')
    for folder in (RESULT,Path('configs/coding_pilot_v1/sparse_repair_01')):
        selected.update(public_tree(root,folder,excluded))
    # The public storage binding is provenance, not private input. Include its
    # exact provider receipt explicitly rather than silently packaging a stale
    # or missing destination identity alongside final scoring evidence.
    if (root/STORAGE_BINDING).exists():
        storage=read_public(root,STORAGE_BINDING)
        receipt=json.loads(storage)['provider_volume_receipt']
        if digest(read_public(root,receipt['path']))!=receipt['sha256']:
            raise ValueError('Frozen private-storage public provider receipt differs')
        selected.add(receipt['path'])
    seed_sources=[]
    for prefix,pattern in [('scripts','sparse_repair_*.py'),('gearshift','sparse_repair*.py'),('tests','test_sparse_repair*.py')]:
        seed_sources += [p.relative_to(root).as_posix() for p in sorted((root/prefix).glob(pattern))]
    selected.update(dependency_closure(root,seed_sources))
    payload={name:read_public(root,name) for name in sorted(selected)}
    check_total_size(sum(map(len,payload.values())))
    identities=git_identities(root) if git_state is None else git_state
    payload['GIT_SOURCE_IDENTITIES.json']=json_bytes(identities)
    payload['REPRODUCE_REPORT_COMMANDS.json']=json_bytes(portable_report_commands(payload))
    coverage={'declaration_sha256':digest(read_public(root,DECLARATION.as_posix())),
        'frozen_calibration_tasks':[x['task_id'] for x in d['calibration_tasks']],
        'frozen_screen_tasks':[x['task_id'] for x in d['screen_tasks']],
        'expected_screen_answers':504,'missing_reports_at_inventory':missing,
        'observed_calibration_case_receipts':sorted(x for x in payload if x.endswith('/calibration.json')),
        'observed_screen_answer_files':sorted(x for x in payload if '/screen/' in x and x.endswith('/answer.json')),
        'observed_screen_score_files':sorted(x for x in payload if '/scor' in x and x.endswith('.json')),
        'limitation':'File counts are artifact coverage only, not passed numerical gates or valid quality observations. SPARSE_REPAIR_RESULTS.md and manifest-bound report snapshot determine actual coverage, failures and limitations.',
        'exclusions':excluded,
        'not_included':['credentials','private test values','model and cache tensors','nested original review/backup archives','unselected original history contents','entire old result trees']}
    payload['REVIEW_COVERAGE_AND_EXCLUSIONS.json']=json_bytes(coverage)
    for name,raw in payload.items():safe_member(name);reject_secrets(raw,name)
    check_total_size(sum(map(len,payload.values()))+len(json_bytes(file_manifest(payload))))
    return payload,coverage


def file_manifest(payload):
    return {'schema':'gearshift.sparse_repair.compact_review.v1','member_count_excluding_manifest':len(payload),
        'size_limits':{'member_bytes':MAX_MEMBER_BYTES,'total_uncompressed_bytes':MAX_TOTAL_BYTES,
            'reason':'The original 768MiB engineering guard was too close to the complete 504-draw public evidence before final scoring/reporting. '
                     'Use 1024MiB rather than omit raw answers, durable resume state or selected-index traces; no user archive size limit was specified. '
                     'Private/credential/tensor/nested-archive exclusions remain unchanged.'},
        'files':[{'path':name,'bytes':len(raw),'sha256':digest(raw)} for name,raw in sorted(payload.items())],
        'manifest_self_hash':'Bound by external review_receipt.json; no circular self-hash.'}


def verify_archive(path,extract=True):
    """Read/hash every member, then safely extract to a temporary verification dir."""
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or MANIFEST not in names:raise ValueError('Duplicate or absent review manifest')
        for info in z.infolist():
            safe_member(info.filename)
            if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('ZIP symlink forbidden')
            if info.file_size>MAX_MEMBER_BYTES:raise ValueError('Oversize ZIP member')
        check_total_size(sum(i.file_size for i in z.infolist()))
        manifest_raw=z.read(MANIFEST);manifest=json.loads(manifest_raw)
        records=manifest['files'];by_name={r['path']:r for r in records}
        if len(by_name)!=len(records) or set(names)!=(set(by_name)|{MANIFEST}):raise ValueError('ZIP member set differs')
        with tempfile.TemporaryDirectory(prefix='gearshift-sparse-review-verify-') as temporary:
            dest=Path(temporary)
            for name in names:
                raw=z.read(name);reject_secrets(raw,name)
                if name!=MANIFEST:
                    r=by_name[name]
                    if len(raw)!=r['bytes'] or digest(raw)!=r['sha256']:raise ValueError('ZIP member checksum differs: '+name)
                if extract:
                    output=dest/safe_member(name);output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(raw)
                    if digest(output.read_bytes())!=digest(raw):raise ValueError('Extracted member bytes differ')
    return {'verified_members':len(names),'manifest_sha256':digest(manifest_raw),'verified_temporary_extraction':extract,
        'bytes':Path(path).stat().st_size,'sha256':digest(Path(path).read_bytes())}


def build(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve existing archive; choose explicit new output')
    payload,coverage=collect(root,require_complete_reports=True)
    payload[MANIFEST]=json_bytes(file_manifest(payload))
    output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_name(output.name+'.tmp')
    if tmp.exists():raise FileExistsError('Existing temporary archive requires inspection')
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for name,raw in sorted(payload.items()):
                info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=(0o100644<<16);z.writestr(info,raw)
        proof=verify_archive(tmp,extract=True)
        os.replace(tmp,output)
    except BaseException:
        # Retain failed bytes for debugging; never overwrite an original ZIP.
        raise
    receipt={**proof,'path':str(output),'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
             'source_root':str(root),'expected_screen_answers':504,
             'scientific_completion_asserted_by_packaging':False,'publication_performed':False}
    receipt_path=output.with_name(output.stem+'_receipt.json');receipt_path.write_bytes(json_bytes(receipt))
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    action=p.add_mutually_exclusive_group();action.add_argument('--build',action='store_true');action.add_argument('--verify',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'gearshift_sparse_repair_review.zip')
    p.add_argument('--inventory-output',type=Path)
    a=p.parse_args(argv)
    if a.verify:result=verify_archive(a.verify)
    elif a.build:result=build(a.root,a.output)
    else:
        payload,coverage=collect(a.root);result={'manifest':file_manifest(payload),'coverage':coverage,'archive_built':False}
        if a.inventory_output:a.inventory_output.write_bytes(json_bytes(result))
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
