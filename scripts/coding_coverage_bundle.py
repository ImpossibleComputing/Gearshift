#!/usr/bin/env python3
"""Compact review overlay; preserves original ZIP and exact executed sources."""
import hashlib,json,os,re,shutil,sys,tarfile,time,zipfile,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import sha,write
from gearshift.coding_coverage import EVIDENCE,PUBLICATION
from coding_post_bundle import permitted
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/EVIDENCE


def preserve_sources():
    objects=E/'runtime_source_objects';objects.mkdir(parents=True,exist_ok=True);index={}
    for archive in (ROOT/'evidence/coding_pilot_v1/control/parallel').glob('coverage_*/*/upload.tar'):
        files={}
        with tarfile.open(archive) as t:
            for m in t:
                p=Path(m.name)
                if m.isfile() and not p.is_absolute() and '..' not in p.parts and p.parts[0] in ['gearshift','scripts'] and p.suffix=='.py' and m.size<1_000_000:
                    b=t.extractfile(m).read();h=hashlib.sha256(b).hexdigest();dest=objects/(h+'.py')
                    if dest.exists() and dest.read_bytes()!=b:raise ValueError('Source object hash conflict')
                    if not dest.exists():dest.write_bytes(b)
                    files[m.name]={'sha256':h,'object':str(dest.relative_to(ROOT))}
        index[str(archive.relative_to(ROOT))]={'files':files,'provenance':'Exact uploaded source before worker execution; source trajectories are included separately from the pinned corpus.'}
    write(E/'runtime_source_snapshots.json',index)


def backup_heavy():
    backup=Path('/Users/qeetbastudio/Gearshift-artifacts/coverage-generalization');entries={}
    for p in sorted((ROOT/'results/coding_pilot_v1').glob('coverage_*/*/*/mapper_step_*.pt')):
        rel=p.relative_to(ROOT);h=sha(p);target=backup/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(p,target)
        if sha(target)!=h:raise ValueError('Heavy backup differs')
        entries[str(rel)]={'bytes':p.stat().st_size,'sha256':h,'canonical':str(p),'verified_backup':str(target),'restore':'Copy verified backup to repository-relative canonical path and check SHA-256.'}
    write(E/'excluded_heavy.json',{'mapper_checkpoints':entries,'large_tensor_caches':'None persisted in this round; full historical caches are reconstructed with frozen models in512-token chunks and released after use.',
        'starting_mapper_inventory':'publication/progress_01_provenance/heavy_backup_verification.json',
        'four_case_mapper_inventory':'evidence/coding_pilot_v1/post_progress01/excluded_heavy.json',
        'model_weights':'Exact pinned revisions and shard hashes in per-run runtime receipts and configs/coding_pilot_v1/pilot.json; recreate with existing download/bootstrap scripts.',
        'private_tests':'Intentionally excluded. The trusted local pinned data provide the exact21 validation and4 training scorer records. No confirmation records are needed.',
        'environment':'Pinned container/bootstrap and requirements.lock.txt retained; environment trees excluded.',
        'backup_limit':'Canonical and external backup are on the Studio disk, outside Git; no off-site durability claim.'})


def build():
    E.mkdir(parents=True,exist_ok=True);preserve_sources();backup_heavy()
    prior=ROOT/'gearshift_post_progress01_review.zip'
    if sha(prior)!='e4a5e8c5ad215ddaaea2d517a1296aee053d42583c20eda3a24620873a869caa':raise ValueError('Completed prior review ZIP changed')
    overlay={}
    for pattern in ['gearshift/*.py','scripts/*.py','tests/*.py','configs/coding_pilot_v1/coverage_generalization/**/*',
        'evidence/coding_pilot_v1/coverage_generalization/**/*','results/coding_pilot_v1/coverage_*/**/*',
        'evidence/coding_pilot_v1/control/parallel/coverage_*/**/*.json','evidence/coding_pilot_v1/parallel/coverage_*/**/*',
        'data/coding_pilot_v1/visible/coverage_generalization.json']:
        for p in ROOT.glob(pattern):
            if p.is_file() and not p.is_symlink() and permitted(str(p.relative_to(ROOT))):overlay['gearshift/'+str(p.relative_to(ROOT))]=p
    for name in ['COVERAGE_GENERALIZATION_RESULTS.md','REPRODUCE_COVERAGE_GENERALIZATION.md','requirements.lock.txt','.gitignore']:
        if (ROOT/name).is_file():overlay['gearshift/'+name]=ROOT/name
    out=ROOT/'gearshift_coverage_generalization_review.zip';temp=out.with_suffix('.zip.tmp');files={}
    def check(name,b):
        if not name.startswith('gearshift/') or '..' in Path(name).parts or not permitted(name):raise ValueError('Forbidden archive member '+name)
        for pattern in [rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',rb'rpa_[A-Za-z0-9]{25,}',rb'hf_[A-Za-z0-9]{30,}',rb'sk-[A-Za-z0-9_-]{35,}']:
            if re.search(pattern,b):raise ValueError('Credential signature in '+name)
    with zipfile.ZipFile(prior) as old,zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        def put(name,b):
            check(name,b);z.writestr(name,b);files[name.removeprefix('gearshift/')]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        for name in old.namelist():
            if name.endswith('/') or name=='gearshift/REVIEW_BUNDLE_MANIFEST.json' or name in overlay or not permitted(name):continue
            put(name,old.read(name))
        for name,p in sorted(overlay.items()):put(name,p.read_bytes())
        manifest={'schema':4,'created_epoch':time.time(),'files':files,'uncompressed_bytes':sum(x['bytes'] for x in files.values()),
            'delivery_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_bundle_sha256':sha(prior),'source_reasoning_trajectories_included':True,'publication_tag':'gearshift-progress-01','publication_commit':PUBLICATION,
            'start_here':'COVERAGE_GENERALIZATION_RESULTS.md','heavy_inventory':EVIDENCE+'/excluded_heavy.json','publishing_authorized':False,
            'excluded':['credentials','environments','large model weights','tensor caches','mapper weights','private hidden tests','Git internals','archive copies'],
            'manifest_self_hash':'Archive SHA-256 in external receipt; no recursive self-hash.'}
        z.writestr('gearshift/REVIEW_BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    os.replace(temp,out)
    with zipfile.ZipFile(out) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate archive members')
        for name,info in files.items():
            b=z.read('gearshift/'+name)
            if len(b)!=info['bytes'] or hashlib.sha256(b).hexdigest()!=info['sha256']:raise ValueError('Archive hash mismatch')
    receipt={'path':str(out),'sha256':sha(out),'bytes':out.stat().st_size,'files':len(files),'all_entry_hashes_verified':True,'credential_signature_scan_passed':True,'created_epoch':time.time()}
    write(ROOT/'gearshift_coverage_generalization_review.zip.receipt.json',receipt);print(json.dumps(receipt));return receipt

if __name__=='__main__':build()
