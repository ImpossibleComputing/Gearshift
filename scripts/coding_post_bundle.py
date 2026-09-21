#!/usr/bin/env python3
"""Verified compact post-progress overlay with separate heavy-artifact backups."""
import hashlib,json,os,re,shutil,sys,time,zipfile,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import sha,write
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence/coding_pilot_v1/post_progress01'
DENY_SUFFIX={'.pt','.bin','.safetensors','.tar','.gz','.zip','.lock','.tmp','.pyc','.npy','.npz'}
DENY_NAMES={'connection.json','pod.json','volume.json','tokenizer.json'}

def permitted(name):
    p=Path(name)
    return not (p.suffix in DENY_SUFFIX or p.name in DENY_NAMES or any(s in {'.git','.venv','.pilot-venv','__pycache__','.pytest_cache','private','cache_lru'} for s in p.parts) or p.name.startswith('.env'))

def backup_heavy():
    backup=Path('/Users/qeetbastudio/Gearshift-artifacts/post-progress-01');entries={};discarded={}
    for p in sorted((ROOT/'results/coding_pilot_v1').glob('post_progress01_*/*/mapper*.pt')):
        if p.name=='mapper_latest.pt':continue
        rel=p.relative_to(ROOT);h=sha(p);target=backup/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(p,target)
        if sha(target)!=h:raise ValueError('Heavy backup changed')
        entries[str(rel)]={'bytes':p.stat().st_size,'sha256':h,'canonical':str(p),'verified_backup':str(target),'restore':'Copy backup to canonical path and verify SHA-256. Starting selected mapper is separately backed up in the Progress01 inventory.'}
    for p in (ROOT/'results/coding_pilot_v1').glob('post_progress01_*/*/cache_inventory.json'):discarded.update(json.loads(p.read_text())['files'])
    write(E/'excluded_heavy.json',{'mapper_checkpoints':entries,'discarded_reproducible_caches':discarded,'starting_artifacts':'publication/progress_01_provenance/heavy_backup_verification.json','model_weights_and_tokenizers':'Exact pinned model revisions/shard hashes and regeneration recipes remain in evidence/coding_pilot_v1/recovery_20260917/excluded_heavy.json.','private_tests':'Excluded; trusted local scorer only. The four-case subset is derived by the predeclared task IDs from the existing pinned training split.','environment':'Recreate pinned container/bootstrap and lockfile.','backup_limitation':'External copies are outside Git on the Studio disk; not off-site disaster recovery.'})

def preserve_sources():
    objects=E/'runtime_source_objects';objects.mkdir(parents=True,exist_ok=True);index={}
    for archive in (ROOT/'evidence/coding_pilot_v1/control/parallel').glob('post_progress01_*/*/upload.tar'):
        files={}
        with tarfile.open(archive) as t:
            for m in t:
                p=Path(m.name)
                if m.isfile() and '..' not in p.parts and p.parts[0] in ('gearshift','scripts') and p.suffix=='.py' and m.size<1000000:
                    b=t.extractfile(m).read();h=hashlib.sha256(b).hexdigest();dest=objects/(h+'.py')
                    if not dest.exists():dest.write_bytes(b)
                    files[m.name]={'sha256':h,'object':str(dest.relative_to(ROOT))}
        index[str(archive.relative_to(ROOT))]={'files':files,'source':'Exact worker upload before execution, not assumed equal to latest packaging code.'}
    write(E/'runtime_source_snapshots.json',index)


def build():
    preserve_sources()
    E.mkdir(parents=True,exist_ok=True);backup_heavy()
    prior=ROOT/'gearshift_recovery_review.zip'
    if sha(prior)!='2aa52917942d25b394b9331f362526f0410a8a90dece3a4ca7e574fcc7bb7221':raise ValueError('Frozen input review ZIP changed')
    overlay={}
    for pattern in ['gearshift/*.py','scripts/*.py','tests/*.py','publication/**/*','configs/coding_pilot_v1/post_progress01/**/*','evidence/coding_pilot_v1/post_progress01/**/*','results/coding_pilot_v1/post_progress01_*/**/*','evidence/coding_pilot_v1/control/parallel/post_progress01_*/**/*.json','evidence/coding_pilot_v1/parallel/post_progress01_*/**/*','data/coding_pilot_v1/visible/post_progress01_*.json']:
        for p in ROOT.glob(pattern):
            if p.is_file() and not p.is_symlink() and permitted(str(p.relative_to(ROOT))):overlay['gearshift/'+str(p.relative_to(ROOT))]=p
    for name in ['POST_PROGRESS_01_RESULTS.md','REPRODUCE_POST_PROGRESS_01.md','.gitignore','uv.lock','pyproject.toml']:
        if (ROOT/name).exists():overlay['gearshift/'+name]=ROOT/name
    out=ROOT/'gearshift_post_progress01_review.zip';tmp=out.with_suffix('.zip.tmp');files={}
    def inspect(name,b):
        if not name.startswith('gearshift/') or '..' in Path(name).parts or not permitted(name):raise ValueError('Forbidden archive member '+name)
        for pat in [rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',rb'rpa_[A-Za-z0-9]{25,}',rb'hf_[A-Za-z0-9]{30,}',rb'sk-[A-Za-z0-9_-]{35,}']:
            if re.search(pat,b):raise ValueError('Credential signature found; inspect '+name)
    with zipfile.ZipFile(prior) as old,zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        def put(name,b):
            inspect(name,b);z.writestr(name,b);files[name.removeprefix('gearshift/')]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        for name in old.namelist():
            if name.endswith('/') or name=='gearshift/REVIEW_BUNDLE_MANIFEST.json' or name in overlay or not permitted(name):continue
            put(name,old.read(name))
        for name,p in sorted(overlay.items()):put(name,p.read_bytes())
        manifest={'schema':3,'created_epoch':time.time(),'files':files,'uncompressed_bytes':sum(x['bytes'] for x in files.values()),'source_bundle_sha256':sha(prior),'source_reasoning_trajectories_included':True,'publication_tag':'gearshift-progress-01','publication_commit':'64725974fa55459350d1c9d09037bab64d0c5ec6','start_here':'POST_PROGRESS_01_RESULTS.md','editorial_master_snapshot':'publication/EDITORIAL_MASTER_20260917.md','excluded_categories':['credentials','environments','model weights','large tensor caches','mapper checkpoints','private hidden tests','Git internals','archive copies'],'heavy_inventory':'evidence/coding_pilot_v1/post_progress01/excluded_heavy.json','publishing_authorized':False,'manifest_self_hash':'Archive SHA-256 supplied in external receipt; no recursive self-hash.'}
        z.writestr('gearshift/REVIEW_BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    os.replace(tmp,out)
    with zipfile.ZipFile(out) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        for name,item in files.items():
            b=z.read('gearshift/'+name)
            if len(b)!=item['bytes'] or hashlib.sha256(b).hexdigest()!=item['sha256']:raise ValueError('Archive verification failed')
    receipt={'path':str(out),'sha256':sha(out),'bytes':out.stat().st_size,'files':len(files),'all_entry_hashes_verified':True,'credential_signature_scan_passed':True,'created_epoch':time.time()};write(ROOT/'gearshift_post_progress01_review.zip.receipt.json',receipt);print(json.dumps(receipt));return receipt

if __name__=='__main__':build()
