#!/usr/bin/env python3
"""Rolling compact review overlay; no weights, tests secrets, caches or credentials."""
import hashlib,json,os,sys,tempfile,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import sha,write,digest
from gearshift.coding_snapshot import verify_archive,GLOBAL_RECEIPTS
ROOT=Path(__file__).resolve().parents[1]
DENY_NAMES={'connection.json','pod.json','volume.json','upload.tar','latest.tar.gz','final.tar.gz'}
DENY_SUFFIX={'.pt','.bin','.safetensors','.tar','.gz','.zip','.lock','.tmp','.pyc'}

def collect_runtime_receipts():
    """Keep each completed worker's global runtime proofs under its own identity."""
    for archive in (ROOT/'evidence/coding_pilot_v1/parallel_backups').glob('recovery_*20260917*/*/final.tar.gz'):
        run,worker=archive.parent.parent.name,archive.parent.name
        control=ROOT/'evidence/coding_pilot_v1/control/parallel'/run
        proof_path=control/worker/'backup_verified.json'
        if not proof_path.exists():continue
        proof=json.loads(proof_path.read_text());plan=json.loads((control/'plan.json').read_text())
        if not proof['verified'] or not proof['worker_confirmed_absent'] or sha(archive)!=proof['sha256']:raise ValueError('Runtime receipt archive not verified')
        target=ROOT/'evidence/coding_pilot_v1/recovery_20260917/runtime_receipts'/run/worker
        with tempfile.TemporaryDirectory(prefix='runtime-review-') as tmp:
            manifest=verify_archive(archive,tmp)
            if manifest['scope']['stage_identity']!=digest(plan):raise ValueError('Runtime receipt scope differs')
            records={}
            for name in GLOBAL_RECEIPTS:
                rel='evidence/coding_pilot_v1/'+name
                if rel not in manifest['files'] or not name.endswith('.json'):continue
                source=Path(tmp)/rel;dest=target/name;data=source.read_bytes()
                if dest.exists() and dest.read_bytes()!=data:raise ValueError('Immutable runtime receipt changed')
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                records[name]={'source_path':rel,'sha256':sha(dest),'bytes':len(data)}
            write(target/'SOURCE.json',{'run_id':run,'worker_id':worker,'stage_identity':digest(plan),'archive_sha256':proof['sha256'],'files':records})


def build():
    collect_runtime_receipts()
    prior=ROOT/'gearshift_coding_review_20260917.zip';out=ROOT/'gearshift_recovery_review.zip';tmp=out.with_suffix('.zip.tmp');files={};overlay={};heavy={}
    for pattern in ['publication/**/*','gearshift/*.py','scripts/*.py','tests/*.py','configs/coding_pilot_v1/recovery_20260917/**/*',
        'data/coding_pilot_v1/visible/recovery_*20260917*.json','evidence/coding_pilot_v1/parallel/recovery_*20260917*/**/*','evidence/coding_pilot_v1/control/parallel/recovery_*20260917*/**/*.json',
        'results/coding_pilot_v1/partial_corpus_20260917_v1/**/*','results/coding_pilot_v1/recovery_*20260917*/**/*','evidence/coding_pilot_v1/recovery_20260917/**/*']:
        for p in ROOT.glob(pattern):
            if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
            rel=str(p.relative_to(ROOT))
            if p.suffix=='.pt':
                if p.name.startswith('mapper'):
                    heavy[rel]={'bytes':p.stat().st_size,'sha256':sha(p),'regenerate':'Pinned partial corpus and declared training schedule in separate namespace; exact trained bytes retained on Studio.'}
                continue
            if p.name in DENY_NAMES or p.suffix in DENY_SUFFIX or '/cache_lru/' in rel:continue
            overlay['gearshift/'+rel]=p
    for name in ['configs/coding_pilot_v1/pilot.json','RECOVERY_RESULTS.md','REPRODUCE_RECOVERY.md','data/coding_pilot_v1/visible/recovery_probe_20260917.json']:
        p=ROOT/name
        if p.exists():overlay['gearshift/'+name]=p
    # Snapshot mutable control status into a dated review receipt, not raw provider credentials.
    status=json.loads((ROOT/'evidence/coding_pilot_v1/control/watchdog_status.json').read_text())
    review=ROOT/'evidence/coding_pilot_v1/recovery_20260917/review_cost_snapshot.json';write(review,status);overlay['gearshift/'+str(review.relative_to(ROOT))]=review
    manifest=json.loads((ROOT/'results/coding_pilot_v1/partial_corpus_20260917_v1/corpus_manifest.json').read_text())
    for rel,info in manifest['files'].items():
        if rel.endswith('.pt'):heavy[rel]={**info,'regenerate':'See original history transaction and pinned source/receiver; regenerate full-prefix paired samples at recorded64positions in a fresh namespace.'}
    for backup in (ROOT/'evidence/coding_pilot_v1/parallel_backups').glob('recovery_train_20260917_*/*/mapper_checkpoints'):
        for checkpoint in backup.rglob('mapper*.pt'):
            logical=str(checkpoint.relative_to(backup));h=sha(checkpoint)
            if logical in heavy and heavy[logical]['sha256']!=h:raise ValueError('Backup checkpoint differs from canonical checkpoint')
            entry=heavy.setdefault(logical,{'bytes':checkpoint.stat().st_size,'sha256':h,'regenerate':'Pinned partial corpus and declared training schedule in a separate namespace; exact bytes retained on Studio.'})
            entry.setdefault('verified_backup_locations',[]).append(str(checkpoint.relative_to(ROOT)))
    for inventory in (ROOT/'results/coding_pilot_v1').glob('recovery_train_20260917_*/*/cache_inventory.json'):
        heavy.update(json.loads(inventory.read_text())['files'])
    model_inventory={}
    cfg=json.loads((ROOT/'configs/coding_pilot_v1/pilot.json').read_text())
    for role,spec in cfg['models'].items():
        candidates=list((ROOT/'evidence/coding_pilot_v1/recovery_20260917').rglob(role+'_weight_pins.json'))+[ROOT/f'evidence/coding_pilot_v1/{role}_weight_pins.json']
        pin=next((p for p in candidates if p.exists()),None)
        if pin is None:continue
        pins=json.loads(pin.read_text())
        if pins['revision']!=spec['revision']:raise ValueError('Weight inventory revision differs')
        for filename,h in pins['sha256'].items():
            model_inventory[spec['id']+'/'+spec['revision']+'/'+filename]={'sha256':h,'location':'Runpod /workspace/hf/hub/models--'+spec['id'].replace('/','--')+'/snapshots/'+spec['revision']+'/'+filename,
                'regenerate':'scripts/coding_download_models.py with pinned config; ephemeral worker copies removed during cleanup, original retained volume may hold duplicate weights.'}
    reproduction_inputs={}
    for rel in ['configs/coding_pilot_v1/reference/Qwen3-32B/tokenizer.json','configs/coding_pilot_v1/reference/Qwen3-8B/tokenizer.json','data/coding_pilot_v1/private/development.json']:
        p=ROOT/rel
        reproduction_inputs[rel]={'bytes':p.stat().st_size,'sha256':sha(p),'location':str(p),
            'regenerate':'Fetch tokenizer.json from the exact corresponding model revision in configs/coding_pilot_v1/pilot.json and verify this hash.' if 'tokenizer' in rel else 'Use scripts/coding_prepare.py with pinned membership/source files in a fresh trusted preparation directory; verify data/coding_pilot_v1/identity.json. Hidden tests stay outside model inputs.'}
    heavyfile=ROOT/'evidence/coding_pilot_v1/recovery_20260917/excluded_heavy.json';write(heavyfile,{'files':heavy,'non_weight_reproduction_inputs':reproduction_inputs,'model_weight_inventory':model_inventory,'model_weights':'Pinned model revisions and per-shard weight-verification receipts; public downloads via scripts/coding_download_models.py; never included.',
        'environments':'Recreate from lockfile and pinned bootstrap; not included.','cache_tensors':'Recompute complete saved prefixes in512-token chunks; cache LRU is disposable.','credentials':'Never included, hashed or enumerated.'})
    overlay['gearshift/'+str(heavyfile.relative_to(ROOT))]=heavyfile
    summary_path=ROOT/'results/coding_pilot_v1/recovery_review_20260917/development_summary.json'
    coverage=json.loads(summary_path.read_text())['coverage'] if summary_path.exists() else {'committed_complete':0,'expected':40}
    with zipfile.ZipFile(prior) as old,zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        old_manifest=json.loads(old.read('gearshift/REVIEW_BUNDLE_MANIFEST.json'))
        def put(name,data):
            if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe archive member')
            z.writestr(name,data);files[name.removeprefix('gearshift/')]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        for name in old.namelist():
            if name=='gearshift/REVIEW_BUNDLE_MANIFEST.json' or name in overlay:continue
            put(name,old.read(name))
        for name,p in sorted(overlay.items()):put(name,p.read_bytes())
        new={'schema':2,'created_epoch':time.time(),'files':files,'uncompressed_bytes':sum(r['bytes'] for r in files.values()),
             'historical_bundle_sha256':sha(prior),'source_trajectories_included':True,'publishing_authorized':False,
             'omissions':old_manifest['omissions'],'excluded_heavy_manifest':'evidence/coding_pilot_v1/recovery_20260917/excluded_heavy.json',
             'excluded_categories':old_manifest['excluded_categories'],'article':'publication/GEARSHIFT_PROGRESS_01.md','start_here':'RECOVERY_RESULTS.md',
             'rolling_snapshot':coverage['committed_complete']!=40,'development_coverage':coverage,'manifest_self_hash':'Not recursive; archive hash in external receipt.'}
        z.writestr('gearshift/REVIEW_BUNDLE_MANIFEST.json',json.dumps(new,indent=2)+'\n')
    os.replace(tmp,out)
    with zipfile.ZipFile(out) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate ZIP member')
        for name,info in files.items():
            data=z.read('gearshift/'+name)
            if len(data)!=info['bytes'] or hashlib.sha256(data).hexdigest()!=info['sha256']:raise ValueError('ZIP verification mismatch')
    receipt={'epoch':time.time(),'path':str(out),'sha256':sha(out),'bytes':out.stat().st_size,'files':len(files),'all_entry_hashes_verified':True}
    write(ROOT/'gearshift_recovery_review.zip.receipt.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':build()
