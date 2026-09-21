"""Verified snapshots of atomic records while their directories keep changing."""
import contextlib,fcntl,hashlib,json,os,shutil,tarfile,tempfile,time
from pathlib import Path
from gearshift.coding_control import write

EXCLUDED_SUFFIXES={'.tmp','.lock','.log','.pt','.bin','.safetensors'}
GLOBAL_RECEIPTS=('download_progress.json','sandbox_gate.json','source_weight_pins.json','receiver_weight_pins.json',
    'weight_verification.json','runtime_lock.json','worker_status.json','ALERT.json','STOP')


def worker_scope(root,spec_path):
    """Current outputs only; immutable uploaded parents are already on Studio."""
    root=Path(root).resolve();spec_path=Path(spec_path)
    if not spec_path.is_absolute():spec_path=root/spec_path
    if spec_path.is_symlink() or not spec_path.resolve().is_relative_to(root):raise ValueError('Unsafe worker snapshot specification')
    data=spec_path.read_bytes();spec=json.loads(data)
    run=spec['run_id'];worker=spec['worker_id']
    from gearshift.coding_parallel import safe_id
    safe_id(run);safe_id(worker)
    result=f'results/coding_pilot_v1/{run}/{worker}'
    evidence=f'evidence/coding_pilot_v1/parallel/{run}/{worker}'
    if spec['result_root']!=result or spec['worker_root']!=evidence:raise ValueError('Worker snapshot roots differ from dispatch identity')
    if spec_path.resolve()!=root/evidence/'worker_spec.json':raise ValueError('Worker spec outside its own evidence directory')
    roots=[root/result,root/evidence]
    if spec['stage']=='memory':
        roots += [root/'results/coding_pilot_v1'/name for name in
            ['memory_v2_both_gpu','memory_v2_source_cpu_diagnostic']]
    files=[root/'evidence/coding_pilot_v1'/name for name in GLOBAL_RECEIPTS]
    logs=[root/evidence/'bootstrap.log',root/'evidence/coding_pilot_v1/bootstrap.log']
    scope={'kind':'worker','run_id':run,'worker_id':worker,'stage_identity':spec['stage_identity'],
        'result_root':result,'worker_root':evidence,'worker_spec_sha256':hashlib.sha256(data).hexdigest(),
        'immutable_uploaded_parents':'Excluded from repeated transfer; retained on Studio and bound by the verified upload manifest.'}
    return roots,files,logs,scope


def scoped_paths(roots,files):
    """Prune before traversal; never enumerate unrelated parent result trees."""
    paths=set()
    for base in roots:
        if not base.exists():continue
        if base.is_symlink():raise ValueError('Snapshot root is a symlink')
        for directory,dirs,names in os.walk(base,followlinks=False):
            dirs[:]=[name for name in dirs if name not in ['control','__pycache__','.cache']
                and not (Path(directory)/name).is_symlink()]
            for name in names:
                p=Path(directory)/name
                if p.suffix not in EXCLUDED_SUFFIXES and not p.is_symlink() and p.is_file():paths.add(p)
    for p in files:
        if p.exists() and not p.is_symlink() and p.is_file():paths.add(p)
    return sorted(paths)


@contextlib.contextmanager
def capture_lock(root):
    """A timed-out SSH client must not create a second competing snapshot."""
    key=hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:20]
    path=Path('/tmp')/('gearshift-snapshot-'+key+'.lock')
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
    with os.fdopen(fd,'w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another snapshot is still active; no overlapping capture')
        yield


def capture(root,stage,*,worker_spec=None):
    started=time.monotonic();root=Path(root).resolve();stage=Path(stage);manifest={}
    if worker_spec is None:
        roots=[root/name for name in ['evidence/coding_pilot_v1','results/coding_pilot_v1']]
        files=[];logs=[root/'evidence/coding_pilot_v1/bootstrap.log'];scope={'kind':'full'}
    else:roots,files,logs,scope=worker_scope(root,worker_spec)
    # Materialize names before reading. A completed-task marker is published last;
    # any captured marker must match every committed file it references.
    paths=scoped_paths(roots,files);enumerated=time.monotonic()
    for p in paths:
        try:data=p.read_bytes()
        except FileNotFoundError:continue # An uncommitted transient disappeared.
        if p.suffix=='.json':json.loads(data)
        rel=str(p.relative_to(root));dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        manifest[rel]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    for rel in list(manifest):
        if rel.endswith('/complete.json'):
            record=json.loads((stage/rel).read_text())
            for name,want in record.get('files',{}).items():
                key=str(Path(rel).parent/name)
                if key not in manifest or manifest[key]['sha256']!=want:raise ValueError('Task transaction not consistent: '+rel)
    for log in logs:
        if log.exists() and not log.is_symlink():
            with log.open('rb') as f:
                f.seek(max(0,os.fstat(f.fileno()).st_size-100000));data=f.read(100000)
            rel=str(log.with_name('bootstrap_tail.txt').relative_to(root));p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            manifest[rel]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    write(stage/'SNAPSHOT_MANIFEST.json',{'schema':1,'epoch':time.time(),'files':manifest,
        'scope':scope,'capture_seconds':time.monotonic()-started,'enumeration_seconds':enumerated-started,
        'uncompressed_bytes':sum(record['bytes'] for record in manifest.values()),
        'consistency':'Each JSON is an atomic committed version. Complete task receipts are cross-checked; in-flight progress is explicitly partial.'})
    return manifest

def stream(root,output,*,worker_spec=None):
    with capture_lock(root), tempfile.TemporaryDirectory(prefix='gearshift-snapshot-',dir='/tmp') as tmp:
        stage=Path(tmp)/'snapshot';stage.mkdir()
        # A race can delay one snapshot; it never turns a healthy worker into a
        # failure. Snapshot staging is immutable before archive streaming starts.
        for attempt in range(3):
            try:capture(root,stage,worker_spec=worker_spec);break
            except (ValueError,FileNotFoundError):
                if attempt==2:raise
                shutil.rmtree(stage);stage.mkdir();time.sleep(.1)
        with tarfile.open(fileobj=output,mode='w|gz') as tar:
            for p in sorted(stage.rglob('*')):
                if p.is_file():tar.add(p,arcname=p.relative_to(stage),recursive=False)

def verify_archive(path,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(path,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive entry')
        if any(not m.isfile() or m.name.startswith('/') or '..' in Path(m.name).parts for m in members):raise ValueError('Unsafe snapshot entry')
        manifest=json.load(tar.extractfile('SNAPSHOT_MANIFEST.json'))
        if set(names)!=set(manifest['files'])|{'SNAPSHOT_MANIFEST.json'}:raise ValueError('Unexpected snapshot membership')
        for m in members:
            data=tar.extractfile(m).read()
            if m.name!='SNAPSHOT_MANIFEST.json':
                want=manifest['files'][m.name]
                if len(data)!=want['bytes'] or hashlib.sha256(data).hexdigest()!=want['sha256']:raise ValueError('Snapshot hash mismatch')
            target=destination/m.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    return manifest
