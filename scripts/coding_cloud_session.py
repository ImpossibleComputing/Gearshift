#!/usr/bin/env python3
"""Studio-owned bounded preflight allocation, transfer, backup and cleanup."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys,tarfile,threading,time,tomllib,urllib.request
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import PREFIX,write,sha,decision
from coding_cloud_guard import cli
from gearshift.coding_snapshot import verify_archive
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'evidence/coding_pilot_v1/control';REMOTE='/workspace/Gearshift'

def quote():
    auth=tomllib.loads((Path.home()/'.runpod/config.toml').read_text())
    query='{ gpuTypes(input: {id: "NVIDIA H200"}) { id securePrice memoryInGb lowestPrice(input: {gpuCount: 1, secureCloud: true}) { uninterruptablePrice stockStatus } } }'
    req=urllib.request.Request('https://api.runpod.io/graphql',data=json.dumps({'query':query}).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+auth['apikey'],'User-Agent':'gearshift-pilot'})
    obj=json.loads(urllib.request.urlopen(req,timeout=30).read())['data']['gpuTypes'][0]
    if not 0<float(obj['securePrice'])<=5.5:raise ValueError('H200 quote outside approved ceiling')
    return obj

def image_digest():
    tag='1.0.2-cu1281-torch280-ubuntu2404';repo='runpod/pytorch'
    token=json.loads(urllib.request.urlopen('https://auth.docker.io/token?service=registry.docker.io&scope=repository:'+repo+':pull',timeout=30).read())['token']
    req=urllib.request.Request('https://registry-1.docker.io/v2/'+repo+'/manifests/'+tag,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.index.v1+json'})
    with urllib.request.urlopen(req,timeout=30) as response:
        data=response.read();digest=response.headers.get('Docker-Content-Digest') or 'sha256:'+hashlib.sha256(data).hexdigest()
    return repo+'@'+digest

def ssh_args(conn):
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
        '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(C/'known_hosts')]
    key=conn.get('ssh_key',{}).get('path')
    if key:args+=['-i',key]
    return args+['-p',str(conn['port']),'root@'+conn['ip']]

def remote(conn,command,timeout=120):
    p=subprocess.run(ssh_args(conn)+[command],capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError('Remote command failed: '+p.stderr[-1500:]+p.stdout[-1500:])
    return p.stdout

def collect(conn):
    dest=ROOT/'evidence/coding_pilot_v1/pod_backup';dest.mkdir(exist_ok=True)
    # One compact tar snapshot (no models, paired tensors, credentials or private tests).
    command="cd "+REMOTE+" && python3 scripts/coding_snapshot.py"
    temp=dest/'latest.tar.gz.tmp'
    with temp.open('wb') as f:
        p=subprocess.run(ssh_args(conn)+[command],stdout=f,stderr=subprocess.PIPE,timeout=120)
        if p.returncode:raise RuntimeError('Backup fetch failed '+p.stderr.decode()[-500:])
        f.flush();os.fsync(f.fileno())
    stage=Path(tempfile.mkdtemp(prefix='verified-',dir=dest))
    try:manifest=verify_archive(temp,stage)
    except BaseException:shutil.rmtree(stage);raise
    unpack=dest/'latest';previous=dest/'previous'
    if previous.exists():shutil.rmtree(previous)
    if unpack.exists():os.replace(unpack,previous)
    os.replace(stage,unpack)
    os.replace(temp,dest/'latest.tar.gz')
    write(C/'backup_ack.json',{'epoch':time.time(),'archive_sha256':sha(dest/'latest.tar.gz'),'archive_bytes':(dest/'latest.tar.gz').stat().st_size,
        'verified_files':len(manifest['files']),'snapshot_epoch':manifest['epoch']})
    return unpack

def main():
    os.chdir(ROOT);C.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--retry-unallocated',action='store_true');parser.add_argument('--retry-setup',action='store_true');parser.add_argument('--retry-protocol',action='store_true');parser.add_argument('--region',default='US-GA-2');args=parser.parse_args()
    previous=None;attempt=1
    if (C/'allocation_intent.json').exists():
        if not (args.retry_unallocated or args.retry_setup or args.retry_protocol):raise RuntimeError('Allocation intent already exists; inspect and recover rather than duplicate spending')
        if args.retry_protocol:
            status=json.loads((C/'latest_worker_status.json').read_text())
            if status.get('error')!='source native control failed at 128':raise RuntimeError('Recovery is restricted to the recorded pre-outcome control failure')
            prior=ROOT/'evidence/coding_pilot_v1/pod_backup/latest/results/coding_pilot_v1'
            if list(prior.glob('*/tasks/*/*.json')):raise RuntimeError('Protocol diagnosis must precede any benchmark output')
            if not (C/'backup_verified.json').exists():raise RuntimeError('Original failure evidence must be backed up')
        elif args.retry_setup:
            status=json.loads((C/'latest_worker_status.json').read_text())
            if status['state'] not in ['bootstrap_failed','bootstrapping']:raise RuntimeError('Setup recovery cannot retry scientific outcomes')
            if (ROOT/'evidence/coding_pilot_v1/pod_backup/latest/results/coding_pilot_v1/preflight_v1/identity.json').exists():raise RuntimeError('Scientific identity already exists; setup-only recovery is forbidden')
            if not (C/'backup_verified.json').exists():raise RuntimeError('Setup evidence must be verified before recovery')
        elif (C/'pod.json').exists():raise RuntimeError('This recovery is only for capacity failures before any GPU allocation')
        if any(r.get('name','').startswith(PREFIX) for kind in ['pod','network-volume'] for r in cli(kind,'list')):raise RuntimeError('Previous resources are not confirmed absent')
        previous=json.loads((C/'allocation_intent.json').read_text());attempt=previous.get('attempt',1)+1
        if attempt>(6 if args.retry_setup or args.retry_protocol else 3):raise RuntimeError('Bounded infrastructure attempts exhausted')
        archive=C/f'capacity_attempt_{attempt-1}';archive.mkdir(exist_ok=True)
        for name in ['allocation_intent.json','volume.json','pod.json','session_error.json','session_status.json','backup_verified.json','latest_worker_status.json','upload_manifest.json','ALERT.json']:
            if (C/name).exists():shutil.copy2(C/name,archive/name)
        if args.retry_setup or args.retry_protocol:
            prior=ROOT/'evidence/coding_pilot_v1/pod_backup';saved=ROOT/f'evidence/coding_pilot_v1/attempt_{attempt-1:02d}'
            shutil.copytree(prior/'latest/evidence/coding_pilot_v1',saved,dirs_exist_ok=True)
            shutil.copy2(prior/'final.tar.gz',prior/f'attempt_{attempt-1:02d}.tar.gz')
            if args.retry_protocol:
                for path in (prior/'latest/results/coding_pilot_v1').iterdir():
                    target=ROOT/'results/coding_pilot_v1'/path.name
                    if target.exists():raise RuntimeError('Original results already present; verify manually instead of overwriting')
                    shutil.copytree(path,target)
            for name in ['pod.json','backup_verified.json','latest_worker_status.json','session_error.json']:(C/name).unlink(missing_ok=True)
        (C/'STOP').unlink(missing_ok=True)
        (C/'ALERT.json').unlink(missing_ok=True)
    if not json.loads((ROOT/'evidence/coding_pilot_v1/data_gate.json').read_text())['passed']:raise RuntimeError('Data gate required')
    if not json.loads((ROOT/'evidence/coding_pilot_v1/local_linux_sandbox_gate.json').read_text())['passed']:raise RuntimeError('Local Linux sandbox probe required')
    auth=json.loads((C/'authorization.json').read_text());assert auth['approved_cap_usd']==300
    if (C/'ledger.json').exists() and decision(json.loads((C/'ledger.json').read_text()))['stop']:
        raise RuntimeError('An approved ceiling has been reached; no new paid allocation is permitted')
    actual_quote=quote();image=image_digest()
    # The independent watchdog must already be installed and alive before mutation.
    probe=subprocess.run(['launchctl','print',f'gui/{os.getuid()}/com.gearshift.coding-pilot.watchdog'],capture_output=True,text=True)
    if probe.returncode or 'state = running' not in probe.stdout:raise RuntimeError('Independent watchdog is not running')
    started=previous['started_epoch'] if previous else time.time()
    intent={'started_epoch':started,'attempt_started_epoch':time.time(),'attempt':attempt,'name_prefix':PREFIX,'quote':actual_quote,'image_digest':image,'region':args.region}
    write(C/'allocation_intent.json',intent)
    write(C/'session_status.json',{'state':'allocating','attempt':attempt,'epoch':time.time()})
    running=True
    def heartbeat():
        while running:
            write(C/'controller_heartbeat.json',{'epoch':time.time(),'pid':os.getpid()});time.sleep(10)
    threading.Thread(target=heartbeat,daemon=True).start()
    conn=None;pod=None;vol=None
    try:
        vol=cli('network-volume','create','--name',PREFIX+f'storage-{attempt:02d}','--size','250','--data-center-id',args.region)
        write(C/'volume.json',{k:vol.get(k) for k in ['id','name','size','dataCenterId']})
        write(C/'resource_receipts'/f"{vol['id']}.json",{'kind':'volume','id':vol['id'],'name':vol['name'],'started_epoch':intent['attempt_started_epoch'],'upper_rate_usd':.04})
        pod=cli('pod','create','--name',PREFIX+f'preflight-{attempt:02d}','--image',image,'--gpu-id','NVIDIA H200','--gpu-count','1',
            '--cloud-type','SECURE','--data-center-ids',args.region,'--container-disk-in-gb','40','--network-volume-id',vol['id'],'--ports','22/tcp','--ssh')
        write(C/'pod.json',{k:pod.get(k) for k in ['id','name','imageName','costPerHr','adjustedCostPerHr','machineId','gpuCount','networkVolumeId']})
        write(C/'resource_receipts'/f"{pod['id']}.json",{'kind':'pod','id':pod['id'],'name':pod['name'],'started_epoch':intent['attempt_started_epoch'],'upper_rate_usd':5.52})
        allocation={'pod_id':pod['id'],'volume_id':vol['id'],'started_epoch':started,'preflight_deadline_epoch':started+4*3600,
            'image_digest':image,'quote':actual_quote,'cap_usd':300,'preflight_cap_usd':25,'preflight_hours':4}
        write(ROOT/'evidence/coding_pilot_v1/allocation.json',allocation)
        for _ in range(60):
            if (C/'STOP').exists():raise RuntimeError('Watchdog stop')
            try:
                info=cli('ssh','info',pod['id']);candidate=info.get('connection',info)
                if candidate and candidate.get('ip'):
                    remote(candidate,'true',30);conn=candidate;break
            except Exception:pass
            time.sleep(10)
        if conn is None:raise TimeoutError('Pod startup exceeded ten minutes')
        write(C/'connection.json',conn)
        files=[p for pat in ['gearshift/*.py','scripts/coding_*.py','configs/coding_pilot_v1/**/*.json','data/coding_pilot_v1/visible/*.json'] for p in ROOT.glob(pat)]
        files += [ROOT/'data/coding_pilot_v1/private/development.json',ROOT/'data/coding_pilot_v1/identity.json',ROOT/'evidence/coding_pilot_v1/allocation.json']
        package=C/'upload.tar'
        with tarfile.open(package,'w') as tar:
            for p in files:tar.add(p,arcname=p.relative_to(ROOT),recursive=False)
        write(C/'upload_manifest.json',{'epoch':time.time(),'archive_sha256':sha(package),
            'files':{str(p.relative_to(ROOT)):sha(p) for p in files}})
        with package.open('rb') as f:
            p=subprocess.run(ssh_args(conn)+['mkdir -p '+REMOTE+' && tar --no-same-owner --no-same-permissions -xf - -C '+REMOTE],stdin=f,capture_output=True,timeout=300)
            if p.returncode:raise RuntimeError('Upload failed')
        launcher="import subprocess,pathlib; p=pathlib.Path("+repr(REMOTE)+"); (p/'evidence/coding_pilot_v1').mkdir(parents=True,exist_ok=True); (p/'results/coding_pilot_v1').mkdir(parents=True,exist_ok=True); f=(p/'evidence/coding_pilot_v1/bootstrap.log').open('ab'); child=subprocess.Popen(['python3','scripts/coding_bootstrap.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(child.pid)"
        pid=remote(conn,'python3 -c '+shlex.quote(launcher),30)
        write(C/'bootstrap_process.json',{'pid':int(pid.strip()),'pod_id':pod['id'],'epoch':time.time()})
        write(C/'session_status.json',{'state':'preflight_running','pod_id':pod['id'],'epoch':time.time()})
        while time.time()<started+4*3600-90 and not (C/'STOP').exists():
            saved=collect(conn);status_path=saved/'evidence/coding_pilot_v1/worker_status.json'
            status=json.loads(status_path.read_text()) if status_path.exists() else {'state':'bootstrapping'}
            write(C/'latest_worker_status.json',status)
            progress=saved/'results/coding_pilot_v1/preflight_v2/progress.json'
            if progress.exists():write(C/'latest_progress.json',json.loads(progress.read_text()))
            if status['state'] in ['stopped','preflight_baselines_finished','bootstrap_failed']:break
            time.sleep(30)
        if conn:
            remote(conn,'touch '+REMOTE+'/evidence/coding_pilot_v1/STOP')
            # Bound the graceful drain; a token callback observes STOP quickly.
            time.sleep(5);saved=collect(conn)
            # A second independent local archive and extracted-file verification.
            archive=ROOT/'evidence/coding_pilot_v1/pod_backup/latest.tar.gz'
            second=ROOT/'evidence/coding_pilot_v1/pod_backup/final.tar.gz';shutil.copy2(archive,second)
            assert sha(archive)==sha(second)
            write(C/'backup_verified.json',{'epoch':time.time(),'pod_id':pod['id'],'volume_id':vol['id'],
                'copies':[str(archive.relative_to(ROOT)),str(second.relative_to(ROOT))],'sha256':sha(archive)})
    except BaseException as exc:
        write(C/'session_error.json',{'epoch':time.time(),'error':str(exc)})
        if conn:
            try:collect(conn)
            except Exception:pass
        raise
    finally:
        (C/'STOP').touch()
        if pod:
            try:cli('pod','delete',pod['id'])
            except Exception as exc:write(C/'cleanup_pending.json',{'error':str(exc)})
        if vol and ((C/'backup_verified.json').exists() or conn is None):
            try:cli('network-volume','delete',vol['id'])
            except Exception as exc:write(C/'volume_cleanup_pending.json',{'error':str(exc)})
        running=False
        write(C/'session_status.json',{'state':'stopped_backed_up' if (C/'backup_verified.json').exists() else 'stopped_inspect_backup','epoch':time.time()})

if __name__=='__main__':main()
