#!/usr/bin/env python3
"""Console-only provisioning/staging; detached jobs never depend on this process."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gearshift.coding_coverage_v2_lease import atomic_json, sha256_file, verify_lease

EXPERIMENT = 'coverage_generalization_v2_20260918T233720Z'
EVIDENCE = 'evidence/coding_pilot_v1/' + EXPERIMENT
REMOTE_ROOT = '/workspace/GearshiftV2'
RESULT_ROOT = 'results/coding_pilot_v1/' + EXPERIMENT
IMAGE = 'runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'


def read(path):
    return json.loads(Path(path).read_text())


def cli(a, *args):
    result = subprocess.run([a.cli, *args], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[:2000])
    return json.loads(result.stdout)


def safe_pod(pod):
    # Never retain environment variables or provider-injected credentials.
    fields = ['id','name','desiredStatus','imageName','image','gpuCount','gpuTypeId',
              'costPerHr','adjustedCostPerHr','networkVolumeId','dataCenterId',
              'containerDiskInGb','volumeInGb','volumeMountPath','publicIp','portMappings',
              'ports','createdAt','lastStartedAt','lastStatusChange','machineId']
    return {key: pod[key] for key in fields if key in pod}


def ssh(a, command, *, input=None, timeout=45):
    return subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12',
        '-o','StrictHostKeyChecking=accept-new','-i',str(Path(a.ssh_key).expanduser()),
        '-p',str(a.ssh_port),'root@'+a.ssh_host,command], input=input,
        capture_output=True, timeout=timeout, check=True).stdout


def upload(a, source, destination):
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=12',
        '-o','StrictHostKeyChecking=accept-new','-i',str(Path(a.ssh_key).expanduser()),
        '-P',str(a.ssh_port),str(source),'root@'+a.ssh_host+':'+destination], check=True, timeout=900)


def evidence_for(a):
    name = getattr(a, "allocation_name", "primary")
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Unsafe allocation identity")
    return ROOT / EVIDENCE if name == "primary" else ROOT / EVIDENCE / "allocations" / name


def lease_path_for(a):
    return ROOT / a.lease if getattr(a, "lease", None) else evidence_for(a) / "allocation_lease.json"


def control_for(lease):
    return lease["allowed_result_root"] + ("/" + lease["control_relative"] if lease.get("control_relative") else "")


def allocate(a):
    evidence = evidence_for(a)
    evidence.mkdir(parents=True, exist_ok=True)
    with (evidence/'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _allocate(a)


def _allocate(a):
    evidence = evidence_for(a)
    if (evidence/'allocation_lease.json').exists():
        raise ValueError('Allocation already recorded; inspect/reuse it instead of duplicating')
    attempts = evidence/'allocation_attempts'
    for previous in attempts.glob('*'):
        proof = previous/'reconciled.json'
        if previous.is_dir() and (not proof.exists() or read(proof).get('provider_resources_confirmed_absent') is not True):
            raise ValueError('Previous allocation attempt requires provider reconciliation before retry')
    epoch = time.time()
    count = getattr(a, 'gpu_count', 4)
    if count not in (1, 2, 4): raise ValueError('Unsupported local GPU count')
    d = read(ROOT/'configs/coding_pilot_v1/coverage_generalization_v2/declaration.json')
    start = d['budget_start']
    lease = {'experiment_id':EXPERIMENT,'pod_id':'reservation_pending',
        'allocation_epoch':epoch,'deadline_epoch':epoch+20*3600,
        'upper_hourly_usd':5.55*count,'gpu_count':count,
        'other_reserved_usd':5.55*(4-count)*20,'other_reserved_gpu_hours':(4-count)*20,
        'baseline_usd':start['upper_usd']+max(0,epoch-start['epoch'])/3600*.08+1,
        'failed_staging_volume_contingency_usd':1,
        'baseline_gpu_hours':start['gpu_hours'],'total_cap_usd':1000,
        'total_cap_gpu_hours':500,'cleanup_reserve_usd':40,
        'allowed_result_root':REMOTE_ROOT+'/'+RESULT_ROOT}
    reservation = verify_lease(lease)
    attempts.mkdir(parents=True,exist_ok=True)
    attempt = attempts/str(time.time_ns()); attempt.mkdir()
    atomic_json(attempt/'reservation.json',{'lease':lease,'reservation':reservation,'region':a.region})
    try:
        if getattr(a, 'network_volume_id', None):
            primary = read(ROOT/EVIDENCE/'allocation_lease.json')
            if a.network_volume_id != primary['network_volume_id']: raise ValueError('Helper must attach the verified primary experiment volume')
            volumes=cli(a,'network-volume','list')
            volume=next(v for v in volumes if v['id']==a.network_volume_id)
            if volume['dataCenterId'] != a.region: raise ValueError('Helper region differs from primary volume')
        else:
            volume = cli(a,'network-volume','create','--name','gearshift-coverage-v2-durable',
                         '--size','250','--data-center-id',a.region)
    except BaseException as exc:
        atomic_json(attempt/'failure.json',{'epoch':time.time(),'type':type(exc).__name__,
            'error':str(exc),'stage':'volume_creation',
            'provider_state_requires_inspection':True,'no_resource_id_returned':True})
        raise
    atomic_json(attempt/'volume.json',volume)
    volume_id = volume['id']
    try:
        pod = cli(a,'pod','create','--name','gearshift-coverage-v2-parallel',
            '--image',IMAGE,'--gpu-id','NVIDIA H200','--gpu-count',str(count),
            '--cloud-type','SECURE','--data-center-ids',a.region,
            '--network-volume-id',volume_id,'--volume-mount-path','/workspace',
            '--container-disk-in-gb','40','--ports','22/tcp','--ssh')
        atomic_json(attempt/'pod.json',safe_pod(pod))
        lease.update(pod_id=pod['id'],network_volume_id=volume_id,control_relative='allocations/'+pod['id'])
        atomic_json(evidence/'allocation_lease.json',lease)
        atomic_json(evidence/'resource_receipt.json',{'pod':safe_pod(pod),'volume':volume,
            'reservation':verify_lease(lease),'allocation_attempt':str(attempt.relative_to(ROOT))})
        print(json.dumps({'pod_id':pod['id'],'volume_id':volume_id,'lease_path':str((evidence/'allocation_lease.json').relative_to(ROOT)),**reservation}),flush=True)
    except BaseException as exc:
        atomic_json(attempt/'failure.json',{'epoch':time.time(),'type':type(exc).__name__,'error':str(exc),
            'volume_retained':True,'pod_state_requires_provider_inspection':True})
        # An ambiguous console/provider failure is never grounds to kill work.
        raise


def verify_remote_identity(a, lease):
    command = "import os,json,pathlib; e=dict(x.split(b'=',1) for x in pathlib.Path('/proc/1/environ').read_bytes().split(b'\\0') if b'=' in x); print(json.dumps({'pod_id':os.environ.get('RUNPOD_POD_ID') or e.get(b'RUNPOD_POD_ID',b'').decode()}))"
    identity = json.loads(ssh(a,'python3 -c '+shlex.quote(command)))
    if identity.get('pod_id') != lease['pod_id']:
        raise ValueError('SSH endpoint does not identify the reserved pod')


def assert_guard_armed(a, lease, *, expected_pid=None, wait_seconds=0):
    """Read back immutable lease and a live matching guard, never just a PID print."""
    lease_path = lease_path_for(a)
    remote_lease = REMOTE_ROOT+'/'+str(lease_path.relative_to(ROOT))
    control = control_for(lease)+'/lease_guard'
    command = f'''import hashlib,json,os,pathlib,time
deadline=time.monotonic()+{float(wait_seconds)!r}
while True:
 try:
  status=json.loads(pathlib.Path({control!r},'status.json').read_text())
  assert status['experiment_id']=={lease['experiment_id']!r} and status['pod_id']=={lease['pod_id']!r}
  assert status['state'] in ('armed','unauthorized_release_ignored')
  pid=status['pid']; assert isinstance(pid,int) and pid>1
  expected={expected_pid!r}; assert expected is None or pid==expected
  os.kill(pid,0)
  args=pathlib.Path('/proc',str(pid),'cmdline').read_bytes().split(b'\\0')
  assert {str(REMOTE_ROOT+'/gearshift/coding_coverage_v2_lease.py').encode()!r} in args
  assert {remote_lease.encode()!r} in args
  assert hashlib.sha256(pathlib.Path({remote_lease!r}).read_bytes()).hexdigest()=={sha256_file(lease_path)!r}
  assert not pathlib.Path({control!r},'STOP').exists()
  assert time.time()<{lease['deadline_epoch']!r}-120
  print(json.dumps({{'guard_pid':pid,'pod_id':status['pod_id'],'state':status['state'],'verified_live':True}})); break
 except (AssertionError,FileNotFoundError,ProcessLookupError,KeyError,json.JSONDecodeError):
  if time.monotonic()>=deadline: raise
  time.sleep(.2)
'''
    return json.loads(ssh(a,'python3 -c '+shlex.quote(command),timeout=max(45,wait_seconds+10)))


def validate_archive(path, *, private, lease_sha256, guard_sha256):
    """Refuse traversal, credentials and changes to the already armed guard."""
    seen = set()
    protected = {EVIDENCE+'/allocation_lease.json':lease_sha256,
                 'gearshift/coding_coverage_v2_lease.py':guard_sha256}
    with tarfile.open(path,'r:gz') as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name).as_posix()
            parts = PurePosixPath(name).parts
            if not parts or name.startswith('/') or '..' in parts or '\\' in name or name in seen:
                raise ValueError('Unsafe or duplicate staged archive path')
            seen.add(name)
            if not (member.isfile() or member.isdir()):
                raise ValueError('Staged archives may contain only ordinary files/directories')
            if any(part in {'.ssh','.runpod','.git','.venv','.pilot-venv'} for part in parts) or Path(name).name in {'.env','.gearshift-runpod-key','config.toml'}:
                raise ValueError('Credential or environment path in staged archive')
            if member.isfile():
                is_private = name.startswith('data/coding_pilot_v1/private/')
                if is_private != private:
                    raise ValueError('Private test archive boundary violated')
                if name in protected and hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=protected[name]:
                    raise ValueError('Archive would change immutable lease or armed guard source')
                if name.startswith(EVIDENCE+'/allocations/') and name.endswith('/allocation_lease.json'):
                    if not (ROOT/name).is_file() or hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=sha256_file(ROOT/name):
                        raise ValueError('Archive would change an immutable helper lease')
                if name.startswith(RESULT_ROOT+'/allocations/'):
                    raise ValueError('Archive cannot supply live per-pod allocation control state')
                if name.startswith(RESULT_ROOT+'/lease_guard/') or name==RESULT_ROOT+'/gpu_release_verified.json':
                    raise ValueError('Archive cannot supply live guard state or a release authorization')
    if not seen:
        raise ValueError('Staged archive is empty')


def arm(a):
    lease_path = lease_path_for(a)
    lease = read(lease_path); verify_lease(lease)
    pod = cli(a,'pod','get',lease['pod_id'])
    if pod.get('gpuCount') != lease['gpu_count'] or pod.get('networkVolumeId') != lease['network_volume_id']:
        raise ValueError('Provisioned GPU count/volume differs from reservation')
    prices=[float(pod[key]) for key in ('costPerHr','adjustedCostPerHr') if pod.get(key) is not None]
    price=max(prices) if prices else 0
    if not all(math.isfinite(value) and value>0 for value in prices) or price <= 0 or price*1.2+.04*lease['gpu_count'] > lease['upper_hourly_usd']:
        raise ValueError('Live quote exceeds immutable conservative reservation')
    atomic_json(lease_path.parent/'pod_ready_receipt.json',safe_pod(pod))
    verify_remote_identity(a,lease)
    if (lease_path.parent/'guard_dispatch.json').exists():
        verified=assert_guard_armed(a,lease)
        print(json.dumps(verified)); return
    root = shlex.quote(REMOTE_ROOT)
    ssh(a,'mkdir -p '+shlex.quote(str(Path(REMOTE_ROOT)/lease_path.relative_to(ROOT).parent))+' '+root+'/gearshift '+shlex.quote(control_for(lease)+'/lease_guard'))
    upload(a,ROOT/'gearshift/coding_coverage_v2_lease.py',REMOTE_ROOT+'/gearshift/coding_coverage_v2_lease.py')
    upload(a,lease_path,REMOTE_ROOT+'/'+str(lease_path.relative_to(ROOT)))
    key = tomllib.loads(Path(a.provider_config).expanduser().read_text())['apikey'].strip()
    # Private token travels only on encrypted stdin to an owner-only file.
    ssh(a,'umask 077; cat > /root/.gearshift-runpod-key',input=key.encode())
    lease_remote=REMOTE_ROOT+'/'+str(lease_path.relative_to(ROOT))
    command=['python3',REMOTE_ROOT+'/gearshift/coding_coverage_v2_lease.py','--lease',lease_remote,
             '--lease-sha256',sha256_file(lease_path),'--key-file','/root/.gearshift-runpod-key']
    launcher="import subprocess,os; f=open("+repr(control_for(lease)+'/lease_guard/launcher.log')+",'ab'); p=subprocess.Popen("+repr(command)+",stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True); print(p.pid)"
    pid=ssh(a,'python3 -c '+shlex.quote(launcher)).decode().strip()
    verified=assert_guard_armed(a,lease,expected_pid=int(pid),wait_seconds=10)
    atomic_json(lease_path.parent/'guard_dispatch.json',{'epoch':time.time(),'pod_id':lease['pod_id'],
        'pid':int(pid),'lease_sha256':sha256_file(lease_path),'ssh_host':a.ssh_host,'ssh_port':a.ssh_port,
        'guard_source_sha256':sha256_file(ROOT/'gearshift/coding_coverage_v2_lease.py'),
        'verified_armed':verified})
    print(json.dumps({'guard_pid':int(pid),'pod_id':lease['pod_id']}))


def launch(a):
    lease_path=lease_path_for(a); lease=read(lease_path); verify_lease(lease)
    verify_remote_identity(a,lease)
    assert_guard_armed(a,lease)
    lease_hash=sha256_file(lease_path)
    guard_hash=sha256_file(ROOT/'gearshift/coding_coverage_v2_lease.py')
    recorded_guard=read(lease_path.parent/'guard_dispatch.json')
    if recorded_guard.get('guard_source_sha256')!=guard_hash:
        raise ValueError('Guard source changed after early arming')
    if not getattr(a,'shared_stage',False):
        validate_archive(a.public_archive,private=False,lease_sha256=lease_hash,guard_sha256=guard_hash)
        validate_archive(a.private_archive,private=True,lease_sha256=lease_hash,guard_sha256=guard_hash)
        for source,name in [(a.public_archive,'inputs.tar.gz'),(a.private_archive,'private_tests.tar.gz')]:
            upload(a,source,'/workspace/'+name)
        command='mkdir -p '+shlex.quote(REMOTE_ROOT)+' && tar -xzf /workspace/inputs.tar.gz -C '+shlex.quote(REMOTE_ROOT)+' && tar -xzf /workspace/private_tests.tar.gz -C '+shlex.quote(REMOTE_ROOT)+' && rm /workspace/private_tests.tar.gz'
        ssh(a,command,timeout=180)
    assert_guard_armed(a,lease)
    plan=a.plan; lease=str(lease_path.relative_to(ROOT))
    upload(a,ROOT/plan,REMOTE_ROOT+'/'+plan)
    command=['python3',REMOTE_ROOT+'/scripts/coding_coverage_v2_bootstrap.py','--plan',plan,'--lease',lease]
    log=control_for(read(lease_path))+'/bootstrap_launcher.log'
    launcher="import subprocess,os; f=open("+repr(log)+",'ab'); p=subprocess.Popen("+repr(command)+",cwd="+repr(REMOTE_ROOT)+",stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True); print(p.pid)"
    pid=ssh(a,'python3 -c '+shlex.quote(launcher)).decode().strip()
    atomic_json(lease_path.parent/'bootstrap_dispatch.json',{'epoch':time.time(),'pid':int(pid),
        'public_archive_sha256':sha256_file(a.public_archive) if a.public_archive else None,
        'private_archive_sha256':sha256_file(a.private_archive) if a.private_archive else None,
        'ssh_host':a.ssh_host,'ssh_port':a.ssh_port,'console_may_disconnect':True})
    print(json.dumps({'bootstrap_pid':int(pid),'console_may_disconnect':True}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['allocate','arm','launch']);p.add_argument('--region',default='US-GA-2')
    p.add_argument('--cli',default=str(Path.home()/'.local/bin/runpodctl-2.14.0'))
    p.add_argument('--ssh-host');p.add_argument('--ssh-port',type=int)
    p.add_argument('--ssh-key',default='~/.ssh/id_ed25519');p.add_argument('--provider-config',default='~/.runpod/config.toml')
    p.add_argument('--public-archive');p.add_argument('--private-archive')
    p.add_argument('--gpu-count',type=int,default=2);p.add_argument('--network-volume-id')
    p.add_argument('--allocation-name',default='primary');p.add_argument('--lease');p.add_argument('--plan')
    p.add_argument('--shared-stage',action='store_true')
    a=p.parse_args()
    if a.action!='allocate' and (not a.ssh_host or not a.ssh_port):p.error('Live verified SSH host/port required')
    if a.action=='launch' and not a.plan:p.error('Frozen per-pod dispatch plan required')
    if a.action=='launch' and not a.shared_stage and (not a.public_archive or not a.private_archive):p.error('Both staged archives required')
    {'allocate':allocate,'arm':arm,'launch':launch}[a.action](a)


if __name__=='__main__':main()
