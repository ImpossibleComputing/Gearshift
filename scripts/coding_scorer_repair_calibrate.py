#!/usr/bin/env python3
"""Public-only resource calibration; no private tests or saved candidates loaded."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import sysconfig
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import digest,sha,write
from gearshift.coding_sandbox_v2 import DEFAULT_POLICY,score,scorer_identity

SOURCES={
 'abc344_e':{'problem':'https://atcoder.jp/contests/abc344/tasks/abc344_e','editorial':'https://atcoder.jp/contests/abc344/editorial/9503','constraints':'N,Q <= 200000; distinct positive values <= 10^9. At most 400000 final integers; conservative canonical bound 4400000 bytes. Valid whitespace can exceed 4 MiB. Linked-list updates are linear overall.'},
 'abc335_c':{'problem':'https://atcoder.jp/contests/abc335/tasks/abc335_c','editorial':'https://atcoder.jp/contests/abc335/editorial/9293','constraints':'N <= 10^6; Q <= 200000. Reverse position history supports constant-time moves/queries; no per-move full-body shift.'},
 'abc339_d':{'problem':'https://atcoder.jp/contests/abc339/tasks/abc339_d','editorial':'https://atcoder.jp/contests/abc339/editorial/9273','constraints':'2 <= N <= 60. Breadth-first search over paired positions has O(N^4) states.'}}
REFERENCES={
'abc344_e':'''import sys
it=iter(map(int,sys.stdin.buffer.read().split()));n=next(it);a=[next(it) for _ in range(n)]
prev={};nxt={};chain=[0]+a+[-1]
for x,y in zip(chain,chain[1:]):nxt[x]=y;prev[y]=x
for _ in range(next(it)):
 t=next(it);x=next(it)
 if t==1:
  y=next(it);z=nxt[x];nxt[x]=y;prev[y]=x;nxt[y]=z;prev[z]=y
 else:
  u=prev[x];v=nxt[x];nxt[u]=v;prev[v]=u
out=[];x=nxt[0]
while x!=-1:out.append(str(x));x=nxt[x]
print('  '.join(out))
''',
'abc335_c':'''import sys
f=sys.stdin.buffer;n,q=map(int,f.readline().split());history=[(1,0)];out=[]
for _ in range(q):
 t,v=f.readline().split()
 if t==b'1':
  x,y=history[-1];dx,dy={b'R':(1,0),b'L':(-1,0),b'U':(0,1),b'D':(0,-1)}[v];history.append((x+dx,y+dy))
 else:
  p=int(v);i=len(history)-p;x,y=history[i] if i>=0 else (p-len(history)+1,0);out.append(f'{x} {y}')
print('\\n'.join(out))
''',
'abc339_d':'''from collections import deque
import sys
f=sys.stdin.buffer;n=int(f.readline());g=[f.readline().decode().strip() for _ in range(n)];size=n*n
players=[i*n+j for i in range(n) for j in range(n) if g[i][j]=='P'];moves=[]
for i in range(n):
 for j in range(n):
  destinations=[]
  for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
   u,v=i+di,j+dj;destinations.append(u*n+v if 0<=u<n and 0<=v<n and g[u][v]!='#' else i*n+j)
  moves.append(destinations)
a,b=players;start=a*size+b;seen=bytearray(size*size);seen[start]=1;q=deque([start]);distance=0;answer=-1
while q:
 for _ in range(len(q)):
  state=q.popleft();a,b=divmod(state,size)
  if a==b:answer=distance;break
  for u,v in zip(moves[a],moves[b]):
   target=u*size+v
   if not seen[target]:seen[target]=1;q.append(target)
 else:distance+=1;continue
 break
print(answer)
'''}


def cases(scale=1):
    n=max(2,int(200000*scale));a=[1000000000-i for i in range(n)];insert=[500000000+i for i in range(n)]
    text=f'{n}\n'+ ' '.join(map(str,a))+f'\n{n}\n'+''.join(f'1 {a[0]} {v}\n' for v in insert)
    expected=' '.join(map(str,[a[0],*reversed(insert),*a[1:]]))+'\n'
    yield 'abc344_e_maximum_output','abc344_e',text,expected
    count=max(2,int(100000*scale));n=1000000
    text=f'{n} {2*count}\n'+('1 R\n2 1000000\n'*count)
    expected=''.join(f'{n-k} 0\n' for k in range(1,count+1))
    yield 'abc335_c_maximum_history','abc335_c',text,expected
    n=max(4,int(60*scale))
    for name,positions,wall in [('opposite_corners',((0,0),(n-1,n-1)),False),('center_adjacent',((n//2-1,n//2-1),(n//2,n//2)),False),('disconnected',((0,0),(n-1,n-1)),True)]:
        g=[['.']*n for _ in range(n)]
        if wall:
            for row in g:row[n//2]='#'
        for i,j in positions:g[i][j]='P'
        # Opposite boundaries require (n-1)+(n-1) moves. Adjacent centered
        # players require touching one horizontal and one vertical boundary.
        expected=-1 if wall else 2*(n-1) if name=='opposite_corners' else 2*min(n//2,n-n//2)
        yield 'abc339_d_'+name,'abc339_d',str(n)+'\n'+'\n'.join(map(''.join,g))+'\n',str(expected)+'\n'


def physical_cpus():
    allowed=sorted(os.sched_getaffinity(0));chosen=[];seen=set()
    for cpu in allowed:
        base=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=tuple((base/n).read_text().strip() for n in ('physical_package_id','core_id')) if base.exists() else (str(cpu),)
        if key not in seen:seen.add(key);chosen.append(cpu)
    return chosen


def environment():
    root=Path('/sys/fs/cgroup');files=['cpu.max','cpu.stat','memory.max','memory.current','cpuset.cpus.effective','cpu,cpuacct/cpu.cfs_quota_us','cpu,cpuacct/cpu.cfs_period_us','cpu,cpuacct/cpu.stat','memory/memory.limit_in_bytes','memory/memory.usage_in_bytes','cpuset/cpuset.cpus']
    return {'platform':platform.platform(),'python':sys.version,'candidate_python':subprocess.check_output(['/usr/bin/python3','--version'],text=True).strip(),
        'hostname':platform.node(),'allowed_cpus':sorted(os.sched_getaffinity(0)),'physical_cpus':physical_cpus(),
        'cgroup':{n:(root/n).read_text().strip() for n in files if (root/n).exists()},
        'memory_total_kib':next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:'))}


def setup_sandbox(root='/opt/gearshift-sandbox'):
    if sys.platform!='linux' or os.geteuid()!=0:raise RuntimeError('Dedicated Linux root CPU scorer required')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    # The isolated candidate is /usr/bin/python3, independent of the orchestrator venv.
    std=Path(subprocess.check_output(['/usr/bin/python3','-I','-c','import sysconfig;print(sysconfig.get_path("stdlib"))'],text=True).strip())
    target=root/std.relative_to('/')
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(std,target,ignore=shutil.ignore_patterns('site-packages','dist-packages','__pycache__','test','tests','ensurepip'))
    (root/'READY').write_text('Versioned scorer: read-only standard library; no host mounts, network, process creation or filesystem writes.\n')


def calibrate(output,workers=8,repetitions=3):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'frozen_policy.json').exists():raise ValueError('Policy already frozen; calibration is immutable')
    setup_sandbox();env=environment();cpus=env['physical_cpus']
    if workers<1 or workers>8 or workers>len(cpus):raise ValueError('Need one distinct physical core per worker, max eight')
    quota=env['cgroup'].get('cpu.max','max').split()
    if quota[0]!='max' and int(quota[0])/int(quota[1])<workers:raise ValueError('CPU quota oversubscribes requested concurrency')
    if 'cpu,cpuacct/cpu.cfs_quota_us' in env['cgroup']:
        q=int(env['cgroup']['cpu,cpuacct/cpu.cfs_quota_us']);period=int(env['cgroup']['cpu,cpuacct/cpu.cfs_period_us'])
        if q>0 and q/period<workers:raise ValueError('Cgroup-v1 CPU quota oversubscribes requested concurrency')
    memory=min(env['memory_total_kib']*1024,*[int(env['cgroup'][k]) for k in ('memory.max','memory/memory.limit_in_bytes') if k in env['cgroup'] and env['cgroup'][k]!='max'])
    if memory<workers*5*1024**3:raise ValueError('Insufficient memory for candidate and checker per worker')
    declaration={'sources':SOURCES,'references_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in REFERENCES.items()},
        'repetitions':repetitions,'workers':workers,'cpu_ids':cpus[:workers],
        'frozen_formula':'cpu_seconds=max(12,ceil(3*max(correct_reference_CPU_seconds))); wall_seconds=3*cpu_seconds+10.',
        'limits_basis':'At most 4 GiB address space; stdout minimum16 MiB, four times trusted expected UTF-8 size plus1 MiB, hard maximum64 MiB. Complete capture; no stdout truncation.',
        'candidate_timeouts_retried':False,'fixed_infrastructure_retries':1,'created_epoch':time.time(),'environment':env}
    write(output/'calibration_declaration.json',declaration)
    p={**DEFAULT_POLICY,'cpu_seconds':180,'wall_seconds':550,'maximum_concurrent_candidates':workers}
    records=[]
    allcases=list(cases())
    def one(item,cpu,rep):
        name,key,inp,expected=item
        r=score(REFERENCES[key],{'fn_name':None,'tests':[{'input':inp,'output':expected}]},policy=p,cpu_id=cpu)
        return {'case':name,'reference':key,'repetition':rep,'cpu_id':cpu,'input_sha256':hashlib.sha256(inp.encode()).hexdigest(),'expected_bytes':len(expected.encode()),'score':r}
    # Fixed three replicates, each with concurrent reference load on all scorer
    # cores, tests the actual bounded allocation instead of an idle-host best run.
    for rep in range(repetitions):
        assignments=[allcases[i%len(allcases)] for i in range(max(workers,len(allcases)))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0,len(assignments),workers):
                batch=[pool.submit(one,item,cpus[i],rep) for i,item in enumerate(assignments[offset:offset+workers])]
                records.extend(f.result() for f in batch)
        write(output/'calibration_progress.json',{'records':records,'repetitions_complete':rep+1})
    receipt={'passed':all(r['score']['passed'] is True for r in records),'records':records,'declaration_sha256':sha(output/'calibration_declaration.json'),'environment':env}
    write(output/'calibration_receipt.json',receipt)
    if not receipt['passed']:raise RuntimeError('Trusted reference calibration failed; no final policy frozen')
    maximum=max(r['score']['receipts'][0]['cpu_seconds'] for r in records);cpu=max(12,math.ceil(3*maximum))
    policy={**DEFAULT_POLICY,'cpu_seconds':cpu,'wall_seconds':3*cpu+10,'maximum_concurrent_candidates':workers,
        'policy_status':'frozen','calibration_receipt_sha256':sha(output/'calibration_receipt.json'),
        'calibration_declaration_sha256':sha(output/'calibration_declaration.json'),'physical_cpu_ids':cpus[:workers],
        'calibrated_environment':env,'maximum_reference_cpu_seconds':maximum}
    write(output/'frozen_policy.json',policy);write(output/'scorer_identity.json',scorer_identity(policy))
    print(json.dumps({'frozen':True,'cpu_seconds':cpu,'wall_seconds':policy['wall_seconds'],'workers':workers,'records':len(records),'policy_sha256':digest(policy)}),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--workers',type=int,default=8);a.add_argument('--setup-only',action='store_true');x=a.parse_args()
    if x.setup_only:setup_sandbox();print(json.dumps(environment()))
    else:calibrate(x.output,x.workers)
if __name__=='__main__':main()
