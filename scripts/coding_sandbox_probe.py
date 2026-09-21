#!/usr/bin/env python3
import json,os,shutil,sys,sysconfig
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_sandbox import run,score,extract,equal_stdio
from gearshift.coding_control import write,sha

def main():
    assert sys.platform=='linux' and os.geteuid()==0
    root=Path('/opt/gearshift-sandbox');root.mkdir(exist_ok=True)
    std=Path(sysconfig.get_path('stdlib'))
    target=root/std.relative_to('/')
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(std,target,ignore=shutil.ignore_patterns('site-packages','dist-packages','__pycache__','test','tests','ensurepip'))
    (root/'READY').write_text('Read-only runtime; zero disk temporary storage; all filesystem writes denied by seccomp.\n')
    good='print(sum(map(int,input().split())))'
    spec={'fn_name':None,'tests':[{'input':'2 3\n','output':'5\n'}]}
    checks={'stdin_good':score(good,spec)['passed'], 'stdin_wrong':not score('print(9)',spec)['passed'],
        'stdin_fd_open':score('print(sum(map(int,open(0).read().split())))',spec)['passed'],
        'stdin_os_read':score('import os; print(sum(map(int,os.read(0,100).split())))',spec)['passed'],
        'malformed':score('def :',spec)['category']=='syntax','early_exit':not score('import os; os._exit(0)',spec)['passed'],
        'output_spoof':not score('print("{\\"passed\\": true}")',spec)['passed'],
        'function_good':score('class Solution:\n def plus(self,a,b):return a+b',{'fn_name':'plus','tests':[{'input':'2\n3','output':'5'}]})['passed']}
    attacks={
      'outside_files':'open("/etc/passwd").read()',
      'outside_parent':'open("/../../workspace/Gearshift/configs/coding_pilot_v1/pilot.json").read()',
      'filesystem_write':'open("/tmp/escape", "w").write("x")',
      'network':'import socket; socket.socket()',
      'fork':'import os; os.fork()',
      'subprocess':'import subprocess; subprocess.run(["/bin/true"])',
      'gpu':'open("/dev/nvidia0","rb")',
      'proc':'open("/proc/1/environ","rb").read()',
      'memory':'a=bytearray(5*1024**3)',
      'cpu':'while True:pass',
      'output_limit':'print("x"*(3*1024**2))'}
    detail={}
    for name,code in attacks.items():
        r=run(code,'',seconds=1);detail[name]=r;checks[name]=r['executed'] and r['status']!='completed'
        if name in ['network','fork','subprocess']:checks[name]=checks[name] and 'PermissionError' in r.get('stderr','')
    uid=run('import os;print(os.getuid())','');checks['unprivileged_uid']=uid.get('stdout','').strip()=='65534'
    receipt={'passed':all(checks.values()),'checks':checks,'probes':detail,
        'isolation':'chroot + uid/gid 65534 + kernel seccomp allowlist; no host mounts, proc, devices, process creation or network',
        'temporary_disk_bytes':0,'memory_limit_bytes':4*1024**3,'per_test_cpu_seconds':6,'checker':'parent compares only candidate output with outside expected answers',
        'implementation_sha256':sha(Path(__file__).resolve().parents[1]/'scripts/coding_sandbox_child.py')}
    private=Path(__file__).resolve().parents[1]/'data/coding_pilot_v1/private/development.json'
    if private.exists():
        tests=json.loads(private.read_text())
        canonical={
            'atcoder/abc356_a':'n,l,r=map(int,open(0).read().split()); a=list(range(1,n+1)); a[l-1:r]=a[l-1:r][::-1]; print(*a)',
            'leetcode/3379':'class Solution:\n def scoreOfString(self,s: str)->int:\n  return sum(abs(ord(a)-ord(b)) for a,b in zip(s,s[1:]))'}
        canonical_receipts={}
        for task_id,code in canonical.items():
            result=score(code,tests[task_id]);checks['development_'+task_id]=result['passed']
            canonical_receipts[task_id]=result
        receipt['canonical_development']=canonical_receipts
        receipt['passed']=all(checks.values())
    write(Path(__file__).resolve().parents[1]/'evidence/coding_pilot_v1/sandbox_gate.json',receipt)
    print(json.dumps({'passed':receipt['passed'],'checks':checks}),flush=True)
    assert receipt['passed'],checks
if __name__=='__main__':main()
