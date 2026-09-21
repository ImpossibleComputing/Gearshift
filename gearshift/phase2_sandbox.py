"""macOS deny-by-default code sandbox; generated code never runs outside it."""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time


def limits():
    resource.setrlimit(resource.RLIMIT_CPU,(6,6))
    resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024**2,2*1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.setsid()


def profile(directory):
    exe=Path(sys.executable).resolve();stdlib=Path(sysconfig.get_path('stdlib')).resolve()
    site=Path(sysconfig.get_path('purelib')).resolve()
    def sub(p):return '(subpath '+json.dumps(str(p))+')'
    rules=['(version 1)','(deny default)',
        '(allow process-exec (literal '+json.dumps(str(exe))+') (literal '+json.dumps(str(exe.parent.parent/'Resources/Python.app/Contents/MacOS/Python'))+'))',
        '(allow sysctl-read)',
        '(allow file-read-metadata)',
        '(allow file-read* (literal "/"))',
        '(allow file-read* (literal '+json.dumps(str(site))+'))',
        '(allow file-read* '+ ' '.join(sub(p) for p in ['/System/Library','/usr/lib',stdlib,exe.parent.parent,site/'numpy',site/'numpy.libs']) +')',
        '(allow file-read* file-write* '+sub(directory)+')',
        '(allow file-read* file-write* (literal "/dev/null"))',
        '(allow file-read* (literal "/dev/urandom") (literal "/dev/random"))']
    return '\n'.join(rules)


def execute(code,timeout=12):
    if sys.platform!='darwin' or not Path('/usr/bin/sandbox-exec').exists():
        return dict(status='sandbox_unavailable',executed=False)
    import psutil
    with tempfile.TemporaryDirectory(prefix='gearshift-exec-') as tmp:
        root=Path(tmp).resolve();path=root/'candidate.py'
        path.write_text('import sys\nsys.path.insert(0, '+repr(sysconfig.get_path('purelib'))+')\n'+code)
        policy=profile(root);sb=root/'policy.sb';sb.write_text(policy)
        output=root/'stdout.txt';error=root/'stderr.txt';start=time.perf_counter();peak=0
        with output.open('w') as out,error.open('w') as err:
            proc=subprocess.Popen(['/usr/bin/sandbox-exec','-f',str(sb),str(Path(sys.executable).resolve()),'-I','-S',str(path)],
                cwd=root,env={'PATH':'/usr/bin:/bin','TMPDIR':str(root),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'},
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,preexec_fn=limits)
            stopped=None
            while proc.poll() is None:
                try:peak=max(peak,psutil.Process(proc.pid).memory_info().rss)
                except psutil.NoSuchProcess:pass
                if time.perf_counter()-start>timeout:stopped='timeout'
                if peak>1024**3:stopped='memory_limit'
                if stopped:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    break
                time.sleep(.02)
            proc.wait()
        stderr=error.read_text(errors='replace');stdout=output.read_text(errors='replace')
        return dict(status=stopped or ('pass' if proc.returncode==0 else 'execution_failure'),executed=True,
            returncode=proc.returncode,stdout=stdout[-12000:],stderr=stderr[-12000:],wall_ms=(time.perf_counter()-start)*1000,
            peak_sampled_rss_bytes=peak,policy=policy,uid=os.getuid())


def probe():
    # Known harmless attacks against our own sentinel/listener; no real secret is accessed.
    with tempfile.TemporaryDirectory(prefix='gearshift-outside-') as tmp,socket.socket() as listener:
        secret=Path(tmp)/'sentinel.txt';secret.write_text('ISOLATION_SENTINEL_NOT_A_SECRET')
        listener.bind(('127.0.0.1',0));listener.listen();port=listener.getsockname()[1]
        code=f'''import json, socket, os, subprocess
result={{}}
try:
    open({str(secret)!r}).read();result['outside_read_denied']=False
except PermissionError:result['outside_read_denied']=True
try:
    open({str(secret.parent/'write-test')!r},'w').write('x');result['outside_write_denied']=False
except PermissionError:result['outside_write_denied']=True
try:
    s=socket.socket();s.connect(('127.0.0.1',{port}));result['network_denied']=False
except PermissionError:result['network_denied']=True
try:
    subprocess.run(['/bin/echo','x'],check=True);result['subprocess_denied']=False
except PermissionError:result['subprocess_denied']=True
import numpy as np
result['numpy_works']=bool(np.allclose([1,2],[1,2]))
result['unprivileged']=os.getuid()!=0
print(json.dumps(result))
assert all(result.values())
'''
        good=execute(code);bad=execute('assert 2+2==5');infinite=execute('while True: pass',timeout=.5)
        try:flags=json.loads(good['stdout'].strip().splitlines()[-1])
        except (KeyError,ValueError,IndexError):flags={}
        passed=good['status']=='pass' and len(flags)==6 and all(flags.values()) and bad['status']=='execution_failure' and infinite['status']=='timeout'
        return dict(passed=passed,restrictions=flags,normal_and_isolation=good,known_test_failure=bad,timeout_control=infinite)


def extract_code(text):
    import re
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S|re.I)
    # Frozen policy: first Python/unlabeled fenced block; otherwise all raw text. No repairs.
    return blocks[0] if blocks else text


def grade_code(task,answer,sandbox_passed):
    import secrets
    code=extract_code(answer)
    try:ast.parse(code)
    except SyntaxError as exc:return dict(status='syntax_failure',passed=False,parsed_code=code,error=str(exc),executed=False)
    if not sandbox_passed:return dict(status='pending_sandbox',passed=None,parsed_code=code,executed=False)
    marker='TESTS_COMPLETED_'+secrets.token_hex(16)
    result=execute(code+'\n'+task.hidden['test']+'\ncheck('+task.hidden['entry_point']+')\nprint('+repr(marker)+')\n')
    completed=marker in result.get('stdout','').splitlines()
    if result['status']=='pass' and not completed:result['status']='no_test_completion_marker'
    result.pop('policy',None)
    return dict(**result,passed=result['status']=='pass' and completed,tests_completed=completed,completion_marker=marker,parsed_code=code)
