"""Versioned bounded scorer; originals and candidate extraction remain unchanged.

All stdout bytes are captured before comparison. Resource/time outcomes are
program failures; verified launch/confinement/capture failures are missing
infrastructure coverage, eligible only for the frozen bounded retry policy.
"""
import ast
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from .coding_control import digest, sha
from .coding_sandbox import extract, equal_stdio

SCORER_VERSION = 'gearshift_scorer_v2_20260919_01'
DEFAULT_POLICY = {
    'schema_version': 2, 'scorer_version': SCORER_VERSION,
    'cpu_seconds': 12, 'wall_seconds': 46,
    'address_space_bytes': 4*1024**3,
    'stdout_min_bytes': 16*1024**2, 'stdout_max_bytes': 64*1024**2,
    'stdout_expected_multiplier': 4, 'stdout_extra_bytes': 1024**2,
    'stderr_limit_bytes': 256*1024,
    'discarded_function_stdout_limit_bytes': 16*1024**2,
    'maximum_infrastructure_retries': 1, 'candidate_timeout_retries': 0,
    'maximum_concurrent_candidates': 8,
    'policy_status': 'calibration_draft_not_frozen',
}


def validate_policy(policy):
    p = dict(policy)
    if p.get('scorer_version') != SCORER_VERSION:
        raise ValueError('Unknown scorer version')
    for name in ('cpu_seconds','wall_seconds','address_space_bytes','stdout_min_bytes','stdout_max_bytes',
                 'stdout_expected_multiplier','stdout_extra_bytes','stderr_limit_bytes',
                 'discarded_function_stdout_limit_bytes','maximum_concurrent_candidates'):
        if type(p.get(name)) is not int or p[name] <= 0:
            raise ValueError('Invalid bounded resource policy: '+name)
    if not p['cpu_seconds'] < p['wall_seconds'] or p['stdout_min_bytes'] > p['stdout_max_bytes']:
        raise ValueError('Invalid CPU/wall/output limits')
    if p.get('maximum_infrastructure_retries') != 1 or p.get('candidate_timeout_retries') != 0:
        raise ValueError('Only one infrastructure retry and no timeout retries are allowed')
    return p


def scorer_identity(policy):
    root = Path(__file__).resolve().parents[1]
    files = ['gearshift/coding_sandbox.py','gearshift/coding_sandbox_v2.py','scripts/coding_sandbox_child_v2.py']
    return {'scorer_version':SCORER_VERSION,'policy_sha256':digest(validate_policy(policy)),
            'files':{n:sha(root/n) for n in files},'extraction_and_comparison':'Imported unchanged original extract/equal_stdio.'}


def output_allowance(policy, expected_output=None, fn_name=None):
    p = validate_policy(policy)
    expected_bytes = len(expected_output.encode('utf-8')) if expected_output is not None else 0
    # A call-based result adds a fixed wrapper; headroom also permits ordinary
    # alternate whitespace/number rendering accepted by the unchanged checker.
    needed = p['stdout_expected_multiplier']*(expected_bytes+32)+p['stdout_extra_bytes']
    if needed > p['stdout_max_bytes']:
        raise ValueError('Trusted expected output exceeds the frozen supported allowance; scorer repair required')
    return max(p['stdout_min_bytes'],needed)


def _missing(reason, **extra):
    return {'status':'infrastructure_failure','executed':False,'missing':True,'infrastructure_reason':reason,**extra}


def run(code, test_input, fn_name=None, *, policy, cpu_id=None,
        expected_output=None, sandbox_root='/opt/gearshift-sandbox', guard=None):
    p = validate_policy(policy)
    if sys.platform != 'linux' or os.geteuid() != 0 or not Path(sandbox_root,'READY').is_file():
        return _missing('sandbox_unavailable')
    if cpu_id is not None and cpu_id not in os.sched_getaffinity(0):
        return _missing('assigned_cpu_not_available',cpu_id=cpu_id)
    try:
        allowance = output_allowance(p,expected_output,fn_name)
    except ValueError as exc:
        return _missing('unsupported_trusted_output_size',detail=str(exc))
    child_policy = {**p,'stdout_limit_bytes':allowance}
    child = Path(__file__).resolve().parents[1]/'scripts/coding_sandbox_child_v2.py'
    with tempfile.TemporaryDirectory(prefix='gearshift-score-v2-') as temp:
        tmp = Path(temp); inp=tmp/'stdin'; out=tmp/'stdout'; err=tmp/'stderr'; payload=tmp/'payload.json'
        inp.write_bytes(test_input.encode('utf-8'))
        status_read,status_write = os.pipe(); os.set_inheritable(status_write,True)
        payload.write_text(json.dumps({'code':code,'input':test_input,'fn_name':fn_name,'root':sandbox_root,
                                     'policy':child_policy,'cpu_id':cpu_id,'status_fd':status_write}))
        process=None; usage=None; exitcode=None; killed_for=None; launch_error=None; started=time.monotonic(); host_started=time.time()
        try:
            with inp.open('rb') as i,out.open('wb') as o,err.open('wb') as e:
                process=subprocess.Popen(['/usr/bin/python3','-I',str(child),str(payload)],stdin=i,stdout=o,stderr=e,
                    pass_fds=(status_write,),env={'LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
                os.close(status_write);status_write=None
                while True:
                    done,status,ru=os.wait4(process.pid,os.WNOHANG)
                    if done:
                        usage=ru;exitcode=os.waitstatus_to_exitcode(status);process.returncode=exitcode;break
                    elapsed=time.monotonic()-started
                    if err.stat().st_size >= p['stderr_limit_bytes']:
                        killed_for='stderr_output_limit'
                    elif elapsed >= p['wall_seconds']:
                        killed_for='wall_timeout'
                    if killed_for:
                        os.killpg(process.pid,signal.SIGKILL)
                        _,status,usage=os.wait4(process.pid,0);exitcode=os.waitstatus_to_exitcode(status);process.returncode=exitcode;break
                    if guard:guard()
                    time.sleep(.01)
        except (OSError,ValueError) as exc:
            launch_error=_missing('process_launch_or_capture_error',exception=type(exc).__name__,detail=str(exc))
        except BaseException:
            os.close(status_read)
            raise
        finally:
            if status_write is not None:os.close(status_write)
            if process is not None and process.returncode is None:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:_,status,usage=os.wait4(process.pid,0);process.returncode=os.waitstatus_to_exitcode(status)
                except ChildProcessError:pass
        try:
            trusted=os.read(status_read,65536)
        finally:os.close(status_read)
        if launch_error is not None:return launch_error
        elapsed=time.monotonic()-started;stdout_bytes=out.read_bytes();stderr_bytes=err.read_bytes()
        try:ready=json.loads(trusted)
        except (ValueError,UnicodeError):ready={'ready':False,'error':'missing_or_malformed_confinement_receipt'}
        cpu_seconds=(usage.ru_utime+usage.ru_stime) if usage else None
        stdout=stdout_bytes.decode('utf-8',errors='replace');stderr=stderr_bytes[:p['stderr_limit_bytes']].decode('utf-8',errors='replace')
        result={'executed':bool(ready.get('ready')),'missing':False,'returncode':exitcode,
            'stdout':stdout,'stderr':stderr,'stdout_bytes':len(stdout_bytes),'stderr_bytes':len(stderr_bytes),
            'stdout_sha256':hashlib.sha256(stdout_bytes).hexdigest(),'stderr_sha256':hashlib.sha256(stderr_bytes).hexdigest(),
            'stdout_capture_complete':len(stdout_bytes)==out.stat().st_size,'stderr_receipt_truncated':len(stderr_bytes)>p['stderr_limit_bytes'],
            'wall_seconds':elapsed,'cpu_seconds':cpu_seconds,'maximum_rss_kib':usage.ru_maxrss if usage else None,
            'stdout_limit_bytes':allowance,'cpu_limit_seconds':p['cpu_seconds'],'wall_limit_seconds':p['wall_seconds'],
            'assigned_cpu':cpu_id,'confinement':ready,'started_epoch':host_started}
        if not ready.get('ready'):
            return {**result,**_missing('confinement_initialization_failed')}
        if not result['stdout_capture_complete']:
            return {**result,**_missing('incomplete_stdout_capture')}
        if killed_for=='stderr_output_limit' or (exitcode!=0 and len(stdout_bytes)>=allowance) or b'File too large' in stderr_bytes or b'OutputLimitError' in stderr_bytes:
            status='output_limit'
        elif killed_for=='wall_timeout':status='wall_timeout'
        elif exitcode==-signal.SIGXCPU or (exitcode==-signal.SIGKILL and cpu_seconds is not None and cpu_seconds >= p['cpu_seconds']-.25):
            status='cpu_timeout'
        elif b'MemoryError' in stderr_bytes:status='memory_limit'
        elif exitcode in (-signal.SIGKILL,-signal.SIGTERM):
            return {**result,**_missing('unexpected_external_termination',executed=True)}
        elif exitcode!=0:status='runtime_error'
        else:status='completed'
        result['status']=status
        return result


def score(code, spec, guard=None, *, policy, cpu_id=None, sandbox_root='/opt/gearshift-sandbox'):
    p=validate_policy(policy);identity=scorer_identity(p)
    try:ast.parse(code)
    except SyntaxError as exc:
        return {'passed':False,'missing':False,'category':'syntax','error':str(exc),'executed_tests':0,'scorer_identity':identity}
    tests=spec.get('tests')
    if not tests:
        return {'passed':None,'missing':True,'category':'infrastructure_failure','reason':'missing_tests','executed_tests':0,'scorer_identity':identity}
    try:
        if spec.get('fn_name'):
            for test in tests:json.loads(test['output'])
        for test in tests:
            if not isinstance(test['input'],str) or not isinstance(test['output'],str):raise ValueError('Expected text test interface')
    except (KeyError,ValueError,TypeError):
        return {'passed':None,'missing':True,'category':'infrastructure_failure','reason':'invalid_trusted_test_spec','executed_tests':0,'scorer_identity':identity}
    receipts=[]
    for index,test in enumerate(tests):
        if guard:guard()
        attempts=[]
        for attempt in range(p['maximum_infrastructure_retries']+1):
            r=run(code,test['input'],spec['fn_name'],policy=p,cpu_id=cpu_id,expected_output=test['output'],sandbox_root=sandbox_root,guard=guard)
            attempts.append({k:v for k,v in r.items() if k not in ('stdout','stderr')})
            if not r.get('missing'):break
        passed=False
        if r['status']=='completed':
            if spec['fn_name']:
                try:
                    value=json.loads(r['stdout']);passed=set(value)=={'result'} and value['result']==json.loads(test['output'])
                except (ValueError,TypeError):pass
            else:passed=equal_stdio(r['stdout'],test['output'])
        receipt={'test_index':index,'passed':None if r.get('missing') else passed,
            'execution_status':r['status'],'missing':bool(r.get('missing')),'returncode':r.get('returncode'),
            'input_sha256':hashlib.sha256(test['input'].encode()).hexdigest(),
            'expected_sha256':hashlib.sha256(test['output'].encode()).hexdigest(),
            'output_sha256':r.get('stdout_sha256'),'stdout_bytes':r.get('stdout_bytes'),
            'stdout_capture_complete':r.get('stdout_capture_complete'),'stdout_limit_bytes':r.get('stdout_limit_bytes'),
            'wall_seconds':r.get('wall_seconds'),'cpu_seconds':r.get('cpu_seconds'),'maximum_rss_kib':r.get('maximum_rss_kib'),
            'stderr':'\n'.join(re.findall(r'(?m)^(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)|KeyboardInterrupt|SystemExit)(?=:|$)',r.get('stderr',''))),
            'stderr_text_policy':'Exception types only; candidate diagnostics may contain private inputs and are withheld. Full bytes are hashed.',
            'stderr_sha256':r.get('stderr_sha256'),'stderr_receipt_truncated':r.get('stderr_receipt_truncated'),
            'attempts':attempts,'frozen_infrastructure_retry_limit':p['maximum_infrastructure_retries']}
        receipts.append(receipt)
        if not passed:
            return {'passed':None if r.get('missing') else False,'missing':bool(r.get('missing')),
                'category':'test_assertion' if r['status']=='completed' else r['status'],
                'executed_tests':len(receipts),'total_tests':len(tests),'receipts':receipts,
                'trusted_checker_complete':not r.get('missing',False),'scorer_identity':identity}
    return {'passed':True,'missing':False,'category':'pass','executed_tests':len(receipts),'total_tests':len(tests),
            'receipts':receipts,'trusted_checker_complete':True,'scorer_identity':identity}
