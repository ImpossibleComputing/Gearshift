import copy
import json
import os
from pathlib import Path
import sys

import pytest

from gearshift import coding_sandbox_v2 as sandbox
from gearshift.coding_sandbox import extract, equal_stdio


def policy(**changes):
    p=copy.deepcopy(sandbox.DEFAULT_POLICY);p.update(changes);return p


def test_original_extraction_and_comparison_are_reused_exactly():
    assert sandbox.extract is extract and sandbox.equal_stdio is equal_stdio
    assert extract('```python\nprint(1)\n```\n```python\nprint(2)\n```')=='print(1)'
    assert not equal_stdio('50000000000000000','50000000000000001')


def test_task_reference_output_headroom_is_bounded_and_candidate_independent():
    p=policy();assert sandbox.output_allowance(p,'x'*4_400_001)==4*4_400_033+1024**2
    assert sandbox.output_allowance(p,'1')==16*1024**2
    with pytest.raises(ValueError):sandbox.output_allowance(p,'x'*17_000_000)


def test_only_verified_infrastructure_gets_one_retry(monkeypatch):
    calls=[]
    def run(*a,**kw):
        calls.append(1)
        if len(calls)==1:return {'status':'infrastructure_failure','missing':True}
        return {'status':'completed','missing':False,'stdout':'1\n'}
    monkeypatch.setattr(sandbox,'run',run)
    result=sandbox.score('print(1)',{'fn_name':None,'tests':[{'input':'','output':'1\n'}]},policy=policy())
    assert result['passed'] is True and len(calls)==2
    calls.clear()
    monkeypatch.setattr(sandbox,'run',lambda *a,**kw:(calls.append(1) or {'status':'cpu_timeout','missing':False}))
    result=sandbox.score('while True: pass',{'fn_name':None,'tests':[{'input':'','output':'1'}]},policy=policy())
    assert result['passed'] is False and result['category']=='cpu_timeout' and len(calls)==1


def test_unresolved_infrastructure_is_missing_not_a_quality_failure(monkeypatch):
    monkeypatch.setattr(sandbox,'run',lambda *a,**kw:{'status':'infrastructure_failure','missing':True})
    result=sandbox.score('print(1)',{'fn_name':None,'tests':[{'input':'','output':'1'}]},policy=policy())
    assert result['passed'] is None and result['missing'] and not result['trusted_checker_complete']
    assert len(result['receipts'][0]['attempts'])==2


LINUX_READY=sys.platform=='linux' and os.geteuid()==0 and Path('/opt/gearshift-sandbox/READY').exists()
linux=pytest.mark.skipif(not LINUX_READY,reason='Requires dedicated Linux root/seccomp scoring environment')


@linux
def test_valid_multimegabyte_stdio_capture_is_complete_and_tail_is_checked():
    n=5*1024**2;expected='x'*n+'TAIL\n'
    r=sandbox.run(f"print('x'*{n}+'TAIL')",'',policy=policy(),expected_output=expected)
    assert r['status']=='completed' and r['stdout_capture_complete'] and r['stdout']==expected
    assert r['stdout_bytes']==len(expected) and r['stdout_bytes']>4*1024**2
    spec={'fn_name':None,'tests':[{'input':'','output':expected}]}
    result=sandbox.score(f"print('x'*{n}+'FAIL')",spec,policy=policy())
    assert result['passed'] is False and result['category']=='test_assertion'


@linux
def test_multimegabyte_function_serialization_has_no_parent_truncation():
    text='x'*(5*1024**2)
    spec={'fn_name':'large','tests':[{'input':'0','output':json.dumps(text)}]}
    result=sandbox.score("class Solution:\n def large(self,n): return 'x'*(5*1024**2)",spec,policy=policy())
    assert result['passed'] is True and result['receipts'][0]['stdout_bytes']>4*1024**2
    assert result['receipts'][0]['stdout_capture_complete']


@linux
@pytest.mark.parametrize('kind',['stdio','discarded_function_stdout','function_result'])
def test_actual_output_flood_is_terminated(kind):
    p=policy(stdout_min_bytes=1024**2,stdout_max_bytes=2*1024**2,stdout_extra_bytes=1024,discarded_function_stdout_limit_bytes=1024**2)
    if kind=='stdio':code="import os\nwhile True: os.write(1,b'x'*65536)";fn=None;inp=''
    elif kind=='discarded_function_stdout':code="def huge(n):\n while True: print('x'*65536)";fn='huge';inp='0'
    else:code="def huge(n): return 'x'*(3*1024**2)";fn='huge';inp='0'
    r=sandbox.run(code,inp,fn,policy=p)
    assert r['status']=='output_limit' and not r['missing'] and r['wall_seconds']<p['wall_seconds']


@linux
def test_infinite_loop_is_algorithmic_timeout_without_retry():
    result=sandbox.score('while True: pass',{'fn_name':None,'tests':[{'input':'','output':'1'}]},
                         policy=policy(cpu_seconds=1,wall_seconds=5))
    assert result['category']=='cpu_timeout' and not result['missing']
    assert len(result['receipts'][0]['attempts'])==1


@linux
@pytest.mark.parametrize('code',["open('/etc/passwd').read()","open('/tmp/escape','w').write('x')",
                                'import socket;socket.socket()','import os;os.fork()',
                                "import subprocess;subprocess.run(['/bin/true'])","open('/proc/1/environ').read()"])
def test_confinement_still_blocks_host_files_writes_network_and_children(code):
    r=sandbox.run(code,'',policy=policy())
    assert r['executed'] and r['status']=='runtime_error' and not r['missing']


@linux
def test_child_remains_unprivileged_and_cpu_pinned():
    cpu=min(os.sched_getaffinity(0));r=sandbox.run('import os;print(os.getuid())','',policy=policy(),cpu_id=cpu)
    assert r['status']=='completed' and r['stdout'].strip()=='65534'
    assert r['confinement']['cpu_affinity']==[cpu]


@linux
def test_address_space_limit_remains_enforced():
    result=sandbox.run('a=bytearray(5*1024**3)','',policy=policy())
    assert result['status']=='memory_limit' and not result['missing']
