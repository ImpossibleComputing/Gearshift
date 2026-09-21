"""Linux chroot, unprivileged UID and kernel syscall allowlist; checker stays outside."""
import ast, decimal, hashlib, json, os, re, signal, subprocess, sys, tempfile, time
from pathlib import Path

def extract(answer):
    matches=re.finditer(r'```([^\n`]*)\n(.*?)```',answer,re.S)
    for m in matches:
        if m[1].strip().lower() in ('','python','python3','py'):return m[2].strip()
    return answer.strip()

def equal_stdio(actual, expected):
    a=[x.strip() for x in actual.strip().split('\n')];b=[x.strip() for x in expected.strip().split('\n')]
    if len(a)!=len(b):return False
    for x,y in zip(a,b):
        if x==y:continue
        try:
            if [decimal.Decimal(t) for t in x.split()]==[decimal.Decimal(t) for t in y.split()]:continue
        except decimal.InvalidOperation:pass
        return False
    return True

def run(code, test_input, fn_name=None, seconds=6, sandbox_root='/opt/gearshift-sandbox'):
    if sys.platform!='linux' or os.geteuid()!=0 or not Path(sandbox_root,'READY').exists():
        return {'status':'sandbox_unavailable','executed':False}
    runner=Path(__file__).resolve().parents[1]/'scripts/coding_sandbox_child.py'
    payload=json.dumps({'code':code,'input':test_input,'fn_name':fn_name,'root':sandbox_root,'seconds':seconds})
    with tempfile.TemporaryDirectory(prefix='coding-check-') as tmp:
        out=Path(tmp)/'stdout';err=Path(tmp)/'stderr';started=time.monotonic()
        payload_path=Path(tmp)/'payload.json';payload_path.write_text(payload)
        with out.open('wb') as o,err.open('wb') as e:
            p=subprocess.Popen(['/usr/bin/python3','-I',str(runner),str(payload_path)],stdin=subprocess.PIPE,stdout=o,stderr=e,
                env={'LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
            try:p.communicate(test_input.encode(),timeout=seconds+3);status='completed'
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait();status='timeout'
        stdout=out.read_bytes()[:2*1024**2].decode(errors='replace');stderr=err.read_bytes()[:32768].decode(errors='replace')
        if p.returncode!=0 and status=='completed':status='timeout' if p.returncode in (-signal.SIGXCPU,-signal.SIGKILL) else 'runtime_error'
        return {'status':status,'executed':True,'returncode':p.returncode,'stdout':stdout,'stderr':stderr,'wall_seconds':time.monotonic()-started}

def score(code, spec, guard=None):
    try:ast.parse(code)
    except SyntaxError as e:return {'passed':False,'category':'syntax','error':str(e),'executed_tests':0}
    tests=spec['tests']; receipts=[]
    if not tests:return {'passed':False,'category':'missing_tests','executed_tests':0}
    for i,t in enumerate(tests):
        if guard:guard()
        r=run(code,t['input'],spec['fn_name']);passed=False
        if r['status']=='completed':
            if spec['fn_name']:
                try:
                    obj=json.loads(r['stdout']);passed=(set(obj)=={'result'} and obj['result']==json.loads(t['output']))
                except (ValueError,TypeError):pass
            else:passed=equal_stdio(r['stdout'],t['output'])
        receipts.append({'test_index':i,'passed':passed,'execution_status':r['status'],'returncode':r.get('returncode'),
            'input_sha256':hashlib.sha256(t['input'].encode()).hexdigest(),'output_sha256':hashlib.sha256(r.get('stdout','').encode()).hexdigest(),
            'expected_sha256':hashlib.sha256(t['output'].encode()).hexdigest(),'wall_seconds':r.get('wall_seconds'), 'stderr':r.get('stderr','')})
        if not passed:return {'passed':False,'category':'test_assertion' if r['status']=='completed' else r['status'],
            'executed_tests':len(receipts),'total_tests':len(tests),'receipts':receipts,'trusted_checker_complete':True}
    return {'passed':True,'category':'pass','executed_tests':len(receipts),'total_tests':len(tests),'receipts':receipts,'trusted_checker_complete':True}
