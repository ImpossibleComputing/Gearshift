#!/usr/bin/env python3
"""Read included-usage permission only; never redeem credits or run inference."""
import json
import select
import subprocess
import tempfile
import time
from phase2_judge_probe import CLI,clean_env


def included_usage():
    with tempfile.TemporaryFile(mode='w+') as errors:
        p=subprocess.Popen([CLI,'app-server','--listen','stdio://'],
            env=clean_env(),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,bufsize=1)
        def send(obj):p.stdin.write(json.dumps(obj)+'\n');p.stdin.flush()
        def reply(identifier):
            deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                if not select.select([p.stdout],[],[],max(0,deadline-time.monotonic()))[0]:break
                line=p.stdout.readline()
                if not line:break
                obj=json.loads(line)
                if obj.get('id')==identifier:
                    if 'error' in obj:raise RuntimeError('Read-only usage request failed')
                    return obj['result']
            raise TimeoutError('Read-only included-usage verification unavailable')
        try:
            send(dict(id=1,method='initialize',params=dict(clientInfo=dict(name='gearshift_usage_guard',version='1'))));reply(1)
            send(dict(method='initialized',params={}))
            send(dict(id=2,method='account/rateLimits/read',params=dict(excludeResetCreditDetails=True,supportsLunaReserve=False)))
            result=reply(2);buckets=result.get('rateLimitsByLimitId') or {}
            bucket=buckets.get('codex',result.get('rateLimits',{}));windows=[]
            for key in ['primary','secondary']:
                if bucket.get(key):windows.append({k:bucket[key].get(k) for k in ['usedPercent','windowDurationMins','resetsAt']})
            allowed=result.get('ordinaryUsageAllowed') is True and bool(windows) and all(type(w['usedPercent']) is int and w['usedPercent']<85 for w in windows)
            return dict(allowed=allowed,ordinary_usage_allowed=result.get('ordinaryUsageAllowed'),windows=windows,
                policy='Run judges only while ordinary included usage is explicitly permitted and every exposed core window has more than 15 percent remaining. Never redeem/reset credits or use a paid fallback.',
                read_method='account/rateLimits/read',checked_unix_seconds=time.time())
        finally:
            p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()


if __name__=='__main__':print(json.dumps(included_usage(),indent=2))
