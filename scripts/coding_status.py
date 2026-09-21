#!/usr/bin/env python3
"""Compact read-only status from verified Studio receipts, without model calls."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'evidence/coding_pilot_v1/control'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def main():
    saved=ROOT/'evidence/coding_pilot_v1/pod_backup/latest'
    namespace=read(C/'session_status.json').get('namespace','preflight_v3');r=saved/'results/coding_pilot_v1'/namespace
    watchdog=read(C/'watchdog_status.json');ack=read(C/'backup_ack.json')
    recovered=[]
    for parent in (saved/'results/coding_pilot_v1/preflight_v2/tasks').glob('*/partial_*.json'):
        old=read(parent);new=read(r/'tasks'/parent.parent.name/parent.name)
        tokens=new.get('token_ids',[]);before=old['token_ids'];n=min(len(tokens),len(before))
        recovered.append({'task_id':old['task_id'],'segment':old['segment'],'saved_tokens':len(before),'recomputed_tokens':len(tokens),
            'checked_tokens':n,'checked_prefix_matches':tokens[:n]==before[:n],'entire_saved_prefix_reproduced':len(tokens)>=len(before) and tokens[:len(before)]==before})
    out={'observed_epoch':time.time(),'controller':read(C/'session_status.json'),'worker':read(C/'latest_worker_status.json'),
        'budget':{k:watchdog.get(k) for k in ['epoch','upper_usd','gpu_hours','stage_gpu_hours','stage_wall_hours','stage_gpu_hour_ceiling','stage_dollar_ceiling','warning','stop','stop_reasons','active_resources']},
        'backup_age_seconds':time.time()-ack.get('epoch',0),'verified_files':ack.get('verified_files'),
        'development':read(r/'progress.json'),'baseline_gate':read(r/'baseline_gate.json'),'recovery':recovered,
        'limit_alert':read(C/'worker_ALERT.json'),'note':'Partial development counts are diagnostic; baseline and transfer claims require their complete prespecified gates.'}
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
