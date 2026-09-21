#!/usr/bin/env python3
"""Read public progress once per physical store; never score candidate content."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import coding_confirmation_dispatch as base
from scripts.coding_confirmation_regional_dispatch import inventory
from scripts.coding_confirmation_completed_primary import completed_primary
CONDITIONS=['A','B','D','P','FIXED_M','ROTATING_M','FIXED_H','ROTATING_H']
SECONDARY_CONDITIONS=['FIXED_M','ROTATING_M','FIXED_H','ROTATING_H']
CODE=r'''import json,pathlib,subprocess,time
p=pathlib.Path('/workspace/GearshiftConfirmationPrimary/results/coding_pilot_v1/confirmation_01_20260919T094418Z')
t=pathlib.Path('/workspace/GearshiftConfirmation/results/coding_pilot_v1/confirmation_01_20260919T094418Z')
read=lambda f:json.loads(f.read_text())
delegated=pathlib.Path('/workspace/GearshiftConfirmationFranceSecondary/results/coding_pilot_v1/confirmation_01_20260919T094418Z')
delegated_value=None
if delegated.is_dir():
    dc={c:len(list(delegated.glob('secondary/tasks/*/'+c+'/*/draw_complete.json'))) for c in ['FIXED_M','ROTATING_M','FIXED_H','ROTATING_H']}
    delegated_value={'epoch':time.time(),'completed_answers':sum(dc.values()),'answers_by_condition':dc,
        'jobs_completed':{'receiver':len(list(delegated.glob('secondary/jobs/receiver_*/complete.json')))},
        'workers':[read(f) for f in delegated.glob('workers/*/latest.json')],
        'failures':[{'path':str(f),'type':read(f).get('type'),'error':read(f).get('error')} for f in delegated.glob('workers/**/failure.json')],
        'ready_primary_controls':len(list(delegated.glob('primary/jobs/receiver_*/complete.json')))}
conditions=['A','B','D','P','FIXED_M','ROTATING_M','FIXED_H','ROTATING_H']
answers={c:len(list(p.glob('primary/tasks/*/'+c+'/*/draw_complete.json'))) for c in conditions}
secondary_answers={c:len(list(p.glob('secondary/tasks/*/'+c+'/*/draw_complete.json'))) for c in conditions[4:]}
print(json.dumps({'epoch':time.time(),'delegated_secondary':delegated_value,'workers':[read(f) for f in p.glob('workers/*/latest.json')],
'answers_by_condition':answers,'completed_answers':sum(answers.values()),
'secondary_answers_by_condition':secondary_answers,'secondary_completed_answers':sum(secondary_answers.values()),
'secondary_jobs_completed':{'receiver':len(list(p.glob('secondary/jobs/receiver_*/complete.json')))},
'source_histories':len(list(p.glob('primary/tasks/*/large_history/history_ready.json'))),
'small_histories':len(list(p.glob('primary/tasks/*/small_history/history_ready.json'))),
'jobs_completed':{k:len(list(p.glob('primary/jobs/'+k+'_*/complete.json'))) for k in ['source','small','receiver']},
'failures':[{'path':str(f),'type':read(f).get('type'),'error':read(f).get('error')} for f in p.glob('workers/**/failure.json')],
'secondary_training_workers':[read(f) for f in t.glob('workers/*/latest.json')],
'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()}))
'''

def collect(pod):
    try:
        r=subprocess.run(base.ssh_args(pod)+['python3 -c '+shlex.quote(CODE)],capture_output=True,text=True,timeout=60)
        if r.returncode:raise RuntimeError(r.stderr[-1000:])
        return {'pod_id':pod['id'],'snapshot':json.loads(r.stdout)}
    except Exception as exc:return {'pod_id':pod['id'],'error':str(exc)}

def summarize(partition, pods, snapshots, cohort='primary', completed=None):
    if cohort not in ('primary', 'secondary'):raise ValueError('Unknown confirmation cohort')
    secondary=cohort=='secondary';worker_suffix='_secondary_' if secondary else '_generation_'
    fields=['completed_answers','answers_by_condition','jobs_completed']
    if not secondary:fields+=['source_histories','small_histories']
    bypod={r['pod_id']:r for r in snapshots}; stores={}
    for sid,store in partition['shard_storage'].items():
        candidates=[p for p in pods if p.get('mounts',{}).get('network')==[{'volumeId':store['network_volume_id'],'path':'/workspace'}]]
        good=[bypod[p['id']]['snapshot'] for p in candidates if 'snapshot' in bypod[p['id']]]
        if not good:
            if not secondary and completed and sid in completed:
                stores[sid]=completed[sid];continue
            stores[sid]={'readable':False,'task_count':len(partition['shards'][sid]),'reason':'No readable live host; inspect durable shard backup before claiming coverage.'};continue
        value=max(good,key=lambda x:x['epoch'])
        liveids={p['id'] for p in candidates}
        stores[sid]={'readable':True,'network_volume_id':store['network_volume_id'],'task_count':len(partition['shards'][sid]),
            'epoch':value['epoch'],**{k:value[('secondary_' if secondary else '')+k] for k in fields},
            'current_workers':[r for r in value['workers'] if r.get('pod_id') in liveids and
                               r.get('worker_id','').startswith(r['pod_id']+worker_suffix)],
            'current_worker_failures':[r for r in value['failures'] if any('/workers/'+pid+worker_suffix in r['path'] for pid in liveids)]}
    complete=all(x['readable'] for x in stores.values())
    totals={k:sum(x.get(k,0) for x in stores.values()) for k in (['completed_answers'] if secondary else ['completed_answers','source_histories','small_histories'])}
    totals['answers_by_condition']={c:sum(x.get('answers_by_condition',{}).get(c,0) for x in stores.values()) for c in (SECONDARY_CONDITIONS if secondary else CONDITIONS)}
    return {'all_shard_stores_readable':all(x['readable'] and x.get('live_store_readable',True) for x in stores.values()),
            'all_shard_coverage_available':complete,'coverage_is_partial_observation':not complete,'shards':stores,'totals':totals}

def delegated_summary(summary,pods,snapshots,owner):
    """One physical destination read for the unchanged logical France population."""
    sid=owner['source_shard'];old=summary['shards'][sid]
    if old.get('completed_answers',0) or old.get('current_workers'):
        raise ValueError('Delegated secondary queue also has original-region activity')
    ids={p['id'] for p in pods if p.get('mounts',{}).get('network')==[{'volumeId':owner['destination_volume_id'],'path':'/workspace'}]}
    values=[r['snapshot']['delegated_secondary'] for r in snapshots if r['pod_id'] in ids
            and r.get('snapshot',{}).get('delegated_secondary') is not None]
    if not values:
        summary['shards'][sid]={'readable':False,'task_count':owner['task_count'],
            'reason':'Delegated secondary destination has no readable live host; inspect durable records.'}
    else:
        d=max(values,key=lambda x:x['epoch'])
        summary['shards'][sid]={'readable':True,'task_count':owner['task_count'],'epoch':d['epoch'],
            'network_volume_id':owner['destination_volume_id'],'execution_location':'US-CO-1',
            'result_root':owner['destination_root'],
            **{k:d[k] for k in ['completed_answers','answers_by_condition','jobs_completed','ready_primary_controls']},
            'current_workers':[r for r in d['workers'] if r.get('pod_id') in ids],
            'current_worker_failures':[r for r in d['failures'] if any('/workers/'+pid+'_secondary_' in r['path'] for pid in ids)]}
    complete=all(x['readable'] for x in summary['shards'].values())
    summary.update(all_shard_stores_readable=complete,coverage_is_partial_observation=not complete)
    summary['totals']={'completed_answers':sum(x.get('completed_answers',0) for x in summary['shards'].values()),
        'answers_by_condition':{c:sum(x.get('answers_by_condition',{}).get(c,0) for x in summary['shards'].values()) for c in SECONDARY_CONDITIONS}}
    return summary


def main():
    partition_path=ROOT/'configs/coding_pilot_v1/confirmation_01/regional_partition.json'
    partition=json.loads(partition_path.read_text())
    rows=[p for p in inventory() if p.get('status')=='RUNNING' and p.get('name','').startswith('gearshift-confirmation-')]
    pods=[base.safe(base.api('pods/'+p['id'])) for p in rows]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:snapshots=list(ex.map(collect,pods))
    now=time.time();observed=datetime.datetime.fromtimestamp(now,datetime.timezone.utc)
    closed=[json.loads(p.read_text()) for p in base.EVIDENCE.glob('*/final_compute_estimate.json')]
    if {r['pod_id'] for r in closed}&{p['id'] for p in pods}:raise ValueError('Released estimate still names a live pod')
    compute=sum(max(0,now-datetime.datetime.fromisoformat(p['createdAt'].replace('Z','+00:00')).timestamp())/3600*p['cost'] for p in pods)+sum(r['compute_estimate_usd'] for r in closed)+0.31019184393141
    secondary=summarize(partition,pods,snapshots,'secondary')
    owner_path=ROOT/'configs/coding_pilot_v1/confirmation_01/secondary_fr_execution_owner_v1.json'
    if owner_path.exists():secondary=delegated_summary(secondary,pods,snapshots,json.loads(owner_path.read_text()))
    completed={}
    for sid in partition['shards']:
        value=completed_primary(sid,partition,base.sha256_file(partition_path),base.EVIDENCE)
        if value is not None:completed[sid]=value
    record={'observed_at_utc':observed.isoformat(),'experiment_id':base.EXPERIMENT,'partition_path':str(partition_path.relative_to(ROOT)),
        'pods':pods,'snapshots':snapshots,'primary':summarize(partition,pods,snapshots,completed=completed),
        'secondary_evaluation':secondary,
        'cost':{'conservative_cumulative_estimate_usd':base.BASELINE+compute,'current_compute_usd_per_hour':sum(p['cost'] for p in pods),
                'completed_compute_estimates':closed,'old_released_scoring_cpu_estimate_usd':0.31019184393141,'new_unposted_storage_excluded':True}}
    path=base.EVIDENCE/('regional_live_check_'+observed.strftime('%Y%m%dT%H%M%SZ')+'.json');base.atomic_json(path,record)
    training=next((r.get('snapshot',{}).get('secondary_training_workers',[]) for r in snapshots if r['pod_id']=='rekbqruqox2bk9'),[])
    local=base.EVIDENCE/'secondary_final_checkpoints/verified_local.json'
    print(json.dumps({'saved':str(path),'observed_at_utc':record['observed_at_utc'],'primary':record['primary'],
        'secondary_evaluation':record['secondary_evaluation'],'secondary_training':training,
        'secondary_final_checkpoint_receipt':str(local) if local.is_file() else None,'cost':record['cost']},indent=2))
if __name__=='__main__':main()
