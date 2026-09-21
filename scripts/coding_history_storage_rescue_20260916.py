#!/usr/bin/env python3
"""One bounded CPU-only forensic pass over the five retained history volumes."""
import json,os,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from gearshift.coding_control import PREFIX,write,sha,decision
from coding_cloud_guard import cli,tick
from coding_parallel_session import remote,pod_detail_rest,sanitized_pod,controller_heartbeat,collect,copy_checkpoints
C=ROOT/'evidence/coding_pilot_v1/control';E=ROOT/'evidence/coding_pilot_v1/history_interruption_20260916'
IMAGE='runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'
PLAN=json.loads((E/'recovery_targets.json').read_text());PARENT=PLAN['parent_run']

def inspect(target):
    worker=target['worker_id'];volume=target['volume_id'];region=target['region']
    assert volume!='jj2zyi9yrc' and target['name'].startswith(PREFIX+PARENT+'-'+worker+'-')
    run='history_rescue_20260916_'+worker;name=PREFIX+run+'-cpu';d=E/'cpu'/worker;d.mkdir(parents=True,exist_ok=False)
    started=time.time();deadline=started+1800;pod=None;verified=None;halt=threading.Event()
    assert not [p for p in cli('pod','list') if p.get('name','').startswith(PREFIX)]
    actual=[v for v in cli('network-volume','list') if v['id']==volume]
    assert len(actual)==1 and actual[0]['name']==target['name'] and actual[0]['dataCenterId']==region
    intent={'purpose':'storage_inspection','started_epoch':started,'deadline_epoch':deadline,'controller_id':run,
        'resource_name':name,'volume_id':volume,'maximum_cpu_pods':1,'gpu_count':0,'maximum_hourly_usd':.50,
        'read_only':True,'model_inference_allowed':False,'approval_sha256':sha(C/'parallel_500h_approved.json')}
    old=C/'storage_inspection_intent.json'
    if old.exists():shutil.copy2(old,d/'previous_inspection_intent.json')
    write(old,intent);intent_sha=sha(old);write(d/'intent.json',intent)
    write(C/'allocation_intents'/(run+'.json'),{'controller_id':run,'started_epoch':started,'resource_names':[name],'gpu_count':0})
    vp=C/'resource_receipts'/(volume+'.json');vr=json.loads(vp.read_text());write(d/'prior_volume_receipt.json',vr);vr['controller_id']=run;write(vp,vr)
    def pulse():
        while not halt.is_set():controller_heartbeat(run,'recovering_history_evidence');halt.wait(10)
    thread=threading.Thread(target=pulse,daemon=True);thread.start()
    try:
        tick();budget=decision(json.loads((C/'ledger.json').read_text()));assert not budget['stop'] and budget['upper_usd']+.27<980
        pod=cli('pod','create','--name',name,'--image',IMAGE,'--compute-type','CPU','--cloud-type','SECURE','--data-center-ids',region,'--container-disk-in-gb','20','--network-volume-id',volume,'--ports','22/tcp','--ssh')
        write(C/'resource_receipts'/(pod['id']+'.json'),{'kind':'pod','id':pod['id'],'name':name,'started_epoch':time.time(),'upper_rate_usd':.50,'controller_id':run,'gpu_count':0,'volume_id':volume,'storage_inspection_intent_sha256':intent_sha})
        detail=pod_detail_rest(pod['id']);write(d/'pod_verified.json',sanitized_pod(detail))
        assert detail['id']==pod['id'] and detail['name']==name and detail['imageName']==IMAGE
        assert detail.get('gpuCount') in [None,0] and detail.get('vcpuCount',0)>0
        assert detail.get('cpuFlavorId') in ['cpu3c','cpu3g','cpu3m','cpu5c','cpu5g','cpu5m']
        assert (detail.get('machine') or {}).get('gpuTypeId') in [None,'unknown']
        assert 0<float(detail.get('costPerHr',0))<=.50
        assert (detail.get('networkVolumeId') or (detail.get('networkVolume') or {}).get('id'))==volume
        conn=None
        for _ in range(40):
            if time.time()>deadline-600:break
            try:
                info=cli('ssh','info',pod['id']);candidate=info.get('connection',info)
                if candidate.get('ip'):remote(candidate,'true',20);conn=candidate;break
            except Exception:pass
            time.sleep(10)
        if conn is None:raise TimeoutError('CPU recovery connection did not become ready')
        write(d/'connection.json',conn)
        spec=json.loads((ROOT/f'evidence/coding_pilot_v1/parallel/{PARENT}/{worker}/worker_spec.json').read_text())
        dest=E/'forensic'/worker;saved=collect(conn,dest,spec)
        if time.time()>=deadline-300:raise TimeoutError('Insufficient CPU recovery time for bounded feature copies')
        features=copy_checkpoints(conn,saved,dest,spec)
        archive=dest/'latest.tar.gz';second=dest/'final.tar.gz';shutil.copy2(archive,second)
        assert sha(archive)==sha(second)
        verified={'epoch':time.time(),'verified':True,'both_copies_verified':True,'pod_id':pod['id'],'volume_id':volume,
            'copies':[str(p.relative_to(ROOT)) for p in [archive,second]],'sha256':sha(archive),'archive_sha256':sha(archive),
            'mapper_checkpoints':features,'stage_identity':spec['stage_identity'],'worker_id':worker,
            'completed_histories':len(list((saved/spec['result_root']).glob('tasks/*/complete.json')))}
        write(dest/'recovery_receipt.json',verified)
        write(d/'status.json',{'state':'recovered','epoch':time.time(),'completed_histories':verified['completed_histories']})
    except BaseException as exc:
        write(d/'error.json',{'epoch':time.time(),'error':str(exc),'traceback':traceback.format_exc()});raise
    finally:
        if pod:
            try:cli('pod','delete',pod['id'])
            except Exception as exc:write(d/'cleanup_error.json',{'error':str(exc)})
        absent=not pod or not any(p['id']==pod['id'] for p in cli('pod','list'))
        if verified and absent:
            verified['worker_confirmed_absent']=True;write(E/'forensic'/worker/'recovery_receipt.json',verified)
            write(C/'backups_verified'/(volume+'.json'),verified)
            # This volume belongs to this interrupted history attempt, not the preserved memory attempt.
            if any(v['id']==volume for v in cli('network-volume','list')):cli('network-volume','delete',volume)
        halt.set();thread.join(timeout=15)
        write(d/'controller_complete.json',{'epoch':time.time(),'cpu_confirmed_absent':absent,'recovery_verified':bool(verified)})
        tick()
    return verified


def main():
    assert len(PLAN['targets'])==5 and PLAN['maximum_gpu_hours']==0
    assert not (E/'cpu_batch_complete.json').exists()
    rows=[]
    for target in PLAN['targets']:
        try:rows.append({'worker_id':target['worker_id'],'passed':True,'receipt':inspect(target)})
        except Exception as exc:
            rows.append({'worker_id':target['worker_id'],'passed':False,'error':str(exc)})
            if [p for p in cli('pod','list') if p.get('name','').startswith(PREFIX)]:break
        write(E/'cpu_batch_progress.json',{'epoch':time.time(),'workers':rows})
    write(E/'cpu_batch_complete.json',{'epoch':time.time(),'passed':len(rows)==5 and all(r['passed'] for r in rows),'workers':rows})

if __name__=='__main__':main()
