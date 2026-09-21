#!/usr/bin/env python3
"""One saved failing stream; no fleet retry, no evaluation/confirmation."""
import argparse,gc,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.coding_control import write,digest,sha
from gearshift.coding_training import read,finish_worker,paired_samples,atomic_tensor
from gearshift.coding_recovery import worker_context,initialize_backends,reason_instrumented
from gearshift.core import CacheExtractor
ROOT=Path(__file__).resolve().parents[1]

def run(c):
    spec=c['spec'];t=c['telemetry'];folder=c['root']/'trajectory'
    if spec['stage']!='recovery_probe' or len(spec['task_ids'])!=1:raise ValueError('Exactly one saved failure required')
    old=ROOT/spec['saved_prefix'];partial=read(old)
    if sha(old)!=spec['saved_prefix_sha256'] or partial['task_id']!=spec['task_ids'][0]:raise ValueError('Saved trajectory changed')
    task=next(x for x in read(ROOT/spec['cohort_file']) if x['task_id']==partial['task_id'])
    source,receiver=initialize_backends(c)
    h,cache=reason_instrumented(source,task['prompt_ids'],task['task_id'],'source_reasoning',24576,folder,t,c['guard'],c['publish'],partial['tokens'])
    c['publish'](stage='receiver_extraction');t.sample(stage='before_receiver_extraction',sequence_length=len(h['prefix_ids']))
    with torch.no_grad():
        sp=CacheExtractor.tensors(cache);native=receiver.prefill_chunked(h['prefix_ids']);tp=CacheExtractor.tensors(native.past_key_values)
        sampled=paired_samples(source,receiver,sp,tp)
    t.sample(stage='after_paired_sampling',sequence_length=len(h['prefix_ids']))
    del sampled,sp,tp,native,cache;gc.collect();torch.cuda.empty_cache()
    t.sample(stage='after_release',sequence_length=len(h['prefix_ids']))
    write(c['root']/'complete.json',{'identity_sha256':digest(c['identity']),'task_id':task['task_id'],'saved_prefix_sha256':sha(old),
        'saved_prefix_tokens':len(partial['tokens']),'saved_prefix_exact':True,'source_history_sha256':sha(folder/'source_history.json'),
        'reasoning_tokens':len(h['reasoning_ids']),'reasoning_capped':h['reasoning_capped'],'paired_extraction_completed':True,
        'memory_policy':'logged warnings; no task-quality claim','new_corpus_inclusion':False})
    finish_worker(c,'complete',stage='recovery_probe_complete')

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker-spec');a=p.parse_args();c=None
    try:c=worker_context(ROOT,a.worker_spec);run(c)
    except BaseException as exc:
        if c:
            c['telemetry'].failure(exc);write(c['root']/'failure.json',{'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
            finish_worker(c,'failed',exception=type(exc).__name__,error=str(exc))
        raise
if __name__=='__main__':main()
