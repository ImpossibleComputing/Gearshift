#!/usr/bin/env python3
"""One immutable shard of training/validation source trajectories; no hidden tests."""
import argparse,gc,json,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import digest,sha,write
from gearshift.coding_training import (worker_context,initialize_backends,finish_worker,read,safe_id,
    paired_samples,atomic_tensor,validate_history)
ROOT=Path(__file__).resolve().parents[1]


def run(context,backends=None):
    import torch
    from gearshift.coding_inference import reason,answer
    from gearshift.core import CacheExtractor,CacheInjector
    c=context;spec=c['spec'];root=c['root'];guard=c['guard'];publish=c['publish']
    if spec['stage']!='histories':raise ValueError('History worker stage differs')
    tasks={}
    for split in ['training','validation']:
        for row in read(ROOT/f'data/coding_pilot_v1/visible/{split}.json'):
            tasks[row['task_id']]={'task_id':row['task_id'],'prompt_ids':row['prompt_ids'],'split':split}
    task_ids=spec['task_ids']
    if len(task_ids)!=len(set(task_ids)) or not set(task_ids)<=set(tasks):raise ValueError('History shard changes fixed membership')
    source,receiver=backends or initialize_backends(c)
    baseline_identity=read(Path(spec['baseline_root'])/'identity.json');cap=baseline_identity['reasoning_cap']
    if cap not in [16384,24576]:raise ValueError('Undeclared reasoning cap')
    completed={};feature_files={};started=time.monotonic()
    for index,tid in enumerate(task_ids):
        guard();task=tasks[tid];folder=root/'tasks'/safe_id(tid);folder.mkdir(parents=True,exist_ok=False)
        publish(stage='source_reasoning',task_id=tid,completed_tasks=index,total_tasks=len(task_ids))
        def partial(tokens):
            guard();write(folder/'inflight_source_reasoning.json',{'task_id':tid,'tokens':tokens,'identity_sha256':digest(c['identity'])})
        history,source_cache=reason(source,task['prompt_ids'],tid,'source_reasoning',cap,callback=partial)
        write(folder/'source_history.json',history)
        # Source features are extracted before answer generation mutates the cache.
        source_pairs=CacheExtractor.tensors(source_cache)
        publish(stage='paired_prefix',task_id=tid)
        with torch.no_grad():native=receiver.prefill_chunked(history['prefix_ids'])
        target_pairs=CacheExtractor.tensors(native.past_key_values)
        if task['split']=='training':
            with torch.no_grad():features=paired_samples(source,receiver,source_pairs,target_pairs)
            feature_sha=atomic_tensor(folder/'paired_features.pt',features);del features
        del target_pairs,native,source_pairs;gc.collect();torch.cuda.empty_cache();guard()
        publish(stage='source_answer',task_id=tid)
        def partial_answer(tokens):
            guard();write(folder/'inflight_teacher_answer.json',{'task_id':tid,'tokens':tokens,'identity_sha256':digest(c['identity'])})
        teacher=answer(source,history,source_cache,tid,'answer_large',4096,callback=partial_answer)
        teacher['provenance']='Native frozen source A, answer_large stream, one sampled answer; no correctness filtering.'
        write(folder/'teacher_answer.json',teacher);del source_cache;gc.collect();torch.cuda.empty_cache();guard()
        obj={'task_id':tid,'split':task['split'],'source_history':history,'teacher_answer':teacher};validate_history(obj)
        files={name:sha(folder/name) for name in ['source_history.json','teacher_answer.json']}
        extra_features={}
        if task['split']=='training':
            extra_features={'paired_features.pt':feature_sha}
            feature_files[str((folder/'paired_features.pt').relative_to(ROOT))]=feature_sha
            write(root/'feature_manifest.json',{'files':feature_files})
        write(folder/'complete.json',{'task_id':tid,'split':task['split'],'identity_sha256':digest(c['identity']),'files':files,'feature_files':extra_features,
            'source_capped':history['reasoning_capped'],'answer_capped':teacher['answer_capped'],
            'source_reasoning_tokens':len(history['reasoning_ids']),'answer_tokens':len(teacher['answer_ids'])})
        completed[tid]=sha(folder/'complete.json')
        write(root/'progress.json',{'completed_tasks':len(completed),'total_tasks':len(task_ids),'wall_seconds':time.monotonic()-started})
    done={'identity_sha256':digest(c['identity']),'tasks':completed,'task_count':len(completed),'teacher':'source_native_A','wall_seconds':time.monotonic()-started}
    write(root/'complete.json',done);finish_worker(c,'complete',stage='histories',completed_tasks=len(completed),total_tasks=len(task_ids))
    return done


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker-spec');args=parser.parse_args();context=None
    try:
        context=worker_context(ROOT,args.worker_spec);run(context)
    except BaseException as exc:
        if context:finish_worker(context,'failed',exception=type(exc).__name__,error=str(exc));write(context['root']/'failure.json',{'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
