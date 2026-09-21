#!/usr/bin/env python3
"""Fixed-cohort development selection and confirmation with pure mapped-cache C.

All training selection is frozen before confirmation. A/C/D share one actual
source-generation cache per task, with distinct identical C/D answer generators.
"""
import argparse,gc,json,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.coding_control import bind,digest,sha,write
from gearshift.coding_training import (worker_context,initialize_backends,finish_worker,read,safe_id,
    choose_development_candidate)
from gearshift.coding_gradients import AffineMapper
from gearshift.coding_resume import verify_prefix
from gearshift.core import CacheExtractor,CacheInjector
ROOT=Path(__file__).resolve().parents[1]


def verified_candidates(spec):
    candidates=spec['candidates']
    if not 1<=len(candidates)<=4 or len({c['candidate_id'] for c in candidates})!=len(candidates):raise ValueError('At most four distinct candidates are permitted')
    counts={};result=[];initializations=set()
    for c in candidates:
        root=Path(c['training_root']);identity=read(root/'identity.json');complete=read(root/'complete.json')
        if identity['config_sha256']!=sha(ROOT/'configs/coding_pilot_v1/pilot.json') or identity['training_protocol_sha256']!=sha(ROOT/'configs/coding_pilot_v1/training_protocol_v1.json'):raise ValueError('Candidate training protocol differs')
        for name in ['gearshift/core.py','gearshift/coding_inference.py','gearshift/coding_gradients.py','gearshift/coding_training.py']:
            if identity['implementation'][name]!=sha(ROOT/name):raise ValueError('Candidate numerical implementation differs')
        if complete['identity_sha256']!=digest(identity):raise ValueError('Training completion identity differs')
        if sha(root/'complete.json')!=c['training_complete_sha256']:raise ValueError('Training completion changed')
        nominee=next((r for r in complete['nominees'] if r['checkpoint']['sha256']==c['checkpoint_sha256']),None)
        if nominee is None or nominee['step']<256:raise ValueError('Checkpoint was not nominated by validation')
        path=Path(c['checkpoint'])
        if path.resolve()!=(root/nominee['checkpoint']['path']).resolve() or sha(path)!=c['checkpoint_sha256']:raise ValueError('Candidate checkpoint differs')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['identity_sha256']!=digest(identity) or checkpoint['step']!=nominee['step'] or checkpoint['seed']!=identity['seed'] or checkpoint['objective']!=identity['objective']:raise ValueError('Checkpoint payload identity differs')
        counts[identity['objective']]=counts.get(identity['objective'],0)+1
        if counts[identity['objective']]>2:raise ValueError('At most two nominees per objective')
        initializations.add(identity['initialization_sha256'])
        result.append({**c,'objective':identity['objective'],'seed':identity['seed'],'step':nominee['step'],
            'initialization_sha256':identity['initialization_sha256'],'state_dict':checkpoint['state_dict']})
    if len(initializations)!=1:raise ValueError('Candidate initializations differ')
    return result


def validate_stage_membership(spec,cfg,tasks,candidates):
    stage=spec['stage'];ids=spec['task_ids'];full={t['task_id'] for t in tasks}
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=full:raise ValueError('Shard task membership changed')
    if stage=='development_candidates':
        if any(c['seed']!=20260915 for c in candidates):raise ValueError('Development comparison uses first seed only')
        return None
    if stage not in ['confirmation','second_seed']:raise ValueError('Unknown evaluation stage')
    selection_path=Path(spec['selection_path']);selection=read(selection_path)
    if sha(selection_path)!=spec['selection_sha256'] or selection.get('confirmation_used_for_selection') is not False:
        raise ValueError('Frozen preconfirmation selection required')
    if len(candidates)!=1:raise ValueError('Confirmation runs exactly one nominated candidate')
    c=candidates[0]
    if stage=='confirmation':
        if c['seed']!=20260915 or c['checkpoint_sha256']!=selection['checkpoint_sha256']:raise ValueError('Headline checkpoint differs from frozen development selection')
    else:
        if c['seed']!=20260916 or c['objective']!=selection['objective']:raise ValueError('Second seed must replicate the selected recipe')
        training=read(Path(c['training_root'])/'complete.json')
        if c['checkpoint_sha256']!=training['nominees'][0]['checkpoint']['sha256']:raise ValueError('Second seed uses first common-validation nominee, never confirmation outcomes')
        reserved={r['platform']+'/'+r['question_id'] for r in cfg['mapper']['second_seed_confirmation_ids']}
        if len(reserved)!=40 or not set(ids)<=reserved:raise ValueError('Second-seed shard exceeds fixed reserved subset')
    return selection


def verify_replayed_history(history,saved):
    verify_prefix(history['reasoning_ids'],saved['reasoning_ids'],final=True)
    keys=['prompt_ids','reasoning_ids','prefix_ids','bridge_ids','natural_boundary','reasoning_capped','early_eos','rng_initial','rng_after_reasoning']
    if any(history[k]!=saved[k] for k in keys):raise ValueError('Exact source trajectory replay changed')


def run(c,backends=None):
    from gearshift.coding_inference import reason,answer,sync,memory_record
    from gearshift.coding_sandbox import extract,score
    from gearshift.coding_reuse import verify_completed
    spec=c['spec'];root=c['root'];guard=c['guard'];publish=c['publish'];cfg=c['config']
    split='development' if spec['stage']=='development_candidates' else 'confirmation'
    tasks=read(ROOT/f'data/coding_pilot_v1/visible/{split}.json');lookup={t['task_id']:t for t in tasks}
    candidates=verified_candidates(spec);selection=validate_stage_membership(spec,cfg,tasks,candidates)
    candidate_receipt=[{k:v for k,v in row.items() if k!='state_dict'} for row in candidates]
    write(root/'candidate_inputs.json',{'candidates':candidate_receipt,'selection_sha256':spec.get('selection_sha256')})
    sandbox_path=Path(spec.get('sandbox_gate',ROOT/'evidence/coding_pilot_v1/sandbox_gate.json'))
    sandbox=read(sandbox_path)
    if not sandbox['passed'] or len(sandbox.get('canonical_development',{}))!=2:raise ValueError('Verified Linux sandbox required')
    write(root/'sandbox_input.json',{'sha256':sha(sandbox_path)})
    source,receiver=backends or initialize_backends(c)
    mapper=AffineMapper(source,receiver);mapper.requires_grad_(False)
    baseline=Path(spec['baseline_root']);baseline_identity=read(baseline/'identity.json');cap=baseline_identity['reasoning_cap']
    if cap not in [16384,24576]:raise ValueError('Reasoning cap changed')
    # Trusted checker material is loaded only in evaluation; never passed to LMs.
    private=read(ROOT/f'data/coding_pilot_v1/private/{split}.json')
    headline={}
    if spec['stage']=='second_seed':
        for parent_root in map(Path,spec['headline_roots']):
            parent_identity=read(parent_root/'identity.json');parent_done=read(parent_root/'complete.json')
            if parent_done['stage']!='confirmation' or parent_done['identity_sha256']!=digest(parent_identity) or parent_done['selection_sha256']!=spec['selection_sha256']:raise ValueError('Second-seed headline lineage differs')
            for task_id,receipt_sha in parent_done['tasks'].items():
                parent_folder=parent_root/'tasks'/safe_id(task_id);receipt=read(parent_folder/'complete.json')
                if sha(parent_folder/'complete.json')!=receipt_sha or receipt['identity_sha256']!=digest(parent_identity):raise ValueError('Headline task transaction differs')
                if sha(parent_folder/'source_history.json')!=receipt['files']['source_history.json']:raise ValueError('Headline source history changed')
                if task_id in headline:raise ValueError('Duplicate headline task')
                headline[task_id]=read(parent_folder/'source_history.json')
        if not set(spec['task_ids'])<=set(headline):raise ValueError('Second-seed exact headline source histories required')
    completed={};started=time.monotonic()
    def graded(folder,name,row):
        row['code']=extract(row['answer_text']);write(folder/(name+'.json'),row)
        row['score']=score(row['code'],private[row['task_id']],guard)
        write(folder/(name+'.json'),row)
        if row['score']['category'] in ['sandbox_unavailable','missing_tests']:raise RuntimeError('Trusted evaluation unavailable')
        return row
    for index,tid in enumerate(spec['task_ids']):
        guard();task=lookup[tid];folder=root/'tasks'/safe_id(tid);folder.mkdir(parents=True,exist_ok=False)
        saved=None
        if split=='development':
            parent=baseline/'tasks'/safe_id(tid);verify_completed(parent,baseline_identity,tid);saved=read(parent/'source_history.json')
        elif spec['stage']=='second_seed':saved=headline[tid]
        publish(stage=spec['stage'],task_id=tid,completed_tasks=index,total_tasks=len(spec['task_ids']),segment='source_reasoning')
        def callback(segment,previous=None):
            def save(tokens):
                guard()
                if previous is not None:verify_prefix(tokens,previous)
                write(folder/('partial_'+segment+'.json'),{'task_id':tid,'segment':segment,'token_ids':tokens,'identity_sha256':digest(c['identity'])})
                publish(segment=segment,generated_tokens=len(tokens))
            return save
        history,cache=reason(source,task['prompt_ids'],tid,'source_reasoning',cap,
            callback=callback('source_reasoning',saved['reasoning_ids'] if saved else None))
        if saved:verify_replayed_history(history,saved)
        write(folder/'source_history.json',history);source_pairs=CacheExtractor.tensors(cache);del cache
        rows={};c_rows={}
        if spec['stage']=='confirmation':
            a=answer(source,history,CacheInjector.create(source_pairs,clone=False),tid,'answer_large',4096,callback=callback('answer_A'))
            a.update(condition='large_only',source_reasoning_seconds=history['reasoning_seconds'],historical_receiver_prefill_tokens=0,
                source_inclusive_seconds=history['reasoning_seconds']+a['answer_seconds'])
            rows['A']=graded(folder,'A',a)
        for candidate in candidates:
            guard();publish(segment='mapped_answer',candidate_id=candidate['candidate_id'])
            mapper.load_state_dict(candidate['state_dict'],strict=True)
            before_tokens=receiver.input_token_count;sync();before=time.monotonic()
            with torch.no_grad():mapped=mapper(source_pairs)
            sync();mapping_seconds=time.monotonic()-before
            if receiver.input_token_count!=before_tokens:raise AssertionError('Pure mapped-cache arm replayed historical receiver tokens')
            cc=answer(receiver,history,CacheInjector.create(mapped,clone=False),tid,'answer_small',4096,
                callback=callback('answer_C_'+safe_id(candidate['candidate_id'])))
            del mapped;gc.collect();torch.cuda.empty_cache()
            cc.update(condition='gearshift',candidate_id=candidate['candidate_id'],checkpoint_sha256=candidate['checkpoint_sha256'],
                source_reasoning_seconds=history['reasoning_seconds'],mapping_seconds=mapping_seconds,historical_receiver_prefill_tokens=0,
                source_inclusive_seconds=history['reasoning_seconds']+mapping_seconds+cc['answer_seconds'])
            name='C' if len(candidates)==1 else 'C_'+safe_id(candidate['candidate_id'])
            c_rows[candidate['candidate_id']]=graded(folder,name,cc)
        del source_pairs;gc.collect();torch.cuda.empty_cache();guard()
        if spec['stage']=='confirmation':
            publish(segment='text_handoff');sync();before=time.monotonic()
            native=receiver.prefill_chunked(history['prefix_ids']);sync();prefill=time.monotonic()-before
            d=answer(receiver,history,native.past_key_values,tid,'answer_small',4096,callback=callback('answer_D'));del native
            d.update(condition='text_handoff',source_reasoning_seconds=history['reasoning_seconds'],native_prefill_seconds=prefill,
                historical_receiver_prefill_tokens=len(history['prefix_ids']),source_inclusive_seconds=history['reasoning_seconds']+prefill+d['answer_seconds'])
            rows['D']=graded(folder,'D',d)
            publish(segment='small_reasoning')
            sh,small_cache=reason(receiver,task['prompt_ids'],tid,'small_reasoning',cap,callback=callback('small_reasoning'))
            write(folder/'small_history.json',sh)
            b=answer(receiver,sh,small_cache,tid,'answer_small',4096,callback=callback('answer_B'));del small_cache
            b.update(condition='small_only',own_reasoning_seconds=sh['reasoning_seconds'],historical_receiver_prefill_tokens=0,
                source_inclusive_seconds=sh['reasoning_seconds']+b['answer_seconds'])
            rows['B']=graded(folder,'B',b)
        guard();memory=memory_record()
        transaction={'task_id':tid,'identity_sha256':digest(c['identity']),'source_capped':history['reasoning_capped'],
            'source_early_eos':history['early_eos'],'candidates':{key:{'passed':r['score']['passed'],'source_inclusive_seconds':r['source_inclusive_seconds'],
                'checkpoint_sha256':r['checkpoint_sha256']} for key,r in c_rows.items()},
            'pass':{arm:r['score']['passed'] for arm,r in rows.items()},'memory':memory,
            'files':{p.name:sha(p) for p in folder.glob('*.json') if p.name!='complete.json'}}
        if spec['stage']=='confirmation':
            transaction['pass']['C']=next(iter(c_rows.values()))['score']['passed'];transaction['small_capped']=sh['reasoning_capped'];transaction['small_early_eos']=sh['early_eos']
        write(folder/'complete.json',transaction);completed[tid]=sha(folder/'complete.json')
        write(root/'progress.json',{'completed_tasks':len(completed),'total_tasks':len(spec['task_ids']),'wall_seconds':time.monotonic()-started})
    done={'identity_sha256':digest(c['identity']),'stage':spec['stage'],'tasks':completed,'candidates':candidate_receipt,'task_count':len(completed),
        'selection_sha256':spec.get('selection_sha256'),'wall_seconds':time.monotonic()-started}
    write(root/'complete.json',done);finish_worker(c,'complete',stage=spec['stage'],completed_tasks=len(completed));return done


def select_development(roots,destination,expected_task_ids):
    """Immutable task-quality selection; confirmation files are never consulted."""
    destination=Path(destination)
    if destination.exists():raise ValueError('Selection is already frozen')
    seen={};descriptors={};sources={}
    for root in map(Path,roots):
        identity=read(root/'identity.json');done=read(root/'complete.json');native=read(root/'native_gate.json')
        if done['stage']!='development_candidates' or done['identity_sha256']!=digest(identity) or not native['passed'] or native['identity_sha256']!=digest(identity):raise ValueError('Invalid development shard/native gate')
        for candidate in done['candidates']:
            cid=candidate['candidate_id']
            if cid in descriptors and descriptors[cid]!=candidate:raise ValueError('Candidate descriptor differs between shards')
            descriptors[cid]=candidate
        for tid,want in done['tasks'].items():
            folder=root/'tasks'/safe_id(tid);row=read(folder/'complete.json')
            if sha(folder/'complete.json')!=want or row['identity_sha256']!=digest(identity):raise ValueError('Development transaction changed')
            for name,value in row['files'].items():
                if Path(name).name!=name or sha(folder/name)!=value:raise ValueError('Development output changed')
            for cid,result in row['candidates'].items():
                if (cid,tid) in seen:raise ValueError('Duplicate candidate task observation')
                if result['checkpoint_sha256']!=descriptors[cid]['checkpoint_sha256']:raise ValueError('Development checkpoint differs')
                seen[cid,tid]=result
        sources[str(root)]=sha(root/'complete.json')
    rows=[];expected=set(expected_task_ids)
    if len(expected)!=40:raise ValueError('Exactly forty fixed development tasks required')
    for cid,descriptor in descriptors.items():
        actual={tid for candidate,tid in seen if candidate==cid}
        if actual!=expected:raise ValueError('Incomplete fixed candidate development cohort')
        records=[seen[cid,tid] for tid in sorted(expected)]
        rows.append({'candidate_id':cid,'passes':sum(r['passed'] for r in records),'source_inclusive_seconds':sum(r['source_inclusive_seconds'] for r in records),
            'step':descriptor['step'],'task_count':len(records)})
    selected=choose_development_candidate(rows);descriptor=descriptors[selected['candidate_id']]
    receipt={**descriptor,'selected_candidate_id':selected['candidate_id'],'development_comparisons':rows,'source_shards':sources,
        'confirmation_used_for_selection':False,'selection_rule':'Maximum full40 passes, lower source-inclusive latency, earlier step; headline seed20260915',
        'frozen_epoch':time.time()}
    write(destination,receipt);return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker-spec');parser.add_argument('--select-development',nargs='+')
    parser.add_argument('--selection-output');args=parser.parse_args();context=None
    try:
        if args.select_development:
            if not args.selection_output:raise ValueError('Selection destination required')
            tasks=read(ROOT/'data/coding_pilot_v1/visible/development.json')
            select_development(args.select_development,args.selection_output,[t['task_id'] for t in tasks]);return
        context=worker_context(ROOT,args.worker_spec)
        # The evaluation implementation is explicitly bound in addition to shared code.
        bind(context['root']/'evaluation_code.json',{'scripts/coding_evaluation_worker.py':sha(Path(__file__))})
        run(context)
    except BaseException as exc:
        if context:finish_worker(context,'failed',exception=type(exc).__name__,error=str(exc));write(context['root']/'failure.json',{'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
