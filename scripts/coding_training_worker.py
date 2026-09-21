#!/usr/bin/env python3
"""Fresh ridge initialization or one bounded, gate-verified mapper training run."""
import argparse,gc,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.coding_control import bind,digest,sha,write
from gearshift.coding_gradients import AffineMapper
from gearshift.coding_training import (worker_context,initialize_backends,finish_worker,verify_history_roots,
    fit_ridge,atomic_tensor,read,checkpoint_manifest,TrainingRuntime,DiskLRU,train_bounded,OBJECTIVES)
ROOT=Path(__file__).resolve().parents[1]


def run(c,backends=None):
    spec=c['spec'];root=c['root'];guard=c['guard'];publish=c['publish']
    if spec['stage'] not in ['initialization','training']:raise ValueError('Unsupported training worker stage')
    histories,features,receipts=verify_history_roots(ROOT,spec['history_roots'])
    training=[o for o in histories if o['split']=='training'];validation=[o for o in histories if o['split']=='validation']
    if len(training)!=128 or len(validation)!=32:raise ValueError('Training/validation split changed')
    write(root/'history_inputs.json',{'receipts':receipts,'training_tasks':128,'validation_tasks':32})
    source,receiver=backends or initialize_backends(c);guard()
    # Every run starts at the identical fitted weights, irrespective of run seed.
    mapper=AffineMapper(source,receiver,seed=20260915)
    if spec['stage']=='initialization':
        rows=fit_ridge(mapper,features,guard=guard,progress=publish)
        init=root/'mapper_initialization.pt'
        checkpoint={'state_dict':{k:v.detach().cpu() for k,v in mapper.state_dict().items()},'identity_sha256':digest(c['identity']),
            'history_receipts_sha256':digest(receipts),'ridge':.01,'training_task_count':128,'paired_positions_per_history_max':64,
            'fresh_source_specific':True,'source_layers':mapper.sources}
        atomic_tensor(init,checkpoint);checkpoint_manifest(ROOT,root)
        done={'identity_sha256':digest(c['identity']),'initialization_sha256':sha(init),'checkpoint':str(init),'fit':rows}
        write(root/'complete.json',done);finish_worker(c,'complete',stage='initialization');return done
    objective=spec['objective'];seed=spec['seed']
    if objective not in OBJECTIVES or seed not in [20260915,20260916]:raise ValueError('Unapproved objective or seed')
    if seed==20260916:
        selection=read(spec['selection_path'])
        if sha(spec['selection_path'])!=spec['selection_sha256'] or selection['objective']!=objective or selection.get('confirmation_used_for_selection') is not False:raise ValueError('Second seed requires frozen development recipe')
        if selection.get('seed')!=20260915 or selection.get('initialization_sha256')!=spec['initialization_sha256']:raise ValueError('Second seed must reuse first-seed initialization and selected objective')
        if not 1<=len(selection.get('development_comparisons',[]))<=4 or any(r.get('task_count')!=40 for r in selection['development_comparisons']):raise ValueError('Development selection is incomplete')
        if spec.get('task_ids'):raise ValueError('Second-seed training cannot consume confirmation task identifiers')
    init=Path(spec['initialization']);checkpoint=torch.load(init,map_location='cpu',weights_only=True)
    if sha(init)!=spec['initialization_sha256'] or checkpoint.get('ridge')!=.01 or not checkpoint.get('fresh_source_specific'):raise ValueError('Initialization differs')
    if checkpoint['history_receipts_sha256']!=digest(receipts):raise ValueError('Initialization history corpus changed')
    if checkpoint['source_layers']!=mapper.sources:raise ValueError('Initialization alignment differs')
    mapper.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint;gc.collect()
    # Initial state is saved in every run's identity; changing only seed changes order.
    identity={**c['identity'],'objective':objective,'seed':seed,'initialization_sha256':sha(init),
        'history_receipts_sha256':digest(receipts),'gradient_predictions_per_update':32,'maximum_updates':2048,'wall_seconds':10800,
        'optimizer':'AdamW lr=1e-5 weight_decay=0','gradient_clip':1.,'minimum_updates_for_nomination':256,
        'validation':'Common full-boundary KL over all valid declared offsets, equal task means',
        'source_cache_recomputation':'Complete saved prefix in chunks of512; no cropping; ordinary extends cache on exact source answer tokens.'}
    # Worker identity remains immutable; the nested training identity adds recipe inputs.
    runroot=root/'run';runroot.mkdir()
    runtime=TrainingRuntime(source,receiver,mapper,DiskLRU(root/'cache_lru'),guard)
    publish(stage='training',objective=objective,seed=seed)
    def backup_manifest():
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(runroot.glob('mapper*.pt'))}
        if any((ROOT/p).stat().st_size>1024**3 for p in files) or sum((ROOT/p).stat().st_size for p in files)>8*1024**3:raise ValueError('Mapper backup bound exceeded')
        write(root/'checkpoint_manifest.json',{'files':files,'policy':'Immutable mapper-only checkpoints; optimizer resume not supported.'})
    done=train_bounded(runtime,training,validation,objective,seed,runroot,spec['deadline_epoch']-120,identity,checkpoint_callback=backup_manifest)
    write(root/'cache_working_set.json',{'disk_limit_bytes':32*1024**3,'events':runtime.lru.events})
    write(root/'complete.json',{'identity_sha256':digest(c['identity']),'run_complete_sha256':sha(runroot/'complete.json'),'training':done})
    finish_worker(c,'complete',stage='training',steps=done['steps'],stop_reason=done['stop_reason'],eligible_checkpoints=len(done['nominees']))
    return done


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker-spec');args=parser.parse_args();context=None
    try:
        context=worker_context(ROOT,args.worker_spec);run(context)
    except BaseException as exc:
        if context:finish_worker(context,'failed',exception=type(exc).__name__,error=str(exc));write(context['root']/'failure.json',{'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
