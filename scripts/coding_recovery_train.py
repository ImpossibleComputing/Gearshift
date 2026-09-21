#!/usr/bin/env python3
"""Single exploratory boundary-objective run on the fixed partial corpus."""
import argparse,gc,os,sys,time,traceback,math,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gearshift.coding_control import bind,digest,sha,write
from gearshift.coding_training import read,finish_worker,TrainingRuntime,DiskLRU,make_schedule,schedule_audit,atomic_tensor
from gearshift.coding_gradients import AffineMapper
from gearshift.coding_recovery import worker_context,initialize_backends
from coding_partial_corpus import load_corpus
ROOT=Path(__file__).resolve().parents[1]

def choose_checkpoint(curve):
    candidates=[r for r in curve if r['step']>=1 and len(r['rows'])==21 and r.get('checkpoint')]
    return min(candidates,key=lambda r:(r['validation_kl'],r['step'])) if candidates else None

class ObservedRuntime(TrainingRuntime):
    def __init__(self,*args,telemetry,**kwargs):super().__init__(*args,**kwargs);self.telemetry=telemetry
    def pair(self,obj,device=None):
        self.telemetry.sample(stage='pair_reconstruction',task_id=obj['task_id'],sequence_length=len(obj['source_history']['prefix_ids']))
        result=super().pair(obj,device);self.telemetry.sample(stage='pair_ready');return result
    def kl(self,obj,positions,objective,sp,tp,require_grad=True,already_extended=False):
        self.telemetry.sample(stage='differentiable_continuation' if require_grad else 'validation_continuation',task_id=obj['task_id'],sequence_length=len(obj['source_history']['prefix_ids'])+max(positions)+1)
        return super().kl(obj,positions,objective,sp,tp,require_grad,already_extended)


def preserve_latest(root):
    """Make the last committed rolling state immutable and externally backed up."""
    info_path=root/'latest_checkpoint.json'
    if not info_path.exists():return
    info=read(info_path);source=root/'mapper_latest.pt'
    if not source.exists() or sha(source)!=info['sha256']:return
    target=root/f"mapper_final_unvalidated_{info['step']:04d}.pt"
    if not target.exists():
        temporary=target.with_suffix('.tmp');shutil.copyfile(source,temporary)
        if sha(temporary)!=info['sha256']:raise ValueError('Final mapper copy differs')
        os.replace(temporary,target)
    path=root/'checkpoint_manifest.json';manifest=read(path) if path.exists() else {'files':{},'policy':'Immutable mapper states; validated and final unvalidated are distinguished by their receipts.'}
    manifest['files'][str(target.relative_to(ROOT))]=sha(target);write(path,manifest)
    write(root/'final_unvalidated_checkpoint.json',{'step':info['step'],'path':str(target.relative_to(ROOT)),'sha256':sha(target),'eligible_for_selection':False})


def run(c):
    spec=c['spec'];root=c['root'];t=c['telemetry'];guard=c['guard'];publish=c['publish']
    if spec['stage']!='exploratory_training':raise ValueError('Wrong stage')
    recipe_path=ROOT/spec['schedule_path'];recipe=read(recipe_path)
    if sha(recipe_path)!=spec['schedule_sha256']:raise ValueError('Training recipe changed')
    manifest_path=ROOT/spec['corpus_manifest']
    if sha(manifest_path)!=recipe['corpus_manifest_sha256']:raise ValueError('Corpus identity differs')
    histories,_,manifest=load_corpus(ROOT,manifest_path,features=False)
    train=[h for h in histories if h['split']=='training'];val=[h for h in histories if h['split']=='validation']
    if len(train)!=104 or len(val)!=21:raise ValueError('Partial split changed')
    source,receiver=initialize_backends(c);mapper=AffineMapper(source,receiver)
    init=ROOT/spec['initialization']
    if sha(init)!=spec['initialization_sha256']:raise ValueError('Initialization differs')
    saved=torch.load(init,map_location='cpu',weights_only=True)
    if saved['corpus_manifest_sha256']!=sha(manifest_path) or saved['source_layers']!=mapper.sources:raise ValueError('Initializer lineage changed')
    mapper.load_state_dict(saved['state_dict'],strict=True);del saved;gc.collect()
    identity={**c['identity'],'recipe_sha256':sha(recipe_path),'corpus_manifest_sha256':sha(manifest_path),'initialization_sha256':sha(init),
              'objective':recipe['objective'],'seed':recipe['seed'],'training_tasks':104,'validation_tasks':21}
    bind(root/'training_identity.json',identity)
    schedule=make_schedule(train,recipe['seed'],recipe['maximum_updates']);write(root/'schedule.json',schedule);write(root/'schedule_audit.json',schedule_audit(train,schedule))
    optimizer=torch.optim.AdamW(mapper.parameters(),lr=1e-5,weight_decay=0);lookup={r['task_id']:r for r in train}
    runtime=ObservedRuntime(source,receiver,mapper,DiskLRU(root/'cache_lru'),guard,telemetry=t)
    started=time.time();deadline=min(started+recipe['training_wall_seconds'],spec['deadline_epoch']-900)
    steps=[];curve=[];files={};last_step=0;stop='maximum_updates'
    def payload(step):return {'state_dict':{k:v.detach().cpu() for k,v in mapper.state_dict().items()},'step':step,'identity_sha256':digest(identity),'optimizer_resume_supported':False}
    def rolling(step):
        path=root/'mapper_latest.pt';tmp=root/'mapper_latest.tmp'
        with tmp.open('wb') as f:torch.save(payload(step),f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        write(root/'latest_checkpoint.json',{'step':step,'sha256':sha(path),'path':str(path.relative_to(ROOT)),'eligible_for_evaluation':False})
    def checkpoint(step,result):
        if not math.isfinite(result['validation_kl']) or result['validation_kl'] < -1e-5 or len(result['rows']) != 21:raise ValueError('Invalid or incomplete validation')
        path=root/f'mapper_step_{step:04d}.pt';atomic_tensor(path,payload(step));files[str(path.relative_to(ROOT))]=sha(path)
        write(root/'checkpoint_manifest.json',{'files':files,'policy':'Immutable validated mapper checkpoints; no optimizer resume.'})
        row={'step':step,**result,'checkpoint':{'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size}}
        curve.append(row);write(root/'validation_curve.json',curve)
    def training_guard():
        guard()
        if time.time()>=deadline:raise TimeoutError('Exploratory training wall bound')
    runtime.guard=training_guard
    try:
        for item in schedule:
            training_guard();publish(stage='functional_training',step=item['step'],maximum_updates=recipe['maximum_updates']);t.sample(stage='update_start',update=item['step'])
            before=time.time();row=runtime.train_update(item,lookup,recipe['objective'],optimizer);last_step=item['step']
            steps.append({'step':last_step,**row,'wall_seconds':time.time()-before});write(root/'training_steps.json',steps)
            rolling(last_step);t.sample(stage='update_committed',update=last_step)
            if last_step in recipe['validation_steps']:
                publish(stage='validation',step=last_step);result=runtime.validate(val);checkpoint(last_step,result)
    except TimeoutError as exc:
        stop='time_or_controlled_stop';write(root/'training_stop.json',{'error':str(exc),'last_full_update':last_step})
    finally:runtime.guard=guard
    # If no prior validation exists, a bounded allocation reserve can validate
    # the last full update. No partial gradient update can be nominated.
    if not curve and last_step and time.time()<spec['deadline_epoch']-600:
        publish(stage='final_reserved_validation',step=last_step);checkpoint(last_step,runtime.validate(val))
    selected=choose_checkpoint(curve)
    done={'identity_sha256':digest(c['identity']),'training_identity_sha256':digest(identity),'steps':last_step,'training_predictions':sum(x['gradient_predictions'] for x in steps),
          'stop_reason':stop,'converged':False,'selection_rule':recipe['checkpoint_selection'],'selected':selected,
          'wall_seconds':time.time()-started,'initialization_sha256':sha(init),'corpus_manifest_sha256':sha(manifest_path),'development_used_for_selection':False}
    preserve_latest(root)
    audit_started=time.time();inventory={}
    for cache_file in sorted((root/'cache_lru').glob('*.pt')):
        guard();inventory[str(cache_file.relative_to(ROOT))]={'bytes':cache_file.stat().st_size,'sha256':sha(cache_file),
            'logical_key_sha256':cache_file.stem,'location_after_cleanup':'Discarded with worker; reproducible full-prefix CPU cache.',
            'regenerate':'TrainingRuntime.pair on exact corpus history using pinned models and512-token full-prefix schedule; no history cropping.'}
    write(root/'cache_inventory.json',{'files':inventory,'audit_seconds':time.time()-audit_started,'retained':False})
    write(root/'complete.json',done);write(root/'cache_working_set.json',runtime.lru.events)
    if selected is None:raise RuntimeError('No fully validated trained checkpoint; inspect bounded attempt')
    write(root/'selection.json',{'checkpoint':selected['checkpoint'],'step':selected['step'],'validation_kl':selected['validation_kl'],
        'training_complete_sha256':sha(root/'complete.json'),'training_identity_sha256':digest(identity),'corpus_manifest_sha256':sha(manifest_path),
        'initialization_sha256':sha(init),'development_used_for_selection':False,'confirmation_used_for_selection':False})
    finish_worker(c,'complete',stage='exploratory_training_complete',steps=last_step,selected_step=selected['step'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker-spec');a=p.parse_args();c=None
    try:c=worker_context(ROOT,a.worker_spec);run(c)
    except BaseException as exc:
        if c:
            try:preserve_latest(c['root'])
            except Exception as save_error:write(c['root']/'checkpoint_preservation_error.json',{'error':str(save_error)})
            c['telemetry'].failure(exc);write(c['root']/'failure.json',{'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()});finish_worker(c,'failed',exception=type(exc).__name__,error=str(exc))
        raise
if __name__=='__main__':main()
