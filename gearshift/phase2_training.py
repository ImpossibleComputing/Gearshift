"""Matched affine continuation versus actual-boundary distillation; frozen LMs."""
from __future__ import annotations
import copy
import time
from pathlib import Path
import numpy as np
import torch
from .core import CacheExtractor,CacheInjector,save_checkpoint,timed,sync,memory,seed_all
from .phase2_io import read,write,immutable,bind,digest,artifact,snapshot,runtime_identity
from .phase2_tasks import from_dict,apply_budgets
from .phase2_inference import source_history,bridge_ids,adapter_from
from .followup import render_measured
from .identity import backend_identity


def training_code():
    return snapshot(['gearshift/phase2_training.py','gearshift/phase2_inference.py','gearshift/phase2_io.py',
        'gearshift/phase2_tasks.py','gearshift/core.py','gearshift/mapping.py','scripts/phase2_train.py','requirements.lock.txt'])


@torch.inference_mode()
def prepare_training(cfg,source,target,pair='1p7_to_0p6'):
    # The characterization must be completed before constructing adaptive objectives.
    if not (Path(cfg['output'])/'characterization/complete.json').exists():
        raise ValueError('Frozen characterization must complete before adaptive training preparation')
    root=Path(cfg['output'])/'training'/pair/'preparation'
    tasks=apply_budgets(cfg,[from_dict(t) for split in ['train','validation'] for t in read(Path(cfg['output'])/'tasks'/f'{split}.json')])
    identity=dict(schema=1,pair=pair,config=cfg,runtime=runtime_identity(),source=backend_identity(source),target=backend_identity(target),
        tasks=[t.visible() for t in tasks],code=training_code(),teacher='native target on same history; no hidden labels',
        protocols={'ordinary':'native assistant continuation after closing thinking','boundary':'actual new-user-turn handoff'},
        target_answer_cap=512)
    bind(root/'manifest.json',identity)
    for i,task in enumerate(tasks):
        path=root/'cases'/f'{task.task_id}.json'
        if path.exists():
            obj=read(path)
            if obj['identity_sha256']!=digest(identity) or obj['payload_sha256']!=digest({k:v for k,v in obj.items() if k!='payload_sha256'}):
                raise ValueError('Training preparation changed')
            continue
        tr,sp=source_history(source,target,task)
        shared=tr['prompt_ids']+tr['source_reasoning_ids'];out=target.prefill(shared);tp=CacheExtractor.tensors(out.past_key_values)
        teachers={}
        for arm,protocol in [('ordinary','native'),('boundary','newturn')]:
            bridge=bridge_ids(target.tokenizer,task.contract,tr['source_completed'],protocol)
            result,wall=timed(lambda:render_measured(target,tp,bridge,512),target.device)
            teachers[arm]=dict(bridge_ids=bridge,answer_ids=result['tokens'],answer=result['text'],stop_reason=result['stop_reason'],wall_ms=wall)
        obj=dict(identity_sha256=digest(identity),task=task.visible(),trajectory=tr,teachers=teachers)
        obj['payload_sha256']=digest(obj);immutable(path,obj)
        print(f'{pair} training preparation {i+1}/{len(tasks)} {task.task_id}',flush=True)
    immutable(root/'complete.json',dict(identity_sha256=digest(identity),cases={str(p.relative_to(root)):artifact(p) for p in sorted((root/'cases').glob('*.json'))}))


def sparse_case(obj,arm,requested_anchor,n=8):
    t=obj['teachers'][arm];bridge=t['bridge_ids'];answers=t['answer_ids']
    # Short answers remain included; anchors fall back to the last available prediction window.
    actual_n=min(n,len(answers));anchor=min(requested_anchor,max(0,len(answers)-actual_n))
    history=obj['trajectory']['prompt_ids']+obj['trajectory']['source_reasoning_ids']
    full=bridge+answers
    start=len(bridge)-1+anchor
    return dict(history=history,prefix_ids=full[:start],input_ids=full[start:start+actual_n],
        target_ids=answers[anchor:anchor+actual_n],anchor=anchor,requested_anchor=requested_anchor,
        predictions=actual_n,family=obj['task']['family'],task_id=obj['task']['task_id'])


def frozen_pair(source,target,history):
    so=source.prefill(history);to=target.prefill(history)
    # Clone outside inference mode to make historical tensors safe autograd constants.
    return CacheExtractor.tensors(so.past_key_values,clone=True),CacheExtractor.tensors(to.past_key_values,clone=True)


def prediction(backend,pairs,case):
    cache=CacheInjector.create(pairs,clone=False)
    if case['prefix_ids']:cache=backend.forward(case['prefix_ids'],cache).past_key_values
    return backend.forward(case['input_ids'],cache,all_logits=True).logits.float().log_softmax(-1)


def make_trainable(adapter):
    params=[]
    for key in ['functional_k','functional_v']:
        for entry in adapter.weights[key]:
            for name in ['weight','bias']:
                entry[name]=torch.nn.Parameter(entry[name].detach().clone());params.append(entry[name])
    return params


def weights(adapter):
    return {key:[{k:(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v) for k,v in m.items()}
                 for m in adapter.weights[key]] for key in ['functional_k','functional_v']}


@torch.no_grad()
def validate(adapter,source,target,objects,anchors):
    rows=[]
    # Both training arms are selected on the SAME mixture of both validation protocols/domains.
    for index,obj in enumerate(objects):
        requested=anchors[index%len(anchors)]
        history=obj['trajectory']['prompt_ids']+obj['trajectory']['source_reasoning_ids']
        sp,tp=frozen_pair(source,target,history);mapped=adapter.transform(sp,'functional')
        for arm in ['ordinary','boundary']:
            case=sparse_case(obj,arm,requested)
            reference=prediction(target,tp,case);candidate=prediction(target,mapped,case)
            kl=(reference.exp()*(reference-candidate)).sum(-1).mean()
            target_ids=torch.tensor(case['target_ids'],device=target.device).view(1,-1,1)
            nll=-candidate.gather(-1,target_ids).mean()
            rows.append(dict(task_id=case['task_id'],family=case['family'],protocol=arm,anchor=case['anchor'],requested_anchor=requested,
                predictions=case['predictions'],kl=float(kl),nll=float(nll),top1=float((reference.argmax(-1)==candidate.argmax(-1)).float().mean())))
    return rows


def train(cfg,source,target,initialization,pair='1p7_to_0p6',seeds=None):
    root=Path(cfg['output'])/'training'/pair;prep=root/'preparation';complete=read(prep/'complete.json')
    for name,desc in complete['cases'].items():
        if artifact(prep/name)!=desc:raise ValueError('Prepared training case changed')
    objects=[read(prep/name) for name in complete['cases']]
    tr=[o for o in objects if o['task']['split']=='train'];dev=[o for o in objects if o['task']['split']=='validation']
    f=cfg['training'];seeds=f['seeds'] if seeds is None else seeds
    objectives=cfg.get('training_objectives',['ordinary','boundary'])
    for seed in seeds:
        seed_all(seed);dest=root/f'seed_{seed}'
        rng=np.random.default_rng(seed);families=sorted({o['task']['family'] for o in tr});schedule=[]
        grouped={family:[o for o in tr if o['task']['family']==family] for family in families}
        while len(schedule)<f['max_steps']:
            for family in rng.permutation(families):
                obj=grouped[family][int(rng.integers(len(grouped[family])))]
                schedule.append(dict(task_id=obj['task']['task_id'],anchor=int(rng.choice(f['later_positions']))))
        schedule=schedule[:f['max_steps']]
        identity=dict(schema=1,pair=pair,config=f,runtime=runtime_identity(),seed=seed,initialization=artifact(initialization),preparation=artifact(prep/'complete.json'),
            source=backend_identity(source),target=backend_identity(target),schedule=schedule,code=training_code(),
            objectives=objectives,selection='equal-weight family then protocol validation KL; shared stopping when all arms plateau',
            gradient='Full gradient through mapped historical K/V and intervening teacher-forced bridge/answer prefix. Frozen LMs; no packing.')
        bind(dest/'manifest.json',identity)
        if (dest/'complete.json').exists():continue
        if (dest/'progress.json').exists():raise ValueError('Partial optimizer state preserved; use a declared new attempt or implement exact optimizer resume')
        arms={}
        for arm in objectives:
            adapter=adapter_from(initialization,source,target);params=make_trainable(adapter)
            arms[arm]=dict(adapter=adapter,params=params,optimizer=torch.optim.Adam(params,lr=f['learning_rate']),best=float('inf'),best_step=0,
                no_improvement=0,curve=[],predictions=0)
        def checkpoint(step):
            for arm,a in arms.items():
                rows=validate(a['adapter'],source,target,dev,f['later_positions'])
                domain={family:float(np.mean([r['kl'] for r in rows if r['family']==family])) for family in families}
                score=float(np.mean(list(domain.values())));entry=dict(step=step,validation_kl=score,by_domain=domain,rows=rows)
                a['curve'].append(entry)
                improved=score < a['best']*(1-f['min_relative_improvement'])
                a['no_improvement']=0 if improved else a['no_improvement']+1
                if score<a['best']:
                    a['best']=score;a['best_step']=step;save_checkpoint(dest/f'{arm}_best.pt',weights(a['adapter']))
                write(dest/f'{arm}_curve.json',a['curve'])
                print(f'{pair} seed {seed} {arm} step {step} validation KL={score:.5f}',flush=True)
        started=time.perf_counter();checkpoint(0);rows=[];lookup={o['task']['task_id']:o for o in tr};stop='max_steps'
        for step,item in enumerate(schedule,1):
            obj=lookup[item['task_id']];history=obj['trajectory']['prompt_ids']+obj['trajectory']['source_reasoning_ids']
            sp,tp=frozen_pair(source,target,history)
            cases={arm:sparse_case(obj,arm,item['anchor'],f['prediction_tokens']) for arm in arms}
            # Match gradient-bearing token count if either teacher stopped before eight tokens.
            n=min(c['predictions'] for c in cases.values())
            for arm,case in cases.items():case['input_ids']=case['input_ids'][:n];case['target_ids']=case['target_ids'][:n];case['predictions']=n
            order=objectives if step%2 else list(reversed(objectives))
            for arm in order:
                a=arms[arm];case=cases[arm];tick=time.perf_counter()
                with torch.no_grad():reference=prediction(target,tp,case)
                mapped=a['adapter'].transform(sp,'functional');candidate=prediction(target,mapped,case)
                kl=(reference.exp()*(reference-candidate)).sum(-1).mean()
                reconstruction=sum((x.float()-y.float()).square().mean()/y.float().square().mean().clamp_min(1e-6)
                    for pred,actual in zip(mapped,tp) for x,y in zip(pred,actual))/(2*len(tp))
                loss=kl+f['reconstruction_weight']*reconstruction
                a['optimizer'].zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(a['params'],f['gradient_clip'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient; both arms stop and retain failure')
                a['optimizer'].step();sync(target.device);a['predictions']+=n
                rows.append(dict(step=step,arm=arm,task_id=item['task_id'],family=case['family'],history_tokens=len(history),
                    requested_anchor=item['anchor'],actual_anchor=case['anchor'],prediction_tokens=n,prefix_tokens=len(case['prefix_ids']),
                    kl=float(kl.detach()),reconstruction=float(reconstruction.detach()),gradient_norm=float(norm),wall_ms=(time.perf_counter()-tick)*1000,**memory()))
                del mapped,candidate,reference,loss,kl,reconstruction
            write(dest/'training_steps.json',rows);write(dest/'progress.json',dict(state='partial',step=step))
            if step%f['validation_every']==0 or step==f['max_steps']:
                checkpoint(step)
                if step>=f['minimum_steps'] and all(a['no_improvement']>=f['patience'] for a in arms.values()):stop='both_arms_validation_plateau';break
                if time.perf_counter()-started>=f['per_seed_wall_cap_seconds']:stop='prespecified_per_seed_wall_cap';break
        assert all(p.grad is None for b in [source,target] for p in b.model.parameters())
        immutable(dest/'complete.json',dict(identity_sha256=digest(identity),steps_per_arm=step,stop_reason=stop,wall_seconds=time.perf_counter()-started,
            arms={arm:dict(best_step=a['best_step'],validation_kl=a['best'],predictions=a['predictions'],checkpoint=artifact(dest/f'{arm}_best.pt')) for arm,a in arms.items()}))
        write(dest/'progress.json',dict(state='complete',step=step,stop_reason=stop))
    # Select objective by mean validation across every declared/completed seed, not the best seed.
    completed={str(seed):read(root/f'seed_{seed}/complete.json') for seed in seeds}
    scores={arm:float(np.mean([r['arms'][arm]['validation_kl'] for r in completed.values()])) for arm in objectives}
    selected=min(scores,key=scores.get)
    immutable(root/'selection.json',dict(selected_objective=selected,validation_mean_by_objective=scores,seeds=seeds,
        representative_seed=seeds[0],representative_policy='first prespecified seed, never best-of-seeds',
        checkpoints={str(s):dict(path=str(root/f'seed_{s}'/f'{selected}_best.pt'),**artifact(root/f'seed_{s}'/f'{selected}_best.pt')) for s in seeds},
        final_confirmation_opened=False,completed_runs=completed))
