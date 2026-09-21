#!/usr/bin/env python3
"""Freeze completed transactions by availability, then fit only training features."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import sha,digest,bind,write
from gearshift.coding_training import read,validate_history,fit_ridge,atomic_tensor
from gearshift.coding_gradients import AffineMapper
from coding_history_stop_report import verify_task
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/coding_pilot_v1/partial_corpus_20260917_v1'

def freeze():
    expected={}
    for split in ['training','validation']:
        for row in read(ROOT/f'data/coding_pilot_v1/visible/{split}.json'):expected[row['task_id']]={**row,'split':split}
    base=ROOT/'results/coding_pilot_v1/history_recovery_20260916_01';rows=[];missing=[];files={}
    for folder in sorted(base.glob('*/tasks/*')):
        if not (folder/'complete.json').exists():continue
        identity=read(folder.parent.parent/'identity.json');tid=read(folder/'complete.json')['task_id']
        if identity['config_sha256']!=sha(ROOT/'configs/coding_pilot_v1/pilot.json'):raise ValueError('Model/config drift')
        native=read(folder.parent.parent/'native_gate.json')
        if not native['passed'] or native['identity_sha256']!=digest(identity):raise ValueError('Native controls differ')
        receipt,h,a=verify_task(folder,identity,expected[tid])
        row={'task_id':tid,'split':expected[tid]['split'],'folder':str(folder.relative_to(ROOT)),
             'reasoning_tokens':len(h['reasoning_ids']),'answer_tokens':len(a['answer_ids']),'source_capped':h['reasoning_capped'],
             'measurement_identity':receipt['identity_sha256'],'files':{}}
        paths=[folder/'complete.json',folder/'source_history.json',folder/'teacher_answer.json',folder.parent.parent/'identity.json',folder.parent.parent/'native_gate.json']
        if row['split']=='training':
            feature=folder/'paired_features.pt'
            if sha(feature)!=receipt['feature_files']['paired_features.pt']:raise ValueError('Feature bytes differ')
            paths.append(feature)
        for p in paths:
            rel=str(p.relative_to(ROOT));files[rel]={'sha256':sha(p),'bytes':p.stat().st_size};row['files'][p.name]=files[rel]
        rows.append(row)
    ids=[r['task_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate history')
    counts={s:sum(r['split']==s for r in rows) for s in ['training','validation']}
    if counts!={'training':104,'validation':21}:raise ValueError(f'Unexpected availability {counts}')
    for tid in sorted(set(expected)-set(ids)):
        matches=list(base.glob('*/tasks/'+tid.replace('/','__')+'/inflight_source_reasoning.json'))
        missing.append({'task_id':tid,'split':expected[tid]['split'],'status':'partial' if matches else 'not_started',
            'saved_tokens':len(read(matches[0])['tokens']) if matches else 0})
    manifest={'schema':1,'corpus_id':'partial_corpus_20260917_v1','amendment_sha256':sha(ROOT/'configs/coding_pilot_v1/recovery_20260917/amendment.json'),
        'config_sha256':sha(ROOT/'configs/coding_pilot_v1/pilot.json'),'counts':counts,'feature_count':104,'histories':rows,'absent':missing,'files':files,
        'selection':'All completed valid transactions at fixed snapshot; no task outcomes or mapper performance used.',
        'limitation':'Not completion of original128/32. Completion selection may underrepresent long/hard histories; four interrupted histories have20k+ tokens.',
        'membership_files':{s:sha(ROOT/f'data/coding_pilot_v1/visible/{s}.json') for s in ['training','validation']}}
    OUT.mkdir(exist_ok=True);bind(OUT/'corpus_manifest.json',manifest);print(json.dumps({'counts':counts,'features':104,'absent':len(missing),'manifest_sha256':sha(OUT/'corpus_manifest.json')}),flush=True)
    return manifest

def load_corpus(repo,manifest_path,features=True):
    repo=Path(repo);m=read(manifest_path);histories=[];paths=[]
    for rel,info in m['files'].items():
        if not features and rel.endswith('.pt'):continue
        p=repo/rel
        if p.is_symlink() or p.stat().st_size!=info['bytes'] or sha(p)!=info['sha256']:raise ValueError('Corpus bytes changed: '+rel)
    for row in m['histories']:
        folder=repo/row['folder'];h=read(folder/'source_history.json');a=read(folder/'teacher_answer.json')
        obj={'task_id':row['task_id'],'split':row['split'],'source_history':h,'teacher_answer':a};validate_history(obj);histories.append(obj)
        if row['split']=='training':paths.append(folder/'paired_features.pt')
    if len(histories)!=125 or len(paths)!=104:raise ValueError('Partial corpus size changed')
    return histories,paths,m

def fit():
    import torch
    torch.set_num_threads(8);m=freeze();histories,paths,_=load_corpus(ROOT,OUT/'corpus_manifest.json')
    identity={'corpus_manifest_sha256':sha(OUT/'corpus_manifest.json'),'config_sha256':m['config_sha256'],'ridge':.01,'training_task_count':104,'validation_task_count':21,
        'source_specific':True,'statistics_dtype':'float64','device':'Studio CPU','code':{s:sha(ROOT/s) for s in ['scripts/coding_partial_corpus.py','gearshift/coding_training.py','gearshift/coding_gradients.py']}}
    bind(OUT/'initialization_identity.json',identity)
    if (OUT/'mapper_initialization.pt').exists():raise ValueError('Immutable initializer exists')
    source=SimpleNamespace(config=SimpleNamespace(**read(ROOT/'configs/coding_pilot_v1/reference/Qwen3-32B/config.json')),device='cpu')
    receiver=SimpleNamespace(config=SimpleNamespace(**read(ROOT/'configs/coding_pilot_v1/reference/Qwen3-8B/config.json')),device='cpu')
    mapper=AffineMapper(source,receiver);started=time.time()
    def progress(**row):write(OUT/'initialization_progress.json',{'epoch':time.time(),**row})
    reports=fit_ridge(mapper,paths,progress=progress)
    path=OUT/'mapper_initialization.pt';atomic_tensor(path,{'state_dict':{k:v.detach().cpu() for k,v in mapper.state_dict().items()},'identity_sha256':digest(identity),'corpus_manifest_sha256':sha(OUT/'corpus_manifest.json'),'ridge':.01,'fresh_source_specific':True,'training_task_count':104,'source_layers':mapper.sources})
    write(OUT/'initialization_complete.json',{'identity_sha256':digest(identity),'checkpoint_sha256':sha(path),'checkpoint_bytes':path.stat().st_size,'wall_seconds':time.time()-started,'fit':reports})
    print('Initialization complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');a=p.parse_args();fit() if a.fit else freeze()
