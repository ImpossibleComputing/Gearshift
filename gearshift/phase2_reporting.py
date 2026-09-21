"""Reports generated from compact records; checkpoints are optional verification only."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv
import math
import numpy as np
from .phase2_io import read,write,digest,artifact,checkpoint_status


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def paired_cluster(rows,a,b,metric,seed=20260915):
    pairs=defaultdict(dict)
    for r in rows:
        if r['condition'] in [a,b] and r['score'].get(metric) is not None:
            pairs[(r['cluster_id'],r['task_id'])][r['condition']]=float(r['score'][metric])
    clusters=defaultdict(list);gains=losses=ties=0
    for (cluster,task),p in pairs.items():
        if set(p)!={a,b}:continue
        delta=p[a]-p[b];clusters[cluster].append(delta)
        gains+=delta>0;losses+=delta<0;ties+=delta==0
    # Equal parent-cluster weighting; nested lengths/branches/variants are retained within parent.
    values=np.array([np.mean(v) for v in clusters.values()]);rng=np.random.default_rng(seed)
    if not len(values):return None
    means=values[rng.integers(0,len(values),(5000,len(values)))].mean(1)
    return dict(a=a,b=b,metric=metric,n_clusters=len(values),n_task_variants=gains+losses+ties,
        difference=float(values.mean()),ci_low=float(np.quantile(means,.025)),ci_high=float(np.quantile(means,.975)),
        gains=gains,losses=losses,ties=ties,unit='original parent task; equal cluster weights',equivalence_established=False)


def legacy_records_report(root,destination,verify_checkpoint=False):
    root=Path(root);dest=Path(destination)
    if dest.resolve()==root.resolve() or dest.resolve()==Path('.').resolve():
        raise ValueError('Use a new destination; historical results are frozen')
    dest.mkdir(parents=True,exist_ok=True)
    mf=read(root/'task_manifest.json');identity=mf['identity']
    if digest(identity)!=mf['identity_sha256']:raise ValueError('Manifest changed')
    complete=read(root/'task_complete.json')
    for name,desc in complete['artifacts'].items():
        if artifact(root/name)!=desc:raise ValueError('Raw records changed')
    rows=read(root/'task_records.json');expected={(i,c) for i in identity['ids'] for c in identity['conditions']}
    keys=[(r['dataset_index'],r['condition']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Incomplete records')
    selection=read(root/'selection.json')
    status=checkpoint_status(root/'selected_mapper.pt',selection['selected_artifact']) if verify_checkpoint else dict(status='not_requested_not_verified',expected=selection['selected_artifact'])
    groups=defaultdict(list)
    for r in rows:groups[r['condition']].append(r)
    summary=[dict(condition=c,n=len(g),**{m:sum(r[m] for r in g) for m in ['correct','format_valid','correct_and_valid']},
        mean_handoff_ms=float(np.mean([r['handoff_wall_ms'] for r in g]))) for c,g in sorted(groups.items())]
    write(dest/'summary.json',dict(checkpoint_verification=status,conditions=summary));csv_write(dest/'task_summary.csv',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh([r['condition'] for r in summary],[100*r['correct']/r['n'] for r in summary]);ax.set(xlim=(0,100),xlabel='Extracted correct (%)')
    fig.tight_layout();fig.savefig(dest/'task_quality.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for arm in ['plaintext','chat']:
        curve=read(root/f'{arm}_learning_curve.json');ax.plot([r['step'] for r in curve],[r['validation_kl'] for r in curve],label=arm)
    ax.set(xlabel='Optimizer updates',ylabel='Mixed validation KL');ax.legend();fig.tight_layout();fig.savefig(dest/'learning_curves.png',dpi=150);plt.close(fig)
    (dest/'README.md').write_text('Regenerated from complete compact follow-up records. Checkpoint verification: '+status['status']+'. Missing weights are never reported as verified. Historical narratives and measurements are unchanged.\n')
    return status
