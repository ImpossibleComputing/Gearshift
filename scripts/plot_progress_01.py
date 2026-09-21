#!/usr/bin/env python3
"""Render a frozen-population figure directly from saved phase-2 summaries."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'results/phase2_v1/report/characterization_summary.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
for ax,family,n,title in zip(axes,['arithmetic','code'],[400,100],['Arithmetic · 400 questions','Code · 100 tasks']):
    rr=[next(r for r in rows if r['family']==family and r['condition']==c) for c in ['C','M']]
    assert all(int(r['n'])==n and int(r['scored_n'])==n for r in rr)
    values=[100*float(r['mean_score']) for r in rr]
    bars=ax.bar(['Native text replay','Frozen mapped'],values,color=['#216e80','#b24a39'],width=.6)
    for bar,r in zip(bars,rr):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{round(float(r['mean_score'])*n)}/{n}",ha='center',fontsize=13)
    ax.set_title(title,fontsize=13,pad=12);ax.set_ylim(0,100)
    ax.set_ylabel('Correct (%)' if family=='arithmetic' else 'Hidden-test passes (%)')
    ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Frozen cache transfer lost quality in the broader study',fontsize=15)
fig.supxlabel('Qwen3-1.7B → Qwen3-0.6B · H100 · Separate populations; no cross-panel pooling',fontsize=10)
out=ROOT/'publication/figures';out.mkdir(exist_ok=True)
fig.savefig(out/'progress_01_frozen_transfer.png',dpi=180)
fig.savefig(out/'progress_01_frozen_transfer.svg')
