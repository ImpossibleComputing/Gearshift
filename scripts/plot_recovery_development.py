#!/usr/bin/env python3
"""Plot only a complete, committed 40-task exploratory comparison."""
from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(summary_path,out):
    summary_path,out=Path(summary_path),Path(out)
    d=json.loads(summary_path.read_text())
    if d['coverage']['committed_complete']!=40 or not d['paired']:
        raise ValueError('A complete paired development comparison is required')
    arms=['A','B','C','D','C_initial'];labels=['Large alone\nA','Small alone\nB','Mapped\ntrained C','Text handoff\nmatched D','Mapped\naffine init']
    if any(d['arms'][a]['n']!=40 for a in arms):raise ValueError('Different populations')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained',width_ratios=[1.4,1])
    values=[d['arms'][a]['passes'] for a in arms]
    bars=axes[0].bar(labels,values,color=['#697b84','#a5adb2','#b85239','#247282','#d49c7a'],width=.67)
    axes[0].bar_label(bars,labels=[f'{n}/40' for n in values],padding=4)
    axes[0].set_ylim(0,44);axes[0].set_ylabel('Tasks passing all hidden tests');axes[0].set_title('Same 40 development tasks')
    keys=['C-B','C-D','C-A']
    for y,key in enumerate(keys):
        row=d['paired']['contrasts'][key];mean=row['difference']*100;lo,hi=[x*100 for x in row['ci95']]
        axes[1].errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',color='#b85239',capsize=5,markersize=6)
        axes[1].annotate(f'{mean:+.1f} pp', (mean,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
    axes[1].set_yticks(range(3),['C minus B','C minus matched D','C minus A']);axes[1].set_ylim(2.65,-.65)
    axes[1].axvline(0,color='#697b84',ls='--',lw=1);axes[1].set_xlabel('Pass-rate difference (percentage points)');axes[1].set_title('Paired differences with 95% intervals')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y' if ax is axes[0] else 'x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Gearshift · first larger-model mapped development comparison',fontsize=15)
    fig.supxlabel('Qwen3-32B → Qwen3-8B · BF16 · H200 · fixed partial corpus: 104 training / 21 validation histories\nInspected development set; not fresh confirmation. Intervals: 10,000 paired task resamples, unadjusted. No speedup claim.',fontsize=9)
    out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:fig.savefig(out.with_suffix('.'+ext),dpi=170)
    plt.close(fig)
    out.with_suffix('.source.json').write_text(json.dumps({'source':str(summary_path),'sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),'coverage':d['coverage'],'passes':dict(zip(arms,values)),'paired':d['paired']},indent=2)+'\n')
    return out.with_suffix('.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('summary');p.add_argument('output_stem');a=p.parse_args();print(plot(a.summary,a.output_stem))
