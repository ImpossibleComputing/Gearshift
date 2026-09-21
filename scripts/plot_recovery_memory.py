#!/usr/bin/env python3
"""Single-trajectory allocator trace, never a model-quality figure."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(root,output):
    root,output=Path(root),Path(output);rows=[json.loads(s) for s in (root/'memory_telemetry.jsonl').read_text().splitlines() if s.strip()]
    rows=[r for r in rows if r['stage']=='source_reasoning' and r.get('sequence_length') is not None]
    if not rows:return None
    resume=json.loads((root/'trajectory/resume.json').read_text());prompt=len(resume['prompt_ids']);x=[r['sequence_length']-prompt for r in rows]
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,layout='constrained',height_ratios=[2,1])
    for k,label,color in [('allocated','Live tensor allocation','#216e80'),('reserved','Allocator reservation','#7a5195')]:
        axes[0].plot(x,[r[k]/1024**3 for r in rows],label=label,color=color,linewidth=2)
    axes[0].set_ylabel('GPU memory (GiB)');axes[0].legend(loc='upper left');axes[0].set_ylim(70,145)
    axes[1].plot(x,[r['free']/1024**3 for r in rows],color='#216e80');axes[1].axhline(10,color='#b24a39',ls='--',label='10 GiB warning threshold')
    axes[1].set_ylabel('Driver free (GiB)');axes[1].set_xlabel('Reasoning tokens completed');axes[1].legend(loc='upper right')
    for ax in axes:
        ax.axvline(resume['saved_prefix_length'],color='#555555',ls=':',label='Last saved original prefix')
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.suptitle('Instrumented recovery · leetcode/2893 · one H200',fontsize=14)
    fig.supxlabel('Dotted line: last saved original prefix. Amended allocator lifecycle; memory thresholds are warnings.\nPeak counters reset after native controls. No coding-quality claim follows from this trace.',fontsize=9)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160);plt.close(fig);return output
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');a=p.parse_args();print(plot(a.root,a.output))
