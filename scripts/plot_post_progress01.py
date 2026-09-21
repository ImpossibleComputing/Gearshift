#!/usr/bin/env python3
"""Regenerate diagnostic plots from compact JSON summaries without model weights."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/coding_pilot_v1/post_progress01_report'

def render():
    sources={}
    p=OUT/'hybrid_summary.json'
    if p.exists():
        d=json.loads(p.read_text())
        if d['completed_tasks']==40:
            fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');labels=['M','H','D','P'];v=[d['arms'][a]['passes'] for a in labels];bars=axes[0].bar(labels,v,color=['#b85239','#9467bd','#247282','#697b84']);axes[0].bar_label(bars,labels=[f'{x}/40' for x in v],padding=3);axes[0].set_ylim(0,44);axes[0].set_ylabel('Hidden-test passes')
            for i,(label,r) in enumerate(d['paired']['contrasts'].items()):
                m=r['difference']*100;lo,hi=[x*100 for x in r['ci95']];axes[1].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color='#9467bd')
            axes[1].set_yticks(range(3),list(d['paired']['contrasts']));axes[1].axvline(0,color='gray',ls='--');axes[1].set_xlabel('Paired pass-rate difference, percentage points');fig.suptitle('Frozen prompt-preservation diagnostic · inspected development tasks');fig.supxlabel('M: mapped · H: native prompt + mapped suffix · D: native whole history · P: prompt only, non-thinking\nExploratory paired 95% bootstrap intervals; nonsignificance is not equivalence.',fontsize=9)
            for ext in ['png','svg']:fig.savefig(OUT/('prompt_preservation.'+ext),dpi=170)
            plt.close(fig);sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=OUT/'memorization_summary.json'
    if p.exists():
        data=json.loads(p.read_text())
        if data:
            fig,(ax,quality)=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
            for d in data:
                curve=d['curve']
                if not curve:continue
                positions=[r['gradient_predictions_so_far'] for r in curve]
                ax.plot(positions,[r['mean_task_kl'] for r in curve],marker='o',color='#9467bd',label='Mean of four histories')
                points=[]
                for r in curve:
                    scored=[x for x in d['scores'] if x['step']==r['step'] and x['arm']=='M_seen' and x['passed'] is not None]
                    if len(scored)==4:points.append((r['gradient_predictions_so_far'],sum(x['passed'] for x in scored)))
                if points:quality.plot([x for x,y in points],[y for x,y in points],marker='o',color='#9467bd',label='Mapped continuation')
                native=[x for x in d['scores'] if x['arm']=='D_seen' and x['passed'] is not None]
                if len(native)==4:quality.axhline(sum(x['passed'] for x in native),color='#247282',ls='--',label='Native replay')
            ax.set_ylabel('Dense teacher-forced KL, four-task mean');ax.set_yscale('log');ax.legend(fontsize=9)
            quality.set_ylabel('Hidden-test passes on the same four cases');quality.set_ylim(-.15,4.4);quality.set_yticks(range(5));quality.legend(loc='lower right',fontsize=9)
            for a in (ax,quality):a.set_xlabel('Actual scored prediction positions');a.grid(alpha=.2)
            fig.suptitle('Four seen training histories · optimization sanity check')
            fig.supxlabel('SEEN-TRAINING-CASE performance only; no generalization claim. Hidden scores do not control optimization or stopping.',fontsize=9)
            for ext in ['png','svg']:fig.savefig(OUT/('seen_training_curve.'+ext),dpi=170)
            plt.close(fig);sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'figure_sources.json').write_text(json.dumps(sources,indent=2)+'\n')

if __name__=='__main__':render()
