#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',default='results/qwen3_1.7b_to_0.6b')
    args=parser.parse_args()
    root=Path(args.results); dest=Path('plots')/root.name; dest.mkdir(parents=True,exist_ok=True)
    manifest=root/'reasoning_manifest.json'
    main_linear=json.loads(manifest.read_text()).get('handoff_variant','normalized_content') if manifest.exists() else 'normalized_content'
    plt.rcParams.update({'figure.dpi':140,'savefig.dpi':180,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'native':'#166534','content':'#2563eb','normalized_content':'#60a5fa','post':'#9333ea','zero':'#dc2626',
            'no_context':'#ea580c','naive':'#777777','random':'#444444','functional':'#0891b2',
            'neighbor':'#92400e','lowrank':'#be185d','mlp':'#4338ca','weighted':'#4d7c0f'}
    labels={'native':'Small native prefill','content':'Linear KV, searched layers','normalized_content':'Linear KV, RoPE corrected',
            'post':'Linear KV, direct post-RoPE','zero':'Zero KV','no_context':'No context','functional':'KL-trained mapper',
            'lowrank':'Rank-128 linear','mlp':'Residual MLP'}
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,tag,title in zip(axes,['k_post','k_content','v'],['K: stored post-RoPE','K: content space','V']):
        obj=json.loads((root/f'alignment_{tag}.json').read_text())
        a=np.array([[m['r2'] for m in r] for r in obj['source_by_target']])
        im=ax.imshow(a,vmin=0,vmax=1,origin='lower',aspect='auto',cmap='viridis')
        ax.plot(range(a.shape[1]),obj['normalized_sources'],'w--',lw=1,label='Normalized depth')
        ax.scatter(range(a.shape[1]),obj['selected_sources'],s=9,c='#ef4444',label='Best screening R²')
        ax.set(xlabel='Target layer (zero-based)',ylabel='Source layer (zero-based)',title=title)
    axes[0].legend(fontsize=7,loc='upper left'); fig.colorbar(im,ax=axes,label='Validation R² (rank-128 screening)')
    fig.savefig(dest/'layer_alignment.png'); plt.close(fig)
    r=pd.read_json(root/'reconstruction_validation.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for v,g in r.groupby('variant'):
        if v.startswith('normalized_'): continue
        axes[0].plot(g.target_layer,g.r2,label=v)
        axes[1].plot(g.target_layer,g.cosine,label=v)
    axes[0].set(ylabel='Validation R²',xlabel='Target layer',title='Full-rank linear reconstruction')
    axes[1].set(ylabel='Mean cosine similarity',xlabel='Target layer',title='Full-rank linear reconstruction')
    axes[0].legend(); axes[1].legend(); fig.savefig(dest/'reconstruction.png'); plt.close(fig)
    if (root/'functional.json').exists():
        records=json.loads((root/'functional.json').read_text()); df=pd.DataFrame(records)
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for condition in ['native',main_linear,'post','functional','zero','no_context']:
            sub=df[df.condition==condition]
            if len(sub)==0: continue
            grouped=sub.groupby('context_length')
            avg=grouped[['nll','kl']].mean()
            axes[0].plot(avg.index,np.exp(avg.nll),marker='o',label=labels.get(condition,condition),color=colors[condition])
            axes[1].plot(avg.index,avg.kl,marker='o',label=labels.get(condition,condition),color=colors[condition])
        for ax in axes:
            ax.set(xscale='log',xlabel='Historical context tokens'); ax.set_xticks(sorted(df.context_length.unique()),sorted(df.context_length.unique())); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter()); ax.grid(alpha=.2)
        axes[0].set(yscale='log',ylabel='Continuation perplexity (64 teacher-forced tokens)',title='Identical continuation endpoint across lengths')
        axes[1].set(ylabel='Next-token KL(native ‖ condition), nats',title='Functional cache fidelity')
        axes[0].legend(fontsize=7); fig.savefig(dest/'context_quality.png'); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for condition in ['native',main_linear,'post','functional','zero','no_context']:
            sub=[r for r in records if r['condition']==condition]
            if not sub: continue
            nll=np.array([r['token_nll'] for r in sub]); kl=np.array([r['token_kl'] for r in sub])
            for ax,array in [(axes[0],nll),(axes[1],kl)]:
                buckets=array.reshape(len(array),-1,8).mean(2).mean(0)
                ax.plot(np.arange(len(buckets))*8+4,buckets,marker='o',label=labels.get(condition,condition),color=colors[condition])
        axes[0].set(ylabel='Mean negative log-likelihood',title='Teacher-forced continuation quality')
        axes[1].set(ylabel='Mean KL from native, nats',title='Recovery as correct native tokens accumulate')
        for ax in axes: ax.set(xlabel='Tokens after handoff'); ax.grid(alpha=.2)
        axes[0].legend(fontsize=7); fig.savefig(dest/'post_handoff_drift.png'); plt.close(fig)
    if (root/'latency.csv').exists():
        df=pd.read_csv(root/'latency.csv')
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for condition,g in df.groupby('condition'):
            if condition not in ['native',main_linear,'functional','post']: continue
            for ax,metric in zip(axes,['handoff_core_ms','time_to_first_token_ms']):
                a=g.groupby('context_length')[metric].agg(['median','min','max'])
                ax.plot(a.index,a['median'],marker='o',label=labels.get(condition,condition),color=colors[condition])
                ax.fill_between(a.index,a['min'],a['max'],alpha=.15,color=colors[condition])
        for ax in axes: ax.set(xlabel='Historical context tokens',ylabel='Milliseconds'); ax.grid(alpha=.2); ax.legend(fontsize=7)
        axes[0].set_title('Native prefill versus cache transform + injection')
        axes[1].set_title('Time to first-token logits (includes common bridge)')
        fig.savefig(dest/'handoff_latency.png'); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        for condition in ['native',main_linear,'functional']:
            g=df[(df.condition==condition)&(df.context_length<=128)]
            if g.empty: continue
            a=g.groupby('context_length').time_to_first_token_ms.agg(['median','min','max'])
            ax.plot(a.index,a['median'],marker='o',color=colors[condition],label=labels.get(condition,condition))
            ax.fill_between(a.index,a['min'],a['max'],color=colors[condition],alpha=.15)
        ax.set(xlabel='Historical context tokens',ylabel='Milliseconds to first-token logits',title='Short-context crossover (median and observed range)')
        ax.set_xticks([8,16,32,64,128]); ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.savefig(dest/'short_context_latency.png'); plt.close(fig)
    if (root/'reasoning.csv').exists():
        df=pd.read_csv(root/'reasoning.csv'); a=df.groupby('condition').correct.agg(['mean','count','sum'])
        df['clean_correct']=[bool(re.fullmatch(r'\s*(?:\\boxed\{)?[-+]?\$?\d[\d,]*(?:\.\d+)?\}?\.?\s*',str(answer))) and bool(correct)
                             for answer,correct in zip(df.answer,df.correct)]
        clean=df.groupby('condition').clean_correct.agg(['mean','sum']).reindex(a.index)
        fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
        short={'A_small_only':'A: Small only','B_large_only':'B: Large only','C_text_handoff':'C: Text handoff','D_kv_handoff':'D: Linear KV','E_functional_kv':'E: KL-trained KV'}
        x=np.arange(len(a))
        ax.bar(x-.19,a['mean'],width=.36,color='#64748b',label='Correct extracted number')
        ax.bar(x+.19,clean['mean'],width=.36,color='#0f766e',label='Correct numeric-only response')
        for i,(_,r) in enumerate(a.iterrows()):
            ax.text(i-.19,r['mean']+.025,f'{int(r["sum"])}/{int(r["count"])}',ha='center',fontsize=8)
            ax.text(i+.19,clean.iloc[i]['mean']+.025,f'{int(clean.iloc[i]["sum"])}/{int(r["count"])}',ha='center',fontsize=8)
        ax.set_xticks(x,[short[c] for c in a.index],fontsize=9)
        ax.set(ylim=(0,1.15),ylabel='Answer accuracy',title='GSM8K: extracted number versus clean final answer')
        ax.legend(fontsize=8,loc='upper right')
        fig.savefig(dest/'reasoning_accuracy.png'); plt.close(fig)
    print(dest)


if __name__=='__main__': main()
