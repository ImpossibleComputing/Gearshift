#!/usr/bin/env python3
"""Regenerate scorer-repair tables/plots from compact receipts only."""
import argparse
import collections
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import digest,sha,write


def read(path):return json.loads(Path(path).read_text())
def label(record):
    condition=record['condition']
    if condition in ('D','P') or condition.startswith('START_'):return condition
    arm,form=condition.rsplit('_',1);return f"{arm}_{record['step']:04d}_{form}"

def cause(old,new):
    if new.get('missing') or new.get('passed') is None:return 'unresolved_infrastructure_missing'
    old_output=any('File too large' in r.get('stderr','') for r in old.get('receipts',[]))
    if old_output:return 'old_output_ceiling_failure_now_pass' if new['passed'] else 'old_output_ceiling_failure_still_fails'
    if old['passed']==new['passed']:return 'unchanged_pass' if new['passed'] else 'unchanged_failure'
    if old.get('category')=='timeout' and new['passed']:return 'old_timeout_now_pass_under_calibrated_policy'
    if old['passed'] and new.get('category') in ('cpu_timeout','wall_timeout'):return 'old_pass_now_timeout_under_calibrated_policy'
    return 'other_fail_to_pass' if new['passed'] else 'other_pass_to_fail'


def statistics(rows,seed=20260918,resamples=10000):
    tasks=sorted({r['task_id'] for r in rows});labels=sorted({r['condition'] for r in rows})
    index=np.random.default_rng(seed).integers(0,len(tasks),(resamples,len(tasks)))
    variants={};task_rows=[]
    for version in ('old','new'):
        lower=np.zeros((len(tasks),len(labels)));upper=lower.copy();counts={}
        for j,name in enumerate(labels):
            selected=[r for r in rows if r['condition']==name];counts[name]=len(selected)
            for i,tid in enumerate(tasks):
                rr=[r for r in selected if r['task_id']==tid]
                if len(rr)!=3 or len({r['seed_index'] for r in rr})!=3:raise ValueError('Expected exactly three distinct draws per task and condition')
                values=[r[version+'_passed'] for r in rr]
                lower[i,j]=sum(v is True for v in values)/3;upper[i,j]=(sum(v is True for v in values)+sum(v is None for v in values))/3
        blo=lower[index].mean(axis=1);bhi=upper[index].mean(axis=1);conditions={}
        for j,name in enumerate(labels):
            rr=[r for r in rows if r['condition']==name];missing=sum(r[version+'_passed'] is None for r in rr)
            conditions[name]={'passed_draws':sum(r[version+'_passed'] is True for r in rr),'draws':len(rr),'missing_draws':missing,
                'pass_rate':None if missing else float(lower[:,j].mean()),
                'ci95':None if missing else np.quantile(blo[:,j],[.025,.975]).tolist(),
                'possible_mean_bounds':[float(lower[:,j].mean()),float(upper[:,j].mean())],
                'conservative_bootstrap_bounds':[float(np.quantile(blo[:,j],.025)),float(np.quantile(bhi[:,j],.975))],
                'categories':dict(collections.Counter(r[version+'_category'] for r in rr))}
        contrasts={};pairs=[]
        for step in (128,256,512,768,1024):
            for form in ('M','H'):pairs.append((f'ROTATING_{step:04d}_{form}',f'FIXED_{step:04d}_{form}'))
            for arm in ('FIXED','ROTATING'):
                pairs.append((f'{arm}_{step:04d}_H',f'{arm}_{step:04d}_M'))
                pairs.extend((f'{arm}_{step:04d}_H',control) for control in ('D','P'))
        for a,b in pairs:
            if a not in labels or b not in labels:continue
            ia,ib=labels.index(a),labels.index(b);complete=not conditions[a]['missing_draws'] and not conditions[b]['missing_draws']
            lo=lower[:,ia]-upper[:,ib];hi=upper[:,ia]-lower[:,ib]
            contrasts[a+'-'+b]={'difference':float(lo.mean()) if complete else None,
                'ci95':np.quantile(blo[:,ia]-blo[:,ib],[.025,.975]).tolist() if complete else None,
                'possible_mean_bounds':[float(lo.mean()),float(hi.mean())],
                'conservative_bootstrap_bounds':[float(np.quantile(blo[:,ia]-bhi[:,ib],.025)),float(np.quantile(bhi[:,ia]-blo[:,ib],.975))],
                'task_gains':int((lo>0).sum()),'task_losses':int((hi<0).sum()),'task_ties':int(((lo==0)&(hi==0)).sum()),'complete':complete}
        variants[version]={'conditions':conditions,'contrasts':contrasts}
        for i,tid in enumerate(tasks):
            for j,name in enumerate(labels):task_rows.append({'version':version,'task_id':tid,'condition':name,'success_lower':float(lower[i,j]),'success_upper':float(upper[i,j])})
    return {'task_clusters':len(tasks),'joint_resamples':resamples,'bootstrap_seed':seed,'variants':variants,
        'method':'Average three fixed draws within each task. Identical joint task resamples across all conditions and old/new variants. Percentile95% intervals, exploratory unadjusted. Missing infrastructure is not scored wrong or dropped; bounds assign missing0/1 conservatively. No best-of-three.'},task_rows,index


def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(root,output=None):
    root=Path(root);out=Path(output) if output else root/'report';out.mkdir(parents=True,exist_ok=True)
    plan=read(root/'rescore_plan.json');manifest=read(root/'rescored_answer_manifest.json')
    if manifest['plan_sha256']!=digest(plan) or manifest['committed']!=plan['expected_answers']:raise ValueError('Rescore plan coverage/binding differs')
    expected={r['record_id']:r for r in plan['records']};rows=[];seen=set()
    for item in manifest['files']:
        path=root/item['path'];record=read(path)
        if sha(path)!=item['sha256'] or record['binding']['plan_sha256']!=digest(plan):raise ValueError('Scoring receipt drift')
        rid=record['binding']['record_id'];frozen=expected[rid]
        if rid in seen or record['binding']['answer_sha256']!=frozen['answer_sha256'] or record['binding']['original_score_sha256']!=frozen['sha256']:raise ValueError('Duplicate or wrong original receipt identity')
        seen.add(rid);old,new=record['original_score'],record['score_v2']
        if new['passed'] is None and not new.get('missing'):raise ValueError('Null outcome without missing receipt')
        rows.append({'record_id':rid,'task_id':record['task_id'],'condition':label(record),'step':record['step'],
            'seed_index':record['seed_index'],'answer_seed':record['answer_seed'],
            'old_passed':old['passed'],'new_passed':new['passed'],'old_category':old['category'],'new_category':new['category'],
            'missing':new.get('missing',False),'change_cause':cause(old,new),
            'old_output_ceiling':any('File too large' in r.get('stderr','') for r in old.get('receipts',[])),
            'original_answer_path':record['original_answer_path'],'original_answer_sha256':frozen['answer_sha256'],
            'original_score_path':record['original_score_path'],'original_score_sha256':frozen['sha256'],
            'new_score_path':item['path'],'new_score_sha256':item['sha256']})
    if seen!=set(expected):raise ValueError('Missing rescore records')
    analysis,task_rows,indices=statistics(rows,seed=plan['analysis']['bootstrap_seed'],resamples=plan['analysis']['joint_bootstrap_resamples'])
    diagnostics=[]
    for selected in plan['diagnostic_subset']:
        records=[]
        for rep in range(plan['diagnostic_repetitions']):
            path=root/'diagnostics'/(selected['record_id']+f'_repeat_{rep}')/'score_v2.json'
            if path.exists():
                value=read(path)
                if value['binding']['plan_sha256']!=digest(plan):raise ValueError('Diagnostic plan drift')
                records.append({'repeat':rep,'passed':value['score_v2']['passed'],'category':value['score_v2']['category'],'path':str(path.relative_to(root)),'sha256':sha(path)})
        diagnostics.append({**selected,'repetitions':records,'complete':len(records)==plan['diagnostic_repetitions'],
            'pass_fail_varied':len({r['passed'] for r in records if r['passed'] is not None})>1})
    summary={**analysis,'answers':len(rows),'missing':sum(r['missing'] for r in rows),'plan_sha256':digest(plan),
        'manifest_sha256':sha(root/'rescored_answer_manifest.json'),'scorer_identity':plan['scorer_identity'],
        'change_causes':dict(collections.Counter(r['change_cause'] for r in rows)),
        'output_ceiling_cases':[r for r in rows if r['old_output_ceiling']],
        'diagnostics':diagnostics,'diagnostic_varied_records':sum(r['pass_fail_varied'] for r in diagnostics),
        'diagnostic_complete':all(r['complete'] for r in diagnostics),'original_scores_preserved':True,'candidate_programs_unchanged':True}
    write(out/'summary.json',summary);write(out/'joint_bootstrap_indices.json',indices.tolist());csv_write(out/'per_draw_comparison.csv',rows);csv_write(out/'per_task_comparison.csv',task_rows)
    lines=['# Scorer repair: saved coverage-v2 validation','',
        'This uniformly rescored the saved programs; no new answers were generated or repaired. Original scores and receipts remain unchanged. These 21 previously examined tasks provide exploratory validation, not fresh confirmation.','',
        f"Coverage: {len(rows)}/{plan['expected_answers']} score transactions; {summary['missing']} unresolved infrastructure outcomes. Three draws per task/condition; {len({r['task_id'] for r in rows})} task clusters.",'',
        '| Condition (update 1024) | Original passes | Corrected passes | Corrected rate / 95% interval |','|---|---:|---:|---|']
    names=['FIXED_1024_M','ROTATING_1024_M','FIXED_1024_H','ROTATING_1024_H','D','P']
    for name in names:
        if name not in analysis['variants']['new']['conditions']:continue
        old=analysis['variants']['old']['conditions'][name];new=analysis['variants']['new']['conditions'][name]
        interval='missing coverage' if new['pass_rate'] is None else f"{new['pass_rate']*100:.1f}% [{new['ci95'][0]*100:.1f},{new['ci95'][1]*100:.1f}]"
        lines.append(f"|{name}|{old['passed_draws']}/{old['draws']}|{new['passed_draws']}/{new['draws']} ({new['missing_draws']} missing)|{interval}|")
    key='ROTATING_1024_M-FIXED_1024_M'
    if key in analysis['variants']['new']['contrasts']:
        old=analysis['variants']['old']['contrasts'][key];new=analysis['variants']['new']['contrasts'][key]
        for version,r in [('Original',old),('Corrected',new)]:
            lines+=['',f"{version} primary ROTATING−FIXED pure-mapping contrast: "+(f"{r['difference']*100:+.1f} percentage points [{r['ci95'][0]*100:+.1f},{r['ci95'][1]*100:+.1f}]." if r['complete'] else f"unresolved; conservative possible mean bounds {r['possible_mean_bounds']}.")]
    lines+=['','All intervals use 10000 identical paired task-cluster resamples and are exploratory, unadjusted. They do not treat an interval crossing zero as equivalence. Missing draws are neither dropped nor treated as wrong; JSON includes conservative sensitivity bounds.','',
        'The repaired scorer captures stdout completely, including valid multi-megabyte output, with bounded output, CPU, wall, memory, file descriptors and process count. Function-result serialization and discarded function stdout are bounded too. Extraction and comparison import the unchanged original functions. Network, host files, subprocesses and writes remain blocked.','',
        f"Frozen per-test CPU={plan['policy']['cpu_seconds']}s, wall={plan['policy']['wall_seconds']}s. Allowance derives from correct public reference stress runs before candidate rescoring. Only verified infrastructure failures receive one retry; algorithmic timeouts receive none. A crashed score transaction is missing coverage.",'',
        'Limits derive from the [ABC344E constraints](https://atcoder.jp/contests/abc344/tasks/abc344_e) and [linked-list editorial](https://atcoder.jp/contests/abc344/editorial/9503), the [ABC335C constraints](https://atcoder.jp/contests/abc335/tasks/abc335_c) and [history editorial](https://atcoder.jp/contests/abc335/editorial/9293), and the [ABC339D constraints](https://atcoder.jp/contests/abc339/tasks/abc339_d) and [paired-state BFS editorial](https://atcoder.jp/contests/abc339/editorial/9273). Reference source hashes, construction rules, allocation and every repeated measurement are retained in calibration receipts.','',
        'Changed outcomes by recorded cause:']
    lines.extend(f'- {name}: {count}' for name,count in sorted(summary['change_causes'].items()))
    lines+=['',f"The fixed diagnostic subset has {sum(r['complete'] for r in diagnostics)}/{len(diagnostics)} complete three-repeat records; {summary['diagnostic_varied_records']} varied between pass and fail. Every repeat is retained; none substitutes for the uniform rescore. A timeout-to-pass change reflects the calibrated policy/environment and is not automatically labeled a spurious original timeout.",'',
        'Reproduction: run `python scripts/coding_scorer_repair_report.py --root <scorer_repair_root>`. Tables and plots require only the frozen plan and compact score receipts; no private tests, candidate execution, model weights or GPU are used.']
    (out/'SCORER_REPAIR_RESULTS.md').write_text('\n'.join(lines)+'\n')
    render(out,summary,names)
    return summary


def render(out,summary,names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[n for n in names if n in summary['variants']['new']['conditions']]
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(names))
    for version,shift,color in [('old',-.18,'#8d9aa4'),('new',.18,'#27746e')]:
        values=[summary['variants'][version]['conditions'][n]['pass_rate'] for n in names]
        ax.bar(x+shift,[np.nan if v is None else v*100 for v in values],width=.36,label='Original' if version=='old' else 'Corrected',color=color)
    ax.set_xticks(x,[n.replace('_1024','') for n in names]);ax.set(ylabel='Mean task success (%)',ylim=(0,105),title='Saved validation outputs: original and corrected scoring');ax.legend();fig.tight_layout()
    fig.savefig(out/'original_corrected_validation.png',dpi=170);fig.savefig(out/'original_corrected_validation.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output');a=p.parse_args();r=report(a.root,a.output);print(json.dumps({'answers':r['answers'],'missing':r['missing'],'diagnostic_complete':r['diagnostic_complete']}))
if __name__=='__main__':main()
