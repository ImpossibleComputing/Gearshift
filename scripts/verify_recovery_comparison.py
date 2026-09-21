#!/usr/bin/env python3
"""Read-only cross-check of final recovery results against immutable baselines."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import sha,digest,write
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def verify():
    summary=read(ROOT/'results/coding_pilot_v1/recovery_review_20260917/development_summary.json')
    if summary['coverage']['committed_complete']!=40:raise ValueError('Full forty-task comparison required')
    selection=read(ROOT/summary['training_root']/'selection.json');selection_sha=sha(ROOT/summary['training_root']/'selection.json')
    if selection['development_used_for_selection'] or selection['confirmation_used_for_selection']:raise ValueError('Outcome-based checkpoint selection')
    baseline=ROOT/'results/coding_pilot_v1/cap_recovery_parallel_06/worker_00'
    expected={x['task_id'] for x in read(ROOT/'data/coding_pilot_v1/visible/development.json')};seen=set();workers=[];transactions={}
    counts={k:0 for k in ['A','B','D_original','C','C_initial','D_matched']}
    from gearshift.coding_parallel import confirmation_input
    for rid in summary['run_ids']:
        plan=read(ROOT/'evidence/coding_pilot_v1/control/parallel'/rid/'plan.json')
        if any(confirmation_input(rel) for rel in plan['files']):raise ValueError('Reserved confirmation inputs present')
        for root in sorted((ROOT/'results/coding_pilot_v1'/rid).glob('*')):
            if not (root/'complete.json').exists():continue
            identity=read(root/'identity.json');complete=read(root/'complete.json');native=read(root/'native_gate.json')
            if complete['identity_sha256']!=digest(identity) or complete['selection_sha256']!=selection_sha or not native['passed'] or native['identity_sha256']!=digest(identity):raise ValueError('Worker identity/selection/native controls differ')
            if identity['stage_identity']!=digest(plan):raise ValueError('Worker does not bind its dispatch plan')
            for arm in ['C','C_initial']:
                control=read(root/(arm+'_injection_control.json'))
                if not control['passed'] or control['metric']['max_abs']!=0 or not control['metric']['top1_equal']:raise ValueError('Mapped reinjection control failed')
            for tid,h in complete['tasks'].items():
                if tid in seen or tid not in expected:raise ValueError('Duplicate or foreign task')
                seen.add(tid);task=root/'tasks'/tid.replace('/','__');old=baseline/'tasks'/tid.replace('/','__')
                receipt=read(task/'complete.json')
                if sha(task/'complete.json')!=h or receipt['identity_sha256']!=digest(identity):raise ValueError('Task identity differs')
                for name,fh in receipt['files'].items():
                    if Path(name).name!=name or sha(task/name)!=fh:raise ValueError('Raw transaction differs')
                for current,original in [('source_history','source_history'),('A_original','A'),('B_original','B'),('D_original','D')]:
                    if sha(task/(current+'.json'))!=sha(old/(original+'.json')):raise ValueError('Saved source/baseline was changed')
                source=read(task/'source_history.json');source_hash=sha(task/'source_history.json')
                matched=read(task/'D_matched.json')
                for arm in ['C','C_initial','D_matched']:
                    row=read(task/(arm+'.json'))
                    if any(row[k]!=matched[k] for k in ['answer_seed','rng_initial','bridge_token_count']):raise ValueError('Current handoff output contracts differ')
                    if row['source_history_sha256']!=source_hash or row['score']['passed']!=receipt['pass'][arm]:raise ValueError('Handoff source or score receipt differs')
                    if arm.startswith('C'):
                        want=selection['checkpoint']['sha256'] if arm=='C' else selection['initialization_sha256']
                        if row['checkpoint_sha256']!=want or row['historical_receiver_prefill_tokens']!=0:raise ValueError('Mapped checkpoint/prefill differs')
                    elif row['historical_receiver_prefill_tokens']!=len(source['prefix_ids']):raise ValueError('Native control did not replay full source prefix')
                for k in counts:counts[k]+=receipt['pass'][k]
                transactions[tid]={'receipt':str((task/'complete.json').relative_to(ROOT)),'sha256':h,'source_history_sha256':source_hash}
            workers.append({'root':str(root.relative_to(ROOT)),'identity_sha256':digest(identity),'native_control_passed':True,'selection_sha256':selection_sha,'completed':len(complete['tasks'])})
    if seen!=expected:raise ValueError('Coverage differs')
    if (counts['A'],counts['B'],counts['D_original'])!=(34,27,34):raise ValueError('Original baseline counts changed')
    for k,target in [('A','A'),('B','B'),('C','C'),('C_initial','C_initial'),('D_original','D_original'),('D_matched','D')]:
        if counts[k]!=summary['arms'][target]['passes']:raise ValueError('Report count differs')
    historical=read(ROOT/'evidence/coding_pilot_v1/recovery_20260917/historical_preservation.json')
    for name,row in historical['files'].items():
        if sha(ROOT/name)!=row['expected']:raise ValueError('Historical report changed: '+name)
    out={'passed':True,'task_count':40,'counts':counts,'workers':workers,'transactions':transactions,'historical_reports_unchanged':True,'confirmation_used':False,'development_summary_sha256':sha(ROOT/'results/coding_pilot_v1/recovery_review_20260917/development_summary.json')}
    write(ROOT/'evidence/coding_pilot_v1/recovery_20260917/final_comparison_verification.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['transactions','workers']}))
if __name__=='__main__':verify()
