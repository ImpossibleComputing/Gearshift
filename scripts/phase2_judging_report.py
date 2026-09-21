#!/usr/bin/env python3
"""Describe actual blinded judgment coverage; never invent labels for pending pairs."""
import collections
import datetime as dt
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read,write
from scripts.phase2_report import judge_tables,md_table

ROOT=Path(__file__).resolve().parents[1]


def main():
    root=ROOT/'results/phase2_v1';dest=root/'report';dest.mkdir(exist_ok=True)
    active=ROOT/'evidence/phase2/studio_transfer/active_judge_job.json'
    job_name=read(active)['job'] if active.exists() else 'final_judge_queue_02_after_reset'
    job_path=ROOT/'evidence/phase2/studio_transfer/jobs'/job_name/'status.json'
    job=read(job_path) if job_path.exists() else {}
    queue=read(root/'judging_queue_status.json') if (root/'judging_queue_status.json').exists() else {}
    state='running' if job.get('state')=='running' else queue.get('state','not_started')
    if job.get('state') in ['failed','deadline']:state='infrastructure_failure_requires_review'
    if (ROOT/'evidence/phase2/studio_transfer/JUDGING_PAUSED.json').exists():state='paused_by_owner_after_partial_result_inspection'
    coverage=[];summary=[];contrasts=[];invalid=[];settings=set();usage=collections.Counter();seconds=[]
    stages=read(root/'judging_execution_policy.json')['primary_stage_order']
    for stage in ['development',*stages]:
        folder=root/stage/'judging'
        if not (folder/'condition_key.json').exists():continue
        table,ci,status=judge_tables(root/stage,dest)
        summary.extend(table);contrasts.extend(ci)
        keys=read(folder/'condition_key.json');pair_counts=collections.Counter();scored=0;attempts=0
        for key in keys:
            path=folder/'judgments'/key['packet_id']/'result.json'
            if not path.exists():continue
            result=read(path);attempts+=1
            if result['status']=='scored':scored+=1;pair_counts[key['pair_id']]+=1
            else:invalid.append(dict(stage=stage,packet_id=key['packet_id'],error=result.get('error'),service_blocked=result.get('service_blocked',False)))
            settings.add((result.get('requested_model'),result.get('requested_reasoning_effort'),result.get('cli_version')))
            seconds.append(result.get('wall_seconds',0))
            for u in result.get('usage',[]):
                for k,v in u.items():
                    if isinstance(v,(int,float)):usage[k]+=v
        coverage.append(dict(stage=stage,exported_orientations=len(keys),attempted=attempts,valid=scored,
            invalid=attempts-scored,unattempted=len(keys)-attempts,complete_pairs=sum(v==2 for v in pair_counts.values()),
            single_valid_orientation_pairs=sum(v==1 for v in pair_counts.values()),
            scope='development diagnostics only' if stage=='development' else 'final evaluation'))
    final=[x for x in coverage if x['stage']!='development']
    totals={k:sum(x[k] for x in final) for k in ['exported_orientations','attempted','valid','invalid','unattempted','complete_pairs','single_valid_orientation_pairs']}
    report=dict(recorded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),state=state,coverage=coverage,final_totals=totals,
        invalid_attempts=invalid,settings=[dict(model=m,reasoning_effort=e,cli_version=v) for m,e,v in sorted(settings)],
        usage_metadata=dict(usage),aggregate_session_wall_seconds=sum(seconds),queue_terminal_state=queue if state!='running' else None,
        human_labels='pending',method='Only complete two-orientation pairs enter preference/acceptability summaries. Orientations remain one task; parent cluster bootstrap uses 5000 resamples. No inferred labels.')
    write(dest/'judging_coverage.json',report)
    lines=['# Blinded judging results',f'**Status: {state.replace("_"," ")}.** Final-evaluation coverage is {totals["valid"]} valid orientations out of {totals["exported_orientations"]} exported, forming {totals["complete_pairs"]} complete two-order pairs. There are {totals["invalid"]} invalid attempted orientations and {totals["unattempted"]} unattempted orientations. Development results are listed separately and do not enlarge final evaluation.',
        'The queue follows the previously frozen resource-prioritized order: balanced primary-comparison subsets across every stage, remaining primary comparisons, then secondary comparisons. Included usage is checked before each six-orientation batch, retaining more than 15% in every exposed core allowance window. The owner separately approved one free reset after an allowance stop. No paid fallback or further reset is assumed. Resource-limited coverage is not a random sample of every exported comparison; do not generalize aggregate coverage or selectively finished rows to the entire matrix.',
        md_table(['Stage','Exported orientations','Valid','Invalid','Unattempted','Complete pairs','One valid order'],[[x['stage'],x['exported_orientations'],x['valid'],x['invalid'],x['unattempted'],x['complete_pairs'],x['single_valid_orientation_pairs']] for x in coverage]),
        '**Owner-directed pause:** this pause followed inspection of partial results because task-validity and scoring-resolution concerns require review and priority is shifting toward coding. It was not prespecified. Subjective evaluation is incomplete; the judged subset is not representative of the whole matrix. See PAUSE_STATUS.md.',
        '## Method and limits',
        '**Task validity caveat:** all characterization evidence parents, and their derived long-output/branching evidence variants, have an undefined pricing unit. EVIDENCE_AUDIT.md documents both interpretations and a separate numeric sensitivity analysis. Those evidence judgments assess the original imperfect packet; they do not resolve the task ambiguity. Confirmation evidence and writing tasks are unaffected by this particular defect.',
        'Each orientation used a separate fresh session containing only the task/evidence, anonymous candidates and frozen rubric. The verified Studio calibration matched all six expected outcomes. Outside-packet canary access was denied; tools were still advertised, so any observed tool use invalidates an attempt. The backend immutable snapshot and temperature were unavailable and are not inferred. Exact requested model, reasoning effort, CLI version, prompt/schema/isolation hashes, raw events and available usage metadata are retained per attempt.',
        'Preference can be left, right, tie or neither acceptable. Preference disagreement across A/B and B/A is reported separately rather than resolved by choosing an order. Acceptability and rubric dimensions are averaged over the two orientations for a complete pair; all dimensions must meet the fixed threshold and material factual/causal violations fail acceptability. A single valid orientation is retained as evidence but excluded from paired summaries.',
        'Warning: all-zero or all-identical bootstrap intervals are degenerate empirical intervals and do not establish population equivalence. The tables below are descriptive by family, comparison and output stratum. Values in the acceptance columns may include half-credit when the two orders disagree. Confidence intervals below resample original parent tasks, keeping repeated variants together. No noninferiority or equivalence claim is made. Incomplete coverage and the many exploratory comparisons limit inference.']
    for stage in stages:
        rows=[r for r in summary if r['stage']==stage]
        lines.append('## '+stage)
        if not rows:lines.append('No complete valid two-orientation pairs yet. No semantic score is inferred.');continue
        lines.append(md_table(['Family / stratum','Left vs right','Pairs / parents','Left / right preferred','Tie / neither','Preference order disagreements','Acceptable left / right'],[
            [r['family']+' / '+r['stratum'],r['left']+' vs '+r['right'],f'{r["n_pairs"]} / {r["n_clusters"]}',f'{r["left_preferred"]} / {r["right_preferred"]}',f'{r["tie"]} / {r["neither_acceptable"]}',r['order_disagreement'],f'{100*r["left_acceptable"]:.1f}% / {100*r["right_acceptable"]:.1f}%'] for r in rows]))
        cis=[r for r in contrasts if r['stage']==stage]
        lines.append(md_table(['Family / stratum','Contrast','Acceptability difference, pp [95% interval]','Parent clusters'],[
            [r['family']+' / '+r['stratum'],r['a']+' minus '+r['b'],f'{100*r["difference"]:.1f} [{100*r["ci_low"]:.1f}, {100*r["ci_high"]:.1f}]',r['n_clusters']] for r in cis]))
    lines+=['## Human and disagreement review',
        'Thirty distinct pre-reserved human pairs (15 per prose family) and an offline form are in `results/phase2_v1/human_review/blinded/`. Human labels remain pending and do not block the research deliverable. The condition key is separate. Selectively identified preference, acceptability or dimension order disagreements are exported to `disagreements_blinded/`; these are diagnostic and must not enter representative human-agreement percentages.',
        'Machine-readable coverage is in `results/phase2_v1/report/judging_coverage.json`; per-stage judge summaries, paired intervals and matched quality/latency plots are in the same report directory. Branching uses the recorded three-output amortized cost, with each method paying source/prefix preparation once. The raw candidates and judgments permit independent review without trusting this narrative.']
    (ROOT/'JUDGING_RESULTS.md').write_text('\n\n'.join(lines)+'\n')
    print(json.dumps(dict(state=state,**totals),indent=2))


if __name__=='__main__':main()
