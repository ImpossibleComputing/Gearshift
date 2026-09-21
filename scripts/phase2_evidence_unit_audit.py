#!/usr/bin/env python3
"""Post-hoc unit-interpretation sensitivity audit; never replaces frozen task grades."""
import collections
import dataclasses
import re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.phase2_io import read,write,artifact
from gearshift.phase2_grading import numeric
from gearshift.phase2_reporting import csv_write
from scripts.phase2_report import md_table

ROOT=Path(__file__).resolve().parents[1]


def alternatives(task):
    prompt=task['prompt']
    match=re.search(r'Expected monthly volume is (\d+) thousand requests\.',prompt)
    if not match:return None
    volume=int(match[1])
    offers=re.findall(r'\[S[23]\] ([^:\n]+): monthly fixed fee (\d+) credits; rate (\d+) credits per unit\.',prompt)
    if len(offers)!=2:raise ValueError('Unexpected ambiguous-unit task structure')
    choices={name:{'per_thousand':int(fixed)+volume*int(rate),'per_request':int(fixed)+1000*volume*int(rate)} for name,fixed,rate in offers}
    selected=task['hidden']['decision']
    result={label:dict(monthly_total=choices[selected][label],difference=abs(choices[offers[0][0]][label]-choices[offers[1][0]][label])) for label in ['per_thousand','per_request']}
    for key,value in result['per_thousand'].items():
        if value!=task['hidden'][key]:raise ValueError('Frozen hidden formula differs from diagnosed interpretation')
    return result


def main():
    root=ROOT/'results/phase2_v1';dest=root/'evidence_unit_audit';dest.mkdir(exist_ok=True)
    tasks={x['task_id']:x for x in read(root/'tasks/characterization.json') if x['family']=='evidence'}
    cases={tid:alternatives(t) for tid,t in tasks.items()}
    if not all(cases.values()):raise ValueError('Expected consistent characterization evidence template')
    rows=[]
    scored=read(root/'characterization/objective_scores.json')['rows']
    for row in scored:
        if row['family']!='evidence':continue
        tid=row['task_id'];s=row['score'];a=cases[tid];parsed=s['parsed_fields']
        matches={label:bool(s['field_checks']['decision'] and all(parsed[k]==numeric(str(v)) for k,v in expected.items())) for label,expected in a.items()}
        if matches['per_thousand']!=s['all_fields_correct']:raise ValueError('Sensitivity baseline fails to reproduce frozen field score')
        rows.append(dict(task_id=tid,condition=row['condition'],decision_correct=s['field_checks']['decision'],
            recorded_per_thousand_all_fields=matches['per_thousand'],alternative_per_request_all_fields=matches['per_request'],
            either_interpretation_all_fields=any(matches.values()),parsed_total=parsed['monthly_total'],parsed_difference=parsed['difference']))
    grouped=collections.defaultdict(list)
    for row in rows:grouped[row['condition']].append(row)
    summary=[dict(condition=c,n=len(rs),decision_correct=sum(r['decision_correct'] for r in rs),
        recorded_per_thousand=sum(r['recorded_per_thousand_all_fields'] for r in rs),alternative_per_request=sum(r['alternative_per_request_all_fields'] for r in rs),
        either_interpretation=sum(r['either_interpretation_all_fields'] for r in rs)) for c,rs in sorted(grouped.items())]
    affected=[]
    for stage in sorted(root.iterdir()):
        if not (stage/'manifest.json').exists():continue
        identity=read(stage/'manifest.json').get('identity',{})
        visible=identity.get('tasks',[])
        if isinstance(visible,list):
            ids=[t['task_id'] for t in visible if isinstance(t,dict) and 'thousand requests' in t.get('prompt','')]
            if ids:affected.append(dict(stage=stage.name,planned_task_variants=len(ids),task_ids=ids))
    metadata=dict(status='post_hoc_diagnostic_only',discovered='During final compact-bundle qualitative review after all objective scoring; no inference, hidden grader, selection, source record or judge packet changed.',
        issue='Volume uses thousand requests while the price says per unit without defining whether one unit is one request or one thousand requests. The frozen hidden grader uses the latter.',
        original_inputs={'tasks':artifact(root/'tasks/characterization.json'),'scores':artifact(root/'characterization/objective_scores.json')},
        audit_source=artifact(Path(__file__)),affected_stages=affected,interpretations=cases,summary=summary,
        caution='Alternative parsing holds the recorded answer and existing field extraction fixed. Either-interpretation matches are a sensitivity diagnostic, not corrected primary scores or full semantic acceptability.')
    write(dest/'audit.json',metadata);csv_write(dest/'case_sensitivity.csv',rows);csv_write(dest/'summary.csv',summary)
    example=cases[sorted(cases)[0]]
    lines=['# Evidence-task unit audit',
        '**The characterization evidence template leaves the pricing unit ambiguous. Its frozen numeric field scores cannot support an unqualified factual-accuracy claim.** Monthly volume is stated in thousand requests, but vendor prices say credits per unit without identifying a unit. The hidden formula multiplies by the displayed number of thousands. Treating a unit as one request instead multiplies by 1,000. This issue was discovered in final qualitative review, after generation and objective scoring.',
        'For `characterization_evidence_000`, the packet gives 28 thousand requests and the eligible vendor charges 230 plus 4 per unit. The hidden monthly total is 342 and difference 148. Under a per-request interpretation, the corresponding values are 112,230 and 28,120. The source-native and native-replay answers use the second interpretation. Their disagreement with the hidden answer is not by itself evidence of arithmetic failure.',
        'All 80 characterization evidence parents share this template. Their derived long-output and branching evidence variants, two evidence timing cases, and the archived partial laptop evidence records inherit the same ambiguity. Training, validation, development, confirmation and the separate contextual-memory diagnostic use different templates and are unaffected by this particular issue. The 4B confirmation uses the unambiguous confirmation template. Raw stage memberships are listed in `audit.json`.',
        md_table(['Characterization condition','N','Decision field correct','Frozen per-thousand all fields','Alternative per-request all fields','Either interpretation'],[[r['condition'],r['n'],r['decision_correct'],r['recorded_per_thousand'],r['alternative_per_request'],r['either_interpretation']] for r in summary]),
        'This sensitivity analysis reuses the same recorded parsed fields and decisions. It changes only the two numeric reference values in a separate analysis; it does not rewrite `objective_scores.json`, score manifests, task files or any candidate output. T and P have 32 cases each, so their aggregate counts are not comparable to the 80-case conditions without matching membership. Neither the alternative nor the either-interpretation column is a replacement primary score.',
        'Blinded judgments remain tied to the exact original task/evidence packet and never receive hidden grading answers. The ambiguous numerical instruction can still affect their factual assessments. Treat characterization and derived-extension evidence judgments as exploratory observations on an imperfect task, rather than a clean semantic benchmark. Preference for a complete, consistent answer may remain informative, but numerical disagreements need this qualification. Writing judgments and the separate confirmation evidence tasks are not affected by this unit defect.',
        'For any future experiment, prices must explicitly say per request or per thousand requests and graders must check that visible-unit conversion. That would require new task identities and fresh outputs. No further inference or retrospective task repair was performed in this bounded run.']
    (ROOT/'EVIDENCE_AUDIT.md').write_text('\n\n'.join(lines)+'\n')
    print('Audited',len(cases),'ambiguous parents and',len(rows),'recorded answers; original scores preserved')


if __name__=='__main__':main()
