"""Objective checks; field/constraint checks do not claim semantic prose quality."""
from __future__ import annotations
import re
from .phase2_sandbox import grade_code


def numeric(text):
    from decimal import Decimal,InvalidOperation
    boxed=re.findall(r'\\boxed\{([^{}]+)\}',text)
    nums=re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',boxed[-1] if boxed else text)
    if not nums:return None
    try:return str(Decimal(nums[-1].replace(',','')).normalize())
    except InvalidOperation:return None


def fields(text,label):
    lines=re.findall(r'^\s*(?:\*\*)?'+re.escape(label)+r'(?:\*\*)?\s*:\s*(?:\*\*)?(.+?)\s*$',text,re.M|re.I)
    return [line.strip().strip('*') for line in lines]


def field(text,label):
    # Repetition is a format issue. Contradictory values remain unresolved, never cherry-picked.
    values=fields(text,label)
    return values[0] if values and len(set(values))==1 else None


def semantic_decision(values,task):
    names=list(task.hidden.get('totals',{}));resolved=[]
    for value in values:
        found={name for name in names if re.search(r'(?<![\w-])'+re.escape(name)+r'(?![\w-])',value,re.I)}
        for i,name in enumerate(names):
            if re.search(r'(?<!\w)S'+str(i+2)+r'(?!\w)',value,re.I):found.add(name)
        if len(found)!=1:return None
        resolved.extend(found)
    return resolved[0] if resolved and len(set(resolved))==1 else None


def semantic_number(values):
    values=[numeric(v) for v in values]
    return values[0] if values and values[0] is not None and len(set(values))==1 else None


def grade(task,answer,sandbox_passed=False):
    words=re.findall(r"\b[\w'-]+\b",answer)
    base=dict(nonempty=bool(answer.strip()),words=len(words),
              repeated_4grams=max(0,len(words)-3)-len({tuple(words[i:i+4]) for i in range(max(0,len(words)-3))}))
    if task.grader=='numeric':
        parsed=numeric(answer);valid=bool(re.fullmatch(r'\s*[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?\s*',answer))
        return dict(**base,correct=parsed==task.hidden['gold'],format_valid=valid,parsed_answer=parsed,
                    correct_and_valid=parsed==task.hidden['gold'] and valid,primary_metric='correct')
    if task.grader=='humanevalplus':
        result=grade_code(task,answer,sandbox_passed)
        return dict(**base,code=result,correct=result['passed'],primary_metric='hidden_test_pass',
                    semantic_quality_status='objective executable test contract')
    if task.grader=='evidence_fields':
        raw_fields={label:fields(answer,label) for label in ['Decision','Monthly total','Difference']}
        decision=semantic_decision(raw_fields['Decision'],task)
        total=semantic_number(raw_fields['Monthly total']);difference=semantic_number(raw_fields['Difference'])
        allowed={task.hidden['decision']}
        if len(set(task.hidden.get('totals',{}).values()))==1 and task.hidden.get('eligibility_update_satisfied',False):
            allowed=set(task.hidden['totals'])  # The visible task does not prescribe a tie-break.
        checks=dict(decision=decision in allowed,monthly_total=total==numeric(str(task.hidden['monthly_total'])),
            difference=difference==numeric(str(task.hidden['difference'])))
        format_valid=all(len(v)==1 for v in raw_fields.values())
        conflicting={label:len(values)>1 and (semantic_decision(values,task) if label=='Decision' else semantic_number(values)) is None
                     for label,values in raw_fields.items()}
        sources={s:bool(re.search(r'\['+s+r'\]',answer)) for s in task.hidden['required_sources']}
        return dict(**base,field_checks=checks,all_fields_correct=all(checks.values()),field_format_valid=format_valid,
            all_fields_correct_and_valid=all(checks.values()) and format_valid,field_coverage=sum(bool(v) for v in raw_fields.values())/3,
            semantic_field_accuracy=sum(checks.values())/3,conflicting_fields=conflicting,raw_fields=raw_fields,
            source_marker_coverage=sum(sources.values())/len(sources),source_markers=sources,
            parsed_fields=dict(decision=decision,monthly_total=total,difference=difference),primary_metric='all_fields_correct',
            semantic_quality_status='pending_blind_judge; normalized field checks do not establish all prose claims or usability')
    if task.grader=='writing_constraints':
        h=task.hidden;checks={s:s.casefold() in answer.casefold() for s in h['required']}
        if 'once' in h:checks['required_phrase_once']=answer.count(h['once'])==1
        if 'ending' in h:checks['exact_ending']=answer.strip().endswith(h['ending'])
        checks['word_range']=h['word_range'][0]<=len(words)<=h['word_range'][1]
        return dict(**base,constraint_checks=checks,all_literal_constraints=all(checks.values()),literal_constraint_fraction=sum(checks.values())/len(checks),
            primary_metric='all_literal_constraints',semantic_quality_status='pending_blind_judge; literal constraints do not establish usefulness/style')
    if task.grader=='evidence_prose':
        markers={s:bool(re.search(r'\['+s+r'\]',answer)) for s in task.hidden['required_sources']}
        return dict(**base,source_markers=markers,source_marker_coverage=sum(markers.values())/len(markers),primary_metric=None,
            semantic_quality_status='pending_blind_judge; source markers alone do not establish claim support or useful coverage')
    return dict(**base,primary_metric=None,semantic_quality_status='unscored_training')
