#!/usr/bin/env python3
"""Post-pause diagnostics from saved records only. Never executes candidates or calls models."""
import argparse, collections, csv, hashlib, json, re
from pathlib import Path

STAGES=['characterization','long_outputs','branching','confirmation_1p7_to_0p6','confirmation_4b_to_0p6']
DIMS=['task_fulfillment','correctness_consistency','coverage','clarity_style']
def read(p): return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def csvout(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def jsonl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code_category(code):
    status=code.get('status','unknown')
    if code.get('passed') is True:return 'pass','saved_hidden_tests_completed'
    if status in ['syntax_failure','timeout','memory_limit','sandbox_unavailable','pending_sandbox','no_test_completion_marker']:return status,'saved_status'
    err=code.get('stderr','').strip();last=err.splitlines()[-1] if err else ''
    if status=='execution_failure':
        if last=='AssertionError' and 'in assertion' in err and 'assert exact_match' in err:
            return 'hidden_test_assertion_failure','saved_trace_names_grader_assertion'
        if last.startswith('AssertionError'):return 'assertion_failure_origin_unknown','trace_not_sufficient_to_attribute'
        if re.match(r'^[A-Za-z_]\w*(?:Error|Exception)(?::|$)',last):
            return 'runtime_exception','saved_exception_line'
        if 'Operation not permitted' in err or 'Permission denied' in err:
            return 'permission_failure_origin_unknown','trace_does_not_prove_sandbox_was_cause'
        return 'execution_failure_unknown','saved_status_without_resolving_trace'
    return 'unknown','saved_record_does_not_establish_category'

def generate(repo,out):
    repo=Path(repo);out=Path(out);root=repo/'results/phase2_v1'
    all_keys={};all_results={};coverage=[];groups=collections.defaultdict(list);observations=[];pairs=[]
    for stage in ['development',*STAGES]:
        folder=root/stage/'judging';keys=read(folder/'condition_key.json');all_keys[stage]=keys
        bypair=collections.defaultdict(list)
        for k in keys:
            path=folder/'judgments'/k['packet_id']/'result.json'
            r=read(path) if path.exists() else None
            if r:all_results[(stage,k['packet_id'])]=r
            bypair[k['pair_id']].append((k,r))
            groups[(stage,k['family'],k['left'],k['right'])].append((k,r))
            if r and r['status']=='scored':
                for label,method in k['condition_by_label'].items():
                    s=r['judgment'][label]
                    observations.append(dict(stage=stage,family=k['family'],task_id=k['task_id'],cluster_id=k['cluster_id'],
                        pair_id=k['pair_id'],packet_id=k['packet_id'],orientation=k['orientation'],method=method,label=label,
                        **{d:s[d] for d in DIMS},acceptable=s['acceptable'],material_veto=bool(s['material_violations']),
                        material_violations=s['material_violations'],rationale=s['rationale'],evidence_quotes=s['evidence_quotes']))
        for pair_id,items in sorted(bypair.items()):
            valid=[(k,r) for k,r in items if r and r['status']=='scored']
            if len(valid)!=2:continue
            valid.sort(key=lambda v:v[0]['orientation'])
            scored=[];prefs=[]
            for k,r in valid:
                scored.append({method:r['judgment'][label] for label,method in k['condition_by_label'].items()})
                pref=r['judgment']['preference'];prefs.append(k['condition_by_label'].get(pref,pref))
            k=valid[0][0]
            changes={method:[d for d in DIMS if scored[0][method][d]!=scored[1][method][d]] for method in scored[0]}
            accept_changes=[method for method in scored[0] if scored[0][method]['acceptable']!=scored[1][method]['acceptable']]
            pairs.append(dict(stage=stage,pair_id=pair_id,task_id=k['task_id'],cluster_id=k['cluster_id'],family=k['family'],
                left=k['left'],right=k['right'],preference_by_order=prefs,preference_disagreement=prefs[0]!=prefs[1],
                acceptability_disagreement=bool(accept_changes),acceptability_changed_methods=accept_changes,
                dimension_disagreement=any(changes.values()),dimension_changed=changes,
                any_order_disagreement=prefs[0]!=prefs[1] or bool(accept_changes) or any(changes.values()),
                left_acceptable_mean=sum(s[k['left']]['acceptable'] for s in scored)/2,
                right_acceptable_mean=sum(s[k['right']]['acceptable'] for s in scored)/2))
    for (stage,family,left,right),items in sorted(groups.items()):
        valid=[k for k,r in items if r and r['status']=='scored'];attempted=sum(r is not None for k,r in items)
        pc=collections.Counter(k['pair_id'] for k in valid)
        coverage.append(dict(stage=stage,family=family,left=left,right=right,exported_orientations=len(items),
            attempted=attempted,valid_orientations=len(valid),invalid_orientations=attempted-len(valid),
            complete_two_order_pairs=sum(v==2 for v in pc.values()),single_valid_order_pairs=sum(v==1 for v in pc.values()),
            pending_unattempted_orientations=len(items)-attempted,unique_exported_tasks=len({k['task_id'] for k,r in items}),
            unique_judged_tasks=len({k['task_id'] for k in valid})))
    csvout(out/'coverage_by_stage_family_comparison.csv',coverage)
    stagecov=[]
    for stage in ['development',*STAGES]:
        rows=[r for r in coverage if r['stage']==stage]
        stagecov.append(dict(stage=stage,**{k:sum(r[k] for r in rows) for k in ['exported_orientations','attempted','valid_orientations','invalid_orientations','complete_two_order_pairs','single_valid_order_pairs','pending_unattempted_orientations']}))
    csvout(out/'coverage_by_stage.csv',stagecov)
    jsonl(out/'candidate_score_observations.jsonl',observations)
    obs_groups=collections.defaultdict(list)
    for r in observations:obs_groups[(r['stage'],r['family'],r['method'])].append(r)
    failures=[];distributions=[];veto_text=[]
    for (stage,family,method),rows in sorted(obs_groups.items()):
        n=len(rows);failed=[r for r in rows if not r['acceptable']]
        expected=sum(1 for k in all_keys[stage] if k['family']==family and method in k['condition_by_label'].values())
        failures.append(dict(stage=stage,family=family,method=method,exported_candidate_orientations=expected,
            valid_candidate_orientation_observations=n,unique_task_outputs=len({r['task_id'] for r in rows}),failed_acceptability=len(failed),
            accepted=n-len(failed),material_veto=sum(r['material_veto'] for r in rows),
            **{d+'_below_3':sum(r[d]<3 for r in rows) for d in DIMS},
            dimension_only_failure=sum(any(r[d]<3 for d in DIMS) and not r['material_veto'] for r in rows),
            veto_only_failure=sum(all(r[d]>=3 for d in DIMS) and r['material_veto'] for r in rows),
            both_veto_and_dimension_failure=sum(any(r[d]<3 for d in DIMS) and r['material_veto'] for r in rows)))
        for d in DIMS:
            distributions.append(dict(stage=stage,family=family,method=method,dimension=d,denominator=n,
                **{'score_'+str(i):sum(r[d]==i for r in rows) for i in range(5)}))
        for r in rows:
            for v in r['material_violations']:
                veto_text.append({k:r[k] for k in ['stage','family','task_id','pair_id','packet_id','orientation','method']}|{'verbatim_violation':v})
    csvout(out/'acceptability_failure_causes.csv',failures)
    csvout(out/'dimension_score_distributions.csv',distributions)
    csvout(out/'material_violations_verbatim.csv',veto_text)
    jsonl(out/'complete_pair_order_diagnostics.jsonl',pairs)
    disgroups=collections.defaultdict(list)
    for r in pairs:disgroups[(r['stage'],r['family'],r['left'],r['right'])].append(r)
    dis=[]
    for (stage,family,left,right),rows in sorted(disgroups.items()):
        diffs=[r['left_acceptable_mean']-r['right_acceptable_mean'] for r in rows]
        dis.append(dict(stage=stage,family=family,left=left,right=right,complete_pair_denominator=len(rows),
            original_parent_denominator=len({r['cluster_id'] for r in rows}),
            preference_disagreements=sum(r['preference_disagreement'] for r in rows),
            acceptability_disagreements=sum(r['acceptability_disagreement'] for r in rows),
            dimension_disagreements=sum(r['dimension_disagreement'] for r in rows),
            any_order_disagreements=sum(r['any_order_disagreement'] for r in rows),
            identical_observed_acceptability_differences=len(set(diffs))==1,
            all_zero_observed_acceptability_differences=all(x==0 for x in diffs),
            interval_warning='Degenerate empirical bootstrap intervals are not population equivalence' if len(set(diffs))==1 else 'Incomplete exploratory sample; no equivalence claim'))
    csvout(out/'order_disagreement_summary.csv',dis)
    salt='gearshift-inspection-order-diagnostics-20260915'
    chosen=[]
    for family in ['evidence','writing']:
        eligible=[r for r in pairs if r['stage'] in STAGES and r['family']==family and r['any_order_disagreement']]
        eligible.sort(key=lambda r:hashlib.sha256((salt+'|'+r['stage']+'|'+r['pair_id']).encode()).hexdigest())
        chosen+=eligible[:6]
    write(out/'diagnostic_pair_selection.json',dict(rule='Post-inspection diagnostic selection: up to six per family from existing complete pairs with any preference, acceptability or dimension disagreement. SHA-256 rank of salt|stage|pair_id; no fill from another family. Distinct pairs, potentially repeated parent tasks. Not representative; never pooled with the reserved 30.',salt=salt,selected=chosen))
    code_rows=[];code_groups=collections.defaultdict(dict);decoding=[]
    for stage in ['characterization','confirmation_1p7_to_0p6','confirmation_4b_to_0p6']:
        folder=root/stage;manifest=read(folder/'manifest.json');ident=manifest['identity']
        scoremap={(r['task_id'],r['condition']):r for r in read(folder/'objective_scores.json')['rows'] if r['family']=='code'}
        tmap={t['task_id']:t for t in ident['tasks']}
        for path in sorted((folder/'questions').glob('*.json')):
            obj=read(path)
            if tmap[obj['task_id']]['family']!='code':continue
            trajectory=obj['trajectory']
            for row in obj['rows']:
                score=scoremap[(obj['task_id'],row['condition'])]['score'];code=score['code']
                category,basis=code_category(code)
                leaf=hashlib.sha256((stage+'|'+obj['task_id']+'|'+row['condition']).encode()).hexdigest()[:24]
                # These text files are copies of saved fields, never repaired or executed.
                raw_path=out/'code_outputs'/stage/leaf/'raw_output.txt';raw_path.parent.mkdir(parents=True,exist_ok=True)
                raw_path.write_text(row['answer'])
                (raw_path.parent/'extracted_for_execution.py').write_text(code.get('parsed_code',''))
                write(raw_path.parent/'saved_execution_record.json',code)
                source_model=ident['config']['source'];receiver=ident['config']['target']
                cond=row['condition'];answer_model=source_model if cond in ['B','B/newturn','B/native'] else receiver
                item=dict(stage=stage,task_id=obj['task_id'],cluster_id=row['cluster_id'],condition=cond,export_id=leaf,
                    passed=code.get('passed'),failure_category=category,category_basis=basis,saved_execution_status=code.get('status','unknown'),
                    parser_status='saved_extraction_available' if 'parsed_code' in code else 'unknown',
                    syntax_status='failed' if code.get('status')=='syntax_failure' else 'passed' if 'parsed_code' in code else 'unknown',
                    executed=code.get('executed'),tests_completed=code.get('tests_completed'),
                    hidden_test_status='passed' if code.get('passed') is True else 'assertion_failure_observed' if category=='hidden_test_assertion_failure' else 'not_completed' if code.get('tests_completed') is False else 'unknown',
                    timeout_status='timeout' if code.get('status')=='timeout' else 'not_recorded_as_timeout',
                    sandbox_status='saved_verified_sandbox_execution' if code.get('executed') else 'not_executed',
                    runtime_exception_last_line=code.get('stderr','').strip().splitlines()[-1] if code.get('stderr','').strip() else '',
                    stop_reason=row.get('answer_stop_reason','unknown'),answer_tokens=row.get('answer_tokens'),
                    source_reasoning_tokens=0 if cond in ['S','S_think'] else len(trajectory.get('source_reasoning_ids',[])),
                    shared_source_reasoning_tokens=len(trajectory.get('source_reasoning_ids',[])),source_history_used_by_condition=cond not in ['S','S_think'],small_reasoning_tokens=row.get('small_reasoning_tokens'),
                    source_stop_reason=trajectory.get('stop_reason','unknown'),source_reasoning_completed=trajectory.get('source_completed'),
                    answer_model=answer_model,source_model=source_model,greedy_decoding=True,
                    raw_output_sha256=hashlib.sha256(row['answer'].encode()).hexdigest(),
                    extracted_code_sha256=hashlib.sha256(code.get('parsed_code','').encode()).hexdigest(),
                    repeated_4grams=score.get('repeated_4grams'),repetition_denominator=max(0,score.get('words',0)-3),
                    source_record=str(path.relative_to(repo)),source_record_sha256=sha(path),
                    handoff_input_ids=row.get('handoff_input_ids'),prompt_ids_ref='trajectory.prompt_ids in source_record',
                    source_reasoning_ids_ref='trajectory.source_reasoning_ids in source_record',
                    source_reasoning_text_ref='trajectory.source_reasoning_text in source_record')
                code_rows.append(item);code_groups[(stage,obj['task_id'])][cond]=item
        models=[ident['config']['source'],ident['config']['target']]
        for model in models:
            for segment in ['reasoning','answer']:
                decoding.append(dict(stage=stage,model=model,segment=segment,decoding_declaration=ident['decoding'],
                    algorithm='argmax',do_sample=False,temperature='not_applied',top_p='not_applied',top_k='not_applied',
                    model_revision=ident['source' if model==ident['config']['source'] else 'target']['revision'],
                    effective_eos_ids=ident['source' if model==ident['config']['source'] else 'target']['effective_eos_ids'],
                    model_config_sha256=ident['source' if model==ident['config']['source'] else 'target']['config_sha256'],
                    tokenizer_identity=ident['source' if model==ident['config']['source'] else 'target']['tokenizer'],
                    seed=ident['config']['seed'],dtype=ident['config']['dtype'],device=ident['config']['device'],
                    reasoning_budget=ident['config'].get('source_reasoning_caps',{}).get('code'),
                    code_answer_budgets=sorted({t['answer_budget'] for t in ident['tasks'] if t['family']=='code'}),
                    manifest=str((folder/'manifest.json').relative_to(repo))))
    jsonl(out/'code_per_output.jsonl',code_rows)
    csvout(out/'code_per_output.csv',[{k:v for k,v in r.items() if k!='handoff_input_ids'} for r in code_rows])
    write(out/'code_decoding_settings.json',decoding)
    totals=collections.defaultdict(list)
    for row in code_rows:totals[(row['stage'],row['condition'])].append(row)
    csvout(out/'code_failure_totals.csv',[dict(stage=s,condition=c,output_denominator=len(rows),task_denominator=len({r['task_id'] for r in rows}),
        **collections.Counter(r['failure_category'] for r in rows)) for (s,c),rows in sorted(totals.items())])
    paired=[]
    for (stage,task),conds in sorted(code_groups.items()):
        comparisons=[('B/native','B/newturn')]+[('C',m) for m in conds if m=='M' or m.startswith('M/')]
        for left,right in comparisons:
            if left not in conds or right not in conds:continue
            a,b=conds[left],conds[right]
            paired.append(dict(stage=stage,task_id=task,cluster_id=a['cluster_id'],left=left,right=right,left_passed=a['passed'],right_passed=b['passed'],
                left_category=a['failure_category'],right_category=b['failure_category'],
                outcome='both_pass' if a['passed'] and b['passed'] else 'left_only_pass' if a['passed'] else 'right_only_pass' if b['passed'] else 'both_fail'))
    csvout(out/'code_paired_outcomes.csv',paired)
    pg=collections.defaultdict(list)
    for r in paired:pg[(r['stage'],r['left'],r['right'])].append(r)
    csvout(out/'code_paired_summary.csv',[dict(stage=s,left=l,right=r,paired_task_denominator=len(rows),
        **collections.Counter(x['outcome'] for x in rows)) for (s,l,r),rows in sorted(pg.items())])
    final=[r for r in stagecov if r['stage']!='development']
    result=dict(analysis='Separate post-inspection diagnostics; frozen scores and thresholds unchanged.',
        final_coverage={k:sum(r[k] for r in final) for k in final[0] if k!='stage'},
        code_outputs=len(code_rows),code_tasks_by_stage={s:len({r['task_id'] for r in code_rows if r['stage']==s}) for s in ['characterization','confirmation_1p7_to_0p6','confirmation_4b_to_0p6']},
        diagnostic_pairs=len(chosen),candidate_observations=len(observations),
        caveats=['Orientation/candidate counts are dependent repeated assessments, not independent sample sizes.',
        'Unjudged outputs have no inferred ratings. Completed subset is not representative.',
        'Veto causes overlap; free-text vetoes are exported verbatim, not newly judged.',
        'All-zero/all-identical bootstrap intervals do not establish population equivalence.',
        'Execution output was historically saved only as the last 12000 characters per stdout/stderr; missing earlier trace detail cannot be reconstructed.',
        'Unknown failure origin stays unknown; no generated code or model was rerun.',
        'All phase-2 code inference used greedy argmax, including reasoning and answers; sampling parameters were not applied.'])
    write(out/'diagnostic_summary.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(generate(a.repo,a.output),indent=2))
