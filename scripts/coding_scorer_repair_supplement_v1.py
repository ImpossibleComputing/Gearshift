#!/usr/bin/env python3
"""Read-only audit of the immutable delivered scorer-repair ZIP; no execution.

Exports a new versioned supplement. Existing output directories are rejected.
Requires NumPy; no models, private tests, network access, or candidate imports.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

DELIVERED_SHA='03a4713c2714ff388afe1bb418c24c44bd0317f58d5900b7aacbf2fb2b05d94a'
RESULT='results/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/'
ORIGINAL='results/coding_pilot_v1/coverage_generalization_v2_20260918T233720Z/evaluation/'


def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def digest(value):return sha_bytes(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def label(record):
    name=record['condition']
    if name in ('D','P') or name.startswith('START_'):return name
    arm,form=name.rsplit('_',1);return f"{arm}_{record['step']:04d}_{form}"
def interval(row):return f"{row['difference']*100:+.1f} [{row['ci95'][0]*100:+.1f}, {row['ci95'][1]*100:+.1f}]"
def cause(old,new):
    if old['passed']==new['passed']:return 'unchanged_pass' if new['passed'] else 'unchanged_failure'
    if new['passed'] and any('File too large' in r.get('stderr','') for r in old.get('receipts',[])):return 'old_output_ceiling_failure_now_pass'
    if new['passed'] and old['category']=='timeout':return 'old_timeout_now_pass_under_calibrated_policy'
    return 'unexpected_change'


def export(archive,output):
    archive=Path(archive).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('Supplement output already exists; do not overwrite a delivered export')
    if sha(archive)!=DELIVERED_SHA:raise ValueError('Delivered archive differs from immutable receipt')
    with zipfile.ZipFile(archive) as z:
        def read(name):return json.loads(z.read(name))
        inventory=read('SCORER_REPAIR_FILE_MANIFEST.json');names=z.namelist()
        wanted={r['path'] for r in inventory['files']}|{'SCORER_REPAIR_FILE_MANIFEST.json'}
        assert len(names)==len(wanted)==7771 and set(names)==wanted
        assert inventory['private_tests_included'] is False and inventory['weights_or_environments_included'] is False
        for item in inventory['files']:
            p=Path(item['path']);assert not p.is_absolute() and '..' not in p.parts
            assert not set(p.parts)&{'private','.git','.venv','__pycache__','.cache'}
            assert p.suffix not in ('.pt','.bin','.safetensors','.pyc')
            raw=z.read(item['path']);assert len(raw)==item['bytes'] and sha_bytes(raw)==item['sha256']
        summary=read(RESULT+'report/summary.json');plan=read(RESULT+'rescore_plan.json')
        manifest=read(RESULT+'rescored_answer_manifest.json');plan_sha=digest(plan)
        assert plan_sha==summary['plan_sha256']==manifest['plan_sha256']
        assert sha_bytes(z.read(RESULT+'rescored_answer_manifest.json'))==summary['manifest_sha256']
        assert plan['expected_answers']==manifest['committed']==1512 and len(manifest['files'])==1512
        frozen={r['record_id']:r for r in plan['records']};rows=[];seen=set()
        for item in manifest['files']:
            raw=z.read(RESULT+item['path']);assert sha_bytes(raw)==item['sha256'];scored=json.loads(raw)
            binding=scored['binding'];rid=binding['record_id'];assert rid not in seen;seen.add(rid);r=frozen[rid]
            assert binding['plan_sha256']==plan_sha and binding['answer_sha256']==r['answer_sha256'] and binding['original_score_sha256']==r['sha256'] and binding['code_sha256']==r['code_sha256']
            assert binding['scorer_identity']==plan['scorer_identity'] and scored['score_v2']['scorer_identity']==plan['scorer_identity']
            assert sha_bytes(z.read(ORIGINAL+r['answer_path']))==r['answer_sha256']
            assert sha_bytes(z.read(ORIGINAL+r['path']))==r['sha256']
            original=read(ORIGINAL+r['path']);assert original['score']==scored['original_score'] and sha_bytes(original['code'].encode())==r['code_sha256']
            for key in ('task_id','condition','step','seed_index','answer_seed'):assert scored[key]==r[key]
            old,new=scored['original_score'],scored['score_v2']
            assert type(old['passed']) is bool and type(new['passed']) is bool and new['missing'] is False
            rows.append({'record_id':rid,'task_id':r['task_id'],'condition':label(r),'seed_index':r['seed_index'],'answer_seed':r['answer_seed'],
                'old_passed':old['passed'],'new_passed':new['passed'],'old_category':old['category'],'new_category':new['category'],'change_cause':cause(old,new)})
        assert seen==set(frozen) and dict(collections.Counter(r['change_cause'] for r in rows))==summary['change_causes']
        tasks=sorted({r['task_id'] for r in rows});conditions=sorted({r['condition'] for r in rows});assert len(tasks)==21 and len(conditions)==24
        indices=np.asarray(read(RESULT+'report/joint_bootstrap_indices.json'),dtype=np.int64)
        assert np.array_equal(indices,np.random.default_rng(summary['bootstrap_seed']).integers(0,21,(10000,21)))
        condition_rows=[];contrast_rows=[]
        for version in ('old','new'):
            counts=np.zeros((21,24),dtype=np.int64)
            for i,t in enumerate(tasks):
                for j,c in enumerate(conditions):
                    selected=[r for r in rows if r['task_id']==t and r['condition']==c]
                    assert len(selected)==3 and {r['seed_index'] for r in selected}=={0,1,2}
                    counts[i,j]=sum(r[version+'_passed'] for r in selected)
            boot=counts[indices].sum(axis=1)/63
            for j,c in enumerate(conditions):
                result=summary['variants'][version]['conditions'][c]
                assert result['draws']==63 and result['missing_draws']==0 and result['passed_draws']==int(counts[:,j].sum())
                assert np.allclose([result['pass_rate'],*result['ci95']],[counts[:,j].sum()/63,*np.quantile(boot[:,j],[.025,.975])],rtol=0,atol=1e-12)
                condition_rows.append({'version':version,'condition':c,'passes':result['passed_draws'],'draws':63,'missing':0,'pass_rate':result['pass_rate'],'ci95_low':result['ci95'][0],'ci95_high':result['ci95'][1]})
            for name,result in summary['variants'][version]['contrasts'].items():
                a,b=name.split('-');ja,jb=conditions.index(a),conditions.index(b)
                diff=(counts[:,ja]-counts[:,jb]).sum()/63
                ci=np.quantile(boot[:,ja]-boot[:,jb],[.025,.975])
                assert np.allclose([result['difference'],*result['ci95']],[diff,*ci],rtol=0,atol=1e-12)
                contrast_rows.append({'version':version,'contrast':name,'difference':result['difference'],'ci95_low':result['ci95'][0],'ci95_high':result['ci95'][1]})
        diagnostics=[];lookup={r['record_id']:r for r in rows}
        for selected in plan['diagnostic_subset']:
            values=[];categories=[];r=lookup[selected['record_id']]
            for repeat in range(plan['diagnostic_repetitions']):
                v=read(RESULT+f"diagnostics/{selected['record_id']}_repeat_{repeat}/score_v2.json")
                assert v['binding']['record_id']==selected['record_id'] and v['binding']['plan_sha256']==plan_sha and v['binding']['diagnostic_repetition']==repeat
                assert v['binding']['scorer_identity']==plan['scorer_identity'] and v['score_v2']['scorer_identity']==plan['scorer_identity']
                assert type(v['score_v2']['passed']) is bool and not v['score_v2']['missing']
                values.append(v['score_v2']['passed']);categories.append(v['score_v2']['category'])
            diagnostics.append({**{k:r[k] for k in ('record_id','task_id','condition','seed_index','answer_seed')},
                'selection_reason':selected['reason'],'repeat_passes':values,'repeat_categories':categories,'varied':len(set(values))>1})
        assert len(diagnostics)==15 and sum(len(r['repeat_passes']) for r in diagnostics)==45 and not any(r['varied'] for r in diagnostics)
        changes=[r for r in rows if r['old_passed']!=r['new_passed']];assert len(changes)==13 and all(r['new_passed'] for r in changes)
        original_report_sha=sha_bytes(z.read('SCORER_REPAIR_RESULTS.md'))
    verification={'supplement_version':1,'scope':'Saved coverage-v2 validation only; no fresh confirmation and no candidate execution.',
        'source_archive_path':str(archive),'source_archive_sha256':DELIVERED_SHA,'source_archive_bytes':archive.stat().st_size,
        'all_archive_members_verified':7771,'original_report_sha256':original_report_sha,'original_artifacts_unchanged':True,
        'uniform_answers_verified':1512,'original_answer_hashes_verified':1512,'original_score_hashes_verified':1512,
        'corrected_score_bindings_verified':1512,'task_clusters':21,'conditions':24,'draws_per_task_condition':3,'missing':0,
        'joint_bootstrap_resamples':10000,'bootstrap_seed':summary['bootstrap_seed'],'condition_statistics_verified':48,'contrast_statistics_verified':80,
        'diagnostic_records':15,'diagnostic_repeats':45,'diagnostic_pass_fail_variation':0,'change_causes':summary['change_causes'],
        'numpy_version':np.__version__,'limits':'Repeatability applies only to the fixed15-program,45-run diagnostic subset under the calibrated CPU environment. Passing benchmark tests and these diagnostics does not establish universal program correctness or universal scorer stability.'}
    out.mkdir(parents=True)
    dump(out/'verification.json',verification);dump(out/'source_summary.json',summary)
    table(out/'all_conditions_original_corrected.csv',condition_rows);table(out/'all_contrasts_original_corrected.csv',contrast_rows)
    table(out/'changed_outcomes.csv',changes);dump(out/'fixed_repeatability_diagnostics.json',diagnostics)
    lines=['# Scorer repair supplement v1','',
        'This is a new supplementary export. The originally delivered report and review ZIP remain unchanged. It adds all24 condition/checkpoint totals, explicit primary/hybrid contrasts, and the exact limits of the repeatability check. No answers or candidate programs were generated, changed, rerun, or selected for this export.','',
        'The compact archive was verified byte-for-byte, including all7,771 members and every one of the1,512 original answer/original score/corrected score bindings. Coverage is21 previously examined tasks ×24 conditions ×3 draws =1,512 uniform corrected scores;0 are missing. These are exploratory validation results, not fresh confirmation.','',
        'M denotes pure mapping; H combines native original-prompt state with mapped reasoning. D is native full-text replay and P is prompt-only non-thinking. START is the shared update-0 initializer; numbered rows identify additional training updates. Each condition has63 draws.','',
        '| Condition | Original passes | Corrected passes | Change |','|---|---:|---:|---:|']
    for c in conditions:
        old=summary['variants']['old']['conditions'][c]['passed_draws'];new=summary['variants']['new']['conditions'][c]['passed_draws']
        lines.append(f'|{c}|{old}/63|{new}/63|{new-old:+d}|')
    lines+=['','At the frozen update-1,024 endpoint, the original and corrected contrasts are below. Entries are percentage-point differences [95% interval]. All use the same10,000 paired task-cluster resamples, averaging all three draws within each of21 tasks. These validation intervals are exploratory and unadjusted for multiple comparisons. An interval touching or crossing zero is inconclusive, not equivalence.','',
        '| Contrast | Original | Corrected |','|---|---:|---:|']
    for name in summary['variants']['new']['contrasts']:
        if '1024' in name:lines.append(f"|{name}|{interval(summary['variants']['old']['contrasts'][name])}|{interval(summary['variants']['new']['contrasts'][name])}|")
    lines+=['',
        'Across all1,512 saved draws,13 changed from failure to pass and none from pass to failure:9 had the old output-ceiling failure;4 changed from timeout to pass under the calibrated CPU/wall policy.711 passes and788 failures were unchanged. The13 exact task/condition/seed records and categories are in changed_outcomes.csv. A timeout-to-pass change is not proof that its historical timeout was spurious; the resource policy and controlled CPU environment changed.','',
        'The fixed diagnostic subset contains15 saved programs:9 audited output-ceiling failures plus6 predeclared timing-audit cases. Each was run3 times, for45 diagnostic executions. All45 committed without unresolved infrastructure failure;0 of15 programs varied between pass and fail. All repeats are retained, and none replaces a uniform score or selects a favorable outcome. Exact membership and outcomes are in fixed_repeatability_diagnostics.json.','',
        'This establishes repeatability only for those15 programs, those45 runs, and the calibrated dedicated CPU environment. It does not establish universal scorer stability, eliminate all residual timing sensitivity, or prove that every passing program is correct for every valid input. A benchmark pass means the unchanged candidate passed the retained hidden tests under the frozen repaired policy.','',
        'The pure-mapping ROTATING-minus-FIXED result remains+20.6 percentage points [11.1,31.7]. The corrected ROTATING hybrid exceeds P by+22.2 points [3.2,42.9] and remains below D by28.6 points [11.1,46.0]. Its advantage over FIXED hybrid is+11.1 points [0.0,25.4], which includes zero. This validation run contains neither an independently reasoning small-model baseline nor a large-model baseline; it cannot establish superiority to those baselines, a useful speedup, or fresh-task generalization.','',
        'The complete condition statistics and all40 paired contrasts per scoring version are included as CSV files. The independent check recomputed their means and intervals from unchanged receipts and the saved joint task resamples, matching within1e-12.','',
        f'Original review ZIP SHA-256: `{DELIVERED_SHA}`. Original report SHA-256: `{original_report_sha}`.','',
        'Reproduce into a new directory: `python coding_scorer_repair_supplement_v1.py --archive <original gearshift_scorer_repair_review.zip> --output <new supplement directory>`. This reads the compact archive only. It does not access private tests, import candidate programs, load weights, or execute candidates.']
    lines+=['', 'The repaired policy remains frozen at12 CPU seconds and46 wall seconds per test,4 GiB address space, and complete bounded stdout capture with16 MiB minimum/64 MiB maximum allowance. Output serialization and discarded function stdout are also bounded. Original extraction and comparison are unchanged; network, process creation, host-file access and writes remain blocked. One verified infrastructure retry is allowed; algorithmic timeouts receive none. Limits were calibrated from trusted public reference stress cases before the uniform rescore.']
    lines+=['','Exact changed outcomes (all are failure → pass):','',
        '| Task | Condition | Seed index | Original category | Recorded change cause |','|---|---|---:|---|---|']
    for r in changes:
        lines.append(f"|{r['task_id']}|{r['condition']}|{r['seed_index']}|{r['old_category']}|{r['change_cause']}|")
    lines+=['','Exact fixed repeatability subset (each row is one saved program, tested three times):','',
        '| Task | Condition | Seed index | Selection | Repeat outcomes |','|---|---|---:|---|---|']
    for r in diagnostics:
        values=', '.join('pass' if v else 'fail' for v in r['repeat_passes'])
        lines.append(f"|{r['task_id']}|{r['condition']}|{r['seed_index']}|{r['selection_reason']}|{values}|")
    # Keep prose spacing readable while preserving scientific identifiers.
    replacements={'all24':'all 24','all7,771':'all 7,771','the1,512':'the 1,512','is21':'is 21','×24':'× 24','×3':'× 3','=1,512':'= 1,512',';0':'; 0','has63':'has 63','same10,000':'same 10,000','of21':'of 21','all1,512':'all 1,512',',13':', 13',':9':': 9',';4':'; 4','.711':'. 711','and788':'and 788','The13':'The 13','contains15':'contains 15','plus6':'plus 6','run3':'run 3','for45':'for 45','All45':'All 45',';0 of15':'; 0 of 15','those15':'those 15','those45':'those 45','remains+20.6':'remains +20.6','by+22.2':'by +22.2','by28.6':'by 28.6','is+11.1':'is +11.1','at12':'at 12','and46':'and 46',',4 GiB':', 4 GiB','with16':'with 16','of15':'of 15','all40':'all 40','within1e-12':'within 1e-12'}
    text='\n'.join(lines)+'\n'
    for a,b in replacements.items():text=text.replace(a,b)
    (out/'SCORER_REPAIR_RESULTS.md').write_text(text)
    shutil.copyfile(__file__,out/Path(__file__).name)
    files=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()]
    dump(out/'FILE_MANIFEST.json',{'version':1,'source_archive_sha256':DELIVERED_SHA,'private_tests_included':False,'files':files})
    zip_path=out.with_suffix('.zip')
    if zip_path.exists():raise ValueError('Supplement ZIP already exists')
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):z.write(p,p.name)
    with zipfile.ZipFile(zip_path) as z:
        for r in files:assert sha_bytes(z.read(r['path']))==r['sha256']
    receipt={'supplement_directory':str(out),'archive_path':str(zip_path),'archive_bytes':zip_path.stat().st_size,'archive_sha256':sha(zip_path),
        'report_path':str(out/'SCORER_REPAIR_RESULTS.md'),'report_sha256':sha(out/'SCORER_REPAIR_RESULTS.md'),'source_archive_unchanged':sha(archive)==DELIVERED_SHA}
    dump(zip_path.with_suffix('.receipt.json'),receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(export(a.archive,a.output),indent=2))
