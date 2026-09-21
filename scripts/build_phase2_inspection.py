#!/usr/bin/env python3
"""Build the inspection snapshot after an owner pause; no inference or judging."""
import argparse, ast, csv, datetime as dt, hashlib, json, os, re, shlex, shutil, subprocess, sys, zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase2_inspection_diagnostics import generate, read, write, csvout, STAGES
ROOT=Path(__file__).resolve().parents[1]
def artifact(p):
    data=Path(p).read_bytes();return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows]])
def copy(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
def compact_files():
    forbidden={'.git','.venv','venv','__pycache__','.pytest_cache','.cache','node_modules'}
    extensions={'.py','.json','.jsonl','.csv','.md','.txt','.png','.svg','.html','.yaml','.yml','.toml','.ini','.patch','.diff','.gz','.lock','.plist'}
    candidates=[]
    for dirname in ['gearshift','scripts','configs','tests','results/phase2_v1','evidence/phase2']:
        for p in (ROOT/dirname).rglob('*'):
            rel=p.relative_to(ROOT)
            if not p.is_file() or p.is_symlink() or forbidden & set(rel.parts) or p.suffix not in extensions:continue
            if rel.parts[:3]==('evidence','phase2','studio_transfer'):
                allowed=rel.parts[3:4]==('inspection_pause_20260915',) or p.name in ['JUDGING_PAUSED.json','compute_completion_verified.json','final_cloud_cleanup_verified.json','billing_post_cleanup_20260915.json']
                if not allowed:continue
            if p.name.startswith('.env') or p.name in ['auth.json','credentials.json']:raise ValueError('Credential filename')
            if p.stat().st_size>64*1024**2:raise ValueError('Unexpected heavy compact file: '+str(rel))
            candidates.append(p)
    for name in ['PHASE2_RESULTS.md','JUDGING_RESULTS.md','EVIDENCE_AUDIT.md','PAUSE_STATUS.md','CODING_PILOT_PLAN.md','REPRODUCE_PHASE2.md','REVIEW_GUIDE.md','requirements.lock.txt','requirements.txt','pyproject.toml','.gitignore']:
        p=ROOT/name
        if p.exists():candidates.append(p)
    return sorted(set(candidates))
def pending_order():
    root=ROOT/'results/phase2_v1';seen=set();pending=[]
    for round_name,scope in [('priority','primary'),('primary','primary'),('all','all')]:
        for stage in STAGES:
            folder=root/stage/'judging';keys=read(folder/'condition_key.json');primary='M'
            selection=root/'training'/stage.removeprefix('confirmation_')/'selection.json'
            if stage.startswith('confirmation_') and selection.exists():
                sel=read(selection);primary=f'M/{sel["selected_objective"]}/{sel["representative_seed"]}'
            if scope=='primary':
                keys=[k for k in keys if (k['left']==primary and k['right'] in ['C','B','B/newturn','B/native','M/frozen','M/initial']) or ('/boundary/' in k['left'] and '/ordinary/' in k['right'])]
            if round_name=='priority':
                chosen=set(read(root/'judging_priority_cases'/f'{stage}.json')['task_ids']);keys=[k for k in keys if k['task_id'] in chosen]
            for k in sorted(keys,key=lambda k:k['packet_id']):
                ident=(stage,k['packet_id'])
                if ident in seen or (folder/'judgments'/k['packet_id']/'result.json').exists():continue
                seen.add(ident);pending.append(dict(queue_index=len(pending),round=round_name,scope=scope,stage=stage,**k))
    return pending
def assert_paused():
    marker=read(ROOT/'evidence/phase2/studio_transfer/JUDGING_PAUSED.json')
    if marker['allow_new_calls'] or marker['inflight_service_calls']!=0:raise ValueError('Pause has not drained')
    active=[]
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2:continue
        try:args=shlex.split(fields[1])
        except ValueError:continue
        if len(args)>1 and 'python' in Path(args[0]).name and Path(args[1]).name in ['phase2_judge_queue.py','phase2_judge_resume.py','phase2_finish_review.py','phase2_review_recovery.py','phase2_judge.py']:
            active.append(line.strip())
        if args and Path(args[0]).name=='judge-cli-0.154.0-alpha.6.2' and 'exec' in args[1:2]:active.append(line.strip())
    if active:raise ValueError('A judge or finalizer is still active')
    return marker
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=ROOT/'inspection_exports/phase2_20260915');args=ap.parse_args()
    pause=assert_paused();snapshot_utc=dt.datetime.now(dt.timezone.utc).isoformat()
    saved=read(ROOT/'evidence/phase2/studio_transfer/inspection_pause_20260915/judgments_snapshot.json')
    for path,a in saved.items():
        if artifact(ROOT/path)!=a:raise ValueError('Saved judgment changed: '+path)
    dest=args.directory
    if dest.exists():raise ValueError('Inspection destination already exists; use another output directory')
    dest.mkdir(parents=True)
    diagnostic=generate(ROOT,dest/'03_diagnostics')
    pending=pending_order();write(dest/'04_provenance/pending_dispatch_order.json',pending)
    stagecov=list(csv.DictReader((dest/'03_diagnostics/coverage_by_stage.csv').open()))
    groupcov=list(csv.DictReader((dest/'03_diagnostics/coverage_by_stage_family_comparison.csv').open()))
    totals=diagnostic['final_coverage']
    text='# Phase-2 judging pause\n\n'
    text+=f'Pause requested {pause["requested_utc"]}; drain verified {pause["paused_utc"]}; snapshot {snapshot_utc} (UTC).\n\n'
    text+='Judging was paused by the owner after inspection of partial results because task-validity and scoring-resolution concerns require review, and research priority is shifting toward coding. This was not a prespecified stopping rule. Subjective evaluation is incomplete; the judged subset is not representative of the whole matrix.\n\n'
    text+=f'The final-evaluation snapshot has {totals["valid_orientations"]} valid judgments, {totals["invalid_orientations"]} invalid judgments, {totals["complete_two_order_pairs"]} complete two-order pairs, {totals["single_valid_order_pairs"]} single-order pairs, and {totals["pending_unattempted_orientations"]} unattempted orientations. Development is separate. There are zero in-flight service calls. No extra calls were launched to complete pairs.\n\n'
    text+='## Verified operation\n\nThe dedicated executable was temporarily gated against new service dispatch while existing service processes finished and their original runner saved results. The original executable was restored with SHA-256 '+pause['original_cli_restored']['sha256']+'. One attempted local dispatch was intercepted before reaching the service; it is not a judgment attempt and has no fabricated result. Its packet ID was not retained by the temporary intercept receipt. Original result files were hash-verified. A completed CLI process by itself is not treated as a saved judgment: coverage counts only atomic result.json files.\n\n'
    text+='The queue wrapper, dispatch runner and automatic finalizer are stopped. The specific recovery launch agent was disabled, unloaded and its plist archived before removal. The laptop heartbeat monitor is PAUSED with its former continuation instructions superseded. Explicit pause guards now block the queue, resume wrapper, recovery helper and finalizer. No reset or paid fallback is authorized. Unrelated work was not touched. Historical process status files remain evidence and may still say running; this verified pause receipt takes precedence.\n\n'
    text+='## Stage coverage\n\n'+md(['Stage','Exported','Valid','Invalid','Complete pairs','Single order','Pending'],[[r[k] for k in ['stage','exported_orientations','valid_orientations','invalid_orientations','complete_two_order_pairs','single_valid_order_pairs','pending_unattempted_orientations']] for r in stagecov])+'\n\n'
    text+='## Stage, family and comparison coverage\n\n'+md(['Stage','Family','Left','Right','Exported','Valid','Complete pairs','Single order','Pending'],[[r[k] for k in ['stage','family','left','right','exported_orientations','valid_orientations','complete_two_order_pairs','single_valid_order_pairs','pending_unattempted_orientations']] for r in groupcov])+'\n\n'
    text+='## Eventual resumption — not authorized now\n\n'
    text+='The active job at pause was final_judge_queue_05_after_service_recovery, in the characterization remaining-primary pass. The exact deduplicated remaining dispatch order is 04_provenance/pending_dispatch_order.json in the inspection ZIP. Its first item is '+json.dumps({k:pending[0][k] for k in ['stage','round','scope','packet_id']})+'. Already saved records, including any unscored attempts, must be reused by packet hash. Do not launch calls merely to complete a pair.\n\n'
    text+='After a new explicit owner instruction to resume this same frozen study: (1) verify original result and runtime hashes against this snapshot; (2) archive, then remove evidence/phase2/studio_transfer/JUDGING_PAUSED.json as an explicitly recorded authorization change; (3) load the environment mapping in configs/phase2_studio_runtime.json; (4) create a new detached phase2_studio_job spec with commands [["scripts/phase2_judge_queue.py"]], a new job name and a recorded pointer update. The queue rechecks the existing priority, primary and secondary order and skips every saved result. Run the existing provenance-only resume wrapper; do not weaken source/rubric/isolation validation. Recheck ordinary allowance under the unchanged 15% reserve. Keep the recovery agent and old finalizer disabled unless separately needed and explicitly re-authorized. No new reset or spending is implied.\n\n'
    text+='A direct foreground launch, only after those authorization and hash checks, is: load configs/phase2_studio_runtime.json["environment"] into os.environ, then runpy.run_path("scripts/phase2_judge_queue.py", run_name="__main__") under .venv/bin/python from the Studio repository. The frozen judging module itself is retained for audit; do not bypass the guarded entry points.\n\n'
    text+='## Limits of operational evidence\n\nThe dispatcher had no built-in pause handler; the temporary executable gate prevented any new service call while allowing version queries needed to save in-flight results. The local intercept receipt does not identify its pending packet, so no packet identity is invented. The immutable packets and result inventory define the exact remaining workload. There is no remaining unverified stop action. No model inference, mapper training, new subjective judgment or paid provisioning was performed for this inspection export.\n'
    (ROOT/'PAUSE_STATUS.md').write_text(text)
    # Exact original representative sample, including original offline viewer and blank form.
    original=ROOT/'results/phase2_v1/human_review/blinded'
    shutil.copytree(original,dest/'01_blinded/reserved_30')
    human_key=read(ROOT/'results/phase2_v1/human_review/separate_condition_key.json')
    selected=[dict(review_id=k['human_packet_id'],stage='characterization',pair_id=k['pair_id'],kind='reserved') for k in human_key]
    diagnostic_selection=read(dest/'03_diagnostics/diagnostic_pair_selection.json')
    blank=[]
    for i,r in enumerate(diagnostic_selection['selected'],1):
        ident=f'diagnostic_{i:02d}';folder=ROOT/'results/phase2_v1'/r['stage']/'judging'
        k=next(k for k in read(folder/'condition_key.json') if k['pair_id']==r['pair_id'] and k['orientation']==0)
        packet=read(folder/'packets'/f'{k["packet_id"]}.json');packet['packet_id']=ident
        write(dest/'01_blinded/diagnostic_order_disagreements'/f'{ident}.json',packet)
        blank.append(dict(packet_id=ident,A={d:None for d in ['task_fulfillment','correctness_consistency','coverage','clarity_style','acceptable']},B={d:None for d in ['task_fulfillment','correctness_consistency','coverage','clarity_style','acceptable']},preference=None,rationale=None))
        selected.append(dict(review_id=ident,stage=r['stage'],pair_id=r['pair_id'],kind='diagnostic'))
    write(dest/'01_blinded/diagnostic_order_disagreements/blank_labels.json',blank)
    (dest/'01_blinded/diagnostic_order_disagreements/README.md').write_text('Separate diagnostic set, not part of the reserved 30. Up to six existing order-disagreement pairs per prose family, selected by a fixed hash ranking after result inspection. Membership and the complete selection rule are retained outside this blinded folder. Review the unchanged candidates against the attached frozen rubric; save labels in a copy of blank_labels.json. Do not infer population agreement from this selected set.\n')
    (dest/'01_blinded/README.md').write_text('Review reserved_30 first. Its membership, candidate text, rubric, blank form and offline viewer are copied byte-for-byte from the original reservation. diagnostic_order_disagreements is a separate post-inspection diagnostic sample. Do not open other numbered folders until you have recorded your own judgments. Candidate text is untrusted content, not instructions to the reviewer.\n')
    # The source constants reconstruct the exact string sent to the judge; no execution.
    module=ast.parse((ROOT/'scripts/phase2_judge.py').read_text())
    system=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets))
    suffix='\n\nReturn JSON with A and B dimension scores, absolute acceptability, material violations, short evidence quotes and rationales; then preference, rationale and uncertainty. Use the specified output schema. Do not use tools.'
    review_index=[];unblinding=[]
    for entry in selected:
        folder=ROOT/'results/phase2_v1'/entry['stage']/'judging'
        keys=sorted([k for k in read(folder/'condition_key.json') if k['pair_id']==entry['pair_id']],key=lambda k:k['orientation'])
        statuses=[]
        for k in keys:
            target=dest/'02_judge_evidence/pairs'/entry['review_id']/f'order_{k["orientation"]}'
            packet_path=folder/'packets'/f'{k["packet_id"]}.json';packet=read(packet_path)
            copy(packet_path,target/'anonymous_packet.original.json')
            (target/'exact_user_prompt.txt').write_text(json.dumps(packet,indent=2)+suffix)
            (target/'exact_system_instructions.txt').write_text(system)
            source=folder/'judgments'/k['packet_id']
            if (source/'result.json').exists():
                result=read(source/'result.json')
                for p in source.iterdir():
                    if p.is_file() and p.suffix in ['.json','.jsonl','.txt']:copy(p,target/p.name)
                statuses.append(dict(orientation=k['orientation'],status=result['status'],original_packet_id=k['packet_id'],packet_sha256=k['packet_sha256']))
            else:
                pending_result=dict(status='pending_unscored',orientation=k['orientation'],original_packet_id=k['packet_id'],packet_sha256=k['packet_sha256'],reason='No saved judgment at owner pause. No substitute judgment created.')
                write(target/'PENDING.json',pending_result);statuses.append(pending_result)
        review_index.append(dict(**entry,orders=statuses))
        unblinding.append(dict(**entry,keys=keys))
    write(dest/'02_judge_evidence/review_pair_status.json',review_index)
    write(dest/'04_provenance/exported_pair_condition_key.json',unblinding)
    write(dest/'04_provenance/diagnostic_selection.json',diagnostic_selection)
    copy(ROOT/'results/phase2_v1/tasks/human_reservation.json',dest/'04_provenance/original_human_reservation.json')
    for name in ['configs/phase2_judging.json','scripts/phase2_judge.py','scripts/phase2_judge_probe.py','scripts/phase2_report.py','scripts/phase2_review_packets.py']:
        copy(ROOT/name,dest/'02_judge_evidence/frozen_definition'/Path(name).name)
    first=next(ROOT.glob('results/phase2_v1/characterization/judging/judgments/*/result.json'))
    write(dest/'02_judge_evidence/frozen_definition/output_schema.json',read(first)['schema'])
    (dest/'02_judge_evidence/frozen_definition/exact_runtime_schema.json').write_text(json.dumps(read(first)['schema']))
    for dirname in ['evidence/phase2/judge_isolation_studio_v2','results/phase2_v1/judge_calibration_studio_v2']:
        source=ROOT/dirname
        if source.exists():
            for p in source.rglob('*'):
                if p.is_file() and p.suffix in ['.json','.jsonl','.txt','.py']:copy(p,dest/'02_judge_evidence/checks'/source.name/p.relative_to(source))
    # Full compact input records and source snapshots are kept in an explicitly unblinding tree.
    for p in compact_files():copy(p,dest/'04_provenance/repository'/p.relative_to(ROOT))
    for name in ['PHASE2_RESULTS.md','JUDGING_RESULTS.md','EVIDENCE_AUDIT.md','PAUSE_STATUS.md','CODING_PILOT_PLAN.md']:
        copy(ROOT/name,dest/'04_provenance'/name)
    copy(ROOT/'EVIDENCE_AUDIT.md',dest/'03_diagnostics/EVIDENCE_AUDIT.md')
    shutil.copytree(ROOT/'results/phase2_v1/evidence_unit_audit',dest/'03_diagnostics/alternative_unit_analysis')
    copy(ROOT/'scripts/phase2_prepare_tasks.py',dest/'03_diagnostics/original_task_templates.py')
    (dest/'03_diagnostics/README.md').write_text('All tables here are newly computed, separately labeled post-inspection diagnostics from existing records. No frozen grades or rubric thresholds were changed. Run ../04_provenance/repository/scripts/phase2_inspection_diagnostics.py with --repo ../04_provenance/repository and --output a NEW directory to reproduce the diagnostic tables and saved-code exports without models, execution of candidate code or network access.\n\nCoverage denominators are explicit. Candidate observations count repeated orientations/comparisons and must not be treated as independent tasks. Dimension failure causes overlap. Preference disagreement, acceptability disagreement and dimension disagreement are distinct columns; the raw per-pair file identifies the changes. All-identical/all-zero empirical bootstrap intervals cannot establish population equivalence. Alternative-unit scores are sensitivity analysis, never replacements for frozen scores.\n\ncode_outputs preserves verbatim raw outputs, exact saved parsed_code and original execution traces. No code was repaired or rerun. code_per_output.jsonl links exact historical stage/task/condition IDs, original prompt and source reasoning tokens/text, and handoff tokens into the repository snapshot. Categories are conservative trace-derived observations; ambiguous origins remain unknown. Historical stdout/stderr were capped to their last 12000 characters; earlier trace material was not saved. Tiny repeated test seeds are not additional task samples. All historical code segments used greedy argmax; do not mistake this for the new pilot sampling proposal.\n')
    # Record files whose bytes define the paused scientific snapshot.
    protected={}
    for p in ROOT.glob('results/phase2_v1/**/*'):
        if p.is_file() and (p.parent.name in ['questions','scores','packets','tasks'] or p.name in ['result.json','manifest.json','complete.json','objective_scores.json','human_reservation.json']):
            if p.suffix in ['.json','.jsonl']:protected[str(p.relative_to(ROOT))]=artifact(p)
    protected['configs/phase2_judging.json']=artifact(ROOT/'configs/phase2_judging.json')
    write(dest/'04_provenance/scientific_record_hashes.json',protected)
    omissions=['4700 final-evaluation judgment orientations are pending; no new judgments were made for selected review pairs.',
        'Human labels remain blank, as requested.',
        'Judge backend immutable snapshot and temperature were not exposed; unknown. Available raw settings and CLI versions are retained.',
        'Code stdout/stderr earlier than the saved 12000-character tails are unavailable; unresolved failure origins are labeled unknown.',
        'The local pre-dispatch interception receipt did not retain a packet ID; no identity was inferred for it.',
        'Model weights, mapper tensors, KV/paired caches, environments, Git internals, credentials and unrelated/private control logs are excluded.',
        'Coding-pilot GPU fit, throughput, baseline quality and costs have not been measured. No pilot experiment was executed.']
    source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    snapshot=dict(snapshot_utc=snapshot_utc,paused_utc=pause['paused_utc'],source_revision=source_revision,
        source_revision_note='This revision is the baseline before the inspection changes; exact delivered source bytes are covered by MANIFEST.json. The external delivery receipt records the final local commit.',
        working_tree_changed_paths=changed,coverage=diagnostic['final_coverage'],review_pairs=review_index,omissions=omissions,
        exact_reserved_membership_count=len(human_key),diagnostic_pair_count=len(diagnostic_selection['selected']),
        checkpoint_inventory='repository/results/phase2_v1/excluded_checkpoints.json',
        scoring='Existing primary scores and thresholds unchanged. Diagnostics are separate post-inspection analyses.',
        publication='Local review export only; no publication, push or project license choice.')
    write(dest/'04_provenance/SNAPSHOT.json',snapshot)
    (dest/'REVIEW_README.md').write_text('# Gearshift phase-2 inspection\n\nThis is an incomplete, owner-paused study snapshot for independent inspection, not a completed subjective evaluation. The pause followed partial-result inspection because task validity and scoring resolution need review and priority is shifting toward coding.\n\n1. Start with 01_blinded/reserved_30/review.html or the unchanged JSON packets; record your own labels before viewing other folders. These are exactly the 30 pre-reserved pairs. The separately labeled diagnostic set contains up to 12 selected disagreement pairs and is not representative.\n2. Open 02_judge_evidence. Each matching anonymous review ID contains the exact prompts and any original saved results for both orders. PENDING.json means no judgment exists. Do not mistake missing judgments for failure or agreement. Raw metadata exposes only available settings; immutable backend version and temperature remain unknown.\n3. Open 03_diagnostics for coverage, failure causes, disagreements and saved code execution diagnostics. These tables identify methods and should be read only after blind labeling. The pricing-unit defect and alternative interpretation are documented separately; original scores remain frozen. All-zero intervals are not evidence of equivalence.\n4. Open 04_provenance/exported_pair_condition_key.json last for the explicit condition mapping. The repository snapshot contains original source reasoning, hidden tests, raw scores, token IDs, manifests and source snapshots; it is unblinding material. Read PAUSE_STATUS.md and the three narrative reports here.\n\nThe proposed larger coding-only pilot is 04_provenance/CODING_PILOT_PLAN.md; draft configs are under 04_provenance/repository/configs/coding_pilot_v1/. It has NOT been executed and needs a separately approved spending cap. Nothing in this ZIP authorizes execution.\n\nSee 04_provenance/SNAPSHOT.json for timestamp, exact coverage and omissions. MANIFEST.json gives every payload file size and SHA-256; it excludes itself to avoid a circular hash. The external receipt supplies the ZIP and manifest hashes. To verify offline, run the included verify_phase2_inspection.py against the ZIP; to reproduce diagnostics, use the stdlib-only script described in 03_diagnostics/README.md. Never run an experimental pipeline to review this archive.\n')
    files={}
    for p in sorted(dest.rglob('*')):
        if p.is_file():files[str(p.relative_to(dest))]=artifact(p)
    write(dest/'MANIFEST.json',dict(schema=1,snapshot_utc=snapshot_utc,files=files,self_excluded=True,omissions=omissions))
    destination=ROOT/'gearshift_phase2_inspection.zip';tmp=destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(dest.rglob('*')):
            if not p.is_file():continue
            data=p.read_bytes()
            if p.suffix not in ['.png','.gz']:
                content=data.decode(errors='replace')
                if re.search(r'(?<![\w-])sk-(?:proj-)?[A-Za-z0-9_-]{32,}',content) or re.search(r'(?m)^-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----\r?$',content):
                    raise ValueError('Possible credential in '+str(p.relative_to(dest)))
            z.writestr(str(p.relative_to(dest)),data)
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
    for path,a in protected.items():
        if artifact(ROOT/path)!=a:raise ValueError('Scientific record changed during snapshot')
    tmp.replace(destination)
    receipt=dict(path=str(destination),**artifact(destination),snapshot_utc=snapshot_utc,manifest=artifact(dest/'MANIFEST.json'),
        files=len(files)+1,uncompressed_bytes=sum(v['bytes'] for v in files.values()),coverage=diagnostic['final_coverage'],state='inspection_snapshot_subjective_judging_paused',pilot_executed=False)
    write(ROOT/'phase2_inspection_receipt.json',receipt)
    copy(dest/'MANIFEST.json',ROOT/'phase2_inspection_manifest.json')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
