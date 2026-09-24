#!/usr/bin/env python3
"""Build a review-only ZIP from an explicit whitelist of committed public blobs."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PATHS=[
 'EARLY_HANDOFF_RESULTS.md','REPRODUCE_EARLY_HANDOFF.md','PROVENANCE.md',
 'LICENSE','NOTICE','THIRD_PARTY_NOTICES.md',
 'third_party/HumanEvalPlus_RELEASE_LICENSE.txt','third_party/HumanEval_LICENSE.txt',
 'third_party/EvalPlus_LICENSE.txt','third_party/Qwen3-8B_LICENSE.txt','third_party/Qwen3-32B_LICENSE.txt',
 'third_party/GSM8K_LICENSE.txt','third_party/README.md',
 'configs/early_handoff_01/declaration.json','configs/early_handoff_01/Scorer.Dockerfile',
 'gearshift/early_handoff.py','gearshift/coding_sandbox.py','gearshift/coding_sandbox_v2.py',
 'scripts/early_handoff_dependencies.py','scripts/early_handoff_prepare.py',
 'scripts/early_handoff_worker.py','scripts/early_handoff_score.py','scripts/early_handoff_pipeline.py',
 'scripts/early_handoff_report.py','scripts/early_handoff_audit.py','scripts/early_handoff_finalize.py',
 'scripts/early_handoff_hygiene.py','scripts/early_handoff_bundle.py',
 'scripts/fetch_livecodebench.py','scripts/coding_sandbox_child_v2.py',
 'tests/test_early_handoff.py',
 'evidence/coding_pilot_v1/confirmation_01_20260919T094418Z/scorer_repair/calibration/frozen_policy.json']
RESULTS=['analysis_input.json','per_draw.json','per_draw.csv','per_task.csv','task_outcomes.csv',
 'condition_summary.csv','paired_comparisons.csv','summary.json','missingness_sensitivity.json',
 'audit.json','frontier.png','verification.json']

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'gearshift_early_handoff_review.zip');a=p.parse_args()
 def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
 if git('branch','--show-current').decode().strip()!='research/early-handoff-01':raise RuntimeError('Wrong branch')
 if git('status','--porcelain').strip():raise RuntimeError('Review and commit safe changes before bundling')
 commit=git('rev-parse','HEAD').decode().strip()
 paths=PATHS+['results/early_handoff_01/'+n for n in RESULTS]
 blobs={n:git('show',commit+':'+n) for n in paths}
 blobs['BUNDLE_README.md']=(
  '# Gearshift early-handoff review\n\n'
  'Read EARLY_HANDOFF_RESULTS.md first. This ZIP is a compact review subset, not the full repository. '
  'No benchmark text, tests, raw answers, reversible tokens, models or private archive are included.\n\n'
  'For full tests and raw reproduction, clone https://github.com/ImpossibleComputing/Gearshift '
  'and check out the release_commit in MANIFEST.json. Do not rerun costly inference merely to review. '
  'The bundled report/audit/finalize scripts support safe numeric regeneration with numpy 2.5.3 '
  'and matplotlib 3.11.2; the raw audit and full experiment require the complete repository. '
  'The final report renderer is intentionally specific to this completed screen.\n\n'
  'MANIFEST.json binds every other file to SHA256 and identifies the reviewed research commit. '
  'Private original raw evidence is retained locally in ignored experiment storage. '
  'Original main/progress-02 and private historical archive remain unchanged.\n').encode()
 manifest={'schema':'gearshift.early_handoff.review_bundle.v1','public_repository':'https://github.com/ImpossibleComputing/Gearshift',
  'branch':'research/early-handoff-01','release_commit':commit,'generation_implementation_commit':'bf85c69508589c8f7027af3cdbe10b91904ed921',
  'scope':'Review-only safe numeric evidence, code, configs, reports, licenses and provenance. No new experiments.',
  'files':{n:{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for n,b in sorted(blobs.items())}}
 blobs['MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n,b in sorted(blobs.items()):
   info=zipfile.ZipInfo(n,date_time=(2026,9,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,b)
 with zipfile.ZipFile(a.output) as z:
  assert z.testzip() is None and set(z.namelist())==set(blobs)
  for n,b in blobs.items():assert z.read(n)==b
 print(json.dumps({'zip':str(a.output.resolve()),'release_commit':commit,'files':len(blobs),'bytes':a.output.stat().st_size,'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
