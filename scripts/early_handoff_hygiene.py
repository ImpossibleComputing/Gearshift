#!/usr/bin/env python3
"""Scan staged public changes against locally fetched official benchmark material.

Defense in depth, not a license/legal proof. Only paths and categories are emitted.
The reviewed release baseline is unchanged; this gate scans every staged blob.
"""
import argparse,ast,hashlib,json,re,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def words(text):return re.findall(r'\w+',text.casefold())
def windows(text):
 w=words(text)
 return {' '.join(w[i:i+24]) for i in range(max(0,len(w)-23))}
def forbidden(value):
 if isinstance(value,dict):
  for k,v in value.items():
   if k in {'prompt','prompt_ids','input_ids','generated_ids','answer_text','private_test_cases','public_test_cases','question_content','starter_code','rng_state'}:return True
   if forbidden(v):return True
 if isinstance(value,list):
  if len(value)>=24 and all(type(x) is int for x in value):return True
  return any(forbidden(x) for x in value)
 return False
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=ROOT/'data/coding_pilot_v1/raw');p.add_argument('--bundle',type=Path,help='Scan every ZIP member instead of the staged diff');a=p.parse_args()
 if a.bundle:
  with zipfile.ZipFile(a.bundle) as z:
   names=z.namelist()
   if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('Unsafe ZIP member paths')
   blobs={n:z.read(n) for n in names}
 else:
  names=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACMR','-z'],cwd=ROOT).decode().split('\0');names=[n for n in names if n]
  if not names:raise SystemExit('No staged changes to scan')
  blobs={n:subprocess.check_output(['git','show',':'+n],cwd=ROOT) for n in names}
 issues=[];by_window={}
 for name,b in blobs.items():
  if name.startswith('data/') or Path(name).suffix in {'.safetensors','.pt','.pth','.npy','.npz','.pkl','.zip'}:issues.append((name,'forbidden artifact type'))
  if b'\x00' in b:
   if Path(name).suffix!='.png':issues.append((name,'unreviewed binary'))
   continue
  text=b.decode('utf8')
  for w in windows(text):by_window.setdefault(w,set()).add(name)
  if name.endswith('.json') and forbidden(json.loads(text)):issues.append((name,'raw field or reversible numeric array'))
  if name.endswith('.py'):
   for node in ast.walk(ast.parse(text)):
    if isinstance(node,(ast.List,ast.Tuple)) and len(node.elts)>=24 and all(isinstance(x,ast.Constant) and type(x.value) is int for x in node.elts):issues.append((name,'long literal numeric array'))
  if re.search(r'\[(?:\s*\d+\s*,){23,}\s*\d+\s*\]',text):issues.append((name,'long numeric array'))
 from scripts.coding_prepare import decode_tests
 seen=0
 for path in sorted(a.raw.glob('*.jsonl')):
  with path.open() as f:
   for line in f:
    row=json.loads(line);seen+=1
    for key in ('question_content','starter_code','public_test_cases'):
     for match in windows(str(row.get(key,''))) & by_window.keys():
      for name in by_window[match]:issues.append((name,'upstream 24-word overlap:'+key))
    # Exact substantial test fragments as well as word runs; no test text is output.
    tests=json.loads(row['public_test_cases'])+decode_tests(row['private_test_cases'])
    for test in tests:
     for value in test.values():
      if isinstance(value,str) and len(value)>=80:
       for name,b in blobs.items():
        if value.encode() in b:issues.append((name,'upstream test fragment'))
 if seen!=1055:raise ValueError('Expected all 1055 pinned upstream rows for scan')
 tracked=subprocess.check_output(['git','ls-files','data/'],cwd=ROOT,text=True).strip()
 if tracked:issues.append(('data/','ignored experiment storage tracked'))
 diff=b'\n'.join(b for n,b in sorted(blobs.items()) if b'\x00' not in b) if a.bundle else subprocess.check_output(['git','diff','--cached','--binary'],cwd=ROOT)
 leak=subprocess.run(['gitleaks','stdin','--redact','--no-banner','--no-color'],input=diff,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if leak.returncode:issues.append(('staged changes','secret scanner failed or found a secret'))
 print(json.dumps({'scan_scope':'bundle' if a.bundle else 'staged','paths':names,'official_rows_scanned':seen,'issues':sorted(set(issues)),'gitleaks_exit':leak.returncode,'scanned_payload_sha256':hashlib.sha256(diff).hexdigest()},indent=2))
 if issues:raise SystemExit(1)
if __name__=='__main__':
 sys.path.insert(0,str(ROOT));main()
