#!/usr/bin/env python3
"""Pin and separate visible problems from test material; no model execution."""
import base64, hashlib, io, json, pickle, sys, urllib.request, zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gearshift.coding_control import write, sha, bind

ROOT=Path(__file__).resolve().parents[1]
class StringsOnly(pickle.Unpickler):
    def find_class(self, module, name): raise ValueError('Dataset pickle contains an executable global')

def decode_tests(s):
    try: return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(StringsOnly(io.BytesIO(zlib.decompress(base64.b64decode(s)))).load())

def prompt(row):
    starter=row.get('starter_code','')
    contract=('Implement the provided Python interface. Return the result; do not read stdin.\n'+starter
              if starter else 'Write a complete Python program that reads standard input and writes the required output.')
    return ('Solve the following programming problem. Think through the solution, then provide the complete Python 3 code in a single Python code fence.\n\n'
        +row['question_content']+'\n\n'+contract)

def main():
    from transformers import AutoTokenizer
    cfg=json.loads((ROOT/'configs/coding_pilot_v1/pilot.json').read_text())
    membership=json.loads((ROOT/'configs/coding_pilot_v1/draft_membership.json').read_text())
    dest=ROOT/'data/coding_pilot_v1'; dest.mkdir(parents=True,exist_ok=True)
    wanted={(r['platform'],str(r['question_id'])):(split,r) for split,rs in membership['selected'].items() for r in rs}
    tok=AutoTokenizer.from_pretrained(ROOT/'configs/coding_pilot_v1/reference/Qwen3-8B',local_files_only=True)
    visible={}; private={}; seen={}; audit=[]; problems=[]
    for filename, meta in membership['source_files'].items():
        raw=dest/'raw'/filename;raw.parent.mkdir(exist_ok=True)
        if not raw.exists():
            tmp=raw.with_suffix('.partial')
            with urllib.request.urlopen(meta['url'],timeout=120) as src,tmp.open('wb') as out:
                while chunk:=src.read(8*1024**2):out.write(chunk)
            tmp.replace(raw)
        if sha(raw)!=meta['sha256'] or raw.stat().st_size!=meta['bytes']:raise ValueError('Dataset content hash mismatch '+filename)
        with raw.open('rb') as f:
            for line in f:
                row=json.loads(line);key=(row['platform'],str(row['question_id']))
                if key not in wanted:continue
                split,expected=wanted[key];task_id='/'.join(key)
                hashes=[hashlib.sha256(x).hexdigest() for x in [line,line.rstrip(b'\r\n'),json.dumps(row,sort_keys=True).encode(),json.dumps(row,sort_keys=True,separators=(',',':')).encode()]]
                if expected['source_row_sha256'] not in hashes:raise ValueError('Selected row hash mismatch '+task_id)
                text=prompt(row);ids=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=True)
                ph=hashlib.sha256(' '.join(row['question_content'].split()).encode()).hexdigest()
                if ph in seen:problems.append({'task_id':task_id,'duplicate_of':seen[ph]})
                seen[ph]=task_id
                if len(ids)>cfg['protocol']['prompt_max_tokens']:problems.append({'task_id':task_id,'oversized_prompt_tokens':len(ids)})
                visible.setdefault(split,[]).append(dict(task_id=task_id,prompt=text,prompt_ids=ids,**expected))
                # Kept outside the generation input directory, never printed.
                tests=json.loads(row['public_test_cases'])+decode_tests(row['private_test_cases'])
                private.setdefault(split,{})[task_id]={'fn_name':json.loads(row['metadata']).get('func_name'), 'tests':tests}
                audit.append({'task_id':task_id,'split':split,'prompt_tokens':len(ids),'visible_prompt_sha256':ph,'raw_row_sha256':expected['source_row_sha256'],'test_count':len(tests)})
        print('verified',filename,flush=True)
    if len(audit)!=400:raise ValueError('Missing selected rows')
    # Earlier local experiments used a separate benchmark. Scan saved JSON text for exact visible problem bodies.
    prior_files=[p for base in ['configs','data','results'] for p in (ROOT/base).rglob('*.json')
        if 'coding_pilot_v1' not in str(p) and p.stat().st_size<10*1024**2]
    prior='\n'.join(p.read_text(errors='replace') for p in prior_files)
    for split, rows in visible.items():
        for r in rows:
            # Exact prompt-body occurrence only; IDs alone do not establish exposure.
            body=r['prompt'].split('\n\n',1)[1].rsplit('\n\n',1)[0]
            if body in prior:problems.append({'task_id':r['task_id'],'prior_exact_body_exposure':True})
    if problems:
        write(ROOT/'evidence/coding_pilot_v1/data_gate.json',{'passed':False,'issues':problems,'rows':audit});raise ValueError('Membership requires declared amendment before generation')
    for split,rs in visible.items():
        write(dest/'visible'/f'{split}.json',rs)
        write(dest/'private'/f'{split}.json',private[split])
    identity={'membership_sha256':sha(ROOT/'configs/coding_pilot_v1/draft_membership.json'),
        'preparation_source_sha256':sha(__file__),'visible_files':{s:sha(dest/'visible'/f'{s}.json') for s in visible},
        'private_files':{s:sha(dest/'private'/f'{s}.json') for s in private}}
    bind(dest/'identity.json',identity)
    write(ROOT/'evidence/coding_pilot_v1/data_gate.json',{'passed':True,'identity':identity,'rows':audit,
        'prior_exposure_check':'Exact visible-body search of prior configs/data/results JSON files smaller than 10 MiB; upstream training exposure remains unknown',
        'prior_files_checked':len(prior_files)})
    print('data gate passed; 400 selected problems; no inference',flush=True)
if __name__=='__main__':main()
