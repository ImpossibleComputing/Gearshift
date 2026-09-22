"""Early text handoff contracts and durable storage; no private archive dependencies."""
import hashlib,json,math,os
from pathlib import Path
CLOSING=151668
EOS={151645,151643}
FRACTIONS=(0,10,25,50,75,100)
CONDITIONS=('SMALL_ONLY','H10','H25','H50','H75','FULL_TEXT','LARGE_ONLY')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def seed(task,draw,stream):
 return int.from_bytes(hashlib.sha256(f'early_handoff_01|{task}|{draw}|{stream}'.encode()).digest()[:8],'big')%(2**63)
def write(path,value,immutable=False):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
 if path.exists() and immutable:
  if path.read_bytes()!=raw:raise ValueError('Refuse to replace completed evidence: '+str(path))
  return
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def read(path):return json.loads(Path(path).read_text())
def cut_index(length,fraction):
 if type(length)!=int or length<0 or type(fraction)!=int or fraction not in FRACTIONS:raise ValueError('Invalid fixed fraction/length')
 return length*fraction//100

def reasoning_body(history):
 ids=history['generated_ids']
 if history['natural_boundary']:
  if not ids or ids[-1]!=CLOSING or CLOSING in ids[:-1]:raise ValueError('Malformed natural boundary')
  return ids[:-1]
 if CLOSING in ids:raise ValueError('Undeclared closing token')
 return list(ids)

def handoff(prompt_ids,history,fraction):
 """Only original prompt and past reasoning tokens. Never takes a source answer."""
 if digest(prompt_ids)!=history['input_ids_sha256']:raise ValueError('Original prompt identity mismatch')
 body=reasoning_body(history);k=cut_index(len(body),fraction)
 visible=list(prompt_ids)+body[:k]
 if fraction==100:visible.append(CLOSING)
 return {'input_ids':visible,'source_index':k,'source_reasoning_length':len(body),
         'fraction_percent':fraction,'answer_only':fraction==100,
         'source_prefix_sha256':digest(body[:k]),'input_ids_sha256':digest(visible)}

def continuation_cap(source_index,maximum=24576):
 if type(source_index)!=int or not 0<=source_index<=maximum:raise ValueError('Context budget invalid')
 return maximum-source_index

def source_prefix_seconds(history,k,fraction):
 if fraction==0:return 0.0
 if fraction==100:return history['active_seconds']
 if not 0<=k<=len(reasoning_body(history)):raise ValueError('Source cut outside history')
 return history['token_elapsed_seconds'][k-1] if k else history['prefill_seconds']

def validate_ledger(ledger):
 if ledger['hard_usd']!=1500 or ledger['soft_usd']!=1000:raise ValueError('Owner budget differs')
 values=[ledger['new_compute_usd'],ledger['reserved_usd'],ledger['cleanup_reserve_usd']]
 if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in values):raise ValueError('Invalid accounting')
 if sum(values)>=ledger['hard_usd']:raise ValueError('Budget ceiling reached')
 return sum(values)
