#!/usr/bin/env python3
"""Local-first, token-exact early text handoff. Raw evidence stays in ignored storage."""
import argparse,copy,fcntl,gc,json,os,platform,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gearshift.early_handoff import *

def stop_requested(storage):
 if (storage/'control/STOP').exists():raise InterruptedError('Explicit stop marker')

def setup_runtime():
 import torch
 torch.set_num_threads(4)
 torch.use_deterministic_algorithms(True)
 if not torch.backends.mps.is_available():raise RuntimeError('Local MPS is not available; no implicit CPU/remote fallback')
 return torch

class Backend:
 def __init__(self,storage,role):
  import torch
  from transformers import AutoTokenizer,AutoModelForCausalLM
  self.torch=torch;self.device='mps';self.role=role;self.storage=storage
  self.tokenizer=AutoTokenizer.from_pretrained(storage/'models'/role,local_files_only=True)
  t=time.monotonic();self.model=AutoModelForCausalLM.from_pretrained(storage/'models'/role,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval()
  self.model.requires_grad_(False);self.model.to('mps');self.sync();self.load_seconds=time.monotonic()-t
  write(storage/'control'/f'{role}_loaded.json',{'epoch':time.time(),'model_load_seconds':self.load_seconds,'torch':torch.__version__,'platform':platform.platform(),'memory':self.memory()})
 def sync(self):self.torch.mps.synchronize()
 def memory(self):
  return {'tensor_allocated_bytes':self.torch.mps.current_allocated_memory(),'driver_allocated_bytes':self.torch.mps.driver_allocated_memory(),'recommended_max_bytes':self.torch.mps.recommended_max_memory()}
 def forward(self,ids,cache=None):
  t=self.torch;past=0 if cache is None else cache.get_seq_length();x=t.tensor([ids],dtype=t.long,device='mps');pos=t.arange(past,past+len(ids),device='mps')
  return self.model(input_ids=x,past_key_values=cache,use_cache=True,attention_mask=t.ones((1,past+len(ids)),dtype=t.long,device='mps'),position_ids=pos.unsqueeze(0),cache_position=pos,logits_to_keep=1)
 def prefill(self,ids):
  if not ids:raise ValueError('Empty native text prefix')
  cache=None
  for i in range(0,len(ids),512):
   stop_requested(self.storage);o=self.forward(ids[i:i+512],cache);cache=o.past_key_values
  return o
 def sample(self,logits,rng):
  t=self.torch;values,indices=t.topk(logits[0,-1].float()/.6,20);p=t.softmax(values,dim=-1);p=p.masked_fill(p.cumsum(-1)-p>=.95,0);p=p/p.sum()
  return int(indices[t.multinomial(p,1,generator=rng)].item())
 def rng(self,s):return self.torch.Generator(device='mps').manual_seed(s)
 def release(self):
  del self.model;gc.collect();self.torch.mps.empty_cache()


def replay(backend,input_ids,generated,stop_kind,initial=None):
 """Same chunked initial prefill, then one-token steps: no altered-shape recovery."""
 o=initial if initial is not None else backend.prefill(input_ids)
 for token in generated:
  # Terminal emissions were deliberately not inserted into the cache.
  if stop_kind=='reason' and token==CLOSING:break
  if token in EOS and stop_kind=='answer':break
  o=backend.forward([token],o.past_key_values)
  if token in EOS:break
 return o


def generate(backend,input_ids,cap,s,kind,path,storage,initial=None):
 """Separate fixed RNG per stage/draw. Completed draws never sampled again."""
 final=path/'complete.json';checkpoint=path/'checkpoint.json';contract={'input_ids_sha256':digest(input_ids),'cap':cap,'seed':s,'kind':kind,'model_role':backend.role}
 if final.exists():
  r=read(final)
  if r['contract']!=contract:raise ValueError('Completed identity differs')
  return r,None
 rng=backend.rng(s);tokens=[];elapsed=[];prior_active=0.;recover_seconds=0.;resumes=[]
 backend.sync();started=time.monotonic()
 if checkpoint.exists():
  cp=read(checkpoint)
  if cp['contract']!=contract:raise ValueError('Checkpoint identity differs')
  tokens=cp['generated_ids'];elapsed=cp['token_elapsed_seconds'];prior_active=cp['active_seconds'];resumes=cp.get('recovery_attempts',[])
  o=replay(backend,input_ids,tokens,kind,initial);backend.sync();recover_seconds=time.monotonic()-started
  rng.set_state(backend.torch.tensor(cp['rng_state'],dtype=backend.torch.uint8,device='cpu'))
  prefill_seconds=cp['prefill_seconds'];resumes.append({'epoch':time.time(),'cache_replay_seconds':recover_seconds,'committed_tokens':len(tokens)})
 else:
  o=initial if initial is not None else backend.prefill(input_ids);backend.sync();prefill_seconds=time.monotonic()-started
 started=time.monotonic();base=prior_active if checkpoint.exists() else prefill_seconds
 natural=False;eos=False
 for i in range(len(tokens),cap):
  if i%16==0:stop_requested(storage)
  token=backend.sample(o.logits,rng);tokens.append(token);backend.sync();elapsed.append(base+time.monotonic()-started)
  natural=kind=='reason' and token==CLOSING;eos=token in EOS
  if natural or (eos and kind=='answer'):break
  o=backend.forward([token],o.past_key_values)
  if eos:break
  if len(tokens)%128==0:
   backend.sync();active=base+time.monotonic()-started
   write(checkpoint,{'contract':contract,'generated_ids':tokens,'token_elapsed_seconds':elapsed,'active_seconds':active,'prefill_seconds':prefill_seconds,'rng_state':rng.get_state().cpu().tolist(),'recovery_attempts':resumes})
   write(storage/'control/progress.json',{'epoch':time.time(),'stage':kind,'record':str(path.relative_to(storage)),'tokens':len(tokens),'active_seconds':active,'memory':backend.memory()})
 backend.sync();active=base+time.monotonic()-started
 # Terminal stops do not need another forward; preserve every emitted token.
 record={'contract':contract,'input_ids_sha256':digest(input_ids),'generated_ids':tokens,'generated_ids_sha256':digest(tokens),'token_elapsed_seconds':elapsed,'prefill_seconds':prefill_seconds,'active_seconds':active,'generation_seconds':active-prefill_seconds,'natural_boundary':natural,'ended_eos':eos,'capped':not natural and not eos,'seed':s,'rng_state':rng.get_state().cpu().tolist(),'recovery_attempts':resumes,'recovery_seconds':sum(x['cache_replay_seconds'] for x in resumes),'gpu_kernel_busy_seconds':None,'gpu_kernel_busy_measurement':'Unavailable; synchronized MPS stage wall time is not kernel-busy time.','memory':backend.memory(),'completed_epoch':time.time()}
 write(final,record,immutable=True)
 return record,o


def answer_draws(backend,input_ids,history,base_out,task,condition,storage,stream,metadata):
 body=reasoning_body(history);answer_prefix=input_ids+body+[CLOSING]
 # Base state before closing token: the unaltered historical reasoning cache.
 if base_out is None:
  t=time.monotonic();base_out=replay(backend,input_ids,history['generated_ids'],'reason');backend.sync();setup=time.monotonic()-t
 else:setup=0.
 for draw in range(3):
  folder=storage/'raw'/task.replace('/','__')/condition/f'answer_{draw}'
  if (folder/'answer.json').exists():
   done=read(folder/'complete.json');expected={'input_ids_sha256':digest(answer_prefix),'cap':4096,'seed':seed(task,draw,stream),'kind':'answer','model_role':backend.role}
   if done['contract']!=expected:raise ValueError('Completed answer identity differs')
   continue
  initial=None;cache=None
  setup_path=folder/'setup.json'
  if not (folder/'complete.json').exists():
   backend.sync();t=time.monotonic();cache=copy.deepcopy(base_out.past_key_values);backend.sync();clone=time.monotonic()-t
   if cache.get_seq_length()!=len(answer_prefix)-1:raise ValueError('Pre-bridge cache/prefix length mismatch')
   t=time.monotonic();initial=backend.forward([CLOSING],cache);backend.sync();bridge=time.monotonic()-t
   if not setup_path.exists():write(setup_path,{'clone_seconds':clone,'bridge_seconds':bridge,'cache_reconstruction_setup_seconds':setup},immutable=True)
  if not setup_path.exists():raise ValueError('Completed answer lacks pre-generation setup receipt')
  record,answer_out=generate(backend,answer_prefix,4096,seed(task,draw,stream),'answer',folder,storage,initial)
  del answer_out
  # Text is raw research output, never a public result field. Complete-but-unpublished
  # draws can be materialized after a crash without drawing any new tokens.
  write(folder/'answer.json',{'task_id':task,'condition':condition,'draw':draw,'answer_text':backend.tokenizer.decode(record['generated_ids'],skip_special_tokens=True),'history_sha256':sha(storage/'raw'/task.replace('/','__')/condition/'reasoning/complete.json') if (storage/'raw'/task.replace('/','__')/condition/'reasoning/complete.json').exists() else metadata['source_history_sha256'],'answer_record_sha256':sha(folder/'complete.json'),**read(setup_path),**metadata},immutable=True)
  del initial,cache;gc.collect();backend.torch.mps.empty_cache()


def source_phase(backend,storage,d,visible):
 for task in d['population']['task_ids']:
  prompt=visible[task]['prompt_ids'];folder=storage/'raw'/task.replace('/','__')/'LARGE_ONLY'
  if all((folder/f'answer_{s}/answer.json').exists() for s in range(3)):continue
  h,o=generate(backend,prompt,24576,seed(task,0,'source_reasoning'),'reason',folder/'reasoning',storage)
  answer_draws(backend,prompt,h,o,task,'LARGE_ONLY',storage,'source_answer',{'fraction_percent':None,'source_index':len(reasoning_body(h)),'source_history_sha256':sha(folder/'reasoning/complete.json'),'source_prefix_seconds':h['active_seconds'],'transfer_seconds':0})
  del o;gc.collect();backend.torch.mps.empty_cache()


def receiver_phase(backend,storage,d,visible):
 for task in d['population']['task_ids']:
  prompt=visible[task]['prompt_ids'];source_file=storage/'raw'/task.replace('/','__')/'LARGE_ONLY/reasoning/complete.json';source=read(source_file)
  for fraction,condition in zip(FRACTIONS,CONDITIONS[:-1]):
   folder=storage/'raw'/task.replace('/','__')/condition
   if all((folder/f'answer_{s}/answer.json').exists() for s in range(3)):continue
   t=time.monotonic();view=handoff(prompt,source,fraction);transfer=time.monotonic()-t
   # Full native reference prefills before the bridge, then feeds closing once.
   if fraction==100:
    inp=view['input_ids'][:-1]
    if (folder/'reasoning/complete.json').exists():
     h=read(folder/'reasoning/complete.json');o=None
    else:
     backend.sync();t=time.monotonic();o=backend.prefill(inp);backend.sync();prefill=time.monotonic()-t
     h={'generated_ids':[CLOSING],'natural_boundary':True,'input_ids_sha256':digest(inp),'active_seconds':prefill,'prefill_seconds':prefill,'token_elapsed_seconds':[prefill],'capped':False,'ended_eos':False,'generation_seconds':0.,'recovery_seconds':0.}
     write(folder/'reasoning/complete.json',h,immutable=True)
   else:
    inp=view['input_ids'];h,o=generate(backend,inp,continuation_cap(view['source_index']),seed(task,0,'receiver_reasoning'),'reason',folder/'reasoning',storage)
   metadata={k:v for k,v in view.items() if k!='input_ids'}
   metadata.update(source_history_sha256=sha(source_file),source_prefix_seconds=source_prefix_seconds(source,view['source_index'],fraction),transfer_seconds=transfer)
   answer_draws(backend,inp,h,o,task,condition,storage,'receiver_answer',metadata)
   del o;gc.collect();backend.torch.mps.empty_cache()


def preflight(backend,storage):
 """Synthetic inputs only; exact-shape replay and RNG continuation must agree."""
 t=backend.torch;tok=backend.tokenizer
 ids=tok.apply_chat_template([{'role':'user','content':'Compute 17 plus 26, explaining the arithmetic briefly.'}],tokenize=True,add_generation_prompt=True,enable_thinking=True)
 backend.sync();started=time.monotonic();o=backend.prefill(ids);backend.sync();prefill=time.monotonic()-started
 rng=backend.rng(20260922);generated=[];beg=time.monotonic()
 for _ in range(64):
  token=backend.sample(o.logits,rng)
  if token in EOS or token==CLOSING:token=tok.encode(' arithmetic',add_special_tokens=False)[0]
  generated.append(token);o=backend.forward([token],o.past_key_values)
 backend.sync();decode=time.monotonic()-beg;saved=rng.get_state().cpu();logits=o.logits.detach().clone()
 native_o=o
 rebuilt=replay(backend,ids,generated,'reason');backend.sync();exact=t.equal(logits,rebuilt.logits)
 rr=backend.rng(0);rr.set_state(saved);continuations=[]
 for _ in range(16):
  a=backend.sample(native_o.logits,rng);b=backend.sample(rebuilt.logits,rr);continuations.append(a==b)
  native_o=backend.forward([a],native_o.past_key_values);rebuilt=backend.forward([b],rebuilt.past_key_values)
 result={'role':backend.role,'exact_shape_replay_logits_bitwise':exact,'rng_replayed_16_tokens_match':all(continuations),'synthetic_decode_tokens':64,'synthetic_decode_seconds':decode,'synthetic_tokens_per_second':64/decode,'prompt_prefill_seconds':prefill,'memory':backend.memory(),'torch':t.__version__,'model_execution_performed':True,'benchmark_tasks_read':False}
 del o,rebuilt,native_o,logits;gc.collect();t.mps.empty_cache()
 # A long native text prefix is required before accepting the local execution path.
 base=tok.encode('This is a synthetic memory-capacity probe, not a benchmark problem. ',add_special_tokens=False)
 long_ids=(base*(36865//len(base)+1))[:36865];backend.sync();start=time.monotonic();out=backend.prefill(long_ids);backend.sync()
 result['long_prefix_tokens']=36865;result['long_prefill_seconds']=time.monotonic()-start;result['long_memory']=backend.memory();result['long_cache_length']=out.past_key_values.get_seq_length()
 result['passed']=exact and all(continuations) and result['long_cache_length']==36865
 write(storage/'control'/f'{backend.role}_mps_preflight.json',result,immutable=True)
 if not result['passed']:raise RuntimeError('Local numerical/shape preflight failed')
 print(json.dumps(result),flush=True)


def seal(storage,d):
 if len(d['population']['task_ids'])*len(CONDITIONS)*3!=840:raise ValueError('Frozen population must contain all 840 answers')
 files=[]
 for task in d['population']['task_ids']:
  for condition in CONDITIONS:
   for draw in range(3):
    for name in ['complete.json','answer.json']:
     p=storage/'raw'/task.replace('/','__')/condition/f'answer_{draw}'/name
     files.append({'path':str(p.relative_to(storage)),'sha256':sha(p),'bytes':p.stat().st_size})
 payload={'expected_answers':840,'all_generation_complete':True,'declaration_sha256':sha(ROOT/'configs/early_handoff_01/declaration.json'),'files':files}
 path=storage/'control/generation_seal.json'
 if path.exists():
  existing=read(path)
  if {k:v for k,v in existing.items() if k!='epoch'}!=payload:raise ValueError('Existing generation seal differs')
  return
 write(path,{**payload,'epoch':time.time()},immutable=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','source','receiver','all']);p.add_argument('--role',choices=['source','receiver']);p.add_argument('--storage',type=Path,default=ROOT/'data/early_handoff_01');a=p.parse_args();a.storage=a.storage.resolve()
 if a.stage=='preflight' and not a.role:p.error('preflight requires --role')
 d=read(ROOT/'configs/early_handoff_01/declaration.json');validate_ledger(read(a.storage/'control/budget_ledger.json'))
 lock=(a.storage/'control/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def interrupted(signum,frame):raise InterruptedError('Controlled worker interruption')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 torch=setup_runtime();started=time.time()
 try:
  with torch.inference_mode():
   if a.stage=='preflight':
    b=Backend(a.storage,a.role);preflight(b,a.storage);b.release();return
   if not all(read(a.storage/'control'/f'{role}_mps_preflight.json')['passed'] for role in ['source','receiver']):raise RuntimeError('Local feasibility unverified')
   prepared=read(a.storage/'control/prepared_inputs.json')
   if sha(a.storage/'inputs/visible.json')!=prepared['visible_sha256'] or sha(ROOT/'configs/early_handoff_01/declaration.json')!=prepared['declaration_sha256']:raise ValueError('Frozen inputs changed')
   visible=read(a.storage/'inputs/visible.json')
   for role,fn in [('source',source_phase),('receiver',receiver_phase)]:
    if a.stage not in ('all',role):continue
    b=Backend(a.storage,role);fn(b,a.storage,d,visible);b.release()
   if a.stage in ('all','receiver'):seal(a.storage,d)
  write(a.storage/'control/worker_complete.json',{'stage':a.stage,'started_epoch':started,'completed_epoch':time.time(),'runpod_spend_usd':0})
 except BaseException as exc:
  write(a.storage/'control'/f'incident_{time.time_ns()}.json',{'stage':a.stage,'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'epoch':time.time(),'completed_outputs_preserved':True});raise
if __name__=='__main__':main()
