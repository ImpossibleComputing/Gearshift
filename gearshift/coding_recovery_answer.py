"""Sampled answers with lightweight token/RNG checkpoints; same original calls."""
import time
from pathlib import Path
import torch
from .coding_control import write,seed_for
from .coding_inference import generator,sample,sync

@torch.no_grad()
def answer_instrumented(backend,history,cache,task_id,stream,cap,folder,telemetry,guard,publish):
    rng=generator(task_id,stream,backend.device);initial=rng.get_state().cpu().tolist()
    assert cache.get_seq_length()==len(history['prefix_ids'])
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True);tokens=[];phase='before_bridge';ended=False;first=None
    def save(state='running'):
        write(folder/'resume.json',{'task_id':task_id,'state':state,'phase':phase,'answer_ids':tokens,'rng_state':rng.get_state().cpu().tolist(),
            'rng_initial':initial,'seed':seed_for(task_id,0,stream),'stream':stream,'cap':cap,'cache_sequence_length':cache.get_seq_length(),
            'historical_prefix_length':len(history['prefix_ids']),'cache_checkpointed':False})
    try:
        save();guard();sync();start=time.monotonic()
        out=backend.forward(history['bridge_ids'],cache);cache=out.past_key_values;logits=out.logits
        sync();bridge=time.monotonic()-start
        for i in range(cap):
            phase='sampling';token=sample(logits,rng);tokens.append(token)
            if first is None:sync();first=time.monotonic()-start
            if token in backend.eos:ended=True;break
            phase='forward_pending';out=backend.forward([token],cache);cache=out.past_key_values;logits=out.logits;phase='forward_complete'
            if i%64==0:
                save();telemetry.sample(stage='answer_generation',task_id=task_id,sequence_length=cache.get_seq_length());publish(generated_answer_tokens=len(tokens));guard()
        sync();elapsed=time.monotonic()-start;phase='answer_complete';save('complete')
        return {'task_id':task_id,'answer_ids':tokens,'answer_text':backend.tokenizer.decode(tokens,skip_special_tokens=True),
            'answer_text_with_special_tokens':backend.tokenizer.decode(tokens,skip_special_tokens=False),'answer_seed':seed_for(task_id,0,stream),
            'rng_initial':initial,'rng_final':rng.get_state().cpu().tolist(),'answer_seconds':elapsed,'bridge_seconds':bridge,
            'first_answer_token_seconds':first,'answer_ended_eos':ended,'answer_capped':not ended,'bridge_token_count':len(history['bridge_ids']),
            'instrumentation':'Elapsed generation includes periodic durable checkpoints and telemetry; final checkpoint excluded.'}
    except BaseException as exc:
        try:save('aborted')
        finally:telemetry.failure(exc)
        raise
