import json
from pathlib import Path
import numpy as np
from gearshift.followup import make_cases, packed_chat, valid_numeric, stop_reason
from types import SimpleNamespace
from gearshift.identity import artifact, array_descriptor


def test_prespecified_format_and_stop_metrics():
    assert valid_numeric('  -1,234.50\n')
    for text in ['\\boxed{42}', 'Answer: 42', '42\n42', '42.', '$42', '', '1,,2', '12,34']:
        assert not valid_numeric(text)
    b=SimpleNamespace(eos={9,10},tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _:7))
    assert stop_reason([1,9],b,2)=='eos'  # cap-length EOS is not a truncation
    assert stop_reason([1,2],b,2)=='cap'
    assert stop_reason([7],b,10,True)=='end_think'


def test_training_budget_domains_lengths_and_validation_separation(tmp_path):
    root=tmp_path; tok=SimpleNamespace(encode=lambda *a,**k:[98,99])
    (root/'trajectories').mkdir()
    for s,base in [('train',100),('validation',1000)]:
        rows=[dict(prompt_ids=[base+i]*4,source_reasoning_ids=[base+i]*24,source_completed=i%2==0) for i in range(8)]
        (root/'trajectories'/f'{s}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    np.save(root/'train_tokens.npy',np.arange(8*64).reshape(8,64))
    np.save(root/'validation_tokens.npy',np.arange(4*64).reshape(4,64)+10000)
    (root/'tokens.json').write_text(json.dumps({'splits':{s:{'array':array_descriptor(root/f'{s}_tokens.npy')} for s in ['train','validation']}}))
    (root/'plaintext_manifest.json').write_text(json.dumps({'root':str(root),'identity':{'tokens':artifact(root/'tokens.json')}}))
    cfg=dict(output=str(root),seed=42,functional=dict(prediction_tokens=8,steps=16,context_lengths=[8,16,32,48],validation_blocks_per_domain=4))
    cases,val=make_cases(cfg,tok)
    assert [c['length'] for c in cases['plaintext']]==[c['length'] for c in cases['chat']]
    for d in cases:
        assert len(cases[d])==16
        assert sum(len(c['ids'])-c['length'] for c in cases[d])==128
        assert {n:sum(c['length']==n for c in cases[d]) for n in [8,16,32,48]}=={8:4,16:4,32:4,48:4}
    assert len(val)==32
    assert all(c['ids'][0]>=10000 for c in val if c['domain']=='plaintext')
    assert all(c['ids'][0]>=1000 for c in val if c['domain']=='chat')
    assert all(c['ids'][0]<1000 for c in cases['chat'])
    assert any(c['packed_boundaries']>0 for c in cases['chat'])
