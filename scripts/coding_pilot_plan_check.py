#!/usr/bin/env python3
"""Metadata-only validator and deterministic seed helper; contains no execution launcher."""
import argparse, hashlib, json
from pathlib import Path

def seed_for(task_id,sample_index,stream):
    payload=f'coding_pilot_v1|{task_id}|{sample_index}|{stream}'
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],'big')%(2**63)

def validate(folder):
    folder=Path(folder);cfg=json.loads((folder/'pilot.json').read_text())
    assert cfg['status']=='draft_prepared_not_executed'
    assert cfg['authorization']['execution_allowed'] is False
    assert cfg['authorization']['approved_spending_cap_usd'] is None
    assert cfg['authorization']['previous_phase_authorization_applies'] is False
    assert cfg['budget']['approved'] is False
    assert cfg['arms']=={'A':'large_only','B':'small_only','C':'gearshift','D':'text_handoff'}
    assert cfg['protocol']['new_user_turn'] is False
    metadata=json.loads((folder/'reference/official_metadata.json').read_text())
    geometries={}
    for role in ['source','receiver']:
        model=cfg['models'][role];official=metadata['models'][model['id']]
        assert model['revision']==official['revision'] and model['frozen'] and model['dtype']=='bfloat16'
        path=folder/'reference'/model['id'].split('/')[-1]
        for name,expected in official['files'].items():
            data=(path/name).read_bytes()
            assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256']
        c=json.loads((path/'config.json').read_text())
        geometries[role]={'layers':c['num_hidden_layers'],'kv_heads':c['num_key_value_heads'],'head_dim':c['head_dim'],
            'cache_bytes_per_token':2*c['num_hidden_layers']*c['num_key_value_heads']*c['head_dim']*2}
    a,b=[metadata['models'][cfg['models'][role]['id']] for role in ['source','receiver']]
    assert a['files']['tokenizer.json']['sha256']==b['files']['tokenizer.json']['sha256']
    assert a['files']['tokenizer_config.json']['sha256']==b['files']['tokenizer_config.json']['sha256']
    data=(folder/'draft_membership.json').read_bytes();membership=json.loads(data)
    assert hashlib.sha256(data).hexdigest()==cfg['data']['membership_sha256']
    assert membership['dataset_revision']==cfg['data']['revision']
    seen=set()
    for split,n in [('training',128),('validation',32),('development',40),('confirmation',200)]:
        assert len(membership['selected'][split])==n
        for r in membership['selected'][split]:
            ident=(r['platform'],str(r['question_id']));assert ident not in seen;seen.add(ident)
            start,end,_=membership['windows'][split]
            assert start<=r['contest_date'][:10]<=end
    conf={(r['platform'],str(r['question_id'])) for r in membership['selected']['confirmation']}
    subset={(r['platform'],str(r['question_id'])) for r in cfg['mapper']['second_seed_confirmation_ids']}
    assert len(subset)==40 and subset<=conf
    assert cfg['decoding']['task_quality']['do_sample'] is True
    assert cfg['decoding']['samples_per_task_per_arm']==1
    assert cfg['protocol']['prompt_max_tokens']+cfg['protocol']['development_cap_gate']['one_allowed_reasoning_cap_increase']+cfg['protocol']['answer_max_tokens']<=40960
    return dict(status='passed_static_planning_checks_only',execution_allowed=False,geometry=geometries,
        combined_weight_gib=sum(m['weight_bytes_from_index'] for m in metadata['models'].values())/1024**3,
        combined_cache_gib_at_32768=sum(g['cache_bytes_per_token'] for g in geometries.values())*32768/1024**3,
        selected_tasks=len(seen),actual_gpu_fit='unmeasured',pilot_inference='not_run',sandbox='not_yet_ported_and_verified_for_linux',
        missing_execution_gates=['Owner approval and new spending cap','Verified natural-boundary sampler','Linux sandbox','Measured full-prefix gradient memory','Task-aware selection and execution watchdog'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config-dir',type=Path,default=Path(__file__).resolve().parents[1]/'configs/coding_pilot_v1');p.add_argument('--output',type=Path);a=p.parse_args()
    result=validate(a.config_dir)
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
