import copy
import json
import pytest
from gearshift.phase2_io import digest, immutable, checkpoint_status, validate_transaction
from gearshift.phase2_tasks import TaskSpec


def test_hidden_material_cannot_enter_generation():
    task=TaskSpec('id','code','test','parent','Implement f(x).','Return code.',100,100,'exec',
                  {'hidden_test':'SECRET_TEST_8172','gold':'SECRET_GOLD_8172'},{'correct':True})
    assert 'SECRET' not in task.generation_text()
    assert 'SECRET' not in json.dumps(task.visible())
    assert 'hidden' not in task.visible()


def test_immutable_write_drift_leaves_bytes(tmp_path):
    path=tmp_path/'manifest.json';immutable(path,{'a':1});before=path.read_bytes()
    with pytest.raises(ValueError):immutable(path,{'a':2})
    assert path.read_bytes()==before


def test_immutable_json_round_trip_keeps_tuple_arrays_but_rejects_boolean_number_drift(tmp_path):
    path=tmp_path/'reservation.json';immutable(path,{'lengths':[(128,'short')],'enabled':True})
    immutable(path,{'lengths':[(128,'short')],'enabled':True})
    immutable(path,{'lengths':[[128,'short']],'enabled':True})
    with pytest.raises(ValueError):immutable(path,{'lengths':[(128,'short')],'enabled':1})


def test_missing_weights_are_explicitly_unverified(tmp_path):
    result=checkpoint_status(tmp_path/'missing.pt',{'bytes':12,'sha256':'x'})
    assert result['status']=='unavailable_not_verified'


def test_manifest_keeps_distinct_handoffs_within_one_task_family():
    from gearshift.phase2_inference import protocol_manifest
    class Tokenizer:
        def encode(self,text,add_special_tokens=False):return list(text.encode())
    a=TaskSpec('short','writing','length','parent','brief','Write fifty words.',100,128,'writing',{}, {})
    b=TaskSpec('long','writing','length','parent','brief','Write five hundred words.',100,1024,'writing',{}, {})
    manifest=protocol_manifest(Tokenizer(),[a,b])
    assert set(manifest)=={'short','long'}
    assert manifest['short']['newturn/True']!=manifest['long']['newturn/True']
    assert b'Write fifty words.' in bytes(manifest['short']['newturn/True'])
    assert b'Write five hundred words.' in bytes(manifest['long']['newturn/True'])


def test_declared_reasoning_cap_override_preserves_prompt_and_hidden_separation():
    from gearshift.phase2_tasks import apply_budgets
    t=TaskSpec('id','code','test','id','Function specification','Return code.',1024,512,'code',{'test':'hidden'}, {})
    changed=apply_budgets({'source_reasoning_caps':{'code':2048}},[t])[0]
    assert changed.reasoning_budget==2048 and t.reasoning_budget==1024
    assert changed.answer_budget==512 and changed.generation_text()==t.generation_text()
    assert 'hidden' not in changed.generation_text()


def test_scoring_rejects_different_hidden_gold_and_allows_only_declared_budget_override():
    from scripts.phase2_score import validate_grading_task
    from dataclasses import replace
    t=TaskSpec('id','arithmetic','test','id','Question','Number.',1024,64,'numeric',{'gold':'12'}, {})
    expected=replace(t,reasoning_budget=2048)
    identity=dict(tasks=[expected.visible()],config={'source_reasoning_caps':{'arithmetic':2048}},
        task_grading_hashes={'id':digest(dict(hidden=t.hidden,grader=t.grader,metadata=t.metadata))})
    validate_grading_task(t,identity)
    with pytest.raises(ValueError):validate_grading_task(replace(t,hidden={'gold':'13'}),identity)
    with pytest.raises(ValueError):validate_grading_task(replace(t,prompt='Different question'),identity)


def transaction():
    tr={'prompt_ids':[1,2],'source_reasoning_ids':[3],'token_sha256':digest([1,2,3])}
    row={'condition':'M','shared_source_trajectory_sha256':tr['token_sha256'],
         'answer_tokens':2,'answer_token_ids':[4,5],'actual_backend_input_tokens':4,'prefill_tokens':0,'bridge_tokens':2}
    obj={'identity_sha256':'identity','task_id':'task','trajectory':tr,'rows':[row]}
    obj['payload_sha256']=digest(obj);return obj


def seal(obj):
    obj['payload_sha256']=digest({k:v for k,v in obj.items() if k!='payload_sha256'});return obj


def test_transaction_integrity_and_exact_matrix():
    obj=transaction();validate_transaction(obj,'identity','task',['M'])
    for mutation in ['tokens','duplicate','prefill','accounting','missing']:
        bad=copy.deepcopy(obj)
        if mutation=='tokens':bad['trajectory']['source_reasoning_ids']=[6]
        if mutation=='duplicate':bad['rows']*=2
        if mutation=='missing':bad['rows']=[]
        if mutation=='prefill':bad['rows'][0]['prefill_tokens']=1;bad['rows'][0]['actual_backend_input_tokens']+=1
        if mutation=='accounting':bad['rows'][0]['actual_backend_input_tokens']-=1
        with pytest.raises(ValueError):validate_transaction(seal(bad),'identity','task',['M'])


def test_payload_and_parent_identity_fail_closed():
    obj=transaction();obj['rows'][0]['answer_token_ids']=[8,9]
    with pytest.raises(ValueError):validate_transaction(obj,'identity','task',['M'])
    with pytest.raises(ValueError):validate_transaction(transaction(),'different','task',['M'])


def test_evidence_field_scoring_rejects_duplicate_and_missing_claims():
    from gearshift.phase2_grading import grade
    task=TaskSpec('e','evidence','test','e','packet','memo',100,100,'evidence_fields',
        dict(decision='Aster',monthly_total=123,difference=22,required_sources=['S1','S2'],totals={'Aster':123,'Birch':145}),{})
    good='Decision: Aster\nMonthly total: 123\nDifference: 22\n[S1] [S2]'
    assert grade(task,good)['all_fields_correct']
    assert not grade(task,good+'\nDecision: Birch')['all_fields_correct']
    assert grade(task,good+'\nDecision: Aster')['all_fields_correct']
    assert not grade(task,good+'\nDecision: Aster')['field_format_valid']
    assert grade(task,good.replace('Decision: Aster','Decision: S2'))['all_fields_correct']
    assert not grade(task,'')['all_fields_correct']
    assert grade(task,good)['semantic_quality_status'].startswith('pending')


def test_code_extraction_is_fixed_without_repair():
    from gearshift.phase2_sandbox import extract_code
    assert extract_code('prose\n```python\ndef f(): return 1\n```')=='def f(): return 1\n'
    assert extract_code('bad prose\ndef f(): return 1')=='bad prose\ndef f(): return 1'
    assert extract_code('```python\nbad\n```\n```python\ngood\n```')=='bad\n'


def test_training_windows_include_first_and_later_answer_predictions():
    from gearshift.phase2_training import sparse_case
    obj={'teachers':{'boundary':{'bridge_ids':[7,8,9],'answer_ids':list(range(20,60))}},
         'trajectory':{'prompt_ids':[1],'source_reasoning_ids':[2,3]},'task':{'family':'writing','task_id':'x'}}
    first=sparse_case(obj,'boundary',0)
    assert first['prefix_ids']==[7,8]
    assert first['input_ids']==[9]+list(range(20,27))
    assert first['target_ids']==list(range(20,28))
    late=sparse_case(obj,'boundary',32)
    assert late['input_ids']==list(range(51,59)) and late['target_ids']==list(range(52,60))
    capped=sparse_case(obj,'boundary',256)
    assert capped['anchor']==32 and capped['requested_anchor']==256


def test_cluster_bootstrap_does_not_treat_repeated_variants_as_independent():
    from gearshift.phase2_reporting import paired_cluster
    rows=[]
    for parent in range(2):
        for v in range(3):
            for c in ['M','C']:rows.append(dict(task_id=f'{parent}/{v}',cluster_id=str(parent),condition=c,score={'correct':c=='M'}))
    out=paired_cluster(rows,'M','C','correct')
    assert out['n_clusters']==2 and out['n_task_variants']==6
    assert out['difference']==1 and not out['equivalence_established']


@pytest.mark.parametrize('anchor',[0,32])
def test_sparse_distillation_real_tiny_decoder_alignment_and_frozen_gradients(anchor):
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from gearshift.core import ModelBackend,CacheExtractor
    from gearshift.mapping import CacheAdapter
    from gearshift.phase2_training import frozen_pair,sparse_case,prediction,make_trainable
    def tiny(seed):
        torch.manual_seed(seed)
        config=Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256)
        b=ModelBackend.__new__(ModelBackend);b.device='cpu';b.dtype=torch.float32;b.config=config
        b.model=Qwen3ForCausalLM(config).eval().requires_grad_(False);b.input_token_count=0
        return b
    source,target=tiny(2),tiny(3)
    obj={'teachers':{'boundary':{'bridge_ids':[7,8,9],'answer_ids':list(range(20,70))}},
         'trajectory':{'prompt_ids':[1,2],'source_reasoning_ids':[3,4,5]},'task':{'family':'writing','task_id':'tiny'}}
    case=sparse_case(obj,'boundary',anchor);sp,tp=frozen_pair(source,target,case['history'])
    with torch.no_grad():
        reference=prediction(target,tp,case)
        whole=target.forward(case['history']+case['prefix_ids']+case['input_ids'],all_logits=True).logits.float().log_softmax(-1)
    assert torch.allclose(reference,whole[:,-case['predictions']:],atol=2e-6,rtol=2e-6)
    maps={name:[dict(sources=[i],weight=torch.eye(16),bias=torch.zeros(16)) for i in range(2)] for name in ['functional_k','functional_v']}
    adapter=CacheAdapter(source,target,maps,'functional');params=make_trainable(adapter)
    mapped=adapter.transform(sp,'functional');student=prediction(target,mapped,case)
    loss=(reference.exp()*(reference-student)).sum(-1).mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert any(p.grad.abs().sum()>0 for p in params)
    assert all(p.grad is None for b in [source,target] for p in b.model.parameters())
    assert all(x.shape[-2]==len(case['history']) for pairs in [sp,tp,mapped] for layer in pairs for x in layer)
