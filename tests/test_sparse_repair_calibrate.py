"""CPU-only validation of runner contracts; never claims actual CUDA gates."""
import importlib.util
import json
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('sparse_calibrate', Path(__file__).resolve().parents[1]/'scripts/sparse_repair_calibrate.py')
cal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cal)


def fixture_inputs(tmp_path, monkeypatch):
    mapper=tmp_path/'mapper.pt';mapper.write_bytes(b'mapper fixture not actual tensor')
    monkeypatch.setattr(cal,'MAPPER_SHA256',cal.sha(mapper))
    cases=[]
    for i in range(4):
        history={'task_id':f'dev/{i}','prompt_ids':[1,2], 'reasoning_ids':[4,5,151668],
                 'prefix_ids':[1,2,4,5], 'prefix_cache_length':4, 'bridge_ids':[151668],
                 'natural_boundary':True}
        p=tmp_path/f'history_{i}.json';p.write_text(json.dumps(history))
        cases.append({'task_id':history['task_id'],'history_path':p.name,'history_sha256':cal.sha(p),'historical_reasoning_positions':2})
    d={'calibration_tasks':cases,'screen_tasks':[{'task_id':'screen/0'}]}
    p=tmp_path/'declaration.json';p.write_text(json.dumps(d))
    return p,mapper,d


def test_only_selected_four_saved_histories_accepted(tmp_path,monkeypatch):
    p,mapper,d=fixture_inputs(tmp_path,monkeypatch)
    declaration,cases=cal.validate_inputs(tmp_path,p.name,cal.sha(p),mapper,['dev/2'])
    assert len(cases)==1 and cases[0][0]['task_id']=='dev/2'
    with pytest.raises(ValueError,match='Only frozen calibration'):
        cal.validate_inputs(tmp_path,p.name,cal.sha(p),mapper,['screen/0'])


def test_changed_history_or_declaration_rejected_before_models(tmp_path,monkeypatch):
    p,mapper,d=fixture_inputs(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='declaration hash'):
        cal.validate_inputs(tmp_path,p.name,'0'*64,mapper)
    (tmp_path/'history_0.json').write_text('{}')
    with pytest.raises(ValueError,match='history hash'):
        cal.validate_inputs(tmp_path,p.name,cal.sha(p),mapper)


def test_private_scope_and_path_escape_rejected(tmp_path,monkeypatch):
    p,mapper,d=fixture_inputs(tmp_path,monkeypatch)
    for path in ['../private.json','data/private/spec.json']:
        with pytest.raises(ValueError,match='public repository'):
            cal.safe_public_path(tmp_path,path)
    (tmp_path/'data/coding_pilot_v1/private').mkdir(parents=True)
    with pytest.raises(ValueError,match='Private-test directory'):
        cal.validate_inputs(tmp_path,p.name,cal.sha(p),mapper)


def test_no_rounding_excuse_for_same_shape():
    metric={'max_abs':.1,'kl':.00001,'top1_equal':True}
    assert not cal.pass_control(metric,False)
    assert cal.pass_control(metric,False,altered_shape=True)
    assert not cal.pass_control({**metric,'max_abs':.126},False,altered_shape=True)
    assert not cal.pass_control({**metric,'kl':.000101},False,altered_shape=True)
    assert not cal.pass_control({**metric,'top1_equal':False},False,altered_shape=True)
    assert cal.pass_control({'max_abs':0.,'kl':0.,'top1_equal':True},True)


def test_runtime_gate_refuses_non_cuda_machine():
    torch=pytest.importorskip('torch')
    if torch.cuda.is_available():
        pytest.skip('Test explicitly targets non-CUDA refusal')
    with pytest.raises(ValueError,match='H200'):
        cal.runtime_gate()


def test_probe_callback_identical_query_no_mutation_tiny_qwen():
    torch=pytest.importorskip('torch')
    transformers=pytest.importorskip('transformers')
    if transformers.__version__!='4.57.6':pytest.skip('Pinned hook API only')
    from gearshift.sparse_repair import HistoryLayout
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from types import SimpleNamespace
    config=Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=24,num_hidden_layers=1,
                      num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64)
    model=Qwen3ForCausalLM(config).eval();model.requires_grad_(False)
    backend=SimpleNamespace(model=model)
    native=((torch.randn(1,1,6,8),torch.randn(1,1,6,8)),)
    mapped=tuple(tuple(t+.1 for t in p) for p in native)
    live=tuple(torch.cat((t,torch.randn(1,1,1,8)),dim=-2) for t in native[0])
    before=[t.clone() for t in live]
    records=[];probe=cal.make_same_query_probe(backend,native,mapped,HistoryLayout(2,6),'dev/0',1,records)
    probe(0,0,torch.randn(1,2,1,8),live,None,8**-.5)
    assert len(records)==9 and all(r['identical_prefix_and_current_query'] for r in records)
    assert all(torch.equal(a,b) for a,b in zip(live,before))
    probe(0,1,torch.randn(1,2,1,8),live,None,8**-.5)
    assert len(records)==9
