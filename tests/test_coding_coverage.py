import copy
import pytest
from gearshift.coding_coverage import paired_schedules, grouped, validation_panels, choose_endpoint, WindowCycle, seed_manifest


def histories():
    return [{'task_id': f'task/{n}', 'teacher_answer': {'answer_ids': [1]*(n-1)+[151645]}}
            for n in [3,9,17,31,32,33,130,513,514,700,1200]]


def test_matched_original_schedule_and_complete_eligibility():
    from gearshift.coding_training import make_schedule
    hs=histories(); schedules=paired_schedules(hs,1024)
    old=make_schedule(hs,20260915,1120)[96:]
    for a,b,r in zip(old,schedules['FIXED'],schedules['ROTATING']):
        assert a['segments']==b['segments']
        assert [len(p) for p in grouped(b).values()]==[len(p) for p in grouped(r).values()]
        assert b['predictions']==r['predictions']==32
    seen={h['task_id']:set() for h in hs}
    for row in schedules['ROTATING']:
        for tid,positions in grouped(row).items():seen[tid].update(positions)
    for h in hs:assert seen[h['task_id']]==set(range(len(h['teacher_answer']['answer_ids'])))
    assert paired_schedules(hs,10)=={a:s[:10] for a,s in schedules.items()}


def test_window_cycle_exhaustion_and_no_padding():
    q=WindowCycle(8,33,123);draws=[]
    for _ in range(20):
        positions,rows=q.take(8)
        assert len(set(positions))==8 and min(positions)>=8 and max(positions)<33
        draws.extend(rows)
    for cycle in range(max(x['cycle'] for x in draws)):
        assert sorted(x['position'] for x in draws if x['cycle']==cycle)==list(range(8,33))


def test_validation_panels_frozen_valid_include_eos():
    for tid,row in validation_panels(histories()).items():
        n=int(tid.split('/')[1])
        assert row['broad']==sorted(set(row['broad']))
        assert row['broad'][0]==0 and row['broad'][-1]==n-1
        assert max(row['legacy'])<n


def test_endpoint_is_resource_only():
    p={'representative_updates':[{'wall_seconds':120}]*12,'validation_seconds':300,'memory_and_gradient_checks_passed':True}
    r=choose_endpoint(p)
    assert r['endpoint']==256 and r['quality_scores_consulted'] is False
    p['new_quality_scores']={'FIXED':1,'ROTATING':0}
    assert choose_endpoint(p)['endpoint']==256
    p['representative_updates']=[{'wall_seconds':1000}]
    with pytest.raises(ValueError):choose_endpoint(p)


def test_seeds_are_task_matched_not_pooled():
    a=seed_manifest(['x','y'],3);b=seed_manifest(['x'],5)
    assert a['x']==b['x'][:3]
    assert len({x['answer_seed'] for x in b['x']})==5


@pytest.mark.parametrize('order',[('FIXED','ROTATING'),('ROTATING','FIXED')])
def test_shared_cache_optimization_matches_independent_updates(order):
    import torch
    from types import SimpleNamespace
    from gearshift.coding_training import TrainingRuntime
    from gearshift.coding_coverage_runtime import paired_update
    class ToyRuntime(TrainingRuntime):
        def __init__(self):
            frozen=torch.nn.Linear(1,1).requires_grad_(False)
            mapper=torch.nn.Linear(1,1).double()
            with torch.no_grad():mapper.weight.fill_(.3);mapper.bias.fill_(.2)
            super().__init__(SimpleNamespace(model=frozen),SimpleNamespace(model=frozen),mapper,None)
            self.telemetry=SimpleNamespace(sample=lambda **kw:None)
        def pair(self,obj,device=None):
            p=((torch.ones(1,1,2,1),torch.ones(1,1,2,1)),)
            return p,p
        def kl(self,obj,positions,objective,sp,tp,require_grad=True,already_extended=False):
            x=torch.tensor(positions,dtype=torch.float64).reshape(-1,1)/30+1
            mapped=self.mapper(x);checks=[]
            def hook(g):
                checks.append({'finite':bool(torch.isfinite(g).all()),'absolute_sum':float(g.abs().sum())})
                return g
            mapped.register_hook(hook);self.cache_gradient_checks.append(checks)
            return (mapped-2).square().flatten()
    items={a:{'predictions':32,'segments':[{'task_id':t,'positions':list(range(start,start+16))} for t in ['a','b']]} for a,start in [('FIXED',0),('ROTATING',16)]}
    lookup={t:{'task_id':t,'source_history':{'prefix_ids':[1,2]},'teacher_answer':{'answer_ids':[1]*64}} for t in ['a','b']}
    actual={a:ToyRuntime() for a in order};expected={a:ToyRuntime() for a in order}
    opts={a:torch.optim.AdamW(r.mapper.parameters(),lr=1e-5,weight_decay=0,foreach=False) for a,r in actual.items()}
    expected_opts={a:torch.optim.AdamW(r.mapper.parameters(),lr=1e-5,weight_decay=0,foreach=False) for a,r in expected.items()}
    for _ in range(3):
        result=paired_update(items,lookup,actual,opts,order)
        for a in order:
            old=expected[a].train_update(items[a],lookup,'natural_handoff_boundary',expected_opts[a])
            assert result['arms'][a]['kl']==old['kl']
            for k,v in actual[a].mapper.state_dict().items():assert torch.equal(v,expected[a].mapper.state_dict()[k])


def test_clustered_analysis_uses_tasks_and_rejects_missing_draws():
    from scripts.coding_coverage_report import clustered
    tids=['one','two','three'];rows=[]
    for tid in tids:
        for condition in ['A','B']:
            for seed in range(3):
                rows.append({'task_id':tid,'condition':condition,'seed_index':seed,'passed':seed<2 if condition=='A' else seed==0,
                    'category':'passed','missing_requested_entrypoint':False,'syntax_valid':True,'EOS':True,'capped':False,
                    **{k:1. for k in ['answer_tokens','repetition_4gram_fraction','answer_seconds','native_prefill_seconds','mapping_seconds','splice_seconds','historical_receiver_prefill_tokens','source_inclusive_estimate_seconds']}})
    result,tasks,indices=clustered(rows,tids,['A','B'],3,[('A','B')])
    assert result['task_clusters']==3 and len(indices)==10000 and len(indices[0])==3
    assert result['contrasts']['A-B']['difference']==pytest.approx(1/3)
    assert result['contrasts']['A-B']['ci95']==pytest.approx([1/3,1/3])
    assert result['conditions']['A']['pass_rate']==pytest.approx(2/3)
    with pytest.raises(ValueError):clustered(rows[:-1],tids,['A','B'],3,[('A','B')])
    with pytest.raises(ValueError):clustered(rows+[rows[0]],tids,['A','B'],3,[('A','B')])


def test_report_rejects_wrong_checkpoint_seed_or_teacher_prefix(tmp_path,monkeypatch):
    import scripts.coding_coverage_report as report
    from gearshift.coding_control import write,sha
    monkeypatch.setattr(report,'ROOT',tmp_path)
    folder=tmp_path/'validation'/'task';path=folder/'START_M'/'seed_0'/'answer.json'
    write(folder/'source_history.json',{'task_id':'task','prefix_ids':[1,2],'bridge_ids':[151668]})
    seeds={'task':[{'seed_index':0,'stream':'fixed','answer_seed':123}]}
    visible={'task':{'prompt':'class Solution:\n    def solve(self, x):\n        pass'}}
    row={'task_id':'task','condition':'START_M','form':'M','seed_index':0,'stream':'fixed','answer_seed':123,
         'checkpoint_sha256':'expected','source_history_sha256':sha(folder/'source_history.json'),'teacher_answer_prefix_supplied':False,
         'code':'class WrongClass:\n    def solve(self, x):\n        return x','score':{'passed':False,'category':'runtime_error'},
         'answer_ids':[1,2,151645],'answer_ended_eos':True,'answer_capped':False}
    write(path,row)
    result=report.program_row(path,visible,seeds,{'START_M':'expected'})
    assert result['missing_requested_entrypoint'] and not result['passed']
    for field,value in [('checkpoint_sha256','wrong'),('answer_seed',124),('teacher_answer_prefix_supplied',True)]:
        write(path,{**row,field:value})
        with pytest.raises(ValueError):report.program_row(path,visible,seeds,{'START_M':'expected'})
