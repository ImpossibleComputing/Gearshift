from scripts.coding_memory_worker import offload_eligible


def test_only_measured_memory_failures_allow_the_offload_fallback():
    assert offload_eligible({'passed':False,'exception':'OutOfMemoryError'})
    assert offload_eligible({'passed':False,'error':'Full-context memory/headroom gate failed at backward'})
    assert not offload_eligible({'passed':False,'error':'Missing, zero or nonfinite mapper gradients'})
    assert not offload_eligible({'passed':False,'error':'A completed recorded sample changed'})
    assert not offload_eligible({'passed':True})


def test_separate_workers_preserve_their_probe_results(tmp_path,monkeypatch):
    import json,os,time
    from types import SimpleNamespace
    from scripts import coding_memory_worker as worker
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    destinations=[]
    def run(command,**kwargs):
        root=tmp_path/command[command.index('--output-root')+1]
        assert not root.exists()
        root.mkdir(parents=True)
        (root/'memory_gate.json').write_text(json.dumps({'passed':True,'training_ready':True}))
        destinations.append(root)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess,'run',run)
    for name in ('first','repair'):
        spec={'result_root':f'results/coding_pilot_v1/{name}/memory',
            'worker_status_path':f'evidence/{name}/worker_status.json','stage_identity':name,
            'worker_id':'memory','baseline_root':'baseline','allocation_path':'allocation',
            'deadline_epoch':time.time()+1000}
        path=tmp_path/(name+'.json');path.write_text(json.dumps(spec))
        monkeypatch.setenv('GEARSHIFT_WORKER_SPEC',str(path));worker.main()
        result=json.loads((tmp_path/spec['result_root']/'memory_gate.json').read_text())
        assert result['passed'] and result['training_ready']
        assert result['selected_probe_root'].startswith(spec['result_root']+'/probes/')
    assert len(set(destinations))==2
    assert all((p/'memory_gate.json').exists() for p in destinations)
