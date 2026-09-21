"""GPU admission must distinguish preexisting owners from its own new context."""
from types import SimpleNamespace
import sys
import pytest
from scripts import coding_confirmation_secondary_sharded_generate_v2 as g


def setup(monkeypatch, problem=None):
    initialized=[problem=='initialized'];calls=[]
    def memory():
        calls.append('memory');initialized[0]=True
        return (90 if problem=='memory' else 99),100
    cuda=SimpleNamespace(is_initialized=lambda:initialized[0],device_count=lambda:2 if problem=='count' else 1,
        get_device_name=lambda _: 'H100' if problem=='gpu' else 'NVIDIA H200',mem_get_info=memory)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','bad-uuid' if problem=='visible' else '0')
    def command(args,**kw):
        if '--query-gpu=uuid' in args:return 'GPU-assigned\nGPU-other\n'
        calls.append('owners_after' if initialized[0] else 'owners_before')
        if problem=='malformed':return 'not-a-valid-row\n'
        if not initialized[0]:return '44, GPU-assigned\n' if problem=='preexisting' else '55, GPU-other\n'
        if problem=='no_owner':return ''
        return '999999, GPU-assigned\n'+('22, GPU-assigned\n' if problem=='racing_owner' else '')
    monkeypatch.setattr(g.subprocess,'check_output',command)
    return calls


def test_host_pid_can_differ_only_after_verified_empty_to_single_context_transition(monkeypatch):
    calls=setup(monkeypatch);result=g.idle_gpu()
    assert calls==['owners_before','memory','owners_after']
    assert result['compute_owner_host_pid']==999999 and result['single_context_transition_verified']
    assert not result['other_compute_owners'] and result['preexisting_compute_owners']==[]


@pytest.mark.parametrize('problem',['initialized','preexisting','memory','count','gpu','visible','malformed','no_owner','racing_owner'])
def test_existing_ambiguous_or_racing_ownership_remains_fatal(monkeypatch,problem):
    calls=setup(monkeypatch,problem)
    with pytest.raises(ValueError):g.idle_gpu()
    if problem=='preexisting':assert calls==['owners_before']
