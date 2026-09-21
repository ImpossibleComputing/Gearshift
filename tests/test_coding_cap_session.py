import importlib.util,sys
from pathlib import Path
import pytest


def controller():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('cap_controller',scripts/'coding_cap_session.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_adoption_cannot_expand_or_change_paid_resources():
    m=controller();pod={'id':m.POD,'name':m.PREFIX+'preflight-10','gpuCount':1,'costPerHr':4.59}
    volume={'id':m.VOLUME,'name':m.PREFIX+'storage-10'}
    m.verify_adoption([pod,{'id':'unrelated','name':'other-project'}],[volume])
    for pods,vols in [([], [volume]),([pod,pod],[volume]),([pod],[{**volume,'id':'other'}]),
                      ([{**pod,'gpuCount':2}],[volume]),([{**pod,'costPerHr':5.51}],[volume])]:
        with pytest.raises(ValueError):m.verify_adoption(pods,vols)


def test_transition_cannot_start_from_a_running_or_failed_original_worker(tmp_path):
    m=controller()
    for state in ['running','stopped','bootstrap_failed']:
        with pytest.raises(ValueError,match='not finished'):m.transition_ready({'state':state},tmp_path,tmp_path)
