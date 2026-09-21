import importlib.util,sys
from pathlib import Path
import pytest


def controller():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('resume_capacity',scripts/'coding_resume_session.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_only_confirmed_unallocated_capacity_failure_can_retry():
    m=controller();prior={'attempt':9,'attempt_started_epoch':100};error={'error':'There are no longer any instances available'}
    ledger={'resources':[{'kind':'pod','started_epoch':10,'absent_epoch':90}]}
    assert m.capacity_retry(prior,error,ledger)==10
    with pytest.raises(ValueError):m.capacity_retry(prior,{'error':'Timeout: unknown allocation state'},ledger)
    ledger['resources'].append({'kind':'pod','started_epoch':100,'absent_epoch':110})
    with pytest.raises(ValueError):m.capacity_retry(prior,error,ledger)
    with pytest.raises(ValueError):m.capacity_retry({'attempt':12,'attempt_started_epoch':120},error,{'resources':[]})
