import importlib.util,sys
from pathlib import Path
import pytest

def load():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('recovery_controller_test',scripts/'coding_recover_session.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_remaining_allocation_excludes_offline_time():
    m=load();ledger={'stage':'preflight','stage_started_epoch':0,'resources':[
        {'kind':'pod','started_epoch':0,'absent_epoch':1800,'upper_rate_usd':5.52},
        {'kind':'volume','started_epoch':0,'upper_rate_usd':.04}]}
    assert m.remaining_seconds(ledger,6*3600)==3.5*3600

def test_no_allocation_at_dollar_or_gpu_limit():
    m=load()
    for ledger in [{'prior_upper_usd':25},{'resources':[{'kind':'pod','started_epoch':0,'absent_epoch':4*3600,'upper_rate_usd':1}]}]:
        with pytest.raises(ValueError):m.remaining_seconds(ledger,4*3600)

def test_dollar_limit_can_be_tighter_than_gpu_hours():
    m=load();assert m.remaining_seconds({'prior_upper_usd':24},1)==pytest.approx(3600/5.56)
