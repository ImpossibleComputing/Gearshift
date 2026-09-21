import pytest
from gearshift.coding_resume import verify_prefix,verify_segment

def test_saved_prefix_is_binding_and_cannot_be_replaced():
    verify_prefix([1,2],[1,2,3])
    verify_prefix([1,2,3,4],[1,2,3],final=True)
    with pytest.raises(ValueError):verify_prefix([1,9],[1,2,3])
    with pytest.raises(ValueError):verify_prefix([1,2],[1,2,3],final=True)

def test_complete_sample_requires_exact_equality():
    old={'s':[1,2]};complete={'s':{'tokens':[1,2]}}
    verify_segment([1,2],'s',old,complete)
    with pytest.raises(ValueError):verify_segment([1,2,3],'s',old,complete)
