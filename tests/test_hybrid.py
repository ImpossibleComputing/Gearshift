import torch
import pytest
from gearshift.core import CacheInjector


def test_hybrid_preserves_absolute_suffix_positions_and_copies():
    full=torch.arange(80).reshape(1,2,10,4).float()
    prefix=torch.full((1,2,3,4),-1.)
    result=CacheInjector.splice_prefix([(prefix,prefix)],[(full,full+10)])
    assert torch.equal(result[0][0][...,:3,:],prefix)
    assert torch.equal(result[0][0][...,3:,:],full[...,3:,:])
    assert torch.equal(result[0][1][...,3:,:],(full+10)[...,3:,:])
    assert CacheInjector.create(result).get_seq_length()==10
    result[0][0].zero_()
    assert torch.equal(full,torch.arange(80).reshape(1,2,10,4).float())
    with pytest.raises(ValueError): CacheInjector.splice_prefix([(full,full)],[(prefix,prefix)])
