import pytest

from scripts.coding_confirmation_regional_bootstrap import validate_visible_gpus


@pytest.mark.parametrize('names,count', [([], 1), (['NVIDIA H200'], 2),
                                      (['NVIDIA H200', 'NVIDIA H200'], 1),
                                      (['NVIDIA H100'], 1), (['H200'], 1),
                                      (['NVIDIA H200'], True), (None, 1)])
def test_missing_or_mismatched_gpus_fail_before_download(names, count):
    with pytest.raises(ValueError, match='Visible H200 count'):
        validate_visible_gpus({'gpu_names': names}, {'gpu_count': count})


def test_all_leased_devices_visible():
    validate_visible_gpus({'gpu_names': ['NVIDIA H200'] * 4}, {'gpu_count': 4})
