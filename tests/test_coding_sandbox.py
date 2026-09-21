from gearshift.coding_sandbox import extract,equal_stdio,score

def test_first_python_or_unlabelled_fence_only():
    assert extract('```js\nwrong\n```\n```python\nprint(1)\n```\n```python\nprint(2)\n```')=='print(1)'
    assert extract('text')=='text'

def test_decimal_comparison_preserves_large_integer_difference():
    assert equal_stdio('1.0 2\n','1 2\n')
    assert not equal_stdio('50000000000000000','50000000000000001')
    assert not equal_stdio('1\n2','1 2')
