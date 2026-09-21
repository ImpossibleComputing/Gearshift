from pathlib import Path
import pytest
from gearshift.phase2_pause import is_paused, require_unpaused, pause_marker
def test_explicit_pause_blocks_dispatch_even_when_marker_body_is_empty(tmp_path):
    require_unpaused(tmp_path)
    p=pause_marker(tmp_path);p.parent.mkdir(parents=True);p.write_text("{}")
    assert is_paused(tmp_path)
    with pytest.raises(SystemExit,match="explicitly paused"):require_unpaused(tmp_path)
def test_pause_is_scoped_to_repository(tmp_path):
    first=tmp_path/"one";second=tmp_path/"two"
    p=pause_marker(first);p.parent.mkdir(parents=True);p.write_text("{}")
    assert is_paused(first)
    assert not is_paused(second)
