"""Explicit owner pause takes precedence over queue/recovery/completion."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def pause_marker(root=None):
    return Path(root or ROOT) / "evidence/phase2/studio_transfer/JUDGING_PAUSED.json"
def is_paused(root=None):
    return pause_marker(root).exists()
def require_unpaused(root=None):
    if is_paused(root):
        raise SystemExit("Phase-2 judging is explicitly paused by the owner. See PAUSE_STATUS.md; no dispatch or automatic resumption is authorized.")
