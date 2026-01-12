# utils/tracking_utils.py
from __future__ import annotations
from typing import Any, Dict, Optional


def get_tracker(
    config: Dict[str, Any],
    *,
    run_name: Optional[str] = None,
    group: Optional[str] = None,
) -> Optional[Any]:
    """
    Returns a Trackio tracker (wandb-like module) or None if disabled.

    The returned object supports:
      - log(dict)
      - finish()
    """
    tcfg = (config or {}).get("tracking", {}) or {}
    if not tcfg.get("enabled", False):
        return None

    import trackio as wandb

    wandb.init(
        project=tcfg.get("project", "gut-microbiome"),
        name=run_name or tcfg.get("run_name"),
        group=group,
        config=config,
        space_id=tcfg.get("space_id"),
    )
    return wandb


def safe_log(tracker: Optional[Any], data: Dict[str, Any]) -> None:
    """
    Safely log metrics or metadata to the tracker if enabled.
    """
    if tracker is None:
        return
    tracker.log(data)