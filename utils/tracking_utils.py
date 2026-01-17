# utils/tracking_utils.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _is_bad_float(v: float) -> bool:
    return math.isnan(v) or math.isinf(v)


def _is_trackio_media_obj(x: Any) -> bool:
    """
    Trackio is wandb-like and may wrap Image/Table types under different modules.
    We detect by:
      - class name (Image/Table)
      - OR wandb-style _type fields (e.g., 'image-file', 'table', etc.)
      - OR module containing trackio/wandb
    """
    if x is None:
        return False

    t = type(x)
    name = getattr(t, "__name__", "")
    mod = getattr(t, "__module__", "") or ""

    if name in {"Image", "Table"}:
        return True

    # wandb-like datatypes have _type
    _type = getattr(x, "_type", None)
    if isinstance(_type, str) and any(tok in _type.lower() for tok in ("image", "table")):
        return True

    if any(tok in mod.lower() for tok in ("trackio", "wandb")):
        # Avoid treating everything from those modules as media,
        if name.lower().endswith("image") or name.lower().endswith("table"):
            return True

    return False


def _sanitize_for_trackio(x: Any) -> Any:
    """
    Recursively make payload JSON-safe for Trackio:
    - float NaN/Inf -> None
    - numpy scalars -> python scalars (and NaN/Inf -> None)
    - dict/list recurse
    - preserve Trackio Image/Table objects (critical!)
    - stringify only known safe non-JSON types (Path)
    """
    # Preserve Trackio media objects EXACTLY
    if _is_trackio_media_obj(x):
        return x

    if x is None or isinstance(x, (bool, int, str)):
        return x

    # Paths should be strings
    if isinstance(x, Path):
        return str(x)

    # float: remove NaN/Inf
    if isinstance(x, float):
        return None if _is_bad_float(x) else x

    # numpy scalars
    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, np.floating):
        v = float(x)
        return None if _is_bad_float(v) else v

    # numpy arrays
    if isinstance(x, np.ndarray):
        return _sanitize_for_trackio(x.tolist())

    # dict
    if isinstance(x, dict):
        return {str(k): _sanitize_for_trackio(v) for k, v in x.items()}

    # list/tuple
    if isinstance(x, (list, tuple)):
        return [_sanitize_for_trackio(v) for v in x]
    
    return x


class _TrackerProxy:
    """
    Wraps a Trackio Run object.
    Ensures EVERY .log() call sanitizes NaN/Inf but preserves media objects.
    """
    def __init__(self, run: Any, wandb_module: Any):
        self._run = run
        # expose constructors for calling code
        self.Image = getattr(wandb_module, "Image", None)
        self.Table = getattr(wandb_module, "Table", None)

    def log(self, data: dict, *args, **kwargs):
        payload = _sanitize_for_trackio(data)
        return self._run.log(payload, *args, **kwargs)

    def finish(self, *args, **kwargs):
        return self._run.finish(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._run, name)


def get_tracker(
    config: Dict[str, Any],
    *,
    run_name: Optional[str] = None,
    group: Optional[str] = None,
) -> Optional[Any]:
    """
    Returns a Trackio run proxy (wandb-like), or None if disabled.
    Proxy guarantees .log() won't crash on NaN/Inf and won't break media objects.
    """
    tcfg = (config or {}).get("tracking", {}) or {}
    if not tcfg.get("enabled", False):
        return None

    import trackio as wandb  # Trackio exposes a wandb-like API

    name = (run_name or tcfg.get("run_name") or "").strip()
    if not name:
        print(
            "⚠️  Tracking enabled but run_name was not provided. "
            "Skipping tracker init to avoid random run names."
        )
        return None

    space_id = tcfg.get("space_id", None) or None

    # sanitize config too (it may contain numpy scalars)
    safe_config = _sanitize_for_trackio(config)

    run = wandb.init(
        project=tcfg.get("project", "gut-microbiome"),
        name=name,
        group=(group or None),
        config=safe_config,
        space_id=space_id,
    )

    return _TrackerProxy(run, wandb)


def safe_log(tracker: Any, data: dict, step: int | None = None) -> None:
    """
    Safe logger for plain metrics/metadata.
    tracker is a proxy so tracker.log is already sanitized.
    """
    if tracker is None:
        return
    payload = _sanitize_for_trackio(data)
    if step is None:
        tracker.log(payload)
    else:
        tracker.log(payload, step=step)