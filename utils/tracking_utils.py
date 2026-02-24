# utils/tracking_utils.py
from __future__ import annotations

import math
from pathlib import Path
from omegaconf import DictConfig
import numpy as np
import trackio as wandb  # Trackio exposes a wandb-like API
from rich.console import Console
import pandas as pd
from PIL import Image as PILImage
import httpx
import json
from typing import Any, Dict, List, Optional
import os
from matplotlib import pyplot as plt
from utils.evaluation_utils import ResultsManager


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
    if isinstance(_type, str) and any(
        tok in _type.lower() for tok in ("image", "table")
    ):
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
    config: DictConfig,
    *,
    run_name: Optional[str] = None,
    group: Optional[str] = None,
    console: Console = Console(),
) -> Optional[Any]:
    """
    Returns a Trackio run proxy (wandb-like), or None if disabled.
    Proxy guarantees .log() won't crash on NaN/Inf and won't break media objects.
    """
    if not config.tracking.enabled:
        return None

    name = run_name or config.tracking.run_name.strip()
    if not name:
        console.print(
            "⚠️  Tracking enabled but run_name was not provided. "
            "Skipping tracker init to avoid random run names.",
            style="error",
        )
        return None

    space_id = config.tracking.space_id

    # sanitize config too (it may contain numpy scalars)
    safe_config = _sanitize_for_trackio(config)

    run = wandb.init(
        project=config.get("project", "gut-microbiome"),
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


def _trackio_table_from_df(tracker, df: pd.DataFrame, *, max_rows: int = 500):
    """
    Create a Trackio Table from a DataFrame using plain Python rows.

    Passing a DataFrame directly can leak NaN/Inf due to dtype coercion.
    Building row/column lists after cleaning keeps the payload JSON-safe.
    """
    if tracker is None:
        return None

    df = df.copy()
    if len(df) > max_rows:
        df = df.head(max_rows)

    # Normalize infinities first.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Use object dtype so None stays None (and doesn't get coerced back to NaN).
    df = df.astype(object)

    def _clean_cell(x: Any) -> Any:
        if isinstance(x, (np.generic,)):
            x = x.item()

        if x is None:
            return None

        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return None
            return float(x)

        if isinstance(x, (int, bool, str)):
            return x

        try:
            if pd.isna(x):
                return None
        except Exception:
            pass

        # stringify anything odd/unexpected.
        return str(x)

    columns = [str(c) for c in df.columns]
    rows: List[List[Any]] = []
    for _, row in df.iterrows():
        rows.append([_clean_cell(v) for v in row.tolist()])

    # Trackio exposes Table either on the run object or via the module alias.
    if hasattr(tracker, "Table"):
        return tracker.Table(data=rows, columns=columns)

    return wandb.Table(data=rows, columns=columns)


def _sanitize_json_payload(x: Any) -> Any:
    """
    Recursively sanitize JSON payloads by converting NaN/Inf to None and
    converting numpy scalars to native Python types.
    """
    if x is None or isinstance(x, (bool, int, str)):
        return x

    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x

    try:
        if isinstance(x, np.floating):
            v = float(x)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.ndarray):
            return _sanitize_json_payload(x.tolist())
    except Exception:
        pass

    if isinstance(x, dict):
        return {str(k): _sanitize_json_payload(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_sanitize_json_payload(v) for v in x]

    # Leave unknown objects untouched (httpx/gradio may still handle them).
    return x


def _patch_httpx_nan_guard() -> None:
    """
    Patch httpx so JSON payloads are sanitized before sending.

    Trackio can push data over httpx, and NaN/Inf values in JSON can cause
    request serialization to fail. This patch keeps uploads robust.
    """

    _real_post = httpx.post
    _real_request = httpx.request

    def post(*args, **kwargs):
        if "json" in kwargs:
            kwargs["json"] = _sanitize_json_payload(kwargs["json"])
        return _real_post(*args, **kwargs)

    def request(method, url, *args, **kwargs):
        if "json" in kwargs:
            kwargs["json"] = _sanitize_json_payload(kwargs["json"])
        return _real_request(method, url, *args, **kwargs)

    httpx.post = post
    httpx.request = request


def _log_table_media(
    tracker_obj: Any, key: str, df: pd.DataFrame, max_rows: int = 500
) -> None:
    if tracker_obj is None:
        return
    if not hasattr(tracker_obj, "Table"):
        return

    table = _df_to_table(tracker_obj, df, max_rows=max_rows)
    tracker_obj.log({key: table})


def _log_tags(tracker, tags: List[str]) -> None:
    """
    Store tags as a single JSON string to keep the dashboard tidy.
    """
    if tracker is None:
        return
    safe_log(tracker, {"meta/tags_json": json.dumps(tags)})


def _sanitize_df_for_trackio_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe for Trackio Table serialization:
      - +/-inf -> NaN -> None
      - numpy scalars -> python scalars
      - column names -> strings
    """

    def _is_bad_float(v: Any) -> bool:
        return isinstance(v, (float, np.floating)) and (
            math.isnan(float(v)) or math.isinf(float(v))
        )

    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.map(lambda x: x.item() if isinstance(x, np.generic) else x)
    df = df.where(pd.notna(df), None)
    df = df.map(lambda x: None if _is_bad_float(x) else x)
    df.columns = [str(c) for c in df.columns]
    return df


def _df_to_table_png(
    df: pd.DataFrame, out_path, *, max_rows: int = 60, max_cols: int = 20
) -> None:
    """
    Render a dataframe to a PNG image (useful when table rendering is flaky).
    """
    df = df.copy()

    if len(df) > max_rows:
        df = df.head(max_rows)
    if df.shape[1] > max_cols:
        df = df.iloc[:, :max_cols]

    df = df.fillna("").astype(str)

    row_h = 0.35
    col_w = 1.8
    fig_w = max(10, min(30, df.shape[1] * col_w))
    fig_h = max(4, min(30, (len(df) + 1) * row_h))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _tracking_cache_hint() -> str:
    """
    Best-effort hint for where Trackio stores its local cache.
    This is informational only and not used programmatically.
    """
    home = os.path.expanduser("~")
    return os.path.join(home, ".cache", "huggingface", "trackio")


def _print_where_to_check_results(
    *, publish: bool, project: str, space_id: str
) -> None:
    print("\n" + "-" * 80)
    print("📍 Where to check Trackio results")

    if publish:
        print("🚀 Publish mode: runs are pushed to the Hugging Face Space dashboard.")
        if space_id:
            print(f"Space:   {space_id}")
        print(f"Project: {project}")
        if space_id:
            print("\nOpen the Space in your browser:")
            print(f"  https://huggingface.co/spaces/{space_id}")

        print("\nIf you logged locally and want to sync later:")
        print(f'  trackio sync --project "{project}" --space-id "{space_id}"')
    else:
        print("🧪 Local mode: runs are logged locally (not pushed to Hugging Face).")
        print(f"Project: {project}")
        print("\nOpen the local Trackio dashboard UI:")
        print(f'  trackio show --project "{project}"')
        print("  # or just: trackio show")
        print("\nLocal Trackio cache (FYI):")
        print(f"  {_tracking_cache_hint()}")

    print("-" * 80 + "\n")


def _apply_tracking_routing(
    config: DictConfig, data_id: str, console: Console = Console()
) -> None:
    """
    Configure Trackio logging mode based on publish_to_main.

    - Local mode:  <data_id>-local  (logs locally only; no HF push)
    - Publish:     <data_id>        (pushes to Space dashboard)

    In local mode, HF-related fields are cleared to avoid accidental pushes.
    """

    if config.tracking.publish_to_main:
        final_project = data_id
        mode = "publish"
    else:
        final_project = f"{data_id}-local"
        mode = "local"

    config.tracking.enabled = True
    config.tracking.project = final_project

    if mode == "local":
        config.tracking.space_id = None

        console.print("\n" + "=" * 80)
        console.print(
            "🧪 LOCAL MODE: Trackio logging is local only (no Hugging Face push).",
            style="info",
        )
        console.print("Runs are stored on this machine.", style="info")
        console.print(f"Project: {final_project}", style="info")
        console.print(
            "To publish to the Space dashboard, re-run with: --publish_to_main",
            style="info",
        )
        console.print("=" * 80 + "\n")
    else:
        console.print("\n" + "=" * 80)
        console.print(
            "⚠️  PUBLISH MODE ENABLED: pushing runs to Hugging Face Space dashboard"
        )
        console.print(f"    Space:   {config.tracking.space_id}", style="info")
        console.print(f"    Project: {final_project}", style="info")
        console.print("=" * 80 + "\n")

    _print_where_to_check_results(
        publish=config.tracking.publish_to_main,
        project=final_project,
        space_id=config.tracking.space_id,
    )


def _get_tags(
    config: DictConfig,
    *,
    dataset_family: str,
    data_id: str,
    pipeline: str,
    clf_type: str,
) -> List[str]:
    """
    Trackio tag support can vary across backends. We avoid passing tags into init()
    and instead log them as metadata for filtering/search.
    """
    tags = config.tracking.tags
    tags += [
        f"dataset:{dataset_family}",
        f"data_id:{data_id}",
        f"pipeline:{pipeline}",
        f"model:{clf_type}",
    ]

    seen = set()
    out: List[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _make_run_name(config: DictConfig, clf_type: str, pipeline: str) -> str:
    dataset_path = config.data.dataset_path
    _, data_id = _dataset_family_and_id(dataset_path)
    if clf_type == "summary":
        return f"{data_id}__summary"
    return f"{data_id}__{clf_type}"


def _make_group_name(config: dict, pipeline: str) -> str:
    dataset_path = config.data.dataset_path
    _, data_id = _dataset_family_and_id(dataset_path)
    return f"{data_id}"


def _run_metadata(config: DictConfig, clf_type: str, pipeline: str) -> Dict[str, Any]:
    dataset_path = config.data.dataset_path
    dataset_family, data_id = _dataset_family_and_id(dataset_path)

    user = (
        os.getenv("TRACKIO_USER")
        or os.getenv("GITHUB_ACTOR")
        or os.getenv("USER")
        or os.getenv("USERNAME")
        or "unknown"
    )
    host = os.uname().nodename if hasattr(os, "uname") else "unknown"

    gs_cv = config.evaluation.grid_search_cv_folds
    fe_cv = config.evaluation.cv_folds
    scoring = config.evaluation.grid_search_scoring
    gs_rs = config.evaluation.grid_search_random_state
    fe_rs = config.evaluation.final_eval_random_state

    return {
        "meta/user": user,
        "meta/host": host,
        "meta/experiment": pipeline,
        "meta/dataset_family": dataset_family,
        "meta/data_id": data_id,
        "meta/dataset_path": dataset_path,
        "meta/classifier": clf_type,
        "meta/protocol_version": "v1",
        "meta/grid_search_cv_folds": str(gs_cv) if gs_cv is not None else "",
        "meta/final_eval_cv_folds": str(fe_cv) if fe_cv is not None else "",
        "meta/grid_search_scoring": str(scoring) if scoring is not None else "",
        "meta/grid_search_random_state": str(gs_rs) if gs_rs is not None else "",
        "meta/final_eval_random_state": str(fe_rs) if fe_rs is not None else "",
    }


def _log_artifacts(
    tracker, results_manager: ResultsManager, classifier_name: str
) -> None:
    if tracker is None:
        return

    out = Path(results_manager.output_dir)
    artifacts = {
        "artifacts/output_dir": str(out),
        "artifacts/classification_report_csv": str(
            out / f"{classifier_name}_classification_report.csv"
        ),
        "artifacts/roc_curve_png": str(out / f"{classifier_name}_roc_curve.png"),
        "artifacts/confusion_matrix_png": str(
            out / f"{classifier_name}_confusion_matrix.png"
        ),
        "artifacts/confusion_matrix_norm_true_png": str(
            out / f"{classifier_name}_confusion_matrix_norm_true.png"
        ),
    }
    safe_log(tracker, artifacts)


def _log_eval_images(tracker, output_dir: Path, classifier_name: str) -> None:
    if tracker is None:
        return

    output_dir = Path(output_dir)
    pngs = {
        "media/roc_curve": output_dir / f"{classifier_name}_roc_curve.png",
        "media/confusion_matrix": output_dir
        / f"{classifier_name}_confusion_matrix.png",
        "media/confusion_matrix_norm_true": output_dir
        / f"{classifier_name}_confusion_matrix_norm_true.png",
    }

    existing = {k: p for k, p in pngs.items() if p.exists()}
    if not existing:
        return

    payload: Dict[str, Any] = {}
    for k, p in existing.items():
        img = PILImage.open(p).convert("RGB")
        payload[k] = wandb.Image(img)

    # Keep a fixed step so the dashboard doesn't treat each upload as a new series.
    tracker.log(payload, step=0)


def _log_only_numeric_metrics(tracker, payload: Dict[str, Any]) -> None:
    """
    Trackio rejects NaN/Inf during JSON serialization. Only log clean numeric values.
    """
    if tracker is None:
        return

    clean: Dict[str, Any] = {}
    for k, v in payload.items():
        if v is None:
            continue
        if isinstance(v, bool):
            continue

        if isinstance(v, (int, float)):
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                continue
            clean[k] = fv
            continue

        if isinstance(v, np.generic) and np.isscalar(v):
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                continue
            clean[k] = fv

    if clean:
        safe_log(tracker, clean)


def _log_metrics_block(
    tracker, *, pipeline: str, clf_type: str, metrics_obj: Any
) -> None:
    if tracker is None:
        return

    numeric = {
        "metrics/roc_auc": getattr(metrics_obj, "roc_auc", None),
        "metrics/final_roc_auc": getattr(metrics_obj, "roc_auc", None),
        "metrics/cv_folds": getattr(metrics_obj, "cv_folds", None),
        "metrics/grid_search_best_score": getattr(metrics_obj, "best_score", None),
    }
    _log_only_numeric_metrics(tracker, numeric)

    best_params = getattr(metrics_obj, "best_params", None)
    safe_log(
        tracker,
        {
            "meta/pipeline": pipeline,
            "meta/classifier_type": clf_type,
            "meta/classifier_name": str(getattr(metrics_obj, "classifier_name", "")),
            "meta/best_params_json": json.dumps(best_params, default=str)
            if best_params is not None
            else "",
        },
    )


def _log_classification_summary_metrics(tracker, metrics_obj: Any) -> None:
    if tracker is None:
        return

    y_true = getattr(metrics_obj, "y_true", None)
    y_pred = getattr(metrics_obj, "y_pred", None)
    if y_true is None or y_pred is None:
        return

    try:
        from sklearn.metrics import classification_report
    except Exception:
        return

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    payload: Dict[str, Any] = {}

    if "accuracy" in report:
        payload["metrics/accuracy"] = report["accuracy"]

    macro = report.get("macro avg", {}) or {}
    payload["metrics/precision_macro"] = macro.get("precision", None)
    payload["metrics/recall_macro"] = macro.get("recall", None)
    payload["metrics/f1_macro"] = macro.get("f1-score", None)

    wavg = report.get("weighted avg", {}) or {}
    payload["metrics/precision_weighted"] = wavg.get("precision", None)
    payload["metrics/recall_weighted"] = wavg.get("recall", None)
    payload["metrics/f1_weighted"] = wavg.get("f1-score", None)

    for label, stats in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        if isinstance(stats, dict):
            payload[f"metrics/f1_class/{label}"] = stats.get("f1-score", None)
            payload[f"metrics/support_class/{label}"] = stats.get("support", None)

    _log_only_numeric_metrics(tracker, payload)


# ----------------------------
# Trackio-safe Table helpers (avoid NaN/Inf in JSON)
# ----------------------------
def _clean_cell(x: Any) -> Any:
    """
    Convert numpy scalars to native types and normalize NaN/Inf/NA values.
    """
    if isinstance(x, np.generic):
        x = x.item()

    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    try:
        if x is pd.NA:
            return None
    except Exception:
        pass

    return x


def _df_to_table(tracker_obj: Any, df: pd.DataFrame, max_rows: int = 500):
    """
    Avoid Table(dataframe=df): dtype coercion can reintroduce NaN/Inf.
    Build Table(data=..., columns=...) after cleaning instead.
    """
    if len(df) > max_rows:
        df = df.head(max_rows)

    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    cols = list(df.columns)
    rows = df.values.tolist()
    rows = [[_clean_cell(v) for v in row] for row in rows]

    return tracker_obj.Table(data=rows, columns=cols)


def _log_comparison_roc_curve(tracker, output_dir: Path) -> None:
    if tracker is None:
        return

    output_dir = Path(output_dir)
    p = output_dir / "comparison_roc_curves.png"
    if not p.exists():
        safe_log(tracker, {"media/comparison_roc_curves_path_missing": str(p)})
        return

    try:
        img = PILImage.open(p).convert("RGB")
        tracker.log({"media/comparison_roc_curves": wandb.Image(img)}, step=0)
    except Exception as e:
        safe_log(
            tracker,
            {
                "media/comparison_roc_curves_error": str(e),
                "media/comparison_roc_curves_path": str(p),
            },
        )


def _round_numeric(v, ndigits: int = 3):
    """
    Normalize numeric values for display:
    - round floats to fixed precision
    - leave non-numeric values untouched
    """
    try:
        if isinstance(v, (np.floating,)):
            v = v.item()
        if isinstance(v, float):
            return round(v, ndigits)
    except Exception:
        pass
    return v


def _log_session_outputs_summary_run(
    config: DictConfig, results_manager: ResultsManager
) -> None:
    """
    Log session-level outputs (combined tables/plots) into a summary run.
    """
    if not config.tracking.enabled:
        return

    output_dir = Path(results_manager.output_dir)
    dataset_path = config.data.dataset_path
    dataset_family, data_id = _dataset_family_and_id(dataset_path)

    run_name = _make_run_name(config, "summary", pipeline="summary")
    group_name = _make_group_name(config, pipeline="summary")
    tracker = get_tracker(config, run_name=run_name, group=group_name)
    if tracker is None:
        return

    _log_comparison_roc_curve(tracker, output_dir)
    tags = _get_tags(
        config,
        dataset_family=dataset_family,
        data_id=data_id,
        pipeline="summary",
        clf_type="summary",
    )
    safe_log(tracker, {"meta/tags_json": json.dumps(tags)})

    safe_log(tracker, _run_metadata(config, clf_type="summary", pipeline="summary"))
    safe_log(tracker, {"artifacts/output_dir": str(output_dir)})

    combined_csv = output_dir / "combined_classification_report.csv"
    if combined_csv.exists():
        try:
            df_raw = pd.read_csv(combined_csv)

            # Build a long-form table: one metric per row
            rows = []
            metric_cols = [
                c for c in df_raw.columns if c not in ("classifier", "class")
            ]

            for _, r in df_raw.iterrows():
                classifier = str(r.get("classifier", ""))
                cls = str(r.get("class", ""))

                for m in metric_cols:
                    v = r.get(m, None)

                    try:
                        if v is None:
                            vv = None
                        elif isinstance(v, (np.generic,)):
                            v = v.item()
                            vv = (
                                None
                                if (
                                    isinstance(v, float)
                                    and (math.isnan(v) or math.isinf(v))
                                )
                                else _round_numeric(v)
                            )
                        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                            vv = None
                        else:
                            vv = _round_numeric(v)
                    except Exception:
                        vv = v

                    if vv is None or vv == "":
                        continue

                    rows.append(
                        {
                            "classifier": classifier,
                            "class": cls,
                            "metric": str(m),
                            "value": vv,
                        }
                    )

            df = pd.DataFrame(rows, columns=["classifier", "class", "metric", "value"])

            # Wide-format table for quick visual inspection
            df_wide = pd.read_csv(combined_csv)
            df_wide.columns = [str(c).strip() for c in df_wide.columns]

            df_wide = df_wide.replace([np.inf, -np.inf], np.nan)
            df_wide = df_wide.where(pd.notna(df_wide), "")

            # Convert numpy scalars and round floats for display
            df_wide = df_wide.applymap(
                lambda x: _round_numeric(x.item() if hasattr(x, "item") else x)
            )

            tracker.log(
                {
                    "tables/combined_classification_report_wide": tracker.Table(
                        dataframe=df_wide
                    )
                },
                step=0,
            )
            print("[trackio] logged tables/combined_classification_report")

        except Exception as e:
            print(f"[trackio] FAILED combined_classification_report table: {e}")
            safe_log(
                tracker,
                {
                    "errors/combined_classification_report_table": str(e),
                    "paths/combined_classification_report_csv": str(combined_csv),
                },
            )

    best_json = output_dir / "best_params_summary.json"
    if best_json.exists():
        try:
            with open(best_json, "r") as f:
                obj = json.load(f)

            rows = []
            for model, params in obj.items():
                for k, v in params.items():
                    rows.append({"model": model, "param": k, "value": v})

            df = pd.DataFrame(rows, columns=["model", "param", "value"])

            tracker.log(
                {"tables/best_params_summary": tracker.Table(dataframe=df)}, step=0
            )

        except Exception:
            safe_log(tracker, {"tables/best_params_summary_path": str(best_json)})

    tracker.finish()


def _dataset_family_and_id(dataset_path: str) -> Tuple[str, str]:
    """
    Extract dataset family and a short dataset id from the dataset path.

    Returns:
      dataset_family: e.g. "Diabimmune", "Goldberg", "Tanaka", "Gadir"
      data_id:        e.g. "month_2", "all_groups", "t1"
    """
    p = Path(dataset_path)
    data_id = (p.stem or "dataset").lower()

    parts = [x.lower() for x in p.parts]
    dataset_family = "dataset"
    if "datasets" in parts:
        idx = parts.index("datasets")
        if idx + 1 < len(parts):
            dataset_family = p.parts[idx + 1]

    dataset_family = dataset_family[:1].upper() + dataset_family[1:]
    return dataset_family, data_id


def _save_best_params_summary(output_dir: Path, best_params: Dict[str, Dict]) -> None:
    summary_path = output_dir / "best_params_summary.json"
    with open(summary_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"\n✓ Best params summary saved: {summary_path}")


def _confirm_publish_to_main() -> None:
    """
    Ask for explicit confirmation before publishing runs/media to a Hugging Face Space.
    This is intentionally conservative: anything other than an explicit "yes" cancels.
    """
    try:
        print("\n⚠️  You are about to publish results to the Hugging Face Space.")
        print(
            "    This may overwrite/update media, tables, and metrics on the shared dashboard."
        )
        resp = input("👉 Continue? [yes / no]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Publish cancelled (no confirmation).")
        raise SystemExit(1)

    if resp not in {"yes", "y"}:
        print("❌ Publish cancelled by user.")
        raise SystemExit(1)

    print("✅ Publish confirmed.\n")
