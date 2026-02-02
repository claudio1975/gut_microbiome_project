# utils/data_utils.py
import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    Values in `override` take precedence over `base`.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def _deep_set(d: Dict[str, Any], keys: Tuple[str, ...], value: Any) -> None:
    """
    Safely set a nested key in a dict, creating intermediate dicts as needed.

    Example:
      _deep_set(cfg, ("data", "dataset_path"), "path.csv")
    """
    cur: Dict[str, Any] = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def load_config() -> Dict[str, Any]:
    """
    Load configuration with the following precedence (low → high):

      1. base.yaml (required)  -> ./configs/base.yaml
      2. dataset-specific config (--config) [optional]
      3. CLI overrides (--dataset_path, --project, --space_id)
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--space_id", type=str, default=None)

    args, _ = parser.parse_known_args()

    base_candidates = [
        Path("configs/base.yaml"),
        Path("base.yaml"),
    ]

    base_path = next((p for p in base_candidates if p.exists()), None)
    if base_path is None:
        raise FileNotFoundError(
            "base.yaml is required but was not found. Looked for: configs/base.yaml and base.yaml"
        )

    with open(base_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f) or {}

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        with open(cfg_path, "r") as f:
            override_cfg = yaml.safe_load(f) or {}
        config = _deep_update(config, override_cfg)

    if args.dataset_path:
        _deep_set(config, ("data", "dataset_path"), args.dataset_path)

    if args.project:
        _deep_set(config, ("tracking", "project"), args.project)

    if args.space_id:
        # Only set this if explicitly passed. In local mode, main.py will remove it anyway.
        _deep_set(config, ("tracking", "space_id"), args.space_id)

    return config


def split_dataset_df(
    dataset_df: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified train/test split on a dataset DataFrame.

    Args:
        dataset_df: DataFrame with at least a 'label' column.
        test_size: Fraction of samples to hold out for testing.
        random_state: Random seed for reproducibility.

    Returns:
        (train_df, test_df) both with reset indices.
    """
    train_df, test_df = train_test_split(
        dataset_df,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset_df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_data(dataset_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array(dataset_df["embedding"].tolist())
    y = np.array(dataset_df["label"])
    return X, y


def df_json_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trackio/Gradio JSON encoding fails on NaN/Inf.
    Make DataFrame safe for tracker.Table by converting:
      inf/-inf -> NaN
      NaN      -> None
    """
    out = df.replace([np.inf, -np.inf], np.nan)
    out = out.astype(object).where(pd.notnull(out), None)
    return out
