import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple

def load_config(config_path: str = "config.yaml"):
    """
    Load configuration from YAML file.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(dataset_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array(dataset_df['embedding'].tolist())
    y = np.array(dataset_df['label'])
    return X, y