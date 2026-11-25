"""Input discovery and loading.

The assessment names files with underscores (``data/train_test.csv``) while some
downloads use hyphens (``train-test.csv``); both spellings are accepted.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FILES = {
    "train": ("train_test.csv", "train-test.csv"),
    "validation": ("validation.csv",),
    "december": ("december_chart_inputs.csv", "december-chart-inputs.csv"),
    "template": ("validation_predictions_template.csv", "validation-predictions-template.csv"),
}


def find_file(data_dir: Path, *names: str) -> Path | None:
    for name in names:
        path = Path(data_dir) / name
        if path.is_file():
            return path
    return None


def require_file(data_dir: Path, key: str) -> Path:
    path = find_file(data_dir, *FILES[key])
    if path is None:
        raise FileNotFoundError(f"{key}: none of {FILES[key]} found in {Path(data_dir).resolve()}")
    return path


def load_inputs(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read train, validation and December inputs; dates are parsed, nothing else is changed."""
    return {
        "train": pd.read_csv(require_file(data_dir, "train"), parse_dates=["date"]),
        "validation": pd.read_csv(require_file(data_dir, "validation"), parse_dates=["date"]),
        "december": pd.read_csv(require_file(data_dir, "december"), parse_dates=["date"]),
    }
