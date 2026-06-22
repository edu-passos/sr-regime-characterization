from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class LabeledTimeSeriesDataset:
    values: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    record_ids: np.ndarray | None = None


def parse_feature_columns(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    return cols or None


def load_labeled_timeseries_csv(
    csv_path: str | Path,
    label_col: str,
    feature_cols: Sequence[str] | None = None,
    time_col: str | None = None,
    record_col: str | None = None,
    normalize: bool = True,
    dropna: bool = True,
) -> LabeledTimeSeriesDataset:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV is empty")

    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'")

    if record_col is not None and record_col not in df.columns:
        raise ValueError(f"Missing record column '{record_col}'")

    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"Missing time column '{time_col}'")
        sort_cols = [time_col]
        if record_col is not None:
            sort_cols.insert(0, record_col)
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    elif record_col is not None:
        df = df.sort_values(record_col, kind="stable").reset_index(drop=True)

    if feature_cols is None:
        excluded = {label_col}
        if time_col is not None:
            excluded.add(time_col)
        numeric_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("No numeric feature columns available")
        selected_features = numeric_cols
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        selected_features = list(feature_cols)

    required_cols = [label_col, *selected_features]

    if dropna:
        before = len(df)
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("All rows were dropped due to missing values")
        if len(df) < before // 2:
            # Guard against accidentally losing most of the dataset.
            raise ValueError("More than half of rows were dropped; check missing-data handling")
    elif df[required_cols].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns")

    values = df[selected_features].to_numpy(dtype=float)
    labels = df[label_col].astype(str).to_numpy()
    record_ids = df[record_col].astype(str).to_numpy() if record_col is not None else None

    if values.ndim != 2:
        raise ValueError("Feature matrix must be 2D")
    if values.shape[0] < 50:
        raise ValueError("Dataset must contain at least 50 rows")
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("Feature matrix contains NaN or inf after processing")

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("Need at least two labels for regime profiling")

    if normalize:
        means = values.mean(axis=0)
        stds = values.std(axis=0)
        values = (values - means) / np.where(stds < 1e-9, 1.0, stds)

    return LabeledTimeSeriesDataset(
        values=values,
        labels=labels,
        feature_names=selected_features,
        record_ids=record_ids,
    )
