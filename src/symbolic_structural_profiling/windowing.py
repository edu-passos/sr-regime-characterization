from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(slots=True)
class FeatureMeta:
    column: int
    variable: int
    lag: int
    name: str


@dataclass(slots=True)
class Window:
    index: int
    start: int
    end: int
    label: str
    values: np.ndarray
    record_id: str | None = None


def segment_labeled_series(
    values: np.ndarray,
    labels: Sequence[str],
    window_size: int,
    step_size: int,
    require_homogeneous_labels: bool = True,
    record_ids: Sequence[str] | None = None,
) -> List[Window]:
    if values.ndim != 2:
        raise ValueError("values must have shape [time, variables]")
    if len(labels) != values.shape[0]:
        raise ValueError("labels length must match time axis")
    if record_ids is not None and len(record_ids) != values.shape[0]:
        raise ValueError("record_ids length must match time axis")
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")

    labels_array = np.asarray(labels)
    records_array = np.asarray(record_ids, dtype=str) if record_ids is not None else None
    windows: List[Window] = []
    idx = 0

    for start in range(0, values.shape[0] - window_size + 1, step_size):
        end = start + window_size
        window_labels = labels_array[start:end]
        unique, counts = np.unique(window_labels, return_counts=True)
        majority = unique[np.argmax(counts)]

        if require_homogeneous_labels and len(unique) > 1:
            continue

        record_id: str | None = None
        if records_array is not None:
            record_slice = records_array[start:end]
            unique_records = np.unique(record_slice)
            if len(unique_records) != 1:
                continue
            record_id = str(unique_records[0])

        window = Window(
            index=idx,
            start=start,
            end=end,
            label=str(majority),
            values=values[start:end],
            record_id=record_id,
        )
        windows.append(window)
        idx += 1
    return windows


def build_lagged_design(
    window_values: np.ndarray,
    max_lag: int,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray, List[FeatureMeta]]:
    n_time, n_vars = window_values.shape
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")
    if n_time <= max_lag:
        raise ValueError("window is too short for requested max_lag")
    if target_index < 0 or target_index >= n_vars:
        raise ValueError("invalid target_index")

    n_samples = n_time - max_lag
    n_features = n_vars * max_lag
    X = np.zeros((n_samples, n_features), dtype=float)
    y = np.zeros(n_samples, dtype=float)

    feature_meta: List[FeatureMeta] = []
    col = 0
    for var in range(n_vars):
        for lag in range(1, max_lag + 1):
            feature_meta.append(
                FeatureMeta(column=col, variable=var, lag=lag, name=f"x{var}_lag{lag}")
            )
            col += 1

    for row, t in enumerate(range(max_lag, n_time)):
        for m in feature_meta:
            X[row, m.column] = window_values[t - m.lag, m.variable]
        y[row] = window_values[t, target_index]
    return X, y, feature_meta


def temporal_train_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0, 1)")
    if validation_split == 0.0:
        return X, y, X, y

    n = X.shape[0]
    split_idx = max(1, min(n - 1, int(round(n * (1.0 - validation_split)))))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def temporal_train_validation_splits(
    X: np.ndarray,
    y: np.ndarray,
    validation_split: float,
    scheme: str = "tail",
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if scheme == "tail":
        return [temporal_train_validation_split(X=X, y=y, validation_split=validation_split)]

    if scheme != "blocked":
        raise ValueError("validation_scheme must be one of: tail, blocked")

    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0, 1)")
    if validation_split == 0.0:
        return [(X, y, X, y)]

    n = X.shape[0]
    val_size = max(1, int(round(n * validation_split)))
    if 2 * val_size >= n:
        return [temporal_train_validation_split(X=X, y=y, validation_split=validation_split)]

    split1_train_end = n - (2 * val_size)
    split1_val_end = split1_train_end + val_size
    split2_train_end = n - val_size

    return [
        (
            X[:split1_train_end],
            y[:split1_train_end],
            X[split1_train_end:split1_val_end],
            y[split1_train_end:split1_val_end],
        ),
        (
            X[:split2_train_end],
            y[:split2_train_end],
            X[split2_train_end:],
            y[split2_train_end:],
        ),
    ]
