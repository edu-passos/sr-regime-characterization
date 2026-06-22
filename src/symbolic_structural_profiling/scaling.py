from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class WindowScaling:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float


def standardize_window_design(
    X: np.ndarray,
    y: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, WindowScaling]:
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float)

    if not enabled:
        scaling = WindowScaling(
            x_mean=np.zeros(X_array.shape[1], dtype=float),
            x_scale=np.ones(X_array.shape[1], dtype=float),
            y_mean=0.0,
            y_scale=1.0,
        )
        return X_array.copy(), y_array.copy(), scaling

    x_mean = np.mean(X_array, axis=0)
    x_scale = np.std(X_array, axis=0)
    x_scale = np.where(x_scale < 1e-9, 1.0, x_scale)

    y_mean = float(np.mean(y_array))
    y_scale = float(np.std(y_array))
    if y_scale < 1e-9:
        y_scale = 1.0

    X_scaled = (X_array - x_mean[None, :]) / x_scale[None, :]
    y_scaled = (y_array - y_mean) / y_scale

    scaling = WindowScaling(
        x_mean=x_mean.astype(float, copy=False),
        x_scale=x_scale.astype(float, copy=False),
        y_mean=y_mean,
        y_scale=y_scale,
    )
    return X_scaled, y_scaled, scaling


def invert_scaled_target(values: np.ndarray, scaling: WindowScaling) -> np.ndarray:
    return np.asarray(values, dtype=float) * float(scaling.y_scale) + float(scaling.y_mean)
