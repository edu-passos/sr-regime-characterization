from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def aggregate_regime_profiles(window_summaries: pd.DataFrame) -> pd.DataFrame:
    if window_summaries.empty:
        return pd.DataFrame()

    feature_cols = [
        c
        for c in window_summaries.columns
        if c.endswith("_median")
        and c
        not in {
            "window_index",
            "start",
            "end",
            "n_models",
            "baseline_fit_mse_median",
            "linear_fit_mse_median",
            "best_symbolic_fit_mse_median",
            "seed_instability_median",
        }
    ]

    rows: List[Dict[str, float | str]] = []
    for label, group in window_summaries.groupby("label"):
        row: Dict[str, float | str] = {"label": str(label), "n_windows": float(len(group))}
        for col in feature_cols:
            base = col[: -len("_median")]
            values = group[col].to_numpy(dtype=float)
            row[f"{base}_p50"] = float(np.median(values))
            row[f"{base}_mean"] = float(np.mean(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            row[f"{base}_iqr"] = float(q75 - q25)
            row[f"{base}_q10"] = float(np.quantile(values, 0.10))
            row[f"{base}_q90"] = float(np.quantile(values, 0.90))
        rows.append(row)
    return pd.DataFrame(rows)
