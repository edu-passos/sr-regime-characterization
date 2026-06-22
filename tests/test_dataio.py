from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from symbolic_structural_profiling.dataio import load_labeled_timeseries_csv


def test_load_labeled_timeseries_csv_validates_and_loads(tmp_path: Path):
    df = pd.DataFrame(
        {
            "t": [3, 1, 2, 4, 5, 6] * 12,
            "regime": ["A", "A", "B", "B", "A", "B"] * 12,
            "y": [0.1, 0.2, 0.3, 0.0, -0.2, 0.4] * 12,
            "x1": [1, 2, 3, 4, 5, 6] * 12,
            "x2": [2, 3, 4, 5, 6, 7] * 12,
            "tag": ["k"] * 72,
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)

    data = load_labeled_timeseries_csv(
        csv_path=path,
        label_col="regime",
        feature_cols=["y", "x1", "x2"],
        time_col="t",
    )

    assert data.values.shape == (72, 3)
    assert list(data.feature_names) == ["y", "x1", "x2"]
    assert set(data.labels.tolist()) == {"A", "B"}


def test_load_labeled_timeseries_csv_rejects_missing_label(tmp_path: Path):
    df = pd.DataFrame({"x": list(range(60)), "y": list(range(60))})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing label column"):
        load_labeled_timeseries_csv(path, label_col="regime")


def test_load_labeled_timeseries_csv_preserves_record_ids_and_sorts_within_record(tmp_path: Path):
    df = pd.DataFrame(
        {
            "record_id": ["r2", "r1", "r2", "r1"] * 20,
            "t": [2, 2, 1, 1] * 20,
            "regime": ["A", "A", "B", "B"] * 20,
            "y": np.linspace(0.0, 1.0, 80),
        }
    )
    path = tmp_path / "with_records.csv"
    df.to_csv(path, index=False)

    data = load_labeled_timeseries_csv(
        path,
        label_col="regime",
        feature_cols=["y"],
        time_col="t",
        record_col="record_id",
        normalize=False,
    )

    assert data.record_ids is not None
    assert data.record_ids[0] == "r1"
    assert data.record_ids[-1] == "r2"
    assert set(data.record_ids.tolist()) == {"r1", "r2"}
