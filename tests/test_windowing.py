import numpy as np
import pytest

from symbolic_structural_profiling.windowing import build_lagged_design, segment_labeled_series


def test_segment_labeled_series_keeps_only_homogeneous_windows_by_default():
    values = np.arange(12, dtype=float).reshape(6, 2)
    labels = ["A", "A", "B", "B", "B", "B"]

    windows = segment_labeled_series(values, labels, window_size=2, step_size=2)

    assert len(windows) == 3
    assert [window.label for window in windows] == ["A", "B", "B"]


def test_build_lagged_design_returns_full_window_fit_arrays():
    window_values = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=float,
    )

    X, y, meta = build_lagged_design(window_values, max_lag=2, target_index=1)

    assert X.shape == (2, 4)
    assert y.shape == (2,)
    np.testing.assert_array_equal(y, np.array([30.0, 40.0], dtype=float))
    assert [m.name for m in meta] == ["x0_lag1", "x0_lag2", "x1_lag1", "x1_lag2"]


def test_build_lagged_design_validates_target_index():
    window_values = np.ones((5, 2), dtype=float)

    with pytest.raises(ValueError, match="invalid target_index"):
        build_lagged_design(window_values, max_lag=2, target_index=5)


def test_segment_labeled_series_respects_record_boundaries():
    values = np.arange(16, dtype=float).reshape(8, 2)
    labels = ["A"] * 8
    record_ids = ["r1"] * 4 + ["r2"] * 4

    windows = segment_labeled_series(
        values,
        labels,
        window_size=4,
        step_size=2,
        record_ids=record_ids,
    )

    assert len(windows) == 2
    assert [window.record_id for window in windows] == ["r1", "r2"]
