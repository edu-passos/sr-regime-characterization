import numpy as np
import pytest

from symbolic_structural_profiling.windowing import (
    temporal_train_validation_split,
    temporal_train_validation_splits,
)


def test_tail_scheme_matches_single_split():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.arange(20, dtype=float)

    single = temporal_train_validation_split(X=X, y=y, validation_split=0.3)
    many = temporal_train_validation_splits(X=X, y=y, validation_split=0.3, scheme="tail")

    assert len(many) == 1
    for got, expected in zip(many[0], single):
        np.testing.assert_array_equal(got, expected)


def test_blocked_scheme_uses_two_disjoint_validation_blocks():
    X = np.arange(100, dtype=float).reshape(50, 2)
    y = np.arange(50, dtype=float)

    splits = temporal_train_validation_splits(X=X, y=y, validation_split=0.2, scheme="blocked")
    assert len(splits) == 2

    X_train_1, y_train_1, X_val_1, y_val_1 = splits[0]
    X_train_2, y_train_2, X_val_2, y_val_2 = splits[1]

    assert X_train_1.shape[0] == 30
    assert X_val_1.shape[0] == 10
    assert X_train_2.shape[0] == 40
    assert X_val_2.shape[0] == 10

    np.testing.assert_array_equal(X_val_1, X[30:40])
    np.testing.assert_array_equal(y_val_1, y[30:40])
    np.testing.assert_array_equal(X_val_2, X[40:50])
    np.testing.assert_array_equal(y_val_2, y[40:50])


def test_blocked_scheme_falls_back_to_tail_if_window_too_short():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)

    blocked = temporal_train_validation_splits(X=X, y=y, validation_split=0.49, scheme="blocked")
    tail = temporal_train_validation_split(X=X, y=y, validation_split=0.49)

    assert len(blocked) == 1
    for got, expected in zip(blocked[0], tail):
        np.testing.assert_array_equal(got, expected)


def test_invalid_validation_scheme_raises():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)

    with pytest.raises(ValueError, match="validation_scheme must be one of: tail, blocked"):
        temporal_train_validation_splits(X=X, y=y, validation_split=0.3, scheme="unknown")


def test_zero_validation_split_uses_full_window_for_train_and_validation():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.arange(12, dtype=float)

    X_train, y_train, X_val, y_val = temporal_train_validation_split(X=X, y=y, validation_split=0.0)
    np.testing.assert_array_equal(X_train, X)
    np.testing.assert_array_equal(y_train, y)
    np.testing.assert_array_equal(X_val, X)
    np.testing.assert_array_equal(y_val, y)


def test_blocked_scheme_with_zero_validation_split_returns_single_full_window_split():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.arange(12, dtype=float)

    splits = temporal_train_validation_splits(X=X, y=y, validation_split=0.0, scheme="blocked")
    assert len(splits) == 1

    X_train, y_train, X_val, y_val = splits[0]
    np.testing.assert_array_equal(X_train, X)
    np.testing.assert_array_equal(y_train, y)
    np.testing.assert_array_equal(X_val, X)
    np.testing.assert_array_equal(y_val, y)
