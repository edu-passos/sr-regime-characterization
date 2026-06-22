import numpy as np
import pandas as pd
import pytest
import warnings

from symbolic_structural_profiling.config import EnsembleConfig
from symbolic_structural_profiling.models import ensemble as ensemble_module
from symbolic_structural_profiling.windowing import FeatureMeta


class _FakePySRRegressor:
    init_kwargs = None

    def __init__(self, **kwargs):
        _FakePySRRegressor.init_kwargs = kwargs
        self.equations_ = pd.DataFrame(
            [
                {
                    "equation": "x0 + x1",
                    "lambda_format": lambda X: X[:, 0] + X[:, 1],
                }
            ]
        )

    def fit(self, X, y):
        return self

    def predict(self, X, index=0):
        return X[:, 0] + X[:, 1]


class _FakePySRRegressorWithInvalidCandidate:
    def __init__(self, **kwargs):
        self.equations_ = pd.DataFrame(
            [
                {
                    "equation": "x0 + x1",
                    "lambda_format": lambda X: X[:, 0] + X[:, 1],
                },
                {
                    "equation": "log(x0)",
                    "lambda_format": lambda X: np.log(-np.abs(X[:, 0])),
                },
            ]
        )

    def fit(self, X, y):
        return self

    def predict(self, X, index=0):
        if int(index) == 0:
            return X[:, 0] + X[:, 1]
        raise RuntimeError("force lambda_format fallback")


class _FakePySRRegressorWithDegenerateCandidate:
    def __init__(self, **kwargs):
        self.equations_ = pd.DataFrame(
            [
                {
                    "equation": "x0 + x1",
                    "lambda_format": lambda X: X[:, 0] + X[:, 1],
                },
                {
                    "equation": "x0 - x0",
                    "lambda_format": lambda X: np.zeros(X.shape[0], dtype=float),
                },
            ]
        )

    def fit(self, X, y):
        return self

    def predict(self, X, index=0):
        if int(index) == 0:
            return X[:, 0] + X[:, 1]
        return np.zeros(X.shape[0], dtype=float)


class _FakePySRRegressorManyCandidates:
    evaluated_indices: set[int] = set()

    def __init__(self, **kwargs):
        _FakePySRRegressorManyCandidates.evaluated_indices = set()
        self.equations_ = pd.DataFrame(
            [
                {"equation": "x0 + x1", "loss": 0.50, "lambda_format": None},
                {"equation": "x0 - x1", "loss": 0.10, "lambda_format": None},
                {"equation": "x0", "loss": 0.20, "lambda_format": None},
                {"equation": "x1", "loss": 0.30, "lambda_format": None},
                {"equation": "x0 + x0", "loss": 0.40, "lambda_format": None},
            ]
        )

    def fit(self, X, y):
        return self

    def predict(self, X, index=0):
        idx = int(index)
        _FakePySRRegressorManyCandidates.evaluated_indices.add(idx)
        if idx == 0:
            return X[:, 0] + X[:, 1]
        if idx == 1:
            return X[:, 0] - X[:, 1]
        if idx == 2:
            return X[:, 0]
        if idx == 3:
            return X[:, 1]
        return X[:, 0] + X[:, 0]


def test_pysr_backend_uses_temp_equation_files(monkeypatch):
    monkeypatch.setattr(ensemble_module, "_load_pysr_regressor", lambda: _FakePySRRegressor)

    X_train = np.random.default_rng(0).normal(size=(40, 3))
    y_train = X_train[:, 0] + X_train[:, 1]
    X_val = np.random.default_rng(1).normal(size=(20, 3))
    y_val = X_val[:, 0] + X_val[:, 1]

    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
        FeatureMeta(column=2, variable=1, lag=1, name="x1_lag1"),
    ]

    cfg = EnsembleConfig()
    cfg.symbolic_backend = "pysr"
    cfg.pysr_population_size = 20
    cfg.pysr_tournament_selection_n = 10
    cfg.pysr_complexity_of_operators = {"tanh": 4.0}

    models, _ = ensemble_module._fit_pysr_seed_models(
        seed=42,
        X_fit=X_train,
        y_fit=y_train,
        feature_meta=feature_meta,
        config=cfg,
    )

    assert len(models) == 1
    assert _FakePySRRegressor.init_kwargs is not None
    assert _FakePySRRegressor.init_kwargs["temp_equation_file"] is True
    assert _FakePySRRegressor.init_kwargs["delete_tempfiles"] is True
    assert _FakePySRRegressor.init_kwargs["tempdir"] is not None
    assert _FakePySRRegressor.init_kwargs["niterations"] == cfg.pysr_niterations
    assert _FakePySRRegressor.init_kwargs["maxsize"] == cfg.pysr_maxsize
    assert _FakePySRRegressor.init_kwargs["should_optimize_constants"] is cfg.pysr_should_optimize_constants
    assert _FakePySRRegressor.init_kwargs["complexity_of_constants"] == cfg.pysr_complexity_of_constants
    assert _FakePySRRegressor.init_kwargs["complexity_of_operators"] == cfg.pysr_complexity_of_operators


def test_pysr_backend_skips_invalid_candidate_without_runtime_warning(monkeypatch):
    monkeypatch.setattr(
        ensemble_module,
        "_load_pysr_regressor",
        lambda: _FakePySRRegressorWithInvalidCandidate,
    )

    X_train = np.random.default_rng(3).normal(size=(40, 3))
    y_train = X_train[:, 0] + X_train[:, 1]
    X_val = np.random.default_rng(4).normal(size=(20, 3))
    y_val = X_val[:, 0] + X_val[:, 1]

    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
        FeatureMeta(column=2, variable=1, lag=1, name="x1_lag1"),
    ]

    cfg = EnsembleConfig()
    cfg.symbolic_backend = "pysr"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        models, _ = ensemble_module._fit_pysr_seed_models(
            seed=11,
            X_fit=X_train,
            y_fit=y_train,
            feature_meta=feature_meta,
            config=cfg,
        )

    assert len(models) == 1
    assert models[0].equation == "x0 + x1"
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_pysr_backend_skips_degenerate_constant_like_candidate(monkeypatch):
    monkeypatch.setattr(
        ensemble_module,
        "_load_pysr_regressor",
        lambda: _FakePySRRegressorWithDegenerateCandidate,
    )

    X_train = np.random.default_rng(9).normal(size=(40, 3))
    y_train = X_train[:, 0] + X_train[:, 1]
    X_val = np.random.default_rng(10).normal(size=(20, 3))
    y_val = X_val[:, 0] + X_val[:, 1]

    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
        FeatureMeta(column=2, variable=1, lag=1, name="x1_lag1"),
    ]

    cfg = EnsembleConfig()
    cfg.symbolic_backend = "pysr"
    cfg.min_prediction_std = 1e-6

    models, _ = ensemble_module._fit_pysr_seed_models(
        seed=13,
        X_fit=X_train,
        y_fit=y_train,
        feature_meta=feature_meta,
        config=cfg,
    )

    assert len(models) == 1
    assert models[0].equation == "x0 + x1"


def test_pysr_backend_caps_evaluated_equations_by_max_evals(monkeypatch):
    monkeypatch.setattr(
        ensemble_module,
        "_load_pysr_regressor",
        lambda: _FakePySRRegressorManyCandidates,
    )

    X_train = np.random.default_rng(12).normal(size=(30, 3))
    y_train = X_train[:, 0] + X_train[:, 1]
    X_val = np.random.default_rng(13).normal(size=(10, 3))
    y_val = X_val[:, 0] + X_val[:, 1]

    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
        FeatureMeta(column=2, variable=1, lag=1, name="x1_lag1"),
    ]

    cfg = EnsembleConfig()
    cfg.symbolic_backend = "pysr"
    cfg.max_evals_per_candidate = 2

    models, _ = ensemble_module._fit_pysr_seed_models(
        seed=21,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_meta=feature_meta,
        config=cfg,
    )

    assert len(models) <= 2
    # The two capped candidates are indices {1, 2}; index 0 is the extra
    # native train-selected winner predict (gate reference, model_selection="best"
    # => the fake's default index=0), which is scored on the tail but does not
    # count against max_evals_per_candidate.
    assert _FakePySRRegressorManyCandidates.evaluated_indices == {0, 1, 2}


def test_pysr_backend_rejects_non_positive_max_evals(monkeypatch):
    monkeypatch.setattr(ensemble_module, "_load_pysr_regressor", lambda: _FakePySRRegressor)

    X_train = np.random.default_rng(14).normal(size=(20, 3))
    y_train = X_train[:, 0] + X_train[:, 1]
    X_val = np.random.default_rng(15).normal(size=(8, 3))
    y_val = X_val[:, 0] + X_val[:, 1]

    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
        FeatureMeta(column=2, variable=1, lag=1, name="x1_lag1"),
    ]

    cfg = EnsembleConfig()
    cfg.symbolic_backend = "pysr"
    cfg.max_evals_per_candidate = 0

    with pytest.raises(ValueError, match="max_evals_per_candidate must be positive"):
        ensemble_module._fit_pysr_seed_models(
            seed=9,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_meta=feature_meta,
            config=cfg,
        )
