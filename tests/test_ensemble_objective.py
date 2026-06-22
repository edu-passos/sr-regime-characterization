import numpy as np
from types import SimpleNamespace

from symbolic_structural_profiling.config import EnsembleConfig, FeatureConfig
from symbolic_structural_profiling.models.ensemble import (
    SymbolicModel,
    _select_retained_frontier_models,
    _fit_random_seed_models,
    fit_symbolic_ensemble,
)
from symbolic_structural_profiling.models.expressions import ExprNode
from symbolic_structural_profiling.windowing import FeatureMeta


def _feature_meta_single() -> list[FeatureMeta]:
    return [FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1")]


def _var_expr() -> ExprNode:
    return ExprNode(op="var", feature_index=0, variable=0, lag=1)


def _feature_meta_pair() -> list[FeatureMeta]:
    return [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
    ]


def test_frontier_retains_only_non_dominated_models(monkeypatch):
    from symbolic_structural_profiling.models import ensemble as ensemble_module

    expr = _var_expr()

    def _fake_seed_models(**kwargs):
        seed = int(kwargs["seed"])
        return [
            SymbolicModel(expression=expr, seed=seed, fit_mse=0.55, complexity=1, equation="c1_bad"),
            SymbolicModel(expression=expr, seed=seed, fit_mse=0.40, complexity=1, equation="c1_good"),
            SymbolicModel(expression=expr, seed=seed, fit_mse=0.32, complexity=2, equation="c2_good"),
            SymbolicModel(expression=expr, seed=seed, fit_mse=0.34, complexity=3, equation="c3_dominated"),
            SymbolicModel(expression=expr, seed=seed, fit_mse=0.25, complexity=4, equation="c4_good"),
        ], float("nan")

    monkeypatch.setattr(ensemble_module, "_seed_models_for_backend", _fake_seed_models)

    cfg = EnsembleConfig()
    cfg.symbolic_backend = "random"
    cfg.n_seeds = 1
    cfg.top_k_per_seed = 3

    X_fit = np.ones((20, 1), dtype=float)
    y_fit = np.ones(20, dtype=float)

    result = fit_symbolic_ensemble(
        X_fit=X_fit,
        y_fit=y_fit,
        feature_meta=_feature_meta_single(),
        target_index=0,
        config=cfg,
    )

    assert [model.equation for model in result.retained_models] == [
        "c4_good",
        "c2_good",
        "c1_good",
    ]


def test_linear_symbolic_gap_is_descriptive_only(monkeypatch):
    from symbolic_structural_profiling.models import ensemble as ensemble_module

    class _FakeRidge:
        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=float)

    monkeypatch.setattr(ensemble_module, "Ridge", _FakeRidge)

    expr = _var_expr()

    def _fake_seed_models(**kwargs):
        return [
            SymbolicModel(
                expression=expr,
                seed=int(kwargs["seed"]),
                fit_mse=2.0,
                complexity=1,
                equation="x0_lag1",
            )
        ], float("nan")

    monkeypatch.setattr(ensemble_module, "_seed_models_for_backend", _fake_seed_models)

    X_fit = np.ones((16, 1), dtype=float)
    y_fit = np.zeros(16, dtype=float)

    cfg = EnsembleConfig()
    cfg.symbolic_backend = "random"
    cfg.n_seeds = 1
    cfg.top_k_per_seed = 1

    result = fit_symbolic_ensemble(
        X_fit=X_fit,
        y_fit=y_fit,
        feature_meta=_feature_meta_single(),
        target_index=0,
        config=cfg,
    )

    assert len(result.retained_models) == 1
    assert result.linear_fit_mse == 0.0
    assert result.best_symbolic_fit_mse == 2.0
    assert result.linear_symbolic_fit_gap == -2.0


def test_empty_search_returns_nan_symbolic_fit_metrics(monkeypatch):
    from symbolic_structural_profiling.models import ensemble as ensemble_module

    monkeypatch.setattr(ensemble_module, "_seed_models_for_backend", lambda **kwargs: ([], float("nan")))

    cfg = EnsembleConfig()
    cfg.symbolic_backend = "random"
    cfg.n_seeds = 1

    X_fit = np.ones((12, 1), dtype=float)
    y_fit = np.ones(12, dtype=float)

    result = fit_symbolic_ensemble(
        X_fit=X_fit,
        y_fit=y_fit,
        feature_meta=_feature_meta_single(),
        target_index=0,
        config=cfg,
    )

    assert result.retained_models == []
    assert np.isnan(result.best_symbolic_fit_mse)
    assert np.isnan(result.linear_symbolic_fit_gap)


def test_random_backend_templates_and_affine_refit_capture_lag_pairs():
    X_fit = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 64, dtype=float),
            np.linspace(1.0, -1.0, 64, dtype=float),
        ]
    )
    y_fit = 1.7 * (X_fit[:, 0] - X_fit[:, 1]) + 0.3

    cfg = EnsembleConfig()
    cfg.symbolic_backend = "random"
    cfg.n_candidates_per_seed = 0
    cfg.pysr_binary_operators = ("+", "-")
    cfg.pysr_unary_operators = ()

    models = _fit_random_seed_models(
        seed=0,
        X_fit=X_fit,
        y_fit=y_fit,
        feature_meta=_feature_meta_pair(),
        config=cfg,
    )

    pair_models = [
        model for model in models if "x0_lag1" in model.equation and "x0_lag2" in model.equation
    ]
    assert pair_models
    assert min(model.fit_mse for model in pair_models) < 1e-10


def test_validation_rebalanced_retention_keeps_simple_model_coverage():
    simple_expr = ExprNode(op="var", feature_index=0, variable=0, lag=1)
    unary_expr = ExprNode(op="tanh", children=(simple_expr,))
    interaction_expr = ExprNode(
        op="mul",
        children=(
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
            ExprNode(op="var", feature_index=1, variable=0, lag=2),
        ),
    )

    frontier = [
        SymbolicModel(expression=unary_expr, seed=0, fit_mse=0.100, complexity=3, equation="tanh(x0_lag1)"),
        SymbolicModel(expression=simple_expr, seed=0, fit_mse=0.105, complexity=1, equation="x0_lag1"),
        SymbolicModel(expression=interaction_expr, seed=0, fit_mse=0.120, complexity=4, equation="(x0_lag1 * x0_lag2)"),
        SymbolicModel(expression=unary_expr, seed=0, fit_mse=0.130, complexity=5, equation="tanh(tanh(x0_lag1))"),
    ]

    cfg = EnsembleConfig()
    cfg.top_k_per_seed = 4
    cfg.rebalance_retained_model_families = True
    cfg.retained_simple_model_fit_ratio = 1.10

    retained = _select_retained_frontier_models(frontier, cfg)

    assert [model.equation for model in retained[:3]] == [
        "tanh(x0_lag1)",
        "x0_lag1",
        "(x0_lag1 * x0_lag2)",
    ]


def test_validation_rebalanced_retention_skips_simple_slot_when_not_adequate():
    simple_expr = ExprNode(op="var", feature_index=0, variable=0, lag=1)
    unary_expr = ExprNode(op="tanh", children=(simple_expr,))
    interaction_expr = ExprNode(
        op="mul",
        children=(
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
            ExprNode(op="var", feature_index=1, variable=0, lag=2),
        ),
    )

    frontier = [
        SymbolicModel(expression=unary_expr, seed=0, fit_mse=0.100, complexity=3, equation="tanh(x0_lag1)"),
        SymbolicModel(expression=simple_expr, seed=0, fit_mse=0.140, complexity=1, equation="x0_lag1"),
        SymbolicModel(expression=interaction_expr, seed=0, fit_mse=0.115, complexity=4, equation="(x0_lag1 * x0_lag2)"),
    ]

    cfg = EnsembleConfig()
    cfg.top_k_per_seed = 4
    cfg.rebalance_retained_model_families = True
    cfg.retained_simple_model_fit_ratio = 1.10

    retained = _select_retained_frontier_models(frontier, cfg)

    assert [model.equation for model in retained] == [
        "tanh(x0_lag1)",
        "(x0_lag1 * x0_lag2)",
        "x0_lag1",
    ]


def test_rebalanced_retention_uses_supported_family_not_raw_family(monkeypatch):
    from symbolic_structural_profiling.models import ensemble as ensemble_module

    simple_expr = ExprNode(op="var", feature_index=0, variable=0, lag=1)
    unary_expr = ExprNode(op="tanh", children=(simple_expr,))
    interaction_expr = ExprNode(
        op="mul",
        children=(
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
            ExprNode(op="var", feature_index=1, variable=0, lag=2),
        ),
    )

    frontier = [
        SymbolicModel(expression=unary_expr, seed=0, fit_mse=0.100, complexity=3, equation="tanh(x0_lag1)"),
        SymbolicModel(expression=simple_expr, seed=0, fit_mse=0.105, complexity=1, equation="x0_lag1"),
        SymbolicModel(expression=interaction_expr, seed=0, fit_mse=0.120, complexity=4, equation="(x0_lag1 * x0_lag2)"),
        SymbolicModel(expression=ExprNode(op="tanh", children=(unary_expr,)), seed=0, fit_mse=0.130, complexity=5, equation="tanh(tanh(x0_lag1))"),
    ]

    family_by_equation = {
        "tanh(x0_lag1)": "simple",
        "x0_lag1": "simple",
        "(x0_lag1 * x0_lag2)": "interaction",
        "tanh(tanh(x0_lag1))": "simple",
    }

    def _fake_supported_structure(expr, **kwargs):
        return SimpleNamespace(family=family_by_equation[expr.to_string()])

    monkeypatch.setattr(ensemble_module, "supported_structure", _fake_supported_structure)

    cfg = EnsembleConfig()
    cfg.top_k_per_seed = 4
    cfg.rebalance_retained_model_families = True
    cfg.retained_simple_model_fit_ratio = 1.10

    retained = _select_retained_frontier_models(
        frontier,
        cfg,
        X_fit=np.ones((8, 2), dtype=float),
        y_fit=np.ones(8, dtype=float),
        linear_fit_mse=1.0,
        feature_config=FeatureConfig(),
    )

    assert [model.equation for model in retained[:3]] == [
        "tanh(x0_lag1)",
        "(x0_lag1 * x0_lag2)",
        "x0_lag1",
    ]
