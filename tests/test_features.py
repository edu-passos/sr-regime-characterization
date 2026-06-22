import numpy as np
import pandas as pd
import warnings
import pytest

from symbolic_structural_profiling.config import FeatureConfig
from symbolic_structural_profiling.features import (
    build_family_transition_diagnostics,
    expression_feature_table,
    extract_expression_features,
    summarize_expression_features,
)
from symbolic_structural_profiling.models.ensemble import SymbolicModel
from symbolic_structural_profiling.models.expressions import (
    ExprNode,
    affine_calibrated_mse,
    safe_eval,
    strip_affine_envelope,
    supported_structure,
)
from symbolic_structural_profiling.windowing import FeatureMeta


def _make_model(expr: ExprNode) -> SymbolicModel:
    return SymbolicModel(
        expression=expr,
        seed=0,
        fit_mse=0.0,
        complexity=1,
        equation=expr.to_string(),
    )


def test_extract_expression_features_coupled_interaction_is_not_counted_as_pure_unary_nonlinearity():
    X = np.random.default_rng(0).normal(size=(80, 4))
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
        FeatureMeta(column=2, variable=1, lag=1, name="x1_lag1"),
        FeatureMeta(column=3, variable=1, lag=2, name="x1_lag2"),
    ]

    expr = ExprNode(
        op="sin",
        children=(
            ExprNode(
                op="mul",
                children=(
                    ExprNode(op="var", feature_index=0, variable=0, lag=1),
                    ExprNode(op="var", feature_index=2, variable=1, lag=1),
                ),
            ),
        ),
    )

    model = SymbolicModel(
        expression=expr,
        seed=0,
        fit_mse=0.0,
        complexity=3,
        equation=expr.to_string(),
    )
    feats = extract_expression_features(model, X, feature_meta, FeatureConfig())

    assert feats.nonlinear_node_count == pytest.approx(0.0)
    assert feats.interaction_term_count >= 1
    assert feats.interaction_order >= 2
    assert feats.lag_mass > 0


def test_extract_expression_features_linear_single_lag_matches_definition():
    X = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 128, dtype=float),
            np.linspace(1.0, -1.0, 128, dtype=float),
        ]
    )
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
    ]
    expr = ExprNode(
        op="mul",
        children=(
            ExprNode(op="const", value=0.8),
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
        ),
    )

    feats = extract_expression_features(_make_model(expr), X, feature_meta, FeatureConfig())

    assert feats.lag_mass == pytest.approx(1.0, abs=1e-3)
    assert feats.long_lag_mass == pytest.approx(0.0)
    assert feats.active_lag_count == pytest.approx(1.0)
    assert feats.nonlinear_node_count == pytest.approx(0.0)


def test_strip_affine_envelope_removes_scale_shift_and_sign_wrappers():
    base = ExprNode(op="var", feature_index=0, variable=0, lag=2)
    expr = ExprNode(
        op="sub",
        children=(
            ExprNode(op="const", value=0.3),
            ExprNode(
                op="mul",
                children=(
                    ExprNode(op="const", value=-1.7),
                    base,
                ),
            ),
        ),
    )

    stripped = strip_affine_envelope(expr)

    assert stripped.op == "var"
    assert stripped.variable == 0
    assert stripped.lag == 2


def test_strip_affine_envelope_keeps_nonlinear_wrappers():
    base = ExprNode(op="var", feature_index=0, variable=0, lag=1)
    expr = ExprNode(
        op="tanh",
        children=(
            ExprNode(
                op="add",
                children=(
                    ExprNode(
                        op="mul",
                        children=(ExprNode(op="const", value=2.0), base),
                    ),
                    ExprNode(op="const", value=1.0),
                ),
            ),
        ),
    )

    stripped = strip_affine_envelope(expr)

    assert stripped is expr


def test_extract_expression_features_are_affine_invariant():
    X = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 128, dtype=float),
            np.linspace(1.0, -1.0, 128, dtype=float),
        ]
    )
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
    ]
    base = ExprNode(op="var", feature_index=1, variable=0, lag=2)
    affine = ExprNode(
        op="add",
        children=(
            ExprNode(
                op="mul",
                children=(ExprNode(op="const", value=2.5), base),
            ),
            ExprNode(op="const", value=-0.8),
        ),
    )

    base_feats = extract_expression_features(_make_model(base), X, feature_meta, FeatureConfig())
    affine_feats = extract_expression_features(_make_model(affine), X, feature_meta, FeatureConfig())

    assert affine_feats.lag_mass == pytest.approx(base_feats.lag_mass, abs=1e-3)
    assert affine_feats.active_lag_count == pytest.approx(base_feats.active_lag_count)
    assert affine_feats.nonlinear_node_count == pytest.approx(base_feats.nonlinear_node_count)
    assert affine_feats.complexity_cost == pytest.approx(base_feats.complexity_cost)


def test_extract_expression_features_interaction_count_matches_product_definition():
    X = np.random.default_rng(4).normal(size=(96, 2))
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
    ]
    expr = ExprNode(
        op="mul",
        children=(
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
            ExprNode(op="var", feature_index=1, variable=0, lag=2),
        ),
    )

    feats = extract_expression_features(_make_model(expr), X, feature_meta, FeatureConfig())

    assert feats.interaction_term_count >= 1.0
    assert feats.interaction_order == pytest.approx(2.0)


def test_extract_expression_features_self_product_counts_as_nonlinear_not_interaction():
    X = np.random.default_rng(14).normal(size=(96, 1))
    feature_meta = [FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1")]
    expr = ExprNode(
        op="mul",
        children=(
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
        ),
    )

    feats = extract_expression_features(_make_model(expr), X, feature_meta, FeatureConfig())

    assert feats.nonlinear_node_count >= 1.0
    assert feats.interaction_term_count == pytest.approx(0.0)
    assert feats.interaction_order == pytest.approx(1.0)


def test_extract_expression_features_cross_lag_product_counts_as_interaction_not_nonlinearity():
    X = np.random.default_rng(15).normal(size=(96, 2))
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
    ]
    expr = ExprNode(
        op="mul",
        children=(
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
            ExprNode(op="var", feature_index=1, variable=0, lag=2),
        ),
    )

    feats = extract_expression_features(_make_model(expr), X, feature_meta, FeatureConfig())

    assert feats.nonlinear_node_count == pytest.approx(0.0)
    assert feats.interaction_term_count >= 1.0
    assert feats.interaction_order == pytest.approx(2.0)


def test_extract_expression_features_unary_nonlinearity_is_not_interaction():
    X = np.random.default_rng(16).normal(size=(96, 1))
    feature_meta = [FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1")]
    expr = ExprNode(
        op="tanh",
        children=(ExprNode(op="var", feature_index=0, variable=0, lag=1),),
    )

    feats = extract_expression_features(_make_model(expr), X, feature_meta, FeatureConfig())

    assert feats.nonlinear_node_count >= 1.0
    assert feats.interaction_term_count == pytest.approx(0.0)


def test_extract_expression_features_coupled_unary_carrier_counts_as_interaction_not_unary():
    X = np.random.default_rng(17).normal(size=(96, 2))
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
    ]
    expr = ExprNode(
        op="tanh",
        children=(
            ExprNode(
                op="add",
                children=(
                    ExprNode(op="var", feature_index=0, variable=0, lag=1),
                    ExprNode(op="var", feature_index=1, variable=0, lag=2),
                ),
            ),
        ),
    )

    feats = extract_expression_features(_make_model(expr), X, feature_meta, FeatureConfig())

    assert feats.nonlinear_node_count == pytest.approx(0.0)
    assert feats.interaction_term_count == pytest.approx(1.0)
    assert feats.interaction_order == pytest.approx(2.0)


def test_extract_expression_features_nested_coupled_carrier_counts_only_once():
    X = np.random.default_rng(18).normal(size=(96, 2))
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
    ]
    expr = ExprNode(
        op="tanh",
        children=(
            ExprNode(
                op="tanh",
                children=(
                    ExprNode(
                        op="sub",
                        children=(
                            ExprNode(op="var", feature_index=0, variable=0, lag=1),
                            ExprNode(op="var", feature_index=1, variable=0, lag=2),
                        ),
                    ),
                ),
            ),
        ),
    )

    feats = extract_expression_features(_make_model(expr), X, feature_meta, FeatureConfig())

    assert feats.interaction_term_count == pytest.approx(1.0)
    assert feats.nonlinear_node_count == pytest.approx(0.0)


def test_extract_expression_features_nonlinear_depth_matches_nested_unary_expression():
    X = np.random.default_rng(9).normal(size=(100, 1))
    feature_meta = [FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1")]
    expr = ExprNode(
        op="tanh",
        children=(
            ExprNode(
                op="sin",
                children=(ExprNode(op="var", feature_index=0, variable=0, lag=1),),
            ),
        ),
    )

    feats = extract_expression_features(_make_model(expr), X, feature_meta, FeatureConfig())

    assert feats.nonlinear_node_count >= 2.0
    assert feats.nonlinear_composition_depth >= 2.0


def test_expression_feature_table_marks_weak_unary_evidence_as_unsupported():
    X = np.linspace(-0.2, 0.2, 200, dtype=float)[:, None]
    y = X[:, 0].copy()
    feature_meta = [FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1")]
    expr = ExprNode(
        op="tanh",
        children=(ExprNode(op="var", feature_index=0, variable=0, lag=1),),
    )
    cfg = FeatureConfig(
        counterfactual_structure_evidence=True,
        unary_structure_evidence_min_ratio=0.05,
        unary_structure_evidence_min_abs=1e-4,
    )

    table = expression_feature_table(
        models=[_make_model(expr)],
        X_reference=X,
        feature_meta=feature_meta,
        config=cfg,
        y_reference=y,
        linear_fit_mse=1.0,
    )
    row = table.iloc[0]

    assert row["raw_has_unary_nonlinearity"] == pytest.approx(1.0)
    assert row["has_unary_nonlinearity"] == pytest.approx(0.0)
    assert row["has_supported_unary_nonlinearity"] == pytest.approx(0.0)
    assert row["raw_is_simple_model"] == pytest.approx(0.0)
    assert row["is_simple_model"] == pytest.approx(1.0)
    assert row["single_lag_unary_surrogate_fit_mse"] < float("inf")
    assert row["simple_surrogate_fit_mse"] < float("inf")
    assert row["unary_curvature_gain"] == pytest.approx(row["unary_evidence_gain"])
    assert row["unary_evidence_gain"] < 0.05
    assert row["nonlinear_node_count"] == pytest.approx(0.0)
    assert row["nonlinear_composition_depth"] == pytest.approx(0.0)


def test_expression_feature_table_keeps_strong_unary_evidence_supported():
    X = np.linspace(-3.0, 3.0, 200, dtype=float)[:, None]
    y = np.tanh(X[:, 0])
    feature_meta = [FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1")]
    expr = ExprNode(
        op="tanh",
        children=(ExprNode(op="var", feature_index=0, variable=0, lag=1),),
    )
    cfg = FeatureConfig(
        counterfactual_structure_evidence=True,
        unary_structure_evidence_min_ratio=0.05,
        unary_structure_evidence_min_abs=1e-4,
    )
    linear_fit_mse = affine_calibrated_mse(
        ExprNode(op="var", feature_index=0, variable=0, lag=1),
        X,
        y,
    )

    table = expression_feature_table(
        models=[_make_model(expr)],
        X_reference=X,
        feature_meta=feature_meta,
        config=cfg,
        y_reference=y,
        linear_fit_mse=linear_fit_mse,
    )
    row = table.iloc[0]

    assert row["raw_has_unary_nonlinearity"] == pytest.approx(1.0)
    assert row["has_unary_nonlinearity"] == pytest.approx(1.0)
    assert row["has_supported_unary_nonlinearity"] == pytest.approx(1.0)
    assert row["raw_is_simple_model"] == pytest.approx(0.0)
    assert row["is_simple_model"] == pytest.approx(0.0)
    assert row["unary_curvature_gain"] == pytest.approx(row["unary_evidence_gain"])
    assert row["unary_evidence_gain"] >= max(linear_fit_mse * 0.05, 1e-4)
    assert row["nonlinear_node_count"] >= 1.0
    assert row["nonlinear_composition_depth"] >= 1.0


def test_expression_feature_table_marks_weak_interaction_evidence_as_unsupported():
    X = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 200, dtype=float),
            np.ones(200, dtype=float),
        ]
    )
    y = X[:, 0].copy()
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
    ]
    expr = ExprNode(
        op="mul",
        children=(
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
            ExprNode(op="var", feature_index=1, variable=0, lag=2),
        ),
    )
    cfg = FeatureConfig(
        counterfactual_structure_evidence=True,
        interaction_structure_evidence_min_ratio=0.05,
        interaction_structure_evidence_min_abs=1e-4,
    )

    table = expression_feature_table(
        models=[_make_model(expr)],
        X_reference=X,
        feature_meta=feature_meta,
        config=cfg,
        y_reference=y,
        linear_fit_mse=1.0,
    )
    row = table.iloc[0]

    assert row["raw_has_interaction"] == pytest.approx(1.0)
    assert row["has_interaction"] == pytest.approx(0.0)
    assert row["has_supported_interaction"] == pytest.approx(0.0)
    assert row["raw_is_simple_model"] == pytest.approx(0.0)
    assert row["is_simple_model"] == pytest.approx(1.0)
    assert row["additive_noninteraction_surrogate_fit_mse"] < float("inf")
    assert row["interaction_synergy_gain"] == pytest.approx(row["interaction_evidence_gain"])
    assert row["interaction_evidence_gain"] < 0.05
    assert row["interaction_term_count"] == pytest.approx(0.0)
    assert row["interaction_order"] == pytest.approx(1.0)


def test_expression_feature_table_keeps_strong_interaction_evidence_supported():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(256, 2))
    y = X[:, 0] * X[:, 1]
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
    ]
    expr = ExprNode(
        op="mul",
        children=(
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
            ExprNode(op="var", feature_index=1, variable=0, lag=2),
        ),
    )
    cfg = FeatureConfig(
        counterfactual_structure_evidence=True,
        interaction_structure_evidence_min_ratio=0.05,
        interaction_structure_evidence_min_abs=1e-4,
    )

    table = expression_feature_table(
        models=[_make_model(expr)],
        X_reference=X,
        feature_meta=feature_meta,
        config=cfg,
        y_reference=y,
        linear_fit_mse=1.0,
    )
    row = table.iloc[0]

    assert row["raw_has_interaction"] == pytest.approx(1.0)
    assert row["has_interaction"] == pytest.approx(1.0)
    assert row["has_supported_interaction"] == pytest.approx(1.0)
    assert row["raw_is_simple_model"] == pytest.approx(0.0)
    assert row["is_simple_model"] == pytest.approx(0.0)
    assert row["interaction_synergy_gain"] == pytest.approx(row["interaction_evidence_gain"])
    assert row["interaction_evidence_gain"] >= 0.05
    assert row["interaction_term_count"] >= 1.0
    assert row["interaction_order"] == pytest.approx(2.0)


def test_supported_structure_prunes_coupled_interaction_to_best_single_lag_surrogate():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(256, 2))
    y = np.tanh(X[:, 0])
    expr = ExprNode(
        op="tanh",
        children=(
            ExprNode(
                op="add",
                children=(
                    ExprNode(op="var", feature_index=0, variable=0, lag=1),
                    ExprNode(op="var", feature_index=1, variable=0, lag=2),
                ),
            ),
        ),
    )
    linear_fit_mse = affine_calibrated_mse(
        ExprNode(op="var", feature_index=0, variable=0, lag=1),
        X,
        y,
    )

    support = supported_structure(
        expr,
        X=X,
        y=y,
        linear_fit_mse=linear_fit_mse,
        enabled=True,
        unary_min_ratio=0.05,
        unary_min_abs=1e-4,
        interaction_min_ratio=0.05,
        interaction_min_abs=1e-4,
    )

    assert support.raw_has_interaction is True
    assert support.raw_has_unary_nonlinearity is False
    assert support.has_interaction is False
    assert support.has_unary_nonlinearity is True
    assert support.unary_curvature_gain > support.interaction_synergy_gain
    assert support.supported_expr.to_string() == "tanh(x0_lag1)"


def test_expression_feature_table_keeps_strong_coupled_interaction_supported():
    rng = np.random.default_rng(32)
    X = rng.normal(size=(256, 2))
    y = np.tanh(X[:, 0] + X[:, 1])
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
    ]
    expr = ExprNode(
        op="tanh",
        children=(
            ExprNode(
                op="add",
                children=(
                    ExprNode(op="var", feature_index=0, variable=0, lag=1),
                    ExprNode(op="var", feature_index=1, variable=0, lag=2),
                ),
            ),
        ),
    )

    table = expression_feature_table(
        models=[_make_model(expr)],
        X_reference=X,
        feature_meta=feature_meta,
        config=FeatureConfig(
            counterfactual_structure_evidence=True,
            unary_structure_evidence_min_ratio=0.05,
            unary_structure_evidence_min_abs=1e-4,
            interaction_structure_evidence_min_ratio=0.05,
            interaction_structure_evidence_min_abs=1e-4,
        ),
        y_reference=y,
        linear_fit_mse=1.0,
    )
    row = table.iloc[0]

    assert row["raw_has_interaction"] == pytest.approx(1.0)
    assert row["raw_has_coupled_interaction"] == pytest.approx(1.0)
    assert row["raw_has_unary_nonlinearity"] == pytest.approx(0.0)
    assert row["has_interaction"] == pytest.approx(1.0)
    assert row["has_coupled_interaction"] == pytest.approx(1.0)
    assert row["has_unary_nonlinearity"] == pytest.approx(0.0)
    assert row["interaction_synergy_gain"] == pytest.approx(row["interaction_evidence_gain"])
    assert row["interaction_synergy_gain"] > row["unary_curvature_gain"]
    assert row["interaction_term_count"] == pytest.approx(1.0)


def test_extract_expression_features_memory_mass_tracks_lag_depth():
    X = np.random.default_rng(1).normal(size=(100, 3))
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
        FeatureMeta(column=2, variable=0, lag=3, name="x0_lag3"),
    ]

    expr = ExprNode(op="var", feature_index=2, variable=0, lag=3)
    model = SymbolicModel(
        expression=expr,
        seed=0,
        fit_mse=0.0,
        complexity=1,
        equation=expr.to_string(),
    )
    feats = extract_expression_features(model, X, feature_meta, FeatureConfig())

    assert feats.lag_mass >= 2.5
    assert feats.long_lag_mass > 0.0
    assert feats.active_lag_count >= 1.0


def test_extract_expression_features_long_lag_mass_ignores_short_lags():
    X = np.random.default_rng(2).normal(size=(100, 3))
    feature_meta = [
        FeatureMeta(column=0, variable=0, lag=1, name="x0_lag1"),
        FeatureMeta(column=1, variable=0, lag=2, name="x0_lag2"),
        FeatureMeta(column=2, variable=0, lag=4, name="x0_lag4"),
    ]
    expr = ExprNode(
        op="add",
        children=(
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
            ExprNode(op="var", feature_index=1, variable=0, lag=2),
        ),
    )

    feats = extract_expression_features(_make_model(expr), X, feature_meta, FeatureConfig())

    assert feats.long_lag_mass == pytest.approx(0.0)


def test_safe_eval_division_guard_avoids_divide_by_zero_warning():
    expr = ExprNode(
        op="div",
        children=(
            ExprNode(op="const", value=1.0),
            ExprNode(op="var", feature_index=0, variable=0, lag=1),
        ),
    )
    X = np.array([[-1e-12], [0.0], [1e-12]], dtype=float)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = safe_eval(expr, X)

    assert np.all(np.isfinite(out))
    assert not any("divide by zero encountered in divide" in str(w.message) for w in caught)


def test_summarize_expression_features_uses_fit_aware_weighted_medians():
    table = pd.DataFrame(
        {
            "lag_mass": [1.0, 5.0],
            "fit_mse": [0.010, 0.014],
            "expression_complexity": [1, 5],
            "model_idx": [0, 1],
            "seed": [0, 0],
            "equation": ["x0_lag1", "sin(x0_lag1)"],
            "max_lag_used": [1.0, 1.0],
            "has_nonlinearity": [0.0, 1.0],
            "has_interaction": [0.0, 0.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.50,
        fit_band_ratio=1.50,
    )

    assert summary["lag_mass_median"] == pytest.approx(1.0)
    assert summary["summary_model_count"] == pytest.approx(2.0)


def test_summarize_expression_features_zeroes_nonlinearity_without_linear_advantage():
    table = pd.DataFrame(
        {
            "lag_mass": [1.0, 5.0],
            "nonlinear_node_count": [0.0, 3.0],
            "nonlinear_composition_depth": [0.0, 2.0],
            "interaction_term_count": [0.0, 1.0],
            "interaction_order": [1.0, 2.0],
            "interaction_depth": [0.0, 3.0],
            "complexity_cost": [1.0, 8.0],
            "fit_mse": [0.100, 0.100],
            "expression_complexity": [1, 5],
            "model_idx": [0, 1],
            "seed": [0, 0],
            "equation": ["x0_lag1", "sin(x0_lag1)"],
            "max_lag_used": [1.0, 1.0],
            "has_nonlinearity": [0.0, 1.0],
            "has_unary_nonlinearity": [0.0, 1.0],
            "has_interaction": [0.0, 1.0],
            "unary_evidence_gain": [0.0, 0.0],
            "interaction_evidence_gain": [0.0, 0.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.100,
        fit_band_ratio=1.50,
    )

    assert summary["lag_mass_median"] == pytest.approx(1.0)
    assert summary["nonlinear_node_count_median"] == pytest.approx(0.0)
    assert summary["nonlinear_composition_depth_median"] == pytest.approx(0.0)
    assert summary["interaction_term_count_median"] == pytest.approx(0.0)
    assert summary["interaction_order_median"] == pytest.approx(0.0)


def test_summarize_expression_features_emits_selected_model_unary_and_interaction_diagnostics():
    table = pd.DataFrame(
        {
            "lag_mass": [1.0, 2.0],
            "long_lag_mass": [0.0, 0.5],
            "lag_entropy": [0.1, 0.2],
            "active_lag_count": [1.0, 2.0],
            "max_poly_degree_weighted": [1.0, 0.0],
            "nonlinear_node_count": [1.0, 0.0],
            "nonlinear_composition_depth": [1.0, 0.0],
            "interaction_term_count": [0.0, 1.0],
            "interaction_order": [0.0, 2.0],
            "interaction_depth": [0.0, 2.0],
            "complexity_cost": [2.0, 3.0],
            "description_length": [2.5, 3.5],
            "fit_mse": [0.10, 0.11],
            "expression_complexity": [2, 3],
            "model_idx": [0, 1],
            "seed": [0, 0],
            "equation": ["tanh(x0_lag1)", "(x0_lag1 * x0_lag2)"],
            "max_lag_used": [1.0, 2.0],
            "unary_evidence_gain": [0.04, 0.0],
            "interaction_evidence_gain": [0.0, 0.001],
            "raw_has_nonlinearity": [1.0, 1.0],
            "raw_has_unary_nonlinearity": [1.0, 0.0],
            "raw_has_interaction": [0.0, 1.0],
            "raw_is_simple_model": [0.0, 0.0],
            "has_nonlinearity": [1.0, 0.0],
            "has_unary_nonlinearity": [1.0, 0.0],
            "has_interaction": [0.0, 1.0],
            "is_simple_model": [0.0, 0.0],
            "has_supported_unary_nonlinearity": [1.0, 0.0],
            "has_supported_interaction": [0.0, 1.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.20,
        fit_band_ratio=1.20,
    )

    assert summary["summary_model_count"] == pytest.approx(2.0)
    assert summary["summary_selection_mode"] == "concept_pools_separability"
    assert summary["adequate_simple_model_count"] == pytest.approx(0.0)
    assert summary["lag_pool_model_count"] == pytest.approx(2.0)
    assert summary["nonlinear_candidate_count"] == pytest.approx(1.0)
    assert summary["interaction_candidate_count"] == pytest.approx(1.0)
    assert summary["nonlinear_pool_admitted"] == pytest.approx(1.0)
    assert summary["interaction_pool_admitted"] == pytest.approx(0.0)
    assert summary["nonlinear_pool_model_count"] == pytest.approx(1.0)
    assert summary["interaction_pool_model_count"] == pytest.approx(0.0)
    assert summary["best_selected_unary_curvature_gain"] == pytest.approx(0.04)
    assert summary["best_selected_interaction_synergy_gain"] == pytest.approx(0.001)
    assert summary["unary_curvature_threshold"] == pytest.approx(0.01)
    assert summary["interaction_synergy_threshold"] == pytest.approx(0.01)
    assert summary["fraction_adequate_models_supported_unary"] == pytest.approx(0.5)
    assert summary["fraction_adequate_models_supported_interaction"] == pytest.approx(0.5)
    assert summary["fraction_nonlinear_pool_models_raw_unary"] == pytest.approx(1.0)
    assert summary["fraction_interaction_pool_models_raw_interaction"] == pytest.approx(0.0)
    assert summary["fraction_summary_models_simple"] == pytest.approx(0.0)
    assert summary["fraction_summary_models_unary_nonlinear"] == pytest.approx(0.5)
    assert summary["fraction_summary_models_interaction"] == pytest.approx(0.5)
    assert summary["fraction_summary_models_raw_simple"] == pytest.approx(0.0)
    assert summary["fraction_summary_models_raw_unary_nonlinear"] == pytest.approx(0.5)
    assert summary["fraction_summary_models_raw_interaction"] == pytest.approx(0.5)
    assert summary["fraction_summary_models_coupled_interaction"] == pytest.approx(0.0)
    assert summary["fraction_summary_models_supported_unary"] == pytest.approx(1.0)
    assert summary["fraction_summary_models_supported_interaction"] == pytest.approx(0.0)
    assert summary["nonlinear_node_count_median"] == pytest.approx(1.0)
    assert summary["interaction_term_count_median"] == pytest.approx(0.0)


def test_build_family_transition_diagnostics_marks_unary_as_removed_at_fit_band():
    table = pd.DataFrame(
        {
            "lag_mass": [1.0, 3.0, 2.0],
            "long_lag_mass": [0.0, 0.3, 0.2],
            "lag_entropy": [0.1, 0.2, 0.2],
            "active_lag_count": [1.0, 2.0, 2.0],
            "max_poly_degree_weighted": [0.0, 2.0, 0.0],
            "nonlinear_node_count": [0.0, 2.0, 0.0],
            "nonlinear_composition_depth": [0.0, 2.0, 0.0],
            "interaction_term_count": [0.0, 0.0, 1.0],
            "interaction_order": [0.0, 0.0, 2.0],
            "interaction_depth": [0.0, 0.0, 2.0],
            "complexity_cost": [1.0, 3.0, 2.0],
            "description_length": [1.2, 3.2, 2.4],
            "fit_mse": [0.100, 0.130, 0.101],
            "expression_complexity": [1, 3, 2],
            "model_idx": [0, 1, 2],
            "seed": [0, 0, 0],
            "equation": ["x0_lag1", "tanh(x0_lag1)", "(x0_lag1 * x0_lag2)"],
            "max_lag_used": [1.0, 1.0, 2.0],
            "unary_evidence_gain": [0.0, 0.02, 0.0],
            "interaction_evidence_gain": [0.0, 0.0, 0.03],
            "has_nonlinearity": [0.0, 1.0, 1.0],
            "has_unary_nonlinearity": [0.0, 1.0, 0.0],
            "has_interaction": [0.0, 0.0, 1.0],
            "is_simple_model": [1.0, 0.0, 0.0],
        }
    )

    diagnostics = build_family_transition_diagnostics(
        table,
        linear_fit_mse=0.20,
        fit_band_ratio=1.05,
    )

    assert diagnostics["unary_present_retained"] == pytest.approx(1.0)
    assert diagnostics["unary_present_fit_band"] == pytest.approx(0.0)
    assert diagnostics["unary_present_post_guard"] == pytest.approx(0.0)
    assert diagnostics["fit_band_removed_unary"] == pytest.approx(1.0)
    assert diagnostics["complex_guard_removed_unary"] == pytest.approx(0.0)
    assert diagnostics["family_trace_retained_unary_count"] == pytest.approx(1.0)
    assert diagnostics["family_trace_fit_band_unary_count"] == pytest.approx(0.0)
    assert diagnostics["current_logic_decision_code"] == "interaction_only"


def test_build_family_transition_diagnostics_marks_unary_as_removed_at_guard():
    table = pd.DataFrame(
        {
            "lag_mass": [1.0, 3.0],
            "long_lag_mass": [0.0, 0.3],
            "lag_entropy": [0.1, 0.2],
            "active_lag_count": [1.0, 2.0],
            "max_poly_degree_weighted": [0.0, 2.0],
            "nonlinear_node_count": [0.0, 2.0],
            "nonlinear_composition_depth": [0.0, 2.0],
            "interaction_term_count": [0.0, 0.0],
            "interaction_order": [0.0, 0.0],
            "interaction_depth": [0.0, 0.0],
            "complexity_cost": [1.0, 3.0],
            "description_length": [1.2, 3.2],
            "fit_mse": [0.1000, 0.09995],
            "expression_complexity": [1, 3],
            "model_idx": [0, 1],
            "seed": [0, 0],
            "equation": ["x0_lag1", "tanh(x0_lag1)"],
            "max_lag_used": [1.0, 1.0],
            "unary_evidence_gain": [0.0, 0.02],
            "interaction_evidence_gain": [0.0, 0.0],
            "has_nonlinearity": [0.0, 1.0],
            "has_unary_nonlinearity": [0.0, 1.0],
            "has_interaction": [0.0, 0.0],
            "is_simple_model": [1.0, 0.0],
        }
    )

    diagnostics = build_family_transition_diagnostics(
        table,
        linear_fit_mse=0.100,
        fit_band_ratio=1.10,
        guard_complex_models=True,
        linear_advantage_min_ratio=0.05,
        linear_advantage_min_abs=1e-4,
    )

    assert diagnostics["unary_present_retained"] == pytest.approx(1.0)
    assert diagnostics["unary_present_fit_band"] == pytest.approx(1.0)
    assert diagnostics["unary_present_post_guard"] == pytest.approx(0.0)
    assert diagnostics["fit_band_removed_unary"] == pytest.approx(0.0)
    assert diagnostics["complex_guard_removed_unary"] == pytest.approx(1.0)
    assert diagnostics["current_logic_decision_code"] == "simple_only"


def test_build_family_transition_diagnostics_family_separated_unary_survives_post_guard_when_current_logic_zeroes_pool():
    table = pd.DataFrame(
        {
            "lag_mass": [1.0, 2.5],
            "long_lag_mass": [0.0, 0.3],
            "lag_entropy": [0.1, 0.2],
            "active_lag_count": [1.0, 2.0],
            "max_poly_degree_weighted": [0.0, 2.0],
            "nonlinear_node_count": [0.0, 2.0],
            "nonlinear_composition_depth": [0.0, 2.0],
            "interaction_term_count": [0.0, 0.0],
            "interaction_order": [0.0, 0.0],
            "interaction_depth": [0.0, 0.0],
            "complexity_cost": [1.0, 3.0],
            "description_length": [1.2, 3.2],
            "fit_mse": [0.100, 0.101],
            "expression_complexity": [1, 3],
            "model_idx": [0, 1],
            "seed": [0, 0],
            "equation": ["x0_lag1", "tanh(x0_lag1)"],
            "max_lag_used": [1.0, 1.0],
            "unary_evidence_gain": [0.0, 0.005],
            "interaction_evidence_gain": [0.0, 0.0],
            "has_nonlinearity": [0.0, 1.0],
            "has_unary_nonlinearity": [0.0, 1.0],
            "has_interaction": [0.0, 0.0],
            "is_simple_model": [1.0, 0.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.20,
        fit_band_ratio=1.05,
    )
    diagnostics = build_family_transition_diagnostics(
        table,
        linear_fit_mse=0.20,
        fit_band_ratio=1.05,
    )

    assert summary["nonlinear_pool_model_count"] == pytest.approx(0.0)
    assert diagnostics["unary_present_post_guard"] == pytest.approx(1.0)
    assert diagnostics["current_logic_decision_code"] == "simple_only"
    assert diagnostics["family_sep_simple_model_count"] == pytest.approx(1.0)
    assert diagnostics["family_sep_unary_model_count"] == pytest.approx(1.0)
    assert diagnostics["family_sep_unary_best_fit_mse"] == pytest.approx(0.101)
    assert diagnostics["family_sep_unary_nonlinear_node_count_median"] > 1.0
    assert diagnostics["family_sep_unary_nonlinear_composition_depth_median"] > 1.0
    assert diagnostics["delta_fit_simple_vs_unary"] == pytest.approx(-0.001)


def test_summarize_expression_features_admits_selected_nonlinear_pool_when_share_threshold_is_met():
    table = pd.DataFrame(
        {
            "lag_mass": [1.0, 4.0, 3.5],
            "long_lag_mass": [0.0, 0.5, 0.4],
            "lag_entropy": [0.1, 0.3, 0.25],
            "active_lag_count": [1.0, 2.0, 2.0],
            "max_poly_degree_weighted": [0.0, 2.0, 1.5],
            "nonlinear_node_count": [0.0, 2.0, 1.0],
            "nonlinear_composition_depth": [0.0, 2.0, 1.0],
            "interaction_term_count": [0.0, 0.0, 0.0],
            "interaction_order": [0.0, 0.0, 0.0],
            "interaction_depth": [0.0, 0.0, 0.0],
            "complexity_cost": [1.0, 4.0, 3.5],
            "description_length": [1.5, 4.5, 4.0],
            "fit_mse": [0.100, 0.090, 0.103],
            "expression_complexity": [1, 3, 2],
            "model_idx": [0, 1, 2],
            "seed": [0, 0, 0],
            "equation": ["x0_lag1", "tanh(x0_lag1)", "tanh(x0_lag2)"],
            "max_lag_used": [1.0, 1.0, 2.0],
            "unary_evidence_gain": [0.0, 0.04, 0.03],
            "interaction_evidence_gain": [0.0, 0.0, 0.0],
            "raw_has_nonlinearity": [0.0, 1.0, 1.0],
            "raw_has_unary_nonlinearity": [0.0, 1.0, 1.0],
            "raw_has_interaction": [0.0, 0.0, 0.0],
            "raw_is_simple_model": [1.0, 0.0, 0.0],
            "has_nonlinearity": [0.0, 1.0, 1.0],
            "has_unary_nonlinearity": [0.0, 1.0, 1.0],
            "has_interaction": [0.0, 0.0, 0.0],
            "is_simple_model": [1.0, 0.0, 0.0],
            "has_supported_unary_nonlinearity": [0.0, 1.0, 1.0],
            "has_supported_interaction": [0.0, 0.0, 0.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.20,
        fit_band_ratio=1.05,
        concept_pool_fit_band_ratio=1.15,
        guard_complex_models=True,
        linear_advantage_min_ratio=0.05,
        linear_advantage_min_abs=1e-4,
    )

    assert summary["summary_selection_mode"] == "concept_pools_separability"
    assert summary["lag_pool_model_count"] == pytest.approx(2.0)
    assert summary["nonlinear_candidate_count"] == pytest.approx(1.0)
    assert summary["nonlinear_pool_admitted"] == pytest.approx(1.0)
    assert summary["nonlinear_pool_model_count"] == pytest.approx(1.0)
    assert summary["best_nonlinear_candidate_fit_mse"] == pytest.approx(0.090)
    assert summary["best_non_nonlinear_fit_mse"] == pytest.approx(0.100)
    assert summary["nonlinear_admission_gain"] == pytest.approx(0.01)
    assert summary["nonlinear_admission_threshold"] == pytest.approx(0.01)
    assert summary["nonlinear_pool_rejected_candidate_count"] == pytest.approx(0.0)
    assert summary["nonlinear_node_count_median"] > 1.0
    assert summary["nonlinear_composition_depth_median"] > 1.0
    assert summary["lag_mass_median"] == pytest.approx(4.0)

def test_summarize_expression_features_rejects_weak_nonlinear_candidate_when_non_unary_comparator_is_nearly_as_good():
    table = pd.DataFrame(
        {
            "lag_mass": [1.0, 4.0, 1.1],
            "long_lag_mass": [0.0, 0.5, 0.0],
            "lag_entropy": [0.1, 0.3, 0.12],
            "active_lag_count": [1.0, 2.0, 1.0],
            "max_poly_degree_weighted": [0.0, 2.0, 0.0],
            "nonlinear_node_count": [0.0, 2.0, 0.0],
            "nonlinear_composition_depth": [0.0, 2.0, 0.0],
            "interaction_term_count": [0.0, 0.0, 0.0],
            "interaction_order": [0.0, 0.0, 0.0],
            "interaction_depth": [0.0, 0.0, 0.0],
            "complexity_cost": [1.0, 4.0, 1.1],
            "description_length": [1.5, 4.5, 1.6],
            "fit_mse": [0.100, 0.100, 0.101],
            "expression_complexity": [1, 3, 1],
            "model_idx": [0, 1, 2],
            "seed": [0, 0, 0],
            "equation": ["x0_lag1", "tanh(x0_lag1)", "x0_lag2"],
            "max_lag_used": [1.0, 1.0, 2.0],
            "unary_evidence_gain": [0.0, 0.001, 0.0],
            "interaction_evidence_gain": [0.0, 0.0, 0.0],
            "raw_has_nonlinearity": [0.0, 1.0, 0.0],
            "raw_has_unary_nonlinearity": [0.0, 1.0, 0.0],
            "raw_has_interaction": [0.0, 0.0, 0.0],
            "raw_is_simple_model": [1.0, 0.0, 1.0],
            "has_nonlinearity": [0.0, 1.0, 0.0],
            "has_unary_nonlinearity": [0.0, 1.0, 0.0],
            "has_interaction": [0.0, 0.0, 0.0],
            "is_simple_model": [1.0, 0.0, 1.0],
            "has_supported_unary_nonlinearity": [0.0, 1.0, 0.0],
            "has_supported_interaction": [0.0, 0.0, 0.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.20,
        fit_band_ratio=1.10,
    )

    assert summary["summary_selection_mode"] == "concept_pools_separability"
    assert summary["nonlinear_candidate_count"] == pytest.approx(1.0)
    assert summary["nonlinear_pool_admitted"] == pytest.approx(0.0)
    assert summary["nonlinear_pool_model_count"] == pytest.approx(0.0)
    assert summary["nonlinear_pool_rejected_candidate_count"] == pytest.approx(1.0)
    assert summary["best_nonlinear_candidate_fit_mse"] == pytest.approx(0.100)
    assert summary["best_non_nonlinear_fit_mse"] == pytest.approx(0.100)
    assert summary["nonlinear_admission_gain"] == pytest.approx(0.0)
    assert summary["nonlinear_admission_threshold"] == pytest.approx(0.01)
    assert summary["nonlinear_node_count_median"] == pytest.approx(0.0)
    assert summary["nonlinear_composition_depth_median"] == pytest.approx(0.0)
    assert summary["max_poly_degree_weighted_median"] == pytest.approx(0.0)
    assert summary["lag_mass_median"] == pytest.approx(1.1)

def test_summarize_expression_features_admits_selected_interaction_pool_when_share_threshold_is_met():
    table = pd.DataFrame(
        {
            "lag_mass": [2.0, 2.5, 2.2],
            "long_lag_mass": [0.3, 0.4, 0.3],
            "lag_entropy": [0.2, 0.3, 0.25],
            "active_lag_count": [2.0, 2.0, 2.0],
            "max_poly_degree_weighted": [0.0, 0.0, 0.0],
            "nonlinear_node_count": [0.0, 0.0, 0.0],
            "nonlinear_composition_depth": [0.0, 0.0, 0.0],
            "interaction_term_count": [0.0, 2.0, 1.0],
            "interaction_order": [0.0, 2.0, 2.0],
            "interaction_depth": [0.0, 2.0, 1.0],
            "complexity_cost": [2.0, 3.0, 2.5],
            "description_length": [2.5, 3.5, 3.0],
            "fit_mse": [0.100, 0.090, 0.103],
            "expression_complexity": [2, 3, 2],
            "model_idx": [0, 1, 2],
            "seed": [0, 0, 0],
            "equation": ["x0_lag1", "(x0_lag1 * x0_lag2)", "(x0_lag2 * x0_lag3)"],
            "max_lag_used": [1.0, 2.0, 2.0],
            "unary_evidence_gain": [0.0, 0.0, 0.0],
            "interaction_evidence_gain": [0.0, 0.06, 0.03],
            "raw_has_nonlinearity": [0.0, 1.0, 1.0],
            "raw_has_unary_nonlinearity": [0.0, 0.0, 0.0],
            "raw_has_interaction": [0.0, 1.0, 1.0],
            "raw_is_simple_model": [1.0, 0.0, 0.0],
            "has_nonlinearity": [0.0, 1.0, 1.0],
            "has_unary_nonlinearity": [0.0, 0.0, 0.0],
            "has_interaction": [0.0, 1.0, 1.0],
            "is_simple_model": [1.0, 0.0, 0.0],
            "has_supported_unary_nonlinearity": [0.0, 0.0, 0.0],
            "has_supported_interaction": [0.0, 1.0, 1.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.20,
        fit_band_ratio=1.05,
        concept_pool_fit_band_ratio=1.15,
    )

    assert summary["summary_selection_mode"] == "concept_pools_separability"
    assert summary["interaction_candidate_count"] == pytest.approx(1.0)
    assert summary["interaction_pool_admitted"] == pytest.approx(1.0)
    assert summary["interaction_pool_model_count"] == pytest.approx(1.0)
    assert summary["best_interaction_candidate_fit_mse"] == pytest.approx(0.090)
    assert summary["best_non_interaction_fit_mse"] == pytest.approx(0.100)
    assert summary["interaction_admission_gain"] == pytest.approx(0.01)
    assert summary["interaction_admission_threshold"] == pytest.approx(0.01)
    assert summary["interaction_pool_rejected_candidate_count"] == pytest.approx(0.0)
    assert summary["interaction_term_count_median"] > 1.0
    assert summary["interaction_order_median"] == pytest.approx(2.0)
    assert summary["interaction_depth_median"] > 1.0


def test_summarize_expression_features_rejects_weak_interaction_candidate_when_non_interaction_comparator_is_nearly_as_good():
    table = pd.DataFrame(
        {
            "lag_mass": [2.0, 2.5, 1.1],
            "long_lag_mass": [0.3, 0.4, 0.0],
            "lag_entropy": [0.2, 0.3, 0.15],
            "active_lag_count": [2.0, 2.0, 1.0],
            "max_poly_degree_weighted": [0.0, 0.0, 0.0],
            "nonlinear_node_count": [0.0, 0.0, 0.0],
            "nonlinear_composition_depth": [0.0, 0.0, 0.0],
            "interaction_term_count": [0.0, 2.0, 0.0],
            "interaction_order": [0.0, 2.0, 0.0],
            "interaction_depth": [0.0, 2.0, 0.0],
            "complexity_cost": [2.0, 3.0, 1.1],
            "description_length": [2.5, 3.5, 1.6],
            "fit_mse": [0.100, 0.100, 0.101],
            "expression_complexity": [2, 3, 1],
            "model_idx": [0, 1, 2],
            "seed": [0, 0, 0],
            "equation": ["x0_lag1", "(x0_lag1 * x0_lag2)", "x0_lag2"],
            "max_lag_used": [1.0, 2.0, 2.0],
            "unary_evidence_gain": [0.0, 0.0, 0.0],
            "interaction_evidence_gain": [0.0, 0.001, 0.0],
            "raw_has_nonlinearity": [0.0, 1.0, 0.0],
            "raw_has_unary_nonlinearity": [0.0, 0.0, 0.0],
            "raw_has_interaction": [0.0, 1.0, 0.0],
            "raw_is_simple_model": [1.0, 0.0, 1.0],
            "has_nonlinearity": [0.0, 1.0, 0.0],
            "has_unary_nonlinearity": [0.0, 0.0, 0.0],
            "has_interaction": [0.0, 1.0, 0.0],
            "is_simple_model": [1.0, 0.0, 1.0],
            "has_supported_unary_nonlinearity": [0.0, 0.0, 0.0],
            "has_supported_interaction": [0.0, 1.0, 0.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.20,
        fit_band_ratio=1.05,
    )

    assert summary["summary_selection_mode"] == "concept_pools_separability"
    assert summary["interaction_candidate_count"] == pytest.approx(1.0)
    assert summary["interaction_pool_admitted"] == pytest.approx(0.0)
    assert summary["interaction_pool_model_count"] == pytest.approx(0.0)
    assert summary["interaction_pool_rejected_candidate_count"] == pytest.approx(1.0)
    assert summary["best_interaction_candidate_fit_mse"] == pytest.approx(0.100)
    assert summary["best_non_interaction_fit_mse"] == pytest.approx(0.100)
    assert summary["interaction_admission_gain"] == pytest.approx(0.0)
    assert summary["interaction_admission_threshold"] == pytest.approx(0.01)
    assert summary["interaction_term_count_median"] == pytest.approx(0.0)
    assert summary["interaction_order_median"] == pytest.approx(0.0)
    assert summary["interaction_depth_median"] == pytest.approx(0.0)

def test_summarize_expression_features_lag_metrics_use_adequate_lag_pool():
    table = pd.DataFrame(
        {
            "lag_mass": [4.0, 1.2],
            "long_lag_mass": [1.0, 0.0],
            "lag_entropy": [0.6, 0.1],
            "active_lag_count": [2.0, 1.0],
            "max_poly_degree_weighted": [0.0, 0.0],
            "nonlinear_node_count": [0.0, 0.0],
            "nonlinear_composition_depth": [0.0, 0.0],
            "interaction_term_count": [0.0, 0.0],
            "interaction_order": [0.0, 0.0],
            "interaction_depth": [0.0, 0.0],
            "complexity_cost": [4.0, 1.0],
            "description_length": [4.5, 1.5],
            "fit_mse": [0.120, 0.100],
            "expression_complexity": [4, 1],
            "model_idx": [0, 1],
            "seed": [0, 0],
            "equation": ["x0_lag4", "x0_lag1"],
            "max_lag_used": [4.0, 1.0],
            "unary_evidence_gain": [0.0, 0.0],
            "interaction_evidence_gain": [0.0, 0.0],
            "has_nonlinearity": [0.0, 0.0],
            "has_unary_nonlinearity": [0.0, 0.0],
            "has_interaction": [0.0, 0.0],
            "is_simple_model": [1.0, 1.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.20,
        fit_band_ratio=1.30,
    )

    assert summary["summary_selection_mode"] == "concept_pools_separability"
    assert summary["lag_pool_model_count"] == pytest.approx(2.0)
    assert summary["lag_mass_median"] == pytest.approx(1.2)
    assert summary["long_lag_mass_median"] == pytest.approx(0.0)
    assert summary["active_lag_count_median"] == pytest.approx(1.0)
    assert summary["complexity_cost_median"] == pytest.approx(1.0)

def test_summarize_expression_features_emits_concept_pool_counts_when_no_specialized_pool_exists():
    table = pd.DataFrame(
        {
            "lag_mass": [3.0, 2.5, 1.0],
            "long_lag_mass": [0.5, 0.4, 0.0],
            "lag_entropy": [0.4, 0.3, 0.1],
            "active_lag_count": [2.0, 2.0, 1.0],
            "max_poly_degree_weighted": [2.0, 1.0, 0.0],
            "nonlinear_node_count": [2.0, 1.0, 0.0],
            "nonlinear_composition_depth": [2.0, 1.0, 0.0],
            "interaction_term_count": [0.0, 1.0, 0.0],
            "interaction_order": [0.0, 2.0, 0.0],
            "interaction_depth": [0.0, 2.0, 0.0],
            "complexity_cost": [4.0, 5.0, 1.0],
            "description_length": [4.5, 5.5, 1.5],
            "fit_mse": [0.100, 0.101, 0.130],
            "expression_complexity": [3, 4, 1],
            "model_idx": [0, 1, 2],
            "seed": [0, 0, 0],
            "equation": ["tanh(x0_lag1)", "(x0_lag1 * x0_lag2)", "x0_lag1"],
            "max_lag_used": [1.0, 2.0, 1.0],
            "unary_evidence_gain": [0.001, 0.0, 0.0],
            "interaction_evidence_gain": [0.0, 0.001, 0.0],
            "has_nonlinearity": [1.0, 0.0, 0.0],
            "has_unary_nonlinearity": [1.0, 0.0, 0.0],
            "has_interaction": [0.0, 1.0, 0.0],
            "is_simple_model": [0.0, 0.0, 1.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.20,
        fit_band_ratio=1.05,
        guard_complex_models=True,
        linear_advantage_min_ratio=0.05,
        linear_advantage_min_abs=1e-4,
    )

    assert summary["summary_model_count_pre_guard"] == pytest.approx(2.0)
    assert summary["summary_model_count_post_guard"] == pytest.approx(2.0)
    assert summary["summary_selection_mode"] == "concept_pools_separability"
    assert summary["lag_pool_model_count"] == pytest.approx(2.0)
    assert summary["nonlinear_pool_admitted"] == pytest.approx(0.0)
    assert summary["interaction_pool_admitted"] == pytest.approx(0.0)
    assert summary["nonlinear_pool_model_count"] == pytest.approx(0.0)
    assert summary["interaction_pool_model_count"] == pytest.approx(0.0)
    assert summary["fraction_summary_models_simple"] == pytest.approx(0.0)
    assert summary["fraction_summary_models_unary_nonlinear"] == pytest.approx(0.5)
    assert summary["fraction_summary_models_interaction"] == pytest.approx(0.5)


def test_summarize_expression_features_guard_filters_weak_complex_models():
    table = pd.DataFrame(
        {
            "lag_mass": [1.0, 4.0, 6.0],
            "nonlinear_node_count": [0.0, 2.0, 0.0],
            "nonlinear_composition_depth": [0.0, 1.0, 0.0],
            "interaction_term_count": [0.0, 0.0, 1.0],
            "interaction_order": [0.0, 0.0, 2.0],
            "interaction_depth": [0.0, 0.0, 2.0],
            "complexity_cost": [1.0, 4.0, 5.0],
            "fit_mse": [0.1000, 0.09995, 0.09996],
            "expression_complexity": [1, 3, 4],
            "model_idx": [0, 1, 2],
            "seed": [0, 0, 0],
            "equation": ["x0_lag1", "tanh(x0_lag1)", "(x0_lag1 * x0_lag2)"],
            "max_lag_used": [1.0, 1.0, 2.0],
            "has_nonlinearity": [0.0, 1.0, 0.0],
            "has_unary_nonlinearity": [0.0, 1.0, 0.0],
            "has_interaction": [0.0, 0.0, 1.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.100,
        fit_band_ratio=1.05,
        guard_complex_models=True,
        linear_advantage_min_ratio=0.05,
        linear_advantage_min_abs=1e-4,
    )

    assert summary["summary_model_count_pre_guard"] == pytest.approx(3.0)
    assert summary["summary_model_count_post_guard"] == pytest.approx(1.0)
    assert summary["summary_model_count"] == pytest.approx(1.0)
    assert summary["fraction_summary_models_unary_nonlinear"] == pytest.approx(0.0)
    assert summary["fraction_summary_models_interaction"] == pytest.approx(0.0)
    assert summary["lag_mass_median"] == pytest.approx(1.0)


def test_summarize_expression_features_guard_keeps_clear_advantage_complex_models():
    table = pd.DataFrame(
        {
            "lag_mass": [1.0, 3.0],
            "nonlinear_node_count": [0.0, 2.0],
            "nonlinear_composition_depth": [0.0, 2.0],
            "interaction_term_count": [0.0, 0.0],
            "interaction_order": [0.0, 0.0],
            "interaction_depth": [0.0, 0.0],
            "complexity_cost": [1.0, 4.0],
            "fit_mse": [0.1000, 0.0800],
            "expression_complexity": [1, 3],
            "model_idx": [0, 1],
            "seed": [0, 0],
            "equation": ["x0_lag1", "tanh(x0_lag1)"],
            "max_lag_used": [1.0, 1.0],
            "has_nonlinearity": [0.0, 1.0],
            "has_unary_nonlinearity": [0.0, 1.0],
            "has_interaction": [0.0, 0.0],
            "unary_evidence_gain": [0.0, 0.02],
            "has_supported_unary_nonlinearity": [0.0, 1.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.100,
        fit_band_ratio=1.30,
        guard_complex_models=True,
        linear_advantage_min_ratio=0.05,
        linear_advantage_min_abs=1e-4,
    )

    assert summary["summary_model_count_pre_guard"] == pytest.approx(2.0)
    assert summary["summary_model_count_post_guard"] == pytest.approx(2.0)
    assert summary["fraction_summary_models_unary_nonlinear"] == pytest.approx(0.5)
    assert summary["nonlinear_pool_admitted"] == pytest.approx(1.0)
    assert summary["nonlinear_node_count_median"] >= 1.0


def test_summarize_expression_features_guard_falls_back_when_all_models_would_be_removed():
    table = pd.DataFrame(
        {
            "lag_mass": [2.0, 3.0],
            "nonlinear_node_count": [2.0, 3.0],
            "nonlinear_composition_depth": [1.0, 2.0],
            "interaction_term_count": [0.0, 1.0],
            "interaction_order": [0.0, 2.0],
            "interaction_depth": [0.0, 2.0],
            "complexity_cost": [3.0, 4.0],
            "fit_mse": [0.09995, 0.09996],
            "expression_complexity": [3, 4],
            "model_idx": [0, 1],
            "seed": [0, 0],
            "equation": ["tanh(x0_lag1)", "(x0_lag1 * x0_lag2)"],
            "max_lag_used": [1.0, 2.0],
            "has_nonlinearity": [1.0, 0.0],
            "has_unary_nonlinearity": [1.0, 0.0],
            "has_interaction": [0.0, 1.0],
        }
    )

    summary = summarize_expression_features(
        table,
        linear_fit_mse=0.100,
        fit_band_ratio=1.05,
        guard_complex_models=True,
        linear_advantage_min_ratio=0.05,
        linear_advantage_min_abs=1e-4,
    )

    assert summary["summary_model_count_pre_guard"] == pytest.approx(2.0)
    assert summary["summary_model_count_post_guard"] == pytest.approx(2.0)
    assert summary["summary_model_count"] == pytest.approx(2.0)
    assert summary["fraction_summary_models_unary_nonlinear"] == pytest.approx(0.5)
    assert summary["fraction_summary_models_interaction"] == pytest.approx(0.5)
