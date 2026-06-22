"""Tests for sr_config_study.py - grid runner, diagnostics, helpers."""

from __future__ import annotations

import numpy as np
import pytest

from symbolic_structural_profiling.config import ProfilingConfig
from symbolic_structural_profiling.models.expressions import ExprNode
from symbolic_structural_profiling.sr_config_study import (
    make_nested_constraints,
    make_complexity_of_operators,
    has_nested_unary,
    ensemble_diagnostics,
    run_sr_grid_study,
    grid_study_summary,
)


# ---------------------------------------------------------------------------
# make_nested_constraints
# ---------------------------------------------------------------------------

class TestMakeNestedConstraints:
    def test_level_0_returns_none(self):
        assert make_nested_constraints(0) is None

    def test_level_1_blocks_all_unary_pairs(self):
        unary = ("tanh", "sin", "cos")
        result = make_nested_constraints(1, unary_ops=unary)
        assert result is not None
        for op in unary:
            assert op in result
            for other in unary:
                assert result[op][other] == 0

    def test_level_2_also_blocks_mul_div(self):
        unary = ("tanh", "sin")
        result = make_nested_constraints(2, unary_ops=unary)
        assert result is not None
        for op in unary:
            assert "mul" in result[op]
            assert "div" in result[op]
            assert result[op]["mul"] == 0
            assert result[op]["div"] == 0

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            make_nested_constraints(3)


# ---------------------------------------------------------------------------
# make_complexity_of_operators
# ---------------------------------------------------------------------------

class TestMakeComplexityOfOperators:
    def test_level_0_returns_none(self):
        assert make_complexity_of_operators(0) is None

    def test_level_1_cost_is_3(self):
        result = make_complexity_of_operators(1)
        assert result is not None
        for v in result.values():
            assert v == 3.0

    def test_level_2_cost_is_5(self):
        result = make_complexity_of_operators(2)
        assert result is not None
        for v in result.values():
            assert v == 5.0

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            make_complexity_of_operators(3)


# ---------------------------------------------------------------------------
# has_nested_unary
# ---------------------------------------------------------------------------

class TestHasNestedUnary:
    def _var(self, i: int = 0) -> ExprNode:
        return ExprNode(op="var", feature_index=i, variable=0, lag=1)

    def _tanh(self, child: ExprNode) -> ExprNode:
        return ExprNode(op="tanh", children=(child,))

    def _sin(self, child: ExprNode) -> ExprNode:
        return ExprNode(op="sin", children=(child,))

    def _add(self, a: ExprNode, b: ExprNode) -> ExprNode:
        return ExprNode(op="add", children=(a, b))

    def test_flat_linear_no_nesting(self):
        node = self._add(self._var(0), self._var(1))
        assert not has_nested_unary(node)

    def test_single_tanh_no_nesting(self):
        node = self._tanh(self._var(0))
        assert not has_nested_unary(node)

    def test_tanh_inside_sin_detected(self):
        node = self._sin(self._tanh(self._var(0)))
        assert has_nested_unary(node)

    def test_tanh_inside_tanh_detected(self):
        node = self._tanh(self._tanh(self._var(0)))
        assert has_nested_unary(node)

    def test_tanh_after_add_not_nested(self):
        # tanh(x0 + x1) - tanh's child is add, not a unary
        node = self._tanh(self._add(self._var(0), self._var(1)))
        assert not has_nested_unary(node)


# ---------------------------------------------------------------------------
# ensemble_diagnostics
# ---------------------------------------------------------------------------

class TestEnsembleDiagnostics:
    class _FakeModel:
        def __init__(self, expr: ExprNode, complexity: int = 3, fit_mse: float = 0.05):
            self.expression = expr
            self.complexity = complexity
            self.fit_mse = fit_mse
            self.equation = expr.to_string()

    class _FakeResult:
        def __init__(self, models, best_fit_mse: float = 0.05):
            self.retained_models = models
            self.best_symbolic_fit_mse = best_fit_mse
            self.linear_fit_mse = 0.10
            self.backend = "random"

    def _var(self, i: int = 0) -> ExprNode:
        return ExprNode(op="var", feature_index=i, variable=0, lag=1)

    def _tanh(self, child: ExprNode) -> ExprNode:
        return ExprNode(op="tanh", children=(child,))

    def test_returns_expected_keys(self):
        model = self._FakeModel(self._var(0))
        result = self._FakeResult([model])
        X = np.ones((10, 1))
        diags = ensemble_diagnostics(result, X, [1])
        expected = {
            "nested_unary_fraction",
            "pareto_frontier_size",
            "best_fit_mse",
            "linear_fit_mse",
            "sr_gain",
            "median_complexity",
            "jacobian_nonlinearity_index_median",
            "jacobian_relative_nonlinearity_index_median",
            "backend_used",
            "backend_fallback",
            "best_equation",
            "best_equation_fit_mse",
        }
        assert expected == set(diags.keys())

    def test_no_nested_unary_gives_zero_fraction(self):
        model = self._FakeModel(self._tanh(self._var(0)))  # single tanh, not nested
        result = self._FakeResult([model])
        X = np.ones((10, 1))
        diags = ensemble_diagnostics(result, X, [1])
        assert diags["nested_unary_fraction"] == 0.0

    def test_nested_unary_gives_nonzero_fraction(self):
        nested = ExprNode(op="sin", children=(self._tanh(self._var(0)),))
        model = self._FakeModel(nested)
        result = self._FakeResult([model])
        X = np.ones((10, 1))
        diags = ensemble_diagnostics(result, X, [1])
        assert diags["nested_unary_fraction"] == 1.0

    def test_empty_models_returns_nans(self):
        result = self._FakeResult([])
        X = np.ones((10, 1))
        diags = ensemble_diagnostics(result, X, [1])
        assert diags["pareto_frontier_size"] == 0.0
        import math
        assert math.isnan(diags["nested_unary_fraction"])


# ---------------------------------------------------------------------------
# run_sr_grid_study (smoke test with random backend)
# ---------------------------------------------------------------------------

class TestRunSrGridStudy:
    def _quick_config(self) -> ProfilingConfig:
        cfg = ProfilingConfig()
        cfg.ensemble.symbolic_backend = "random"
        cfg.ensemble.max_lag = 3
        cfg.ensemble.n_seeds = 1
        cfg.ensemble.n_candidates_per_seed = 20
        cfg.ensemble.top_k_per_seed = 2
        return cfg

    def test_returns_dataframe_with_expected_columns(self):
        df = run_sr_grid_study(
            base_config=self._quick_config(),
            scenarios=["nonlinearity_only", "amplitude_only"],
            seeds=[0],
            nesting_levels=[0, 1],
            cost_levels=[0],
            n_steps=400,
            window_size=60,
        )
        assert not df.empty
        for col in ("nesting_level", "cost_level", "scenario", "seed",
                    "nested_unary_fraction", "pareto_frontier_size",
                    "best_fit_mse", "jacobian_nonlinearity_index_median"):
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count(self):
        df = run_sr_grid_study(
            base_config=self._quick_config(),
            scenarios=["nonlinearity_only"],
            seeds=[0, 1],
            nesting_levels=[0, 2],
            cost_levels=[0, 1],
            n_steps=400,
            window_size=60,
        )
        # 2 nesting × 2 cost × 1 scenario × 2 seeds = 8
        assert len(df) == 8

    def test_nesting_labels_present(self):
        df = run_sr_grid_study(
            base_config=self._quick_config(),
            scenarios=["amplitude_only"],
            seeds=[0],
            nesting_levels=[0, 1, 2],
            cost_levels=[0],
            n_steps=400,
            window_size=60,
        )
        assert set(df["nesting_label"]) == {"none", "no_unary_in_unary", "strict"}


# ---------------------------------------------------------------------------
# grid_study_summary
# ---------------------------------------------------------------------------

class TestGridStudySummary:
    def _quick_config(self) -> ProfilingConfig:
        cfg = ProfilingConfig()
        cfg.ensemble.symbolic_backend = "random"
        cfg.ensemble.max_lag = 3
        cfg.ensemble.n_seeds = 1
        cfg.ensemble.n_candidates_per_seed = 20
        cfg.ensemble.top_k_per_seed = 2
        return cfg

    def test_summary_is_string(self):
        df = run_sr_grid_study(
            base_config=self._quick_config(),
            scenarios=["nonlinearity_only"],
            seeds=[0],
            nesting_levels=[0],
            cost_levels=[0],
            n_steps=400,
            window_size=60,
        )
        summary = grid_study_summary(df)
        assert isinstance(summary, str)
        assert "nonlinearity_only" in summary

    def test_empty_df_returns_no_results(self):
        import pandas as pd
        summary = grid_study_summary(pd.DataFrame())
        assert "No results" in summary
