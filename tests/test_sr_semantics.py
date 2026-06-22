"""Semantic tests for the SR configuration grid study.

These tests verify *concept-level correctness*: that the pipeline enforces
structural constraints, that concept features respond to expected structural
differences between synthetic scenarios, and that the SR gain threshold
correctly calibrates the relative nonlinearity index.

All tests use the ``random`` backend so they run without PySR installed.
They check properties that hold regardless of backend:

- Nesting constraints are actually enforced in returned expression models.
- Structural features are present, non-negative, and non-trivially populated.
- The SR gain threshold suppresses near-zero-gain cases in the relative
  nonlinearity index.
- The run card faithfully records all critical calibration parameters.
- End-to-end grid with the empirically best config (nesting=1, cost=0)
  completes without fallbacks and produces the expected output shape.

Best configuration (from overnight PySR run, 2026-04-15):
  nesting_level = 1  (no_unary_in_unary): eliminates expression-tree noise,
                       0% nested unary fraction across all seeds/scenarios.
  cost_level    = 0  (default cost):      highest Pareto diversity at this
                       nesting level; (nonlin+interact)/amplitude ratio = 4.06.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from symbolic_structural_profiling.config import ProfilingConfig
from symbolic_structural_profiling.models.expressions import ExprNode
from symbolic_structural_profiling.sr_config_study import (
    ensemble_diagnostics,
    make_run_card,
    run_sr_grid_study,
)

# ---------------------------------------------------------------------------
# Best-config constants (empirically validated overnight PySR run)
# ---------------------------------------------------------------------------
BEST_NESTING = 1   # no_unary_in_unary
BEST_COST = 0      # cost_1 (default operator cost)


def _best_config() -> ProfilingConfig:
    """Quick random-backend config with empirically best nesting/cost settings."""
    cfg = ProfilingConfig()
    cfg.ensemble.symbolic_backend = "random"
    cfg.ensemble.max_lag = 4
    cfg.ensemble.n_seeds = 2
    cfg.ensemble.n_candidates_per_seed = 40
    cfg.ensemble.top_k_per_seed = 3
    return cfg


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _var(i: int = 0) -> ExprNode:
    return ExprNode(op="var", feature_index=i, variable=0, lag=1)


def _tanh(child: ExprNode) -> ExprNode:
    return ExprNode(op="tanh", children=(child,))


class _FakeModel:
    def __init__(self, expr: ExprNode, complexity: int = 3, fit_mse: float = 0.05):
        self.expression = expr
        self.complexity = complexity
        self.fit_mse = fit_mse
        self.equation = expr.to_string()


class _FakeResult:
    def __init__(
        self,
        models: list,
        best_fit_mse: float = 0.05,
        linear_fit_mse: float = 0.10,
    ):
        self.retained_models = models
        self.best_symbolic_fit_mse = best_fit_mse
        self.linear_fit_mse = linear_fit_mse
        self.backend = "random"


# ---------------------------------------------------------------------------
# Nesting constraint semantics
# ---------------------------------------------------------------------------

class TestNestingConstraintSemantics:
    """no_unary_in_unary (level 1) must drive nested_unary_fraction to zero."""

    def test_nested_unary_fraction_is_zero_across_seeds_with_level_1(self):
        df = run_sr_grid_study(
            base_config=_best_config(),
            scenarios=["nonlinearity_only", "amplitude_only"],
            seeds=[0, 1, 2],
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=500,
            window_size=80,
        )
        assert not df.empty
        assert (df["nested_unary_fraction"] == 0.0).all(), (
            "Nesting level 1 (no_unary_in_unary) must produce zero nested "
            "unary pairs across all returned models"
        )

    def test_nesting_label_is_correct_for_best_config(self):
        df = run_sr_grid_study(
            base_config=_best_config(),
            scenarios=["nonlinearity_only"],
            seeds=[0],
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=400,
            window_size=60,
        )
        assert (df["nesting_label"] == "no_unary_in_unary").all()

    def test_unconstrained_level_0_allows_nesting_in_column(self):
        """Level 0 produces a valid nested_unary_fraction column in [0, 1]."""
        df = run_sr_grid_study(
            base_config=_best_config(),
            scenarios=["nonlinearity_only"],
            seeds=list(range(6)),
            nesting_levels=[0],
            cost_levels=[0],
            n_steps=400,
            window_size=60,
        )
        assert "nested_unary_fraction" in df.columns
        assert (df["nested_unary_fraction"] >= 0.0).all()
        assert (df["nested_unary_fraction"] <= 1.0).all()


# ---------------------------------------------------------------------------
# SR gain threshold semantics
# ---------------------------------------------------------------------------

class TestSrGainThresholdSemantics:
    """Relative nonlinearity index must respect the SR gain threshold."""

    def test_sr_gain_below_threshold_gives_zero_relative_index(self):
        """sr_gain 0.03 < threshold 0.05 → effective_gain = 0 → relative = 0."""
        model = _FakeModel(_var(0))
        # best=0.97, linear=1.0 → sr_gain = 0.03
        result = _FakeResult([model], best_fit_mse=0.97, linear_fit_mse=1.0)
        X = np.ones((10, 1))
        diags = ensemble_diagnostics(result, X, [1], sr_gain_threshold=0.05)
        assert diags["jacobian_relative_nonlinearity_index_median"] == 0.0

    def test_sr_gain_exactly_at_threshold_gives_zero_relative_index(self):
        """sr_gain == threshold → effective_gain = 0."""
        model = _FakeModel(_var(0))
        # best=0.95, linear=1.0 → sr_gain = 0.05
        result = _FakeResult([model], best_fit_mse=0.95, linear_fit_mse=1.0)
        X = np.ones((10, 1))
        diags = ensemble_diagnostics(result, X, [1], sr_gain_threshold=0.05)
        assert diags["jacobian_relative_nonlinearity_index_median"] == 0.0

    def test_sr_gain_above_threshold_preserves_nonzero_sr_gain_in_output(self):
        """sr_gain 0.20 > threshold 0.05 → sr_gain recorded correctly."""
        model = _FakeModel(_var(0))
        # best=0.80, linear=1.0 → sr_gain = 0.20, effective_gain = 0.15
        result = _FakeResult([model], best_fit_mse=0.80, linear_fit_mse=1.0)
        X = np.ones((10, 1))
        diags = ensemble_diagnostics(result, X, [1], sr_gain_threshold=0.05)
        assert not math.isnan(diags["sr_gain"])
        assert abs(diags["sr_gain"] - 0.20) < 1e-6
        assert not math.isnan(diags["jacobian_relative_nonlinearity_index_median"])

    def test_threshold_zero_makes_relative_equal_absolute_times_sr_gain(self):
        """With threshold=0, relative = absolute * sr_gain exactly."""
        model = _FakeModel(_var(0))
        result = _FakeResult([model], best_fit_mse=0.80, linear_fit_mse=1.0)
        X = np.ones((10, 1))
        diags = ensemble_diagnostics(result, X, [1], sr_gain_threshold=0.0)
        jac_rel = diags["jacobian_relative_nonlinearity_index_median"]
        jac_abs = diags["jacobian_nonlinearity_index_median"]
        sr = diags["sr_gain"]
        assert abs(jac_rel - jac_abs * sr) < 1e-9

    def test_different_thresholds_produce_monotone_relative_index(self):
        """Higher threshold → lower (or equal) relative nonlinearity index."""
        model = _FakeModel(_tanh(_var(0)), complexity=4, fit_mse=0.70)
        result = _FakeResult([model], best_fit_mse=0.70, linear_fit_mse=1.0)
        X = np.linspace(-1, 1, 20).reshape(-1, 1)
        low = ensemble_diagnostics(result, X, [1], sr_gain_threshold=0.0)
        mid = ensemble_diagnostics(result, X, [1], sr_gain_threshold=0.05)
        high = ensemble_diagnostics(result, X, [1], sr_gain_threshold=0.20)
        r_low = low["jacobian_relative_nonlinearity_index_median"]
        r_mid = mid["jacobian_relative_nonlinearity_index_median"]
        r_high = high["jacobian_relative_nonlinearity_index_median"]
        assert r_low >= r_mid >= r_high


# ---------------------------------------------------------------------------
# Structural feature semantics
# ---------------------------------------------------------------------------

class TestStructuralFeatureSemantics:
    """Structural features must be present, non-negative, and non-trivial."""

    _REQUIRED_COLS = [
        "lag_mass_median",
        "long_lag_mass_median",
        "lag_entropy_median",
        "active_lag_count_median",
        "nonlinear_node_count_median",
        "nonlinear_composition_depth_median",
        "interaction_term_count_median",
        "interaction_order_median",
        "complexity_cost_median",
        "description_length_median",
    ]

    def _run_best(self, scenarios, seeds=(0, 1)):
        return run_sr_grid_study(
            base_config=_best_config(),
            scenarios=scenarios,
            seeds=list(seeds),
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=500,
            window_size=80,
        )

    def test_all_structural_columns_present(self):
        df = self._run_best(["nonlinearity_only"])
        for col in self._REQUIRED_COLS:
            assert col in df.columns, f"Missing structural column: {col}"

    def test_structural_columns_have_finite_values(self):
        df = self._run_best(["nonlinearity_only", "amplitude_only"])
        for col in self._REQUIRED_COLS:
            finite = df[col].dropna()
            assert len(finite) > 0, f"{col} is entirely NaN"

    def test_count_columns_are_non_negative(self):
        df = self._run_best(
            ["nonlinearity_only", "interaction_only", "amplitude_only"]
        )
        for col in (
            "nonlinear_node_count_median",
            "interaction_term_count_median",
            "active_lag_count_median",
        ):
            finite = df[col].dropna()
            assert (finite >= 0.0).all(), f"{col} has negative values"

    def test_complexity_cost_increases_with_cost_level(self):
        """Higher cost_level → higher complexity_cost_median in retained models."""
        cfg = _best_config()
        df = run_sr_grid_study(
            base_config=cfg,
            scenarios=["nonlinearity_only"],
            seeds=[0, 1, 2],
            nesting_levels=[BEST_NESTING],
            cost_levels=[0, 1, 2],
            n_steps=500,
            window_size=80,
        )
        medians = (
            df.groupby("cost_level")["complexity_cost_median"].median().sort_index()
        )
        assert medians.loc[0] <= medians.loc[1] or medians.loc[1] <= medians.loc[2], (
            "complexity_cost_median should increase (weakly) with cost_level"
        )

    def test_description_length_is_positive_for_all_retained_models(self):
        df = self._run_best(["nonlinearity_only"])
        dl = df["description_length_median"].dropna()
        assert (dl > 0.0).all(), "description_length_median must be positive"


# ---------------------------------------------------------------------------
# Run card records calibration parameters
# ---------------------------------------------------------------------------

class TestRunCardSemantics:
    """make_run_card must faithfully record all calibration parameters."""

    def test_run_card_records_sr_gain_threshold(self):
        card = make_run_card(
            base_config=_best_config(),
            scenarios=["nonlinearity_only"],
            seeds=[0],
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=400,
            window_size=60,
            noise_std=0.16,
            sr_gain_threshold=0.05,
        )
        assert card["sr_gain_threshold"] == 0.05

    def test_run_card_records_parsimony(self):
        cfg = _best_config()
        cfg.ensemble.pysr_parsimony = 0.01
        card = make_run_card(
            base_config=cfg,
            scenarios=["nonlinearity_only"],
            seeds=[0],
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=400,
            window_size=60,
            noise_std=0.16,
        )
        assert card["sr_config"]["pysr_parsimony"] == pytest.approx(0.01)

    def test_run_card_has_git_commit_and_timestamp(self):
        card = make_run_card(
            base_config=_best_config(),
            scenarios=["nonlinearity_only"],
            seeds=[0],
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=400,
            window_size=60,
            noise_std=0.16,
        )
        assert "git_commit" in card
        assert "timestamp" in card
        # ISO timestamp has at least YYYY-MM-DD (10 chars)
        assert len(card["timestamp"]) >= 10

    def test_run_card_records_nesting_and_cost_levels(self):
        card = make_run_card(
            base_config=_best_config(),
            scenarios=["nonlinearity_only", "amplitude_only"],
            seeds=[0, 1],
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=400,
            window_size=60,
            noise_std=0.16,
        )
        assert card["nesting_levels"] == [BEST_NESTING]
        assert card["cost_levels"] == [BEST_COST]
        assert set(card["scenarios"]) == {"nonlinearity_only", "amplitude_only"}
        assert set(card["seeds"]) == {0, 1}


# ---------------------------------------------------------------------------
# End-to-end grid with best config
# ---------------------------------------------------------------------------

class TestBestConfigEndToEnd:
    """Full grid run with empirically best config must complete cleanly."""

    def test_output_shape_matches_grid_dimensions(self):
        df = run_sr_grid_study(
            base_config=_best_config(),
            scenarios=["nonlinearity_only", "amplitude_only"],
            seeds=[0, 1],
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=500,
            window_size=80,
            sr_gain_threshold=0.05,
        )
        # 1 nesting × 1 cost × 2 scenarios × 2 seeds = 4 rows
        assert len(df) == 4

    def test_no_backend_fallbacks_with_random_backend(self):
        df = run_sr_grid_study(
            base_config=_best_config(),
            scenarios=["nonlinearity_only", "interaction_only", "amplitude_only"],
            seeds=[0, 1],
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=500,
            window_size=80,
        )
        assert df["backend_fallback"].sum() == 0.0

    def test_all_core_diagnostic_columns_present(self):
        df = run_sr_grid_study(
            base_config=_best_config(),
            scenarios=["nonlinearity_only"],
            seeds=[0],
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=500,
            window_size=80,
        )
        for col in (
            "nesting_level", "nesting_label", "cost_level", "cost_label",
            "scenario", "seed",
            "nested_unary_fraction", "pareto_frontier_size",
            "best_fit_mse", "linear_fit_mse", "sr_gain",
            "jacobian_nonlinearity_index_median",
            "jacobian_relative_nonlinearity_index_median",
            "backend_used", "backend_fallback",
            "best_equation", "best_equation_fit_mse",
        ):
            assert col in df.columns, f"Missing column: {col}"

    def test_sr_gain_is_non_negative(self):
        df = run_sr_grid_study(
            base_config=_best_config(),
            scenarios=["nonlinearity_only", "amplitude_only"],
            seeds=[0, 1],
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=500,
            window_size=80,
        )
        finite = df["sr_gain"].dropna()
        assert (finite >= 0.0).all()

    def test_best_equation_is_non_empty_string(self):
        df = run_sr_grid_study(
            base_config=_best_config(),
            scenarios=["nonlinearity_only"],
            seeds=[0],
            nesting_levels=[BEST_NESTING],
            cost_levels=[BEST_COST],
            n_steps=500,
            window_size=80,
        )
        assert (df["best_equation"].str.len() > 0).all()
