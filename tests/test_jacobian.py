"""Tests for models/jacobian.py - symbolic differentiation and Jacobian concepts."""

from __future__ import annotations

import numpy as np

from symbolic_structural_profiling.models.expressions import ExprNode, safe_eval
from symbolic_structural_profiling.models.jacobian import (
    symbolic_diff,
    gradient_mean_abs,
    gradient_std,
    jacobian_features,
    ensemble_jacobian_features,
    _zero_jacobian_features,
)


# ---------------------------------------------------------------------------
# Helpers to build simple expression trees
# ---------------------------------------------------------------------------

def _var(feature_index: int, lag: int = 1) -> ExprNode:
    return ExprNode(op="var", feature_index=feature_index, variable=0, lag=lag)


def _const(v: float) -> ExprNode:
    return ExprNode(op="const", value=v)


def _add(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(op="add", children=(a, b))


def _mul(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(op="mul", children=(a, b))


def _tanh(a: ExprNode) -> ExprNode:
    return ExprNode(op="tanh", children=(a,))


def _sin(a: ExprNode) -> ExprNode:
    return ExprNode(op="sin", children=(a,))


def _neg(a: ExprNode) -> ExprNode:
    return ExprNode(op="neg", children=(a,))


# ---------------------------------------------------------------------------
# Layer 1: symbolic_diff correctness
# ---------------------------------------------------------------------------

class TestSymbolicDiff:
    def test_diff_var_self(self):
        """∂x_0/∂x_0 = 1."""
        node = _var(0)
        grad = symbolic_diff(node, 0)
        X = np.random.default_rng(0).normal(size=(10, 2))
        result = safe_eval(grad, X)
        np.testing.assert_allclose(result, 1.0)

    def test_diff_var_other(self):
        """∂x_1/∂x_0 = 0."""
        node = _var(1)
        grad = symbolic_diff(node, 0)
        X = np.random.default_rng(0).normal(size=(10, 2))
        result = safe_eval(grad, X)
        np.testing.assert_allclose(result, 0.0)

    def test_diff_const(self):
        """∂c/∂x_0 = 0."""
        node = _const(3.14)
        grad = symbolic_diff(node, 0)
        X = np.random.default_rng(0).normal(size=(10, 1))
        result = safe_eval(grad, X)
        np.testing.assert_allclose(result, 0.0)

    def test_diff_add(self):
        """∂(x0 + x1)/∂x0 = 1."""
        node = _add(_var(0), _var(1))
        grad = symbolic_diff(node, 0)
        X = np.random.default_rng(0).normal(size=(10, 2))
        result = safe_eval(grad, X)
        np.testing.assert_allclose(result, 1.0)

    def test_diff_mul_product_rule(self):
        """∂(x0 * x1)/∂x0 = x1."""
        node = _mul(_var(0), _var(1))
        grad = symbolic_diff(node, 0)
        X = np.random.default_rng(0).normal(size=(10, 2))
        expected = X[:, 1]
        result = safe_eval(grad, X)
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_diff_linear_expression(self):
        """∂(a*x0 + b*x1)/∂x0 = a (a constant scalar)."""
        # 0.5 * x0 + 2.0 * x1
        node = _add(_mul(_const(0.5), _var(0)), _mul(_const(2.0), _var(1)))
        grad = symbolic_diff(node, 0)
        X = np.ones((8, 2))
        result = safe_eval(grad, X)
        np.testing.assert_allclose(result, 0.5, atol=1e-9)

    def test_diff_tanh_chain_rule(self):
        """∂tanh(x0)/∂x0 = 1 - tanh²(x0) = sech²(x0)."""
        node = _tanh(_var(0))
        grad = symbolic_diff(node, 0)
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 1))
        expected = 1.0 - np.tanh(X[:, 0]) ** 2
        result = safe_eval(grad, X)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_diff_sin_chain_rule(self):
        """∂sin(x0)/∂x0 = cos(x0)."""
        node = _sin(_var(0))
        grad = symbolic_diff(node, 0)
        rng = np.random.default_rng(7)
        X = rng.normal(size=(15, 1))
        expected = np.cos(X[:, 0])
        result = safe_eval(grad, X)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_diff_neg(self):
        """∂(-x0)/∂x0 = -1."""
        node = _neg(_var(0))
        grad = symbolic_diff(node, 0)
        X = np.random.default_rng(0).normal(size=(5, 1))
        result = safe_eval(grad, X)
        np.testing.assert_allclose(result, -1.0, atol=1e-9)

    def test_diff_tanh_nested_chain(self):
        """∂tanh(0.5*x0)/∂x0 = 0.5 * sech²(0.5*x0)."""
        inner = _mul(_const(0.5), _var(0))
        node = _tanh(inner)
        grad = symbolic_diff(node, 0)
        rng = np.random.default_rng(11)
        X = rng.uniform(-1, 1, size=(20, 1))
        arg = 0.5 * X[:, 0]
        expected = 0.5 * (1.0 - np.tanh(arg) ** 2)
        result = safe_eval(grad, X)
        np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Layer 2: gradient_std as nonlinearity signal
# ---------------------------------------------------------------------------

class TestGradientStd:
    def test_linear_expression_has_zero_gradient_std(self):
        """A purely linear expression has constant gradient → std = 0."""
        # f(x) = 0.7*x0 + 0.3*x1
        node = _add(_mul(_const(0.7), _var(0)), _mul(_const(0.3), _var(1)))
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2))
        stds = gradient_std(node, X)
        np.testing.assert_allclose(stds, 0.0, atol=1e-9)

    def test_tanh_in_saturation_has_nonzero_gradient_std(self):
        """tanh with large inputs produces varying gradient → std > 0."""
        # f(x) = tanh(3*x0) - x0 values span -2..2, well into saturation.
        node = _tanh(_mul(_const(3.0), _var(0)))
        X = np.linspace(-2.0, 2.0, 50)[:, None]
        stds = gradient_std(node, X)
        assert stds[0] > 0.01, "tanh in saturation should have non-trivial gradient variance"

    def test_tanh_in_linear_regime_has_low_gradient_std(self):
        """tanh with tiny inputs acts linearly → std ≈ 0."""
        node = _tanh(_mul(_const(0.001), _var(0)))
        X = np.linspace(-0.1, 0.1, 50)[:, None]
        stds = gradient_std(node, X)
        # Not exactly zero (some float noise), but much smaller than the saturation case.
        assert stds[0] < 1e-4, "tanh in linear regime should have near-zero gradient variance"

    def test_gradient_mean_abs_shape(self):
        node = _add(_var(0), _var(1))
        X = np.random.default_rng(0).normal(size=(20, 2))
        result = gradient_mean_abs(node, X)
        assert result.shape == (2,)

    def test_gradient_std_shape(self):
        node = _tanh(_var(0))
        X = np.random.default_rng(0).normal(size=(20, 1))
        result = gradient_std(node, X)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# Layer 3: jacobian_features
# ---------------------------------------------------------------------------

class TestJacobianFeatures:
    def _lag_for_feature(self, n: int, start: int = 1) -> list[int]:
        return list(range(start, start + n))

    def test_keys_present(self):
        node = _add(_var(0), _var(1))
        X = np.random.default_rng(0).normal(size=(30, 2))
        feats = jacobian_features(node, X, self._lag_for_feature(2))
        expected_keys = set(_zero_jacobian_features().keys())
        assert expected_keys == set(feats.keys())

    def test_linear_has_zero_nonlinearity_index(self):
        """Linear expression → jacobian_nonlinearity_index = 0."""
        node = _add(_mul(_const(0.5), _var(0)), _mul(_const(0.3), _var(1)))
        X = np.random.default_rng(0).normal(size=(40, 2))
        feats = jacobian_features(node, X, self._lag_for_feature(2))
        assert feats["jacobian_nonlinearity_index"] < 1e-6

    def test_nonlinear_has_positive_nonlinearity_index(self):
        """tanh in saturation → jacobian_nonlinearity_index > 0."""
        node = _tanh(_mul(_const(3.0), _var(0)))
        X = np.linspace(-2.0, 2.0, 40)[:, None]
        feats = jacobian_features(node, X, [1])
        assert feats["jacobian_nonlinearity_index"] > 0.1

    def test_nonlinearity_index_linear_less_than_nonlinear(self):
        """Linear expression should have lower index than true nonlinear."""
        linear_node = _mul(_const(0.5), _var(0))
        nonlinear_node = _tanh(_mul(_const(4.0), _var(0)))
        X = np.linspace(-1.5, 1.5, 40)[:, None]
        lin_feats = jacobian_features(linear_node, X, [1])
        nonlin_feats = jacobian_features(nonlinear_node, X, [1])
        assert lin_feats["jacobian_nonlinearity_index"] < nonlin_feats["jacobian_nonlinearity_index"]

    def test_lag_mass_positive_for_active_var(self):
        node = _var(0)
        X = np.random.default_rng(0).normal(size=(20, 1))
        feats = jacobian_features(node, X, [1])
        assert feats["jacobian_lag_mass"] > 0.0

    def test_empty_X_returns_zeros(self):
        node = _var(0)
        X = np.empty((0, 1))
        feats = jacobian_features(node, X, [1])
        assert all(v == 0.0 for v in feats.values())

    def test_long_lag_mass_fraction(self):
        """Variables at long lags should contribute to jacobian_long_lag_mass."""
        # x0 is lag 1 (short), x1 is lag 5 (long, cutoff=3)
        node = _add(_var(0), _var(1))
        X = np.ones((20, 2))
        feats = jacobian_features(node, X, lag_for_feature=[1, 5], memory_long_lag_cutoff=3)
        # Both gradients are 1, so long fraction = 0.5
        assert abs(feats["jacobian_long_lag_mass"] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# ensemble_jacobian_features
# ---------------------------------------------------------------------------

class TestEnsembleJacobianFeatures:
    class _FakeModel:
        def __init__(self, expr: ExprNode):
            self.expression = expr

    def test_returns_median_keys(self):
        models = [
            self._FakeModel(_mul(_const(0.5), _var(0))),
            self._FakeModel(_tanh(_var(0))),
        ]
        X = np.linspace(-1, 1, 30)[:, None]
        result = ensemble_jacobian_features(models, X, [1])
        expected = {f"{k}_median" for k in _zero_jacobian_features()}
        assert expected == set(result.keys())

    def test_empty_models_returns_zeros(self):
        X = np.ones((10, 1))
        result = ensemble_jacobian_features([], X, [1])
        assert all(v == 0.0 for v in result.values())
