"""Analytical Jacobian extraction from symbolic expression trees.

Three layers:
  1. symbolic_diff  - produces an ExprNode representing ∂f/∂x_j exactly.
  2. gradient_*     - evaluates those derivative expressions over data arrays.
  3. jacobian_features - aggregates gradient statistics into concept-level scalars.

The key insight is that gradient *variance* across data points distinguishes
real nonlinearity from syntactic nonlinearity:
  - A linear expression has a constant gradient → variance = 0.
  - A tanh in its saturated/curved region has a varying gradient → variance > 0.
  - A tanh in its linear regime (small inputs) behaves like a linear term
    and also shows near-zero gradient variance - the signal we were missing.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .expressions import ExprNode, safe_eval


# ---------------------------------------------------------------------------
# Layer 1 - symbolic differentiation
# ---------------------------------------------------------------------------

_ZERO = ExprNode(op="const", value=0.0)
_ONE = ExprNode(op="const", value=1.0)


def symbolic_diff(node: ExprNode, feature_index: int) -> ExprNode:
    """Return an ExprNode representing ∂node/∂x_{feature_index}.

    The result is a valid ExprNode that can be evaluated with ``safe_eval``.
    All standard chain / product / quotient rules are applied recursively.
    """
    op = node.op

    if op == "var":
        return _ONE if node.feature_index == feature_index else _ZERO

    if op == "const":
        return _ZERO

    if op == "add":
        da = symbolic_diff(node.children[0], feature_index)
        db = symbolic_diff(node.children[1], feature_index)
        return ExprNode(op="add", children=(da, db))

    if op == "sub":
        da = symbolic_diff(node.children[0], feature_index)
        db = symbolic_diff(node.children[1], feature_index)
        return ExprNode(op="sub", children=(da, db))

    if op == "mul":
        a, b = node.children
        da = symbolic_diff(a, feature_index)
        db = symbolic_diff(b, feature_index)
        # d(a*b)/dx = a*db + b*da
        return ExprNode(
            op="add",
            children=(
                ExprNode(op="mul", children=(a, db)),
                ExprNode(op="mul", children=(b, da)),
            ),
        )

    if op == "div":
        a, b = node.children
        da = symbolic_diff(a, feature_index)
        db = symbolic_diff(b, feature_index)
        # d(a/b)/dx = (da*b - a*db) / b²
        numerator = ExprNode(
            op="sub",
            children=(
                ExprNode(op="mul", children=(da, b)),
                ExprNode(op="mul", children=(a, db)),
            ),
        )
        denominator = ExprNode(op="mul", children=(b, b))
        return ExprNode(op="div", children=(numerator, denominator))

    if op == "neg":
        da = symbolic_diff(node.children[0], feature_index)
        return ExprNode(op="neg", children=(da,))

    if op == "sin":
        a = node.children[0]
        da = symbolic_diff(a, feature_index)
        cos_a = ExprNode(op="cos", children=(a,))
        return ExprNode(op="mul", children=(cos_a, da))

    if op == "cos":
        a = node.children[0]
        da = symbolic_diff(a, feature_index)
        sin_a = ExprNode(op="sin", children=(a,))
        neg_sin_a = ExprNode(op="neg", children=(sin_a,))
        return ExprNode(op="mul", children=(neg_sin_a, da))

    if op == "tanh":
        a = node.children[0]
        da = symbolic_diff(a, feature_index)
        # d(tanh(a))/dx = (1 - tanh²(a)) * da/dx
        tanh_sq = ExprNode(op="mul", children=(node, node))
        sech_sq = ExprNode(op="sub", children=(_ONE, tanh_sq))
        return ExprNode(op="mul", children=(sech_sq, da))

    if op == "exp":
        a = node.children[0]
        da = symbolic_diff(a, feature_index)
        # d(exp(a))/dx = exp(a) * da/dx
        return ExprNode(op="mul", children=(node, da))

    if op == "log":
        a = node.children[0]
        da = symbolic_diff(a, feature_index)
        # d(log|a|)/dx = da/dx / a  (safe_eval handles |a|+eps)
        return ExprNode(op="div", children=(da, a))

    # Unknown op - treat as constant w.r.t. the feature.
    return _ZERO


# ---------------------------------------------------------------------------
# Layer 2 - gradient evaluation
# ---------------------------------------------------------------------------

def gradient_mean_abs(node: ExprNode, X: np.ndarray) -> np.ndarray:
    """Mean absolute gradient for each input feature over all rows of X.

    Returns shape ``(n_features,)``.  Entry j is ``mean(|∂f/∂x_j|)``
    evaluated over the ``X.shape[0]`` data points.
    """
    n_features = X.shape[1]
    out = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        grad_expr = symbolic_diff(node, j)
        grad_vals = safe_eval(grad_expr, X)
        out[j] = float(np.mean(np.abs(grad_vals)))
    return out


def gradient_std(node: ExprNode, X: np.ndarray) -> np.ndarray:
    """Standard deviation of the gradient for each feature over all rows of X.

    Returns shape ``(n_features,)``.  Entry j is ``std(∂f/∂x_j)``
    evaluated over the data.  This is the nonlinearity signal:
      - linear expression  → gradient is constant → std ≈ 0
      - true nonlinear     → gradient varies      → std > 0
    """
    n_features = X.shape[1]
    out = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        grad_expr = symbolic_diff(node, j)
        grad_vals = safe_eval(grad_expr, X)
        out[j] = float(np.std(grad_vals))
    return out


# ---------------------------------------------------------------------------
# Layer 3 - concept features
# ---------------------------------------------------------------------------

def jacobian_features(
    node: ExprNode,
    X: np.ndarray,
    lag_for_feature: Sequence[int],
    memory_long_lag_cutoff: int = 3,
) -> Dict[str, float]:
    """Compute Jacobian-based concept features for one expression over one window.

    Parameters
    ----------
    node:
        The fitted symbolic expression.
    X:
        Window design matrix, shape ``(n_samples, n_features)``.  Each column
        corresponds to one lag of the target variable.
    lag_for_feature:
        ``lag_for_feature[j]`` is the lag index (1-based) of column j in X.
        Used to distinguish short-memory vs long-memory lags.
    memory_long_lag_cutoff:
        Lags at or above this index are considered "long-memory" lags.
        Convention matches the structural-layer ``long_lag_mass``
        (``lags >= cutoff``) so that the two features can be compared
        directly.

    Returns
    -------
    dict with keys:
      jacobian_lag_mass
          Sum across lags of the per-window mean of
          ``|df/dx_j|`` (unnormalized). Total gradient magnitude
          aggregated across lags; the Jacobian analogue of the
          structural lag_mass total, NOT a position-weighted
          centroid. See features.lag_mass for the position-weighted
          structural counterpart.
      jacobian_long_lag_mass
          Fraction of gradient mass attributable to lags >=
          ``memory_long_lag_cutoff``.
      jacobian_centroid_norm
          ||∇f||₂ at the mean data point - the Euclidean norm of the
          gradient vector evaluated at mean(X, axis=0).  Named for the
          fact that the gradient is taken at the data centroid; this is
          a single-point evaluation, not a sample-averaged quantity.
      jacobian_nonlinearity_index
          Sum over features of ``std(∂f/∂x_j) / (mean(|∂f/∂x_j|) + ε)``.
          Zero for linear expressions; positive for expressions whose
          gradient varies across the observed data range.
      jacobian_grad_variance_total
          Sum of per-feature gradient variance.  Raw scale version of
          the nonlinearity index.
    """
    n_features = X.shape[1]
    if n_features == 0 or X.shape[0] == 0:
        return _zero_jacobian_features()

    mean_abs = gradient_mean_abs(node, X)
    stds = gradient_std(node, X)

    # --- jacobian_lag_mass ---
    total_mean_abs = float(np.sum(mean_abs))
    jacobian_lag_mass = total_mean_abs  # unnormalized sum; analogue of lag_mass

    # --- jacobian_long_lag_mass ---
    # Convention: lags >= memory_long_lag_cutoff are "long" (matches the
    # structural-layer long_lag_mass in features.py).
    long_mask = np.array(
        [int(lag_for_feature[j]) >= memory_long_lag_cutoff for j in range(n_features)],
        dtype=float,
    )
    long_sum = float(np.dot(mean_abs, long_mask))
    jacobian_long_lag_mass = long_sum / max(total_mean_abs, 1e-9)

    # --- jacobian_centroid_norm (L2 norm of gradient at the data centroid) ---
    X_mean = X.mean(axis=0, keepdims=True)
    grad_at_mean = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        grad_expr = symbolic_diff(node, j)
        val = safe_eval(grad_expr, X_mean)
        grad_at_mean[j] = float(val[0])
    jacobian_centroid_norm = float(np.linalg.norm(grad_at_mean))

    # --- jacobian_nonlinearity_index ---
    eps = 1e-6
    nonlin_index = float(
        np.sum(stds / np.maximum(mean_abs, eps))
    )

    # --- jacobian_grad_variance_total ---
    grad_var_total = float(np.sum(stds ** 2))

    return {
        "jacobian_lag_mass": jacobian_lag_mass,
        "jacobian_long_lag_mass": jacobian_long_lag_mass,
        "jacobian_centroid_norm": jacobian_centroid_norm,
        "jacobian_nonlinearity_index": nonlin_index,
        "jacobian_grad_variance_total": grad_var_total,
    }


def _zero_jacobian_features() -> Dict[str, float]:
    return {
        "jacobian_lag_mass": 0.0,
        "jacobian_long_lag_mass": 0.0,
        "jacobian_centroid_norm": 0.0,
        "jacobian_nonlinearity_index": 0.0,
        "jacobian_grad_variance_total": 0.0,
    }


# ---------------------------------------------------------------------------
# Layer 3b - Hessian (second-order) concept features
# ---------------------------------------------------------------------------
#
# The Jacobian reads first-order behaviour. The Hessian d²f/dx_i dx_j separates
# the two structural signatures that the *syntactic* concepts confound (because
# PySR sprinkles spurious operators on linear windows, and an operator count
# cannot tell a genuine term from a disguised-linear one):
#   - off-diagonal (i != j) -> cross-lag MULTIPLICATIVE coupling. A real bilinear
#     term y_{t-k}y_{t-l} has a constant O(1) cross-partial; a disguised-linear
#     spurious product has tiny constants, so its cross-partial is suppressed by
#     those same constants. This reads the interaction functionally, where the
#     syntactic interaction_term_count is contaminated (it fires on the memory
#     scenario, which has no interaction, because deeper dynamics -> more complex
#     fits -> more spurious products).
#   - diagonal (i == i) -> single-lag CURVATURE (nonlinearity).
# Both summed as |·| and aggregated as the median over window rows, mirroring the
# Jacobian features' row aggregation.


def hessian_features(node: ExprNode, X: np.ndarray) -> Dict[str, float]:
    """Off-diagonal and diagonal Hessian magnitudes for one expression.

    Returns
    -------
    dict with keys:
      hessian_interaction
          Median over rows of ``sum_{i<j} |d²f/dx_i dx_j|`` -- cross-lag
          multiplicative coupling.
      hessian_curvature
          Median over rows of ``sum_i |d²f/dx_i²|`` -- single-lag curvature.
    """
    n_features = X.shape[1]
    if n_features == 0 or X.shape[0] == 0:
        return _zero_hessian_features()

    first = [symbolic_diff(node, i) for i in range(n_features)]
    off = np.zeros(X.shape[0], dtype=float)
    dia = np.zeros(X.shape[0], dtype=float)
    for i in range(n_features):
        dia += np.abs(safe_eval(symbolic_diff(first[i], i), X))
        for j in range(i + 1, n_features):
            off += np.abs(safe_eval(symbolic_diff(first[i], j), X))

    off = off[np.isfinite(off)]
    dia = dia[np.isfinite(dia)]
    return {
        "hessian_interaction": float(np.median(off)) if off.size else 0.0,
        "hessian_curvature": float(np.median(dia)) if dia.size else 0.0,
    }


def _zero_hessian_features() -> Dict[str, float]:
    return {"hessian_interaction": 0.0, "hessian_curvature": 0.0}


def representative_hessian_features(
    models: Sequence,
    X: np.ndarray,
) -> Dict[str, float]:
    """Hessian features of the per-window representative (best-fit) expression.

    NOT a median across the ensemble, unlike :func:`ensemble_jacobian_features`.
    The Hessian interaction signal is carried by a MINORITY of the ensemble: a
    near-linear window's set is mostly linear fits, so fewer than half of the
    expressions contain any cross-lag term, and a median across models collapses
    the off-diagonal magnitude to zero on almost every window. We therefore read
    the Hessian of the single lowest-MSE expression -- the same representative
    used for ``medoid_equation`` -- which carries the structure and reproduces the
    validated probe. Keys keep the ``_median`` suffix so the statistics layer
    (which tests every ``_median``/``_mean`` column) picks them up.
    """
    if not models or X.shape[0] == 0:
        return {f"{k}_median": 0.0 for k in _zero_hessian_features()}

    best = min(models, key=lambda m: float(getattr(m, "fit_mse", float("inf"))))
    try:
        feats = hessian_features(best.expression, X)
    except Exception:
        feats = _zero_hessian_features()
    return {f"{k}_median": float(feats[k]) for k in _zero_hessian_features()}


def ensemble_hessian_features(
    models: Sequence,
    X: np.ndarray,
) -> Dict[str, float]:
    """Aggregate the second-order (Hessian) concepts across a symbolic ensemble.

    Mirrors :func:`ensemble_jacobian_features` -- one value per model, aggregated
    over the ensemble -- so the second-order behavioural concepts sit on the same
    footing as the first-order ones rather than being read off a single best-fit
    representative.

    Aggregation is the ensemble MEAN, not the median. The off-diagonal Hessian is
    exactly zero for every additively-separable expression, and those dominate most
    windows' ensembles, so a median collapses the interaction signal to zero on
    almost every window. The mean reads prevalence x magnitude -- the minority of
    expressions that carry a genuine cross-lag term -- which is the same
    zero-inflation handling the gated index already uses. Keys carry the ``_mean``
    suffix so the statistics layer picks them up and the name states the aggregation
    directly.
    """
    if not models or X.shape[0] == 0:
        return {f"{k}_mean": 0.0 for k in _zero_hessian_features()}

    rows: List[Dict[str, float]] = []
    for model in models:
        try:
            feats = hessian_features(model.expression, X)
        except Exception:
            feats = _zero_hessian_features()
        rows.append(feats)

    keys = list(_zero_hessian_features().keys())
    return {f"{k}_mean": float(np.mean([float(r[k]) for r in rows])) for k in keys}


def ensemble_jacobian_features(
    models: Sequence,
    X: np.ndarray,
    lag_for_feature: Sequence[int],
    memory_long_lag_cutoff: int = 3,
) -> Dict[str, float]:
    """Aggregate Jacobian features across a symbolic ensemble.

    For each model in ``models``, computes ``jacobian_features`` and then
    returns the median across the ensemble for each feature key.

    Parameters
    ----------
    models:
        Sequence of ``SymbolicModel`` objects (each has an ``.expression``
        attribute that is an ``ExprNode``).
    X:
        Window design matrix used for gradient evaluation.
    lag_for_feature:
        Lag index (1-based) per column of X.
    memory_long_lag_cutoff:
        Threshold for long-memory lag classification.

    Returns
    -------
    dict with keys ``jacobian_*_median`` (one per feature from
    ``jacobian_features``).
    """
    if not models or X.shape[0] == 0:
        return {f"{k}_median": 0.0 for k in _zero_jacobian_features()}

    rows: List[Dict[str, float]] = []
    for model in models:
        try:
            feats = jacobian_features(
                model.expression,
                X,
                lag_for_feature,
                memory_long_lag_cutoff=memory_long_lag_cutoff,
            )
        except Exception:
            feats = _zero_jacobian_features()
        rows.append(feats)

    keys = list(_zero_jacobian_features().keys())
    result: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        result[f"{k}_median"] = float(np.median(vals))
    return result
