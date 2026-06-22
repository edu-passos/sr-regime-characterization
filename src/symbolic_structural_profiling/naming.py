"""Plain-language display names for the dual-layer concept-vector features.

Single source of truth consumed by the table and figure scripts (the paper
tables and captions are transcribed by hand).

Internal CSV column keys (e.g. ``jacobian_lag_mass_median``) are unchanged so
existing pipeline outputs still load.  This module only maps keys to labels
for presentation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-feature labels.  Keys match the column names emitted by features.py and
# jacobian.py; values are the plain-language labels used in the paper.
# ---------------------------------------------------------------------------

_BASE = {
    # Memory features (structural)
    "lag_mass": "Lag centroid",
    "long_lag_mass": "High lag order share",
    "lag_entropy": "Lag diversity",
    "active_lag_count": "Active lag count",
    # Interaction features
    "interaction_term_count": "Interaction term count",
    "interaction_order": "Highest interaction order",
    "interaction_depth": "Interaction depth",
    # Nonlinearity features (structural)
    "max_poly_degree_weighted": "Polynomial degree",
    "nonlinear_node_count": "Nonlinear operator count",
    "nonlinear_composition_depth": "Operator nesting depth",
    # Complexity features
    "complexity_cost": "Expression cost",
    "description_length": "Description length",
    # Behavioural-Jacobian features
    "jacobian_lag_mass": "Gradient magnitude",
    "jacobian_long_lag_mass": "Gradient high lag order share",
    "jacobian_centroid_norm": "Gradient sensitivity",
    "jacobian_nonlinearity_index": "Gradient variation index",
    "jacobian_grad_variance_total": "Total gradient variance",
    "jacobian_relative_nonlinearity_index": "Gated gradient variation",
    # Behavioural-Hessian features (off-diagonal = interaction curvature, diagonal = single-lag curvature)
    "hessian_interaction": "Interaction curvature",
    "hessian_curvature": "Single-lag curvature",
}


def _expand(base: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, label in base.items():
        out[key] = label
        out[f"{key}_median"] = f"{label} (median)"
        out[f"{key}_mean"] = f"{label} (mean)"
        out[f"{key}_iqr"] = f"{label} (IQR)"
    return out


FEATURE_DISPLAY_NAMES: dict[str, str] = _expand(_BASE)


# ---------------------------------------------------------------------------
# Group ordering used by the per-subject heatmap.  Order matches §2.2 / §2.3.
# Group separations show up as thin gaps between bands of features in plots.
# ---------------------------------------------------------------------------

FEATURE_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Memory",
        [
            "lag_mass_median",
            "lag_mass_mean",
            "long_lag_mass_median",
            "long_lag_mass_mean",
            "lag_entropy_median",
            "lag_entropy_mean",
            "active_lag_count_median",
            "active_lag_count_mean",
        ],
    ),
    (
        "Interaction",
        [
            "interaction_term_count_median",
            "interaction_order_median",
            "interaction_depth_median",
            "interaction_depth_mean",
        ],
    ),
    (
        "Nonlinearity",
        [
            "max_poly_degree_weighted_median",
            "nonlinear_node_count_median",
            "nonlinear_composition_depth_median",
        ],
    ),
    (
        "Complexity",
        [
            "complexity_cost_median",
            "description_length_median",
            "description_length_mean",
        ],
    ),
    (
        "Behavioural Jacobian",
        [
            "jacobian_lag_mass_median",
            "jacobian_long_lag_mass_median",
            "jacobian_centroid_norm_median",
            "jacobian_nonlinearity_index_median",
            "jacobian_grad_variance_total_median",
            "jacobian_relative_nonlinearity_index_median",
        ],
    ),
    (
        "Behavioural Hessian",
        [
            "hessian_interaction_median",
            "hessian_curvature_median",
        ],
    ),
]


def display_name(key: str) -> str:
    """Return the plain-language label for ``key``, or ``key`` itself if unknown."""
    return FEATURE_DISPLAY_NAMES.get(key, key)
