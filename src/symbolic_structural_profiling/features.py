from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .models.ensemble import SymbolicModel
from .models.expressions import (
    ExprNode,
    affine_calibrated_mse,
    audit_supported_structure,
    expression_depth,
    finite_difference_importance,
    interaction_stats,
    max_lag_used,
    max_nonlinear_composition_depth,
    nonlinear_node_count,
    node_count,
    operator_histogram,
    polynomial_degree_proxy,
    strip_affine_envelope,
)
from .windowing import FeatureMeta


@dataclass(slots=True)
class ExpressionFeatures:
    lag_mass: float
    long_lag_mass: float
    lag_entropy: float
    active_lag_count: float
    max_poly_degree_weighted: float
    nonlinear_node_count: float
    nonlinear_composition_depth: float
    interaction_term_count: float
    interaction_order: float
    interaction_depth: float
    complexity_cost: float
    description_length: float


STRUCTURE_ADVANTAGE_COLS = {
    "nonlinear_node_count",
    "nonlinear_composition_depth",
    "interaction_term_count",
    "interaction_order",
    "interaction_depth",
}

LAG_SUMMARY_COLS = (
    "lag_mass",
    "long_lag_mass",
    "lag_entropy",
    "active_lag_count",
)
NONLINEAR_SUMMARY_COLS = (
    "max_poly_degree_weighted",
    "nonlinear_node_count",
    "nonlinear_composition_depth",
)
INTERACTION_SUMMARY_COLS = (
    "interaction_term_count",
    "interaction_order",
    "interaction_depth",
)
COMPLEXITY_SUMMARY_COLS = (
    "complexity_cost",
    "description_length",
)


@dataclass(slots=True)
class _SummarySelectionState:
    feature_table: pd.DataFrame
    pre_guard_selected: pd.DataFrame
    selected: pd.DataFrame
    lag_pool: pd.DataFrame
    nonlinear_pool: pd.DataFrame
    interaction_pool: pd.DataFrame
    nonlinear_evidence_col: str
    interaction_evidence_col: str
    pre_guard_count: int
    post_guard_count: int
    adequate_simple_count: int
    best_simple_fit_ratio: float
    best_simple_fit_mse: float
    best_complex_fit_mse: float
    complex_gain_over_simple: float
    complex_gain_over_simple_threshold: float
    nonlinear_threshold: float
    interaction_threshold: float
    nonlinear_admission: Dict[str, float]
    interaction_admission: Dict[str, float]
    nonlinear_admission_debug: Dict[str, float]
    interaction_admission_debug: Dict[str, float]


@dataclass(slots=True)
class SummarySelectionTrace:
    feature_table: pd.DataFrame
    pre_guard_selected: pd.DataFrame
    selected: pd.DataFrame
    lag_pool: pd.DataFrame
    nonlinear_pool: pd.DataFrame
    interaction_pool: pd.DataFrame
    summary: Dict[str, float | str]


def _lag_importance(
    importances: np.ndarray,
    feature_meta: Sequence[FeatureMeta],
) -> Dict[int, float]:
    lag_scores: Dict[int, float] = {}
    for m in feature_meta:
        lag_scores[m.lag] = lag_scores.get(m.lag, 0.0) + float(importances[m.column])
    return lag_scores


def _lag_stats(
    lag_scores: Dict[int, float],
    threshold: float,
    long_lag_cutoff: int,
) -> tuple[float, float, float, float]:
    lags = np.array(sorted(lag_scores.keys()), dtype=float)
    scores = np.array([lag_scores[int(l)] for l in lags], dtype=float)

    total = float(scores.sum())
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    probs = scores / total
    lag_mass = float(np.sum(lags * probs))
    long_lag_mass = float(np.sum(probs[lags >= float(long_lag_cutoff)]))

    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero + 1e-12)))

    max_score = float(np.max(scores))
    active = float(np.sum(scores >= threshold * max_score))
    return lag_mass, long_lag_mass, entropy, active


def _complexity_cost(op_hist: Dict[str, int], weights: Dict[str, float]) -> float:
    total = 0.0
    for op, count in op_hist.items():
        total += float(weights.get(op, 1.0) * count)
    return total


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if vals.size == 0:
        return float("nan")

    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if float(w.sum()) <= 0.0:
        return float(np.quantile(vals, quantile))

    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cum = np.cumsum(w) / float(np.sum(w))
    idx = int(np.searchsorted(cum, quantile, side="left"))
    idx = min(max(idx, 0), vals.size - 1)
    return float(vals[idx])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if vals.size == 0:
        return float("nan")
    if float(w.sum()) <= 0.0:
        return float(np.mean(vals))
    return float(np.average(vals, weights=w))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return float("nan")
    mean = _weighted_mean(vals, weights)
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if float(w.sum()) <= 0.0:
        return float(np.std(vals))
    variance = float(np.average((vals - mean) ** 2, weights=w))
    return float(np.sqrt(max(variance, 0.0)))


def _rank_concept_pool(
    pool: pd.DataFrame,
    *,
    evidence_col: str,
    limit: int,
) -> pd.DataFrame:
    if pool.empty:
        return pool.copy()
    if evidence_col not in pool.columns:
        pool = pool.copy()
        pool[evidence_col] = 0.0
    ranked = pool.sort_values(
        [evidence_col, "fit_mse", "expression_complexity", "model_idx"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    return ranked.head(max(int(limit), 1)).copy()


def _concept_admission_threshold(
    linear_fit_mse: float,
    *,
    min_ratio: float,
    min_abs: float,
) -> float:
    if not np.isfinite(linear_fit_mse):
        return float(min_abs)
    return max(float(linear_fit_mse) * float(min_ratio), float(min_abs))


def _admit_concept_pool(
    feature_table: pd.DataFrame,
    *,
    concept_flag_col: str,
    evidence_col: str,
    fit_band_ratio: float,
    pool_size: int,
    linear_fit_mse: float,
    min_ratio: float,
    min_abs: float,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    empty = feature_table.iloc[0:0].copy()
    diagnostics: Dict[str, float] = {
        "candidate_count": 0.0,
        "pool_admitted": 0.0,
        "best_candidate_fit_mse": float("nan"),
        "best_comparator_fit_mse": float("nan"),
        "admission_gain": float("nan"),
        "admission_threshold": float(
            _concept_admission_threshold(
                linear_fit_mse,
                min_ratio=float(min_ratio),
                min_abs=float(min_abs),
            )
        ),
        "rejected_candidate_count": 0.0,
    }
    if feature_table.empty or concept_flag_col not in feature_table.columns:
        return empty, diagnostics

    concept_mask = feature_table[concept_flag_col].to_numpy(dtype=float) > 0.0
    candidates = feature_table.loc[concept_mask].copy()
    diagnostics["candidate_count"] = float(len(candidates))
    if candidates.empty:
        return empty, diagnostics

    best_candidate_fit = float(candidates["fit_mse"].min())
    diagnostics["best_candidate_fit_mse"] = best_candidate_fit
    threshold = float(diagnostics["admission_threshold"])

    comparator = feature_table.loc[~concept_mask].copy()
    admitted = False
    if comparator.empty:
        if evidence_col in candidates.columns:
            best_candidate = candidates.sort_values(
                ["fit_mse", "expression_complexity", "model_idx"],
                ascending=[True, True, True],
            ).iloc[0]
            admitted = float(best_candidate.get(evidence_col, 0.0)) > 0.0
    else:
        best_comparator_fit = float(comparator["fit_mse"].min())
        admission_gain = float(best_comparator_fit - best_candidate_fit)
        diagnostics["best_comparator_fit_mse"] = best_comparator_fit
        diagnostics["admission_gain"] = admission_gain
        admitted = admission_gain >= threshold - 1e-12

    if evidence_col not in candidates.columns:
        candidates = candidates.copy()
        candidates[evidence_col] = 0.0
    candidates = candidates.loc[candidates[evidence_col].to_numpy(dtype=float) > 0.0].copy()
    if candidates.empty or not admitted:
        diagnostics["rejected_candidate_count"] = float(diagnostics["candidate_count"])
        return empty, diagnostics

    band_threshold = best_candidate_fit * float(fit_band_ratio) + 1e-12
    banded = candidates.loc[candidates["fit_mse"].to_numpy(dtype=float) <= band_threshold].copy()
    ranked = _rank_concept_pool(
        banded,
        evidence_col=evidence_col,
        limit=int(pool_size),
    )
    if ranked.empty:
        diagnostics["rejected_candidate_count"] = float(diagnostics["candidate_count"])
        return empty, diagnostics

    diagnostics["pool_admitted"] = 1.0
    diagnostics["rejected_candidate_count"] = float(max(len(candidates) - len(ranked), 0))
    return ranked, diagnostics


def _best_gain_concept_pool(
    selected: pd.DataFrame,
    *,
    concept_flag_col: str,
    evidence_col: str,
    pool_size: int,
    threshold: float,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    empty = selected.iloc[0:0].copy()
    diagnostics: Dict[str, float] = {
        "candidate_count": 0.0,
        "pool_admitted": 0.0,
        "best_selected_gain": float("nan"),
        "threshold": float(threshold),
        "rejected_candidate_count": 0.0,
    }
    if selected.empty or concept_flag_col not in selected.columns:
        return empty, diagnostics

    concept_mask = selected[concept_flag_col].to_numpy(dtype=float) > 0.0
    candidates = selected.loc[concept_mask].copy()
    diagnostics["candidate_count"] = float(len(candidates))
    if candidates.empty:
        return empty, diagnostics

    if evidence_col not in candidates.columns:
        candidates = candidates.copy()
        candidates[evidence_col] = 0.0
    diagnostics["best_selected_gain"] = float(candidates[evidence_col].max())
    cleared = candidates.loc[candidates[evidence_col].to_numpy(dtype=float) >= float(threshold) - 1e-12].copy()
    if cleared.empty:
        diagnostics["rejected_candidate_count"] = float(len(candidates))
        return empty, diagnostics

    ranked = _rank_concept_pool(
        cleared,
        evidence_col=evidence_col,
        limit=int(pool_size),
    )
    if ranked.empty:
        diagnostics["rejected_candidate_count"] = float(len(candidates))
        return empty, diagnostics

    diagnostics["pool_admitted"] = 1.0
    diagnostics["rejected_candidate_count"] = float(max(len(candidates) - len(ranked), 0))
    return ranked, diagnostics


def _selection_evidence_cols(feature_table: pd.DataFrame) -> tuple[str, str]:
    nonlinear_evidence_col = (
        "unary_curvature_gain"
        if "unary_curvature_gain" in feature_table.columns
        else "unary_evidence_gain"
    )
    interaction_evidence_col = (
        "interaction_synergy_gain"
        if "interaction_synergy_gain" in feature_table.columns
        else "interaction_evidence_gain"
    )
    return nonlinear_evidence_col, interaction_evidence_col


def _summarize_pool_columns(
    summary: Dict[str, float],
    *,
    pool: pd.DataFrame,
    columns: Sequence[str],
    weights: np.ndarray | None,
    zero_if_empty: bool,
) -> None:
    if pool.empty:
        if zero_if_empty:
            for col in columns:
                summary[f"{col}_median"] = 0.0
                summary[f"{col}_mean"] = 0.0
                summary[f"{col}_iqr"] = 0.0
        return

    if weights is None:
        weights_arr = np.ones(len(pool), dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)

    if zero_if_empty and float(np.sum(np.maximum(weights_arr, 0.0))) <= 0.0:
        for col in columns:
            summary[f"{col}_median"] = 0.0
            summary[f"{col}_mean"] = 0.0
            summary[f"{col}_iqr"] = 0.0
        return

    for col in columns:
        if col not in pool.columns:
            continue
        values = pool[col].to_numpy(dtype=float)
        summary[f"{col}_median"] = _weighted_quantile(values, weights_arr, 0.50)
        summary[f"{col}_mean"] = _weighted_mean(values, weights_arr)
        q25 = _weighted_quantile(values, weights_arr, 0.25)
        q75 = _weighted_quantile(values, weights_arr, 0.75)
        summary[f"{col}_iqr"] = float(q75 - q25)


def _expression_features_from_expr(
    expr: object,
    X_reference: np.ndarray,
    feature_meta: Sequence[FeatureMeta],
    config: FeatureConfig,
) -> ExpressionFeatures:
    importances = finite_difference_importance(expr, X_reference, config.finite_diff_epsilon)
    lag_scores = _lag_importance(importances, feature_meta)
    lag_mass, long_lag_mass, lag_entropy, active_lag_count = _lag_stats(
        lag_scores,
        config.active_lag_threshold,
        int(config.memory_long_lag_cutoff),
    )

    poly_degree = polynomial_degree_proxy(expr)
    depth = expression_depth(expr)
    max_poly_degree_weighted = float(poly_degree / max(depth, 1))

    n_nonlin = float(nonlinear_node_count(expr))
    nonlin_comp = float(max_nonlinear_composition_depth(expr))

    interact_count, interact_order, interact_depth = interaction_stats(expr)

    op_hist = operator_histogram(expr)
    complexity = _complexity_cost(op_hist, config.complexity_weights)
    mdl_proxy = float(node_count(expr) + 0.5 * complexity)

    return ExpressionFeatures(
        lag_mass=lag_mass,
        long_lag_mass=long_lag_mass,
        lag_entropy=lag_entropy,
        active_lag_count=active_lag_count,
        max_poly_degree_weighted=max_poly_degree_weighted,
        nonlinear_node_count=n_nonlin,
        nonlinear_composition_depth=nonlin_comp,
        interaction_term_count=float(interact_count),
        interaction_order=float(interact_order),
        interaction_depth=float(interact_depth),
        complexity_cost=complexity,
        description_length=mdl_proxy,
    )


def select_summary_models(
    feature_table: pd.DataFrame,
    *,
    linear_fit_mse: float,
    fit_band_ratio: float,
    min_models: int = 2,
) -> pd.DataFrame:
    if feature_table.empty:
        return feature_table

    ordered = feature_table.sort_values(["fit_mse", "expression_complexity", "model_idx"]).reset_index(drop=True)
    best_fit = float(ordered["fit_mse"].min())
    candidates = [best_fit * float(fit_band_ratio)]
    if np.isfinite(linear_fit_mse):
        candidates.append(float(linear_fit_mse))
    threshold = float(min(candidates))
    selected = ordered[ordered["fit_mse"].astype(float) <= threshold + 1e-12].copy()

    minimum = min(int(max(min_models, 1)), int(len(ordered)))
    if len(selected) < minimum:
        selected = ordered.head(minimum).copy()
    return selected


def _rescued_nonlinear_mask(feature_table: pd.DataFrame) -> np.ndarray:
    required = {
        "raw_family",
        "strict_supported_family",
        "unary_support_path",
        "nonlinear_node_count",
        "has_interaction",
    }
    if feature_table.empty or not required.issubset(feature_table.columns):
        return np.zeros(len(feature_table), dtype=bool)
    raw_unary = feature_table["raw_family"].astype(str).to_numpy() == "unary"
    strict_not_unary = feature_table["strict_supported_family"].astype(str).to_numpy() != "unary"
    preserve_rescue = feature_table["unary_support_path"].astype(str).to_numpy() == "preserve_fraction"
    nonlinear = feature_table["nonlinear_node_count"].to_numpy(dtype=float) > 0.0
    no_interaction = feature_table["has_interaction"].to_numpy(dtype=float) <= 0.0
    return raw_unary & strict_not_unary & preserve_rescue & nonlinear & no_interaction


def _apply_rescued_row_fitband_override(
    feature_table: pd.DataFrame,
    pre_guard_selected: pd.DataFrame,
    *,
    enabled: bool,
    rescue_fitband_tol_ratio: float,
    rescue_fitband_tol_abs: float,
    rescue_max_extra_per_window: int,
    rescue_complexity_cap_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = feature_table.copy()
    working["rescued_nonlinear_eligible"] = _rescued_nonlinear_mask(working)
    working["added_by_rescue_fitband_override"] = False
    working["rescue_fitband_pass_reason"] = ""
    working["fit_gap_to_worst_selected"] = float("nan")
    working["fit_tolerance_used"] = float("nan")
    working["complexity_cap_passed"] = False

    selected = pre_guard_selected.copy()
    if (
        not enabled
        or working.empty
        or selected.empty
        or int(rescue_max_extra_per_window) <= 0
    ):
        return working, selected

    if rescue_complexity_cap_mode != "median_plus_2":
        raise ValueError(f"Unsupported rescue_complexity_cap_mode: {rescue_complexity_cap_mode}")

    best_selected_fit = float(selected["fit_mse"].min())
    worst_selected_fit = float(selected["fit_mse"].max())
    if not np.isfinite(best_selected_fit) or not np.isfinite(worst_selected_fit):
        return working, selected

    fit_tol = max(
        float(best_selected_fit) * float(rescue_fitband_tol_ratio),
        float(rescue_fitband_tol_abs),
    )
    complexity_cap = float(np.median(selected["expression_complexity"].to_numpy(dtype=float))) + 2.0

    candidate_mask = working["rescued_nonlinear_eligible"].to_numpy(dtype=bool) & (
        ~working.index.isin(selected.index)
    )
    if not np.any(candidate_mask):
        return working, selected

    working.loc[candidate_mask, "fit_gap_to_worst_selected"] = (
        working.loc[candidate_mask, "fit_mse"].to_numpy(dtype=float) - float(worst_selected_fit)
    )
    working.loc[candidate_mask, "fit_tolerance_used"] = float(fit_tol)
    complexity_pass = (
        working.loc[candidate_mask, "expression_complexity"].to_numpy(dtype=float)
        <= float(complexity_cap) + 1e-12
    )
    working.loc[candidate_mask, "complexity_cap_passed"] = complexity_pass

    admissible = working.loc[candidate_mask].copy()
    admissible = admissible.loc[
        admissible["fit_mse"].to_numpy(dtype=float) <= float(worst_selected_fit + fit_tol) + 1e-12
    ].copy()
    admissible = admissible.loc[
        admissible["expression_complexity"].to_numpy(dtype=float) <= float(complexity_cap) + 1e-12
    ].copy()
    if admissible.empty:
        return working, selected

    ranked = admissible.sort_values(
        ["fit_mse", "nonlinear_node_count", "expression_complexity", "model_idx"],
        ascending=[True, False, True, True],
    )
    added = ranked.head(int(rescue_max_extra_per_window)).copy()
    working.loc[added.index, "added_by_rescue_fitband_override"] = True
    working.loc[added.index, "rescue_fitband_pass_reason"] = "within_fit_and_complexity_cap"

    selected = pd.concat([selected, working.loc[added.index]], axis=0).sort_values(
        ["fit_mse", "expression_complexity", "model_idx"],
        ascending=[True, True, True],
    )
    return working, selected.copy()


def _guard_failure_reason(
    *,
    removed: bool,
    ratio_component: float,
    abs_component: float,
) -> str:
    if not removed:
        return ""
    if not np.isfinite(ratio_component) or not np.isfinite(abs_component):
        return "other"
    if np.isclose(ratio_component, abs_component, rtol=1e-9, atol=1e-12):
        return "mixed_guard_failure"
    if ratio_component > abs_component:
        return "linear_advantage_below_ratio_threshold"
    if abs_component > ratio_component:
        return "linear_advantage_below_abs_threshold"
    return "other"


def _apply_rescued_row_softened_guard(
    selected: pd.DataFrame,
    *,
    current_keep_mask: np.ndarray,
    linear_advantages: np.ndarray,
    threshold: float,
    enabled: bool,
    softened_guard_margin_ratio: float,
    softened_guard_min_unary_preservation_ratio: float,
) -> np.ndarray:
    saved_mask = np.zeros(len(selected), dtype=bool)
    if not enabled or selected.empty or not np.isfinite(threshold) or threshold <= 0.0:
        return saved_mask

    required = {
        "raw_family",
        "strict_supported_family",
        "unary_support_path",
        "nonlinear_node_count",
        "has_interaction",
        "added_by_rescue_fitband_override",
        "unary_preservation_ratio",
    }
    if not required.issubset(selected.columns):
        return saved_mask

    raw_unary = selected["raw_family"].astype(str).to_numpy() == "unary"
    strict_not_unary = selected["strict_supported_family"].astype(str).to_numpy() != "unary"
    preserve_rescue = selected["unary_support_path"].astype(str).to_numpy() == "preserve_fraction"
    nonlinear = selected["nonlinear_node_count"].to_numpy(dtype=float) > 0.0
    no_interaction = selected["has_interaction"].to_numpy(dtype=float) <= 0.0
    added_by_rescue = selected["added_by_rescue_fitband_override"].astype(bool).to_numpy()
    high_preservation = (
        selected["unary_preservation_ratio"].to_numpy(dtype=float)
        >= float(softened_guard_min_unary_preservation_ratio) - 1e-12
    )
    shortfall = np.maximum(float(threshold) - linear_advantages, 0.0)
    within_margin = shortfall <= float(softened_guard_margin_ratio) * float(threshold) + 1e-12

    eligible = (
        (~current_keep_mask)
        & raw_unary
        & strict_not_unary
        & preserve_rescue
        & nonlinear
        & no_interaction
        & added_by_rescue
        & high_preservation
        & within_margin
    )
    saved_mask[eligible] = True
    return saved_mask


def _apply_complex_model_guard(
    selected: pd.DataFrame,
    *,
    linear_fit_mse: float,
    enabled: bool,
    linear_advantage_min_ratio: float,
    linear_advantage_min_abs: float,
    softened_guard_enabled: bool = False,
    softened_guard_margin_ratio: float = 0.10,
    softened_guard_min_unary_preservation_ratio: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    pre_guard_count = int(len(selected))
    annotated = selected.copy()
    annotated["guard_considered"] = False
    annotated["guard_removed"] = False
    annotated["guard_linear_advantage"] = float("nan")
    annotated["guard_threshold"] = float("nan")
    annotated["guard_ratio_threshold_component"] = float("nan")
    annotated["guard_abs_threshold_component"] = float("nan")
    annotated["guard_threshold_shortfall"] = float("nan")
    annotated["guard_failure_reason"] = ""
    annotated["softened_guard_saved"] = False
    if (
        annotated.empty
        or not enabled
        or not np.isfinite(linear_fit_mse)
    ):
        return annotated, annotated, pre_guard_count, pre_guard_count

    unary_flags = (
        annotated["has_unary_nonlinearity"].to_numpy(dtype=float)
        if "has_unary_nonlinearity" in annotated.columns
        else np.zeros(len(annotated), dtype=float)
    )
    interaction_flags = (
        annotated["has_interaction"].to_numpy(dtype=float)
        if "has_interaction" in annotated.columns
        else np.zeros(len(annotated), dtype=float)
    )
    complex_mask = (unary_flags > 0.0) | (interaction_flags > 0.0)
    if not np.any(complex_mask):
        return annotated, annotated, pre_guard_count, pre_guard_count

    ratio_component = float(linear_fit_mse) * float(linear_advantage_min_ratio)
    abs_component = float(linear_advantage_min_abs)
    threshold = max(float(ratio_component), float(abs_component))
    advantages = float(linear_fit_mse) - annotated["fit_mse"].to_numpy(dtype=float)
    current_keep_mask = (~complex_mask) | (advantages >= threshold - 1e-12)
    removed_mask = complex_mask & (~current_keep_mask)

    annotated.loc[complex_mask, "guard_considered"] = True
    annotated.loc[complex_mask, "guard_linear_advantage"] = advantages[complex_mask]
    annotated.loc[complex_mask, "guard_threshold"] = float(threshold)
    annotated.loc[complex_mask, "guard_ratio_threshold_component"] = float(ratio_component)
    annotated.loc[complex_mask, "guard_abs_threshold_component"] = float(abs_component)
    annotated.loc[complex_mask, "guard_threshold_shortfall"] = np.maximum(
        float(threshold) - advantages[complex_mask],
        0.0,
    )
    annotated.loc[removed_mask, "guard_removed"] = True
    if np.any(removed_mask):
        failure_reason = _guard_failure_reason(
            removed=True,
            ratio_component=float(ratio_component),
            abs_component=float(abs_component),
        )
        annotated.loc[removed_mask, "guard_failure_reason"] = failure_reason

    softened_saved_mask = _apply_rescued_row_softened_guard(
        annotated,
        current_keep_mask=current_keep_mask,
        linear_advantages=advantages,
        threshold=float(threshold),
        enabled=bool(softened_guard_enabled),
        softened_guard_margin_ratio=float(softened_guard_margin_ratio),
        softened_guard_min_unary_preservation_ratio=float(
            softened_guard_min_unary_preservation_ratio
        ),
    )
    annotated.loc[softened_saved_mask, "softened_guard_saved"] = True
    final_keep_mask = current_keep_mask | softened_saved_mask
    guarded = annotated.loc[final_keep_mask].copy()
    if guarded.empty:
        annotated["guard_removed"] = False
        annotated["guard_failure_reason"] = ""
        return annotated, annotated, pre_guard_count, pre_guard_count
    return guarded, annotated, pre_guard_count, int(len(guarded))


def _simple_model_mask(feature_table: pd.DataFrame) -> np.ndarray:
    if "is_simple_model" in feature_table.columns:
        return feature_table["is_simple_model"].to_numpy(dtype=float) > 0.0
    unary_flags = (
        feature_table["has_unary_nonlinearity"].to_numpy(dtype=float)
        if "has_unary_nonlinearity" in feature_table.columns
        else np.zeros(len(feature_table), dtype=float)
    )
    interaction_flags = (
        feature_table["has_interaction"].to_numpy(dtype=float)
        if "has_interaction" in feature_table.columns
        else np.zeros(len(feature_table), dtype=float)
    )
    return (unary_flags <= 0.0) & (interaction_flags <= 0.0)


def _select_best_simple_model(simple_models: pd.DataFrame) -> pd.DataFrame:
    ordered = simple_models.sort_values(
        ["expression_complexity", "fit_mse", "model_idx"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return ordered.head(1).copy()


def _simple_model_fallback_stats(
    feature_table: pd.DataFrame,
    *,
    best_selected_fit: float,
    simple_fallback_ratio: float,
) -> tuple[pd.DataFrame, int, float, float]:
    simple_mask = _simple_model_mask(feature_table)
    if not np.any(simple_mask):
        return feature_table.iloc[0:0].copy(), 0, float("nan"), float("nan")

    simple_models = feature_table.loc[simple_mask].copy()
    best_simple_fit = float(simple_models["fit_mse"].min())
    fit_ratio = (
        float(best_simple_fit / best_selected_fit)
        if np.isfinite(best_selected_fit) and best_selected_fit > 1e-12
        else float("nan")
    )
    adequate_mask = simple_models["fit_mse"].to_numpy(dtype=float) <= (
        best_selected_fit * float(simple_fallback_ratio) + 1e-12
    )
    adequate_simple_models = simple_models.loc[adequate_mask].copy()
    return adequate_simple_models, int(len(adequate_simple_models)), fit_ratio, best_simple_fit


def _best_simple_candidate(
    feature_table: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    simple_fallback_ratio: float,
) -> tuple[pd.DataFrame, int, float, float]:
    if selected.empty:
        return feature_table.iloc[0:0].copy(), 0, float("nan"), float("nan")

    best_selected_fit = float(selected["fit_mse"].min())
    adequate_simple_models, adequate_count, fit_ratio, best_simple_fit = _simple_model_fallback_stats(
        feature_table,
        best_selected_fit=best_selected_fit,
        simple_fallback_ratio=float(simple_fallback_ratio),
    )
    selected_simple = selected.loc[_simple_model_mask(selected)].copy()
    if not selected_simple.empty:
        return _select_best_simple_model(selected_simple), adequate_count, fit_ratio, best_simple_fit
    if not adequate_simple_models.empty:
        return _select_best_simple_model(adequate_simple_models), adequate_count, fit_ratio, best_simple_fit
    return feature_table.iloc[0:0].copy(), adequate_count, fit_ratio, best_simple_fit


def _complex_candidate_stats(selected: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if selected.empty:
        return selected, float("nan")
    complex_selected = selected.loc[~_simple_model_mask(selected)].copy()
    if complex_selected.empty:
        return complex_selected, float("nan")
    best_complex_fit = float(complex_selected["fit_mse"].min())
    return complex_selected, best_complex_fit


def _complex_over_simple_threshold(
    linear_fit_mse: float,
    *,
    complex_over_simple_min_ratio: float,
    complex_over_simple_min_abs: float,
) -> float:
    if not np.isfinite(linear_fit_mse):
        return float(complex_over_simple_min_abs)
    return max(
        float(linear_fit_mse) * float(complex_over_simple_min_ratio),
        float(complex_over_simple_min_abs),
    )


def _apply_simple_model_preference(
    feature_table: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    enabled: bool,
    simple_fallback_ratio: float,
    linear_fit_mse: float,
    complex_over_simple_min_ratio: float,
    complex_over_simple_min_abs: float,
) -> tuple[pd.DataFrame, str, int, float, float, float, float, float]:
    if selected.empty or not enabled:
        return selected, "fit_band", 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    simple_candidate, adequate_count, fit_ratio, best_simple_fit = _best_simple_candidate(
        feature_table,
        selected,
        simple_fallback_ratio=float(simple_fallback_ratio),
    )
    complex_selected, best_complex_fit = _complex_candidate_stats(selected)
    threshold = _complex_over_simple_threshold(
        float(linear_fit_mse),
        complex_over_simple_min_ratio=float(complex_over_simple_min_ratio),
        complex_over_simple_min_abs=float(complex_over_simple_min_abs),
    )
    complex_gain = (
        float(best_simple_fit - best_complex_fit)
        if np.isfinite(best_simple_fit) and np.isfinite(best_complex_fit)
        else float("nan")
    )

    if simple_candidate.empty:
        return (
            selected,
            "complex_guard_fallback",
            adequate_count,
            fit_ratio,
            best_simple_fit,
            best_complex_fit,
            complex_gain,
            threshold,
        )

    if complex_selected.empty:
        simple_mask = _simple_model_mask(selected)
        chosen = selected.loc[simple_mask].copy() if np.any(simple_mask) else simple_candidate
        if chosen.empty:
            chosen = simple_candidate
        return (
            chosen,
            "fit_band",
            adequate_count,
            fit_ratio,
            float(simple_candidate["fit_mse"].iloc[0]),
            best_complex_fit,
            float("nan"),
            threshold,
        )

    chosen_simple_fit = float(simple_candidate["fit_mse"].iloc[0])
    complex_gain = float(chosen_simple_fit - best_complex_fit)
    if complex_gain >= threshold - 1e-12:
        return (
            selected,
            "complex_preserved",
            adequate_count,
            fit_ratio,
            chosen_simple_fit,
            best_complex_fit,
            complex_gain,
            threshold,
        )
    return (
        simple_candidate,
        "simple_fallback",
        adequate_count,
        fit_ratio,
        chosen_simple_fit,
        best_complex_fit,
        complex_gain,
        threshold,
    )


def extract_expression_features(
    model: SymbolicModel,
    X_reference: np.ndarray,
    feature_meta: Sequence[FeatureMeta],
    config: FeatureConfig,
) -> ExpressionFeatures:
    expr = strip_affine_envelope(model.expression)
    return _expression_features_from_expr(expr, X_reference, feature_meta, config)


def _replace_long_lag_refs(node: ExprNode, *, long_lag_cutoff: int) -> ExprNode:
    if node.op == "var":
        if node.lag is not None and int(node.lag) >= int(long_lag_cutoff):
            return ExprNode(op="const", value=0.0)
        return node
    if node.op == "const":
        return node
    children = tuple(
        _replace_long_lag_refs(child, long_lag_cutoff=long_lag_cutoff)
        for child in node.children
    )
    return ExprNode(
        op=node.op,
        children=children,
        feature_index=node.feature_index,
        variable=node.variable,
        lag=node.lag,
        value=node.value,
    )


def _memory_reduced_surrogate(node: ExprNode, *, long_lag_cutoff: int) -> ExprNode:
    return strip_affine_envelope(
        _replace_long_lag_refs(node, long_lag_cutoff=max(int(long_lag_cutoff), 1))
    )


def _need_delta(raw_delta: float) -> float:
    if not np.isfinite(float(raw_delta)):
        return 0.0
    return float(max(float(raw_delta), 0.0))


def _compute_need_delta_rows(
    *,
    audit,
    full_fit_mse: float,
    X_reference: np.ndarray | None,
    y_reference: np.ndarray | None,
    long_lag_cutoff: int,
) -> Dict[str, float]:
    unary_raw = float(audit.simple_surrogate_fit_mse - audit.single_lag_unary_surrogate_fit_mse)
    interaction_raw = float(audit.additive_noninteraction_surrogate_fit_mse - full_fit_mse)

    if X_reference is None or y_reference is None:
        return {
            "delta_unary_need_raw": float(unary_raw),
            "delta_unary_need": _need_delta(unary_raw),
            "delta_interaction_need_raw": float(interaction_raw),
            "delta_interaction_need": _need_delta(interaction_raw),
            "memory_reduced_fit_mse": float("nan"),
            "delta_memory_need_raw": float("nan"),
            "delta_memory_need": 0.0,
            "complexity_surrogate_fit_mse": float("nan"),
            "delta_complexity_need_raw": float("nan"),
            "delta_complexity_need": 0.0,
        }

    base_expr = audit.supported_expr_before_final_strip
    memory_surrogate_expr = _memory_reduced_surrogate(
        base_expr,
        long_lag_cutoff=int(long_lag_cutoff),
    )
    memory_fit = float(affine_calibrated_mse(memory_surrogate_expr, X_reference, y_reference))
    memory_raw = float(memory_fit - full_fit_mse)

    candidate_exprs = []
    for candidate in (
        audit.single_lag_unary_expr,
        audit.additive_noninteraction_expr,
        audit.simple_surrogate_expr,
        memory_surrogate_expr,
    ):
        simplified = strip_affine_envelope(candidate)
        if node_count(simplified) < node_count(audit.core_expr):
            candidate_exprs.append(simplified)

    best_complexity_fit = float("nan")
    if candidate_exprs:
        best_complexity_fit = min(
            float(affine_calibrated_mse(candidate, X_reference, y_reference))
            for candidate in candidate_exprs
        )
    complexity_raw = (
        float(best_complexity_fit - full_fit_mse)
        if np.isfinite(best_complexity_fit)
        else float("nan")
    )

    return {
        "delta_unary_need_raw": float(unary_raw),
        "delta_unary_need": _need_delta(unary_raw),
        "delta_interaction_need_raw": float(interaction_raw),
        "delta_interaction_need": _need_delta(interaction_raw),
        "memory_reduced_fit_mse": float(memory_fit),
        "delta_memory_need_raw": float(memory_raw),
        "delta_memory_need": _need_delta(memory_raw),
        "complexity_surrogate_fit_mse": float(best_complexity_fit),
        "delta_complexity_need_raw": float(complexity_raw),
        "delta_complexity_need": _need_delta(complexity_raw),
    }


def expression_feature_table(
    models: Sequence[SymbolicModel],
    X_reference: np.ndarray,
    feature_meta: Sequence[FeatureMeta],
    config: FeatureConfig,
    y_reference: np.ndarray | None = None,
    linear_fit_mse: float | None = None,
    support_semantics_kwargs: dict[str, object] | None = None,
    source_model_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    support_kwargs = dict(support_semantics_kwargs or {})
    if source_model_ids is not None and len(source_model_ids) != len(models):
        raise ValueError("source_model_ids must match models length")
    for idx, model in enumerate(models):
        support, audit = audit_supported_structure(
            model.expression,
            X=X_reference,
            y=y_reference,
            linear_fit_mse=linear_fit_mse,
            enabled=bool(config.counterfactual_structure_evidence),
            unary_min_ratio=float(config.unary_structure_evidence_min_ratio),
            unary_min_abs=float(config.unary_structure_evidence_min_abs),
            interaction_min_ratio=float(config.interaction_structure_evidence_min_ratio),
            interaction_min_abs=float(config.interaction_structure_evidence_min_abs),
            **support_kwargs,
        )
        features = _expression_features_from_expr(support.supported_expr, X_reference, feature_meta, config)
        row: Dict[str, float | int | str] = asdict(features)
        row["model_idx"] = idx
        if source_model_ids is not None:
            row["source_model_id"] = str(source_model_ids[idx])
        row["seed"] = model.seed
        row["fit_mse"] = model.fit_mse
        row["expression_complexity"] = model.complexity
        row["equation"] = model.equation
        row["max_lag_used"] = float(max_lag_used(support.core_expr))
        row["raw_has_nonlinearity"] = float(support.raw_has_nonlinearity)
        row["raw_has_unary_nonlinearity"] = float(support.raw_has_unary_nonlinearity)
        row["raw_has_interaction"] = float(support.raw_has_interaction)
        row["raw_has_coupled_interaction"] = float(support.raw_has_coupled_interaction)
        row["raw_is_simple_model"] = float(support.raw_is_simple_model)
        row["raw_family"] = str(support.raw_family)
        row["has_nonlinearity"] = float(support.has_nonlinearity)
        row["has_unary_nonlinearity"] = float(support.has_unary_nonlinearity)
        row["has_interaction"] = float(support.has_interaction)
        row["has_coupled_interaction"] = float(support.has_coupled_interaction)
        row["is_simple_model"] = float(support.is_simple_model)
        row["family"] = str(support.family)
        row["full_fit_mse"] = float(support.full_fit_mse)
        row["simple_surrogate_fit_mse"] = float(support.simple_surrogate_fit_mse)
        row["single_lag_unary_surrogate_fit_mse"] = float(support.single_lag_unary_surrogate_fit_mse)
        row["additive_noninteraction_surrogate_fit_mse"] = float(support.additive_noninteraction_surrogate_fit_mse)
        row["unary_pruned_fit_mse"] = float(support.unary_pruned_fit_mse)
        row["interaction_pruned_fit_mse"] = float(support.interaction_pruned_fit_mse)
        row["unary_curvature_gain"] = float(support.unary_curvature_gain)
        row["interaction_synergy_gain"] = float(support.interaction_synergy_gain)
        row["unary_evidence_gain"] = float(support.unary_evidence_gain)
        row["interaction_evidence_gain"] = float(support.interaction_evidence_gain)
        row["has_supported_unary_nonlinearity"] = float(support.has_unary_nonlinearity)
        row["has_supported_interaction"] = float(support.has_interaction)
        row.update(
            _compute_need_delta_rows(
                audit=audit,
                full_fit_mse=float(support.full_fit_mse),
                X_reference=X_reference,
                y_reference=y_reference,
                long_lag_cutoff=int(config.memory_long_lag_cutoff),
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _supported_family_mask(feature_table: pd.DataFrame, family: str) -> np.ndarray:
    if feature_table.empty:
        return np.zeros(0, dtype=bool)
    if "family" in feature_table.columns:
        return feature_table["family"].astype(str).to_numpy() == str(family)
    if family == "simple":
        return _simple_model_mask(feature_table)
    if family == "unary":
        return (
            feature_table["has_unary_nonlinearity"].to_numpy(dtype=float) > 0.0
            if "has_unary_nonlinearity" in feature_table.columns
            else np.zeros(len(feature_table), dtype=bool)
        )
    if family == "interaction":
        return (
            feature_table["has_interaction"].to_numpy(dtype=float) > 0.0
            if "has_interaction" in feature_table.columns
            else np.zeros(len(feature_table), dtype=bool)
        )
    raise ValueError(f"Unsupported family: {family}")


def _family_best_evidence(
    family_table: pd.DataFrame,
    *,
    family: str,
    unary_evidence_col: str,
    interaction_evidence_col: str,
) -> float:
    if family_table.empty:
        return float("nan")
    if family == "simple":
        return 0.0
    evidence_col = unary_evidence_col if family == "unary" else interaction_evidence_col
    if evidence_col not in family_table.columns:
        return float("nan")
    return float(family_table[evidence_col].max())


def _family_stage_stats(
    feature_table: pd.DataFrame,
    *,
    stage: str,
    unary_evidence_col: str,
    interaction_evidence_col: str,
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for family in ("simple", "unary", "interaction"):
        mask = _supported_family_mask(feature_table, family)
        subset = feature_table.loc[mask].copy() if len(mask) > 0 else feature_table.iloc[0:0].copy()
        stats[f"family_trace_{stage}_{family}_count"] = float(len(subset))
        stats[f"family_trace_{stage}_{family}_best_fit_mse"] = (
            float(subset["fit_mse"].min()) if not subset.empty else float("nan")
        )
        stats[f"family_trace_{stage}_{family}_best_evidence"] = _family_best_evidence(
            subset,
            family=family,
            unary_evidence_col=unary_evidence_col,
            interaction_evidence_col=interaction_evidence_col,
        )
    return stats


def _family_stage_subset(feature_table: pd.DataFrame, family: str) -> pd.DataFrame:
    if feature_table.empty:
        return feature_table.iloc[0:0].copy()
    mask = _supported_family_mask(feature_table, family)
    return feature_table.loc[mask].copy()


def _family_separated_summary(
    feature_table: pd.DataFrame,
    *,
    family: str,
    evidence_col: str | None,
) -> Dict[str, float]:
    subset = _family_stage_subset(feature_table, family)
    summary: Dict[str, float] = {
        f"family_sep_{family}_model_count": float(len(subset)),
        f"family_sep_{family}_best_fit_mse": float(subset["fit_mse"].min()) if not subset.empty else float("nan"),
    }
    if subset.empty:
        if family == "unary":
            summary["family_sep_unary_nonlinear_node_count_median"] = 0.0
            summary["family_sep_unary_nonlinear_composition_depth_median"] = 0.0
        elif family == "interaction":
            summary["family_sep_interaction_term_count_median"] = 0.0
            summary["family_sep_interaction_order_median"] = 0.0
        return summary

    if family == "simple":
        return summary

    if evidence_col is not None and evidence_col in subset.columns:
        weights = (
            np.maximum(subset[evidence_col].to_numpy(dtype=float), 0.0)
            / np.maximum(subset["fit_mse"].to_numpy(dtype=float), 1e-6)
        )
    else:
        weights = np.ones(len(subset), dtype=float)

    if family == "unary":
        local_summary: Dict[str, float] = {}
        _summarize_pool_columns(
            local_summary,
            pool=subset,
            columns=("nonlinear_node_count", "nonlinear_composition_depth"),
            weights=weights,
            zero_if_empty=True,
        )
        summary["family_sep_unary_nonlinear_node_count_median"] = float(
            local_summary.get("nonlinear_node_count_median", 0.0)
        )
        summary["family_sep_unary_nonlinear_composition_depth_median"] = float(
            local_summary.get("nonlinear_composition_depth_median", 0.0)
        )
    elif family == "interaction":
        local_summary = {}
        _summarize_pool_columns(
            local_summary,
            pool=subset,
            columns=("interaction_term_count", "interaction_order"),
            weights=weights,
            zero_if_empty=True,
        )
        summary["family_sep_interaction_term_count_median"] = float(
            local_summary.get("interaction_term_count_median", 0.0)
        )
        summary["family_sep_interaction_order_median"] = float(
            local_summary.get("interaction_order_median", 0.0)
        )
    return summary


def _delta_or_nan(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float(left - right)


def _current_logic_decision_code(
    *,
    post_guard_simple_count: int,
    nonlinear_pool_count: int,
    interaction_pool_count: int,
) -> str:
    if nonlinear_pool_count > 0 and interaction_pool_count > 0:
        return "mixed_current_logic"
    if nonlinear_pool_count > 0:
        return "unary_only"
    if interaction_pool_count > 0:
        return "interaction_only"
    if post_guard_simple_count > 0:
        return "simple_only"
    return "no_family_with_evidence"


def _build_summary_selection_state(
    feature_table: pd.DataFrame,
    *,
    linear_fit_mse: float | None,
    fit_band_ratio: float,
    guard_complex_models: bool,
    simple_model_fallback_ratio: float,
    concept_pool_size: int,
    concept_pool_fit_band_ratio: float,
    nonlinear_pool_min_ratio: float,
    nonlinear_pool_min_abs: float,
    interaction_pool_min_ratio: float,
    interaction_pool_min_abs: float,
    complex_over_simple_min_ratio: float,
    complex_over_simple_min_abs: float,
    linear_advantage_min_ratio: float,
    linear_advantage_min_abs: float,
    rescue_fitband_enabled: bool = False,
    rescue_fitband_tol_ratio: float = 0.02,
    rescue_fitband_tol_abs: float = 1e-4,
    rescue_max_extra_per_window: int = 1,
    rescue_complexity_cap_mode: str = "median_plus_2",
    softened_guard_enabled: bool = False,
    softened_guard_margin_ratio: float = 0.10,
    softened_guard_min_unary_preservation_ratio: float = 0.50,
) -> _SummarySelectionState:
    reference_linear_fit = float(linear_fit_mse) if linear_fit_mse is not None else float("inf")
    working_feature_table = feature_table.copy()
    pre_guard_selected = select_summary_models(
        feature_table,
        linear_fit_mse=reference_linear_fit,
        fit_band_ratio=float(fit_band_ratio),
    )
    working_feature_table, pre_guard_selected = _apply_rescued_row_fitband_override(
        working_feature_table,
        pre_guard_selected,
        enabled=bool(rescue_fitband_enabled),
        rescue_fitband_tol_ratio=float(rescue_fitband_tol_ratio),
        rescue_fitband_tol_abs=float(rescue_fitband_tol_abs),
        rescue_max_extra_per_window=int(rescue_max_extra_per_window),
        rescue_complexity_cap_mode=str(rescue_complexity_cap_mode),
    )
    pre_guard_count = int(len(pre_guard_selected))
    selected, guard_audit_selected, _, post_guard_count = _apply_complex_model_guard(
        pre_guard_selected,
        linear_fit_mse=reference_linear_fit,
        enabled=bool(guard_complex_models),
        linear_advantage_min_ratio=float(linear_advantage_min_ratio),
        linear_advantage_min_abs=float(linear_advantage_min_abs),
        softened_guard_enabled=bool(softened_guard_enabled),
        softened_guard_margin_ratio=float(softened_guard_margin_ratio),
        softened_guard_min_unary_preservation_ratio=float(
            softened_guard_min_unary_preservation_ratio
        ),
    )
    guard_audit_cols = [
        "guard_considered",
        "guard_removed",
        "guard_linear_advantage",
        "guard_threshold",
        "guard_ratio_threshold_component",
        "guard_abs_threshold_component",
        "guard_threshold_shortfall",
        "guard_failure_reason",
        "softened_guard_saved",
    ]
    for col in guard_audit_cols:
        if col not in working_feature_table.columns:
            if col in {"guard_considered", "guard_removed", "softened_guard_saved"}:
                working_feature_table[col] = False
            elif col == "guard_failure_reason":
                working_feature_table[col] = ""
            else:
                working_feature_table[col] = float("nan")
    working_feature_table.loc[guard_audit_selected.index, guard_audit_cols] = guard_audit_selected[
        guard_audit_cols
    ]
    best_selected_fit = float(selected["fit_mse"].min())
    adequate_simple_models, adequate_simple_count, best_simple_fit_ratio, best_simple_fit_mse = (
        _simple_model_fallback_stats(
            feature_table,
            best_selected_fit=best_selected_fit,
            simple_fallback_ratio=float(simple_model_fallback_ratio),
        )
    )
    simple_candidate = (
        _select_best_simple_model(adequate_simple_models)
        if not adequate_simple_models.empty
        else feature_table.iloc[0:0].copy()
    )
    best_complex_fit_mse = float("nan")
    complex_gain_over_simple = float("nan")
    if np.any(~_simple_model_mask(selected)):
        best_complex_fit_mse = float(selected.loc[~_simple_model_mask(selected), "fit_mse"].min())
    if not simple_candidate.empty and np.isfinite(best_complex_fit_mse):
        complex_gain_over_simple = float(float(simple_candidate["fit_mse"].iloc[0]) - best_complex_fit_mse)
    complex_gain_over_simple_threshold = _complex_over_simple_threshold(
        reference_linear_fit,
        complex_over_simple_min_ratio=float(complex_over_simple_min_ratio),
        complex_over_simple_min_abs=float(complex_over_simple_min_abs),
    )

    nonlinear_evidence_col, interaction_evidence_col = _selection_evidence_cols(feature_table)
    _, nonlinear_admission_debug = _admit_concept_pool(
        feature_table,
        concept_flag_col="has_unary_nonlinearity",
        evidence_col=nonlinear_evidence_col,
        fit_band_ratio=float(concept_pool_fit_band_ratio),
        pool_size=int(concept_pool_size),
        linear_fit_mse=reference_linear_fit,
        min_ratio=float(nonlinear_pool_min_ratio),
        min_abs=float(nonlinear_pool_min_abs),
    )
    _, interaction_admission_debug = _admit_concept_pool(
        feature_table,
        concept_flag_col="has_interaction",
        evidence_col=interaction_evidence_col,
        fit_band_ratio=float(concept_pool_fit_band_ratio),
        pool_size=int(concept_pool_size),
        linear_fit_mse=reference_linear_fit,
        min_ratio=float(interaction_pool_min_ratio),
        min_abs=float(interaction_pool_min_abs),
    )
    nonlinear_threshold = _concept_admission_threshold(
        reference_linear_fit,
        min_ratio=float(nonlinear_pool_min_ratio),
        min_abs=float(nonlinear_pool_min_abs),
    )
    interaction_threshold = _concept_admission_threshold(
        reference_linear_fit,
        min_ratio=float(interaction_pool_min_ratio),
        min_abs=float(interaction_pool_min_abs),
    )
    nonlinear_pool, nonlinear_admission = _best_gain_concept_pool(
        selected,
        concept_flag_col="has_unary_nonlinearity",
        evidence_col=nonlinear_evidence_col,
        pool_size=int(concept_pool_size),
        threshold=float(nonlinear_threshold),
    )
    interaction_pool, interaction_admission = _best_gain_concept_pool(
        selected,
        concept_flag_col="has_interaction",
        evidence_col=interaction_evidence_col,
        pool_size=int(concept_pool_size),
        threshold=float(interaction_threshold),
    )
    return _SummarySelectionState(
        feature_table=working_feature_table,
        pre_guard_selected=pre_guard_selected,
        selected=selected,
        lag_pool=selected.copy(),
        nonlinear_pool=nonlinear_pool,
        interaction_pool=interaction_pool,
        nonlinear_evidence_col=nonlinear_evidence_col,
        interaction_evidence_col=interaction_evidence_col,
        pre_guard_count=pre_guard_count,
        post_guard_count=post_guard_count,
        adequate_simple_count=int(adequate_simple_count),
        best_simple_fit_ratio=float(best_simple_fit_ratio),
        best_simple_fit_mse=float(best_simple_fit_mse),
        best_complex_fit_mse=float(best_complex_fit_mse),
        complex_gain_over_simple=float(complex_gain_over_simple),
        complex_gain_over_simple_threshold=float(complex_gain_over_simple_threshold),
        nonlinear_threshold=float(nonlinear_threshold),
        interaction_threshold=float(interaction_threshold),
        nonlinear_admission=nonlinear_admission,
        interaction_admission=interaction_admission,
        nonlinear_admission_debug=nonlinear_admission_debug,
        interaction_admission_debug=interaction_admission_debug,
    )


def summarize_expression_features(
    feature_table: pd.DataFrame,
    *,
    linear_fit_mse: float | None = None,
    fit_band_ratio: float = 1.15,
    guard_complex_models: bool = False,
    prefer_simple_models: bool = False,
    simple_model_fit_ratio: float = 1.10,
    simple_model_fallback_ratio: float = 0.0,
    concept_pool_size: int = 3,
    concept_pool_fit_band_ratio: float = 1.10,
    nonlinear_pool_min_ratio: float = 0.05,
    nonlinear_pool_min_abs: float = 1e-4,
    interaction_pool_min_ratio: float = 0.05,
    interaction_pool_min_abs: float = 1e-4,
    complex_over_simple_min_ratio: float = 0.05,
    complex_over_simple_min_abs: float = 1e-4,
    linear_advantage_min_ratio: float = 0.0,
    linear_advantage_min_abs: float = 0.0,
) -> Dict[str, float | str]:
    trace = build_summary_selection_trace(
        feature_table,
        linear_fit_mse=linear_fit_mse,
        fit_band_ratio=fit_band_ratio,
        guard_complex_models=guard_complex_models,
        prefer_simple_models=prefer_simple_models,
        simple_model_fit_ratio=simple_model_fit_ratio,
        simple_model_fallback_ratio=simple_model_fallback_ratio,
        concept_pool_size=concept_pool_size,
        concept_pool_fit_band_ratio=concept_pool_fit_band_ratio,
        nonlinear_pool_min_ratio=nonlinear_pool_min_ratio,
        nonlinear_pool_min_abs=nonlinear_pool_min_abs,
        interaction_pool_min_ratio=interaction_pool_min_ratio,
        interaction_pool_min_abs=interaction_pool_min_abs,
        complex_over_simple_min_ratio=complex_over_simple_min_ratio,
        complex_over_simple_min_abs=complex_over_simple_min_abs,
        linear_advantage_min_ratio=linear_advantage_min_ratio,
        linear_advantage_min_abs=linear_advantage_min_abs,
    )
    return trace.summary


def _summary_from_selection_state(state: _SummarySelectionState) -> Dict[str, float | str]:
    if state.pre_guard_selected.empty:
        return {}
    summary: Dict[str, float | str] = {}
    lag_pool = state.lag_pool
    nonlinear_pool = state.nonlinear_pool
    interaction_pool = state.interaction_pool
    nonlinear_evidence_col = state.nonlinear_evidence_col
    interaction_evidence_col = state.interaction_evidence_col

    lag_weights = 1.0 / np.maximum(lag_pool["fit_mse"].to_numpy(dtype=float), 1e-6)
    nonlinear_weights = (
        np.maximum(nonlinear_pool[nonlinear_evidence_col].to_numpy(dtype=float), 0.0)
        / np.maximum(nonlinear_pool["fit_mse"].to_numpy(dtype=float), 1e-6)
        if not nonlinear_pool.empty and nonlinear_evidence_col in nonlinear_pool.columns
        else np.ones(len(nonlinear_pool), dtype=float)
    )
    interaction_weights = (
        np.maximum(interaction_pool[interaction_evidence_col].to_numpy(dtype=float), 0.0)
        / np.maximum(interaction_pool["fit_mse"].to_numpy(dtype=float), 1e-6)
        if not interaction_pool.empty and interaction_evidence_col in interaction_pool.columns
        else np.ones(len(interaction_pool), dtype=float)
    )
    complexity_weights = lag_weights

    _summarize_pool_columns(
        summary,
        pool=lag_pool,
        columns=LAG_SUMMARY_COLS,
        weights=lag_weights,
        zero_if_empty=False,
    )
    _summarize_pool_columns(
        summary,
        pool=nonlinear_pool,
        columns=NONLINEAR_SUMMARY_COLS,
        weights=nonlinear_weights,
        zero_if_empty=True,
    )
    _summarize_pool_columns(
        summary,
        pool=interaction_pool,
        columns=INTERACTION_SUMMARY_COLS,
        weights=interaction_weights,
        zero_if_empty=True,
    )
    _summarize_pool_columns(
        summary,
        pool=lag_pool,
        columns=COMPLEXITY_SUMMARY_COLS,
        weights=complexity_weights,
        zero_if_empty=False,
    )

    stability_targets = [
        "lag_mass",
        "nonlinear_node_count",
        "interaction_term_count",
        "complexity_cost",
    ]
    cv_values: List[float] = []
    for col in stability_targets:
        if col in LAG_SUMMARY_COLS or col in COMPLEXITY_SUMMARY_COLS:
            pool = lag_pool
            weights = lag_weights
        elif col in NONLINEAR_SUMMARY_COLS:
            pool = nonlinear_pool
            weights = nonlinear_weights
        else:
            pool = interaction_pool
            weights = interaction_weights
        if pool.empty:
            cv_values.append(0.0)
            continue
        if col not in pool.columns:
            continue
        if col in STRUCTURE_ADVANTAGE_COLS and float(np.sum(weights)) <= 0.0:
            cv_values.append(0.0)
            continue
        values = pool[col].to_numpy(dtype=float)
        mean = float(_weighted_mean(np.abs(values), weights))
        if mean <= 1e-9:
            cv_values.append(0.0)
            continue
        cv = float(_weighted_std(values, weights) / max(mean, 1e-9))
        cv_values.append(cv)

    summary["structural_instability"] = float(np.median(cv_values) if cv_values else 1.0)
    summary["summary_model_count_pre_guard"] = float(state.pre_guard_count)
    summary["summary_model_count_post_guard"] = float(state.post_guard_count)
    summary["summary_model_count"] = float(state.post_guard_count)
    summary["summary_selection_mode"] = "concept_pools_separability"
    summary["adequate_simple_model_count"] = float(state.adequate_simple_count)
    summary["best_simple_fit_ratio_to_best_selected"] = (
        float(state.best_simple_fit_ratio) if np.isfinite(state.best_simple_fit_ratio) else float("nan")
    )
    summary["best_simple_fit_mse"] = (
        float(state.best_simple_fit_mse) if np.isfinite(state.best_simple_fit_mse) else float("nan")
    )
    summary["best_complex_fit_mse"] = (
        float(state.best_complex_fit_mse) if np.isfinite(state.best_complex_fit_mse) else float("nan")
    )
    summary["complex_gain_over_simple"] = (
        float(state.complex_gain_over_simple) if np.isfinite(state.complex_gain_over_simple) else float("nan")
    )
    summary["complex_gain_over_simple_threshold"] = float(state.complex_gain_over_simple_threshold)
    summary["lag_pool_model_count"] = float(len(lag_pool))
    summary["nonlinear_pool_model_count"] = float(len(nonlinear_pool))
    summary["interaction_pool_model_count"] = float(len(interaction_pool))
    summary["nonlinear_candidate_count"] = float(state.nonlinear_admission["candidate_count"])
    summary["interaction_candidate_count"] = float(state.interaction_admission["candidate_count"])
    summary["nonlinear_pool_admitted"] = float(state.nonlinear_admission["pool_admitted"])
    summary["interaction_pool_admitted"] = float(state.interaction_admission["pool_admitted"])
    summary["best_selected_unary_curvature_gain"] = float(state.nonlinear_admission["best_selected_gain"])
    summary["best_selected_interaction_synergy_gain"] = float(state.interaction_admission["best_selected_gain"])
    summary["unary_curvature_threshold"] = float(state.nonlinear_admission["threshold"])
    summary["interaction_synergy_threshold"] = float(state.interaction_admission["threshold"])
    summary["best_nonlinear_candidate_fit_mse"] = float(
        state.nonlinear_admission_debug["best_candidate_fit_mse"]
    )
    summary["best_non_nonlinear_fit_mse"] = float(
        state.nonlinear_admission_debug["best_comparator_fit_mse"]
    )
    summary["nonlinear_admission_gain"] = float(state.nonlinear_admission_debug["admission_gain"])
    summary["nonlinear_admission_threshold"] = float(state.nonlinear_admission_debug["admission_threshold"])
    summary["best_interaction_candidate_fit_mse"] = float(
        state.interaction_admission_debug["best_candidate_fit_mse"]
    )
    summary["best_non_interaction_fit_mse"] = float(
        state.interaction_admission_debug["best_comparator_fit_mse"]
    )
    summary["interaction_admission_gain"] = float(state.interaction_admission_debug["admission_gain"])
    summary["interaction_admission_threshold"] = float(
        state.interaction_admission_debug["admission_threshold"]
    )
    summary["nonlinear_pool_rejected_candidate_count"] = float(
        state.nonlinear_admission["rejected_candidate_count"]
    )
    summary["interaction_pool_rejected_candidate_count"] = float(
        state.interaction_admission["rejected_candidate_count"]
    )
    summary["fraction_adequate_models_supported_unary"] = float(
        np.mean(state.selected["has_unary_nonlinearity"].to_numpy(dtype=float))
    ) if "has_unary_nonlinearity" in state.selected.columns else 0.0
    summary["fraction_adequate_models_supported_interaction"] = float(
        np.mean(state.selected["has_interaction"].to_numpy(dtype=float))
    ) if "has_interaction" in state.selected.columns else 0.0
    summary["fraction_nonlinear_pool_models_raw_unary"] = float(
        np.mean(nonlinear_pool["raw_has_unary_nonlinearity"].to_numpy(dtype=float))
    ) if not nonlinear_pool.empty and "raw_has_unary_nonlinearity" in nonlinear_pool.columns else 0.0
    summary["fraction_interaction_pool_models_raw_interaction"] = float(
        np.mean(interaction_pool["raw_has_interaction"].to_numpy(dtype=float))
    ) if not interaction_pool.empty and "raw_has_interaction" in interaction_pool.columns else 0.0
    summary["fraction_summary_models_coupled_interaction"] = float(
        np.mean(state.selected["has_coupled_interaction"].to_numpy(dtype=float))
    ) if "has_coupled_interaction" in state.selected.columns else 0.0
    summary["fraction_summary_models_simple"] = float(np.mean(_simple_model_mask(state.selected)))
    summary["fraction_summary_models_unary_nonlinear"] = float(
        np.mean(state.selected["has_unary_nonlinearity"].to_numpy(dtype=float))
    ) if "has_unary_nonlinearity" in state.selected.columns else 0.0
    summary["fraction_summary_models_interaction"] = float(
        np.mean(state.selected["has_interaction"].to_numpy(dtype=float))
    ) if "has_interaction" in state.selected.columns else 0.0
    summary["fraction_summary_models_raw_simple"] = float(
        np.mean(state.selected["raw_is_simple_model"].to_numpy(dtype=float))
    ) if "raw_is_simple_model" in state.selected.columns else summary["fraction_summary_models_simple"]
    summary["fraction_summary_models_raw_unary_nonlinear"] = float(
        np.mean(state.selected["raw_has_unary_nonlinearity"].to_numpy(dtype=float))
    ) if "raw_has_unary_nonlinearity" in state.selected.columns else summary["fraction_summary_models_unary_nonlinear"]
    summary["fraction_summary_models_raw_interaction"] = float(
        np.mean(state.selected["raw_has_interaction"].to_numpy(dtype=float))
    ) if "raw_has_interaction" in state.selected.columns else summary["fraction_summary_models_interaction"]
    summary["fraction_summary_models_supported_unary"] = float(
        np.mean(nonlinear_pool["has_supported_unary_nonlinearity"].to_numpy(dtype=float))
    ) if not nonlinear_pool.empty and "has_supported_unary_nonlinearity" in nonlinear_pool.columns else 0.0
    summary["fraction_summary_models_supported_interaction"] = float(
        np.mean(interaction_pool["has_supported_interaction"].to_numpy(dtype=float))
    ) if not interaction_pool.empty and "has_supported_interaction" in interaction_pool.columns else 0.0
    return summary


def build_summary_selection_trace(
    feature_table: pd.DataFrame,
    *,
    linear_fit_mse: float | None = None,
    fit_band_ratio: float = 1.15,
    guard_complex_models: bool = False,
    prefer_simple_models: bool = False,
    simple_model_fit_ratio: float = 1.10,
    simple_model_fallback_ratio: float = 0.0,
    concept_pool_size: int = 3,
    concept_pool_fit_band_ratio: float = 1.10,
    nonlinear_pool_min_ratio: float = 0.05,
    nonlinear_pool_min_abs: float = 1e-4,
    interaction_pool_min_ratio: float = 0.05,
    interaction_pool_min_abs: float = 1e-4,
    complex_over_simple_min_ratio: float = 0.05,
    complex_over_simple_min_abs: float = 1e-4,
    linear_advantage_min_ratio: float = 0.0,
    linear_advantage_min_abs: float = 0.0,
    rescue_fitband_enabled: bool = False,
    rescue_fitband_tol_ratio: float = 0.02,
    rescue_fitband_tol_abs: float = 1e-4,
    rescue_max_extra_per_window: int = 1,
    rescue_complexity_cap_mode: str = "median_plus_2",
    softened_guard_enabled: bool = False,
    softened_guard_margin_ratio: float = 0.10,
    softened_guard_min_unary_preservation_ratio: float = 0.50,
) -> SummarySelectionTrace:
    if feature_table.empty:
        empty = feature_table.iloc[0:0].copy()
        return SummarySelectionTrace(
            feature_table=empty,
            pre_guard_selected=empty,
            selected=empty,
            lag_pool=empty,
            nonlinear_pool=empty,
            interaction_pool=empty,
            summary={},
        )

    del prefer_simple_models, simple_model_fit_ratio

    effective_simple_fallback_ratio = (
        float(simple_model_fallback_ratio)
        if float(simple_model_fallback_ratio) > 0.0
        else float(fit_band_ratio)
    )
    state = _build_summary_selection_state(
        feature_table,
        linear_fit_mse=linear_fit_mse,
        fit_band_ratio=float(fit_band_ratio),
        guard_complex_models=bool(guard_complex_models),
        simple_model_fallback_ratio=effective_simple_fallback_ratio,
        concept_pool_size=int(concept_pool_size),
        concept_pool_fit_band_ratio=float(concept_pool_fit_band_ratio),
        nonlinear_pool_min_ratio=float(nonlinear_pool_min_ratio),
        nonlinear_pool_min_abs=float(nonlinear_pool_min_abs),
        interaction_pool_min_ratio=float(interaction_pool_min_ratio),
        interaction_pool_min_abs=float(interaction_pool_min_abs),
        complex_over_simple_min_ratio=float(complex_over_simple_min_ratio),
        complex_over_simple_min_abs=float(complex_over_simple_min_abs),
        linear_advantage_min_ratio=float(linear_advantage_min_ratio),
        linear_advantage_min_abs=float(linear_advantage_min_abs),
        rescue_fitband_enabled=bool(rescue_fitband_enabled),
        rescue_fitband_tol_ratio=float(rescue_fitband_tol_ratio),
        rescue_fitband_tol_abs=float(rescue_fitband_tol_abs),
        rescue_max_extra_per_window=int(rescue_max_extra_per_window),
        rescue_complexity_cap_mode=str(rescue_complexity_cap_mode),
        softened_guard_enabled=bool(softened_guard_enabled),
        softened_guard_margin_ratio=float(softened_guard_margin_ratio),
        softened_guard_min_unary_preservation_ratio=float(
            softened_guard_min_unary_preservation_ratio
        ),
    )
    return SummarySelectionTrace(
        feature_table=state.feature_table.copy(),
        pre_guard_selected=state.pre_guard_selected.copy(),
        selected=state.selected.copy(),
        lag_pool=state.lag_pool.copy(),
        nonlinear_pool=state.nonlinear_pool.copy(),
        interaction_pool=state.interaction_pool.copy(),
        summary=_summary_from_selection_state(state),
    )


def build_family_transition_diagnostics(
    feature_table: pd.DataFrame,
    *,
    linear_fit_mse: float | None = None,
    fit_band_ratio: float = 1.15,
    guard_complex_models: bool = False,
    simple_model_fallback_ratio: float = 0.0,
    concept_pool_size: int = 3,
    concept_pool_fit_band_ratio: float = 1.10,
    nonlinear_pool_min_ratio: float = 0.05,
    nonlinear_pool_min_abs: float = 1e-4,
    interaction_pool_min_ratio: float = 0.05,
    interaction_pool_min_abs: float = 1e-4,
    complex_over_simple_min_ratio: float = 0.05,
    complex_over_simple_min_abs: float = 1e-4,
    linear_advantage_min_ratio: float = 0.0,
    linear_advantage_min_abs: float = 0.0,
) -> Dict[str, float | str]:
    if feature_table.empty:
        return {}

    effective_simple_fallback_ratio = (
        float(simple_model_fallback_ratio)
        if float(simple_model_fallback_ratio) > 0.0
        else float(fit_band_ratio)
    )
    state = _build_summary_selection_state(
        feature_table,
        linear_fit_mse=linear_fit_mse,
        fit_band_ratio=float(fit_band_ratio),
        guard_complex_models=bool(guard_complex_models),
        simple_model_fallback_ratio=effective_simple_fallback_ratio,
        concept_pool_size=int(concept_pool_size),
        concept_pool_fit_band_ratio=float(concept_pool_fit_band_ratio),
        nonlinear_pool_min_ratio=float(nonlinear_pool_min_ratio),
        nonlinear_pool_min_abs=float(nonlinear_pool_min_abs),
        interaction_pool_min_ratio=float(interaction_pool_min_ratio),
        interaction_pool_min_abs=float(interaction_pool_min_abs),
        complex_over_simple_min_ratio=float(complex_over_simple_min_ratio),
        complex_over_simple_min_abs=float(complex_over_simple_min_abs),
        linear_advantage_min_ratio=float(linear_advantage_min_ratio),
        linear_advantage_min_abs=float(linear_advantage_min_abs),
    )
    if state.pre_guard_selected.empty:
        return {}

    retained_family_stats = _family_stage_stats(
        feature_table,
        stage="retained",
        unary_evidence_col=state.nonlinear_evidence_col,
        interaction_evidence_col=state.interaction_evidence_col,
    )
    fit_band_family_stats = _family_stage_stats(
        state.pre_guard_selected,
        stage="fit_band",
        unary_evidence_col=state.nonlinear_evidence_col,
        interaction_evidence_col=state.interaction_evidence_col,
    )
    post_guard_family_stats = _family_stage_stats(
        state.selected,
        stage="post_guard",
        unary_evidence_col=state.nonlinear_evidence_col,
        interaction_evidence_col=state.interaction_evidence_col,
    )
    retained_unary_count = int(retained_family_stats["family_trace_retained_unary_count"])
    fit_band_unary_count = int(fit_band_family_stats["family_trace_fit_band_unary_count"])
    post_guard_unary_count = int(post_guard_family_stats["family_trace_post_guard_unary_count"])
    retained_interaction_count = int(retained_family_stats["family_trace_retained_interaction_count"])
    fit_band_interaction_count = int(fit_band_family_stats["family_trace_fit_band_interaction_count"])
    post_guard_interaction_count = int(post_guard_family_stats["family_trace_post_guard_interaction_count"])
    post_guard_simple_count = int(post_guard_family_stats["family_trace_post_guard_simple_count"])

    diagnostics: Dict[str, float | str] = {}
    diagnostics.update(retained_family_stats)
    diagnostics.update(fit_band_family_stats)
    diagnostics.update(post_guard_family_stats)
    diagnostics["unary_present_retained"] = float(retained_unary_count > 0)
    diagnostics["unary_present_fit_band"] = float(fit_band_unary_count > 0)
    diagnostics["unary_present_post_guard"] = float(post_guard_unary_count > 0)
    diagnostics["interaction_present_retained"] = float(retained_interaction_count > 0)
    diagnostics["interaction_present_fit_band"] = float(fit_band_interaction_count > 0)
    diagnostics["interaction_present_post_guard"] = float(post_guard_interaction_count > 0)
    diagnostics["fit_band_removed_unary"] = float(retained_unary_count > 0 and fit_band_unary_count == 0)
    diagnostics["fit_band_removed_interaction"] = float(
        retained_interaction_count > 0 and fit_band_interaction_count == 0
    )
    diagnostics["complex_guard_removed_unary"] = float(fit_band_unary_count > 0 and post_guard_unary_count == 0)
    diagnostics["complex_guard_removed_interaction"] = float(
        fit_band_interaction_count > 0 and post_guard_interaction_count == 0
    )
    diagnostics["delta_fit_unary_vs_interaction"] = _delta_or_nan(
        float(post_guard_family_stats["family_trace_post_guard_unary_best_fit_mse"]),
        float(post_guard_family_stats["family_trace_post_guard_interaction_best_fit_mse"]),
    )
    diagnostics["delta_fit_simple_vs_unary"] = _delta_or_nan(
        float(post_guard_family_stats["family_trace_post_guard_simple_best_fit_mse"]),
        float(post_guard_family_stats["family_trace_post_guard_unary_best_fit_mse"]),
    )
    diagnostics["delta_evidence_unary_vs_interaction"] = _delta_or_nan(
        float(post_guard_family_stats["family_trace_post_guard_unary_best_evidence"]),
        float(post_guard_family_stats["family_trace_post_guard_interaction_best_evidence"]),
    )
    diagnostics.update(
        _family_separated_summary(
            state.selected,
            family="simple",
            evidence_col=None,
        )
    )
    diagnostics.update(
        _family_separated_summary(
            state.selected,
            family="unary",
            evidence_col=state.nonlinear_evidence_col,
        )
    )
    diagnostics.update(
        _family_separated_summary(
            state.selected,
            family="interaction",
            evidence_col=state.interaction_evidence_col,
        )
    )
    diagnostics["current_logic_decision_code"] = _current_logic_decision_code(
        post_guard_simple_count=post_guard_simple_count,
        nonlinear_pool_count=len(state.nonlinear_pool),
        interaction_pool_count=len(state.interaction_pool),
    )
    return diagnostics


def summarize_model_family_diagnostics(
    feature_table: pd.DataFrame,
    *,
    max_lag: int,
) -> Dict[str, float | str]:
    if feature_table.empty:
        return {
            "best_equation": "",
            "retained_model_count": 0.0,
            "fraction_models_nonlinear": 0.0,
            "fraction_models_simple": 0.0,
            "fraction_models_unary_nonlinear": 0.0,
            "fraction_models_interaction": 0.0,
            "fraction_models_coupled_interaction": 0.0,
            "fraction_models_raw_nonlinear": 0.0,
            "fraction_models_raw_simple": 0.0,
            "fraction_models_raw_unary_nonlinear": 0.0,
            "fraction_models_raw_interaction": 0.0,
            "fraction_models_supported_unary": 0.0,
            "fraction_models_supported_interaction": 0.0,
            "retained_model_count_simple": 0.0,
            "retained_model_count_unary": 0.0,
            "retained_model_count_interaction": 0.0,
            "retained_model_count_raw_simple": 0.0,
            "retained_model_count_raw_unary": 0.0,
            "retained_model_count_raw_interaction": 0.0,
            "fraction_models_using_max_lag": 0.0,
        }

    ordered = feature_table.sort_values(["fit_mse", "expression_complexity", "model_idx"]).reset_index(drop=True)
    simple_mask = _simple_model_mask(ordered)
    raw_simple_mask = (
        ordered["raw_is_simple_model"].to_numpy(dtype=float) > 0.0
        if "raw_is_simple_model" in ordered.columns
        else simple_mask
    )
    return {
        "best_equation": str(ordered.loc[0, "equation"]),
        "retained_model_count": float(len(ordered)),
        "fraction_models_nonlinear": float(np.mean(ordered["has_nonlinearity"].to_numpy(dtype=float))),
        "fraction_models_simple": float(np.mean(simple_mask)),
        "fraction_models_unary_nonlinear": float(
            np.mean(ordered["has_unary_nonlinearity"].to_numpy(dtype=float))
        ) if "has_unary_nonlinearity" in ordered.columns else 0.0,
        "fraction_models_interaction": float(np.mean(ordered["has_interaction"].to_numpy(dtype=float))),
        "fraction_models_coupled_interaction": float(
            np.mean(ordered["has_coupled_interaction"].to_numpy(dtype=float))
        ) if "has_coupled_interaction" in ordered.columns else 0.0,
        "fraction_models_raw_nonlinear": float(
            np.mean(ordered["raw_has_nonlinearity"].to_numpy(dtype=float))
        ) if "raw_has_nonlinearity" in ordered.columns else float(np.mean(ordered["has_nonlinearity"].to_numpy(dtype=float))),
        "fraction_models_raw_simple": float(np.mean(raw_simple_mask)),
        "fraction_models_raw_unary_nonlinear": float(
            np.mean(ordered["raw_has_unary_nonlinearity"].to_numpy(dtype=float))
        ) if "raw_has_unary_nonlinearity" in ordered.columns else (
            float(np.mean(ordered["has_unary_nonlinearity"].to_numpy(dtype=float)))
            if "has_unary_nonlinearity" in ordered.columns
            else 0.0
        ),
        "fraction_models_raw_interaction": float(
            np.mean(ordered["raw_has_interaction"].to_numpy(dtype=float))
        ) if "raw_has_interaction" in ordered.columns else (
            float(np.mean(ordered["has_interaction"].to_numpy(dtype=float)))
            if "has_interaction" in ordered.columns
            else 0.0
        ),
        "fraction_models_supported_unary": float(
            np.mean(ordered["has_supported_unary_nonlinearity"].to_numpy(dtype=float))
        ) if "has_supported_unary_nonlinearity" in ordered.columns else 0.0,
        "fraction_models_supported_interaction": float(
            np.mean(ordered["has_supported_interaction"].to_numpy(dtype=float))
        ) if "has_supported_interaction" in ordered.columns else 0.0,
        "retained_model_count_simple": float(np.sum(simple_mask)),
        "retained_model_count_unary": float(
            np.sum(
                (ordered["has_unary_nonlinearity"].to_numpy(dtype=float) > 0.0)
                & (ordered["has_interaction"].to_numpy(dtype=float) <= 0.0)
            )
        ) if {"has_unary_nonlinearity", "has_interaction"}.issubset(ordered.columns) else 0.0,
        "retained_model_count_interaction": float(np.sum(ordered["has_interaction"].to_numpy(dtype=float)))
        if "has_interaction" in ordered.columns else 0.0,
        "retained_model_count_raw_simple": float(np.sum(raw_simple_mask)),
        "retained_model_count_raw_unary": float(
            np.sum(
                (ordered["raw_has_unary_nonlinearity"].to_numpy(dtype=float) > 0.0)
                & (ordered["raw_has_interaction"].to_numpy(dtype=float) <= 0.0)
            )
        ) if {"raw_has_unary_nonlinearity", "raw_has_interaction"}.issubset(ordered.columns) else 0.0,
        "retained_model_count_raw_interaction": float(
            np.sum(ordered["raw_has_interaction"].to_numpy(dtype=float))
        ) if "raw_has_interaction" in ordered.columns else 0.0,
        "fraction_models_using_max_lag": float(
            np.mean(ordered["max_lag_used"].to_numpy(dtype=float) >= float(max_lag))
        ),
    }


def medoid_equation(feature_table: pd.DataFrame, metric: str = "fit_mse") -> str:
    if feature_table.empty:
        return ""
    idx = int(feature_table[metric].idxmin())
    return str(feature_table.loc[idx, "equation"])
