from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .config import ProfilingConfig
from .features import expression_feature_table, summarize_expression_features
from .models.ensemble import fit_symbolic_ensemble, mse
from .scaling import standardize_window_design
from .windowing import FeatureMeta

if TYPE_CHECKING:
    from .checkpoint import WindowCheckpointStore

_BASELINE_SYMBOLIC_PHASE = "baseline_symbolic"


@dataclass(slots=True)
class WindowModelContext:
    window_index: int
    label: str
    start: int
    end: int
    record_id: str | None
    target_index: int
    window_values: np.ndarray
    X_fit: np.ndarray
    y_fit: np.ndarray
    feature_meta: Sequence[FeatureMeta]


def _lag_stats_from_importance(
    importances: np.ndarray,
    feature_meta: Sequence[FeatureMeta],
    threshold: float,
    long_lag_cutoff: int,
) -> tuple[float, float, float, float]:
    lag_scores: Dict[int, float] = {}
    for m in feature_meta:
        lag_scores[m.lag] = lag_scores.get(m.lag, 0.0) + float(importances[m.column])

    lags = np.array(sorted(lag_scores.keys()), dtype=float)
    vals = np.array([lag_scores[int(l)] for l in lags], dtype=float)

    total = float(vals.sum())
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    probs = vals / total
    lag_mass = float(np.sum(lags * probs))
    long_lag_mass = float(np.sum(probs[lags >= float(long_lag_cutoff)]))
    entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0] + 1e-12)))
    active = float(np.sum(vals >= threshold * np.max(vals)))
    return lag_mass, long_lag_mass, entropy, active


def _linear_equation_from_coeffs(
    coeffs: np.ndarray,
    feature_meta: Sequence[FeatureMeta],
    intercept: float,
) -> str:
    terms = []
    for coef, meta in zip(coeffs, feature_meta):
        if abs(float(coef)) < 1e-8:
            continue
        terms.append(f"({coef:.4f} * x{meta.variable}_lag{meta.lag})")

    if not terms:
        return f"{intercept:.4f}"
    return f"({intercept:.4f} + " + " + ".join(terms) + ")"


def run_linear_lasso_baseline(
    contexts: Sequence[WindowModelContext],
    config: ProfilingConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []

    for ctx in contexts:
        model = Lasso(
            alpha=config.baselines.lasso_alpha,
            max_iter=config.baselines.lasso_max_iter,
            random_state=0,
        )
        model.fit(ctx.X_fit, ctx.y_fit)
        fit_pred = model.predict(ctx.X_fit)
        fit_mse = mse(ctx.y_fit, fit_pred)

        importances = np.abs(model.coef_)
        lag_mass, long_lag_mass, lag_entropy, active_lag_count = _lag_stats_from_importance(
            importances=importances,
            feature_meta=ctx.feature_meta,
            threshold=config.features.active_lag_threshold,
            long_lag_cutoff=int(config.features.memory_long_lag_cutoff),
        )

        nonzero = int(np.sum(importances > 1e-8))
        complexity = float(nonzero + 1)
        description_length = float(nonzero + 0.5 * complexity)

        rows.append(
            {
                "baseline_name": "linear_lasso_ar",
                "window_index": ctx.window_index,
                "label": ctx.label,
                "start": ctx.start,
                "end": ctx.end,
                "record_id": ctx.record_id,
                "n_models": 1,
                "lag_mass_median": lag_mass,
                "long_lag_mass_median": long_lag_mass,
                "lag_entropy_median": lag_entropy,
                "active_lag_count_median": active_lag_count,
                "nonlinear_node_count_median": 0.0,
                "nonlinear_composition_depth_median": 0.0,
                "interaction_term_count_median": 0.0,
                "interaction_order_median": 1.0,
                "interaction_depth_median": 0.0,
                "complexity_cost_median": complexity,
                "description_length_median": description_length,
                "best_symbolic_fit_mse_median": fit_mse,
                "medoid_equation": _linear_equation_from_coeffs(
                    coeffs=model.coef_,
                    feature_meta=ctx.feature_meta,
                    intercept=float(model.intercept_),
                ),
                "structural_instability": 0.0,
            }
        )

    return pd.DataFrame(rows)


def run_non_ensemble_symbolic_baseline(
    contexts: Sequence[WindowModelContext],
    config: ProfilingConfig,
    checkpoint_store: Optional["WindowCheckpointStore"] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []

    baseline_cfg = copy.deepcopy(config)
    baseline_cfg.ensemble.n_seeds = 1
    baseline_cfg.ensemble.top_k_per_seed = 1
    baseline_cfg.ensemble.seed_offset += 10000

    if checkpoint_store is not None:
        cached = checkpoint_store.count(_BASELINE_SYMBOLIC_PHASE)
        if cached:
            print(
                f"[checkpoint] resuming with {cached}/{len(contexts)} baseline-symbolic windows cached"
            )

    for ctx in contexts:
        if checkpoint_store is not None:
            cached_row = checkpoint_store.get(_BASELINE_SYMBOLIC_PHASE, int(ctx.window_index))
            if cached_row is not None:
                if isinstance(cached_row, dict) and cached_row:
                    rows.append(cached_row)
                continue

        X_model, y_model, _ = standardize_window_design(
            ctx.X_fit,
            ctx.y_fit,
            enabled=bool(config.ensemble.normalize_window_for_symbolic),
        )
        ensemble = fit_symbolic_ensemble(
            X_fit=X_model,
            y_fit=y_model,
            feature_meta=ctx.feature_meta,
            target_index=ctx.target_index,
            config=baseline_cfg.ensemble,
            feature_config=config.features,
        )

        if not ensemble.retained_models:
            if checkpoint_store is not None:
                # Cache the empty result so we don't re-fit on resume.
                checkpoint_store.put(
                    _BASELINE_SYMBOLIC_PHASE, int(ctx.window_index), {}
                )
            continue

        table = expression_feature_table(
            models=ensemble.retained_models,
            X_reference=X_model,
            feature_meta=ctx.feature_meta,
            config=config.features,
            y_reference=y_model,
            linear_fit_mse=float(ensemble.linear_fit_mse),
        )
        summary = summarize_expression_features(
            table,
            linear_fit_mse=float(ensemble.linear_fit_mse),
            fit_band_ratio=float(config.ensemble.window_summary_fit_band_ratio),
            concept_pool_size=int(config.ensemble.window_summary_concept_pool_size),
            concept_pool_fit_band_ratio=float(config.ensemble.window_summary_concept_pool_fit_band_ratio),
            nonlinear_pool_min_ratio=float(config.ensemble.window_summary_nonlinear_pool_min_ratio),
            nonlinear_pool_min_abs=float(config.ensemble.window_summary_nonlinear_pool_min_abs),
            interaction_pool_min_ratio=float(config.ensemble.window_summary_interaction_pool_min_ratio),
            interaction_pool_min_abs=float(config.ensemble.window_summary_interaction_pool_min_abs),
        )

        row = {
            "baseline_name": "symbolic_no_ensemble",
            "window_index": ctx.window_index,
            "label": ctx.label,
            "start": ctx.start,
            "end": ctx.end,
            "record_id": ctx.record_id,
            "n_models": int(len(ensemble.retained_models)),
            "lag_mass_median": summary.get("lag_mass_median", 0.0),
            "long_lag_mass_median": summary.get("long_lag_mass_median", 0.0),
            "lag_entropy_median": summary.get("lag_entropy_median", 0.0),
            "active_lag_count_median": summary.get("active_lag_count_median", 0.0),
            "nonlinear_node_count_median": summary.get("nonlinear_node_count_median", 0.0),
            "nonlinear_composition_depth_median": summary.get(
                "nonlinear_composition_depth_median", 0.0
            ),
            "interaction_term_count_median": summary.get("interaction_term_count_median", 0.0),
            "interaction_order_median": summary.get("interaction_order_median", 1.0),
            "interaction_depth_median": summary.get("interaction_depth_median", 0.0),
            "complexity_cost_median": summary.get("complexity_cost_median", 0.0),
            "description_length_median": summary.get("description_length_median", 0.0),
            "best_symbolic_fit_mse_median": float(ensemble.best_symbolic_fit_mse),
            "medoid_equation": str(table.loc[table["fit_mse"].idxmin(), "equation"]),
            "structural_instability": float(summary.get("structural_instability", 0.0)),
        }
        rows.append(row)
        if checkpoint_store is not None:
            checkpoint_store.put(_BASELINE_SYMBOLIC_PHASE, int(ctx.window_index), row)

    return pd.DataFrame(rows)


def _window_embedding(values: np.ndarray) -> np.ndarray:
    # Simple descriptive embedding baseline for regime separability checks.
    emb: List[float] = []
    for v in range(values.shape[1]):
        series = values[:, v]
        mean = float(np.mean(series))
        std = float(np.std(series))
        diff = np.diff(series)
        mad = float(np.mean(np.abs(diff))) if diff.size else 0.0
        if series.size > 1:
            a = series[:-1]
            b = series[1:]
            denom = float(np.std(a) * np.std(b))
            ac1 = float(np.corrcoef(a, b)[0, 1]) if denom > 1e-12 else 0.0
        else:
            ac1 = 0.0
        emb.extend([mean, std, mad, ac1])
    return np.array(emb, dtype=float)


def run_embedding_baseline(
    contexts: Sequence[WindowModelContext],
    config: ProfilingConfig,
) -> pd.DataFrame:
    if not contexts:
        return pd.DataFrame()

    X = np.vstack([_window_embedding(ctx.window_values) for ctx in contexts])
    y = np.array([ctx.label for ctx in contexts], dtype=str)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    max_components = min(config.baselines.embedding_n_components, Xs.shape[1], max(1, Xs.shape[0] - 1))
    pca = PCA(n_components=max_components, random_state=0)
    Xp = pca.fit_transform(Xs)

    class_counts = pd.Series(y).value_counts()
    min_class = int(class_counts.min()) if not class_counts.empty else 0
    n_splits = min(config.baselines.embedding_cv_folds, min_class)

    accuracy = np.nan
    balanced = np.nan
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        acc_scores = []
        bal_scores = []
        for train_idx, test_idx in cv.split(Xp, y):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(Xp[train_idx], y[train_idx])
            pred = clf.predict(Xp[test_idx])
            acc_scores.append(accuracy_score(y[test_idx], pred))
            bal_scores.append(balanced_accuracy_score(y[test_idx], pred))

        accuracy = float(np.mean(acc_scores))
        balanced = float(np.mean(bal_scores))

    sil = np.nan
    if len(np.unique(y)) > 1 and Xp.shape[0] > len(np.unique(y)):
        sil = float(silhouette_score(Xp, y))

    return pd.DataFrame(
        [
            {
                "baseline_name": "embedding_separation",
                "n_windows": int(X.shape[0]),
                "embedding_dims": int(Xp.shape[1]),
                "cv_accuracy": accuracy,
                "cv_balanced_accuracy": balanced,
                "silhouette_score": sil,
            }
        ]
    )


def _aggregate_baseline_regimes(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, float | str]] = []
    feature_cols = [c for c in window_df.columns if c.endswith("_median")]

    for (baseline_name, label), group in window_df.groupby(["baseline_name", "label"]):
        row: Dict[str, float | str] = {
            "baseline_name": str(baseline_name),
            "label": str(label),
            "n_windows": float(len(group)),
        }
        for col in feature_cols:
            vals = group[col].to_numpy(dtype=float)
            row[f"{col}_p50"] = float(np.median(vals))
            row[f"{col}_iqr"] = float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))
        rows.append(row)

    return pd.DataFrame(rows)


def run_baseline_suite(
    contexts: Sequence[WindowModelContext],
    config: ProfilingConfig,
    checkpoint_store: Optional["WindowCheckpointStore"] = None,
) -> dict[str, pd.DataFrame]:
    if not config.baselines.run_baselines:
        return {}

    lasso_df = run_linear_lasso_baseline(contexts, config)
    no_ens_df = run_non_ensemble_symbolic_baseline(
        contexts, config, checkpoint_store=checkpoint_store
    )
    combined = pd.concat([lasso_df, no_ens_df], ignore_index=True)

    return {
        "baseline_window_summaries": combined,
        "baseline_regime_profiles": _aggregate_baseline_regimes(combined),
        "embedding_baseline_metrics": run_embedding_baseline(contexts, config),
    }
