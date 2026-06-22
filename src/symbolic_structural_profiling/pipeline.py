from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .aggregation import aggregate_regime_profiles
from .baselines import WindowModelContext, run_baseline_suite
from .checkpoint import (
    WindowCheckpointStore,
    make_config_signature,
    make_data_fingerprint,
)
from .config import ProfilingConfig
from .features import (
    expression_feature_table,
    medoid_equation,
    summarize_expression_features,
    summarize_model_family_diagnostics,
)
from .models.ensemble import fit_symbolic_ensemble, mse, persistence_baseline_fit_mse
from .models.expressions import safe_eval
from .models.jacobian import ensemble_jacobian_features, ensemble_hessian_features
from .scaling import invert_scaled_target, standardize_window_design
from .statistics import run_pairwise_feature_tests
from .windowing import build_lagged_design, segment_labeled_series

_PIPELINE_PHASE = "pipeline_main"


@dataclass(slots=True)
class PrototypeFit:
    label: str
    window_index: int
    y_true: np.ndarray
    y_pred: np.ndarray
    fit_mse: float


@dataclass(slots=True)
class ProfilingResult:
    raw_window_summaries: pd.DataFrame
    window_summaries: pd.DataFrame
    regime_profiles: pd.DataFrame
    pairwise_stats: pd.DataFrame
    medoid_equations: pd.DataFrame
    window_evaluation_all: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_outputs: dict[str, pd.DataFrame] = field(default_factory=dict)
    prototype_fits: list[PrototypeFit] = field(default_factory=list)
    window_fits: list[PrototypeFit] = field(default_factory=list)


class StructuralProfiler:
    def __init__(self, config: ProfilingConfig | None = None) -> None:
        self.config = config or ProfilingConfig()

    def fit(
        self,
        values,
        labels,
        record_ids=None,
        checkpoint_dir: Optional[Path] = None,
    ) -> ProfilingResult:
        cfg = self.config
        values_array = np.asarray(values, dtype=float)
        if values_array.ndim != 2:
            raise ValueError("values must have shape [time, variables]")
        if len(labels) != values_array.shape[0]:
            raise ValueError("labels length must match time axis")
        if record_ids is not None and len(record_ids) != values_array.shape[0]:
            raise ValueError("record_ids length must match time axis")

        target_index = int(cfg.target_index)
        if target_index < 0 or target_index >= values_array.shape[1]:
            raise ValueError("target_index out of bounds for values")

        model_values = (
            values_array[:, [target_index]]
            if values_array.shape[1] > 1
            else values_array
        )
        model_target_index = 0 if values_array.shape[1] > 1 else target_index

        windows = segment_labeled_series(
            values=model_values,
            labels=labels,
            window_size=cfg.window.window_size,
            step_size=cfg.window.step_size,
            require_homogeneous_labels=cfg.window.require_homogeneous_labels,
            record_ids=record_ids,
        )
        if not windows:
            raise ValueError("No valid windows were produced. Adjust windowing settings.")

        store: Optional[WindowCheckpointStore] = None
        if checkpoint_dir is not None:
            store = WindowCheckpointStore(
                root=Path(checkpoint_dir),
                config_signature=make_config_signature(
                    cfg.window, cfg.ensemble, cfg.features
                ),
                data_fingerprint=make_data_fingerprint(values_array, labels, record_ids),
                n_windows_expected=len(windows),
            )
            cached = store.count(_PIPELINE_PHASE)
            if cached:
                print(
                    f"[checkpoint] resuming with {cached}/{len(windows)} pipeline windows cached"
                )
        rows = []
        evaluation_rows = []
        contexts: list[WindowModelContext] = []
        best_window_fits: dict[int, PrototypeFit] = {}

        for window in windows:
            X_raw, y_raw, feature_meta = build_lagged_design(
                window_values=window.values,
                max_lag=cfg.ensemble.max_lag,
                target_index=model_target_index,
            )
            X_model, y_model, scaling = standardize_window_design(
                X_raw,
                y_raw,
                enabled=bool(cfg.ensemble.normalize_window_for_symbolic),
            )

            contexts.append(
                WindowModelContext(
                    window_index=int(window.index),
                    label=str(window.label),
                    start=int(window.start),
                    end=int(window.end),
                    record_id=window.record_id,
                    target_index=model_target_index,
                    window_values=window.values,
                    X_fit=X_raw,
                    y_fit=y_raw,
                    feature_meta=feature_meta,
                )
            )

            if store is not None:
                cached_payload = store.get(_PIPELINE_PHASE, int(window.index))
                if cached_payload is not None:
                    summary_cached = cached_payload["summary"]
                    eval_row_cached = cached_payload["evaluation_row"]
                    proto_cached = cached_payload.get("prototype_fit")
                    if proto_cached is not None:
                        best_window_fits[int(window.index)] = PrototypeFit(
                            label=str(proto_cached["label"]),
                            window_index=int(proto_cached["window_index"]),
                            y_true=np.asarray(proto_cached["y_true"], dtype=float),
                            y_pred=np.asarray(proto_cached["y_pred"], dtype=float),
                            fit_mse=float(proto_cached["fit_mse"]),
                        )
                    rows.append(summary_cached)
                    evaluation_rows.append(eval_row_cached)
                    continue

            ensemble = fit_symbolic_ensemble(
                X_fit=X_model,
                y_fit=y_model,
                feature_meta=feature_meta,
                target_index=model_target_index,
                config=cfg.ensemble,
                feature_config=cfg.features,
            )

            all_models = ensemble.retained_models
            if not all_models:
                continue

            best_model = min(all_models, key=lambda m: m.fit_mse)
            y_pred_model = np.asarray(safe_eval(best_model.expression, X_model), dtype=float)
            y_pred = invert_scaled_target(y_pred_model, scaling)
            best_symbolic_fit_mse = mse(y_raw, y_pred)
            baseline_fit_mse = persistence_baseline_fit_mse(
                X_raw,
                y_raw,
                feature_meta,
                model_target_index,
            )
            linear_model = Ridge(alpha=1.0)
            linear_model.fit(X_model, y_model)
            linear_pred = invert_scaled_target(np.asarray(linear_model.predict(X_model), dtype=float), scaling)
            linear_fit_mse = mse(y_raw, linear_pred)
            best_window_fits[int(window.index)] = PrototypeFit(
                label=str(window.label),
                window_index=int(window.index),
                y_true=np.asarray(y_raw, dtype=float).copy(),
                y_pred=y_pred,
                fit_mse=float(best_symbolic_fit_mse),
            )

            feature_table = expression_feature_table(
                models=all_models,
                X_reference=X_model,
                feature_meta=feature_meta,
                config=cfg.features,
                y_reference=y_model,
                linear_fit_mse=float(ensemble.linear_fit_mse),
            )
            if feature_table.empty:
                continue

            summary = summarize_expression_features(
                feature_table,
                linear_fit_mse=float(ensemble.linear_fit_mse),
                fit_band_ratio=float(cfg.ensemble.window_summary_fit_band_ratio),
                guard_complex_models=bool(cfg.ensemble.window_summary_guard_complex_models),
                prefer_simple_models=bool(cfg.ensemble.window_summary_prefer_simple_models),
                simple_model_fit_ratio=float(cfg.ensemble.window_summary_simple_fit_ratio),
                simple_model_fallback_ratio=float(cfg.ensemble.window_summary_simple_fallback_ratio),
                concept_pool_size=int(cfg.ensemble.window_summary_concept_pool_size),
                concept_pool_fit_band_ratio=float(cfg.ensemble.window_summary_concept_pool_fit_band_ratio),
                nonlinear_pool_min_ratio=float(cfg.ensemble.window_summary_nonlinear_pool_min_ratio),
                nonlinear_pool_min_abs=float(cfg.ensemble.window_summary_nonlinear_pool_min_abs),
                interaction_pool_min_ratio=float(cfg.ensemble.window_summary_interaction_pool_min_ratio),
                interaction_pool_min_abs=float(cfg.ensemble.window_summary_interaction_pool_min_abs),
                complex_over_simple_min_ratio=float(
                    cfg.ensemble.window_summary_complex_over_simple_min_ratio
                ),
                complex_over_simple_min_abs=float(
                    cfg.ensemble.window_summary_complex_over_simple_min_abs
                ),
                linear_advantage_min_ratio=float(cfg.ensemble.window_summary_linear_advantage_min_ratio),
                linear_advantage_min_abs=float(cfg.ensemble.window_summary_linear_advantage_min_abs),
            )
            summary.update(
                summarize_model_family_diagnostics(
                    feature_table,
                    max_lag=int(cfg.ensemble.max_lag),
                )
            )
            lag_for_feature = [fm.lag for fm in feature_meta]
            summary.update(
                ensemble_jacobian_features(
                    models=all_models,
                    X=X_model,
                    lag_for_feature=lag_for_feature,
                    memory_long_lag_cutoff=int(cfg.features.memory_long_lag_cutoff),
                )
            )
            # Second-order behavioural concepts: off-diagonal Hessian reads
            # cross-lag interaction functionally (where the syntactic interaction
            # count is contaminated), diagonal reads single-lag curvature. Aggregated
            # over the full ensemble by the MEAN (not median): the interaction is
            # carried by a minority of the set, which a median would zero out, so the
            # mean reads prevalence x magnitude -- the same footing as the first-order
            # Jacobian concepts, no special best-fit representative.
            summary.update(
                ensemble_hessian_features(models=all_models, X=X_model)
            )
            summary["window_index"] = int(window.index)
            summary["label"] = str(window.label)
            summary["start"] = int(window.start)
            summary["end"] = int(window.end)
            if window.record_id is not None:
                summary["record_id"] = str(window.record_id)
            summary["n_models"] = int(len(all_models))
            summary["baseline_fit_mse_median"] = float(baseline_fit_mse)
            summary["linear_fit_mse_median"] = float(linear_fit_mse)
            summary["best_symbolic_fit_mse_median"] = float(best_symbolic_fit_mse)
            summary["linear_symbolic_fit_gap_median"] = float(linear_fit_mse - best_symbolic_fit_mse)

            # Calibrate Jacobian nonlinearity index by SR gain.
            # Raw index is noisy because PySR overfits linear windows with
            # nonlinear expressions.  Weighting by effective SR gain suppresses
            # these: if SR doesn't beat linear, the nonlinearity signal is zero.
            raw_jac_nonlin = float(summary.get("jacobian_nonlinearity_index_median", 0.0))
            # SR gain must be OUT-OF-SAMPLE for the gate to mean anything: in-sample,
            # an overfit flexible expression is guaranteed to beat the linear
            # baseline (that is what overfitting is), which would defeat the gate on
            # exactly the linear windows it must catch. The ensemble evaluates both
            # the best symbolic model and the linear baseline on the held-out tail
            # (validation_split); use those held-out errors, so the gate opens only
            # when SR beats linear out of sample. Falls back to in-sample when no
            # holdout is configured (validation_split == 0).
            oos_linear = float(ensemble.linear_fit_mse)
            # gate_symbolic_fit_mse, not best_symbolic_fit_mse: the latter is the
            # min over tail-SELECTED expressions, which double-dips the same held-out
            # tail the gate scores on and biases the gain upward on near-zero-gain
            # (memory/scaling) windows. gate_symbolic_fit_mse is PySR's train-selected
            # winner scored on the tail -- symmetric with the held-out linear baseline.
            oos_symbolic = float(ensemble.gate_symbolic_fit_mse)
            if np.isfinite(oos_linear) and np.isfinite(oos_symbolic) and oos_linear > 1e-9:
                sr_gain = max(0.0, oos_linear - oos_symbolic) / oos_linear
            else:
                sr_gain = 0.0
            effective_gain = max(0.0, sr_gain - float(cfg.ensemble.sr_gain_threshold))
            summary["sr_gain"] = sr_gain
            summary["jacobian_relative_nonlinearity_index_median"] = raw_jac_nonlin * effective_gain

            summary["seed_instability_median"] = float(ensemble.seed_instability)
            summary["symbolic_backend"] = str(ensemble.backend)
            summary["structural_instability"] = float(
                max(
                    float(summary.get("structural_instability", 0.0)),
                    float(summary.get("behavioral_instability_evidence_median", 0.0)),
                    float(ensemble.seed_instability),
                )
            )
            summary["medoid_equation"] = medoid_equation(feature_table)

            evaluation_row = {
                "window_index": int(window.index),
                "label": str(window.label),
                "start": int(window.start),
                "end": int(window.end),
                "n_models_retained": int(len(all_models)),
                "baseline_fit_mse_median": float(baseline_fit_mse),
                "linear_fit_mse_median": float(linear_fit_mse),
                "best_symbolic_fit_mse_median": float(best_symbolic_fit_mse),
                "linear_symbolic_fit_gap_median": float(linear_fit_mse - best_symbolic_fit_mse),
                "linear_symbolic_gap_median": float(linear_fit_mse - best_symbolic_fit_mse),
                "seed_instability_median": float(ensemble.seed_instability),
                "has_structural_summary": True,
                "symbolic_backend": str(ensemble.backend),
            }
            rows.append(summary)
            evaluation_rows.append(evaluation_row)

            if store is not None:
                proto = best_window_fits.get(int(window.index))
                proto_payload = (
                    None
                    if proto is None
                    else {
                        "label": proto.label,
                        "window_index": proto.window_index,
                        "y_true": np.asarray(proto.y_true, dtype=float),
                        "y_pred": np.asarray(proto.y_pred, dtype=float),
                        "fit_mse": float(proto.fit_mse),
                    }
                )
                store.put(
                    _PIPELINE_PHASE,
                    int(window.index),
                    {
                        "summary": summary,
                        "evaluation_row": evaluation_row,
                        "prototype_fit": proto_payload,
                    },
                )

        raw_window_summaries = pd.DataFrame(rows)
        if raw_window_summaries.empty:
            raise ValueError("Pipeline produced no window summaries.")
        window_evaluation_all = pd.DataFrame(evaluation_rows)

        window_summaries = raw_window_summaries.copy()
        disagreement_col = (
            "behavioral_instability_evidence_median"
            if "behavioral_instability_evidence_median" in window_summaries.columns
            else "structural_instability"
        )
        disagreement = pd.to_numeric(
            window_summaries[disagreement_col],
            errors="coerce",
        ).fillna(0.0)
        window_summaries["is_uncertain"] = disagreement > float(cfg.aggregation.instability_threshold)
        window_summaries["uncertainty_filter_fallback_used"] = False
        if not window_evaluation_all.empty:
            raw_window_indices = set(raw_window_summaries["window_index"].astype(int).tolist())
            kept_window_indices = set(window_summaries["window_index"].astype(int).tolist())
            window_evaluation_all["has_structural_summary"] = window_evaluation_all[
                "window_index"
            ].isin(raw_window_indices)
            window_evaluation_all["included_in_window_summaries"] = window_evaluation_all[
                "window_index"
            ].isin(kept_window_indices)
            window_evaluation_all["excluded_by_uncertainty_filter"] = (
                window_evaluation_all["has_structural_summary"]
                & ~window_evaluation_all["included_in_window_summaries"]
            )
            window_evaluation_all = window_evaluation_all.sort_values("window_index").reset_index(
                drop=True
            )

        regime_profiles = aggregate_regime_profiles(window_summaries)

        excluded_feature_cols = {
            "window_index",
            "start",
            "end",
            "n_models",
            "baseline_fit_mse_median",
            "linear_fit_mse_median",
            "best_symbolic_fit_mse_median",
            "linear_symbolic_fit_gap_median",
            "seed_instability_median",
        }
        excluded_feature_cols.update(
            {
                c.replace("_median", "_mean")
                for c in excluded_feature_cols
                if c.endswith("_median")
            }
        )

        feature_columns = [
            c
            for c in window_summaries.columns
            if (c.endswith("_median") or c.endswith("_mean")) and c not in excluded_feature_cols
        ]

        pairwise_stats = run_pairwise_feature_tests(
            window_summaries=window_summaries,
            feature_columns=feature_columns,
            config=cfg.stats,
        )

        medoid_rows = []
        prototype_fits: list[PrototypeFit] = []
        window_fits = sorted(best_window_fits.values(), key=lambda fit: float(fit.fit_mse))
        for label, group in window_summaries.groupby("label"):
            pick = group.sort_values("best_symbolic_fit_mse_median").iloc[0]
            selected_window_index = int(pick["window_index"])
            medoid_rows.append(
                {
                    "label": str(label),
                    "window_index": selected_window_index,
                    "medoid_equation": str(pick["medoid_equation"]),
                    "best_symbolic_fit_mse": float(pick["best_symbolic_fit_mse_median"]),
                }
            )
            fit_payload = best_window_fits.get(selected_window_index)
            if fit_payload is not None:
                prototype_fits.append(fit_payload)
        medoid_equations = pd.DataFrame(medoid_rows)

        baseline_outputs = run_baseline_suite(contexts, cfg, checkpoint_store=store)

        return ProfilingResult(
            raw_window_summaries=raw_window_summaries,
            window_summaries=window_summaries,
            regime_profiles=regime_profiles,
            pairwise_stats=pairwise_stats,
            medoid_equations=medoid_equations,
            window_evaluation_all=window_evaluation_all,
            baseline_outputs=baseline_outputs,
            prototype_fits=prototype_fits,
            window_fits=window_fits,
        )
