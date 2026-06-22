from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ProfilingConfig, StatsConfig
from .dataio import LabeledTimeSeriesDataset
from .pipeline import ProfilingResult, StructuralProfiler
from .statistics import run_pairwise_feature_tests
from .synthetic import generate_synthetic_regime_series


PRIMARY_TARGETS: dict[str, tuple[str, ...]] = {
    "memory_only": ("lag_mass_median", "long_lag_mass_median"),
    "nonlinearity_only": ("nonlinear_node_count_median", "nonlinear_composition_depth_median"),
    "interaction_only": ("interaction_term_count_median", "interaction_order_median"),
    "parameter_only": (),
    "amplitude_only": (),
    "offset_only": (),
    "noise_only": (),
}

SUPPORT_SENSITIVE_FEATURES: tuple[str, ...] = (
    "nonlinear_node_count_median",
    "nonlinear_composition_depth_median",
    "interaction_term_count_median",
    "interaction_order_median",
    "interaction_depth_median",
)

SPECIFICITY_FEATURES_BY_SCENARIO: dict[str, tuple[str, ...]] = {
    "parameter_only": SUPPORT_SENSITIVE_FEATURES,
    "amplitude_only": SUPPORT_SENSITIVE_FEATURES + ("lag_mass_median", "long_lag_mass_median"),
    "offset_only": SUPPORT_SENSITIVE_FEATURES + ("lag_mass_median", "long_lag_mass_median"),
}


EXPECTED_SIGNS: dict[str, int] = {
    "memory_only": 1,
    "nonlinearity_only": 1,
    "interaction_only": 1,
}

VALIDATION_STOP_AFTER_STAGES: tuple[str, ...] = (
    "synthetic.validation_matrix",
    "synthetic.targeted_sensitivity",
    "synthetic.specificity_checks",
    "synthetic.seed_stability",
    "synthetic.nuisance_robustness",
    "synthetic.block_sensitivity",
    "real.block_sensitivity",
    "real.record_replication",
    "claim_audit",
)


@dataclass(slots=True)
class ValidationResult:
    validation_matrix: pd.DataFrame
    targeted_sensitivity: pd.DataFrame
    specificity_checks: pd.DataFrame
    seed_stability: pd.DataFrame
    nuisance_robustness: pd.DataFrame
    synthetic_diagnostics: pd.DataFrame
    block_sensitivity: pd.DataFrame
    record_replication: pd.DataFrame
    claim_audit: pd.DataFrame


def validation_stage_choices() -> tuple[str, ...]:
    return VALIDATION_STOP_AFTER_STAGES


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()


def _load_existing_started_at(output_dir: str | Path) -> datetime | None:
    status_path = Path(output_dir) / "validation_status.json"
    if not status_path.exists():
        return None
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = payload.get("started_at_utc")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _load_validation_checkpoint(output_dir: str | Path) -> ValidationResult:
    out = Path(output_dir)
    return ValidationResult(
        validation_matrix=_read_optional_csv(out / "validation_matrix.csv"),
        targeted_sensitivity=_read_optional_csv(out / "targeted_sensitivity.csv"),
        specificity_checks=_read_optional_csv(out / "specificity_checks.csv"),
        seed_stability=_read_optional_csv(out / "seed_stability.csv"),
        nuisance_robustness=_read_optional_csv(out / "nuisance_robustness.csv"),
        synthetic_diagnostics=_read_optional_csv(out / "synthetic_diagnostics.csv"),
        block_sensitivity=_read_optional_csv(out / "block_sensitivity.csv"),
        record_replication=_read_optional_csv(out / "record_replication.csv"),
        claim_audit=pd.DataFrame(),
    )


def _concept_features(config: ProfilingConfig) -> list[str]:
    return list(config.validation.concept_features)


def _merge_frames(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new.copy()
    if new.empty:
        return existing.copy()
    return pd.concat([existing, new], ignore_index=True)


def _nuisance_resume_key(perturbation: str, parameter_value: float) -> tuple[str, str]:
    if np.isfinite(parameter_value):
        return str(perturbation), f"{float(parameter_value):.12g}"
    return str(perturbation), "__nan__"


def _stage_allowed_for_suite(stage: str, suite: str) -> bool:
    if stage == "claim_audit":
        return True
    if suite == "synthetic":
        return stage.startswith("synthetic.")
    if suite == "real":
        return stage.startswith("real.")
    return stage.startswith(("synthetic.", "real."))


def _validation_matrix_complete(validation_matrix: pd.DataFrame, config: ProfilingConfig) -> bool:
    if validation_matrix.empty or "scenario" not in validation_matrix.columns:
        return False
    completed = set(validation_matrix["scenario"].astype(str).tolist())
    expected = {str(spec["scenario"]) for spec in _scenario_specs(config)}
    return expected.issubset(completed)


def _targeted_sensitivity_complete(targeted_sensitivity: pd.DataFrame, config: ProfilingConfig) -> bool:
    if targeted_sensitivity.empty or not {"scenario", "parameter_value"}.issubset(targeted_sensitivity.columns):
        return False
    completed = {
        (str(scenario), float(parameter_value))
        for scenario, parameter_value in targeted_sensitivity[["scenario", "parameter_value"]].itertuples(index=False, name=None)
    }
    expected = {
        (str(spec["scenario"]), float(value))
        for spec in _targeted_sweep_specs(config)
        for value in spec["values"]
    }
    return expected.issubset(completed)


def _nuisance_robustness_complete(nuisance_robustness: pd.DataFrame, config: ProfilingConfig) -> bool:
    if nuisance_robustness.empty or not {"perturbation", "parameter_value"}.issubset(nuisance_robustness.columns):
        return False
    completed = {
        _nuisance_resume_key(str(perturbation), float(parameter_value))
        for perturbation, parameter_value in nuisance_robustness[["perturbation", "parameter_value"]].itertuples(index=False, name=None)
    }
    expected: set[tuple[str, str]] = set()
    for value in config.validation.nuisance_scale_levels:
        expected.add(_nuisance_resume_key("scale", float(value)))
    for value in config.validation.nuisance_offset_levels:
        expected.add(_nuisance_resume_key("offset", float(value)))
    expected.add(_nuisance_resume_key("jitter", float(config.validation.nuisance_jitter_steps)))
    expected.add(_nuisance_resume_key("dropout", float(config.validation.nuisance_dropout_fraction)))
    noise_scale = float(max(config.validation.noise_levels) / max(config.validation.synthetic_noise_std, 1e-9))
    expected.add(_nuisance_resume_key("noise_only", noise_scale))
    expected.add(_nuisance_resume_key("nonlinearity_only", float("nan")))
    expected.add(_nuisance_resume_key("interaction_only", float("nan")))
    return expected.issubset(completed)


def _clone_validation_config(
    base_config: ProfilingConfig,
    *,
    seed_offset: int | None = None,
    synthetic_profile: bool = False,
) -> ProfilingConfig:
    cfg = copy.deepcopy(base_config)
    cfg.baselines.run_baselines = False
    if seed_offset is not None:
        cfg.ensemble.seed_offset = int(seed_offset)
    if synthetic_profile:
        cfg.ensemble.max_lag = max(
            int(cfg.ensemble.max_lag),
            int(max(cfg.validation.memory_lag_levels)),
        )
        cfg.ensemble.n_seeds = max(int(cfg.ensemble.n_seeds), 4)
        cfg.ensemble.n_candidates_per_seed = max(int(cfg.ensemble.n_candidates_per_seed), 160)
        cfg.ensemble.top_k_per_seed = max(
            int(cfg.ensemble.top_k_per_seed),
            int(cfg.validation.synthetic_top_k_per_seed),
        )
        cfg.ensemble.pysr_populations = max(int(cfg.ensemble.pysr_populations), 10)
        cfg.ensemble.pysr_population_size = max(int(cfg.ensemble.pysr_population_size), 64)
        cfg.ensemble.pysr_maxdepth = max(int(cfg.ensemble.pysr_maxdepth), 6)
        cfg.ensemble.pysr_maxsize = min(
            int(cfg.ensemble.pysr_maxsize),
            int(cfg.validation.synthetic_pysr_maxsize),
        )
        cfg.ensemble.pysr_parsimony = max(
            float(cfg.ensemble.pysr_parsimony),
            float(cfg.validation.synthetic_pysr_parsimony),
        )
        cfg.ensemble.pysr_ncycles_per_iteration = max(int(cfg.ensemble.pysr_ncycles_per_iteration), 160)
        cfg.ensemble.pysr_tournament_selection_n = max(int(cfg.ensemble.pysr_tournament_selection_n), 12)
        cfg.ensemble.pysr_binary_operators = tuple(cfg.validation.synthetic_binary_operators)
        cfg.ensemble.pysr_unary_operators = tuple(cfg.validation.synthetic_unary_operators)
        cfg.ensemble.window_summary_fit_band_ratio = min(
            float(cfg.ensemble.window_summary_fit_band_ratio),
            float(cfg.validation.synthetic_window_summary_fit_band_ratio),
        )
        cfg.ensemble.rebalance_retained_model_families = True
        cfg.ensemble.retained_simple_model_fit_ratio = float(
            cfg.validation.synthetic_simple_model_fit_ratio
        )
        cfg.ensemble.window_summary_guard_complex_models = True
        cfg.ensemble.window_summary_prefer_simple_models = True
        cfg.ensemble.window_summary_simple_fit_ratio = float(
            cfg.validation.synthetic_simple_summary_fit_ratio
        )
        cfg.ensemble.window_summary_simple_fallback_ratio = float(
            cfg.validation.synthetic_simple_summary_fallback_ratio
        )
        cfg.ensemble.window_summary_complex_over_simple_min_ratio = float(
            cfg.validation.synthetic_complex_over_simple_min_ratio
        )
        cfg.ensemble.window_summary_complex_over_simple_min_abs = float(
            cfg.validation.synthetic_complex_over_simple_min_abs
        )
        cfg.ensemble.window_summary_linear_advantage_min_ratio = float(
            cfg.validation.synthetic_linear_advantage_min_ratio
        )
        cfg.ensemble.window_summary_linear_advantage_min_abs = float(
            cfg.validation.synthetic_linear_advantage_min_abs
        )
        cfg.features.counterfactual_structure_evidence = True
        cfg.features.unary_structure_evidence_min_ratio = float(
            cfg.validation.synthetic_unary_structure_evidence_min_ratio
        )
        cfg.features.unary_structure_evidence_min_abs = float(
            cfg.validation.synthetic_unary_structure_evidence_min_abs
        )
        cfg.features.interaction_structure_evidence_min_ratio = float(
            cfg.validation.synthetic_interaction_structure_evidence_min_ratio
        )
        cfg.features.interaction_structure_evidence_min_abs = float(
            cfg.validation.synthetic_interaction_structure_evidence_min_abs
        )
    return cfg


def _profile(
    config: ProfilingConfig,
    values: np.ndarray,
    labels: np.ndarray,
    record_ids: np.ndarray | None = None,
) -> ProfilingResult:
    profiler = StructuralProfiler(config)
    return profiler.fit(values, labels, record_ids=record_ids)


def _feature_columns(window_summaries: pd.DataFrame, features: Iterable[str]) -> list[str]:
    return [feature for feature in features if feature in window_summaries.columns]


def _pairwise_row(pairwise_stats: pd.DataFrame, feature: str) -> pd.Series | None:
    if pairwise_stats.empty or feature not in set(pairwise_stats["feature"].astype(str)):
        return None
    subset = pairwise_stats[pairwise_stats["feature"].astype(str) == feature]
    if subset.empty:
        return None
    return subset.iloc[0]


def _signed_effect_ratio(row: pd.Series | None) -> float:
    if row is None:
        return 0.0
    ratio = float(row.get("effect_iqr_ratio", np.nan))
    if not np.isfinite(ratio):
        return 0.0
    diff = float(row.get("median_diff", 0.0))
    if diff == 0.0:
        return 0.0
    return float(np.sign(diff) * ratio)


def _median_diff(row: pd.Series | None) -> float:
    if row is None:
        return 0.0
    return float(row.get("median_diff", 0.0))


def _safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    sx = pd.Series(list(x), dtype=float)
    sy = pd.Series(list(y), dtype=float)
    if len(sx) < 2 or len(sy) < 2:
        return float("nan")
    if sx.nunique(dropna=True) < 2 or sy.nunique(dropna=True) < 2:
        return float("nan")
    rx = sx.rank(method="average")
    ry = sy.rank(method="average")
    corr = rx.corr(ry, method="pearson")
    return float(corr) if corr is not None else float("nan")


def _scenario_specs(config: ProfilingConfig) -> list[dict[str, Any]]:
    validation = config.validation
    base_noise = float(validation.synthetic_noise_std)
    noise_scale = max(validation.noise_levels) / max(base_noise, 1e-9)
    return [
        {
            "scenario": "memory_only",
            "display_name": "memory_depth",
            "extra": {"memory_lag_b": int(max(validation.memory_lag_levels))},
        },
        {
            "scenario": "nonlinearity_only",
            "display_name": "nonlinearity_strength",
            "extra": {"nonlinearity_strength_b": float(max(validation.nonlinearity_levels))},
        },
        {
            "scenario": "interaction_only",
            "display_name": "lag_interaction",
            "extra": {"interaction_strength_b": float(max(validation.interaction_levels))},
        },
        {
            "scenario": "parameter_only",
            "display_name": "parameter_only",
            "extra": {},
        },
        {
            "scenario": "amplitude_only",
            "display_name": "amplitude_only",
            "extra": {"amplitude_scale_b": float(max(validation.amplitude_levels))},
        },
        {
            "scenario": "offset_only",
            "display_name": "offset_only",
            "extra": {"offset_b": float(max(validation.offset_levels))},
        },
        {
            "scenario": "noise_only",
            "display_name": "noise_only",
            "extra": {"noise_scale_b": float(noise_scale)},
        },
    ]


def _synthetic_dataset(
    config: ProfilingConfig,
    *,
    scenario: str,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> LabeledTimeSeriesDataset:
    kwargs: dict[str, Any] = {
        "n_steps": int(config.validation.synthetic_n_steps),
        "scenario": scenario,
        "noise_std": float(config.validation.synthetic_noise_std),
        "seed": int(seed),
        "regime_block_size": int(config.window.window_size + config.window.step_size),
    }
    if extra:
        kwargs.update(extra)
    dataset = generate_synthetic_regime_series(**kwargs)
    return LabeledTimeSeriesDataset(
        values=dataset.values,
        labels=dataset.labels,
        feature_names=dataset.feature_names,
        record_ids=None,
    )


def _synthetic_diagnostic_rows(
    result: ProfilingResult,
    *,
    source: str,
    scenario: str,
    parameter: str | None = None,
    parameter_value: float | None = None,
) -> list[dict[str, Any]]:
    if result.window_summaries.empty:
        return []

    candidate_cols = [
        "window_index",
        "label",
        "n_models",
        "summary_model_count",
        "summary_model_count_pre_guard",
        "summary_model_count_post_guard",
        "summary_selection_mode",
        "adequate_simple_model_count",
        "best_simple_fit_ratio_to_best_selected",
        "best_simple_fit_mse",
        "best_complex_fit_mse",
        "complex_gain_over_simple",
        "complex_gain_over_simple_threshold",
        "nonlinear_candidate_count",
        "interaction_candidate_count",
        "nonlinear_pool_admitted",
        "interaction_pool_admitted",
        "best_selected_unary_curvature_gain",
        "best_selected_interaction_synergy_gain",
        "unary_curvature_threshold",
        "interaction_synergy_threshold",
        "best_nonlinear_candidate_fit_mse",
        "best_non_nonlinear_fit_mse",
        "nonlinear_admission_gain",
        "nonlinear_admission_threshold",
        "best_interaction_candidate_fit_mse",
        "best_non_interaction_fit_mse",
        "interaction_admission_gain",
        "interaction_admission_threshold",
        "nonlinear_pool_rejected_candidate_count",
        "interaction_pool_rejected_candidate_count",
        "fraction_models_supported_unary",
        "fraction_models_supported_interaction",
        "lag_pool_model_count",
        "nonlinear_pool_model_count",
        "interaction_pool_model_count",
        "fraction_adequate_models_supported_unary",
        "fraction_adequate_models_supported_interaction",
        "fraction_nonlinear_pool_models_raw_unary",
        "fraction_interaction_pool_models_raw_interaction",
        "best_equation",
        "medoid_equation",
        "fraction_models_nonlinear",
        "fraction_models_simple",
        "fraction_models_unary_nonlinear",
        "fraction_models_interaction",
        "fraction_models_coupled_interaction",
        "fraction_models_raw_nonlinear",
        "fraction_models_raw_simple",
        "fraction_models_raw_unary_nonlinear",
        "fraction_models_raw_interaction",
        "retained_model_count_simple",
        "retained_model_count_unary",
        "retained_model_count_interaction",
        "retained_model_count_raw_simple",
        "retained_model_count_raw_unary",
        "retained_model_count_raw_interaction",
        "fraction_summary_models_simple",
        "fraction_summary_models_unary_nonlinear",
        "fraction_summary_models_interaction",
        "fraction_summary_models_coupled_interaction",
        "fraction_summary_models_raw_simple",
        "fraction_summary_models_raw_unary_nonlinear",
        "fraction_summary_models_raw_interaction",
        "fraction_summary_models_supported_unary",
        "fraction_summary_models_supported_interaction",
        "fraction_models_using_max_lag",
        "best_symbolic_fit_mse_median",
        "linear_fit_mse_median",
        "linear_symbolic_fit_gap_median",
        "symbolic_backend",
    ]
    cols = [col for col in candidate_cols if col in result.window_summaries.columns]
    rows: list[dict[str, Any]] = []
    for _, row in result.window_summaries[cols].iterrows():
        payload = row.to_dict()
        payload["diagnostic_source"] = str(source)
        payload["scenario"] = str(scenario)
        payload["parameter"] = str(parameter) if parameter is not None else ""
        payload["parameter_value"] = float(parameter_value) if parameter_value is not None else float("nan")
        rows.append(payload)
    return rows


def _build_validation_matrix(
    base_config: ProfilingConfig,
    *,
    existing: pd.DataFrame | None = None,
    existing_diagnostics: pd.DataFrame | None = None,
    checkpoint: Callable[[pd.DataFrame, pd.DataFrame], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _concept_features(base_config)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    completed_scenarios: set[str] = set()
    if existing is not None and not existing.empty:
        rows.extend(existing.to_dict(orient="records"))
        if "scenario" in existing.columns:
            completed_scenarios = set(existing["scenario"].astype(str).tolist())
    if existing_diagnostics is not None and not existing_diagnostics.empty:
        diagnostics.extend(existing_diagnostics.to_dict(orient="records"))

    def flush() -> None:
        if checkpoint is not None:
            checkpoint(pd.DataFrame(rows), pd.DataFrame(diagnostics))

    for spec in _scenario_specs(base_config):
        scenario = str(spec["scenario"])
        if scenario in completed_scenarios:
            continue
        dataset = _synthetic_dataset(
            base_config,
            scenario=scenario,
            seed=0,
            extra=dict(spec["extra"]),
        )
        result = _profile(
            _clone_validation_config(base_config, synthetic_profile=True),
            dataset.values,
            dataset.labels,
            record_ids=dataset.record_ids,
        )
        diagnostics.extend(
            _synthetic_diagnostic_rows(
                result,
                source="validation_matrix",
                scenario=scenario,
            )
        )

        row: dict[str, Any] = {
            "scenario": scenario,
            "display_name": str(spec["display_name"]),
            "target_features": ",".join(PRIMARY_TARGETS.get(scenario, ())),
        }
        for feature in features:
            pairwise = _pairwise_row(result.pairwise_stats, feature)
            row[feature] = _signed_effect_ratio(pairwise)
        rows.append(row)
        completed_scenarios.add(scenario)
        flush()

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def _targeted_sweep_specs(config: ProfilingConfig) -> list[dict[str, Any]]:
    validation = config.validation
    return [
        {
            "scenario": "memory_only",
            "parameter": "memory_lag_b",
            "values": tuple(int(v) for v in validation.memory_lag_levels),
            "targets": PRIMARY_TARGETS["memory_only"],
        },
        {
            "scenario": "nonlinearity_only",
            "parameter": "nonlinearity_strength_b",
            "values": tuple(float(v) for v in validation.nonlinearity_levels),
            "targets": PRIMARY_TARGETS["nonlinearity_only"],
        },
        {
            "scenario": "interaction_only",
            "parameter": "interaction_strength_b",
            "values": tuple(float(v) for v in validation.interaction_levels),
            "targets": PRIMARY_TARGETS["interaction_only"],
        },
    ]


def _build_targeted_sensitivity(
    base_config: ProfilingConfig,
    *,
    existing: pd.DataFrame | None = None,
    existing_diagnostics: pd.DataFrame | None = None,
    checkpoint: Callable[[pd.DataFrame, pd.DataFrame], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _concept_features(base_config)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    completed_tasks: set[tuple[str, float]] = set()
    if existing is not None and not existing.empty:
        rows.extend(existing.to_dict(orient="records"))
        if {"scenario", "parameter_value"}.issubset(existing.columns):
            completed_tasks = {
                (str(scenario), float(parameter_value))
                for scenario, parameter_value in existing[["scenario", "parameter_value"]].itertuples(index=False, name=None)
            }
    if existing_diagnostics is not None and not existing_diagnostics.empty:
        diagnostics.extend(existing_diagnostics.to_dict(orient="records"))

    def flush() -> None:
        if checkpoint is not None:
            checkpoint(pd.DataFrame(rows), pd.DataFrame(diagnostics))

    for spec in _targeted_sweep_specs(base_config):
        scenario = str(spec["scenario"])
        parameter = str(spec["parameter"])
        for value in spec["values"]:
            task_key = (scenario, float(value))
            if task_key in completed_tasks:
                continue
            dataset = _synthetic_dataset(
                base_config,
                scenario=scenario,
                seed=0,
                extra={parameter: value},
            )
            result = _profile(
                _clone_validation_config(base_config, synthetic_profile=True),
                dataset.values,
                dataset.labels,
                record_ids=dataset.record_ids,
            )
            diagnostics.extend(
                _synthetic_diagnostic_rows(
                    result,
                    source="targeted_sensitivity",
                    scenario=scenario,
                    parameter=parameter,
                    parameter_value=float(value),
                )
            )
            for feature in features:
                pairwise = _pairwise_row(result.pairwise_stats, feature)
                rows.append(
                    {
                        "scenario": scenario,
                        "parameter": parameter,
                        "parameter_value": float(value),
                        "feature": feature,
                        "is_primary_target": feature in set(spec["targets"]),
                        "expected_sign": int(EXPECTED_SIGNS.get(scenario, 0)),
                        "median_diff": _median_diff(pairwise),
                        "standardized_effect": _signed_effect_ratio(pairwise),
                        "abs_standardized_effect": abs(_signed_effect_ratio(pairwise)),
                        "q_value": float(pairwise.get("q_value", 1.0)) if pairwise is not None else 1.0,
                    }
                )
            completed_tasks.add(task_key)
            flush()
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def _build_specificity_checks(validation_matrix: pd.DataFrame, config: ProfilingConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    threshold = float(config.validation.specificity_effect_ratio_max)
    for scenario, features in SPECIFICITY_FEATURES_BY_SCENARIO.items():
        subset = validation_matrix[validation_matrix["scenario"].astype(str) == scenario]
        if subset.empty:
            continue
        row = subset.iloc[0]
        feature_effects = {feature: abs(float(row.get(feature, 0.0))) for feature in features}
        for feature, effect in feature_effects.items():
            rows.append(
                {
                    "row_type": "feature",
                    "scenario": scenario,
                    "feature": feature,
                    "abs_effect_ratio": float(effect),
                    "passes_specificity": bool(effect <= threshold),
                }
            )
        max_effect = float(max(feature_effects.values())) if feature_effects else 0.0
        rows.append(
            {
                "row_type": "summary",
                "scenario": scenario,
                "feature": "__summary__",
                "abs_effect_ratio": max_effect,
                "passes_specificity": bool(max_effect <= threshold),
            }
        )
    return pd.DataFrame(rows)


def _concept_vector_distances(
    left: pd.DataFrame,
    right: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    join_cols = ["window_index", "label"]
    if "record_id" in left.columns and "record_id" in right.columns:
        join_cols.append("record_id")

    lhs = left[join_cols + features].copy()
    rhs = right[join_cols + features].copy()
    merged = lhs.merge(rhs, on=join_cols, how="inner", suffixes=("_left", "_right"))
    if merged.empty:
        return np.array([], dtype=float)

    deltas = []
    for feature in features:
        deltas.append(
            (
                merged[f"{feature}_left"].to_numpy(dtype=float)
                - merged[f"{feature}_right"].to_numpy(dtype=float)
            )
            ** 2
        )
    return np.sqrt(np.sum(np.vstack(deltas), axis=0))


def _cross_label_distances(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    if df.empty or "label" not in df.columns:
        return np.array([], dtype=float)

    labels = sorted(df["label"].astype(str).unique().tolist())
    if len(labels) < 2:
        return np.array([], dtype=float)

    a = df[df["label"].astype(str) == labels[0]][features].to_numpy(dtype=float)
    b = df[df["label"].astype(str) == labels[1]][features].to_numpy(dtype=float)
    if a.size == 0 or b.size == 0:
        return np.array([], dtype=float)

    distances = []
    for vec_a in a:
        diff = b - vec_a[None, :]
        distances.extend(np.sqrt(np.sum(diff**2, axis=1)).tolist())
    return np.asarray(distances, dtype=float)


def _build_seed_stability(base_config: ProfilingConfig) -> pd.DataFrame:
    features = _concept_features(base_config)
    dataset = _synthetic_dataset(
        base_config,
        scenario="memory_only",
        seed=17,
        extra={"memory_lag_b": int(max(base_config.validation.memory_lag_levels))},
    )

    run_tables: list[pd.DataFrame] = []
    seed_values = tuple(int(seed) for seed in base_config.validation.seed_values)
    for run_seed in seed_values:
        cfg = _clone_validation_config(
            base_config,
            seed_offset=int(base_config.ensemble.seed_offset + run_seed * 1000),
            synthetic_profile=True,
        )
        result = _profile(cfg, dataset.values, dataset.labels, record_ids=dataset.record_ids)
        table = result.raw_window_summaries[["window_index", "label", *features]].copy()
        table["seed_run"] = int(run_seed)
        run_tables.append(table)

    combined = pd.concat(run_tables, ignore_index=True)
    rows: list[dict[str, Any]] = []

    for feature in features:
        pivot = combined.pivot_table(index=["window_index", "label"], columns="seed_run", values=feature)
        pivot = pivot.dropna(how="any")
        if pivot.empty:
            continue

        values = pivot.to_numpy(dtype=float)
        mean_abs = np.mean(np.abs(values), axis=1)
        row_cv = np.std(values, axis=1) / np.maximum(mean_abs, 1e-9)
        row_iqr = np.quantile(values, 0.75, axis=1) - np.quantile(values, 0.25, axis=1)

        rank_scores = []
        for idx, left_seed in enumerate(seed_values):
            for right_seed in seed_values[idx + 1 :]:
                pair = pivot[[left_seed, right_seed]].dropna()
                if len(pair) < 2:
                    continue
                corr = _safe_spearman(pair[left_seed].tolist(), pair[right_seed].tolist())
                if np.isfinite(corr):
                    rank_scores.append(corr)

        rank_mean = float(np.mean(rank_scores)) if rank_scores else float("nan")
        rows.append(
            {
                "row_type": "feature",
                "feature": feature,
                "n_windows_matched": int(len(pivot)),
                "median_window_cv": float(np.median(row_cv)),
                "median_window_iqr": float(np.median(row_iqr)),
                "mean_rank_spearman": rank_mean,
                "passes_rank_stability": bool(
                    np.isfinite(rank_mean)
                    and rank_mean >= float(base_config.validation.seed_rank_min_spearman)
                ),
                "passes_cv_stability": bool(
                    float(np.median(row_cv)) <= float(base_config.validation.seed_window_cv_max)
                ),
            }
        )

    reference = run_tables[0]
    seed_drifts = []
    for idx, left in enumerate(run_tables):
        for right in run_tables[idx + 1 :]:
            distances = _concept_vector_distances(left, right, features)
            if distances.size:
                seed_drifts.append(float(np.median(distances)))

    regime_distances = _cross_label_distances(reference, features)
    median_seed_drift = float(np.median(seed_drifts)) if seed_drifts else float("nan")
    median_regime_drift = (
        float(np.median(regime_distances)) if regime_distances.size else float("nan")
    )
    ratio = (
        float(median_seed_drift / median_regime_drift)
        if np.isfinite(median_seed_drift) and np.isfinite(median_regime_drift) and median_regime_drift > 1e-12
        else float("nan")
    )
    rows.append(
        {
            "row_type": "summary",
            "feature": "__seed_vs_regime_drift__",
            "n_windows_matched": int(len(reference)),
            "median_window_cv": float("nan"),
            "median_window_iqr": float("nan"),
            "mean_rank_spearman": float("nan"),
            "median_seed_drift": median_seed_drift,
            "median_regime_drift": median_regime_drift,
            "seed_vs_regime_ratio": ratio,
            "passes_seed_vs_regime": bool(
                np.isfinite(ratio)
                and ratio <= float(base_config.validation.seed_vs_regime_drift_ratio_max)
            ),
        }
    )
    return pd.DataFrame(rows)


def _smooth_jitter(values: np.ndarray, n_steps: int) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for _ in range(max(1, int(n_steps))):
        prev_vals = np.vstack([out[:1], out[:-1]])
        next_vals = np.vstack([out[1:], out[-1:]])
        out = 0.5 * out + 0.25 * (prev_vals + next_vals)
    return out


def _dropout_interpolate(values: np.ndarray, fraction: float) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    n = out.shape[0]
    step = max(3, int(round(1.0 / max(float(fraction), 1e-6))))
    mask = np.ones(n, dtype=bool)
    mask[step - 1 : n - 1 : step] = False
    mask[0] = True
    mask[-1] = True
    x = np.arange(n, dtype=float)
    keep_idx = np.where(mask)[0]
    for col in range(out.shape[1]):
        out[:, col] = np.interp(x, keep_idx.astype(float), out[keep_idx, col])
    return out


def _build_nuisance_robustness(
    base_config: ProfilingConfig,
    *,
    existing: pd.DataFrame | None = None,
    checkpoint: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    features = _concept_features(base_config)
    base_dataset = _synthetic_dataset(
        base_config,
        scenario="memory_only",
        seed=29,
        extra={"memory_lag_b": int(max(base_config.validation.memory_lag_levels))},
    )
    base_result = _profile(
        _clone_validation_config(base_config, synthetic_profile=True),
        base_dataset.values,
        base_dataset.labels,
        record_ids=base_dataset.record_ids,
    )
    base_table = base_result.raw_window_summaries[["window_index", "label", *features]].copy()

    rows: list[dict[str, Any]] = []
    completed_tasks: set[tuple[str, str]] = set()
    if existing is not None and not existing.empty:
        rows.extend(existing.to_dict(orient="records"))
        if {"perturbation", "parameter_value"}.issubset(existing.columns):
            completed_tasks = {
                _nuisance_resume_key(str(perturbation), float(parameter_value))
                for perturbation, parameter_value in existing[["perturbation", "parameter_value"]].itertuples(index=False, name=None)
            }

    def flush() -> None:
        if checkpoint is not None:
            checkpoint(pd.DataFrame(rows))

    nuisance_specs = [
        ("scale", "nuisance", [float(v) for v in base_config.validation.nuisance_scale_levels]),
        ("offset", "nuisance", [float(v) for v in base_config.validation.nuisance_offset_levels]),
        ("jitter", "nuisance", [float(base_config.validation.nuisance_jitter_steps)]),
        ("dropout", "nuisance", [float(base_config.validation.nuisance_dropout_fraction)]),
    ]

    for perturbation, kind, values in nuisance_specs:
        for value in values:
            task_key = _nuisance_resume_key(perturbation, float(value))
            if task_key in completed_tasks:
                continue
            if perturbation == "scale":
                perturbed_values = base_dataset.values * float(value)
            elif perturbation == "offset":
                perturbed_values = base_dataset.values + float(value)
            elif perturbation == "jitter":
                perturbed_values = _smooth_jitter(base_dataset.values, int(value))
            else:
                perturbed_values = _dropout_interpolate(base_dataset.values, float(value))

            result = _profile(
                _clone_validation_config(base_config, synthetic_profile=True),
                perturbed_values,
                base_dataset.labels,
                record_ids=base_dataset.record_ids,
            )
            distances = _concept_vector_distances(
                base_table,
                result.raw_window_summaries[["window_index", "label", *features]].copy(),
                features,
            )
            rows.append(
                {
                    "perturbation": perturbation,
                    "kind": kind,
                    "parameter_value": float(value),
                    "median_concept_drift": float(np.median(distances)) if distances.size else float("nan"),
                    "n_windows_compared": int(distances.size),
                }
            )
            completed_tasks.add(task_key)
            flush()

    noise_scale = float(max(base_config.validation.noise_levels) / max(base_config.validation.synthetic_noise_std, 1e-9))
    noise_key = _nuisance_resume_key("noise_only", noise_scale)
    if noise_key not in completed_tasks:
        noise_dataset = _synthetic_dataset(
            base_config,
            scenario="noise_only",
            seed=29,
            extra={"noise_scale_b": noise_scale},
        )
        noise_result = _profile(
            _clone_validation_config(base_config, synthetic_profile=True),
            noise_dataset.values,
            noise_dataset.labels,
            record_ids=noise_dataset.record_ids,
        )
        noise_distances = _concept_vector_distances(
            base_table,
            noise_result.raw_window_summaries[["window_index", "label", *features]].copy(),
            features,
        )
        rows.append(
            {
                "perturbation": "noise_only",
                "kind": "nuisance",
                "parameter_value": noise_scale,
                "median_concept_drift": float(np.median(noise_distances)) if noise_distances.size else float("nan"),
                "n_windows_compared": int(noise_distances.size),
            }
        )
        completed_tasks.add(noise_key)
        flush()

    structural_specs = [
        ("nonlinearity_only", {"nonlinearity_strength_b": float(max(base_config.validation.nonlinearity_levels))}),
        ("interaction_only", {"interaction_strength_b": float(max(base_config.validation.interaction_levels))}),
    ]
    for scenario, extra in structural_specs:
        task_key = _nuisance_resume_key(scenario, float("nan"))
        if task_key in completed_tasks:
            continue
        dataset = _synthetic_dataset(base_config, scenario=scenario, seed=29, extra=extra)
        result = _profile(
            _clone_validation_config(base_config, synthetic_profile=True),
            dataset.values,
            dataset.labels,
            record_ids=dataset.record_ids,
        )
        distances = _concept_vector_distances(
            base_table,
            result.raw_window_summaries[["window_index", "label", *features]].copy(),
            features,
        )
        rows.append(
            {
                "perturbation": scenario,
                "kind": "structural",
                "parameter_value": float("nan"),
                "median_concept_drift": float(np.median(distances)) if distances.size else float("nan"),
                "n_windows_compared": int(distances.size),
            }
        )
        completed_tasks.add(task_key)
        flush()

    df = pd.DataFrame(rows)
    structural_reference = df.loc[
        (df["kind"] == "structural") & df["median_concept_drift"].notna(),
        "median_concept_drift",
    ]
    reference = float(np.median(structural_reference.to_numpy(dtype=float))) if not structural_reference.empty else float("nan")
    if not df.empty:
        df["drift_ratio_vs_structural"] = (
            df["median_concept_drift"].astype(float) / reference
            if np.isfinite(reference) and reference > 1e-12
            else np.nan
        )
        df["passes_nuisance_robustness"] = (
            (df["kind"].astype(str) == "structural")
            | (
                df["drift_ratio_vs_structural"].astype(float)
                <= float(base_config.validation.nuisance_vs_structural_ratio_max)
            )
        )
    return df


def _build_block_sensitivity(
    window_summaries: pd.DataFrame,
    base_stats: StatsConfig,
    config: ProfilingConfig,
    *,
    source: str,
) -> pd.DataFrame:
    features = _feature_columns(window_summaries, _concept_features(config))
    rows: list[pd.DataFrame] = []
    if window_summaries.empty or not features:
        return pd.DataFrame()

    for block_size in config.validation.block_size_grid:
        stats_cfg = copy.deepcopy(base_stats)
        stats_cfg.block_size = int(block_size)
        out = run_pairwise_feature_tests(window_summaries, features, stats_cfg)
        if out.empty:
            continue
        out = out.copy()
        out["sensitivity_source"] = str(source)
        out["block_size_tested"] = int(block_size)
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df["direction_consistent_across_grid"] = False
    for _, group in df.groupby(["sensitivity_source", "feature", "group_a", "group_b"]):
        signs = np.sign(group["median_diff"].to_numpy(dtype=float))
        finite_signs = signs[signs != 0]
        consistent = bool(finite_signs.size <= 1 or np.all(finite_signs == finite_signs[0]))
        df.loc[group.index, "direction_consistent_across_grid"] = consistent
    return df


def _build_record_replication(
    dataset: LabeledTimeSeriesDataset,
    base_config: ProfilingConfig,
    *,
    existing: pd.DataFrame | None = None,
    checkpoint: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    features = _concept_features(base_config)
    if dataset.record_ids is None:
        result = pd.DataFrame(
            [
                {
                    "replication_mode": "not_available",
                    "record_id": "n/a",
                    "status": "record_col_missing",
                    "feature": "__summary__",
                }
            ]
        )
        if checkpoint is not None:
            checkpoint(result)
        return result

    record_ids = np.asarray(dataset.record_ids, dtype=str)
    unique_records = sorted(np.unique(record_ids).tolist())
    if len(unique_records) < int(base_config.validation.min_records_for_replication):
        result = pd.DataFrame(
            [
                {
                    "replication_mode": "not_available",
                    "record_id": "n/a",
                    "status": "insufficient_records",
                    "feature": "__summary__",
                }
            ]
        )
        if checkpoint is not None:
            checkpoint(result)
        return result

    rows: list[dict[str, Any]] = []
    completed_tasks: set[tuple[str, str]] = set()
    if existing is not None and not existing.empty:
        rows.extend(existing.to_dict(orient="records"))
        if {"replication_mode", "record_id"}.issubset(existing.columns):
            completed_tasks = {
                (str(mode), str(record_id))
                for mode, record_id in existing[["replication_mode", "record_id"]].astype(str).drop_duplicates().itertuples(index=False, name=None)
            }

    def flush_rows() -> None:
        if checkpoint is not None:
            checkpoint(pd.DataFrame(rows))

    for record_id in unique_records:
        task_key = ("within_record", str(record_id))
        if task_key in completed_tasks:
            continue
        mask = record_ids == record_id
        try:
            result = _profile(
                _clone_validation_config(base_config),
                dataset.values[mask],
                dataset.labels[mask],
            )
            pairwise = result.pairwise_stats.copy()
            pairwise = pairwise[pairwise["feature"].astype(str).isin(features)].copy()
            if pairwise.empty:
                rows.append(
                    {
                        "replication_mode": "within_record",
                        "record_id": record_id,
                        "status": "no_pairwise_stats",
                        "feature": "__summary__",
                    }
                )
            else:
                for _, row in pairwise.iterrows():
                    rows.append(
                        {
                            "replication_mode": "within_record",
                            "record_id": record_id,
                            "status": "ok",
                            "feature": str(row["feature"]),
                            "median_diff": float(row["median_diff"]),
                            "q_value": float(row["q_value"]),
                        }
                    )
        except Exception as exc:
            rows.append(
                {
                    "replication_mode": "within_record",
                    "record_id": record_id,
                    "status": f"skipped:{type(exc).__name__}",
                    "feature": "__summary__",
                }
            )
        completed_tasks.add(task_key)
        flush_rows()

    if len(unique_records) >= 2:
        for held_out in unique_records:
            task_key = ("leave_one_record_out", str(held_out))
            if task_key in completed_tasks:
                continue
            mask = record_ids != held_out
            try:
                result = _profile(
                    _clone_validation_config(base_config),
                    dataset.values[mask],
                    dataset.labels[mask],
                    record_ids=record_ids[mask],
                )
                pairwise = result.pairwise_stats.copy()
                pairwise = pairwise[pairwise["feature"].astype(str).isin(features)].copy()
                if pairwise.empty:
                    rows.append(
                        {
                            "replication_mode": "leave_one_record_out",
                            "record_id": held_out,
                            "status": "no_pairwise_stats",
                            "feature": "__summary__",
                        }
                    )
                else:
                    for _, row in pairwise.iterrows():
                        rows.append(
                            {
                                "replication_mode": "leave_one_record_out",
                                "record_id": held_out,
                                "status": "ok",
                                "feature": str(row["feature"]),
                                "median_diff": float(row["median_diff"]),
                                "q_value": float(row["q_value"]),
                            }
                        )
            except Exception as exc:
                rows.append(
                    {
                        "replication_mode": "leave_one_record_out",
                        "record_id": held_out,
                        "status": f"skipped:{type(exc).__name__}",
                        "feature": "__summary__",
                    }
                )
            completed_tasks.add(task_key)
            flush_rows()
    return pd.DataFrame(rows)


def _has_block_sensitivity_source(block_sensitivity: pd.DataFrame, source: str) -> bool:
    if block_sensitivity.empty or "sensitivity_source" not in block_sensitivity.columns:
        return False
    return bool((block_sensitivity["sensitivity_source"].astype(str) == str(source)).any())


def _expected_record_replication_tasks(
    dataset: LabeledTimeSeriesDataset,
    base_config: ProfilingConfig,
) -> set[tuple[str, str]]:
    if dataset.record_ids is None:
        return {("not_available", "n/a")}
    record_ids = np.asarray(dataset.record_ids, dtype=str)
    unique_records = sorted(np.unique(record_ids).tolist())
    if len(unique_records) < int(base_config.validation.min_records_for_replication):
        return {("not_available", "n/a")}
    tasks = {("within_record", str(record_id)) for record_id in unique_records}
    if len(unique_records) >= 2:
        tasks.update(("leave_one_record_out", str(record_id)) for record_id in unique_records)
    return tasks


def _record_replication_complete(
    record_replication: pd.DataFrame,
    dataset: LabeledTimeSeriesDataset,
    base_config: ProfilingConfig,
) -> bool:
    if record_replication.empty or not {"replication_mode", "record_id"}.issubset(record_replication.columns):
        return False
    completed = {
        (str(mode), str(record_id))
        for mode, record_id in record_replication[["replication_mode", "record_id"]].astype(str).drop_duplicates().itertuples(index=False, name=None)
    }
    return _expected_record_replication_tasks(dataset, base_config).issubset(completed)


def _append_claim_row(
    rows: list[dict[str, Any]],
    *,
    dimension: str,
    check_name: str,
    status: str,
    observed_value: float,
    threshold: float,
    details: str,
    blocking: bool = True,
    priority: str = "core",
) -> None:
    rows.append(
        {
            "dimension": str(dimension),
            "check_name": str(check_name),
            "status": str(status),
            "observed_value": float(observed_value) if np.isfinite(observed_value) else float("nan"),
            "threshold": float(threshold) if np.isfinite(threshold) else float("nan"),
            "details": str(details),
            "blocking": bool(blocking),
            "priority": str(priority),
        }
    )


def _build_claim_audit(
    validation_matrix: pd.DataFrame,
    targeted_sensitivity: pd.DataFrame,
    specificity_checks: pd.DataFrame,
    seed_stability: pd.DataFrame,
    nuisance_robustness: pd.DataFrame,
    block_sensitivity: pd.DataFrame,
    record_replication: pd.DataFrame,
    config: ProfilingConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = _concept_features(config)
    core_scenarios = {"memory_only", "nonlinearity_only"}

    if not validation_matrix.empty and "scenario" in validation_matrix.columns:
        for scenario in ("memory_only", "nonlinearity_only", "interaction_only"):
            subset = validation_matrix[validation_matrix["scenario"].astype(str) == scenario]
            if subset.empty:
                continue
            row = subset.iloc[0]
            targets = PRIMARY_TARGETS[scenario]
            target_abs = max(abs(float(row.get(feature, 0.0))) for feature in targets)
            off_target_abs = max(
                [abs(float(row.get(feature, 0.0))) for feature in features if feature not in set(targets)] or [0.0]
            )
            observed_sign = int(np.sign(float(row.get(targets[0], 0.0)))) if targets else 0
            expected_sign = int(EXPECTED_SIGNS.get(scenario, 0))
            passes = bool(
                target_abs >= float(config.validation.fidelity_target_ratio_min) * max(off_target_abs, 1e-9)
                and observed_sign == expected_sign
            )
            _append_claim_row(
                rows,
                dimension="fidelity",
                check_name=scenario,
                status="pass" if passes else "fail",
                observed_value=target_abs,
                threshold=float(config.validation.fidelity_target_ratio_min) * max(off_target_abs, 1e-9),
                details=f"target_abs={target_abs:.4f}, off_target_abs={off_target_abs:.4f}, sign={observed_sign}",
                blocking=scenario in core_scenarios,
                priority="core" if scenario in core_scenarios else "secondary",
            )

    if not targeted_sensitivity.empty and "scenario" in targeted_sensitivity.columns:
        for scenario in ("memory_only", "nonlinearity_only", "interaction_only"):
            subset = targeted_sensitivity[targeted_sensitivity["scenario"].astype(str) == scenario]
            if subset.empty:
                continue
            target_subset = subset[subset["is_primary_target"]]
            off_subset = subset[~subset["is_primary_target"]]
            if target_subset.empty:
                _append_claim_row(
                    rows,
                    dimension="sensitivity",
                    check_name=scenario,
                    status="fail",
                    observed_value=float("nan"),
                    threshold=float(config.validation.monotonicity_min_spearman),
                    details="no primary-target rows available under current concept feature mapping",
                    blocking=scenario in core_scenarios,
                    priority="core" if scenario in core_scenarios else "secondary",
                )
                continue
            max_target = (
                target_subset.groupby("parameter_value")["abs_standardized_effect"].max().reset_index()
            )
            if max_target.empty:
                _append_claim_row(
                    rows,
                    dimension="sensitivity",
                    check_name=scenario,
                    status="fail",
                    observed_value=float("nan"),
                    threshold=float(config.validation.monotonicity_min_spearman),
                    details="primary-target rows did not yield any parameter sweep summary",
                    blocking=scenario in core_scenarios,
                    priority="core" if scenario in core_scenarios else "secondary",
                )
                continue
            target_spearman = _safe_spearman(
                max_target["parameter_value"].tolist(),
                max_target["abs_standardized_effect"].tolist(),
            )
            max_row = target_subset.loc[target_subset["parameter_value"].idxmax()]
            max_target_effect = float(max_row["abs_standardized_effect"])
            off_at_max = off_subset[off_subset["parameter_value"] == float(max_row["parameter_value"])]
            off_target_effect = (
                float(off_at_max["abs_standardized_effect"].max()) if not off_at_max.empty else 0.0
            )
            passes = bool(
                np.isfinite(target_spearman)
                and target_spearman >= float(config.validation.monotonicity_min_spearman)
                and max_target_effect > off_target_effect
                and int(np.sign(float(max_row["median_diff"]))) == int(EXPECTED_SIGNS.get(scenario, 0))
            )
            _append_claim_row(
                rows,
                dimension="sensitivity",
                check_name=scenario,
                status="pass" if passes else "fail",
                observed_value=float(target_spearman) if np.isfinite(target_spearman) else float("nan"),
                threshold=float(config.validation.monotonicity_min_spearman),
                details=f"max_target_effect={max_target_effect:.4f}, max_off_target_effect={off_target_effect:.4f}",
                blocking=scenario in core_scenarios,
                priority="core" if scenario in core_scenarios else "secondary",
            )

    if not specificity_checks.empty and "row_type" in specificity_checks.columns:
        specificity_summary = specificity_checks[specificity_checks["row_type"].astype(str) == "summary"]
        for _, row in specificity_summary.iterrows():
            _append_claim_row(
                rows,
                dimension="specificity",
                check_name=str(row["scenario"]),
                status="pass" if bool(row["passes_specificity"]) else "fail",
                observed_value=float(row["abs_effect_ratio"]),
                threshold=float(config.validation.specificity_effect_ratio_max),
                details=f"max_abs_effect_ratio={float(row['abs_effect_ratio']):.4f}",
            )

    feature_stability = (
        seed_stability[seed_stability["row_type"].astype(str) == "feature"]
        if not seed_stability.empty and "row_type" in seed_stability.columns
        else pd.DataFrame()
    )
    if not feature_stability.empty:
        mean_rank = float(feature_stability["mean_rank_spearman"].dropna().median())
        mean_cv = float(feature_stability["median_window_cv"].dropna().median())
        _append_claim_row(
            rows,
            dimension="stability",
            check_name="seed_rank",
            status="pass" if mean_rank >= float(config.validation.seed_rank_min_spearman) else "fail",
            observed_value=mean_rank,
            threshold=float(config.validation.seed_rank_min_spearman),
            details=f"median feature rank spearman={mean_rank:.4f}",
        )
        _append_claim_row(
            rows,
            dimension="stability",
            check_name="seed_cv",
            status="pass" if mean_cv <= float(config.validation.seed_window_cv_max) else "fail",
            observed_value=mean_cv,
            threshold=float(config.validation.seed_window_cv_max),
            details=f"median feature CV={mean_cv:.4f}",
        )

    drift_row = (
        seed_stability[seed_stability["feature"].astype(str) == "__seed_vs_regime_drift__"]
        if not seed_stability.empty and "feature" in seed_stability.columns
        else pd.DataFrame()
    )
    if not drift_row.empty:
        ratio = float(drift_row.iloc[0].get("seed_vs_regime_ratio", np.nan))
        _append_claim_row(
            rows,
            dimension="stability",
            check_name="seed_vs_regime",
            status=(
                "pass"
                if np.isfinite(ratio)
                and ratio <= float(config.validation.seed_vs_regime_drift_ratio_max)
                else "fail"
            ),
            observed_value=ratio,
            threshold=float(config.validation.seed_vs_regime_drift_ratio_max),
            details=(
                f"seed_drift={float(drift_row.iloc[0].get('median_seed_drift', np.nan)):.4f}, "
                f"regime_drift={float(drift_row.iloc[0].get('median_regime_drift', np.nan)):.4f}"
            ),
        )

    nuisance_subset = (
        nuisance_robustness[nuisance_robustness["kind"].astype(str) == "nuisance"]
        if not nuisance_robustness.empty and "kind" in nuisance_robustness.columns
        else pd.DataFrame()
    )
    if not nuisance_subset.empty:
        max_ratio = float(nuisance_subset["drift_ratio_vs_structural"].dropna().max())
        _append_claim_row(
            rows,
            dimension="stability",
            check_name="nuisance_vs_structural",
            status=(
                "pass"
                if np.isfinite(max_ratio)
                and max_ratio <= float(config.validation.nuisance_vs_structural_ratio_max)
                else "fail"
            ),
            observed_value=max_ratio,
            threshold=float(config.validation.nuisance_vs_structural_ratio_max),
            details=f"max nuisance drift ratio={max_ratio:.4f}",
        )

    if not block_sensitivity.empty:
        grouped = block_sensitivity.groupby(["sensitivity_source", "feature"], as_index=False).agg(
            direction_consistent=("direction_consistent_across_grid", "min")
        )
        consistency = float(grouped["direction_consistent"].mean()) if not grouped.empty else float("nan")
        _append_claim_row(
            rows,
            dimension="inference",
            check_name="block_sensitivity",
            status="pass" if consistency == 1.0 else "fail",
            observed_value=consistency,
            threshold=1.0,
            details=f"direction-consistent groups={int(grouped['direction_consistent'].sum())}/{len(grouped)}",
        )

    if not record_replication.empty and "status" in record_replication.columns:
        ok = record_replication[record_replication["status"].astype(str) == "ok"]
        if not ok.empty and "median_diff" in ok.columns:
            sign_agreement_rows = []
            for feature, group in ok.groupby("feature"):
                signs = np.sign(group["median_diff"].to_numpy(dtype=float))
                finite = signs[signs != 0]
                agreement = float(np.mean(finite == finite[0])) if finite.size else float("nan")
                sign_agreement_rows.append(agreement)
            finite_agreement = [value for value in sign_agreement_rows if np.isfinite(value)]
            sign_agreement = (
                float(np.mean(finite_agreement)) if finite_agreement else float("nan")
            )
            _append_claim_row(
                rows,
                dimension="inference",
                check_name="record_replication",
                status=(
                    "pass"
                    if np.isfinite(sign_agreement)
                    and sign_agreement >= float(config.validation.record_sign_consistency_min)
                    else "fail"
                ),
                observed_value=sign_agreement,
                threshold=float(config.validation.record_sign_consistency_min),
                details=f"mean sign agreement across records={sign_agreement:.4f}",
            )
        elif not ok.empty:
            _append_claim_row(
                rows,
                dimension="inference",
                check_name="record_replication",
                status="pass",
                observed_value=float("nan"),
                threshold=float(config.validation.record_sign_consistency_min),
                details="record-level replication executed without feature-level effects",
            )

    return pd.DataFrame(rows)


def _plot_validation_matrix(validation_matrix: pd.DataFrame, config: ProfilingConfig, output_path: Path) -> None:
    if validation_matrix.empty:
        return

    features = _concept_features(config)
    data = validation_matrix[features].to_numpy(dtype=float)
    vmax = float(np.max(np.abs(data))) if data.size else 1.0
    vmax = max(vmax, 1.0)

    fig_h = max(4.0, 0.55 * len(validation_matrix) + 1.8)
    fig, ax = plt.subplots(figsize=(11.2, fig_h))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_yticks(np.arange(len(validation_matrix)))
    ax.set_yticklabels(validation_matrix["display_name"].astype(str).tolist())
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels([feature.replace("_median", "") for feature in features], rotation=45, ha="right")
    ax.set_title("Concept-Generator Validation Matrix")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Signed standardized effect")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.4f}"
    return str(value)


def _synthetic_support_lines(synthetic_diagnostics: pd.DataFrame) -> list[str]:
    if synthetic_diagnostics.empty:
        return []
    required_cols = {"scenario", "summary_model_count", "n_models"}
    if not required_cols.issubset(synthetic_diagnostics.columns):
        return []

    support_df = synthetic_diagnostics.copy()
    if "diagnostic_source" in support_df.columns:
        validation_matrix_df = support_df[support_df["diagnostic_source"].astype(str) == "validation_matrix"].copy()
        if not validation_matrix_df.empty:
            support_df = validation_matrix_df

    lines = ["## Synthetic Ensemble Support"]
    core_scenarios = {"memory_only", "nonlinearity_only"}
    for scenario in sorted(support_df["scenario"].astype(str).unique().tolist()):
        group = support_df[support_df["scenario"].astype(str) == scenario].copy()
        if group.empty:
            continue
        median_summary_model_count = float(group["summary_model_count"].astype(float).median())
        median_n_models = float(group["n_models"].astype(float).median())
        lines.append(
            f"- {scenario}: median summary_model_count={_fmt(median_summary_model_count)}, "
            f"median n_models={_fmt(median_n_models)}"
        )
        single_model_share = float(np.mean(group["summary_model_count"].astype(float).to_numpy() <= 1.0))
        if scenario in core_scenarios and single_model_share > 0.5:
            lines.append(
                f"- Warning: {scenario} uses a single summary model in most validation-matrix windows "
                f"({single_model_share:.0%})."
            )
    lines.append("")
    return lines


def _write_claim_audit(claim_audit: pd.DataFrame, output_path: Path) -> None:
    lines = ["# Claim Audit", ""]
    if claim_audit.empty:
        lines.append("- No claim-audit rows were produced.")
    else:
        for dimension in ["fidelity", "stability", "sensitivity", "specificity", "inference"]:
            subset = claim_audit[claim_audit["dimension"].astype(str) == dimension]
            if subset.empty:
                continue
            lines.append(f"## {dimension.title()}")
            for _, row in subset.iterrows():
                lines.append(
                    "- "
                    + f"{row['check_name']}: status={row['status']}, "
                    + f"priority={row.get('priority', 'core')}, "
                    + f"blocking={bool(row.get('blocking', True))}, "
                    + f"observed={_fmt(row['observed_value'])}, "
                    + f"threshold={_fmt(row['threshold'])}, "
                    + f"{row['details']}"
                )
            lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_validation_summary(
    result: ValidationResult,
    output_path: Path,
    suite: str,
    config: ProfilingConfig,
) -> None:
    claim_audit = result.claim_audit
    lines = ["# Validation Summary", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Suite: {suite}")
    lines.append("")
    lines.append("## Thresholds")
    lines.append(f"- Fidelity target/off-target ratio: {_fmt(config.validation.fidelity_target_ratio_min)}")
    lines.append(f"- Monotonicity Spearman minimum: {_fmt(config.validation.monotonicity_min_spearman)}")
    lines.append(f"- Specificity max effect ratio: {_fmt(config.validation.specificity_effect_ratio_max)}")
    lines.append(f"- Seed rank Spearman minimum: {_fmt(config.validation.seed_rank_min_spearman)}")
    lines.append(f"- Seed CV maximum: {_fmt(config.validation.seed_window_cv_max)}")
    lines.append(f"- Nuisance/structural drift max ratio: {_fmt(config.validation.nuisance_vs_structural_ratio_max)}")
    lines.append(f"- Seed/regime drift max ratio: {_fmt(config.validation.seed_vs_regime_drift_ratio_max)}")
    lines.append("")

    if claim_audit.empty:
        lines.append("## Status")
        lines.append("- No validation checks were produced.")
    else:
        blocking_subset = (
            claim_audit[claim_audit["blocking"].astype(bool)]
            if "blocking" in claim_audit.columns
            else claim_audit
        )
        if not blocking_subset.empty:
            blocking_passed = int((blocking_subset["status"].astype(str) == "pass").sum())
            lines.append("## Overall")
            lines.append(f"- Blocking checks passed: {blocking_passed}/{len(blocking_subset)}")
            lines.append("")
        for dimension in ["fidelity", "stability", "sensitivity", "specificity", "inference"]:
            subset = claim_audit[claim_audit["dimension"].astype(str) == dimension]
            if subset.empty:
                continue
            passed = int((subset["status"].astype(str) == "pass").sum())
            total = int(len(subset))
            lines.append(f"## {dimension.title()}")
            lines.append(f"- Passed: {passed}/{total}")
            for _, row in subset.iterrows():
                lines.append(
                    "- "
                    + f"{row['check_name']}: {row['status']} "
                    + f"[priority={row.get('priority', 'core')}, blocking={bool(row.get('blocking', True))}] "
                    + f"(observed={_fmt(row['observed_value'])}, threshold={_fmt(row['threshold'])})"
                )
            lines.append("")

    if not result.validation_matrix.empty:
        lines.append("## Validation Matrix")
        lines.append(f"- Rows: {len(result.validation_matrix)}")
        lines.append(f"- Columns: {len(_concept_features(config))}")
        lines.append("")

    if not result.synthetic_diagnostics.empty:
        lines.append("## Synthetic Diagnostics")
        lines.append(f"- Rows: {len(result.synthetic_diagnostics)}")
        if "diagnostic_source" in result.synthetic_diagnostics.columns:
            source_counts = result.synthetic_diagnostics["diagnostic_source"].astype(str).value_counts()
            for source, count in source_counts.items():
                lines.append(f"- {source}: {int(count)}")
        lines.append("")
        lines.extend(_synthetic_support_lines(result.synthetic_diagnostics))

    if not result.record_replication.empty:
        lines.append("## Record-Level Replication")
        status_counts = result.record_replication["status"].astype(str).value_counts()
        for status, count in status_counts.items():
            lines.append(f"- {status}: {int(count)}")
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_validation_report(
    result: ValidationResult,
    output_dir: str | Path,
    config: ProfilingConfig,
    suite: str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.validation_matrix.to_csv(out / "validation_matrix.csv", index=False)
    result.targeted_sensitivity.to_csv(out / "targeted_sensitivity.csv", index=False)
    result.specificity_checks.to_csv(out / "specificity_checks.csv", index=False)
    result.seed_stability.to_csv(out / "seed_stability.csv", index=False)
    result.nuisance_robustness.to_csv(out / "nuisance_robustness.csv", index=False)
    result.synthetic_diagnostics.to_csv(out / "synthetic_diagnostics.csv", index=False)
    result.block_sensitivity.to_csv(out / "block_sensitivity.csv", index=False)
    result.record_replication.to_csv(out / "record_replication.csv", index=False)
    _plot_validation_matrix(result.validation_matrix, config, out / "validation_matrix.png")
    _write_claim_audit(result.claim_audit, out / "claim_audit.md")
    _write_validation_summary(result, out / "validation_summary.md", suite=suite, config=config)
    return out


def _write_validation_status(
    output_dir: str | Path,
    *,
    suite: str,
    stage: str,
    status: str,
    started_at: datetime,
    result: ValidationResult,
    error: str | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "suite": str(suite),
        "stage": str(stage),
        "status": str(status),
        "started_at_utc": started_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": {
            "validation_matrix_rows": int(len(result.validation_matrix)),
            "targeted_sensitivity_rows": int(len(result.targeted_sensitivity)),
            "specificity_checks_rows": int(len(result.specificity_checks)),
            "seed_stability_rows": int(len(result.seed_stability)),
            "nuisance_robustness_rows": int(len(result.nuisance_robustness)),
            "synthetic_diagnostics_rows": int(len(result.synthetic_diagnostics)),
            "block_sensitivity_rows": int(len(result.block_sensitivity)),
            "record_replication_rows": int(len(result.record_replication)),
            "claim_audit_rows": int(len(result.claim_audit)),
        },
    }
    if error is not None:
        payload["error"] = str(error)

    status_path = out / "validation_status.json"
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status_path


def _checkpoint_validation_report(
    output_dir: str | Path,
    *,
    suite: str,
    config: ProfilingConfig,
    started_at: datetime,
    stage: str,
    status: str,
    validation_matrix: pd.DataFrame,
    targeted_sensitivity: pd.DataFrame,
    specificity_checks: pd.DataFrame,
    seed_stability: pd.DataFrame,
    nuisance_robustness: pd.DataFrame,
    synthetic_diagnostics: pd.DataFrame,
    block_sensitivity: pd.DataFrame,
    record_replication: pd.DataFrame,
    claim_audit: pd.DataFrame,
    error: str | None = None,
) -> ValidationResult:
    result = ValidationResult(
        validation_matrix=validation_matrix,
        targeted_sensitivity=targeted_sensitivity,
        specificity_checks=specificity_checks,
        seed_stability=seed_stability,
        nuisance_robustness=nuisance_robustness,
        synthetic_diagnostics=synthetic_diagnostics,
        block_sensitivity=block_sensitivity,
        record_replication=record_replication,
        claim_audit=claim_audit,
    )
    write_validation_report(result, output_dir=output_dir, config=config, suite=suite)
    _write_validation_status(
        output_dir,
        suite=suite,
        stage=stage,
        status=status,
        started_at=started_at,
        result=result,
        error=error,
    )
    return result


def _complete_if_requested_stop(
    *,
    stop_after_stage: str | None,
    current_stage: str,
    output_dir: str | Path,
    suite: str,
    config: ProfilingConfig,
    started_at: datetime,
    validation_matrix: pd.DataFrame,
    targeted_sensitivity: pd.DataFrame,
    specificity_checks: pd.DataFrame,
    seed_stability: pd.DataFrame,
    nuisance_robustness: pd.DataFrame,
    synthetic_diagnostics: pd.DataFrame,
    block_sensitivity: pd.DataFrame,
    record_replication: pd.DataFrame,
    claim_audit: pd.DataFrame,
) -> ValidationResult | None:
    if stop_after_stage != current_stage:
        return None
    return _checkpoint_validation_report(
        output_dir,
        suite=suite,
        config=config,
        started_at=started_at,
        stage=current_stage,
        status="completed",
        validation_matrix=validation_matrix,
        targeted_sensitivity=targeted_sensitivity,
        specificity_checks=specificity_checks,
        seed_stability=seed_stability,
        nuisance_robustness=nuisance_robustness,
        synthetic_diagnostics=synthetic_diagnostics,
        block_sensitivity=block_sensitivity,
        record_replication=record_replication,
        claim_audit=claim_audit,
    )


def run_validation(
    output_dir: str | Path,
    base_config: ProfilingConfig | None = None,
    suite: str = "all",
    dataset: LabeledTimeSeriesDataset | None = None,
    *,
    resume: bool = False,
    stop_after_stage: str | None = None,
) -> ValidationResult:
    config = copy.deepcopy(base_config or ProfilingConfig())
    if stop_after_stage is not None:
        if stop_after_stage not in VALIDATION_STOP_AFTER_STAGES:
            raise ValueError(f"Unsupported stop_after_stage: {stop_after_stage}")
        if not _stage_allowed_for_suite(stop_after_stage, suite):
            raise ValueError(f"stop_after_stage '{stop_after_stage}' is not reachable for suite '{suite}'")
    checkpoint = _load_validation_checkpoint(output_dir) if resume else ValidationResult(
        validation_matrix=pd.DataFrame(),
        targeted_sensitivity=pd.DataFrame(),
        specificity_checks=pd.DataFrame(),
        seed_stability=pd.DataFrame(),
        nuisance_robustness=pd.DataFrame(),
        synthetic_diagnostics=pd.DataFrame(),
        block_sensitivity=pd.DataFrame(),
        record_replication=pd.DataFrame(),
        claim_audit=pd.DataFrame(),
    )
    started_at = _load_existing_started_at(output_dir) or datetime.now(timezone.utc)
    current_stage = "resuming" if resume else "initializing"

    validation_matrix = checkpoint.validation_matrix.copy()
    targeted_sensitivity = checkpoint.targeted_sensitivity.copy()
    specificity_checks = checkpoint.specificity_checks.copy()
    seed_stability = checkpoint.seed_stability.copy()
    nuisance_robustness = checkpoint.nuisance_robustness.copy()
    synthetic_diagnostics = checkpoint.synthetic_diagnostics.copy()
    record_replication = checkpoint.record_replication.copy()
    claim_audit = pd.DataFrame()
    block_sensitivity = checkpoint.block_sensitivity.copy()

    _checkpoint_validation_report(
        output_dir,
        suite=suite,
        config=config,
        started_at=started_at,
        stage=current_stage,
        status="running",
        validation_matrix=validation_matrix,
        targeted_sensitivity=targeted_sensitivity,
        specificity_checks=specificity_checks,
        seed_stability=seed_stability,
        nuisance_robustness=nuisance_robustness,
        synthetic_diagnostics=synthetic_diagnostics,
        block_sensitivity=block_sensitivity,
        record_replication=record_replication,
        claim_audit=claim_audit,
    )

    try:
        if suite in {"synthetic", "all"}:
            current_stage = "synthetic.validation_matrix"
            if not _validation_matrix_complete(validation_matrix, config):
                def checkpoint_validation_matrix(partial_matrix: pd.DataFrame, partial_diagnostics: pd.DataFrame) -> None:
                    _checkpoint_validation_report(
                        output_dir,
                        suite=suite,
                        config=config,
                        started_at=started_at,
                        stage=current_stage,
                        status="running",
                        validation_matrix=partial_matrix,
                        targeted_sensitivity=targeted_sensitivity,
                        specificity_checks=specificity_checks,
                        seed_stability=seed_stability,
                        nuisance_robustness=nuisance_robustness,
                        synthetic_diagnostics=partial_diagnostics,
                        block_sensitivity=block_sensitivity,
                        record_replication=record_replication,
                        claim_audit=claim_audit,
                    )

                validation_matrix, matrix_diagnostics = _build_validation_matrix(
                    config,
                    existing=validation_matrix,
                    existing_diagnostics=synthetic_diagnostics[
                        synthetic_diagnostics["diagnostic_source"].astype(str) == "validation_matrix"
                    ].copy()
                    if not synthetic_diagnostics.empty and "diagnostic_source" in synthetic_diagnostics.columns
                    else pd.DataFrame(),
                    checkpoint=checkpoint_validation_matrix,
                )
                if not matrix_diagnostics.empty:
                    non_matrix = (
                        synthetic_diagnostics[synthetic_diagnostics["diagnostic_source"].astype(str) != "validation_matrix"].copy()
                        if not synthetic_diagnostics.empty and "diagnostic_source" in synthetic_diagnostics.columns
                        else pd.DataFrame()
                    )
                    synthetic_diagnostics = _merge_frames(non_matrix, matrix_diagnostics)
                _checkpoint_validation_report(
                    output_dir,
                    suite=suite,
                    config=config,
                    started_at=started_at,
                    stage=current_stage,
                    status="running",
                    validation_matrix=validation_matrix,
                    targeted_sensitivity=targeted_sensitivity,
                    specificity_checks=specificity_checks,
                    seed_stability=seed_stability,
                    nuisance_robustness=nuisance_robustness,
                    synthetic_diagnostics=synthetic_diagnostics,
                    block_sensitivity=block_sensitivity,
                    record_replication=record_replication,
                    claim_audit=claim_audit,
                )
            stopped = _complete_if_requested_stop(
                stop_after_stage=stop_after_stage,
                current_stage=current_stage,
                output_dir=output_dir,
                suite=suite,
                config=config,
                started_at=started_at,
                validation_matrix=validation_matrix,
                targeted_sensitivity=targeted_sensitivity,
                specificity_checks=specificity_checks,
                seed_stability=seed_stability,
                nuisance_robustness=nuisance_robustness,
                synthetic_diagnostics=synthetic_diagnostics,
                block_sensitivity=block_sensitivity,
                record_replication=record_replication,
                claim_audit=claim_audit,
            )
            if stopped is not None:
                return stopped

            current_stage = "synthetic.targeted_sensitivity"
            if not _targeted_sensitivity_complete(targeted_sensitivity, config):
                def checkpoint_targeted_sensitivity(partial_sensitivity: pd.DataFrame, partial_diagnostics: pd.DataFrame) -> None:
                    merged_diagnostics = _merge_frames(
                        synthetic_diagnostics[
                            synthetic_diagnostics["diagnostic_source"].astype(str) != "targeted_sensitivity"
                        ].copy()
                        if not synthetic_diagnostics.empty and "diagnostic_source" in synthetic_diagnostics.columns
                        else pd.DataFrame(),
                        partial_diagnostics,
                    )
                    _checkpoint_validation_report(
                        output_dir,
                        suite=suite,
                        config=config,
                        started_at=started_at,
                        stage=current_stage,
                        status="running",
                        validation_matrix=validation_matrix,
                        targeted_sensitivity=partial_sensitivity,
                        specificity_checks=specificity_checks,
                        seed_stability=seed_stability,
                        nuisance_robustness=nuisance_robustness,
                        synthetic_diagnostics=merged_diagnostics,
                        block_sensitivity=block_sensitivity,
                        record_replication=record_replication,
                        claim_audit=claim_audit,
                    )

                existing_targeted_diagnostics = (
                    synthetic_diagnostics[
                        synthetic_diagnostics["diagnostic_source"].astype(str) == "targeted_sensitivity"
                    ].copy()
                    if not synthetic_diagnostics.empty and "diagnostic_source" in synthetic_diagnostics.columns
                    else pd.DataFrame()
                )
                targeted_sensitivity, targeted_diagnostics = _build_targeted_sensitivity(
                    config,
                    existing=targeted_sensitivity,
                    existing_diagnostics=existing_targeted_diagnostics,
                    checkpoint=checkpoint_targeted_sensitivity,
                )
                if not targeted_diagnostics.empty:
                    non_targeted = (
                        synthetic_diagnostics[
                            synthetic_diagnostics["diagnostic_source"].astype(str) != "targeted_sensitivity"
                        ].copy()
                        if not synthetic_diagnostics.empty and "diagnostic_source" in synthetic_diagnostics.columns
                        else pd.DataFrame()
                    )
                    synthetic_diagnostics = _merge_frames(non_targeted, targeted_diagnostics)
                _checkpoint_validation_report(
                    output_dir,
                    suite=suite,
                    config=config,
                    started_at=started_at,
                    stage=current_stage,
                    status="running",
                    validation_matrix=validation_matrix,
                    targeted_sensitivity=targeted_sensitivity,
                    specificity_checks=specificity_checks,
                    seed_stability=seed_stability,
                    nuisance_robustness=nuisance_robustness,
                    synthetic_diagnostics=synthetic_diagnostics,
                    block_sensitivity=block_sensitivity,
                    record_replication=record_replication,
                    claim_audit=claim_audit,
                )
            stopped = _complete_if_requested_stop(
                stop_after_stage=stop_after_stage,
                current_stage=current_stage,
                output_dir=output_dir,
                suite=suite,
                config=config,
                started_at=started_at,
                validation_matrix=validation_matrix,
                targeted_sensitivity=targeted_sensitivity,
                specificity_checks=specificity_checks,
                seed_stability=seed_stability,
                nuisance_robustness=nuisance_robustness,
                synthetic_diagnostics=synthetic_diagnostics,
                block_sensitivity=block_sensitivity,
                record_replication=record_replication,
                claim_audit=claim_audit,
            )
            if stopped is not None:
                return stopped

            current_stage = "synthetic.specificity_checks"
            if specificity_checks.empty:
                specificity_checks = _build_specificity_checks(validation_matrix, config)
                _checkpoint_validation_report(
                    output_dir,
                    suite=suite,
                    config=config,
                    started_at=started_at,
                    stage=current_stage,
                    status="running",
                    validation_matrix=validation_matrix,
                    targeted_sensitivity=targeted_sensitivity,
                    specificity_checks=specificity_checks,
                    seed_stability=seed_stability,
                    nuisance_robustness=nuisance_robustness,
                    synthetic_diagnostics=synthetic_diagnostics,
                    block_sensitivity=block_sensitivity,
                    record_replication=record_replication,
                    claim_audit=claim_audit,
                )
            stopped = _complete_if_requested_stop(
                stop_after_stage=stop_after_stage,
                current_stage=current_stage,
                output_dir=output_dir,
                suite=suite,
                config=config,
                started_at=started_at,
                validation_matrix=validation_matrix,
                targeted_sensitivity=targeted_sensitivity,
                specificity_checks=specificity_checks,
                seed_stability=seed_stability,
                nuisance_robustness=nuisance_robustness,
                synthetic_diagnostics=synthetic_diagnostics,
                block_sensitivity=block_sensitivity,
                record_replication=record_replication,
                claim_audit=claim_audit,
            )
            if stopped is not None:
                return stopped

            current_stage = "synthetic.seed_stability"
            if seed_stability.empty:
                seed_stability = _build_seed_stability(config)
                _checkpoint_validation_report(
                    output_dir,
                    suite=suite,
                    config=config,
                    started_at=started_at,
                    stage=current_stage,
                    status="running",
                    validation_matrix=validation_matrix,
                    targeted_sensitivity=targeted_sensitivity,
                    specificity_checks=specificity_checks,
                    seed_stability=seed_stability,
                    nuisance_robustness=nuisance_robustness,
                    synthetic_diagnostics=synthetic_diagnostics,
                    block_sensitivity=block_sensitivity,
                    record_replication=record_replication,
                    claim_audit=claim_audit,
                )
            stopped = _complete_if_requested_stop(
                stop_after_stage=stop_after_stage,
                current_stage=current_stage,
                output_dir=output_dir,
                suite=suite,
                config=config,
                started_at=started_at,
                validation_matrix=validation_matrix,
                targeted_sensitivity=targeted_sensitivity,
                specificity_checks=specificity_checks,
                seed_stability=seed_stability,
                nuisance_robustness=nuisance_robustness,
                synthetic_diagnostics=synthetic_diagnostics,
                block_sensitivity=block_sensitivity,
                record_replication=record_replication,
                claim_audit=claim_audit,
            )
            if stopped is not None:
                return stopped

            current_stage = "synthetic.nuisance_robustness"
            if not _nuisance_robustness_complete(nuisance_robustness, config):
                def checkpoint_nuisance_robustness(partial_nuisance: pd.DataFrame) -> None:
                    _checkpoint_validation_report(
                        output_dir,
                        suite=suite,
                        config=config,
                        started_at=started_at,
                        stage=current_stage,
                        status="running",
                        validation_matrix=validation_matrix,
                        targeted_sensitivity=targeted_sensitivity,
                        specificity_checks=specificity_checks,
                        seed_stability=seed_stability,
                        nuisance_robustness=partial_nuisance,
                        synthetic_diagnostics=synthetic_diagnostics,
                        block_sensitivity=block_sensitivity,
                        record_replication=record_replication,
                        claim_audit=claim_audit,
                    )

                nuisance_robustness = _build_nuisance_robustness(
                    config,
                    existing=nuisance_robustness,
                    checkpoint=checkpoint_nuisance_robustness,
                )
                _checkpoint_validation_report(
                    output_dir,
                    suite=suite,
                    config=config,
                    started_at=started_at,
                    stage=current_stage,
                    status="running",
                    validation_matrix=validation_matrix,
                    targeted_sensitivity=targeted_sensitivity,
                    specificity_checks=specificity_checks,
                    seed_stability=seed_stability,
                    nuisance_robustness=nuisance_robustness,
                    synthetic_diagnostics=synthetic_diagnostics,
                    block_sensitivity=block_sensitivity,
                    record_replication=record_replication,
                    claim_audit=claim_audit,
                )
            stopped = _complete_if_requested_stop(
                stop_after_stage=stop_after_stage,
                current_stage=current_stage,
                output_dir=output_dir,
                suite=suite,
                config=config,
                started_at=started_at,
                validation_matrix=validation_matrix,
                targeted_sensitivity=targeted_sensitivity,
                specificity_checks=specificity_checks,
                seed_stability=seed_stability,
                nuisance_robustness=nuisance_robustness,
                synthetic_diagnostics=synthetic_diagnostics,
                block_sensitivity=block_sensitivity,
                record_replication=record_replication,
                claim_audit=claim_audit,
            )
            if stopped is not None:
                return stopped

            current_stage = "synthetic.block_sensitivity"
            if not _has_block_sensitivity_source(block_sensitivity, "synthetic_memory"):
                synthetic_reference = _synthetic_dataset(
                    config,
                    scenario="memory_only",
                    seed=41,
                    extra={"memory_lag_b": int(max(config.validation.memory_lag_levels))},
                )
                synthetic_result = _profile(
                    _clone_validation_config(config, synthetic_profile=True),
                    synthetic_reference.values,
                    synthetic_reference.labels,
                    record_ids=synthetic_reference.record_ids,
                )
                synthetic_block = _build_block_sensitivity(
                    synthetic_result.window_summaries,
                    config.stats,
                    config,
                    source="synthetic_memory",
                )
                if block_sensitivity.empty:
                    block_sensitivity = synthetic_block
                elif not synthetic_block.empty:
                    block_sensitivity = pd.concat([block_sensitivity, synthetic_block], ignore_index=True)
                _checkpoint_validation_report(
                    output_dir,
                    suite=suite,
                    config=config,
                    started_at=started_at,
                    stage=current_stage,
                    status="running",
                    validation_matrix=validation_matrix,
                    targeted_sensitivity=targeted_sensitivity,
                    specificity_checks=specificity_checks,
                    seed_stability=seed_stability,
                    nuisance_robustness=nuisance_robustness,
                    synthetic_diagnostics=synthetic_diagnostics,
                    block_sensitivity=block_sensitivity,
                    record_replication=record_replication,
                    claim_audit=claim_audit,
                )
            stopped = _complete_if_requested_stop(
                stop_after_stage=stop_after_stage,
                current_stage=current_stage,
                output_dir=output_dir,
                suite=suite,
                config=config,
                started_at=started_at,
                validation_matrix=validation_matrix,
                targeted_sensitivity=targeted_sensitivity,
                specificity_checks=specificity_checks,
                seed_stability=seed_stability,
                nuisance_robustness=nuisance_robustness,
                synthetic_diagnostics=synthetic_diagnostics,
                block_sensitivity=block_sensitivity,
                record_replication=record_replication,
                claim_audit=claim_audit,
            )
            if stopped is not None:
                return stopped

        if suite in {"real", "all"}:
            if dataset is None:
                raise ValueError("Real validation requires a loaded dataset")

            current_stage = "real.block_sensitivity"
            if not _has_block_sensitivity_source(block_sensitivity, "real_dataset"):
                real_result = _profile(
                    _clone_validation_config(config),
                    dataset.values,
                    dataset.labels,
                    record_ids=dataset.record_ids,
                )
                real_block = _build_block_sensitivity(
                    real_result.window_summaries,
                    config.stats,
                    config,
                    source="real_dataset",
                )
                if block_sensitivity.empty:
                    block_sensitivity = real_block
                elif not real_block.empty:
                    block_sensitivity = pd.concat([block_sensitivity, real_block], ignore_index=True)
                _checkpoint_validation_report(
                    output_dir,
                    suite=suite,
                    config=config,
                    started_at=started_at,
                    stage=current_stage,
                    status="running",
                    validation_matrix=validation_matrix,
                    targeted_sensitivity=targeted_sensitivity,
                    specificity_checks=specificity_checks,
                    seed_stability=seed_stability,
                    nuisance_robustness=nuisance_robustness,
                    synthetic_diagnostics=synthetic_diagnostics,
                    block_sensitivity=block_sensitivity,
                    record_replication=record_replication,
                    claim_audit=claim_audit,
                )
            stopped = _complete_if_requested_stop(
                stop_after_stage=stop_after_stage,
                current_stage=current_stage,
                output_dir=output_dir,
                suite=suite,
                config=config,
                started_at=started_at,
                validation_matrix=validation_matrix,
                targeted_sensitivity=targeted_sensitivity,
                specificity_checks=specificity_checks,
                seed_stability=seed_stability,
                nuisance_robustness=nuisance_robustness,
                synthetic_diagnostics=synthetic_diagnostics,
                block_sensitivity=block_sensitivity,
                record_replication=record_replication,
                claim_audit=claim_audit,
            )
            if stopped is not None:
                return stopped

            current_stage = "real.record_replication"
            if not _record_replication_complete(record_replication, dataset, config):
                def checkpoint_record_replication(partial: pd.DataFrame) -> None:
                    _checkpoint_validation_report(
                        output_dir,
                        suite=suite,
                        config=config,
                        started_at=started_at,
                        stage=current_stage,
                        status="running",
                        validation_matrix=validation_matrix,
                        targeted_sensitivity=targeted_sensitivity,
                        specificity_checks=specificity_checks,
                        seed_stability=seed_stability,
                        nuisance_robustness=nuisance_robustness,
                        synthetic_diagnostics=synthetic_diagnostics,
                        block_sensitivity=block_sensitivity,
                        record_replication=partial,
                        claim_audit=claim_audit,
                    )

                record_replication = _build_record_replication(
                    dataset,
                    config,
                    existing=record_replication,
                    checkpoint=checkpoint_record_replication,
                )
                checkpoint_record_replication(record_replication)
            stopped = _complete_if_requested_stop(
                stop_after_stage=stop_after_stage,
                current_stage=current_stage,
                output_dir=output_dir,
                suite=suite,
                config=config,
                started_at=started_at,
                validation_matrix=validation_matrix,
                targeted_sensitivity=targeted_sensitivity,
                specificity_checks=specificity_checks,
                seed_stability=seed_stability,
                nuisance_robustness=nuisance_robustness,
                synthetic_diagnostics=synthetic_diagnostics,
                block_sensitivity=block_sensitivity,
                record_replication=record_replication,
                claim_audit=claim_audit,
            )
            if stopped is not None:
                return stopped

        current_stage = "claim_audit"
        claim_audit = _build_claim_audit(
            validation_matrix=validation_matrix,
            targeted_sensitivity=targeted_sensitivity,
            specificity_checks=specificity_checks,
            seed_stability=seed_stability,
            nuisance_robustness=nuisance_robustness,
            block_sensitivity=block_sensitivity,
            record_replication=record_replication,
            config=config,
        )
        result = _checkpoint_validation_report(
            output_dir,
            suite=suite,
            config=config,
            started_at=started_at,
            stage=current_stage,
            status="running",
            validation_matrix=validation_matrix,
            targeted_sensitivity=targeted_sensitivity,
            specificity_checks=specificity_checks,
            seed_stability=seed_stability,
            nuisance_robustness=nuisance_robustness,
            synthetic_diagnostics=synthetic_diagnostics,
            block_sensitivity=block_sensitivity,
            record_replication=record_replication,
            claim_audit=claim_audit,
        )
        _write_validation_status(
            output_dir,
            suite=suite,
            stage="completed",
            status="completed",
            started_at=started_at,
            result=result,
        )
        return result
    except Exception as exc:
        partial_result = ValidationResult(
            validation_matrix=validation_matrix,
            targeted_sensitivity=targeted_sensitivity,
            specificity_checks=specificity_checks,
            seed_stability=seed_stability,
            nuisance_robustness=nuisance_robustness,
            synthetic_diagnostics=synthetic_diagnostics,
            block_sensitivity=block_sensitivity,
            record_replication=record_replication,
            claim_audit=claim_audit,
        )
        write_validation_report(partial_result, output_dir=output_dir, config=config, suite=suite)
        _write_validation_status(
            output_dir,
            suite=suite,
            stage=current_stage,
            status="failed",
            started_at=started_at,
            result=partial_result,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
