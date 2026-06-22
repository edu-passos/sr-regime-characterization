from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(slots=True)
class WindowConfig:
    window_size: int = 160
    step_size: int = 160
    require_homogeneous_labels: bool = True


@dataclass(slots=True)
class EnsembleConfig:
    symbolic_backend: str = "pysr"
    fallback_backend: str = "random"
    max_lag: int = 6
    n_seeds: int = 4
    n_candidates_per_seed: int = 220
    top_k_per_seed: int = 3
    max_expression_depth: int = 4
    validation_split: float = 0.25
    validation_scheme: str = "tail"
    min_relative_gain_over_baseline: float = 0.03
    min_relative_gain_over_linear: float = 0.0
    gain_reference: str = "linear"
    min_models_per_window: int = 3
    seed_offset: int = 100
    pysr_niterations: int = 40
    pysr_populations: int = 10
    pysr_population_size: int = 64
    pysr_maxsize: int = 18
    pysr_maxdepth: int | None = 6
    pysr_parsimony: float = 0.005
    pysr_ncycles_per_iteration: int = 160
    pysr_tournament_selection_n: int = 12
    pysr_binary_operators: tuple[str, ...] = ("+", "-", "*", "/")
    pysr_unary_operators: tuple[str, ...] = ("sin", "cos", "log", "exp", "tanh")
    pysr_should_optimize_constants: bool = True
    pysr_complexity_of_constants: float = 1.0
    pysr_complexity_of_operators: Dict[str, float] | None = None
    pysr_nested_constraints: Dict[str, Dict[str, int]] | None = None
    pysr_weight_mutate_constant: float = 0.0346
    pysr_optimize_probability: float = 0.14
    min_prediction_std: float = 1e-6
    pysr_parallelism: str = "serial"
    pysr_procs: int = 0
    pysr_deterministic: bool = True
    pysr_progress: bool = False
    pysr_verbosity: int = 0
    max_evals_per_candidate: int = 1000
    normalize_window_for_symbolic: bool = True
    rebalance_retained_model_families: bool = False
    retained_simple_model_fit_ratio: float = 1.10
    window_summary_fit_band_ratio: float = 1.15
    window_summary_guard_complex_models: bool = False
    window_summary_prefer_simple_models: bool = False
    window_summary_simple_fit_ratio: float = 1.10
    window_summary_simple_fallback_ratio: float = 0.0
    window_summary_concept_pool_size: int = 3
    window_summary_concept_pool_fit_band_ratio: float = 1.10
    window_summary_nonlinear_pool_min_ratio: float = 0.05
    window_summary_nonlinear_pool_min_abs: float = 1e-4
    window_summary_interaction_pool_min_ratio: float = 0.05
    window_summary_interaction_pool_min_abs: float = 1e-4
    window_summary_complex_over_simple_min_ratio: float = 0.05
    window_summary_complex_over_simple_min_abs: float = 1e-4
    window_summary_linear_advantage_min_ratio: float = 0.0
    window_summary_linear_advantage_min_abs: float = 0.0
    sr_gain_threshold: float = 0.05


@dataclass(slots=True)
class FeatureConfig:
    active_lag_threshold: float = 0.2
    memory_long_lag_cutoff: int = 3
    finite_diff_epsilon: float = 1e-3
    counterfactual_structure_evidence: bool = True
    unary_structure_evidence_min_ratio: float = 0.05
    unary_structure_evidence_min_abs: float = 1e-4
    interaction_structure_evidence_min_ratio: float = 0.05
    interaction_structure_evidence_min_abs: float = 1e-4
    complexity_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "add": 1.0,
            "sub": 1.0,
            "mul": 1.4,
            "div": 2.2,
            "sin": 2.1,
            "cos": 2.1,
            "tanh": 2.4,
            "exp": 3.0,
            "log": 2.8,
            "neg": 0.7,
            "const": 0.4,
            "var": 0.5,
        }
    )


@dataclass(slots=True)
class AggregationConfig:
    instability_threshold: float = 0.55
    window_exclusion_if_uncertain: bool = True


@dataclass(slots=True)
class StatsConfig:
    block_size: int = 4
    n_permutations: int = 1500
    ci_bootstrap_repeats: int = 1000
    ci_alpha: float = 0.05
    ci_seed: int = 0
    alpha: float = 0.05
    min_effect_iqr_ratio: float = 0.2
    balance_labels_for_pairwise: bool = False
    balance_seed: int = 0


@dataclass(slots=True)
class BaselineConfig:
    run_baselines: bool = True
    lasso_alpha: float = 0.01
    lasso_max_iter: int = 10000
    embedding_n_components: int = 6
    embedding_cv_folds: int = 5


@dataclass(slots=True)
class ValidationConfig:
    concept_features: tuple[str, ...] = (
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
    )
    synthetic_n_steps: int = 2200
    synthetic_noise_std: float = 0.16
    seed_values: tuple[int, ...] = (0, 1, 2, 3)
    memory_lag_levels: tuple[int, ...] = (2, 3, 4, 5, 6, 8)
    nonlinearity_levels: tuple[float, ...] = (0.0, 0.15, 0.3, 0.45, 0.6)
    interaction_levels: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)
    amplitude_levels: tuple[float, ...] = (1.0, 1.4, 1.8)
    offset_levels: tuple[float, ...] = (0.0, 0.2, 0.4)
    noise_levels: tuple[float, ...] = (0.08, 0.12, 0.16, 0.24, 0.32)
    nuisance_scale_levels: tuple[float, ...] = (0.85, 1.15)
    nuisance_offset_levels: tuple[float, ...] = (-0.2, 0.2)
    nuisance_jitter_steps: int = 2
    nuisance_dropout_fraction: float = 0.08
    synthetic_binary_operators: tuple[str, ...] = ("+", "-", "*")
    synthetic_unary_operators: tuple[str, ...] = ("tanh",)
    synthetic_pysr_parsimony: float = 0.01
    synthetic_pysr_maxsize: int = 12
    synthetic_window_summary_fit_band_ratio: float = 1.05
    synthetic_linear_advantage_min_ratio: float = 0.05
    synthetic_linear_advantage_min_abs: float = 1e-4
    synthetic_top_k_per_seed: int = 4
    synthetic_simple_model_fit_ratio: float = 1.10
    synthetic_simple_summary_fit_ratio: float = 1.10
    synthetic_simple_summary_fallback_ratio: float = 1.20
    synthetic_complex_over_simple_min_ratio: float = 0.05
    synthetic_complex_over_simple_min_abs: float = 1e-4
    synthetic_unary_structure_evidence_min_ratio: float = 0.05
    synthetic_unary_structure_evidence_min_abs: float = 1e-4
    synthetic_interaction_structure_evidence_min_ratio: float = 0.05
    synthetic_interaction_structure_evidence_min_abs: float = 1e-4
    block_size_grid: tuple[int, ...] = (2, 3, 4, 5, 6)
    fidelity_target_ratio_min: float = 1.0
    monotonicity_min_spearman: float = 0.60
    specificity_effect_ratio_max: float = 0.35
    seed_rank_min_spearman: float = 0.60
    seed_window_cv_max: float = 0.60
    nuisance_vs_structural_ratio_max: float = 0.80
    seed_vs_regime_drift_ratio_max: float = 1.0
    record_sign_consistency_min: float = 0.60
    min_records_for_replication: int = 2


@dataclass(slots=True)
class ProfilingConfig:
    target_index: int = 0
    window: WindowConfig = field(default_factory=WindowConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
