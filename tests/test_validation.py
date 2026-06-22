from pathlib import Path
import json

import numpy as np
import pytest
import symbolic_structural_profiling.validation as validation_module

from symbolic_structural_profiling.config import ProfilingConfig
from symbolic_structural_profiling.dataio import LabeledTimeSeriesDataset
from symbolic_structural_profiling.validation import run_validation


def _validation_config() -> ProfilingConfig:
    cfg = ProfilingConfig()
    cfg.window.window_size = 60
    cfg.window.step_size = 60
    cfg.ensemble.symbolic_backend = "random"
    cfg.ensemble.max_lag = 4
    cfg.ensemble.n_seeds = 1
    cfg.ensemble.n_candidates_per_seed = 24
    cfg.ensemble.top_k_per_seed = 1
    cfg.ensemble.max_expression_depth = 3
    cfg.stats.n_permutations = 12
    cfg.stats.ci_bootstrap_repeats = 10
    cfg.baselines.run_baselines = False

    cfg.validation.synthetic_n_steps = 1000
    cfg.validation.seed_values = (0, 1)
    cfg.validation.memory_lag_levels = (3, 5)
    cfg.validation.nonlinearity_levels = (0.0, 0.4)
    cfg.validation.interaction_levels = (0.0, 0.3)
    cfg.validation.nuisance_scale_levels = (0.9,)
    cfg.validation.nuisance_offset_levels = (0.1,)
    cfg.validation.block_size_grid = (2, 3)
    return cfg


def test_run_validation_synthetic_writes_expected_artifacts(tmp_path: Path):
    cfg = _validation_config()

    result = run_validation(
        output_dir=tmp_path,
        base_config=cfg,
        suite="synthetic",
    )

    assert not result.validation_matrix.empty
    assert not result.targeted_sensitivity.empty
    assert not result.claim_audit.empty
    assert not result.synthetic_diagnostics.empty
    assert (tmp_path / "validation_matrix.csv").exists()
    assert (tmp_path / "validation_matrix.png").exists()
    assert (tmp_path / "validation_summary.md").exists()
    assert (tmp_path / "claim_audit.md").exists()
    assert (tmp_path / "synthetic_diagnostics.csv").exists()
    assert "summary_model_count_pre_guard" in result.synthetic_diagnostics.columns
    assert "summary_model_count_post_guard" in result.synthetic_diagnostics.columns
    assert "summary_selection_mode" in result.synthetic_diagnostics.columns
    assert "current_logic_decision_code" not in result.synthetic_diagnostics.columns
    assert "family_sep_unary_model_count" not in result.synthetic_diagnostics.columns
    assert "family_trace_post_guard_unary_count" not in result.synthetic_diagnostics.columns
    assert "adequate_simple_model_count" in result.synthetic_diagnostics.columns
    assert "best_simple_fit_ratio_to_best_selected" in result.synthetic_diagnostics.columns
    assert "best_simple_fit_mse" in result.synthetic_diagnostics.columns
    assert "best_complex_fit_mse" in result.synthetic_diagnostics.columns
    assert "complex_gain_over_simple" in result.synthetic_diagnostics.columns
    assert "complex_gain_over_simple_threshold" in result.synthetic_diagnostics.columns
    assert "nonlinear_candidate_count" in result.synthetic_diagnostics.columns
    assert "interaction_candidate_count" in result.synthetic_diagnostics.columns
    assert "nonlinear_pool_admitted" in result.synthetic_diagnostics.columns
    assert "interaction_pool_admitted" in result.synthetic_diagnostics.columns
    assert "best_selected_unary_curvature_gain" in result.synthetic_diagnostics.columns
    assert "best_selected_interaction_synergy_gain" in result.synthetic_diagnostics.columns
    assert "unary_curvature_threshold" in result.synthetic_diagnostics.columns
    assert "interaction_synergy_threshold" in result.synthetic_diagnostics.columns
    assert "nonlinear_admission_share" not in result.synthetic_diagnostics.columns
    assert "interaction_admission_share" not in result.synthetic_diagnostics.columns
    assert "nonlinear_admission_gain" in result.synthetic_diagnostics.columns
    assert "nonlinear_admission_threshold" in result.synthetic_diagnostics.columns
    assert "interaction_admission_gain" in result.synthetic_diagnostics.columns
    assert "interaction_admission_threshold" in result.synthetic_diagnostics.columns
    assert "nonlinear_pool_rejected_candidate_count" in result.synthetic_diagnostics.columns
    assert "interaction_pool_rejected_candidate_count" in result.synthetic_diagnostics.columns
    assert "fraction_models_simple" in result.synthetic_diagnostics.columns
    assert "fraction_models_raw_simple" in result.synthetic_diagnostics.columns
    assert "fraction_models_raw_unary_nonlinear" in result.synthetic_diagnostics.columns
    assert "fraction_models_raw_interaction" in result.synthetic_diagnostics.columns
    assert "fraction_models_coupled_interaction" in result.synthetic_diagnostics.columns
    assert "fraction_models_supported_unary" in result.synthetic_diagnostics.columns
    assert "fraction_models_supported_interaction" in result.synthetic_diagnostics.columns
    assert "lag_pool_model_count" in result.synthetic_diagnostics.columns
    assert "nonlinear_pool_model_count" in result.synthetic_diagnostics.columns
    assert "interaction_pool_model_count" in result.synthetic_diagnostics.columns
    assert "fraction_adequate_models_supported_unary" in result.synthetic_diagnostics.columns
    assert "fraction_adequate_models_supported_interaction" in result.synthetic_diagnostics.columns
    assert "fraction_nonlinear_pool_models_raw_unary" in result.synthetic_diagnostics.columns
    assert "fraction_interaction_pool_models_raw_interaction" in result.synthetic_diagnostics.columns
    assert "fraction_summary_models_simple" in result.synthetic_diagnostics.columns
    assert "fraction_summary_models_raw_simple" in result.synthetic_diagnostics.columns
    assert "fraction_summary_models_raw_unary_nonlinear" in result.synthetic_diagnostics.columns
    assert "fraction_summary_models_raw_interaction" in result.synthetic_diagnostics.columns
    assert "fraction_summary_models_coupled_interaction" in result.synthetic_diagnostics.columns
    assert "fraction_summary_models_supported_unary" in result.synthetic_diagnostics.columns
    assert "fraction_summary_models_supported_interaction" in result.synthetic_diagnostics.columns
    assert "retained_model_count_simple" in result.synthetic_diagnostics.columns
    assert "retained_model_count_unary" in result.synthetic_diagnostics.columns
    assert "retained_model_count_interaction" in result.synthetic_diagnostics.columns
    assert "retained_model_count_raw_simple" in result.synthetic_diagnostics.columns
    assert "retained_model_count_raw_unary" in result.synthetic_diagnostics.columns
    assert "retained_model_count_raw_interaction" in result.synthetic_diagnostics.columns
    assert "fraction_summary_models_unary_nonlinear" in result.synthetic_diagnostics.columns
    assert "fraction_summary_models_interaction" in result.synthetic_diagnostics.columns
    summary_text = (tmp_path / "validation_summary.md").read_text(encoding="utf-8")
    assert "## Synthetic Ensemble Support" in summary_text
    assert "median summary_model_count" in summary_text
    status_payload = json.loads((tmp_path / "validation_status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == "completed"
    assert status_payload["stage"] == "completed"
    assert status_payload["artifacts"]["validation_matrix_rows"] > 0
    assert status_payload["artifacts"]["synthetic_diagnostics_rows"] > 0


def test_run_validation_real_generates_record_replication(tmp_path: Path):
    cfg = _validation_config()
    record_a = np.sin(np.linspace(0.0, 10.0, 240, dtype=float))
    record_b = np.cos(np.linspace(0.0, 10.0, 240, dtype=float))
    values = np.concatenate([record_a, record_b])[:, None]
    labels = np.array((["A"] * 120 + ["B"] * 120) * 2, dtype=str)
    record_ids = np.array(["r1"] * 240 + ["r2"] * 240, dtype=str)
    dataset = LabeledTimeSeriesDataset(
        values=values,
        labels=labels,
        feature_names=["y"],
        record_ids=record_ids,
    )

    result = run_validation(
        output_dir=tmp_path,
        base_config=cfg,
        suite="real",
        dataset=dataset,
    )

    assert not result.block_sensitivity.empty
    assert not result.record_replication.empty
    assert "replication_mode" in result.record_replication.columns
    assert (tmp_path / "record_replication.csv").exists()
    status_payload = json.loads((tmp_path / "validation_status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == "completed"
    assert status_payload["artifacts"]["record_replication_rows"] > 0


def test_run_validation_resume_reuses_existing_synthetic_artifacts(tmp_path: Path, monkeypatch):
    cfg = _validation_config()

    initial = run_validation(
        output_dir=tmp_path,
        base_config=cfg,
        suite="synthetic",
    )
    assert not initial.validation_matrix.empty

    def _should_not_run(*args, **kwargs):
        raise AssertionError("resume should reuse existing validation_matrix artifact")

    monkeypatch.setattr(validation_module, "_build_validation_matrix", _should_not_run)

    resumed = run_validation(
        output_dir=tmp_path,
        base_config=cfg,
        suite="synthetic",
        resume=True,
    )

    assert not resumed.validation_matrix.empty
    status_payload = json.loads((tmp_path / "validation_status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == "completed"


def test_run_validation_can_stop_after_stage_and_resume(tmp_path: Path):
    cfg = _validation_config()

    partial = run_validation(
        output_dir=tmp_path,
        base_config=cfg,
        suite="synthetic",
        stop_after_stage="synthetic.validation_matrix",
    )

    assert not partial.validation_matrix.empty
    assert partial.targeted_sensitivity.empty
    status_payload = json.loads((tmp_path / "validation_status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == "completed"
    assert status_payload["stage"] == "synthetic.validation_matrix"

    resumed = run_validation(
        output_dir=tmp_path,
        base_config=cfg,
        suite="synthetic",
        resume=True,
    )

    assert not resumed.targeted_sensitivity.empty
    final_status = json.loads((tmp_path / "validation_status.json").read_text(encoding="utf-8"))
    assert final_status["status"] == "completed"
    assert final_status["stage"] == "completed"


def test_run_validation_resume_finishes_partial_validation_matrix(tmp_path: Path):
    cfg = _validation_config()
    validation_matrix, diagnostics = validation_module._build_validation_matrix(cfg)

    first_scenario = str(validation_matrix.iloc[0]["scenario"])
    validation_matrix.iloc[[0]].to_csv(tmp_path / "validation_matrix.csv", index=False)
    diagnostics[diagnostics["scenario"].astype(str) == first_scenario].to_csv(
        tmp_path / "synthetic_diagnostics.csv",
        index=False,
    )

    resumed = run_validation(
        output_dir=tmp_path,
        base_config=cfg,
        suite="synthetic",
        resume=True,
        stop_after_stage="synthetic.validation_matrix",
    )

    assert len(resumed.validation_matrix["scenario"].astype(str).unique()) == len(validation_module._scenario_specs(cfg))
    status_payload = json.loads((tmp_path / "validation_status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == "completed"
    assert status_payload["stage"] == "synthetic.validation_matrix"


def test_validation_routes_noise_only_to_nuisance_and_clamps_synthetic_operator_profile(tmp_path: Path):
    cfg = _validation_config()
    cfg.ensemble.pysr_populations = 2
    cfg.ensemble.pysr_population_size = 20
    cfg.ensemble.pysr_maxdepth = 4
    cfg.ensemble.pysr_maxsize = 20
    cfg.ensemble.pysr_parsimony = 0.001
    cfg.ensemble.pysr_ncycles_per_iteration = 40
    cfg.ensemble.pysr_tournament_selection_n = 6
    cfg.ensemble.window_summary_fit_band_ratio = 1.20

    result = run_validation(
        output_dir=tmp_path,
        base_config=cfg,
        suite="synthetic",
    )

    assert "noise_only" not in set(result.specificity_checks["scenario"].astype(str))
    nuisance_rows = result.nuisance_robustness[result.nuisance_robustness["perturbation"].astype(str) == "noise_only"]
    assert not nuisance_rows.empty
    assert set(nuisance_rows["kind"].astype(str)) == {"nuisance"}

    synthetic_cfg = validation_module._clone_validation_config(cfg, synthetic_profile=True)
    assert "lag_mass_median" in synthetic_cfg.validation.concept_features
    assert "nonlinear_node_count_median" in synthetic_cfg.validation.concept_features
    assert "curvature_response_evidence_median" not in synthetic_cfg.validation.concept_features
    assert synthetic_cfg.ensemble.max_lag == max(cfg.ensemble.max_lag, max(cfg.validation.memory_lag_levels))
    assert synthetic_cfg.ensemble.n_seeds == 4
    assert synthetic_cfg.ensemble.n_candidates_per_seed == 160
    assert synthetic_cfg.ensemble.top_k_per_seed == 4
    assert synthetic_cfg.ensemble.window_summary_concept_pool_size == 3
    assert synthetic_cfg.ensemble.window_summary_concept_pool_fit_band_ratio == pytest.approx(1.10)
    assert synthetic_cfg.ensemble.window_summary_nonlinear_pool_min_ratio == pytest.approx(0.05)
    assert synthetic_cfg.ensemble.window_summary_nonlinear_pool_min_abs == pytest.approx(1e-4)
    assert synthetic_cfg.ensemble.window_summary_interaction_pool_min_ratio == pytest.approx(0.05)
    assert synthetic_cfg.ensemble.window_summary_interaction_pool_min_abs == pytest.approx(1e-4)
    assert synthetic_cfg.ensemble.pysr_populations == 10
    assert synthetic_cfg.ensemble.pysr_population_size == 64
    assert synthetic_cfg.ensemble.pysr_maxdepth == 6
    assert synthetic_cfg.ensemble.pysr_maxsize == 12
    assert synthetic_cfg.ensemble.pysr_parsimony == 0.01
    assert synthetic_cfg.ensemble.pysr_ncycles_per_iteration == 160
    assert synthetic_cfg.ensemble.pysr_tournament_selection_n == 12
    assert synthetic_cfg.ensemble.rebalance_retained_model_families is True
    assert synthetic_cfg.ensemble.retained_simple_model_fit_ratio == pytest.approx(1.10)
    assert synthetic_cfg.ensemble.window_summary_fit_band_ratio == 1.05
    assert synthetic_cfg.ensemble.window_summary_guard_complex_models is True
    assert synthetic_cfg.ensemble.window_summary_prefer_simple_models is True
    assert synthetic_cfg.ensemble.window_summary_simple_fit_ratio == pytest.approx(1.10)
    assert synthetic_cfg.ensemble.window_summary_simple_fallback_ratio == pytest.approx(1.20)
    assert synthetic_cfg.ensemble.window_summary_complex_over_simple_min_ratio == pytest.approx(0.05)
    assert synthetic_cfg.ensemble.window_summary_complex_over_simple_min_abs == pytest.approx(1e-4)
    assert synthetic_cfg.ensemble.window_summary_linear_advantage_min_ratio == pytest.approx(0.05)
    assert synthetic_cfg.ensemble.window_summary_linear_advantage_min_abs == pytest.approx(1e-4)
    assert synthetic_cfg.features.counterfactual_structure_evidence is True
    assert synthetic_cfg.features.unary_structure_evidence_min_ratio == pytest.approx(0.05)
    assert synthetic_cfg.features.unary_structure_evidence_min_abs == pytest.approx(1e-4)
    assert synthetic_cfg.features.interaction_structure_evidence_min_ratio == pytest.approx(0.05)
    assert synthetic_cfg.features.interaction_structure_evidence_min_abs == pytest.approx(1e-4)
    assert synthetic_cfg.ensemble.pysr_binary_operators == cfg.validation.synthetic_binary_operators
    assert synthetic_cfg.ensemble.pysr_unary_operators == cfg.validation.synthetic_unary_operators
    assert cfg.ensemble.pysr_binary_operators != synthetic_cfg.ensemble.pysr_binary_operators
