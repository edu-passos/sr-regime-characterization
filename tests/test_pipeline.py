import numpy as np
import pytest

from symbolic_structural_profiling.config import ProfilingConfig
from symbolic_structural_profiling.models.ensemble import SymbolicModel, WindowEnsembleResult
from symbolic_structural_profiling.models.expressions import ExprNode
from symbolic_structural_profiling.pipeline import StructuralProfiler
from symbolic_structural_profiling.synthetic import generate_synthetic_regime_series


def _quick_config(symbolic_backend: str = "random", run_baselines: bool = False) -> ProfilingConfig:
    cfg = ProfilingConfig()
    cfg.window.window_size = 120
    cfg.window.step_size = 120

    cfg.ensemble.symbolic_backend = symbolic_backend
    cfg.ensemble.max_lag = 5
    cfg.ensemble.n_seeds = 2
    cfg.ensemble.n_candidates_per_seed = 60
    cfg.ensemble.top_k_per_seed = 2
    cfg.ensemble.max_expression_depth = 3

    cfg.ensemble.pysr_niterations = 2
    cfg.ensemble.pysr_populations = 2
    cfg.ensemble.pysr_population_size = 24
    cfg.ensemble.pysr_maxsize = 10
    cfg.ensemble.pysr_maxdepth = 4
    cfg.ensemble.pysr_tournament_selection_n = 8

    cfg.stats.n_permutations = 80

    cfg.baselines.run_baselines = run_baselines
    return cfg


def test_pipeline_runs_end_to_end_random_backend():
    data = generate_synthetic_regime_series(n_steps=1200, scenario="memory_only", seed=11)

    cfg = _quick_config(symbolic_backend="random", run_baselines=False)
    profiler = StructuralProfiler(cfg)
    result = profiler.fit(data.values, data.labels)

    assert not result.window_summaries.empty
    assert not result.window_evaluation_all.empty
    assert not result.regime_profiles.empty
    assert "medoid_equation" in result.medoid_equations.columns
    assert "label" in result.window_summaries.columns
    assert "included_in_window_summaries" in result.window_evaluation_all.columns
    assert "linear_symbolic_gap_median" in result.window_evaluation_all.columns
    assert "active_lag_count_median" in result.window_summaries.columns
    assert "complexity_cost_median" in result.window_summaries.columns
    assert "description_length_median" in result.window_summaries.columns


def test_pipeline_pysr_request_falls_back_when_unavailable(monkeypatch):
    from symbolic_structural_profiling.models import ensemble as ensemble_module

    def _raise_import_error():
        raise ImportError("pysr unavailable in this test")

    monkeypatch.setattr(ensemble_module, "_load_pysr_regressor", _raise_import_error)

    data = generate_synthetic_regime_series(n_steps=1000, scenario="nonlinearity_only", seed=7)

    cfg = _quick_config(symbolic_backend="pysr", run_baselines=False)
    cfg.ensemble.n_seeds = 1
    cfg.ensemble.top_k_per_seed = 1

    profiler = StructuralProfiler(cfg)
    result = profiler.fit(data.values, data.labels)

    assert not result.window_summaries.empty
    assert "symbolic_backend" in result.window_summaries.columns
    assert "random" in set(result.window_summaries["symbolic_backend"].astype(str))


def test_baseline_outputs_present():
    data = generate_synthetic_regime_series(n_steps=1200, scenario="interaction_only", seed=5)

    cfg = _quick_config(symbolic_backend="random", run_baselines=True)
    profiler = StructuralProfiler(cfg)
    result = profiler.fit(data.values, data.labels)

    assert "baseline_window_summaries" in result.baseline_outputs
    assert "baseline_regime_profiles" in result.baseline_outputs
    assert "embedding_baseline_metrics" in result.baseline_outputs

    assert not result.baseline_outputs["baseline_window_summaries"].empty
    assert not result.baseline_outputs["embedding_baseline_metrics"].empty


def test_amplitude_only_retains_windows():
    data = generate_synthetic_regime_series(n_steps=1200, scenario="amplitude_only", seed=5)

    cfg = _quick_config(symbolic_backend="random", run_baselines=False)
    profiler = StructuralProfiler(cfg)
    result = profiler.fit(data.values, data.labels)

    assert len(result.window_summaries) > 0


def test_pipeline_uses_fit_named_schema_without_validation_fields():
    data = generate_synthetic_regime_series(n_steps=1200, scenario="memory_only", seed=17)

    cfg = _quick_config(symbolic_backend="random", run_baselines=False)
    profiler = StructuralProfiler(cfg)
    result = profiler.fit(data.values, data.labels)

    assert not result.window_summaries.empty
    assert "baseline_fit_mse_median" in result.window_summaries.columns
    assert "linear_fit_mse_median" in result.window_summaries.columns
    assert "best_symbolic_fit_mse_median" in result.window_summaries.columns
    assert "linear_symbolic_fit_gap_median" in result.window_summaries.columns
    assert "validation_scheme" not in result.window_summaries.columns
    assert "symbolic_fit_mode" not in result.window_summaries.columns


def test_pipeline_standardizes_symbolic_inputs_and_restores_raw_prototype_scale(monkeypatch):
    import symbolic_structural_profiling.pipeline as pipeline_module

    captured: dict[str, np.ndarray | float] = {}

    def _fake_fit_symbolic_ensemble(X_fit, y_fit, feature_meta, target_index, config, feature_config=None):
        captured["x_mean"] = np.mean(X_fit, axis=0)
        captured["x_std"] = np.std(X_fit, axis=0)
        captured["y_mean"] = float(np.mean(y_fit))
        captured["y_std"] = float(np.std(y_fit))
        expr = ExprNode(op="var", feature_index=0, variable=0, lag=1)
        model = SymbolicModel(
            expression=expr,
            seed=0,
            fit_mse=0.01,
            complexity=1,
            equation=expr.to_string(),
        )
        return WindowEnsembleResult(
            retained_models=[model],
            candidate_models=[model],
            baseline_fit_mse=1.0,
            linear_fit_mse=0.02,
            best_symbolic_fit_mse=0.01,
            gate_symbolic_fit_mse=0.01,
            gate_symbolic_per_seed=(0.01,),
            linear_symbolic_fit_gap=0.01,
            seed_instability=0.0,
            backend="random",
        )

    monkeypatch.setattr(pipeline_module, "fit_symbolic_ensemble", _fake_fit_symbolic_ensemble)

    values = (5.0 + 3.0 * np.sin(np.linspace(0.0, 20.0, 360, dtype=float)))[:, None]
    labels = np.array(["A"] * 180 + ["B"] * 180, dtype=str)

    cfg = _quick_config(symbolic_backend="random", run_baselines=False)
    cfg.window.window_size = 120
    cfg.window.step_size = 120
    profiler = StructuralProfiler(cfg)
    result = profiler.fit(values, labels)

    assert np.allclose(captured["x_mean"], 0.0, atol=1e-6)
    assert np.allclose(captured["x_std"], 1.0, atol=1e-6)
    assert captured["y_mean"] == pytest.approx(0.0, abs=1e-6)
    assert captured["y_std"] == pytest.approx(1.0, abs=1e-6)
    assert result.prototype_fits
    assert abs(float(np.mean(result.prototype_fits[0].y_pred))) > 1.0


def test_pipeline_emits_structural_feature_columns_and_keeps_uncertain_windows(monkeypatch):
    import symbolic_structural_profiling.pipeline as pipeline_module

    def _fake_fit_symbolic_ensemble(X_fit, y_fit, feature_meta, target_index, config, feature_config=None):
        simple = ExprNode(op="var", feature_index=0, variable=0, lag=1)
        unary = ExprNode(
            op="tanh",
            children=(ExprNode(op="var", feature_index=2, variable=0, lag=3),),
        )
        simple_model = SymbolicModel(
            expression=simple,
            seed=0,
            fit_mse=0.05,
            complexity=1,
            equation=simple.to_string(),
        )
        unary_model = SymbolicModel(
            expression=unary,
            seed=1,
            fit_mse=0.06,
            complexity=3,
            equation=unary.to_string(),
        )
        return WindowEnsembleResult(
            retained_models=[simple_model, unary_model],
            candidate_models=[simple_model, unary_model],
            baseline_fit_mse=0.50,
            linear_fit_mse=0.20,
            best_symbolic_fit_mse=0.05,
            gate_symbolic_fit_mse=0.05,
            gate_symbolic_per_seed=(0.05,),
            linear_symbolic_fit_gap=0.15,
            seed_instability=0.80,
            backend="random",
        )

    monkeypatch.setattr(pipeline_module, "fit_symbolic_ensemble", _fake_fit_symbolic_ensemble)

    values = np.sin(np.linspace(0.0, 18.0, 360, dtype=float))[:, None]
    labels = np.array(["A"] * 180 + ["B"] * 180, dtype=str)

    cfg = _quick_config(symbolic_backend="random", run_baselines=False)
    cfg.window.window_size = 120
    cfg.window.step_size = 120

    result = StructuralProfiler(cfg).fit(values, labels)

    for column in (
        "lag_mass_median",
        "nonlinear_node_count_median",
        "interaction_term_count_median",
        "complexity_cost_median",
    ):
        assert column in result.raw_window_summaries.columns
        assert column in result.window_summaries.columns

    for column in (
        "lag_mass_p50",
        "nonlinear_node_count_p50",
        "interaction_term_count_p50",
        "complexity_cost_p50",
    ):
        assert column in result.regime_profiles.columns

    assert len(result.window_summaries) == len(result.raw_window_summaries)
    assert bool(result.window_summaries["is_uncertain"].all()) is True
    assert bool((result.window_summaries["uncertainty_filter_fallback_used"] == False).all()) is True
    assert int(result.window_evaluation_all["excluded_by_uncertainty_filter"].sum()) == 0
    assert bool(result.window_evaluation_all["included_in_window_summaries"].all()) is True
    assert "nonlinear_node_count_median" in result.window_summaries.columns
