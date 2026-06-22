from pathlib import Path

import pandas as pd

import numpy as np

from symbolic_structural_profiling.pipeline import ProfilingResult, PrototypeFit
from symbolic_structural_profiling.reporting import write_full_report
from symbolic_structural_profiling.summary import generate_run_summary


def test_generate_run_summary_handles_missing_inputs(tmp_path: Path):
    summary_path = generate_run_summary(tmp_path)
    text = summary_path.read_text(encoding="utf-8")

    assert summary_path.exists()
    assert "No window summaries found." in text
    assert "Pairwise stats not available." in text
    assert "Regime profile table not available." in text


def test_generate_run_summary_includes_key_sections(tmp_path: Path):
    pd.DataFrame(
        [
            {
                "window_index": 0,
                "label": "A",
                "symbolic_backend": "pysr",
                "record_id": "r1",
                "linear_symbolic_fit_gap_median": 0.08,
                "best_symbolic_fit_mse_median": 0.20,
                "linear_fit_mse_median": 0.30,
                "structural_instability": 0.1,
                "is_uncertain": False,
                "uncertainty_filter_fallback_used": False,
            },
            {
                "window_index": 1,
                "label": "B",
                "symbolic_backend": "pysr",
                "record_id": "r2",
                "linear_symbolic_fit_gap_median": -0.02,
                "best_symbolic_fit_mse_median": 0.40,
                "linear_fit_mse_median": 0.35,
                "structural_instability": 0.2,
                "is_uncertain": False,
                "uncertainty_filter_fallback_used": False,
            },
        ]
    ).to_csv(tmp_path / "window_summaries.csv", index=False)
    pd.DataFrame(
        [
            {
                "window_index": 0,
                "label": "A",
                "linear_symbolic_fit_gap_median": 0.08,
                "best_symbolic_fit_mse_median": 0.20,
                "linear_fit_mse_median": 0.30,
                "has_structural_summary": True,
                "included_in_window_summaries": True,
            },
            {
                "window_index": 1,
                "label": "B",
                "linear_symbolic_fit_gap_median": -0.02,
                "best_symbolic_fit_mse_median": 0.40,
                "linear_fit_mse_median": 0.35,
                "has_structural_summary": True,
                "included_in_window_summaries": True,
            },
            {
                "window_index": 2,
                "label": "B",
                "linear_symbolic_fit_gap_median": 0.01,
                "best_symbolic_fit_mse_median": 0.33,
                "linear_fit_mse_median": 0.34,
                "has_structural_summary": False,
                "included_in_window_summaries": False,
            },
        ]
    ).to_csv(tmp_path / "window_evaluation_all.csv", index=False)

    pd.DataFrame(
        [
            {
                "label": "A",
                "n_windows": 1,
                "lag_mass_p50": 1.4,
                "nonlinear_node_count_p50": 2.0,
                "interaction_term_count_p50": 0.5,
                "complexity_cost_p50": 5.0,
                "linear_symbolic_fit_gap_p50": 0.1,
            }
        ]
    ).to_csv(tmp_path / "regime_profiles.csv", index=False)

    pd.DataFrame(
        [
            {
                "feature": "lag_mass_median",
                "median_diff": 0.8,
                "cliffs_delta": 0.7,
                "p_value": 0.002,
                "q_value": 0.01,
                "reject": True,
            }
        ]
    ).to_csv(tmp_path / "pairwise_stats.csv", index=False)

    pd.DataFrame([{"label": "A", "medoid_equation": "x0 + x1"}]).to_csv(
        tmp_path / "medoid_equations.csv", index=False
    )

    pd.DataFrame(
        [
            {"baseline_name": "lasso_ar", "window_index": 0, "best_symbolic_fit_mse_median": 0.25},
            {"baseline_name": "lasso_ar", "window_index": 1, "best_symbolic_fit_mse_median": 0.35},
        ]
    ).to_csv(tmp_path / "baseline_window_summaries.csv", index=False)

    pd.DataFrame([{"cv_accuracy": 0.9, "cv_balanced_accuracy": 0.88, "silhouette_score": 0.22}]).to_csv(
        tmp_path / "embedding_baseline_metrics.csv", index=False
    )

    summary_path = generate_run_summary(tmp_path)
    text = summary_path.read_text(encoding="utf-8")

    assert "Windows retained: 2" in text
    assert "Windows evaluated (all): 3" in text
    assert "Retained coverage:" in text
    assert "All windows (objective evaluation)" in text
    assert "Retained windows (structural analysis)" in text
    assert "Labels: A, B" in text
    assert "Records represented: 2" in text
    assert "Canonical fit: full-window direct symbolic fitting" in text
    assert "## Fidelity" in text
    assert "## Stability" in text
    assert "## Sensitivity" in text
    assert "## Specificity" in text
    assert "FDR-significant tests: 1" in text
    assert "lag_mass_median" in text
    assert "Label A: `x0 + x1`" in text
    assert "lasso_ar: windows=2" in text
    assert "embedding_separation:" in text


def test_write_full_report_generates_summary(tmp_path: Path):
    result = ProfilingResult(
        raw_window_summaries=pd.DataFrame([{"window_index": 0, "label": "A"}]),
        window_summaries=pd.DataFrame([{"window_index": 0, "label": "A"}]),
        window_evaluation_all=pd.DataFrame(
            [
                {
                    "window_index": 0,
                    "label": "A",
                    "linear_symbolic_fit_gap_median": 0.02,
                    "best_symbolic_fit_mse_median": 0.15,
                    "linear_fit_mse_median": 0.17,
                    "has_structural_summary": True,
                    "included_in_window_summaries": True,
                }
            ]
        ),
        regime_profiles=pd.DataFrame([{"label": "A", "n_windows": 1}]),
        pairwise_stats=pd.DataFrame(),
        medoid_equations=pd.DataFrame([{"label": "A", "medoid_equation": "x0"}]),
        baseline_outputs={},
    )

    output_dir = write_full_report(result, tmp_path)
    summary_path = output_dir / "summary.md"

    assert summary_path.exists()


def test_write_full_report_generates_visual_pack_when_inputs_available(tmp_path: Path):
    window_df = pd.DataFrame(
        [
            {
                "window_index": 0,
                "label": "A",
                "lag_mass_median": 1.2,
                "best_symbolic_fit_mse_median": 0.12,
                "structural_instability": 0.15,
            },
            {
                "window_index": 1,
                "label": "B",
                "lag_mass_median": 1.8,
                "best_symbolic_fit_mse_median": 0.22,
                "structural_instability": 0.25,
            },
        ]
    )

    result = ProfilingResult(
        raw_window_summaries=window_df.copy(),
        window_summaries=window_df,
        window_evaluation_all=pd.DataFrame(
            [
                {
                    "window_index": 0,
                    "label": "A",
                    "linear_symbolic_fit_gap_median": 0.03,
                    "best_symbolic_fit_mse_median": 0.12,
                    "linear_fit_mse_median": 0.14,
                    "has_structural_summary": True,
                    "included_in_window_summaries": True,
                },
                {
                    "window_index": 1,
                    "label": "B",
                    "linear_symbolic_fit_gap_median": 0.05,
                    "best_symbolic_fit_mse_median": 0.22,
                    "linear_fit_mse_median": 0.27,
                    "has_structural_summary": True,
                    "included_in_window_summaries": True,
                },
            ]
        ),
        regime_profiles=pd.DataFrame(
            [
                {
                    "label": "A",
                    "n_windows": 1,
                    "lag_mass_p50": 1.2,
                    "lag_entropy_p50": 0.8,
                    "nonlinear_node_count_p50": 2.0,
                    "interaction_term_count_p50": 0.2,
                    "complexity_cost_p50": 5.0,
                    "linear_symbolic_fit_gap_p50": 0.03,
                },
                {
                    "label": "B",
                    "n_windows": 1,
                    "lag_mass_p50": 1.8,
                    "lag_entropy_p50": 1.1,
                    "nonlinear_node_count_p50": 3.0,
                    "interaction_term_count_p50": 0.5,
                    "complexity_cost_p50": 6.0,
                    "linear_symbolic_fit_gap_p50": 0.05,
                },
            ]
        ),
        pairwise_stats=pd.DataFrame(
            [
                {
                    "feature": "lag_mass_median",
                    "group_a": "A",
                    "group_b": "B",
                    "median_diff": -0.6,
                    "p_value": 0.01,
                    "q_value": 0.04,
                }
            ]
        ),
        medoid_equations=pd.DataFrame(
            [
                {
                    "label": "A",
                    "window_index": 0,
                    "medoid_equation": "x0_lag1 + 0.1 * x0_lag2",
                    "best_symbolic_fit_mse": 0.12,
                },
                {
                    "label": "B",
                    "window_index": 1,
                    "medoid_equation": "sin(x0_lag1) + x0_lag3",
                    "best_symbolic_fit_mse": 0.22,
                },
            ]
        ),
        baseline_outputs={},
        prototype_fits=[
            PrototypeFit(
                label="A",
                window_index=0,
                y_true=np.array([0.0, 0.3, -0.1, 0.2], dtype=float),
                y_pred=np.array([0.02, 0.28, -0.06, 0.24], dtype=float),
                fit_mse=0.01,
            ),
            PrototypeFit(
                label="B",
                window_index=1,
                y_true=np.array([0.1, -0.2, 0.4, 0.0], dtype=float),
                y_pred=np.array([0.12, -0.18, 0.35, -0.05], dtype=float),
                fit_mse=0.02,
            ),
        ],
        window_fits=[
            PrototypeFit(
                label="A",
                window_index=0,
                y_true=np.array([0.0, 0.3, -0.1, 0.2], dtype=float),
                y_pred=np.array([0.02, 0.28, -0.06, 0.24], dtype=float),
                fit_mse=0.01,
            ),
            PrototypeFit(
                label="B",
                window_index=1,
                y_true=np.array([0.1, -0.2, 0.4, 0.0], dtype=float),
                y_pred=np.array([0.12, -0.18, 0.35, -0.05], dtype=float),
                fit_mse=0.02,
            ),
        ],
    )

    output_dir = write_full_report(result, tmp_path)
    assert (output_dir / "medoid_equations.png").exists()
    assert (output_dir / "top_pairwise_effects.png").exists()
    assert (output_dir / "window_quality.png").exists()
    assert (output_dir / "prototype_fit.png").exists()
    assert (output_dir / "window_fit_examples.png").exists()
