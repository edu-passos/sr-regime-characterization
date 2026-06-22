from argparse import Namespace

import pytest

from symbolic_structural_profiling.cli import _apply_config, _apply_preset, build_parser
from symbolic_structural_profiling.config import ProfilingConfig


def _modeling_args(**overrides) -> Namespace:
    args = {
        "preset": "standard",
        "output_dir": "outputs/test",
        "window_size": None,
        "step_size": None,
        "max_lag": None,
        "n_seeds": None,
        "n_candidates": None,
        "top_k": None,
        "pysr_niterations": None,
        "pysr_populations": None,
        "pysr_population_size": None,
        "pysr_ncycles_per_iteration": None,
        "instability_threshold": None,
        "n_permutations": None,
        "block_size": None,
        "effect_size_iqr_ratio": None,
        "balance_pairwise_labels": False,
        "balance_seed": None,
        "safe_operators": False,
        "skip_baselines": False,
        "symbolic_backend": None,
    }
    args.update(overrides)
    return Namespace(**args)


def test_apply_preset_quick_sets_small_budget():
    cfg = ProfilingConfig()
    _apply_preset(cfg, "quick")

    assert cfg.window.window_size == 120
    assert cfg.ensemble.n_seeds == 1
    assert cfg.ensemble.n_candidates_per_seed == 60
    assert cfg.ensemble.pysr_populations == 2
    assert cfg.ensemble.pysr_ncycles_per_iteration == 60
    assert cfg.ensemble.pysr_maxsize == 18
    assert cfg.stats.n_permutations == 120


def test_apply_preset_mvp_increases_window_density_and_relaxes_stability_filter():
    cfg = ProfilingConfig()
    _apply_preset(cfg, "mvp")

    assert cfg.window.window_size == 120
    assert cfg.window.step_size == 60
    assert cfg.aggregation.instability_threshold == 0.9
    assert cfg.stats.block_size == 3
    assert cfg.stats.n_permutations == 1200
    # PySR search hyperparameters are left at the library's v1.5.9 defaults
    # (no per-dataset tuning), pinned explicitly in the preset.
    assert cfg.ensemble.pysr_niterations == 100
    assert cfg.ensemble.pysr_populations == 31
    assert cfg.ensemble.pysr_population_size == 27
    assert cfg.ensemble.pysr_ncycles_per_iteration == 380
    assert cfg.ensemble.pysr_maxsize == 30
    assert cfg.ensemble.pysr_maxdepth is None


def test_apply_config_overrides_preset_values():
    cfg = ProfilingConfig()
    args = _modeling_args(
        preset="quick",
        n_seeds=3,
        n_candidates=99,
        top_k=2,
        pysr_niterations=7,
        n_permutations=345,
        effect_size_iqr_ratio=0.33,
        skip_baselines=True,
        symbolic_backend="random",
    )
    _apply_config(args, cfg)

    assert cfg.ensemble.n_seeds == 3
    assert cfg.ensemble.n_candidates_per_seed == 99
    assert cfg.ensemble.top_k_per_seed == 2
    assert cfg.ensemble.pysr_niterations == 7
    assert cfg.stats.n_permutations == 345
    assert cfg.stats.min_effect_iqr_ratio == 0.33
    assert cfg.baselines.run_baselines is False
    assert cfg.ensemble.symbolic_backend == "random"


def test_apply_config_safe_operators_and_balanced_pairwise_switches():
    cfg = ProfilingConfig()
    args = _modeling_args(
        preset="quick",
        safe_operators=True,
        balance_pairwise_labels=True,
        balance_seed=123,
    )
    _apply_config(args, cfg)

    assert cfg.stats.balance_labels_for_pairwise is True
    assert cfg.stats.balance_seed == 123
    assert cfg.ensemble.pysr_binary_operators == ("+", "-", "*")
    assert cfg.ensemble.pysr_unary_operators == ("sin", "cos", "tanh")


def test_parser_accepts_preset_for_each_command():
    parser = build_parser()
    run_syn = parser.parse_args(["run-synthetic", "--preset", "mvp"])
    run_csv = parser.parse_args(["run-csv", "--data-csv", "x.csv", "--label-col", "regime", "--preset", "quick"])
    run_validation = parser.parse_args(
        [
            "run-validation",
            "--preset",
            "mvp",
            "--suite",
            "synthetic",
            "--resume",
            "--stop-after-stage",
            "synthetic.validation_matrix",
        ]
    )

    assert run_syn.preset == "mvp"
    assert run_csv.preset == "quick"
    assert run_validation.preset == "mvp"
    assert run_validation.resume is True
    assert run_validation.stop_after_stage == "synthetic.validation_matrix"


def test_parser_rejects_removed_run_sweeps_command():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["run-sweeps", "--preset", "mvp"])


def test_parser_rejects_removed_validation_and_residual_flags():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["run-synthetic", "--validation-scheme", "tail"])

    with pytest.raises(SystemExit):
        parser.parse_args(["run-synthetic", "--fit-on-residuals"])
