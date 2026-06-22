from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import ProfilingConfig
from .dataio import load_labeled_timeseries_csv, parse_feature_columns
from .pipeline import ProfilingResult, StructuralProfiler
from .reporting import write_full_report
from .sr_config_study import (
    run_sr_grid_study, grid_study_summary, make_run_card, grid_study_plots,
    make_nested_constraints, make_complexity_of_operators,
)
from .synthetic import generate_synthetic_regime_series
from .validation import run_validation, validation_stage_choices


def _apply_preset(config: ProfilingConfig, preset: str) -> None:
    if preset == "quick":
        config.window.window_size = 120
        config.window.step_size = 120

        config.ensemble.max_lag = 5
        config.ensemble.n_seeds = 1
        config.ensemble.n_candidates_per_seed = 60
        config.ensemble.top_k_per_seed = 1

        config.ensemble.pysr_populations = 2
        config.ensemble.pysr_population_size = 24
        config.ensemble.pysr_maxdepth = 4
        config.ensemble.pysr_ncycles_per_iteration = 60
        config.ensemble.pysr_tournament_selection_n = 8

        config.stats.block_size = 4
        config.stats.n_permutations = 120
        return

    if preset == "standard":
        config.window.window_size = 160
        config.window.step_size = 160

        config.ensemble.max_lag = 6
        config.ensemble.n_seeds = 4
        config.ensemble.n_candidates_per_seed = 220
        config.ensemble.top_k_per_seed = 3

        config.ensemble.pysr_populations = 10
        config.ensemble.pysr_population_size = 64
        config.ensemble.pysr_maxdepth = 6
        config.ensemble.pysr_ncycles_per_iteration = 160
        config.ensemble.pysr_tournament_selection_n = 12

        config.stats.block_size = 4
        config.stats.n_permutations = 1500
        return

    if preset == "mvp":
        # Window / step are tied to the synthetic regime-block length (200
        # samples): 120/60 yields ~2 homogeneous windows per block, where 160
        # would yield ~1 and waste half the data. These stay dataset-specific.
        config.window.window_size = 120
        config.window.step_size = 60

        # PySR search hyperparameters are left at the library's own defaults
        # (v1.5.9), pinned explicitly here for reproducibility -- no per-dataset
        # tuning of the symbolic search. Only the ensemble-level knobs (max_lag,
        # S = n_seeds, K = top_k) are configured by this work.
        config.ensemble.max_lag = 6
        config.ensemble.n_seeds = 4
        config.ensemble.n_candidates_per_seed = 220
        config.ensemble.top_k_per_seed = 3

        config.ensemble.pysr_niterations = 100
        config.ensemble.pysr_populations = 31
        config.ensemble.pysr_population_size = 27
        config.ensemble.pysr_maxsize = 30
        config.ensemble.pysr_maxdepth = None
        config.ensemble.pysr_ncycles_per_iteration = 380
        config.ensemble.pysr_tournament_selection_n = 15

        config.aggregation.instability_threshold = 0.9
        config.stats.block_size = 3
        config.stats.n_permutations = 1200
        return

    if preset == "research":
        config.window.window_size = 180
        config.window.step_size = 180

        config.ensemble.max_lag = 8
        config.ensemble.n_seeds = 8
        config.ensemble.n_candidates_per_seed = 320
        config.ensemble.top_k_per_seed = 4

        config.ensemble.pysr_populations = 14
        config.ensemble.pysr_population_size = 96
        config.ensemble.pysr_maxdepth = 7
        config.ensemble.pysr_ncycles_per_iteration = 240
        config.ensemble.pysr_tournament_selection_n = 14

        config.stats.block_size = 5
        config.stats.n_permutations = 4000
        return

    raise ValueError(f"Unsupported preset: {preset}")


def _add_modeling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=["quick", "standard", "mvp", "research"], default="standard")
    parser.add_argument("--output-dir", default="outputs/run")

    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--max-lag", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-candidates", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)

    parser.add_argument("--pysr-niterations", type=int, default=None)
    parser.add_argument("--pysr-populations", type=int, default=None)
    parser.add_argument("--pysr-population-size", type=int, default=None)
    parser.add_argument("--pysr-ncycles-per-iteration", type=int, default=None)
    parser.add_argument(
        "--pysr-parallelism",
        choices=["serial", "multithreading", "multiprocessing"],
        default=None,
        help="PySR Julia parallelism mode (default: serial).",
    )
    parser.add_argument(
        "--pysr-procs",
        type=int,
        default=None,
        help="Number of Julia worker procs/threads when --pysr-parallelism is not serial.",
    )
    parser.add_argument(
        "--pysr-deterministic",
        dest="pysr_deterministic",
        action="store_true",
        default=None,
        help="Force PySR deterministic mode (incompatible with non-serial parallelism).",
    )
    parser.add_argument(
        "--no-pysr-deterministic",
        dest="pysr_deterministic",
        action="store_false",
        help="Disable PySR deterministic mode (required when using --pysr-parallelism != serial).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Per-window checkpoint directory. Defaults to <output-dir>/checkpoints. "
             "Resumed runs reuse cached windows when manifest matches.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable per-window checkpointing entirely.",
    )
    parser.add_argument(
        "--nesting-level", type=int, default=None,
        help="SR nesting constraint level: 0=none, 1=no_unary_in_unary, 2=strict",
    )
    parser.add_argument(
        "--cost-level", type=int, default=None,
        help="SR operator cost level: 0=default, 1=cost×3, 2=cost×5",
    )
    parser.add_argument(
        "--parsimony", type=float, default=None,
        help="PySR parsimony penalty for expression complexity",
    )

    parser.add_argument("--instability-threshold", type=float, default=None)
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--effect-size-iqr-ratio", type=float, default=None)
    parser.add_argument(
        "--balance-pairwise-labels",
        action="store_true",
        help="Downsample larger label groups before pairwise tests to enforce balanced group sizes.",
    )
    parser.add_argument("--balance-seed", type=int, default=None)
    parser.add_argument(
        "--safe-operators",
        action="store_true",
        help="Use a safer PySR operator set without division, log, or exp.",
    )

    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument(
        "--symbolic-backend",
        choices=["pysr", "random"],
        default=None,
        help="Primary symbolic regression backend",
    )


def _apply_config(args: argparse.Namespace, config: ProfilingConfig) -> None:
    _apply_preset(config, args.preset)

    if args.window_size is not None:
        config.window.window_size = args.window_size
    if args.step_size is not None:
        config.window.step_size = args.step_size

    if args.max_lag is not None:
        config.ensemble.max_lag = args.max_lag
    if args.n_seeds is not None:
        config.ensemble.n_seeds = args.n_seeds
    if args.n_candidates is not None:
        config.ensemble.n_candidates_per_seed = args.n_candidates
    if args.top_k is not None:
        config.ensemble.top_k_per_seed = args.top_k

    if args.pysr_niterations is not None:
        config.ensemble.pysr_niterations = args.pysr_niterations
    if args.pysr_populations is not None:
        config.ensemble.pysr_populations = args.pysr_populations
    if args.pysr_population_size is not None:
        config.ensemble.pysr_population_size = args.pysr_population_size
    if args.pysr_ncycles_per_iteration is not None:
        config.ensemble.pysr_ncycles_per_iteration = args.pysr_ncycles_per_iteration
    if getattr(args, "pysr_parallelism", None) is not None:
        config.ensemble.pysr_parallelism = args.pysr_parallelism
    if getattr(args, "pysr_procs", None) is not None:
        config.ensemble.pysr_procs = int(args.pysr_procs)
    if getattr(args, "pysr_deterministic", None) is not None:
        config.ensemble.pysr_deterministic = bool(args.pysr_deterministic)

    # Auto-relax determinism when caller picked non-serial parallelism without
    # explicitly toggling the determinism flag, since PySR errors otherwise.
    if (
        config.ensemble.pysr_parallelism != "serial"
        and config.ensemble.pysr_deterministic
        and getattr(args, "pysr_deterministic", None) is None
    ):
        config.ensemble.pysr_deterministic = False

    if args.symbolic_backend is not None:
        config.ensemble.symbolic_backend = args.symbolic_backend

    if args.instability_threshold is not None:
        config.aggregation.instability_threshold = args.instability_threshold

    if args.block_size is not None:
        config.stats.block_size = args.block_size
    if args.n_permutations is not None:
        config.stats.n_permutations = args.n_permutations
    if args.effect_size_iqr_ratio is not None:
        config.stats.min_effect_iqr_ratio = args.effect_size_iqr_ratio
    if args.balance_pairwise_labels:
        config.stats.balance_labels_for_pairwise = True
    if args.balance_seed is not None:
        config.stats.balance_seed = args.balance_seed

    if args.safe_operators:
        config.ensemble.pysr_binary_operators = ("+", "-", "*")
        config.ensemble.pysr_unary_operators = ("sin", "cos", "tanh")

    # Apply nesting / cost / parsimony after safe_operators so unary_ops is final
    if getattr(args, "nesting_level", None) is not None:
        unary_ops = tuple(config.ensemble.pysr_unary_operators)
        nc = make_nested_constraints(int(args.nesting_level), unary_ops)
        if nc is not None:
            config.ensemble.pysr_nested_constraints = nc
    if getattr(args, "cost_level", None) is not None:
        cc = make_complexity_of_operators(int(args.cost_level))
        if cc is not None:
            config.ensemble.pysr_complexity_of_operators = cc
    if getattr(args, "parsimony", None) is not None:
        config.ensemble.pysr_parsimony = float(args.parsimony)

    config.baselines.run_baselines = not args.skip_baselines


def _resolve_checkpoint_dir(args: argparse.Namespace) -> Optional[Path]:
    if getattr(args, "no_checkpoint", False):
        return None
    explicit = getattr(args, "checkpoint_dir", None)
    if explicit:
        return Path(explicit)
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        return Path(output_dir) / "checkpoints"
    return None


def _print_result_summary(result: ProfilingResult, output_dir: Path) -> None:
    print(f"Windows retained: {len(result.window_summaries)}")
    print(f"Regimes: {', '.join(result.regime_profiles['label'].astype(str).tolist())}")
    print(f"Output directory: {output_dir}")

    if "symbolic_backend" in result.window_summaries.columns:
        backends = sorted(result.window_summaries["symbolic_backend"].astype(str).unique().tolist())
        if backends:
            print(f"Symbolic backend(s): {', '.join(backends)}")

    if not result.pairwise_stats.empty:
        top = result.pairwise_stats.head(5)
        print("Top pairwise effects:")
        for _, row in top.iterrows():
            print(
                f"  {row['feature']}: {row['group_a']} vs {row['group_b']} "
                f"median_diff={row['median_diff']:.4f}, q={row['q_value']:.4g}"
            )


def _print_validation_summary(result, output_dir: Path) -> None:
    print(f"Validation output directory: {output_dir}")
    if not result.claim_audit.empty:
        audit = result.claim_audit
        blocking = audit[audit["blocking"].astype(bool)] if "blocking" in audit.columns else audit
        passed = int((blocking["status"].astype(str) == "pass").sum())
        total = int(len(blocking))
        print(f"Blocking validation checks passed: {passed}/{total}")
    if not result.validation_matrix.empty:
        print(f"Validation matrix rows: {len(result.validation_matrix)}")
    if hasattr(result, "synthetic_diagnostics") and not result.synthetic_diagnostics.empty:
        print(f"Synthetic diagnostics rows: {len(result.synthetic_diagnostics)}")
    if not result.record_replication.empty and "status" in result.record_replication.columns:
        statuses = result.record_replication["status"].astype(str).value_counts().to_dict()
        if statuses:
            print("Record replication status counts:")
            for key, value in sorted(statuses.items()):
                print(f"  {key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssp",
        description="Symbolic Structural Profiling CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-synthetic", help="Run a synthetic structural profiling experiment")
    run.add_argument(
        "--scenario",
        choices=[
            "memory_only",
            "nonlinearity_only",
            "interaction_only",
            "parameter_only",
            "amplitude_only",
            "offset_only",
            "noise_only",
            "iid_noise",
        ],
        default="memory_only",
    )
    run.add_argument("--n-steps", type=int, default=3200)
    run.add_argument("--noise-std", type=float, default=0.16)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--memory-lag-b", type=int, default=5)
    run.add_argument("--nonlinearity-strength-b", type=float, default=0.42)
    run.add_argument("--interaction-strength-b", type=float, default=0.33)
    run.add_argument("--amplitude-scale-b", type=float, default=1.9)
    run.add_argument("--offset-b", type=float, default=0.35)
    run.add_argument("--noise-scale-b", type=float, default=1.75)
    _add_modeling_args(run)

    run_csv = sub.add_parser("run-csv", help="Run structural profiling on a labeled CSV dataset")
    run_csv.add_argument("--data-csv", required=True)
    run_csv.add_argument("--label-col", required=True)
    run_csv.add_argument("--feature-cols", default=None, help="Comma-separated feature columns")
    run_csv.add_argument("--time-col", default=None)
    run_csv.add_argument("--record-col", default=None)
    run_csv.add_argument("--target-col", default=None)
    run_csv.add_argument("--target-index", type=int, default=0)
    run_csv.add_argument("--no-normalize", action="store_true")
    run_csv.add_argument("--keep-na", action="store_true")
    _add_modeling_args(run_csv)

    validation = sub.add_parser("run-validation", help="Run the three-level validation harness")
    validation.add_argument("--suite", choices=["synthetic", "real", "all"], default="all")
    validation.add_argument("--data-csv", default=None)
    validation.add_argument("--label-col", default=None)
    validation.add_argument("--feature-cols", default=None, help="Comma-separated feature columns")
    validation.add_argument("--time-col", default=None)
    validation.add_argument("--record-col", default=None)
    validation.add_argument("--target-col", default=None)
    validation.add_argument("--target-index", type=int, default=0)
    validation.add_argument("--no-normalize", action="store_true")
    validation.add_argument("--keep-na", action="store_true")
    validation.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous validation run from artifacts already present in --output-dir.",
    )
    validation.add_argument(
        "--stop-after-stage",
        choices=validation_stage_choices(),
        default=None,
        help="Stop cleanly after the selected validation stage and mark that stage as completed.",
    )
    _add_modeling_args(validation)

    suite = sub.add_parser(
        "run-scenario-suite",
        help="Run the full pipeline on multiple synthetic scenarios and compare concept p-values",
    )
    suite.add_argument(
        "--scenarios",
        default="memory_only,nonlinearity_only,interaction_only,amplitude_only,parameter_only,noise_only,offset_only",
        help="Comma-separated scenarios to run (default: all seven)",
    )
    suite.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic data (default: 0)")
    suite.add_argument("--n-steps", type=int, default=3200, help="Synthetic series length (default: 3200)")
    suite.add_argument("--output-dir", default="outputs/scenario_suite")
    suite.add_argument("--preset", choices=["quick", "standard", "mvp", "research"], default="standard")
    suite.add_argument(
        "--window-size", type=int, default=None,
        help="Override window length W (default: from preset)",
    )
    suite.add_argument(
        "--step-size", type=int, default=None,
        help="Override window stride (default: from preset; set equal to --window-size for non-overlapping windows)",
    )
    suite.add_argument(
        "--pysr-maxsize", type=int, default=None,
        help="Override PySR maxsize (max expression nodes; bound to window data per parsimony)",
    )
    suite.add_argument(
        "--pysr-maxdepth", type=int, default=None,
        help="Override PySR maxdepth (max expression depth)",
    )
    suite.add_argument(
        "--pysr-populations", type=int, default=None,
        help="Override PySR populations (search-effort knob)",
    )
    suite.add_argument(
        "--pysr-niterations", type=int, default=None,
        help="Override PySR niterations (search-effort knob)",
    )
    suite.add_argument(
        "--pysr-ncycles-per-iteration", type=int, default=None,
        help="Override PySR ncycles_per_iteration (search-effort knob)",
    )
    suite.add_argument(
        "--nesting-level", type=int, default=1,
        help="SR nesting constraint level: 0=none, 1=no_unary_in_unary (default: 1)",
    )
    suite.add_argument(
        "--cost-level", type=int, default=0,
        help="SR operator cost level: 0=default, 1=cost×3, 2=cost×5 (default: 0)",
    )
    suite.add_argument(
        "--parsimony", type=float, default=0.01,
        help="PySR parsimony penalty (default: 0.01)",
    )
    suite.add_argument(
        "--symbolic-backend", choices=["pysr", "random"], default=None,
        help="Override symbolic backend",
    )
    suite.add_argument(
        "--top-k", type=int, default=None,
        help="Override top_k_per_seed (default: from preset)",
    )
    suite.add_argument(
        "--n-seeds", type=int, default=None,
        help="Override n_seeds, the per-window PySR ensemble size S (default: from preset)",
    )
    suite.add_argument(
        "--amplitude-scale-b", type=float, default=1.9,
        help="amplitude_only regime-B rescaling factor (default: 1.9)",
    )
    suite.add_argument(
        "--pysr-parallelism",
        choices=["serial", "multithreading", "multiprocessing"],
        default=None,
        help="PySR Julia parallelism mode (default: serial).",
    )
    suite.add_argument(
        "--pysr-procs", type=int, default=None,
        help="Julia worker procs/threads when --pysr-parallelism is not serial.",
    )
    suite.add_argument(
        "--no-pysr-deterministic",
        dest="pysr_deterministic", action="store_false", default=None,
        help="Disable PySR deterministic mode (required for non-serial parallelism).",
    )

    grid = sub.add_parser(
        "run-sr-grid-study",
        help="Run the 3x3 SR configuration grid study (nesting x operator cost)",
    )
    grid.add_argument(
        "--scenarios",
        default="nonlinearity_only,interaction_only,amplitude_only",
        help="Comma-separated scenarios (default: nonlinearity_only,interaction_only,amplitude_only)",
    )
    grid.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated random seeds (default: 0,1,2)",
    )
    grid.add_argument(
        "--nesting-levels",
        default="0,1,2",
        help="Comma-separated nesting constraint levels to sweep (default: 0,1,2)",
    )
    grid.add_argument(
        "--cost-levels",
        default="0,1,2",
        help="Comma-separated operator cost levels to sweep (default: 0,1,2)",
    )
    grid.add_argument(
        "--backend",
        default="random",
        choices=["random", "pysr"],
        help="Symbolic backend (random=fast, pysr=real; default: random)",
    )
    grid.add_argument("--n-steps", type=int, default=1200)
    grid.add_argument("--window-size", type=int, default=120)
    grid.add_argument("--noise-std", type=float, default=0.16)
    grid.add_argument("--output-dir", default="outputs/sr_grid")
    grid.add_argument("--preset", choices=["quick", "standard", "mvp", "research"], default="standard")
    grid.add_argument(
        "--parsimony",
        type=float,
        default=None,
        help=(
            "PySR parsimony penalty for expression complexity. "
            "Higher values favour simpler expressions. "
            "If not set, uses the config default (0.005)."
        ),
    )
    grid.add_argument(
        "--sr-gain-threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum SR gain above which the relative nonlinearity index is "
            "credited. effective_gain = max(0, sr_gain - threshold). "
            "Default: 0.05."
        ),
    )

    return parser


def run_synthetic(args: argparse.Namespace) -> Path:
    dataset = generate_synthetic_regime_series(
        n_steps=args.n_steps,
        scenario=args.scenario,
        noise_std=args.noise_std,
        seed=args.seed,
        memory_lag_b=args.memory_lag_b,
        nonlinearity_strength_b=args.nonlinearity_strength_b,
        interaction_strength_b=args.interaction_strength_b,
        amplitude_scale_b=args.amplitude_scale_b,
        offset_b=args.offset_b,
        noise_scale_b=args.noise_scale_b,
    )

    config = ProfilingConfig()
    _apply_config(args, config)

    profiler = StructuralProfiler(config)
    result = profiler.fit(
        dataset.values,
        dataset.labels,
        checkpoint_dir=_resolve_checkpoint_dir(args),
    )

    output_dir = write_full_report(result, args.output_dir)
    print(f"Scenario: {args.scenario}")
    print(f"Preset: {args.preset}")
    _print_result_summary(result, output_dir)
    return Path(output_dir)


def run_csv(args: argparse.Namespace) -> Path:
    feature_cols = parse_feature_columns(args.feature_cols)
    dataset = load_labeled_timeseries_csv(
        csv_path=args.data_csv,
        label_col=args.label_col,
        feature_cols=feature_cols,
        time_col=args.time_col,
        record_col=args.record_col,
        normalize=not args.no_normalize,
        dropna=not args.keep_na,
    )

    config = ProfilingConfig()
    _apply_config(args, config)

    if args.target_col is not None:
        if args.target_col not in dataset.feature_names:
            raise ValueError(f"target_col '{args.target_col}' must be one of {dataset.feature_names}")
        config.target_index = dataset.feature_names.index(args.target_col)
    else:
        if args.target_index < 0 or args.target_index >= len(dataset.feature_names):
            raise ValueError("target_index out of bounds for selected feature columns")
        config.target_index = args.target_index

    profiler = StructuralProfiler(config)
    result = profiler.fit(
        dataset.values,
        dataset.labels,
        record_ids=dataset.record_ids,
        checkpoint_dir=_resolve_checkpoint_dir(args),
    )

    output_dir = write_full_report(result, args.output_dir)
    print(f"Dataset: {args.data_csv}")
    print(f"Preset: {args.preset}")
    print(f"Target: {dataset.feature_names[config.target_index]}")
    if dataset.record_ids is not None:
        print(f"Records: {len(set(dataset.record_ids.astype(str).tolist()))}")
    _print_result_summary(result, output_dir)
    return Path(output_dir)


def run_validation_command(args: argparse.Namespace) -> Path:
    config = ProfilingConfig()
    _apply_config(args, config)

    dataset = None
    if args.suite in {"real", "all"}:
        if args.data_csv is None or args.label_col is None:
            raise ValueError("run-validation with suite real/all requires --data-csv and --label-col")
        feature_cols = parse_feature_columns(args.feature_cols)
        dataset = load_labeled_timeseries_csv(
            csv_path=args.data_csv,
            label_col=args.label_col,
            feature_cols=feature_cols,
            time_col=args.time_col,
            record_col=args.record_col,
            normalize=not args.no_normalize,
            dropna=not args.keep_na,
        )

        if args.target_col is not None:
            if args.target_col not in dataset.feature_names:
                raise ValueError(f"target_col '{args.target_col}' must be one of {dataset.feature_names}")
            config.target_index = dataset.feature_names.index(args.target_col)
        else:
            if args.target_index < 0 or args.target_index >= len(dataset.feature_names):
                raise ValueError("target_index out of bounds for selected feature columns")
            config.target_index = args.target_index

    print(f"Launching validation suite: {args.suite}")
    print(f"Preset: {args.preset}")
    print(f"Validation output directory: {args.output_dir}")

    result = run_validation(
        output_dir=args.output_dir,
        base_config=config,
        suite=args.suite,
        dataset=dataset,
        resume=args.resume,
        stop_after_stage=args.stop_after_stage,
    )
    print(f"Validation suite: {args.suite}")
    print(f"Preset: {args.preset}")
    if dataset is not None:
        print(f"Validation dataset: {args.data_csv}")
    _print_validation_summary(result, Path(args.output_dir))
    return Path(args.output_dir)


def run_sr_grid_study_command(args: argparse.Namespace) -> Path:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = ProfilingConfig()
    _apply_preset(config, args.preset)
    config.ensemble.symbolic_backend = args.backend
    if args.parsimony is not None:
        config.ensemble.pysr_parsimony = float(args.parsimony)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    nesting_levels = [int(v) for v in args.nesting_levels.split(",") if v.strip()]
    cost_levels = [int(v) for v in args.cost_levels.split(",") if v.strip()]

    print(
        f"SR grid study: {len(nesting_levels)} nesting × {len(cost_levels)} cost × "
        f"{len(scenarios)} scenarios × {len(seeds)} seeds = "
        f"{len(nesting_levels) * len(cost_levels) * len(scenarios) * len(seeds)} cells"
    )

    df = run_sr_grid_study(
        base_config=config,
        scenarios=scenarios,
        seeds=seeds,
        nesting_levels=nesting_levels,
        cost_levels=cost_levels,
        n_steps=args.n_steps,
        window_size=args.window_size,
        noise_std=args.noise_std,
        sr_gain_threshold=args.sr_gain_threshold,
    )

    results_path = out / "sr_grid_study_results.csv"
    df.to_csv(results_path, index=False)
    print(f"Results: {results_path}")

    # Parquet for efficient downstream analysis (optional - requires pyarrow)
    try:
        parquet_path = out / "sr_grid_study_results.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Parquet:  {parquet_path}")
    except Exception:
        pass

    summary_md = grid_study_summary(df)
    summary_path = out / "sr_grid_study_summary.md"
    summary_path.write_text(summary_md)
    print(f"Summary:  {summary_path}")

    # Run card
    run_card = make_run_card(
        base_config=config,
        scenarios=scenarios,
        seeds=seeds,
        nesting_levels=nesting_levels,
        cost_levels=cost_levels,
        n_steps=args.n_steps,
        window_size=args.window_size,
        noise_std=args.noise_std,
        sr_gain_threshold=args.sr_gain_threshold,
    )
    run_card_path = out / "run_card.json"
    run_card_path.write_text(json.dumps(run_card, indent=2))
    print(f"Run card: {run_card_path}")

    # Plots
    print("Generating plots...")
    plot_paths = grid_study_plots(
        df=df,
        output_dir=out,
        base_config=config,
        n_steps=args.n_steps,
        window_size=args.window_size,
        noise_std=args.noise_std,
    )
    for p in plot_paths:
        print(f"Plot:     {p}")

    return out


def run_scenario_suite_command(args: argparse.Namespace) -> Path:
    """Run the full pipeline on multiple synthetic scenarios and compare concepts."""
    import datetime

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    # Build config with best nesting/cost settings
    config = ProfilingConfig()
    _apply_preset(config, args.preset)
    if getattr(args, "window_size", None) is not None:
        config.window.window_size = int(args.window_size)
    if getattr(args, "step_size", None) is not None:
        config.window.step_size = int(args.step_size)
    if getattr(args, "pysr_maxsize", None) is not None:
        config.ensemble.pysr_maxsize = int(args.pysr_maxsize)
    if getattr(args, "pysr_maxdepth", None) is not None:
        config.ensemble.pysr_maxdepth = int(args.pysr_maxdepth)
    if getattr(args, "pysr_populations", None) is not None:
        config.ensemble.pysr_populations = int(args.pysr_populations)
    if getattr(args, "pysr_niterations", None) is not None:
        config.ensemble.pysr_niterations = int(args.pysr_niterations)
    if getattr(args, "pysr_ncycles_per_iteration", None) is not None:
        config.ensemble.pysr_ncycles_per_iteration = int(args.pysr_ncycles_per_iteration)
    if getattr(args, "symbolic_backend", None) is not None:
        config.ensemble.symbolic_backend = args.symbolic_backend
    if getattr(args, "parsimony", None) is not None:
        config.ensemble.pysr_parsimony = float(args.parsimony)
    if getattr(args, "top_k", None) is not None:
        config.ensemble.top_k_per_seed = int(args.top_k)
    if getattr(args, "n_seeds", None) is not None:
        config.ensemble.n_seeds = int(args.n_seeds)

    nesting_level = int(args.nesting_level)
    cost_level = int(args.cost_level)
    unary_ops = tuple(config.ensemble.pysr_unary_operators)
    nc = make_nested_constraints(nesting_level, unary_ops)
    if nc is not None:
        config.ensemble.pysr_nested_constraints = nc
    cc = make_complexity_of_operators(cost_level)
    if cc is not None:
        config.ensemble.pysr_complexity_of_operators = cc

    if getattr(args, "pysr_parallelism", None) is not None:
        config.ensemble.pysr_parallelism = args.pysr_parallelism
    if getattr(args, "pysr_procs", None) is not None:
        config.ensemble.pysr_procs = int(args.pysr_procs)
    if getattr(args, "pysr_deterministic", None) is not None:
        config.ensemble.pysr_deterministic = bool(args.pysr_deterministic)

    seed = int(args.seed)
    n_steps = int(args.n_steps)

    all_pairwise: list[pd.DataFrame] = []
    all_regime_profiles: list[pd.DataFrame] = []

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"{'='*60}")

        scenario_out = out / scenario
        scenario_out.mkdir(parents=True, exist_ok=True)

        dataset = generate_synthetic_regime_series(
            n_steps=n_steps,
            scenario=scenario,
            seed=seed,
            amplitude_scale_b=float(args.amplitude_scale_b),
        )

        profiler = StructuralProfiler(config)
        scenario_ckpt = (
            None
            if getattr(args, "no_checkpoint", False)
            else scenario_out / "checkpoints"
        )
        try:
            result = profiler.fit(
                dataset.values, dataset.labels, checkpoint_dir=scenario_ckpt
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        write_full_report(result, str(scenario_out))

        _print_result_summary(result, scenario_out)

        if not result.pairwise_stats.empty:
            tagged = result.pairwise_stats.copy()
            tagged.insert(0, "scenario", scenario)
            all_pairwise.append(tagged)

        if not result.regime_profiles.empty:
            tagged_rp = result.regime_profiles.copy()
            tagged_rp.insert(0, "scenario", scenario)
            all_regime_profiles.append(tagged_rp)

    # Consolidated outputs
    if all_pairwise:
        combined = pd.concat(all_pairwise, ignore_index=True)
        combined.to_csv(out / "pairwise_comparison.csv", index=False)
        print(f"\nConsolidated pairwise stats: {out / 'pairwise_comparison.csv'}")

        # Markdown comparison: p_value and median_diff per feature × scenario
        _write_pairwise_comparison_md(combined, out / "pairwise_comparison.md")
        print(f"Pairwise comparison table: {out / 'pairwise_comparison.md'}")

    if all_regime_profiles:
        pd.concat(all_regime_profiles, ignore_index=True).to_csv(
            out / "regime_profiles_all.csv", index=False
        )

    # Run card
    run_card = {
        "timestamp": datetime.datetime.now().isoformat(),
        "scenarios": scenarios,
        "seed": seed,
        "n_steps": n_steps,
        "preset": args.preset,
        "nesting_level": nesting_level,
        "cost_level": cost_level,
        "sr_config": {
            "symbolic_backend": str(config.ensemble.symbolic_backend),
            "pysr_parsimony": float(config.ensemble.pysr_parsimony)
            if config.ensemble.pysr_parsimony is not None else None,
            "n_seeds": int(config.ensemble.n_seeds),
            "max_lag": int(config.ensemble.max_lag),
        },
    }
    (out / "run_card.json").write_text(json.dumps(run_card, indent=2))
    print(f"Run card: {out / 'run_card.json'}")

    return out


def _write_pairwise_comparison_md(df: pd.DataFrame, path: Path) -> None:
    """Write a human-readable markdown table comparing p_values across scenarios."""
    alpha = 0.05
    lines = [
        "# Scenario Comparison - Pairwise Feature Tests",
        "",
        "Regime A vs B.  "
        "`p` = block permutation p-value.  "
        "`d` = median difference (A − B).  "
        "`CI` = 95 % bootstrap CI.  "
        "`*` = p < 0.05.",
        "",
    ]

    for scenario in df["scenario"].unique():
        s_df = df[df["scenario"] == scenario].copy()
        # Only A vs B rows (skip any multi-label pairs)
        pairs = s_df[["group_a", "group_b"]].drop_duplicates()
        for _, pair in pairs.iterrows():
            ga, gb = pair["group_a"], pair["group_b"]
            pair_df = s_df[(s_df["group_a"] == ga) & (s_df["group_b"] == gb)].copy()
            pair_df = pair_df.sort_values("p_value")
            lines.append(f"## {scenario}  ({ga} vs {gb})")
            lines.append("")
            header = "| feature | p | d | CI 95% | effect_iqr | n_a | n_b |"
            sep    = "|---|---|---|---|---|---|---|"
            lines += [header, sep]
            for _, row in pair_df.iterrows():
                sig = " *" if float(row["p_value"]) < alpha else ""
                ci = f"[{float(row['median_diff_ci_low']):.3f}, {float(row['median_diff_ci_high']):.3f}]"
                lines.append(
                    f"| {row['feature']}{sig} "
                    f"| {float(row['p_value']):.4f} "
                    f"| {float(row['median_diff']):.4f} "
                    f"| {ci} "
                    f"| {float(row['effect_iqr_ratio']):.3f} "
                    f"| {int(row['n_group_a_used'])} "
                    f"| {int(row['n_group_b_used'])} |"
                )
            lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-synthetic":
        run_synthetic(args)
        return

    if args.command == "run-csv":
        run_csv(args)
        return

    if args.command == "run-validation":
        run_validation_command(args)
        return

    if args.command == "run-sr-grid-study":
        run_sr_grid_study_command(args)
        return

    if args.command == "run-scenario-suite":
        run_scenario_suite_command(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
