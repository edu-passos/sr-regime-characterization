"""SR configuration grid study.

Runs a 3 × 3 grid of SR configurations (nesting_level × operator_cost_level)
on a set of synthetic scenarios and returns ensemble-level diagnostics *before*
any concept extraction.  The purpose is to identify the SR configuration that
produces clean ensembles - fewest nested-unary expressions, competitive fit,
adequate Pareto diversity - as a prerequisite for the Jacobian concept comparison.

Usage
-----
Run via the CLI subcommand ``run-sr-grid-study``, or import and call
``run_sr_grid_study`` directly.
"""

from __future__ import annotations

import copy
import datetime
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import ProfilingConfig
from .features import expression_feature_table, summarize_expression_features
from .models.expressions import ExprNode, UNARY_NONLINEAR_OPS, safe_eval
from .models.jacobian import jacobian_features, symbolic_diff
from .models.ensemble import fit_symbolic_ensemble
from .scaling import standardize_window_design
from .synthetic import Scenario, generate_synthetic_regime_series
from .windowing import build_lagged_design


_STRUCTURAL_KEYS: List[str] = [
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
]


# ---------------------------------------------------------------------------
# Nesting constraint presets
# ---------------------------------------------------------------------------

_UNARY_OPS_DEFAULT = ("sin", "cos", "tanh", "exp", "log", "neg")


def make_nested_constraints(
    level: int,
    unary_ops: Sequence[str] = _UNARY_OPS_DEFAULT,
) -> Dict[str, Dict[str, int]] | None:
    """Build a ``nested_constraints`` dict for PySR.

    Parameters
    ----------
    level:
        0 - no constraints (default PySR behaviour).
        1 - no unary-inside-unary (moderate restriction).
        2 - strict: unary operators may only take linear combinations
            of lag variables as input (also blocks ``mul`` / ``div``
            from appearing inside a unary).
    unary_ops:
        Sequence of unary operator names to constrain.

    Returns
    -------
    A dict suitable for ``PySRRegressor(nested_constraints=...)``,
    or ``None`` for level 0.
    """
    if level == 0:
        return None
    unary_set = tuple(unary_ops)
    if level == 1:
        return {op: {other: 0 for other in unary_set} for op in unary_set}
    if level == 2:
        blocked = set(unary_set) | {"mul", "div"}
        return {op: {b: 0 for b in blocked} for op in unary_set}
    raise ValueError(f"level must be 0, 1, or 2; got {level}")


# ---------------------------------------------------------------------------
# Operator cost presets
# ---------------------------------------------------------------------------

_NONLINEAR_UNARY_OPS = ("sin", "cos", "tanh", "exp", "log")


def make_complexity_of_operators(cost_level: int) -> Dict[str, float] | None:
    """Build a ``complexity_of_operators`` dict.

    Parameters
    ----------
    cost_level:
        0 - default PySR cost (1 for all operators).
        1 - nonlinear unary operators cost 3.
        2 - nonlinear unary operators cost 5.
    """
    if cost_level == 0:
        return None
    cost = {1: 3.0, 2: 5.0}.get(cost_level)
    if cost is None:
        raise ValueError(f"cost_level must be 0, 1, or 2; got {cost_level}")
    return {op: cost for op in _NONLINEAR_UNARY_OPS}


# ---------------------------------------------------------------------------
# Ensemble diagnostics helpers
# ---------------------------------------------------------------------------

def has_nested_unary(
    node: ExprNode,
    unary_ops: frozenset[str] | None = None,
    _parent_is_unary: bool = False,
) -> bool:
    """Return True if ``node`` contains a unary operator directly inside another.

    Parameters
    ----------
    node:
        Root of the expression tree to inspect.
    unary_ops:
        Set of operator names considered unary.  Defaults to
        ``UNARY_NONLINEAR_OPS`` (sin, cos, tanh, exp, log).
    _parent_is_unary:
        Internal recursion flag - do not set manually.
    """
    if unary_ops is None:
        unary_ops = frozenset(UNARY_NONLINEAR_OPS)
    is_unary = node.op in unary_ops
    if is_unary and _parent_is_unary:
        return True
    for child in node.children:
        if has_nested_unary(child, unary_ops, is_unary):
            return True
    return False


def ensemble_diagnostics(
    result,
    X: np.ndarray,
    lag_for_feature: Sequence[int],
    memory_long_lag_cutoff: int = 3,
    requested_backend: str = "",
    sr_gain_threshold: float = 0.05,
) -> Dict[str, float]:
    """Compute diagnostic metrics for a fitted ensemble (no concept extraction).

    Parameters
    ----------
    result:
        ``WindowEnsembleResult`` from ``fit_symbolic_ensemble``.
    X:
        Window design matrix (standardised), shape ``(n_samples, n_features)``.
    lag_for_feature:
        Lag index (1-based) per column of X.
    memory_long_lag_cutoff:
        Threshold for long-lag classification.
    requested_backend:
        The backend that was requested (e.g. "pysr").  Compared against
        ``result.backend`` to detect silent fallbacks.

    Returns
    -------
    dict with keys:
      nested_unary_fraction
          Fraction of retained models containing a nested unary pair.
      pareto_frontier_size
          Number of retained models.
      best_fit_mse
          MSE of the best retained model on the training window.
      linear_fit_mse
          MSE of the ridge linear baseline on the same window.
      median_complexity
          Median expression complexity across retained models.
      jacobian_nonlinearity_index_median
          Median Jacobian nonlinearity index across retained models.
      jacobian_relative_nonlinearity_index_median
          jacobian_nonlinearity_index_median weighted by how much SR beats
          the linear baseline.  Zero when SR offers no improvement over linear
          (e.g. amplitude_only with a well-calibrated linear model).
      backend_used
          Actual backend used (may differ from requested if PySR crashed).
      backend_fallback
          1.0 if a silent fallback occurred, 0.0 otherwise.
      best_equation
          String representation of the best-fit retained model.
      best_equation_fit_mse
          Fit MSE of that model.
    """
    backend_used = getattr(result, "backend", "")
    fallback = 1.0 if (requested_backend and backend_used != requested_backend) else 0.0
    linear_fit_mse = float(getattr(result, "linear_fit_mse", float("nan")))

    models = result.retained_models
    if not models:
        return {
            "nested_unary_fraction": float("nan"),
            "pareto_frontier_size": 0.0,
            "best_fit_mse": float("nan"),
            "linear_fit_mse": linear_fit_mse,
            "median_complexity": float("nan"),
            "jacobian_nonlinearity_index_median": float("nan"),
            "jacobian_relative_nonlinearity_index_median": float("nan"),
            "backend_used": backend_used,
            "backend_fallback": fallback,
            "best_equation": "",
            "best_equation_fit_mse": float("nan"),
        }

    nested_count = sum(1 for m in models if has_nested_unary(m.expression))
    complexities = [m.complexity for m in models]
    best_model = min(models, key=lambda m: m.fit_mse)
    best_fit_mse = float(result.best_symbolic_fit_mse)

    # Jacobian nonlinearity index per model
    nonlin_indices: List[float] = []
    for m in models:
        try:
            feats = jacobian_features(
                m.expression, X, lag_for_feature,
                memory_long_lag_cutoff=memory_long_lag_cutoff,
            )
            nonlin_indices.append(float(feats["jacobian_nonlinearity_index"]))
        except Exception:
            nonlin_indices.append(0.0)

    jac_nonlin = float(np.median(nonlin_indices))

    # Relative nonlinearity: weight by how much SR beats the linear baseline,
    # with a threshold to suppress near-zero-gain noise fits.
    # effective_gain = max(0, sr_gain - threshold) so that amplitude_only runs
    # with sr_gain ~0.05 contribute zero, while true nonlinearity targets
    # (sr_gain 0.10-0.24) retain meaningful signal.
    if np.isfinite(linear_fit_mse) and np.isfinite(best_fit_mse) and linear_fit_mse > 1e-9:
        sr_gain = max(0.0, linear_fit_mse - best_fit_mse) / linear_fit_mse
        effective_gain = max(0.0, sr_gain - float(sr_gain_threshold))
        jac_relative = jac_nonlin * effective_gain
    else:
        jac_relative = float("nan")

    sr_gain = (
        max(0.0, linear_fit_mse - best_fit_mse) / max(linear_fit_mse, 1e-9)
        if np.isfinite(linear_fit_mse) and np.isfinite(best_fit_mse)
        else float("nan")
    )

    return {
        "nested_unary_fraction": float(nested_count) / len(models),
        "pareto_frontier_size": float(len(models)),
        "best_fit_mse": best_fit_mse,
        "linear_fit_mse": linear_fit_mse,
        "sr_gain": sr_gain,
        "median_complexity": float(np.median(complexities)),
        "jacobian_nonlinearity_index_median": jac_nonlin,
        "jacobian_relative_nonlinearity_index_median": jac_relative,
        "backend_used": backend_used,
        "backend_fallback": fallback,
        "best_equation": best_model.equation,
        "best_equation_fit_mse": float(best_model.fit_mse),
    }


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

_NESTING_LABELS = {0: "none", 1: "no_unary_in_unary", 2: "strict"}
_COST_LABELS = {0: "cost_1", 1: "cost_3", 2: "cost_5"}

_DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    "nonlinearity_only",
    "interaction_only",
    "amplitude_only",
)


def run_sr_grid_study(
    base_config: ProfilingConfig,
    scenarios: Sequence[Scenario] = _DEFAULT_SCENARIOS,
    seeds: Sequence[int] = (0, 1, 2),
    nesting_levels: Sequence[int] = (0, 1, 2),
    cost_levels: Sequence[int] = (0, 1, 2),
    n_steps: int = 1200,
    window_size: int = 120,
    noise_std: float = 0.16,
    sr_gain_threshold: float = 0.05,
) -> pd.DataFrame:
    """Run the 3 × 3 SR configuration grid study.

    For each combination of ``nesting_level`` × ``cost_level`` × ``scenario``
    × ``seed``, generates a synthetic window, fits the ensemble, and records
    diagnostic metrics.

    Parameters
    ----------
    base_config:
        Base ``ProfilingConfig`` to clone for each grid cell.  Only
        ``ensemble``-level SR parameters are overridden; window / feature /
        stats settings are preserved.
    scenarios:
        Synthetic scenarios to evaluate.
    seeds:
        Random seeds for synthetic data generation.
    nesting_levels:
        Nesting constraint levels to sweep (0=none, 1=moderate, 2=strict).
    cost_levels:
        Operator cost levels to sweep (0=default, 1=cost×3, 2=cost×5).
    n_steps:
        Length of synthetic series per seed.
    window_size:
        Window size to extract a single window from the series.
    noise_std:
        Noise standard deviation for synthetic generation.

    Returns
    -------
    Long-format ``pd.DataFrame`` with one row per
    (nesting_level, cost_level, scenario, seed) combination, plus all
    diagnostic columns from ``ensemble_diagnostics``.
    """
    rows: List[dict] = []

    unary_ops_for_config = tuple(base_config.ensemble.pysr_unary_operators)

    for nesting_level in nesting_levels:
        for cost_level in cost_levels:
            # Build the SR config override for this grid cell.
            cfg = copy.deepcopy(base_config)
            nc = make_nested_constraints(nesting_level, unary_ops_for_config)
            cfg.ensemble.pysr_nested_constraints = nc
            cfg.ensemble.pysr_complexity_of_operators = make_complexity_of_operators(cost_level)

            for scenario in scenarios:
                for seed in seeds:
                    row = _run_single_cell(
                        cfg=cfg,
                        scenario=scenario,
                        seed=seed,
                        n_steps=n_steps,
                        window_size=window_size,
                        noise_std=noise_std,
                        sr_gain_threshold=sr_gain_threshold,
                    )
                    row["nesting_level"] = nesting_level
                    row["nesting_label"] = _NESTING_LABELS[nesting_level]
                    row["cost_level"] = cost_level
                    row["cost_label"] = _COST_LABELS[cost_level]
                    row["scenario"] = scenario
                    row["seed"] = seed
                    rows.append(row)

    if not rows:
        return pd.DataFrame()

    col_order = [
        "nesting_level", "nesting_label", "cost_level", "cost_label",
        "scenario", "seed",
        "backend_used", "backend_fallback",
        "nested_unary_fraction", "pareto_frontier_size",
        "best_fit_mse", "linear_fit_mse", "sr_gain", "median_complexity",
        "jacobian_nonlinearity_index_median",
        "jacobian_relative_nonlinearity_index_median",
        *_STRUCTURAL_KEYS,
        "best_equation", "best_equation_fit_mse",
    ]
    df = pd.DataFrame(rows)
    present = [c for c in col_order if c in df.columns]
    extra = [c for c in df.columns if c not in col_order]
    return df[present + extra].reset_index(drop=True)


def _run_single_cell(
    cfg: ProfilingConfig,
    scenario: Scenario,
    seed: int,
    n_steps: int,
    window_size: int,
    noise_std: float,
    sr_gain_threshold: float = 0.05,
) -> dict:
    """Fit one ensemble cell and return diagnostics."""
    dataset = generate_synthetic_regime_series(
        n_steps=n_steps,
        scenario=scenario,
        noise_std=noise_std,
        seed=seed,
    )

    # Extract one window from regime B (the structurally distinct regime).
    values = dataset.values
    labels = dataset.labels
    b_indices = np.where(labels == "B")[0]
    if len(b_indices) < window_size:
        return {k: float("nan") for k in _diagnostic_keys()}

    start = int(b_indices[0])
    end = start + window_size
    if end > len(values):
        return {k: float("nan") for k in _diagnostic_keys()}

    window_vals = values[start:end]
    target_index = 0

    X_raw, y_raw, feature_meta = build_lagged_design(
        window_values=window_vals,
        target_index=target_index,
        max_lag=int(cfg.ensemble.max_lag),
    )
    if X_raw.shape[0] == 0:
        return {k: float("nan") for k in _diagnostic_keys()}

    X_model, y_model, scaling = standardize_window_design(
        X_raw, y_raw, enabled=bool(cfg.ensemble.normalize_window_for_symbolic)
    )

    try:
        result = fit_symbolic_ensemble(
            X_fit=X_model,
            y_fit=y_model,
            feature_meta=feature_meta,
            target_index=target_index,
            config=cfg.ensemble,
            feature_config=cfg.features,
        )
    except Exception:
        return {k: float("nan") for k in _diagnostic_keys()}

    lag_for_feature = [fm.lag for fm in feature_meta]
    row = ensemble_diagnostics(
        result,
        X=X_model,
        lag_for_feature=lag_for_feature,
        memory_long_lag_cutoff=int(cfg.features.memory_long_lag_cutoff),
        requested_backend=str(cfg.ensemble.symbolic_backend),
        sr_gain_threshold=sr_gain_threshold,
    )

    # Structural features (expression-tree layer)
    try:
        ft = expression_feature_table(
            models=result.retained_models,
            X_reference=X_model,
            feature_meta=feature_meta,
            config=cfg.features,
            y_reference=y_model,
            linear_fit_mse=float(result.linear_fit_mse),
        )
        if not ft.empty:
            struct = summarize_expression_features(
                ft, linear_fit_mse=float(result.linear_fit_mse)
            )
            for k in _STRUCTURAL_KEYS:
                row[k] = float(struct.get(k, float("nan")))
    except Exception:
        pass

    for k in _STRUCTURAL_KEYS:
        row.setdefault(k, float("nan"))

    return row


def _diagnostic_keys() -> List[str]:
    return [
        "nested_unary_fraction",
        "pareto_frontier_size",
        "best_fit_mse",
        "linear_fit_mse",
        "sr_gain",
        "median_complexity",
        "jacobian_nonlinearity_index_median",
        "jacobian_relative_nonlinearity_index_median",
        "backend_used",
        "backend_fallback",
        "best_equation",
        "best_equation_fit_mse",
        *_STRUCTURAL_KEYS,
    ]


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _df_to_md(df: pd.DataFrame) -> str:
    """Minimal markdown table without requiring tabulate."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for _, row in df.iterrows():
        parts = []
        for c in cols:
            v = row[c]
            parts.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        rows.append("| " + " | ".join(parts) + " |")
    return "\n".join([header, sep] + rows)


def grid_study_summary(df: pd.DataFrame) -> str:
    """Generate a markdown summary of grid study results.

    Groups by (nesting_label, cost_label, scenario) and reports median
    diagnostic metrics.
    """
    if df.empty:
        return "No results.\n"

    lines = ["# SR Configuration Grid Study - Summary\n"]
    lines.append(
        "Metrics are medians across seeds.  "
        "`nested_unary_fraction`: lower is better.  "
        "`pareto_frontier_size`: higher (more diverse) is better.  "
        "`best_fit_mse`: lower is better.  "
        "`jacobian_nonlinearity_index_median`: nonlinearity of fitted expressions.  "
        "`jacobian_relative_nonlinearity_index_median`: nonlinearity weighted by "
        "SR-vs-linear gain - should be elevated for nonlinearity_only, "
        "near zero for amplitude_only.  "
        "`backend_fallback`: 1.0 = PySR crashed and fell back to random (results invalid).\n"
    )

    metric_cols = [
        "backend_fallback",
        "nested_unary_fraction",
        "pareto_frontier_size",
        "best_fit_mse",
        "linear_fit_mse",
        "jacobian_nonlinearity_index_median",
        "jacobian_relative_nonlinearity_index_median",
    ]
    present_metrics = [c for c in metric_cols if c in df.columns]

    for scenario in df["scenario"].unique():
        lines.append(f"## Scenario: {scenario}\n")
        subset = df[df["scenario"] == scenario]
        grouped = (
            subset.groupby(["nesting_label", "cost_label"])[present_metrics]
            .median()
            .reset_index()
        )
        lines.append(_df_to_md(grouped))
        lines.append("")

        # Best equation per config cell (lowest best_equation_fit_mse across seeds)
        if "best_equation" in df.columns and "best_equation_fit_mse" in df.columns:
            lines.append("### Best equations per configuration\n")
            for (nesting_label, cost_label), grp in subset.groupby(["nesting_label", "cost_label"]):
                best_row = grp.loc[grp["best_equation_fit_mse"].idxmin()]
                eq = str(best_row.get("best_equation", ""))
                mse_val = best_row.get("best_equation_fit_mse", float("nan"))
                seed_val = int(best_row.get("seed", -1))
                mse_str = f"{float(mse_val):.4f}" if str(mse_val) != "nan" else "nan"
                lines.append(
                    f"- **{nesting_label} / {cost_label}** "
                    f"(seed {seed_val}, fit_mse={mse_str}): `{eq}`"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run card
# ---------------------------------------------------------------------------

def make_run_card(
    base_config: ProfilingConfig,
    scenarios: Sequence[str],
    seeds: Sequence[int],
    nesting_levels: Sequence[int],
    cost_levels: Sequence[int],
    n_steps: int,
    window_size: int,
    noise_std: float,
    sr_gain_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Build a reproducibility record for one grid study execution."""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": git_commit,
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "nesting_levels": list(nesting_levels),
        "cost_levels": list(cost_levels),
        "n_steps": n_steps,
        "window_size": window_size,
        "noise_std": noise_std,
        "sr_gain_threshold": sr_gain_threshold,
        "sr_config": {
            "symbolic_backend": str(base_config.ensemble.symbolic_backend),
            "n_seeds": int(base_config.ensemble.n_seeds),
            "n_candidates_per_seed": int(base_config.ensemble.n_candidates_per_seed),
            "top_k_per_seed": int(base_config.ensemble.top_k_per_seed),
            "max_lag": int(base_config.ensemble.max_lag),
            "pysr_parsimony": float(base_config.ensemble.pysr_parsimony)
            if base_config.ensemble.pysr_parsimony is not None else None,
            "pysr_niterations": int(base_config.ensemble.pysr_niterations)
            if base_config.ensemble.pysr_niterations is not None else None,
            "pysr_populations": int(base_config.ensemble.pysr_populations)
            if base_config.ensemble.pysr_populations is not None else None,
        },
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

_SCENARIO_COLORS = {
    "nonlinearity_only": "#e15759",
    "interaction_only": "#4e79a7",
    "amplitude_only": "#59a14f",
    "memory_only": "#f28e2b",
    "parameter_only": "#b07aa1",
    "noise_only": "#9c755f",
    "offset_only": "#bab0ac",
}


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _violin_by_scenario(
    df: pd.DataFrame,
    feature: str,
    ax,
    title: str,
    ylabel: str,
) -> None:
    """Draw a violin/strip plot of *feature* split by scenario."""
    scenarios = list(df["scenario"].unique())
    data = [
        df.loc[df["scenario"] == s, feature].dropna().values
        for s in scenarios
    ]
    colors = [_SCENARIO_COLORS.get(s, "#aaaaaa") for s in scenarios]
    parts = ax.violinplot(data, positions=range(len(scenarios)), showmedians=True)
    for i, (body, color) in enumerate(zip(parts["bodies"], colors)):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    for col in ("cmedians", "cmins", "cmaxes", "cbars"):
        if col in parts:
            parts[col].set_color("black")
            parts[col].set_linewidth(1.0)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace("_only", "") for s in scenarios], rotation=20, ha="right")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)


def grid_study_plots(
    df: pd.DataFrame,
    output_dir: Path,
    base_config: ProfilingConfig | None = None,
    n_steps: int = 2000,
    window_size: int = 200,
    noise_std: float = 0.16,
) -> List[Path]:
    """Generate all grid study visualisations and save them to *output_dir*.

    Returns a list of paths to the files written.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return []

    output_dir = Path(output_dir)
    written: List[Path] = []

    # Filter out fallback rows for all plots
    clean = df[df.get("backend_fallback", pd.Series(0, index=df.index)).fillna(0) == 0].copy()

    # ---- 1. Violin: Jacobian relative nonlinearity by scenario ----------------
    if "jacobian_relative_nonlinearity_index_median" in clean.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        _violin_by_scenario(
            clean, "jacobian_relative_nonlinearity_index_median", ax,
            title="Relative Jacobian Nonlinearity Index by Scenario",
            ylabel="jacobian_relative_nonlinearity_index",
        )
        fig.tight_layout()
        p = output_dir / "plot_jacobian_relative_by_scenario.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # ---- 2. Violin: structural concepts by scenario ---------------------------
    struct_plot_pairs = [
        ("lag_mass_median", "Lag Mass (memory depth)"),
        ("interaction_term_count_median", "Interaction Term Count"),
        ("nonlinear_node_count_median", "Nonlinear Node Count"),
    ]
    available_struct = [(f, lbl) for f, lbl in struct_plot_pairs if f in clean.columns]
    if available_struct:
        fig, axes = plt.subplots(1, len(available_struct), figsize=(5 * len(available_struct), 4))
        if len(available_struct) == 1:
            axes = [axes]
        for ax, (feat, lbl) in zip(axes, available_struct):
            _violin_by_scenario(clean, feat, ax, title=lbl, ylabel=feat.replace("_median", ""))
        fig.suptitle("Structural Concepts by Scenario", fontsize=11)
        fig.tight_layout()
        p = output_dir / "plot_structural_concepts_by_scenario.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # ---- 3. Sensitivity: operator cost vs relative nonlinearity ---------------
    if "cost_label" in clean.columns and "jacobian_relative_nonlinearity_index_median" in clean.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        cost_order = ["cost_1", "cost_3", "cost_5"]
        for scenario, grp in clean.groupby("scenario"):
            color = _SCENARIO_COLORS.get(scenario, "#aaaaaa")
            medians = (
                grp.groupby("cost_label")["jacobian_relative_nonlinearity_index_median"]
                .median()
                .reindex(cost_order)
            )
            ax.plot(
                range(len(cost_order)), medians.values,
                marker="o", label=scenario.replace("_only", ""),
                color=color, linewidth=1.5,
            )
        ax.set_xticks(range(len(cost_order)))
        ax.set_xticklabels(cost_order)
        ax.set_xlabel("Operator cost level")
        ax.set_ylabel("Relative Jacobian nonlinearity (median)")
        ax.set_title("Sensitivity to Operator Cost")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = output_dir / "plot_cost_sensitivity.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # ---- 4. Scatter: SR gain vs raw Jacobian nonlinearity ---------------------
    if "sr_gain" in clean.columns and "jacobian_nonlinearity_index_median" in clean.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        for scenario, grp in clean.groupby("scenario"):
            color = _SCENARIO_COLORS.get(scenario, "#aaaaaa")
            ax.scatter(
                grp["sr_gain"],
                grp["jacobian_nonlinearity_index_median"],
                color=color, alpha=0.6, s=30,
                label=scenario.replace("_only", ""),
            )
        ax.set_xlabel("SR gain  (linear_mse − best_mse) / linear_mse")
        ax.set_ylabel("Raw Jacobian nonlinearity index")
        ax.set_title("SR Gain vs Jacobian Nonlinearity\n(color = scenario)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = output_dir / "plot_sr_gain_vs_jacobian.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # ---- 5. Gradient profiles -------------------------------------------------
    if base_config is not None:
        gradient_paths = _plot_gradient_profiles(
            df=clean,
            base_config=base_config,
            output_dir=output_dir,
            n_steps=n_steps,
            window_size=window_size,
            noise_std=noise_std,
        )
        written.extend(gradient_paths)

    return written


def _plot_gradient_profiles(
    df: pd.DataFrame,
    base_config: ProfilingConfig,
    output_dir: Path,
    n_steps: int,
    window_size: int,
    noise_std: float,
) -> List[Path]:
    """Re-fit the best cell per scenario and plot ∂f/∂x₀ vs x₀."""
    plt = _try_import_matplotlib()
    if plt is None:
        return []

    written: List[Path] = []
    target_scenarios = [s for s in ["nonlinearity_only", "amplitude_only"] if s in df["scenario"].unique()]

    for scenario in target_scenarios:
        sub = df[df["scenario"] == scenario]
        if sub.empty:
            continue
        # Pick the row with lowest best_fit_mse among non-fallback rows
        valid = sub[sub.get("backend_fallback", pd.Series(0, index=sub.index)).fillna(0) == 0]
        if valid.empty:
            valid = sub
        best_row = valid.loc[valid["best_fit_mse"].idxmin()]
        seed = int(best_row["seed"])
        nesting_level = int(best_row["nesting_level"])
        cost_level = int(best_row["cost_level"])

        cfg = copy.deepcopy(base_config)
        unary_ops = tuple(cfg.ensemble.pysr_unary_operators)
        cfg.ensemble.pysr_nested_constraints = make_nested_constraints(nesting_level, unary_ops)
        cfg.ensemble.pysr_complexity_of_operators = make_complexity_of_operators(cost_level)

        try:
            X_model, y_model, feature_meta, best_model = _refit_cell_for_plot(
                cfg=cfg, scenario=scenario, seed=seed,
                n_steps=n_steps, window_size=window_size, noise_std=noise_std,
            )
        except Exception:
            continue

        if X_model is None or best_model is None:
            continue

        fig, axes = plt.subplots(1, min(3, X_model.shape[1]), figsize=(4 * min(3, X_model.shape[1]), 3.5))
        if X_model.shape[1] == 1:
            axes = [axes]
        for j, ax in enumerate(axes):
            if j >= X_model.shape[1]:
                break
            try:
                grad_node = symbolic_diff(best_model.expression, j)
                grad_vals = np.asarray(safe_eval(grad_node, X_model), dtype=float)
                x_vals = X_model[:, j]
                order = np.argsort(x_vals)
                ax.scatter(x_vals[order], grad_vals[order], s=4, alpha=0.5, color="#4e79a7")
                ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
                fm = feature_meta[j]
                ax.set_xlabel(f"x lag{fm.lag}", fontsize=9)
                ax.set_ylabel(f"∂f/∂x_lag{fm.lag}", fontsize=9)
            except Exception:
                ax.set_visible(False)
        title_eq = str(best_row.get("best_equation", ""))[:80]
        fig.suptitle(
            f"Gradient profile - {scenario}\n`{title_eq}`",
            fontsize=8, wrap=True,
        )
        fig.tight_layout()
        p = output_dir / f"plot_gradient_profile_{scenario}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    return written


def _refit_cell_for_plot(
    cfg: ProfilingConfig,
    scenario: Scenario,
    seed: int,
    n_steps: int,
    window_size: int,
    noise_std: float,
):
    """Re-generate the window and refit the ensemble for gradient plotting.

    Returns (X_model, y_model, feature_meta, best_model) or (None, None, None, None).
    """
    dataset = generate_synthetic_regime_series(
        n_steps=n_steps, scenario=scenario, noise_std=noise_std, seed=seed,
    )
    values = dataset.values
    labels = dataset.labels
    b_indices = np.where(labels == "B")[0]
    if len(b_indices) < window_size:
        return None, None, None, None

    start = int(b_indices[0])
    end = start + window_size
    if end > len(values):
        return None, None, None, None

    window_vals = values[start:end]
    X_raw, y_raw, feature_meta = build_lagged_design(
        window_values=window_vals, target_index=0, max_lag=int(cfg.ensemble.max_lag),
    )
    if X_raw.shape[0] == 0:
        return None, None, None, None

    X_model, y_model, _ = standardize_window_design(
        X_raw, y_raw, enabled=bool(cfg.ensemble.normalize_window_for_symbolic),
    )
    result = fit_symbolic_ensemble(
        X_fit=X_model, y_fit=y_model, feature_meta=feature_meta,
        target_index=0, config=cfg.ensemble, feature_config=cfg.features,
    )
    if not result.retained_models:
        return None, None, None, None

    best_model = min(result.retained_models, key=lambda m: m.fit_mse)
    return X_model, y_model, list(feature_meta), best_model
