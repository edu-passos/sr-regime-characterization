from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pipeline import ProfilingResult
from .summary import generate_run_summary


DEFAULT_RADAR_FEATURES = [
    "lag_mass",
    "long_lag_mass",
    "nonlinear_node_count",
    "interaction_term_count",
    "complexity_cost",
    "linear_symbolic_fit_gap",
]


def save_result_tables(result: ProfilingResult, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.raw_window_summaries.to_csv(out / "raw_window_summaries.csv", index=False)
    result.window_summaries.to_csv(out / "window_summaries.csv", index=False)
    result.window_evaluation_all.to_csv(out / "window_evaluation_all.csv", index=False)
    result.regime_profiles.to_csv(out / "regime_profiles.csv", index=False)
    result.pairwise_stats.to_csv(out / "pairwise_stats.csv", index=False)
    result.medoid_equations.to_csv(out / "medoid_equations.csv", index=False)
    for name, df in result.baseline_outputs.items():
        if df is None or df.empty:
            continue
        df.to_csv(out / f"{name}.csv", index=False)
    return out


def _feature_exists(df: pd.DataFrame, base: str) -> bool:
    return f"{base}_p50" in df.columns


def plot_radar_profile(
    regime_profiles: pd.DataFrame,
    output_path: str | Path,
    features: Sequence[str] = DEFAULT_RADAR_FEATURES,
) -> None:
    if regime_profiles.empty:
        return

    selected = [f for f in features if _feature_exists(regime_profiles, f)]
    if len(selected) < 3:
        return

    labels = regime_profiles["label"].astype(str).tolist()
    values = regime_profiles[[f"{f}_p50" for f in selected]].to_numpy(dtype=float)

    # Scale each feature for comparability on radar axes.
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    norm = (values - mins) / np.where(np.abs(maxs - mins) < 1e-9, 1.0, maxs - mins)

    angles = np.linspace(0, 2 * np.pi, len(selected), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)

    for i, label in enumerate(labels):
        vals = np.concatenate([norm[i], [norm[i, 0]]])
        ax.plot(angles, vals, linewidth=2, label=label)
        ax.fill(angles, vals, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(selected)
    ax.set_yticklabels([])
    ax.set_title("Regime Structural Fingerprint (normalized)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_densities(
    window_summaries: pd.DataFrame,
    output_dir: str | Path,
    features: Sequence[str],
) -> None:
    out = Path(output_dir)
    if window_summaries.empty:
        return

    labels = sorted(window_summaries["label"].astype(str).unique().tolist())

    for feature in features:
        col = f"{feature}_median"
        if col not in window_summaries.columns:
            continue

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for label in labels:
            vals = window_summaries.loc[window_summaries["label"] == label, col].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            bins = min(28, max(8, vals.size // 2))
            ax.hist(vals, bins=bins, alpha=0.42, density=True, label=label)

        ax.set_title(f"Density: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"density_{feature}.png", dpi=160)
        plt.close(fig)


def plot_medoid_equations(
    medoid_equations: pd.DataFrame,
    output_path: str | Path,
) -> None:
    if medoid_equations.empty or "label" not in medoid_equations.columns:
        return
    if "medoid_equation" not in medoid_equations.columns:
        return

    n = len(medoid_equations)
    fig, ax = plt.subplots(figsize=(11, max(3.2, 1.8 * n)))
    ax.axis("off")
    ax.set_title("Prototype (Medoid) Equations by Regime", loc="left", fontsize=13, pad=12)

    y = 0.92
    step = 0.82 / max(n, 1)
    for _, row in medoid_equations.iterrows():
        label = str(row.get("label", "n/a"))
        equation = str(row.get("medoid_equation", "n/a"))
        mse = row.get("best_symbolic_fit_mse", np.nan)
        mse_text = f"  (best_fit_mse={float(mse):.4f})" if np.isfinite(mse) else ""
        wrapped = textwrap.fill(equation, width=95)
        ax.text(
            0.01,
            y,
            f"{label}{mse_text}\n{wrapped}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            family="monospace",
        )
        y -= step

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_top_pairwise_effects(
    pairwise_stats: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 10,
) -> None:
    required = {"feature", "group_a", "group_b", "median_diff", "q_value", "p_value"}
    if pairwise_stats.empty or not required.issubset(set(pairwise_stats.columns)):
        return

    top = pairwise_stats.sort_values(["q_value", "p_value"], na_position="last").head(top_n).copy()
    if top.empty:
        return

    labels = [
        f"{str(r['feature'])} ({str(r['group_a'])} vs {str(r['group_b'])})"
        for _, r in top.iterrows()
    ]
    values = top["median_diff"].to_numpy(dtype=float)
    significant = top["q_value"].to_numpy(dtype=float) <= 0.05
    colors = np.where(significant, "#1b9e77", "#6c757d")

    fig_h = max(3.2, 0.52 * len(top) + 1.6)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, values, color=colors, alpha=0.9)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Median Difference")
    ax.set_title("Top Pairwise Structural Effects")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_window_quality(
    window_summaries: pd.DataFrame,
    output_path: str | Path,
) -> None:
    x_col = (
        "behavioral_instability_evidence_median"
        if "behavioral_instability_evidence_median" in window_summaries.columns
        else (
            "instability_evidence_median"
            if "instability_evidence_median" in window_summaries.columns
            else (
            "disagreement_evidence_median"
            if "disagreement_evidence_median" in window_summaries.columns
            else "structural_instability"
            )
        )
    )
    required = {"label", x_col, "best_symbolic_fit_mse_median"}
    if window_summaries.empty or not required.issubset(set(window_summaries.columns)):
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for label, group in window_summaries.groupby("label"):
        ax.scatter(
            group[x_col].to_numpy(dtype=float),
            group["best_symbolic_fit_mse_median"].to_numpy(dtype=float),
            label=str(label),
            alpha=0.72,
            s=28,
        )

    if x_col == "behavioral_instability_evidence_median":
        xlabel = "Behavioral Instability Evidence"
    elif x_col == "instability_evidence_median":
        xlabel = "Instability Evidence"
    elif x_col == "disagreement_evidence_median":
        xlabel = "Disagreement Evidence"
    else:
        xlabel = "Structural Instability"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best Symbolic Fit MSE (median)")
    ax.set_title("Window Quality Map")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_prototype_fit(
    prototype_fits,
    window_summaries: pd.DataFrame,
    output_path: str | Path,
) -> None:
    if not prototype_fits:
        return

    _plot_window_fit_panels(
        fits=prototype_fits,
        window_summaries=window_summaries,
        output_path=output_path,
        title_prefix="prototype",
    )


def plot_top_window_fits(
    window_fits,
    window_summaries: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 6,
) -> None:
    if not window_fits:
        return
    if top_n <= 0:
        return

    ranked = sorted(window_fits, key=lambda fit: float(fit.fit_mse))[:top_n]
    _plot_window_fit_panels(
        fits=ranked,
        window_summaries=window_summaries,
        output_path=output_path,
        title_prefix="window",
    )


def _plot_window_fit_panels(
    fits,
    window_summaries: pd.DataFrame,
    output_path: str | Path,
    title_prefix: str,
) -> None:
    if not fits:
        return

    n = len(fits)
    fig_h = max(3.4, 2.8 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(9.4, fig_h))
    if n == 1:
        axes = [axes]

    for ax, fit in zip(axes, fits):
        y_true = np.asarray(fit.y_true, dtype=float)
        y_pred = np.asarray(fit.y_pred, dtype=float)
        t = np.arange(y_true.shape[0], dtype=int)

        ax.plot(t, y_true, color="#1f77b4", linewidth=1.9, label="true")
        ax.plot(t, y_pred, color="#d62728", linewidth=1.7, alpha=0.88, label=title_prefix)

        mse = float(np.mean((y_true - y_pred) ** 2))
        denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 1e-12 else np.nan
        r2_text = f"{r2:.3f}" if np.isfinite(r2) else "n/a"

        mse_pct_text = ""
        if {
            "label",
            "best_symbolic_fit_mse_median",
        }.issubset(set(window_summaries.columns)):
            same = window_summaries[window_summaries["label"].astype(str) == str(fit.label)]
            if not same.empty:
                vals = same["best_symbolic_fit_mse_median"].to_numpy(dtype=float)
                pct = float(np.mean(vals <= mse) * 100.0)
                mse_pct_text = f", fit_mse_pct={pct:.1f}%"

        ax.set_title(
            f"Regime {fit.label} | {title_prefix} window {fit.window_index} | "
            f"fit_mse={mse:.4f}, r2={r2_text}{mse_pct_text}",
            fontsize=11,
        )
        ax.set_xlabel("Lagged Sample")
        ax.set_ylabel("Target")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_full_report(
    result: ProfilingResult,
    output_dir: str | Path,
    radar_features: Sequence[str] = DEFAULT_RADAR_FEATURES,
    density_features: Sequence[str] = DEFAULT_RADAR_FEATURES,
    summary_filename: str = "summary.md",
) -> Path:
    out = save_result_tables(result, output_dir)
    plot_radar_profile(result.regime_profiles, out / "radar_profile.png", radar_features)
    plot_feature_densities(result.window_summaries, out, density_features)
    plot_medoid_equations(result.medoid_equations, out / "medoid_equations.png")
    plot_top_pairwise_effects(result.pairwise_stats, out / "top_pairwise_effects.png")
    plot_window_quality(result.window_summaries, out / "window_quality.png")
    plot_prototype_fit(result.prototype_fits, result.window_summaries, out / "prototype_fit.png")
    plot_top_window_fits(result.window_fits, result.window_summaries, out / "window_fit_examples.png")
    generate_run_summary(out, filename=summary_filename)
    return out
