from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def generate_run_summary(output_dir: str | Path, filename: str = "summary.md") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    window_df = _safe_read_csv(out / "window_summaries.csv")
    eval_all_df = _safe_read_csv(out / "window_evaluation_all.csv")
    regime_df = _safe_read_csv(out / "regime_profiles.csv")
    pairwise_df = _safe_read_csv(out / "pairwise_stats.csv")
    medoid_df = _safe_read_csv(out / "medoid_equations.csv")
    baseline_df = _safe_read_csv(out / "baseline_window_summaries.csv")
    embedding_df = _safe_read_csv(out / "embedding_baseline_metrics.csv")

    lines: list[str] = []
    lines.append("# Run Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    lines.append("## Overview")
    if window_df.empty:
        lines.append("- No window summaries found.")
    else:
        labels = sorted(window_df["label"].astype(str).unique().tolist()) if "label" in window_df.columns else []
        backends = (
            sorted(window_df["symbolic_backend"].astype(str).unique().tolist())
            if "symbolic_backend" in window_df.columns
            else []
        )
        lines.append(f"- Windows retained: {len(window_df)}")
        lines.append(f"- Labels: {', '.join(labels) if labels else 'n/a'}")
        if "record_id" in window_df.columns:
            records = sorted(window_df["record_id"].dropna().astype(str).unique().tolist())
            lines.append(f"- Records represented: {len(records)}")
        lines.append(f"- Symbolic backend(s): {', '.join(backends) if backends else 'n/a'}")
        lines.append("- Canonical fit: full-window direct symbolic fitting")
    if not eval_all_df.empty:
        lines.append(f"- Windows evaluated (all): {len(eval_all_df)}")
        if "has_structural_summary" in eval_all_df.columns:
            n_structural = int(
                pd.to_numeric(eval_all_df["has_structural_summary"], errors="coerce").fillna(0).sum()
            )
            lines.append(f"- Windows with structural summaries: {n_structural}")
        if "included_in_window_summaries" in eval_all_df.columns:
            n_kept = int(
                pd.to_numeric(eval_all_df["included_in_window_summaries"], errors="coerce")
                .fillna(0)
                .sum()
            )
            coverage = float(n_kept / max(len(eval_all_df), 1))
            lines.append(f"- Retained coverage: {_fmt(coverage)}")
    lines.append("")

    lines.append("## Performance")
    if eval_all_df.empty and window_df.empty:
        lines.append("- Performance tables not available.")
    else:
        def _append_perf_block(name: str, frame: pd.DataFrame) -> None:
            gap_col = None
            best_col = None
            linear_col = None
            if {"linear_symbolic_fit_gap_median", "best_symbolic_fit_mse_median", "linear_fit_mse_median"}.issubset(
                set(frame.columns)
            ):
                gap_col = "linear_symbolic_fit_gap_median"
                best_col = "best_symbolic_fit_mse_median"
                linear_col = "linear_fit_mse_median"
            elif {"linear_symbolic_gap_median", "best_symbolic_mse_median", "linear_mse_median"}.issubset(
                set(frame.columns)
            ):
                gap_col = "linear_symbolic_gap_median"
                best_col = "best_symbolic_mse_median"
                linear_col = "linear_mse_median"
            if gap_col is None or best_col is None or linear_col is None:
                return

            gap = pd.to_numeric(frame[gap_col], errors="coerce")
            best = pd.to_numeric(frame[best_col], errors="coerce")
            linear = pd.to_numeric(frame[linear_col], errors="coerce")
            valid = gap.notna() & best.notna() & linear.notna()
            if not valid.any():
                return

            gap_v = gap[valid]
            best_v = best[valid]
            linear_v = linear[valid]
            lines.append(
                "- "
                + f"{name}: windows={int(valid.sum())}, "
                + f"gap_p50={_fmt(float(gap_v.median()))}, "
                + f"win_rate(gap>0)={_fmt(float((gap_v > 0).mean()))}, "
                + f"symbolic_mse_p50={_fmt(float(best_v.median()))}, "
                + f"linear_mse_p50={_fmt(float(linear_v.median()))}"
            )

        _append_perf_block("All windows (objective evaluation)", eval_all_df)
        _append_perf_block("Retained windows (structural analysis)", window_df)
    lines.append("")

    lines.append("## Fidelity")
    if pairwise_df.empty:
        lines.append("- Pairwise stats not available.")
    else:
        reject_count = int(pairwise_df["reject"].sum()) if "reject" in pairwise_df.columns else 0
        lines.append(f"- Tests run: {len(pairwise_df)}")
        lines.append(f"- FDR-significant tests: {reject_count}")
        if "reject_and_relevant" in pairwise_df.columns:
            lines.append(f"- Significant + practically relevant: {int(pairwise_df['reject_and_relevant'].sum())}")
        if "effect_iqr_ratio" in pairwise_df.columns:
            threshold = pairwise_df["effect_iqr_ratio"].quantile(0.5)
            lines.append(f"- Median effect/IQR ratio: {_fmt(float(threshold))}")

        cols = [c for c in ["feature", "median_diff", "cliffs_delta", "q_value"] if c in pairwise_df.columns]
        top = pairwise_df.sort_values(["q_value", "p_value"], na_position="last").head(6)
        if cols and not top.empty:
            lines.append("")
            lines.append("Top effects:")
            for _, row in top.iterrows():
                ci_text = ""
                if "median_diff_ci_low" in top.columns and "median_diff_ci_high" in top.columns:
                    ci_text = (
                        f"95% CI=[{_fmt(row.get('median_diff_ci_low'))}, "
                        f"{_fmt(row.get('median_diff_ci_high'))}]"
                    )
                lines.append(
                    "- "
                    + f"{row.get('feature', 'n/a')}: "
                    + f"median_diff={_fmt(row.get('median_diff'))}, "
                    + (ci_text + ", " if ci_text else "")
                    + f"Cliff's delta={_fmt(row.get('cliffs_delta'))}, "
                    + f"q={_fmt(row.get('q_value'))}"
                )
    lines.append("")

    lines.append("## Stability")
    if window_df.empty:
        lines.append("- Window-level stability is not available.")
    else:
        if "structural_instability" in window_df.columns:
            instability = window_df["structural_instability"].to_numpy(dtype=float)
            lines.append(f"- Median structural instability: {_fmt(float(np.median(instability)))}")
            lines.append(f"- P90 structural instability: {_fmt(float(np.quantile(instability, 0.90)))}")
        if "is_uncertain" in window_df.columns:
            uncertain = int(window_df["is_uncertain"].astype(bool).sum())
            lines.append(f"- Windows flagged uncertain: {uncertain}")
        if "uncertainty_filter_fallback_used" in window_df.columns:
            fallback = bool(window_df["uncertainty_filter_fallback_used"].astype(bool).any())
            lines.append(f"- Uncertainty-filter fallback used: {fallback}")
    lines.append("")

    lines.append("## Sensitivity")
    if pairwise_df.empty:
        lines.append("- Directed sensitivity evidence is not available in this run.")
    else:
        top = pairwise_df.sort_values(["q_value", "p_value"], na_position="last").head(3)
        if top.empty:
            lines.append("- No ranked structural effects available.")
        else:
            lines.append("- Strongest observed structural contrasts:")
            for _, row in top.iterrows():
                lines.append(
                    "- "
                    + f"{row.get('feature', 'n/a')}: "
                    + f"median_diff={_fmt(row.get('median_diff'))}, "
                    + f"q={_fmt(row.get('q_value'))}"
                )
    lines.append("")

    lines.append("## Specificity")
    lines.append("- Symbolic equations remain descriptive exemplars, not identified mechanisms.")
    if (out / "validation_summary.md").exists():
        lines.append("- A dedicated validation harness is available in `validation_summary.md`.")
    else:
        lines.append("- No standalone validation harness report was found in this output directory.")
    if "record_id" not in window_df.columns:
        lines.append("- Record-level replication metadata was not provided for this run.")
    lines.append("")

    lines.append("## Regime Profiles (P50)")
    if regime_df.empty:
        lines.append("- Regime profile table not available.")
    else:
        candidate_cols = [
            "label",
            "n_windows",
            "lag_mass_p50",
            "lag_mass_mean",
            "long_lag_mass_p50",
            "long_lag_mass_mean",
            "lag_entropy_p50",
            "lag_entropy_mean",
            "active_lag_count_p50",
            "active_lag_count_mean",
            "nonlinear_node_count_p50",
            "nonlinear_node_count_mean",
            "nonlinear_composition_depth_p50",
            "nonlinear_composition_depth_mean",
            "interaction_term_count_p50",
            "interaction_term_count_mean",
            "interaction_order_p50",
            "interaction_order_mean",
            "complexity_cost_p50",
            "complexity_cost_mean",
            "description_length_p50",
            "description_length_mean",
            "linear_symbolic_fit_gap_p50",
            "linear_symbolic_fit_gap_mean",
        ]
        cols = [c for c in candidate_cols if c in regime_df.columns]
        if cols:
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            for _, row in regime_df[cols].iterrows():
                lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
        else:
            lines.append("- Regime profile columns not found.")
    lines.append("")

    lines.append("## Medoid Equations")
    if medoid_df.empty:
        lines.append("- Medoid equations not available.")
    else:
        for _, row in medoid_df.iterrows():
            lines.append(f"- Label {row.get('label', 'n/a')}: `{row.get('medoid_equation', 'n/a')}`")
    lines.append("")

    lines.append("## Baseline Snapshot")
    if baseline_df.empty:
        lines.append("- Baseline window summaries not available.")
    else:
        fit_col = (
            "best_symbolic_fit_mse_median"
            if "best_symbolic_fit_mse_median" in baseline_df.columns
            else "best_symbolic_mse_median"
        )
        grouped = baseline_df.groupby("baseline_name", as_index=False).agg(
            n_windows=("window_index", "count"),
            median_fit_mse=(fit_col, "median"),
        )
        for _, row in grouped.iterrows():
            lines.append(
                f"- {row['baseline_name']}: windows={int(row['n_windows'])}, "
                f"median_fit_mse={_fmt(float(row['median_fit_mse']))}"
            )

    if not embedding_df.empty:
        r = embedding_df.iloc[0]
        lines.append(
            "- embedding_separation: "
            + f"cv_accuracy={_fmt(r.get('cv_accuracy'))}, "
            + f"balanced_accuracy={_fmt(r.get('cv_balanced_accuracy'))}, "
            + f"silhouette={_fmt(r.get('silhouette_score'))}"
        )

    summary_path = out / filename
    summary_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return summary_path
