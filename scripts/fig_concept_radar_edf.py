"""Sleep-EDF concept radar (Sleep baseline -> Wake departure), v2 run.

Single panel (one regime pair), same visual language as the synthetic radar
(scripts/fig_concept_radar.py); each is zoomed to its own departures. Built from the v2 run's per-subject
window summaries (outputs/sleep_edf/subject_*/), pooling whatever subjects
have completed, over all 16 canonical concepts (incl. the two ensemble-mean Hessian
concepts). Departure z = median over subjects of the per-subject (median Wake - median
Sleep), in units of the Sleep-window IQR pooled across subjects -- the across-subject
analog of the synthetic radar's per-seed departure. Sleep is the reference regime so the
Wake departure reads outward, matching the synthetic radar.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent

from _paths import SLEEP_EDF
SRC_DIR = SLEEP_EDF
# Sleep is the baseline reference, Wake is the departure (swapped so the departure reads
# outward, matching the synthetic radar's positive perturbations).
BASELINE, DEPARTURE = "state_sleep", "state_wake"

# The canonical 16-concept descriptor (matches scripts/across_seed_inference.py GROUPS):
# 6 lag-order + 4 multiplicative-interaction + 5 nonlinearity + 1 complexity.
GROUPS = [
    ("Lag order", "#a6cee3", "#1f78b4", [
        "lag_mass_median", "long_lag_mass_median", "lag_entropy_median",
        "active_lag_count_median", "jacobian_lag_mass_median",
        "jacobian_long_lag_mass_median"]),
    ("Mult. interaction", "#fb9a99", "#e31a1c", [
        "interaction_term_count_median", "interaction_order_median",
        "interaction_depth_median", "hessian_interaction_mean"]),
    ("Nonlinearity", "#b2df8a", "#33a02c", [
        "nonlinear_node_count_median", "jacobian_centroid_norm_median",
        "jacobian_nonlinearity_index_median", "hessian_curvature_mean",
        "jacobian_relative_nonlinearity_index_median"]),
    ("Complexity", "#cab2d6", "#6a3d9a", [
        "complexity_cost_median"]),
]
CONCEPTS = [c for _, _, _, cs in GROUPS for c in cs]
assert len(CONCEPTS) == 16

SHORT = {
    "lag_mass_median": "Lag centroid", "long_lag_mass_median": "High-lag share",
    "lag_entropy_median": "Lag diversity", "active_lag_count_median": "Active lags",
    "jacobian_lag_mass_median": "Grad. magnitude",
    "jacobian_long_lag_mass_median": "Grad. high-lag share",
    "interaction_term_count_median": "Interaction terms",
    "interaction_order_median": "Interaction order",
    "interaction_depth_median": "Interaction depth",
    "hessian_interaction_mean": "Interaction curvature",
    "nonlinear_node_count_median": "Nonlinear ops",
    "jacobian_centroid_norm_median": "Grad. sensitivity",
    "jacobian_nonlinearity_index_median": "Grad. variation",
    "hessian_curvature_mean": "Single-lag curvature",
    "jacobian_relative_nonlinearity_index_median": "Gated grad. variation",
    "complexity_cost_median": "Expression cost",
}

Z_MIN, Z_MAX = -0.25, 0.6
Z_RINGS = [-0.25, 0.0, 0.25, 0.5]
A_STYLE = ("0.30", "--")
B_STYLE = ("#c0392b", "-")


def _radius(z):
    return (np.clip(z, Z_MIN, Z_MAX) - Z_MIN) / (Z_MAX - Z_MIN)


def departures() -> pd.DataFrame:
    dirs = sorted([p for p in SRC_DIR.glob("subject_*") if p.is_dir()],
                  key=lambda p: int(p.name.split("_")[1]))
    files = [d / "window_summaries.csv" for d in dirs
             if (d / "window_summaries.csv").exists()]
    if not files:
        raise SystemExit(f"no completed subjects under {SRC_DIR}")
    frames = [pd.read_csv(f)[CONCEPTS + ["label"]] for f in files]
    pooled = pd.concat(frames, ignore_index=True)
    rows = []
    for c in CONCEPTS:
        base = pooled.loc[pooled.label == BASELINE, c].to_numpy(float)
        # baseline spread; fall back to both-regime IQR if Sleep is flat (zero-inflated
        # concepts), and treat a near-constant concept as no departure (z = 0).
        iqr = abs(np.subtract(*np.percentile(base, [75, 25])))
        if iqr < 1e-9:
            iqr = abs(np.subtract(*np.percentile(pooled[c].to_numpy(float), [75, 25])))
        # per-subject median(Wake) - median(Sleep), then median across subjects (robust
        # to the zero-inflated, spike-prone concepts where a mean explodes).
        diffs = [np.median(fr.loc[fr.label == DEPARTURE, c].to_numpy(float))
                 - np.median(fr.loc[fr.label == BASELINE, c].to_numpy(float))
                 for fr in frames
                 if (fr.label == DEPARTURE).any() and (fr.label == BASELINE).any()]
        z = 0.0 if iqr < 1e-9 else float(np.median(diffs)) / iqr
        rows.append({"concept": c, "z": z})
    out = pd.DataFrame(rows).set_index("concept").loc[CONCEPTS]
    out.attrs["n_subjects"] = len(files)
    return out


def main() -> None:
    cells = departures()
    n_subj = cells.attrs.get("n_subjects", 0)
    print(f"Sleep-EDF v2 departures ({n_subj} subjects, "
          "z = median_subj(median Wake - median Sleep)/IQR_Sleep):")
    for c in CONCEPTS:
        print(f"  {SHORT[c]:24s} z={cells.loc[c,'z']:+.2f}")

    n = len(CONCEPTS)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    th = np.concatenate([angles, angles[:1]])
    dtheta = 2.0 * np.pi / n
    numbers = [str(i + 1) for i in range(n)]
    r0 = float(_radius(0.0))

    fig = plt.figure(figsize=(12.8, 6.3))
    # No standalone title (the bottom legend names Wake/Sleep; the figure caption names the
    # dataset). Radar sits left with a margin on BOTH sides, so its spoke numbers never run
    # into the numbered key on the right. Axes ~square (in inches) -> no internal slack.
    ax = fig.add_axes([0.035, 0.09, 0.37, 0.84], projection="polar")
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.0)

    start = 0
    for _, colour, _, cs in GROUPS:
        span = len(cs) * dtheta
        mid = angles[start] - dtheta / 2.0 + span / 2.0
        ax.bar(mid, 1.0, width=span, bottom=0.0, color=colour, alpha=0.30,
               edgecolor="none", zorder=0)
        start += len(cs)

    ax.set_rgrids([_radius(zr) for zr in Z_RINGS], labels=[], angle=90)
    ax.grid(color="0.75", lw=0.5)
    for zr in Z_RINGS:
        if zr == 0:
            continue
        ax.text(np.pi / n, _radius(zr), f"{zr:+g}", fontsize=12.5, color="0.45",
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    ax.plot(th, np.full_like(th, r0), color=A_STYLE[0], lw=1.8, ls=A_STYLE[1], zorder=3)
    z = cells["z"].to_numpy()
    rb = np.concatenate([_radius(z), _radius(z)[:1]])
    ax.fill(np.concatenate([th, th[::-1]]),
            np.concatenate([rb, np.full_like(rb, r0)[::-1]]),
            color=B_STYLE[0], alpha=0.15, lw=0, zorder=2)
    ax.plot(th, rb, color=B_STYLE[0], lw=1.8, ls=B_STYLE[1], zorder=3)
    for i, zi in enumerate(z):
        if zi > Z_MAX or zi < Z_MIN:
            ax.annotate(f"{zi:+.1f}", (angles[i], float(_radius(zi))), fontsize=13,
                        color=B_STYLE[0], fontweight="bold", ha="center", va="bottom", zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels(numbers, fontsize=16)
    ax.tick_params(axis="x", pad=2)
    ax.spines["polar"].set_color("0.6")

    # numbered key: two grouped columns filling the right half, numbering 1..n in order
    key_cols = [(0.45, GROUPS[:2]), (0.71, GROUPS[2:])]
    y_top, dy, gap = 0.965, 0.074, 0.022
    num = 1
    for x, groups in key_cols:
        y = y_top
        for gname, _, dark, cs in groups:
            fig.text(x, y, gname, fontsize=17, fontweight="bold", color=dark)
            y -= dy
            for c in cs:
                fig.text(x, y, f"{num}.  {SHORT[c]}", fontsize=16, color="0.15")
                num += 1
                y -= dy
            y -= gap
    # Wake/Sleep legend centred under the radar
    fig.legend(handles=[Line2D([], [], color=A_STYLE[0], ls=A_STYLE[1], lw=2.0, label="Sleep (baseline)"),
                        Line2D([], [], color=B_STYLE[0], ls=B_STYLE[1], lw=2.0, label="Wake (departure)")],
               loc="lower center", ncol=2, fontsize=17, frameon=False,
               bbox_to_anchor=(0.20, 0.004), columnspacing=2.0)
    fig.text(0.72, 0.035, f"Sleep-EDF, {n_subj} subjects.",
             fontsize=12, style="italic", color="0.45", ha="center")

    for ext in ("png", "pdf"):
        fig.savefig(REPO / f"figures/fig_concept_radar_edf.{ext}", dpi=200)
    print("saved figures/fig_concept_radar_edf.{png,pdf}")


if __name__ == "__main__":
    main()
