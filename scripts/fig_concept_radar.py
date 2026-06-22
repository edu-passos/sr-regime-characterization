"""Radar plots of the 17-dim concept descriptor per synthetic scenario (20-seed run).

Unlike the regime-separation figure (which shows |Cliff's delta| of the A-vs-B test),
this figure shows the descriptor values themselves, expressed as departure from baseline:
regime A is a fixed reference circle, and regime B is plotted per concept as

    z = median over seeds of (mean_B - mean_A over windows, paired per seed)
        -------------------------------------------------------------------
        IQR of regime-A window values, pooled across all scenarios and seeds

i.e. "how many baseline-IQRs does the perturbation push this concept off regime A".
The unit is per concept but shared across panels (A is the same base regime in every
scenario), so departures are comparable across axes and across panels, and the scaling
null reads correctly as B sitting on the A circle.

SINGLE SOURCE OF TRUTH: the values are NOT computed here.  They are read from
figures/results_master.csv, the canonical artifact produced by
scripts/across_seed_inference.py (which also carries the across-seed Wilcoxon/BH
inference).  Run that script first; this one only draws.

Background sectors are colour-coded by concept group.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from symbolic_structural_profiling.naming import display_name  # noqa: E402

CANONICAL = REPO / "figures/results_master.csv"
# Three perturbation panels only; the scaling null (B sits on the A circle) is
# reported in text/table rather than shown here.
SCEN = [("memory_only", "High lag order"), ("nonlinearity_only", "Nonlinearity"),
        ("interaction_only", "Interaction")]

# Display spec only (axis order + sector colours).  The concept family itself is
# defined once in scripts/across_seed_inference.py and arrives via the CSV.
# Four core dynamical properties. Every concept -- structural (form), behavioural
# (gradient), or second-order (Hessian) -- folds into the property it measures, so the
# radar has four colour groups rather than per-layer compound groups. Within each group
# concepts run structural -> gradient -> Hessian.
GROUPS: list[tuple[str, str, list[str]]] = [
    ("Lag order", "#a6cee3", [
        "lag_mass_median", "long_lag_mass_median",
        "lag_entropy_median", "active_lag_count_median",
        "jacobian_lag_mass_median", "jacobian_long_lag_mass_median"]),
    ("Multiplicative interaction", "#fb9a99", [
        "interaction_term_count_median", "interaction_order_median",
        "interaction_depth_median", "hessian_interaction_mean"]),
    ("Nonlinearity", "#b2df8a", [
        "nonlinear_node_count_median",
        "jacobian_centroid_norm_median", "jacobian_nonlinearity_index_median",
        "hessian_curvature_mean", "jacobian_relative_nonlinearity_index_median"]),
    ("Complexity", "#cab2d6", [
        "complexity_cost_median"]),
]
# 16-concept descriptor: 14 first-order (three r>0.9 redundant pairs cut: polynomial
# degree, description length, total gradient variance) PLUS the two second-order Hessian
# concepts (interaction curvature off-diagonal, single-lag curvature diagonal).
CONCEPTS = [c for _, _, cs in GROUPS for c in cs]
assert len(CONCEPTS) == 16, f"expected 16 concepts, got {len(CONCEPTS)}"

# Short radar-axis labels (the full plain-language names live in naming.py, the caption
# and the tables). Keeps the 16 axes legible without overlapping around the circle.
SHORT_LABELS = {
    "lag_mass_median": "Lag centroid",
    "long_lag_mass_median": "High-lag share",
    "lag_entropy_median": "Lag diversity",
    "active_lag_count_median": "Active lags",
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

# Radial axis: departure z in baseline-IQR units, linear, A circle at z = 0.
# Scale zoomed to the synthetic departures (z in [-0.99, 2.05]) so the polygons fill the plot.
Z_MIN, Z_MAX = -0.6, 2.2
Z_RINGS = [-0.5, 0, 1, 2]

A_STYLE = ("0.30", "--")       # baseline reference circle
B_STYLE = ("#c0392b", "-")     # perturbed regime polygon


def _radius(z: np.ndarray | float) -> np.ndarray | float:
    return (np.clip(z, Z_MIN, Z_MAX) - Z_MIN) / (Z_MAX - Z_MIN)


def load_departures() -> pd.DataFrame:
    if not CANONICAL.exists():
        sys.exit(f"{CANONICAL} not found -- run scripts/across_seed_inference.py first "
                 "(it produces the canonical values this figure draws)")
    cells = pd.read_csv(CANONICAL)
    # descriptor concepts are kind=concept; the two Hessian concepts may be emitted
    # as kind=hessian (instrument) or kind=concept (folded-in) depending on the run.
    cells = cells[cells.kind.isin(["concept", "hessian"])].rename(columns={"feature": "concept"})
    # normalise the Hessian name: the s10 post-hoc cache used *_median, the v2
    # ensemble-mean pipeline uses *_mean (the descriptor name).
    cells["concept"] = cells["concept"].replace({
        "hessian_interaction_median": "hessian_interaction_mean",
        "hessian_curvature_median": "hessian_curvature_mean"})
    missing = [c for c in CONCEPTS if c not in set(cells.concept)]
    if missing:
        sys.exit(f"canonical CSV is missing concepts: {missing}")
    return cells


def main() -> None:
    cells = load_departures()

    # ---- radar figure ----
    n = len(CONCEPTS)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    th_closed = np.concatenate([angles, angles[:1]])
    dtheta = 2.0 * np.pi / n
    # Number the 16 spokes (1..16 in group order); the colour-coded key below maps the
    # numbers to names. No text around the circles -> the circles stay large and there is
    # no label overlap, so the data reads big on the page.
    numbers = [str(i + 1) for i in range(n)]
    r0 = float(_radius(0.0))

    # Wide banner spanning the full column width: 3 circles in a row on top, key beneath.
    # Native fonts are deliberately LARGE because the column scales this ~9.5in figure
    # down to ~4.8in (x0.5); the big native sizes land at a readable ~6-8pt on the page.
    # The width gives the four key columns room so the long names do not overlap.
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 6.0),
                             subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.875, bottom=0.50, wspace=0.16)
    for ax, (sc, sc_label) in zip(axes, SCEN):
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_direction(-1)
        ax.set_ylim(0.0, 1.0)

        # group sectors (background wedges)
        start = 0
        for _, colour, cs in GROUPS:
            span = len(cs) * dtheta
            mid = angles[start] - dtheta / 2.0 + span / 2.0
            ax.bar(mid, 1.0, width=span, bottom=0.0, color=colour,
                   alpha=0.30, edgecolor="none", zorder=0)
            start += len(cs)

        # departure rings (labelled in baseline-IQR units)
        ring_r = [_radius(zr) for zr in Z_RINGS]
        ax.set_rgrids(ring_r, labels=[], angle=90)
        ax.grid(color="0.75", lw=0.5)
        for zr, rr in zip(Z_RINGS, ring_r):
            if zr == 0:
                continue
            ax.text(np.pi / n, rr, f"{zr:+g}", fontsize=8, color="0.45",
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none",
                              alpha=0.7))

        # regime A = baseline reference circle at z = 0
        a_colour, a_ls = A_STYLE
        ax.plot(th_closed, np.full_like(th_closed, r0), color=a_colour,
                lw=1.6, ls=a_ls, zorder=3)

        # regime B polygon + shaded departure band between B and the A circle
        sub = cells[cells.scenario == sc].set_index("concept").loc[CONCEPTS]
        z = sub["z"].to_numpy()
        rb = np.asarray(_radius(z))
        rb_closed = np.concatenate([rb, rb[:1]])
        b_colour, b_ls = B_STYLE
        ax.fill(np.concatenate([th_closed, th_closed[::-1]]),
                np.concatenate([rb_closed, np.full_like(rb_closed, r0)[::-1]]),
                color=b_colour, alpha=0.15, lw=0, zorder=2)
        ax.plot(th_closed, rb_closed, color=b_colour, lw=1.6, ls=b_ls, zorder=3)
        # flag clipped axes with the true departure value (never clip silently)
        for i, zi in enumerate(z):
            if zi > Z_MAX or zi < Z_MIN:
                ax.annotate(f"{zi:+.1f}", (angles[i], float(_radius(zi))),
                            fontsize=7, color=b_colour, fontweight="bold",
                            ha="center", va="bottom", zorder=5)
                print(f"WARNING: {sc}/{CONCEPTS[i]} departure {zi:+.2f} clipped to "
                      f"[{Z_MIN}, {Z_MAX}] (annotated on figure)")

        # spokes carry only their number; the key beneath decodes them
        ax.set_xticks(angles)
        ax.set_xticklabels(numbers, fontsize=12)
        ax.tick_params(axis="x", pad=2)
        ax.spines["polar"].set_color("0.6")
        ax.set_title(sc_label, fontsize=15, pad=10, fontweight="bold")

    # number -> name key: one colour-coded column per property group, beneath the circles
    DARK = {"Lag order": "#1f78b4", "Multiplicative interaction": "#e31a1c",
            "Nonlinearity": "#33a02c", "Complexity": "#6a3d9a"}
    HDR = {"Lag order": "Lag order", "Multiplicative interaction": "Mult. interaction",
           "Nonlinearity": "Nonlinearity", "Complexity": "Complexity"}
    col_x = [0.04, 0.29, 0.54, 0.79]
    y_top, dy = 0.42, 0.055
    num = 1
    for (gname, _, cs), x in zip(GROUPS, col_x):
        fig.text(x, y_top, HDR[gname], fontsize=14, fontweight="bold", color=DARK[gname])
        for j, c in enumerate(cs):
            name = SHORT_LABELS.get(c, display_name(c))
            fig.text(x, y_top - (j + 1) * dy, f"{num}.  {name}", fontsize=13, color="0.15")
            num += 1

    # regime A/B legend on one row at the very bottom, clear of the key columns
    fig.legend(handles=[Line2D([], [], color=A_STYLE[0], ls=A_STYLE[1], lw=1.8,
                               label="Regime A (baseline)"),
                        Line2D([], [], color=B_STYLE[0], ls=B_STYLE[1], lw=1.8,
                               label="Regime B (departure)")],
               loc="lower center", ncol=2, fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, 0.005), columnspacing=2.5)
    # NOTE: no tight_layout -- it would clobber the manual subplots_adjust / fig.text key.
    for ext in ("png", "pdf"):
        fig.savefig(REPO / f"figures/fig_concept_radar.{ext}", dpi=200)
    print(f"saved figures/fig_concept_radar.{{png,pdf}} "
          f"(values from {CANONICAL.name})")


if __name__ == "__main__":
    main()
