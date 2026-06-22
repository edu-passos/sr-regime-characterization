"""The single statistical standard for the synthetic experiments (20-seed run).

ONE canonical artifact -- figures/results_master.csv -- feeding every table
and figure, with a `kind` column:

  kind=concept        the descriptor concepts (final_v2 run): real regime values in
                      native units, departure from baseline in pooled-A window-IQR
                      units, and the across-seed test.
  kind=baseline       the 7 classical baselines (DFA, Lyapunov, spectral entropy,
                      sample entropy, AR(2) coef norm, skewness, excess kurtosis) from
                      the 20-seed recompute -- raw-series features, so they are valid
                      alongside the descriptor run -- under the IDENTICAL standard.
  kind=equation_term  equation-level recovery: per window, does the medoid equation
                      contain the generative term (lag-5 variable / nonlinear
                      operator / cross-lag product)?  For binary indicators Cliff's
                      delta equals the difference in proportions, so the same
                      effect-size column applies exactly.
  kind=redundancy_pair  concept-redundancy audit: Spearman correlation between each
                      pair of the 17 concepts on the pooled windows (4 scenarios x
                      20 seeds x both regimes); `value` = rho, scenario = pooled_all.
  kind=dimensionality   PCA spectrum of the rank-standardized 17-dim descriptor on
                      the same pooled windows; `value` = variance share for pc1..pc17
                      and the effective-dimension count for participation_ratio
                      (1 / sum of squared shares).

The standard, everywhere (Level 2; seeds are independent replicates):
  per feature x scenario, test the per-seed Cliff's deltas (B vs A) against zero
  with a two-sided Wilcoxon signed-rank test (Demsar 2006, JMLR), Benjamini-Hochberg
  within (family x scenario).  detected == (q < 0.05 AND |median delta| >= 0.147), i.e.
  significant AND at least a small effect -- significance alone, over 20 lock-step seeds,
  flags negligible consistent shifts (even nulls).  Effect magnitudes use the standard
  Cliff's-delta bands (|d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large;
  Romano et al. 2006).  Median delta with 95% bootstrap CI.

Interpretable values: per regime, the within-seed mean over windows (mean, not
median: several concepts are zero-inflated), aggregated as the median across seeds.
Departure z = (B - A) / IQR of regime-A windows pooled across the four AR scenarios
(A is the same base regime in all of them; iid_noise has a different base, so its
baseline rows carry no z).

Level 1 (per seed: block permutation, per-seed FDR) remains the method's
single-recording readout and is NOT the paper's evidence layer.

Output: figures/results_master.csv (canonical; fig_concept_radar.py reads
the kind=concept rows) + per-scenario stdout tables.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from symbolic_structural_profiling.naming import display_name              # noqa: E402
from symbolic_structural_profiling.statistics import (                     # noqa: E402
    benjamini_hochberg, cliffs_delta)

from _paths import SYNTHETIC, SYNTHETIC_BASELINES                          # noqa: E402
ROOT = SYNTHETIC
BASE_DIR = SYNTHETIC_BASELINES
SCEN = [("memory_only", "Memory"), ("nonlinearity_only", "Nonlinearity"),
        ("interaction_only", "Interaction"), ("amplitude_only", "Scaling (null)")]
BASE_SCEN = SCEN + [("iid_noise", "White noise (null)")]
N_SEEDS = 20
ALPHA = 0.05
# A concept/baseline counts as "detected" only when it is BOTH significant (BH q<ALPHA)
# AND shows at least a small effect (|Cliff's delta| >= 0.147). The across-seed Wilcoxon
# over 20 lock-step (common-random-number) seeds is powerful enough to flag negligible,
# consistent shifts (it otherwise marks even the scaling null on DFA/Lyapunov); gating on
# effect size keeps "detected" meaningful, not just non-zero.
MIN_EFFECT = 0.147
BOOT = 10_000

# The paper's 16-concept descriptor, ordered by group (single source for the family
# used in BH, the radar axes, and the tables). 14 first-order concepts (three r>0.9
# redundant pairs cut: polynomial degree, description length, total gradient variance)
# PLUS the two second-order Hessian concepts (ensemble-mean, hessian_*_mean). The
# whole 16 form one BH family. Requires a run whose pairwise_stats/window_summaries
# carry hessian_*_mean (the v2 ensemble-mean pipeline) -- not the old s10 dir.
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
CONCEPTS = [c for _, _, cs in GROUPS for c in cs]
GROUP_OF = {c: g for g, _, cs in GROUPS for c in cs}
assert len(CONCEPTS) == 16

BASELINES = ["dfa", "lyap", "spectral_ent", "sample_ent", "ar2_coef_norm",
             "skewness", "excess_kurtosis"]
BASELINE_NAMES = {"dfa": "DFA exponent", "lyap": "Largest Lyapunov exponent",
                  "spectral_ent": "Spectral entropy", "sample_ent": "Sample entropy",
                  "ar2_coef_norm": "AR(2) coefficient norm",
                  "skewness": "Skewness", "excess_kurtosis": "Excess kurtosis"}

EQ_TERMS = ["medoid_lag5_term", "medoid_nonlinear_op", "medoid_cross_product"]
EQ_NAMES = {"medoid_lag5_term": "Medoid contains lag-5 term",
            "medoid_nonlinear_op": "Medoid contains nonlinear operator",
            "medoid_cross_product": "Medoid contains cross-lag product"}
_PAT_LAG5 = re.compile(r"\bx4\b")
_PAT_NL = re.compile(r"sin|cos|tanh|exp|log")
_PAT_PROD = re.compile(r"x(\d)[^x]*\*[^x]*x(\d)")


def _magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def _infer_rows(kind: str, feature_order: list[str], names: dict[str, str],
                group_of: dict[str, str], scenario: str,
                deltas: dict[str, list[float]], rng: np.random.Generator) -> list[dict]:
    """Across-seed Wilcoxon + BH within (this family x scenario); one row per feature."""
    pvals, rows = [], []
    for f in feature_order:
        d = np.asarray(deltas[f], dtype=float)
        nz = d[d != 0.0]
        p = float(wilcoxon(nz, alternative="two-sided").pvalue) if nz.size else 1.0
        pvals.append(p)
        boots = np.median(rng.choice(d, size=(BOOT, d.size), replace=True), axis=1)
        rows.append({
            "kind": kind, "feature": f, "display_name": names[f],
            "group": group_of[f], "scenario": scenario, "n_seeds": int(d.size),
            "median_delta": float(np.median(d)),
            "delta_ci_low": float(np.quantile(boots, 0.025)),
            "delta_ci_high": float(np.quantile(boots, 0.975)),
            "n_seeds_same_sign": int(max(np.sum(d >= 0), np.sum(d < 0))),
            "p_wilcoxon": p,
        })
    reject, q = benjamini_hochberg(np.asarray(pvals), ALPHA)
    for r, qq, rj in zip(rows, q, reject):
        r["q_value"] = float(qq)
        r["detected"] = bool(rj) and abs(r["median_delta"]) >= MIN_EFFECT
        r["effect_magnitude"] = _magnitude(r["median_delta"])
    return rows


# ----------------------------------------------------------------- concepts
def concept_rows(rng: np.random.Generator) -> tuple[list[dict], list[int], pd.DataFrame]:
    seeds = [s for s in range(N_SEEDS)
             if all((ROOT / f"seed_{s}" / sc / "pairwise_stats.csv").exists()
                    for sc, _ in SCEN)]
    if not seeds:
        sys.exit(f"no complete seeds found under {ROOT}")

    rows: list[dict] = []
    names = {c: display_name(c).replace(" (median)", "").replace(" (mean)", "")
             for c in CONCEPTS}
    a_means, b_means, diffs = ({sc: [] for sc, _ in SCEN} for _ in range(3))
    a_windows, all_windows = [], []
    for sc, _ in SCEN:
        deltas = {c: [] for c in CONCEPTS}
        for s in seeds:
            pw = pd.read_csv(ROOT / f"seed_{s}" / sc / "pairwise_stats.csv").set_index("feature")
            for c in CONCEPTS:
                # CSV delta is A-vs-B (positive = A higher); flip to B-vs-A
                deltas[c].append(-float(pw.loc[c, "cliffs_delta"]))
            ws = pd.read_csv(ROOT / f"seed_{s}" / sc / "window_summaries.csv",
                             usecols=["label"] + CONCEPTS)
            # Winsorize continuous concepts per seed-scenario (median +/- 10*IQR) before the
            # departure mean, so a single near-singular-Jacobian window cannot detonate it.
            # Leaves zero-inflated concepts (IQR=0) and the rank-based detection untouched.
            wc = ws[CONCEPTS].copy()
            iqrw = wc.quantile(0.75) - wc.quantile(0.25)
            medw = wc.median()
            for c in CONCEPTS:
                if iqrw[c] > 1e-9:
                    wc[c] = wc[c].clip(medw[c] - 10 * iqrw[c], medw[c] + 10 * iqrw[c])
            wc["label"] = ws["label"].values
            m = wc.groupby("label")[CONCEPTS].mean()
            a_means[sc].append(m.loc["A"])
            b_means[sc].append(m.loc["B"])
            diffs[sc].append(m.loc["B"] - m.loc["A"])
            a_windows.append(ws[ws.label == "A"][CONCEPTS])
            all_windows.append(ws[CONCEPTS])
        rows.extend(_infer_rows("concept", CONCEPTS, names, GROUP_OF, sc, deltas, rng))

    # one unit per concept: regime-A window spread pooled over the four AR scenarios
    A = pd.concat(a_windows)
    iqr = A.quantile(0.75) - A.quantile(0.25)
    q1090_eq = (A.quantile(0.90) - A.quantile(0.10)) * (1.349 / 2.563)
    std_eq = A.std() * 1.349
    unit = iqr.where(iqr > 1e-9, q1090_eq.where(q1090_eq > 1e-9, std_eq))
    source = pd.Series(np.where(iqr > 1e-9, "iqr",
                                np.where(q1090_eq > 1e-9, "q10-90", "std")),
                       index=unit.index)
    vals = {}
    for sc, _ in SCEN:
        ra = pd.concat(a_means[sc], axis=1).median(axis=1)
        rb = pd.concat(b_means[sc], axis=1).median(axis=1)
        d = pd.concat(diffs[sc], axis=1).median(axis=1)
        for c in CONCEPTS:
            u = float(unit[c])
            vals[(c, sc)] = (float(ra[c]), float(rb[c]), float(d[c]), u,
                             str(source[c]), float(d[c]) / u if u > 1e-12 else 0.0)
    for r in rows:
        ra, rb, d, u, src, z = vals[(r["feature"], r["scenario"])]
        r.update(raw_a=ra, raw_b=rb, diff=d, unit=u, unit_source=src, z=z)
    return rows, seeds, pd.concat(all_windows, ignore_index=True)


# ----------------------------------------------------------------- baselines
def baseline_rows(rng: np.random.Generator) -> list[dict]:
    pw = pd.read_csv(BASE_DIR / "baseline_pairwise.csv")
    wf = pd.read_csv(BASE_DIR / "baseline_window_features.csv")
    group_of = {b: "Classical baseline" for b in BASELINES}

    rows: list[dict] = []
    for sc, _ in BASE_SCEN:
        sub = pw[pw.scenario == sc]
        deltas = {b: (-sub[sub.feature == b].cliffs_delta).tolist() for b in BASELINES}
        rows.extend(_infer_rows("baseline", BASELINES, BASELINE_NAMES,
                                group_of, sc, deltas, rng))

    # values + departure unit (pooled A over the four AR scenarios; iid_noise has a
    # different base process, so its rows carry no z)
    ar_scen = [sc for sc, _ in SCEN]
    A = wf[(wf.label == "A") & (wf.scenario.isin(ar_scen))][BASELINES]
    unit = (A.quantile(0.75) - A.quantile(0.25)).clip(lower=1e-12)
    per_seed = wf.groupby(["scenario", "label", "seed"])[BASELINES].mean()
    typical = per_seed.groupby(["scenario", "label"]).median()
    for r in rows:
        sc, b = r["scenario"], r["feature"]
        ra = float(typical.loc[(sc, "A"), b])
        rb = float(typical.loc[(sc, "B"), b])
        d_seed = (per_seed.loc[(sc, "B"), b] - per_seed.loc[(sc, "A"), b]).median()
        z = float(d_seed / unit[b]) if sc in ar_scen else float("nan")
        r.update(raw_a=ra, raw_b=rb, diff=float(d_seed), unit=float(unit[b]),
                 unit_source="iqr", z=z)
    return rows


# ------------------------------------------------------------ equation terms
def equation_rows(seeds: list[int], rng: np.random.Generator) -> list[dict]:
    group_of = {t: "Equation recovery" for t in EQ_TERMS}

    def indicators(eq: str) -> tuple[int, int, int]:
        m = _PAT_PROD.search(eq)
        return (int(bool(_PAT_LAG5.search(eq))),
                int(bool(_PAT_NL.search(eq))),
                int(bool(m and m.group(1) != m.group(2))))

    rows: list[dict] = []
    for sc, _ in SCEN:
        deltas = {t: [] for t in EQ_TERMS}
        fr_a = {t: [] for t in EQ_TERMS}
        fr_b = {t: [] for t in EQ_TERMS}
        for s in seeds:
            ws = pd.read_csv(ROOT / f"seed_{s}" / sc / "window_summaries.csv",
                             usecols=["label", "medoid_equation"])
            ind = pd.DataFrame([indicators(str(e)) for e in ws.medoid_equation],
                               columns=EQ_TERMS)
            ind["label"] = ws.label.to_numpy()
            frac = ind.groupby("label")[EQ_TERMS].mean()
            for t in EQ_TERMS:
                fa, fb = float(frac.loc["A", t]), float(frac.loc["B", t])
                fr_a[t].append(fa)
                fr_b[t].append(fb)
                # for binary indicators Cliff's delta == difference in proportions
                deltas[t].append(fb - fa)
        new = _infer_rows("equation_term", EQ_TERMS, EQ_NAMES, group_of, sc, deltas, rng)
        for r in new:
            t = r["feature"]
            r.update(raw_a=float(np.median(fr_a[t])), raw_b=float(np.median(fr_b[t])),
                     diff=float(np.median(np.asarray(fr_b[t]) - np.asarray(fr_a[t]))),
                     unit=float("nan"), unit_source="proportion", z=float("nan"))
        rows.extend(new)
    return rows



# ------------------------------------------------------ redundancy audit
def redundancy_rows(X: pd.DataFrame, n_seeds: int) -> list[dict]:
    """Pairwise Spearman correlations + PCA spectrum of the 17 concepts on the
    pooled windows (all scenarios, both regimes)."""
    names = {c: display_name(c).replace(" (median)", "") for c in CONCEPTS}
    rows: list[dict] = []
    C = X.corr(method="spearman")
    for i, a in enumerate(CONCEPTS):
        for b in CONCEPTS[i + 1:]:
            grp = GROUP_OF[a] if GROUP_OF[a] == GROUP_OF[b] else "cross-group"
            rows.append({
                "kind": "redundancy_pair", "feature": f"{a}~{b}",
                "display_name": f"{names[a]} ~ {names[b]}", "group": grp,
                "scenario": "pooled_all", "n_seeds": n_seeds,
                "value": float(C.loc[a, b]),
            })
    ranks = X.rank()
    R = (ranks - ranks.mean()) / ranks.std()
    ev = np.linalg.eigvalsh(np.cov(R.fillna(0.0).T))[::-1]
    shares = ev / ev.sum()
    for i, s in enumerate(shares, start=1):
        rows.append({
            "kind": "dimensionality", "feature": f"pc{i}",
            "display_name": f"PC{i} variance share", "group": "PCA spectrum",
            "scenario": "pooled_all", "n_seeds": n_seeds, "value": float(s),
        })
    rows.append({
        "kind": "dimensionality", "feature": "participation_ratio",
        "display_name": "Effective dimensionality (participation ratio)",
        "group": "PCA spectrum", "scenario": "pooled_all", "n_seeds": n_seeds,
        "value": float(1.0 / np.sum(shares ** 2)),
    })
    return rows


def main() -> None:
    rng = np.random.default_rng(0)
    c_rows, seeds, pooled = concept_rows(rng)   # 16-concept descriptor incl. Hessian
    b_rows = baseline_rows(rng)
    e_rows = equation_rows(seeds, rng)
    r_rows = redundancy_rows(pooled, len(seeds))

    cols = ["kind", "feature", "display_name", "group", "scenario", "n_seeds",
            "raw_a", "raw_b", "diff", "unit", "unit_source", "z",
            "median_delta", "delta_ci_low", "delta_ci_high", "n_seeds_same_sign",
            "p_wilcoxon", "q_value", "detected", "effect_magnitude", "value"]
    out = pd.DataFrame(c_rows + b_rows + e_rows + r_rows).reindex(columns=cols)
    os.makedirs(REPO / "figures", exist_ok=True)
    csv_path = REPO / "figures/results_master.csv"
    out.to_csv(csv_path, index=False, float_format="%.6g")

    # ---- stdout: per-scenario tables ----
    print(f"across-seed inference, {len(seeds)} seeds (delta = Cliff's delta, B vs A; "
          f"Wilcoxon signed-rank across seeds; BH within family x scenario)")
    for sc, label in BASE_SCEN:
        sub = out[out.scenario == sc]
        if sub.empty:
            continue
        print(f"\n=== {label} ===")
        print(f"{'feature':34s} {'A value':>10s} {'B value':>10s} {'dep(IQR)':>9s} "
              f"{'delta':>6s} {'95% CI':>16s} {'q':>8s}  effect")
        for kind in ("concept", "baseline", "equation_term"):
            k = sub[sub.kind == kind].sort_values("q_value")
            if k.empty:
                continue
            for _, r in k.iterrows():
                flag = "*" if r.detected else " "
                ci = f"[{r.delta_ci_low:+.2f},{r.delta_ci_high:+.2f}]"
                dep = f"{r.z:+9.2f}" if np.isfinite(r.z) else f"{'--':>9s}"
                print(f"{r.display_name:34s} {r.raw_a:10.3g} {r.raw_b:10.3g} {dep} "
                      f"{r.median_delta:+6.2f} {ci:>16s} {r.q_value:8.3g} {flag} "
                      f"{r.effect_magnitude}")
            print()
    inf = out[out.kind.isin(["concept", "baseline", "equation_term"])]
    det = inf.groupby(["kind", "scenario"])["detected"].sum().unstack(fill_value=0)
    print("detected counts:")
    print(det.to_string())

    pairs = out[out.kind == "redundancy_pair"].sort_values(
        "value", key=lambda s: s.abs(), ascending=False)
    print("\nconcept redundancy (top pairs, |Spearman| on pooled windows):")
    for _, r in pairs.head(8).iterrows():
        print(f"  {r['value']:+.2f}  {r.display_name}")
    pr = out[(out.kind == "dimensionality")
             & (out.feature == "participation_ratio")]["value"].iloc[0]
    pcs = out[out.kind == "dimensionality"].set_index("feature")["value"]
    print(f"PCA: PC1 {pcs['pc1']*100:.0f}%, PC2 {pcs['pc2']*100:.0f}%, "
          f"PC3 {pcs['pc3']*100:.0f}%; effective dimensionality = {pr:.1f} / 17")
    print(f"\nsaved {csv_path}")


if __name__ == "__main__":
    main()
