"""Build the paper's LaTeX tables.

  specificity  tab:specificity, the synthetic per-scenario Cliff's delta table.
               Reads figures/results_master.csv (from across_seed_inference.py
               over outputs/synthetic). Each cell is the across-seed median Cliff's
               delta; a dash marks no detection (BH q<0.05 and |delta|>=0.147). Bold
               marks the concept built for the perturbed property.

  sleep        tab:sleep-edf-baselines, the Sleep-EDF within-subject table. Reads
               outputs/sleep_edf. Median across subjects of the per-subject Wake-vs-Sleep
               Cliff's delta and the number of subjects separated, where a subject counts
               when the block-permutation p < 0.05 and |Cliff's delta| >= 0.147. Concepts
               come from each subject's pairwise_stats.csv; baselines run through the same
               block-permutation test on the per-window baseline features.

Run: python3 scripts/build_tables.py [specificity|sleep|all]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from _paths import SLEEP_EDF, FIGURES                                          # noqa: E402
from symbolic_structural_profiling.config import StatsConfig                   # noqa: E402
from symbolic_structural_profiling.statistics import run_pairwise_feature_tests  # noqa: E402


# ============================================================ specificity (synthetic)
RM = FIGURES / "results_master.csv"
SPEC_ROWS = [("High lag order", "memory_only"), ("Nonlinearity", "nonlinearity_only"),
             ("Interaction", "interaction_only"), ("Scaling", "amplitude_only"),
             ("White noise", "iid_noise")]
SPEC_BASE = [("DFA", "dfa"), ("AR(2)", "ar2_coef_norm"),
             ("Skewness", "skewness"), ("Kurtosis", "excess_kurtosis")]
SPEC_CONC = [("Lag order", "jacobian_long_lag_mass_median"),
             ("Nonlinear operator count", "nonlinear_node_count_median"),
             ("Gradient variation", "jacobian_nonlinearity_index_median"),
             ("Interaction curvature", "hessian_interaction_mean")]
SPEC_BOLD = {("memory_only", "Lag order"),
             ("nonlinearity_only", "Nonlinear operator count"),
             ("nonlinearity_only", "Gradient variation"),
             ("interaction_only", "Interaction curvature")}


def specificity_table() -> str:
    rm = pd.read_csv(RM)
    n = int(rm[rm.kind == "concept"].n_seeds.dropna().max())

    def cell(kind, feat, scenario, bold=False):
        r = rm[(rm.kind == kind) & (rm.feature == feat) & (rm.scenario == scenario)]
        if r.empty or not bool(r.iloc[0].detected):
            return "---"
        d = float(r.iloc[0].median_delta)
        return rf"$\mathbf{{{d:.2f}}}$" if bold else f"${d:.2f}$"

    lines = [r"\begin{table}[t]", r"\centering", r"\renewcommand{\arraystretch}{1.2}",
             r"\caption{Cliff's $\delta$ per scenario for the standard baselines and "
             r"a representative concept of each kind. Positive when the perturbed regime "
             r"exceeds the baseline regime. A dash marks no detection (Benjamini-Hochberg "
             rf"$q<0.05$ and $|\delta|\ge0.147$ across ${n}$ seeds). \textbf{{Bold}} marks "
             r"the concept built for the perturbed property.}",
             r"\label{tab:specificity}",
             r"\begin{tabular}{@{}l cccc @{\hspace{1.2em}} cccc@{}}", r"\toprule",
             r"& \multicolumn{4}{c}{Standard baselines} & \multicolumn{4}{c}{Descriptor concepts} \\",
             r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
             r"Scenario & DFA & AR(2) & Skewness & Kurtosis & "
             r"\makecell{Lag\\order} & \makecell{Nonlinear\\operator count} & "
             r"\makecell{Gradient\\variation} & \makecell{Interaction\\curvature} \\",
             r"\midrule"]
    for label, sc in SPEC_ROWS:
        cells = [cell("baseline", f, sc) for _, f in SPEC_BASE]
        cells += [cell("concept", f, sc, bold=(sc, h) in SPEC_BOLD) for h, f in SPEC_CONC]
        lines.append(f"{label:<12} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ============================================================ sleep (Sleep-EDF)
# Block-permutation settings of the sleep run (matches the committed pairwise_stats).
CFG = StatsConfig(block_size=3, n_permutations=1500, min_effect_iqr_ratio=0.2,
                  alpha=0.05, balance_labels_for_pairwise=False)
# A subject counts as separated under the paper's per-case criterion: the
# block-permutation p is significant and the effect is at least small on the
# Romano scale (block_permutation p < 0.05 and |Cliff's delta| >= 0.147).
SIG_ALPHA, MIN_DELTA = 0.05, 0.147


def _separated(p_value: float, cliffs_delta: float) -> bool:
    return float(p_value) < SIG_ALPHA and abs(float(cliffs_delta)) >= MIN_DELTA
SLEEP_CONC = [("Gradient sensitivity", "jacobian_centroid_norm_median"),
              ("Gradient magnitude", "jacobian_lag_mass_median"),
              ("Lag diversity", "lag_entropy_median"),
              ("Lag centroid", "lag_mass_median"),
              ("Active lag count", "active_lag_count_median")]
SLEEP_BASE = [("DFA exponent", "dfa"),
              ("AR(2) coefficient norm", "ar2_coef_norm"),
              ("Sample entropy", "sample_ent"),
              ("Excess kurtosis", "excess_kurtosis"),
              ("Spectral entropy", "spectral_ent"),
              ("Largest Lyapunov", "lyap"),
              ("Skewness", "skewness")]


def _sleep_subjects() -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in SLEEP_EDF.glob("subject_*") if p.is_dir())


def _sleep_concept_rows(subs: list[int]) -> list[tuple[str, float, int]]:
    rows = []
    for disp, feat in SLEEP_CONC:
        deltas, sep = [], 0
        for s in subs:
            df = pd.read_csv(SLEEP_EDF / f"subject_{s}/pairwise_stats.csv")
            r = df[df.feature == feat]
            if r.empty:
                continue
            deltas.append(abs(float(r.cliffs_delta.iloc[0])))
            sep += int(_separated(r.p_value.iloc[0], r.cliffs_delta.iloc[0]))
        rows.append((disp, float(np.median(deltas)), sep))
    return rows


def _sleep_baseline_rows(subs: list[int]) -> list[tuple[str, float, int]]:
    cols = [c for _, c in SLEEP_BASE]
    agg = {c: {"d": [], "sep": 0} for c in cols}
    for s in subs:
        feat = pd.read_csv(SLEEP_EDF / f"subject_{s}/baseline_window_features_v2.csv")
        summ = pd.read_csv(SLEEP_EDF / f"subject_{s}/baseline_window_summaries.csv")
        df = feat.merge(summ[["record_id", "start", "end", "label"]],
                        on=["record_id", "start", "end"], how="inner")
        res = run_pairwise_feature_tests(df, cols, CFG).set_index("feature")
        for c in cols:
            if c in res.index:
                agg[c]["d"].append(abs(float(res.loc[c, "cliffs_delta"])))
                agg[c]["sep"] += int(_separated(res.loc[c, "p_value"], res.loc[c, "cliffs_delta"]))
    return [(disp, float(np.median(agg[c]["d"])), agg[c]["sep"]) for disp, c in SLEEP_BASE]


def sleep_table() -> str:
    subs = _sleep_subjects()
    n = len(subs)
    concepts, baselines = _sleep_concept_rows(subs), _sleep_baseline_rows(subs)
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Within-subject Wake versus Sleep separation on Sleep-EDF, "
           rf"${n}$ subjects. Median $|\delta|$ across subjects and the number of "
           r"subjects with a significant within-subject separation.}",
           r"\label{tab:sleep-edf-baselines}", r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
           r"& Median $|\delta|$ & Subjects separated \\", r"\midrule",
           r"\multicolumn{3}{@{}l}{\textit{Descriptor concepts}} \\"]
    for disp, d, sep in concepts:
        out.append(rf"{disp} & ${d:.2f}$ & ${sep}/{n}$ \\")
    out += [r"\midrule", r"\multicolumn{3}{@{}l}{\textit{Standard baselines}} \\"]
    for disp, d, sep in baselines:
        out.append(rf"{disp} & ${d:.2f}$ & ${sep}/{n}$ \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


TABLES = {"specificity": specificity_table, "sleep": sleep_table}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = list(TABLES) if which == "all" else [which]
    for k in keys:
        print(f"% ===== {k} =====")
        print(TABLES[k]())
