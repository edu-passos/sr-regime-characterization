"""Across-subject inference for the Sleep-EDF v2 run (EEG analog of across_seed_inference.py).

Each subject is one independent replicate (the EEG analog of a synthetic seed). For each
concept we take the within-subject Cliff's delta (Sleep vs Wake, oriented Sleep - Wake from
each subject's pairwise_stats.csv) and test the per-subject deltas against zero with a
two-sided Wilcoxon signed-rank test (Demsar 2006), Benjamini-Hochberg across concepts.

    detected == (q < 0.05 AND |median delta| >= 0.147)

i.e. significant AND at least a small effect -- the identical standard used for the synthetic
experiments. Effect bands (Romano et al. 2006): |d|<0.147 negligible, <0.33 small,
<0.474 medium, else large.

Runs on whatever subjects have completed (pairwise_stats.csv + window_summaries.csv present),
so it can be used as a preliminary readout mid-run and re-run unchanged when all land.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from symbolic_structural_profiling.naming import display_name              # noqa: E402
from symbolic_structural_profiling.statistics import benjamini_hochberg    # noqa: E402

from _paths import SLEEP_EDF                                               # noqa: E402
ROOT = SLEEP_EDF
ALPHA = 0.05
MIN_EFFECT = 0.147


def _magnitude(d: float) -> str:
    d = abs(d)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def completed_subjects() -> list[Path]:
    out = []
    dirs = [d for d in ROOT.glob("subject_*") if d.is_dir()]
    for d in sorted(dirs, key=lambda p: int(p.name.split("_")[1])):
        if (d / "pairwise_stats.csv").exists() and (d / "window_summaries.csv").exists():
            out.append(d)
    return out


def main() -> None:
    subjects = completed_subjects()
    if not subjects:
        print(f"No completed subjects under {ROOT}")
        return

    # feature -> {subject_id: (cliffs_delta, median_diff)}
    deltas: dict[str, dict[int, float]] = {}
    real: dict[str, dict[int, float]] = {}
    for d in subjects:
        sid = int(d.name.split("_")[1])
        df = pd.read_csv(d / "pairwise_stats.csv")
        # orientation guard: every row should be Sleep (A) vs Wake (B)
        assert (df["group_a"] == "state_sleep").all() and (df["group_b"] == "state_wake").all()
        for _, row in df.iterrows():
            f = row["feature"]
            deltas.setdefault(f, {})[sid] = float(row["cliffs_delta"])
            real.setdefault(f, {})[sid] = float(row["median_diff"])

    sids = [int(d.name.split("_")[1]) for d in subjects]
    n = len(sids)

    rows, pvals = [], []
    for f, per in deltas.items():
        d = np.array([per[s] for s in sids], dtype=float)
        nz = d[d != 0]
        p = float(wilcoxon(nz, alternative="two-sided").pvalue) if nz.size else 1.0
        rows.append({
            "feature": f,
            "median_delta": float(np.median(d)),
            "median_diff_real": float(np.median([real[f][s] for s in sids])),
            "n_same_sign": int(max(np.sum(d > 0), np.sum(d < 0))),
            "p_wilcoxon": p,
        })
        pvals.append(p)

    reject, q = benjamini_hochberg(np.asarray(pvals), ALPHA)
    for r, qv in zip(rows, q):
        r["q_value"] = float(qv)
        r["detected"] = bool(qv < ALPHA and abs(r["median_delta"]) >= MIN_EFFECT)
        r["effect"] = _magnitude(r["median_delta"])

    rows.sort(key=lambda r: abs(r["median_delta"]), reverse=True)
    out = pd.DataFrame(rows)
    dest = ROOT / "across_subject_results.csv"
    out.to_csv(dest, index=False)

    print(f"Sleep-EDF across-subject inference -- {n} completed subjects "
          f"(ids: {sorted(sids)})")
    print("Per-subject Cliff's delta (Sleep - Wake), two-sided Wilcoxon signed-rank, "
          "BH across concepts.")
    print(f"detected = q<{ALPHA} AND |median delta|>={MIN_EFFECT}.  "
          f"Wilcoxon floor at n={n}: p>={2/2**n:.4f} (so q can clear 0.05 only with strong sign agreement).\n")
    print(f"{'concept':<42s} {'med d':>7s} {'real':>8s} {'sign':>5s} {'q':>7s}  {'effect':<10s} det")
    for r in rows:
        print(f"{display_name(r['feature']):<42s} {r['median_delta']:>7.3f} "
              f"{r['median_diff_real']:>8.3f} {r['n_same_sign']:>3d}/{n} "
              f"{r['q_value']:>7.3f}  {r['effect']:<10s} {'YES' if r['detected'] else ''}")
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
