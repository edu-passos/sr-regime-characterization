"""Skewness / kurtosis baselines for the Sleep-EDF run -- same standard as the concepts.

Trivial per-window signal-moment descriptors (skewness, excess kurtosis) computed on the
SAME W=160 non-overlapping homogeneous windows as the concept run, then aggregated
ACROSS SUBJECTS exactly like scripts/across_subject_inference_edf.py: per-subject Cliff's delta
(Sleep - Wake), two-sided Wilcoxon signed-rank across subjects, BH, gate |median d| >= 0.147.

Purpose: a leakage / sanity baseline -- does a 2-moment descriptor separate Wake vs Sleep as
strongly as the 16-concept vector? Auto-detects completed subjects, so it tracks the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, wilcoxon

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from symbolic_structural_profiling.statistics import (             # noqa: E402
    benjamini_hochberg, cliffs_delta)
from symbolic_structural_profiling.windowing import segment_labeled_series  # noqa: E402

from _paths import SLEEP_EDF                                       # noqa: E402
ROOT = SLEEP_EDF
SUBJ = ROOT / "subjects"
WAKE, SLEEP = "state_wake", "state_sleep"
W = 160                       # same window as the v2 concept run (non-overlapping)
ALPHA, MIN_EFFECT = 0.05, 0.147
# scipy: skew/kurtosis are bias-corrected; kurtosis is Fisher (excess; normal -> 0).
FEATURES = ("skewness", "excess_kurtosis")


def _magnitude(d: float) -> str:
    d = abs(d)
    return ("negligible" if d < 0.147 else "small" if d < 0.33
            else "medium" if d < 0.474 else "large")


def completed_subjects() -> list[tuple[int, Path]]:
    out = []
    dirs = [p for p in ROOT.glob("subject_*") if p.is_dir()]
    for d in sorted(dirs, key=lambda p: int(p.name.split("_")[1])):
        sid = int(d.name.split("_")[1])
        csv = SUBJ / f"subject_{sid}.csv"
        if (d / "window_summaries.csv").exists() and csv.exists():
            out.append((sid, csv))
    return out


def window_moments(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    windows = segment_labeled_series(
        values=df[["V1"]].to_numpy(float),
        labels=df["regime"].astype(str).to_numpy(),
        window_size=W, step_size=W,
        require_homogeneous_labels=True,
        record_ids=df["record_id"].astype(str).to_numpy(),
    )
    rows = []
    for w in windows:
        y = w.values[:, 0]
        rows.append({"label": str(w.label),
                     "skewness": float(skew(y)),
                     "excess_kurtosis": float(kurtosis(y))})
    return pd.DataFrame(rows)


def main() -> None:
    subs = completed_subjects()
    if not subs:
        print(f"No completed subjects under {ROOT}")
        return

    per = {f: [] for f in FEATURES}     # per-subject Cliff's delta (Sleep - Wake)
    real = {f: [] for f in FEATURES}    # per-subject real median(Sleep) - median(Wake)
    sids = []
    for sid, csv in subs:
        wm = window_moments(csv)
        a, b = wm[wm.label == SLEEP], wm[wm.label == WAKE]
        if a.empty or b.empty:
            continue
        sids.append(sid)
        for f in FEATURES:
            s, w_ = a[f].to_numpy(), b[f].to_numpy()
            per[f].append(cliffs_delta(s, w_))
            real[f].append(float(np.median(s) - np.median(w_)))

    n = len(sids)
    rows, pvals = [], []
    for f in FEATURES:
        d = np.asarray(per[f], float)
        nz = d[d != 0]
        p = float(wilcoxon(nz, alternative="two-sided").pvalue) if nz.size else 1.0
        rows.append({"feature": f, "median_delta": float(np.median(d)),
                     "median_diff_real": float(np.median(real[f])),
                     "n_same_sign": int(max((d > 0).sum(), (d < 0).sum())), "p_wilcoxon": p})
        pvals.append(p)

    _, q = benjamini_hochberg(np.asarray(pvals), ALPHA)
    for r, qv in zip(rows, q):
        r["q_value"] = float(qv)
        r["detected"] = bool(qv < ALPHA and abs(r["median_delta"]) >= MIN_EFFECT)
        r["effect"] = _magnitude(r["median_delta"])

    dest = ROOT / "across_subject_baselines.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)

    print(f"Sleep-EDF skew/kurtosis baselines -- {n} subjects {sorted(sids)}")
    print("Per-subject Cliff's delta (Sleep - Wake), Wilcoxon across subjects, BH.")
    print(f"detected = q<{ALPHA} AND |median delta|>={MIN_EFFECT}\n")
    print(f"{'baseline':<18s} {'med d':>7s} {'real':>9s} {'sign':>5s} {'q':>7s}  {'effect':<10s} det")
    for r in sorted(rows, key=lambda r: abs(r["median_delta"]), reverse=True):
        print(f"{r['feature']:<18s} {r['median_delta']:>7.3f} {r['median_diff_real']:>9.3f} "
              f"{r['n_same_sign']:>3d}/{n} {r['q_value']:>7.3f}  {r['effect']:<10s} "
              f"{'YES' if r['detected'] else ''}")
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
