"""Validate that the descriptor's lag-order axis is a faithful read-out of the
classical temporal-memory of the (1 Hz) Sleep-EDF signal.

The v2 EEG is analysed at ~1 Hz, so the descriptor's "lag order" indexes the
temporal persistence of slow (<0.5 Hz) cortical dynamics. This script checks that
claim model-free: on the SAME homogeneous 160 s windows the descriptor used
(outputs/sleep_edf/subject_*/window_summaries.csv), it recomputes two
classical memory proxies straight from the raw window samples --

  * AR model order chosen by AIC (Levinson-Durbin, orders 0..max_lag), the direct
    autoregressive analogue of the descriptor's lag order;
  * the spectral exponent (slope of log power vs log frequency over 0.02-0.45 Hz),
    i.e. the aperiodic 1/f slope -- steeper => more low-frequency => longer memory.

It then correlates these against the descriptor's lag centroid per window and per
subject x regime (Spearman rho about 0.88 for AR order and 0.87 for the spectral
slope), and checks which subjects reverse Wake vs Sleep using Cliff's delta, the
same statistic the descriptor uses (median differences are too coarse for the
integer AR order). The spectral slope reverses on the same five subjects as the
descriptor's lag concepts, and the AR order reverses on five subjects as well, so
the inter-subject heterogeneity is a genuine signal property rather than an
artefact of the symbolic-regression pipeline.

Outputs:
  figures/fig_edf_memory_validation.{png,pdf}
  figures/edf_modelfree_memory_windows.csv     (per-window proxies)
  figures/edf_modelfree_departure_by_subject.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from _paths import SLEEP_EDF, FIGURES                                       # noqa: E402
from symbolic_structural_profiling.statistics import cliffs_delta           # noqa: E402
SRC = SLEEP_EDF
FIGS = FIGURES
WAKE, SLEEP = "state_wake", "state_sleep"

W = 160          # window = step (non-overlapping), matches run_sleep_edf.sh
MAX_LAG = 6      # matches the ensemble max_lag in the v2 run


def ar_order_aic(x: np.ndarray, pmax: int = MAX_LAG) -> int:
    """AR order minimising AIC, via Levinson-Durbin recursion on the biased acvf."""
    x = x - x.mean()
    n = len(x)
    r = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(pmax + 1)])
    if r[0] <= 0:
        return 0
    a = np.zeros(pmax + 1)
    err = r[0]
    aic = [n * np.log(err) + 2.0]
    for p in range(1, pmax + 1):
        acc = r[p] - sum(a[j] * r[p - j] for j in range(1, p))
        k = acc / err if err > 0 else 0.0
        new = a.copy()
        new[p] = k
        for j in range(1, p):
            new[j] = a[j] - k * a[p - j]
        a = new
        err = max(err * (1.0 - k * k), 1e-12)
        aic.append(n * np.log(err) + 2.0 * (p + 1))
    return int(np.argmin(aic))


def spectral_exponent(x: np.ndarray) -> float:
    """Slope of log10 PSD vs log10 f over 0.02-0.45 Hz (fs = 1 Hz); the 1/f slope."""
    f, P = periodogram(x - x.mean(), fs=1.0)
    m = (f >= 0.02) & (f <= 0.45) & (P > 0)
    if m.sum() < 4:
        return np.nan
    return float(np.polyfit(np.log10(f[m]), np.log10(P[m]), 1)[0])


def per_window_table() -> pd.DataFrame:
    rows = []
    for rid in range(20):
        sub = SRC / f"subject_{rid}"
        ws_path = sub / "window_summaries.csv"
        csv = SRC / "subjects" / f"subject_{rid}.csv"
        if not (ws_path.exists() and csv.exists()):
            continue
        df = pd.read_csv(csv)
        vals, labs = df["V1"].to_numpy(float), df["regime"].to_numpy()
        ws = pd.read_csv(ws_path)
        desc_lag = ws["lag_mass_median"].to_numpy(float)
        wi = 0
        for s in range(0, len(vals) - W + 1, W):
            seg = labs[s : s + W]
            if len(set(seg)) != 1:          # keep homogeneous windows (as the run did)
                continue
            x = vals[s : s + W]
            rows.append({
                "rid": rid, "label": seg[0],
                "ar_order": ar_order_aic(x),
                "spec_exp": spectral_exponent(x),
                "desc_lag": desc_lag[wi],
            })
            wi += 1
    return pd.DataFrame(rows)


def main() -> None:
    D = per_window_table()
    print(f"windows: {len(D)} "
          f"(wake {(D.label == WAKE).sum()}, sleep {(D.label == SLEEP).sum()})")

    for mf in ("ar_order", "spec_exp"):
        rho, p = spearmanr(D["desc_lag"], D[mf])
        print(f"  window-level   desc_lag vs {mf:9s}: rho={rho:+.3f} p={p:.1e}")

    g = D.groupby(["rid", "label"]).median(numeric_only=True).reset_index()
    for mf in ("ar_order", "spec_exp"):
        rho, p = spearmanr(g["desc_lag"], g[mf])
        print(f"  subj x regime  desc_lag vs {mf:9s}: rho={rho:+.3f} p={p:.1e}")

    # Per-subject Wake-vs-Sleep effect with Cliff's delta, the same statistic the
    # descriptor uses. A subject reverses when its sign opposes the cohort (Sleep
    # exceeds Wake on lag order, i.e. delta < 0). Median differences are too coarse
    # for the integer AR order, where ties hide reversals.
    def by_subject(metric: str) -> pd.Series:
        out = {}
        for rid, d in D.groupby("rid"):
            w = d.loc[d.label == WAKE, metric].to_numpy(float)
            s = d.loc[d.label == SLEEP, metric].to_numpy(float)
            if w.size and s.size:
                out[rid] = cliffs_delta(w, s)
        return pd.Series(out)

    dep = pd.DataFrame({"desc": by_subject("desc_lag"),
                        "ar": by_subject("ar_order"),
                        "spec": by_subject("spec_exp")})
    for mf in ("ar", "spec"):
        rho, p = spearmanr(dep["desc"], dep[mf])
        print(f"  per-subject    desc vs {mf:4s}: rho={rho:+.3f} p={p:.3f}")
    for col in ("desc", "ar", "spec"):
        print(f"  {col:4s} reversers:", sorted(int(i) for i in dep.index[dep[col] < 0]))

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.3))
    cw = g.label == WAKE
    ax[0].scatter(g.ar_order[cw], g.desc_lag[cw], s=34, c="#c0392b",
                  label="Wake", alpha=.8, edgecolor="none")
    ax[0].scatter(g.ar_order[~cw], g.desc_lag[~cw], s=34, c="0.35",
                  label="Sleep", alpha=.8, edgecolor="none")
    rho, _ = spearmanr(g.desc_lag, g.ar_order)
    ax[0].set(xlabel="Classical AR model order (AIC)",
              ylabel="Descriptor lag order (lag centroid)")
    ax[0].set_title(f"Per subject × regime\nSpearman ρ = {rho:+.2f}", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")

    isf = (dep.desc < 0).to_numpy()
    ax[1].axhline(0, color="0.8", lw=.8)
    ax[1].axvline(0, color="0.8", lw=.8)
    ax[1].scatter(dep.spec[~isf], dep.desc[~isf], s=38, c="0.35",
                  label=f"aligned ({(~isf).sum()})", edgecolor="none")
    ax[1].scatter(dep.spec[isf], dep.desc[isf], s=46, c="#c0392b",
                  label=f"reversed ({isf.sum()})", edgecolor="none")
    for i in dep.index[isf]:
        ax[1].annotate(str(i), (dep.spec[i], dep.desc[i]), fontsize=7,
                       ha="left", va="bottom", color="#c0392b")
    rho2, _ = spearmanr(dep.desc, dep.spec)
    ax[1].set(xlabel="Spectral-slope Cliff's delta  (Wake vs Sleep)",
              ylabel="Descriptor lag Cliff's delta  (Wake vs Sleep)")
    ax[1].set_title(f"Per subject departure\nSpearman ρ = {rho2:+.2f}", fontsize=11)
    ax[1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()

    D.to_csv(FIGS / "edf_modelfree_memory_windows.csv", index=False)
    dep.round(4).to_csv(FIGS / "edf_modelfree_departure_by_subject.csv")
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fig_edf_memory_validation.{ext}", dpi=200)
    print("saved fig_edf_memory_validation.{png,pdf} and the two CSVs to figures/")


if __name__ == "__main__":
    main()
