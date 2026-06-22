"""Classical time-series baselines for the synthetic suite.

Matches the descriptor run (outputs/synthetic):
non-overlapping windows W=step=100, 20 seeds, amplitude_scale_b=2.0, n_steps=6000,
and the same stats config (block_size=3, n_permutations=1200) so the baseline vs
descriptor comparison is apples-to-apples. Windows are regenerated deterministically
per seed (the synthetic generator is seeded), so this needs no PySR and no stored
window signals. Includes the two signal moments (skewness, excess kurtosis) so the
moment baselines appear in the same table as the dynamical ones.

Writes to outputs/synthetic/baselines/
  baseline_window_features.csv
  baseline_pairwise.csv
  baseline_summary.csv   (per scenario x baseline: median delta, median q)

These two files feed scripts/across_seed_inference.py (BASE_DIR), which re-tests them
under the SAME across-seed Wilcoxon + BH pipeline used for the concepts.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from _paths import SYNTHETIC_BASELINES                                             # noqa: E402
from symbolic_structural_profiling.config import StatsConfig                       # noqa: E402
from symbolic_structural_profiling.statistics import run_pairwise_feature_tests    # noqa: E402
from symbolic_structural_profiling.synthetic import generate_synthetic_regime_series  # noqa: E402
from symbolic_structural_profiling.windowing import segment_labeled_series         # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUT = SYNTHETIC_BASELINES
SCENARIOS = ["memory_only", "nonlinearity_only", "interaction_only", "amplitude_only", "iid_noise"]
SEEDS = list(range(20))
WINDOW_SIZE, STEP_SIZE = 100, 100
N_STEPS = 6000
AMPLITUDE_SCALE_B = 2.0
FEATS = ["dfa", "lyap", "spectral_ent", "sample_ent", "ar2_coef_norm",
         "skewness", "excess_kurtosis"]


def safe(fn, default=np.nan, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float(default)


def compute_baselines(y: np.ndarray) -> dict[str, float]:
    import nolds
    import antropy as ant
    from scipy.stats import kurtosis, skew
    out: dict[str, float] = {}
    out["skewness"] = safe(skew, np.nan, y)
    out["excess_kurtosis"] = safe(kurtosis, np.nan, y)  # Fisher: normal -> 0
    out["dfa"] = safe(nolds.dfa, np.nan, y)
    out["lyap"] = safe(nolds.lyap_r, np.nan, y, emb_dim=4, lag=1, min_tsep=2, trajectory_len=10)
    out["spectral_ent"] = safe(ant.spectral_entropy, np.nan, y, sf=1.0, method="welch", normalize=True)
    std = float(np.std(y))
    out["sample_ent"] = np.nan if std < 1e-12 else safe(ant.sample_entropy, np.nan, y, order=2, metric="chebyshev")
    if len(y) < 4:
        out["ar2_coef_norm"] = np.nan
    else:
        Y = y[2:]
        X = np.column_stack([y[1:-1], y[:-2]])
        try:
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            out["ar2_coef_norm"] = float(np.linalg.norm(coef))
        except Exception:
            out["ar2_coef_norm"] = np.nan
    return out


def window_features(seed: int, scenario: str) -> pd.DataFrame:
    ds = generate_synthetic_regime_series(n_steps=N_STEPS, scenario=scenario, seed=seed,
                                          amplitude_scale_b=AMPLITUDE_SCALE_B)
    windows = segment_labeled_series(values=ds.values, labels=ds.labels,
                                     window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                     require_homogeneous_labels=True)
    rows = []
    for w in windows:
        feats = compute_baselines(w.values[:, 0])
        feats.update(label=str(w.label), seed=seed, scenario=scenario)
        rows.append(feats)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = StatsConfig(block_size=3, n_permutations=1200)
    print(f"Computing baselines over {len(SEEDS)} seeds x {len(SCENARIOS)} scenarios "
          f"(W={WINDOW_SIZE}, step={STEP_SIZE}, amp_b={AMPLITUDE_SCALE_B}, "
          f"block_size={cfg.block_size}, n_perm={cfg.n_permutations})")
    feat_frames, stat_frames = [], []
    for seed in SEEDS:
        for scen in SCENARIOS:
            df = window_features(seed, scen)
            feat_frames.append(df)
            stats = run_pairwise_feature_tests(df, FEATS, cfg)
            stats["seed"], stats["scenario"] = seed, scen
            stat_frames.append(stats)
        print(f"  seed {seed} done")
    full = pd.concat(feat_frames, ignore_index=True)
    full.to_csv(OUT / "baseline_window_features.csv", index=False)
    combined = pd.concat(stat_frames, ignore_index=True)
    combined.to_csv(OUT / "baseline_pairwise.csv", index=False)

    rows = []
    for scen in SCENARIOS:
        for feat in FEATS:
            sub = combined[(combined.scenario == scen) & (combined.feature == feat)]
            if sub.empty:
                continue
            rows.append({
                "scenario": scen, "feature": feat,
                "median_delta": float(sub.cliffs_delta.median()),
                "median_q": float(sub.q_value.median()),
                "n_seeds_sig": int((sub.q_value < 0.05).sum()),
                "n_seeds": int(len(sub)),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "baseline_summary.csv", index=False)
    print(f"\n=== baseline summary (median delta, median q, n_sig/{len(SEEDS)}) ===")
    for scen in SCENARIOS:
        print(f"\n{scen}")
        for feat in FEATS:
            r = summary[(summary.scenario == scen) & (summary.feature == feat)]
            if r.empty:
                continue
            r = r.iloc[0]
            sig = "SIG" if r.median_q < 0.05 else "   "
            print(f"  {feat:14s} median|d|={abs(r.median_delta):.2f} med_q={r.median_q:.3f} {sig} "
                  f"({int(r.n_seeds_sig)}/{len(SEEDS)})")
    print(f"\nWrote {OUT}/baseline_summary.csv")


if __name__ == "__main__":
    main()
