"""Leave-one-subject-out (LOSO) AUC for the Sleep-EDF run.

A leave-one-subject-out logistic regression (median-impute + standardise on the
training subjects, C=1.0, ROC-AUC on the held-out subject), mean +/- sd across
folds (= completed subjects).

Feature sets are structural_12 / behavioural_5 / calibrated_1 / dual_layer_18, the
ensemble-mean Hessian pair (hessian_2 -> dual_layer_20), and a 2-moment skew/kurtosis
baseline computed on the SAME windows for comparison.

Per-window concept values come from each subject's window_summaries.csv; skew/kurtosis
are recomputed from the raw per-subject CSV and merged on (record_id, start, end).
Auto-detects completed subjects, so it tracks the run.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from symbolic_structural_profiling.windowing import segment_labeled_series  # noqa: E402

from _paths import SLEEP_EDF                                                # noqa: E402
warnings.filterwarnings("ignore")
ROOT = SLEEP_EDF
SUBJ = ROOT / "subjects"
WAKE, SLEEP = "state_wake", "state_sleep"
W = 160

# The canonical v2 descriptor is 16 concepts (scripts/across_seed_inference.py CONCEPTS),
# split by layer into 9 structural + 6 behavioural (Jacobian+Hessian) + 1 gating -- exactly
# the concept vector in Fig. 1. The three r>0.9 redundant concepts (polynomial degree,
# description length, total gradient variance) and nonlinear_composition_depth are NOT in it.
STRUCTURAL_9 = [
    "lag_mass_median", "long_lag_mass_median", "lag_entropy_median", "active_lag_count_median",
    "interaction_term_count_median", "interaction_order_median", "interaction_depth_median",
    "nonlinear_node_count_median", "complexity_cost_median",
]
BEHAVIOURAL_6 = [
    "jacobian_lag_mass_median", "jacobian_long_lag_mass_median", "jacobian_centroid_norm_median",
    "jacobian_nonlinearity_index_median", "hessian_interaction_mean", "hessian_curvature_mean",
]
GATING_1 = ["jacobian_relative_nonlinearity_index_median"]
DUAL_LAYER_16 = STRUCTURAL_9 + BEHAVIOURAL_6 + GATING_1
assert len(DUAL_LAYER_16) == 16

# Classical descriptor baselines (same five as the synthetic benchmark / old EDF probe)
# plus the two signal moments, all computed per window on the raw trace.
CLASSICAL_5 = ["dfa", "lyap", "spectral_ent", "sample_ent", "ar2_coef_norm"]
MOMENT_2 = ["skewness", "excess_kurtosis"]
BASELINE_COLS = MOMENT_2 + CLASSICAL_5

FEATURE_SETS = {
    "structural_9": STRUCTURAL_9,
    "behavioural_6": BEHAVIOURAL_6,
    "gating_1": GATING_1,
    "dual_layer_16": DUAL_LAYER_16,
    "dfa": ["dfa"],
    "lyap": ["lyap"],
    "spectral_ent": ["spectral_ent"],
    "sample_ent": ["sample_ent"],
    "ar2_coef_norm": ["ar2_coef_norm"],
    "skew+kurtosis": MOMENT_2,
    "all_baselines (5 classical)": CLASSICAL_5,
    "all_baselines+moments (7)": BASELINE_COLS,
    "dual_layer_16 + all_baselines": DUAL_LAYER_16 + CLASSICAL_5,
}


def completed_subjects() -> list[int]:
    out = []
    for d in [p for p in ROOT.glob("subject_*") if p.is_dir()]:
        sid = int(d.name.split("_")[1])
        if (d / "window_summaries.csv").exists() and (SUBJ / f"subject_{sid}.csv").exists():
            out.append(sid)
    return sorted(out)


def _safe(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def _classical(y: np.ndarray) -> dict[str, float]:
    import antropy as ant
    import nolds
    out = {
        "dfa": _safe(nolds.dfa, y),
        "lyap": _safe(nolds.lyap_r, y, emb_dim=4, lag=1, min_tsep=2, trajectory_len=10),
        "spectral_ent": _safe(ant.spectral_entropy, y, sf=1.0, method="welch", normalize=True),
        "sample_ent": (float("nan") if float(np.std(y)) < 1e-12
                       else _safe(ant.sample_entropy, y, order=2, metric="chebyshev")),
    }
    if len(y) < 4:
        out["ar2_coef_norm"] = float("nan")
    else:
        Y, X = y[2:], np.column_stack([y[1:-1], y[:-2]])
        try:
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            out["ar2_coef_norm"] = float(np.linalg.norm(coef))
        except Exception:
            out["ar2_coef_norm"] = float("nan")
    return out


def baseline_windows(sid: int) -> pd.DataFrame:
    """Per-window moments + classical baselines, cached (the classical ones are slow)."""
    cache = ROOT / f"subject_{sid}" / "baseline_window_features_v2.csv"
    if cache.exists():
        return pd.read_csv(cache)
    df = pd.read_csv(SUBJ / f"subject_{sid}.csv")
    windows = segment_labeled_series(
        values=df[["V1"]].to_numpy(float),
        labels=df["regime"].astype(str).to_numpy(),
        window_size=W, step_size=W, require_homogeneous_labels=True,
        record_ids=df["record_id"].astype(str).to_numpy(),
    )
    rows = []
    for w in windows:
        y = w.values[:, 0]
        feats = {"record_id": int(w.record_id), "start": int(w.start), "end": int(w.end),
                 "skewness": float(skew(y)), "excess_kurtosis": float(kurtosis(y))}
        feats.update(_classical(y))
        rows.append(feats)
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def load_subject(sid: int) -> pd.DataFrame:
    ws = pd.read_csv(ROOT / f"subject_{sid}" / "window_summaries.csv")
    ws = ws[ws["label"].isin([WAKE, SLEEP])].copy()
    base = baseline_windows(sid)
    merged = ws.merge(base, on=["record_id", "start", "end"], how="left")
    merged["subject"] = sid
    merged["y"] = (merged["label"] == SLEEP).astype(int)
    return merged


def main() -> None:
    sids = completed_subjects()
    if len(sids) < 3:
        print(f"need >=3 completed subjects for LOSO; have {sids}")
        return
    data = pd.concat([load_subject(s) for s in sids], ignore_index=True)

    print(f"Sleep-EDF LOSO AUC probe -- {len(sids)} subjects {sids} "
          f"({len(data)} windows)  [PROVISIONAL until all 14 land]")
    print("Leave-one-subject-out logistic regression (median-impute + standardise on train, "
          "C=1.0); ROC-AUC on held-out subject.\n")
    print(f"{'feature set':<26s} {'dim':>4s} {'folds':>6s}  {'mean AUC':>9s}  {'sd':>6s}")

    rows = []
    for name, feats in FEATURE_SETS.items():
        feats = [f for f in feats if f in data.columns]
        aucs = []
        for held in sids:
            tr, te = data[data.subject != held], data[data.subject == held]
            if te["y"].nunique() < 2 or tr["y"].nunique() < 2:
                continue
            pipe = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000),
            )
            pipe.fit(tr[feats].to_numpy(float), tr["y"].to_numpy())
            p = pipe.predict_proba(te[feats].to_numpy(float))[:, 1]
            aucs.append(roc_auc_score(te["y"].to_numpy(), p))
        m, sd = float(np.mean(aucs)), float(np.std(aucs))
        rows.append({"feature_set": name, "dim": len(feats), "folds": len(aucs),
                     "mean_auc": m, "sd_auc": sd})
        print(f"{name:<26s} {len(feats):>4d} {len(aucs):>6d}  {m:>9.3f}  {sd:>6.3f}")

    dest = ROOT / "probe_loso_auc.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
