"""Bilinear / quadratic-AR baselines on the moment-matched cross-lag scenario.

Motivation
----------
The paper's headline is that the multiplicative-interaction ("cross-lag") regime
pair is matched on first- and second-order summaries, so the five summary-statistic
baselines (DFA, largest Lyapunov, spectral entropy, sample entropy, AR(2)
coefficient norm) cannot tell the regimes apart, while the calibrated NLI can.

A natural reviewer objection: *any* cross-lag-aware feature would catch it, so the
comparison against five first/second-order baselines is unfair. This script answers
that objection directly by adding bilinear (quadratic) autoregression baselines that
are explicitly built to look for multiplicative structure, in three variants that
differ only in how much they are *told* in advance:

  bilinear_oracle   -- the lag design is augmented with the single product
                       y_{t-1}*y_{t-2}, i.e. the exact interaction term used to
                       generate regime B is pre-specified.
  bilinear_blind    -- the lag design is augmented with ALL 21 pairwise lag
                       products; the model must discover which one matters.
  bilinear_blind_l1 -- same full quadratic basis, but fitted with an L1 (Lasso)
                       penalty, i.e. a SINDy-style sparse selection over the basis.

For each window and variant we measure the relative MSE reduction of the augmented
model over a purely linear AR model on the same lags ("bilinear gain"), both
in-sample and out-of-sample (contiguous K-fold; shuffling would leak adjacent
autocorrelated samples). Each per-window gain is then tested for a regime effect
with the SAME pipeline the paper uses on the synthetic suite (block-permutation test,
block size 3, 1200 permutations, Cliff's delta, Benjamini-Hochberg across the 20
seeds).

Expected outcome and how it strengthens the paper
-------------------------------------------------
The oracle bilinear detects the regime difference (so the difference is genuinely
multiplicative and second-order, not an artefact of the calibrated NLI). The blind
bilinear -- dense or sparse, scored out-of-sample -- does NOT: the 21-term quadratic
basis cannot be identified from ~110 within-window samples. The calibrated NLI
recovers the effect WITHOUT specifying any interaction basis. This sharpens the
novelty claim from the unguarded "nothing detects it" to the defensible "recovering
it requires either knowing the interaction term in advance (oracle bilinear) or our
prior-free calibrated NLI; a blind quadratic search cannot."

The amplitude-rescaling nuisance scenario is run as a specificity check: every
variant should stay at zero there.

Run:  .venv/bin/python scripts/bilinear_baseline_interaction.py
Writes: outputs/bilinear_baseline/bilinear_baseline_results.csv  (+ console summary)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from symbolic_structural_profiling.statistics import (
    benjamini_hochberg,
    block_permutation_test,
    cliffs_delta,
)
from symbolic_structural_profiling.synthetic import generate_synthetic_regime_series
from symbolic_structural_profiling.windowing import (
    build_lagged_design,
    segment_labeled_series,
)

# Match the synthetic interaction runs exactly (outputs/synthetic:
# W=100, S=100, L=6, 20 seeds) and the synthetic statistical settings reported in the
# paper (block size 3, 1200 permutations).
N_STEPS = 6000
WINDOW_SIZE = 100
STEP_SIZE = 100
MAX_LAG = 6
SEEDS = list(range(20))
AMPLITUDE_SCALE_B = 2.0
SCENARIOS = ["interaction_only", "amplitude_only"]
CV_FOLDS = 5
RIDGE_ALPHA = 1.0  # matches the paper's linear-baseline Ridge(alpha=1.0)
LASSO_ALPHA = 0.001
BLOCK_SIZE = 3
N_PERMUTATIONS = 1200
ALPHA = 0.05

OUT_DIR = Path("outputs/bilinear_baseline")


def _oracle_column(X: np.ndarray) -> np.ndarray:
    """The single generative interaction term: lag1 * lag2 of the target var.

    build_lagged_design lays out columns as var0 lag1..lag6, so columns 0 and 1
    are y_{t-1} and y_{t-2}.
    """
    return (X[:, 0] * X[:, 1]).reshape(-1, 1)


def _all_products(X: np.ndarray) -> np.ndarray:
    """All unique pairwise products of the lag columns (i<=j includes squares)."""
    p = X.shape[1]
    cols = [X[:, i] * X[:, j] for i in range(p) for j in range(i, p)]
    return np.column_stack(cols) if cols else np.empty((X.shape[0], 0))


def _zscore(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def _insample_gain(X_lin: np.ndarray, X_extra: np.ndarray, y: np.ndarray) -> float:
    def mse(X: np.ndarray) -> float:
        model = LinearRegression().fit(X, y)
        return float(np.mean((y - model.predict(X)) ** 2))

    mse_lin = mse(X_lin)
    mse_full = mse(np.column_stack([X_lin, X_extra])) if X_extra.shape[1] else mse_lin
    return max(0.0, (mse_lin - mse_full) / mse_lin) if mse_lin > 1e-12 else 0.0


def _oos_gain(
    X_lin: np.ndarray, X_extra: np.ndarray, y: np.ndarray, *, sparse: bool
) -> float:
    n = len(y)
    fold_id = np.array_split(np.arange(n), CV_FOLDS)

    def cv_mse(X: np.ndarray, model_factory) -> float:
        if X.shape[1] == 0:
            return float("nan")
        errs = []
        for k in range(CV_FOLDS):
            test_idx = fold_id[k]
            train_idx = np.concatenate([fold_id[j] for j in range(CV_FOLDS) if j != k])
            if len(test_idx) == 0 or len(train_idx) < 2:
                continue
            Xtr, Xte = _zscore(X[train_idx], X[test_idx])
            model = model_factory().fit(Xtr, y[train_idx])
            errs.append(float(np.mean((y[test_idx] - model.predict(Xte)) ** 2)))
        return float(np.mean(errs)) if errs else float("nan")

    lin_factory = LinearRegression
    full_factory = (
        (lambda: Lasso(alpha=LASSO_ALPHA, max_iter=5000))
        if sparse
        else (lambda: Ridge(alpha=RIDGE_ALPHA))
    )
    mse_lin = cv_mse(X_lin, lin_factory)
    mse_full = cv_mse(np.column_stack([X_lin, X_extra]), full_factory)
    if not np.isfinite(mse_lin) or mse_lin <= 1e-12 or not np.isfinite(mse_full):
        return 0.0
    return max(0.0, (mse_lin - mse_full) / mse_lin)


# (label, extra-basis fn, gain fn) for each bilinear variant.
VARIANTS = {
    "bilinear_oracle_insample": (_oracle_column, lambda Xl, Xe, y: _insample_gain(Xl, Xe, y)),
    "bilinear_oracle_oos": (_oracle_column, lambda Xl, Xe, y: _oos_gain(Xl, Xe, y, sparse=False)),
    "bilinear_blind_oos": (_all_products, lambda Xl, Xe, y: _oos_gain(Xl, Xe, y, sparse=False)),
    "bilinear_blind_l1_oos": (_all_products, lambda Xl, Xe, y: _oos_gain(Xl, Xe, y, sparse=True)),
}


def _windows(scenario: str, seed: int):
    ds = generate_synthetic_regime_series(n_steps=N_STEPS, scenario=scenario, seed=seed,
                                          amplitude_scale_b=AMPLITUDE_SCALE_B)
    return segment_labeled_series(
        values=ds.values,
        labels=ds.labels,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        require_homogeneous_labels=True,
    )


def test_one(variant: str, scenario: str, seed: int) -> dict:
    basis_fn, gain_fn = VARIANTS[variant]
    vals, labs = [], []
    for w in _windows(scenario, seed):
        X_lin, y, _ = build_lagged_design(w.values, max_lag=MAX_LAG, target_index=0)
        vals.append(gain_fn(X_lin, basis_fn(X_lin), y))
        labs.append(w.label)
    vals = np.asarray(vals, dtype=float)
    labs = np.asarray(labs)
    obs, p = block_permutation_test(
        values=vals,
        labels=labs,
        group_a="A",
        group_b="B",
        block_size=BLOCK_SIZE,
        n_permutations=N_PERMUTATIONS,
        seed=37,
    )
    return {
        "variant": variant,
        "scenario": scenario,
        "seed": seed,
        "median_diff": float(obs),
        "cliffs_delta": float(cliffs_delta(vals[labs == "A"], vals[labs == "B"])),
        "p_value": float(p),
        "median_gain_a": float(np.median(vals[labs == "A"])),
        "median_gain_b": float(np.median(vals[labs == "B"])),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                records.append(test_one(variant, scenario, seed))

    df = pd.DataFrame(records)

    # BH-FDR across the seeds within each (variant, scenario) family.
    df["reject"] = False
    df["q_value"] = np.nan
    for (variant, scenario), idx in df.groupby(["variant", "scenario"]).groups.items():
        reject, q = benjamini_hochberg(df.loc[idx, "p_value"].to_numpy(), ALPHA)
        df.loc[idx, "reject"] = reject
        df.loc[idx, "q_value"] = q

    df.to_csv(OUT_DIR / "bilinear_baseline_results.csv", index=False)

    print(f"=== Bilinear-AR baselines, A vs B, {len(SEEDS)} seeds (block=3, 1200 perms) ===")
    print(f"{'variant':26s} {'scenario':16s}  FDR-sig  med_delta  sign-consistency  medGainB")
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            sub = df[(df["variant"] == variant) & (df["scenario"] == scenario)]
            deltas = sub["cliffs_delta"].to_numpy()
            n_neg = int(np.sum(deltas < 0))
            n_pos = int(np.sum(deltas > 0))
            direction = "B>A" if n_neg >= n_pos else "A>B"
            print(
                f"{variant:26s} {scenario:16s}  "
                f"{int(sub['reject'].sum()):2d}/{len(SEEDS)}    "
                f"{np.median(deltas):+.3f}    "
                f"{max(n_neg, n_pos)}/{len(SEEDS)} ({direction})    "
                f"{sub['median_gain_b'].median():.4f}"
            )
    print(f"\nWrote {OUT_DIR / 'bilinear_baseline_results.csv'}")


if __name__ == "__main__":
    main()
