import numpy as np
import pandas as pd

from symbolic_structural_profiling.config import StatsConfig
from symbolic_structural_profiling.statistics import (
    benjamini_hochberg,
    block_permutation_test,
    cliffs_delta,
    run_pairwise_feature_tests,
)


def test_cliffs_delta_direction():
    a = np.array([3, 4, 5, 6])
    b = np.array([0, 1, 2, 3])
    assert cliffs_delta(a, b) > 0.0


def test_bh_outputs_shape_and_bounds():
    p = [0.001, 0.01, 0.2, 0.5]
    reject, q = benjamini_hochberg(p, alpha=0.05)
    assert reject.shape == q.shape
    assert np.all((q >= 0.0) & (q <= 1.0))


def test_block_permutation_test_smoke():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0.0, 1.0, 80), rng.normal(0.7, 1.0, 80)])
    labels = np.array(["A"] * 80 + ["B"] * 80)
    obs, p = block_permutation_test(values, labels, "A", "B", block_size=8, n_permutations=120)
    assert np.isfinite(obs)
    assert 0.0 <= p <= 1.0


def test_block_permutation_auto_fallback_for_degenerate_block_layout():
    values = np.arange(64, dtype=float)
    labels = np.array(["A" if i % 2 == 0 else "B" for i in range(64)], dtype=str)

    obs, p, meta = block_permutation_test(
        values,
        labels,
        "A",
        "B",
        block_size=4,
        n_permutations=120,
        return_meta=True,
    )
    assert np.isfinite(obs)
    assert 0.0 <= p <= 1.0
    assert meta["block_size_requested"] == 4
    assert meta["block_size_used"] != 4
    assert bool(meta["block_size_fallback_used"]) is True
    assert float(meta["null_range"]) > 0.0


def test_effect_size_relevance_filter_marks_small_effect_as_not_relevant():
    rows = []
    for i in range(120):
        label = "A" if i < 60 else "B"
        # Small deterministic shift relative to wide global spread.
        base = float(i % 20) / 10.0
        value = base + (0.05 if label == "A" else 0.0)
        rows.append({"label": label, "lag_mass_median": value})
    df = pd.DataFrame(rows)

    cfg = StatsConfig(
        block_size=6,
        n_permutations=120,
        ci_bootstrap_repeats=30,
        ci_alpha=0.05,
        ci_seed=3,
        alpha=0.2,
        min_effect_iqr_ratio=0.5,
    )
    out = run_pairwise_feature_tests(df, ["lag_mass_median"], cfg)

    assert not out.empty
    assert "effect_iqr_ratio" in out.columns
    assert "practically_relevant" in out.columns
    assert "reject_and_relevant" in out.columns
    assert "block_size_requested" in out.columns
    assert "block_size_used" in out.columns
    assert "block_size_fallback_used" in out.columns
    assert "null_range" in out.columns
    assert "median_diff_ci_low" in out.columns
    assert "median_diff_ci_high" in out.columns
    assert np.isfinite(float(out.iloc[0]["median_diff_ci_low"]))
    assert np.isfinite(float(out.iloc[0]["median_diff_ci_high"]))
    assert bool(out.iloc[0]["practically_relevant"]) is False
    assert bool(out.iloc[0]["reject_and_relevant"]) is False


def test_pairwise_balancing_downsamples_majority_label_group():
    rows = []
    for i in range(90):
        rows.append({"label": "A", "lag_mass_median": float(i % 10)})
    for i in range(30):
        rows.append({"label": "B", "lag_mass_median": float(i % 10)})
    df = pd.DataFrame(rows)

    cfg = StatsConfig(
        block_size=5,
        n_permutations=80,
        ci_bootstrap_repeats=20,
        alpha=0.2,
        min_effect_iqr_ratio=0.1,
        balance_labels_for_pairwise=True,
        balance_seed=7,
    )
    out = run_pairwise_feature_tests(df, ["lag_mass_median"], cfg)

    assert not out.empty
    row = out.iloc[0]
    assert bool(row["pairwise_balanced"]) is True
    assert bool(row["balanced_downsampled"]) is True
    assert int(row["n_group_a_raw"]) == 90
    assert int(row["n_group_b_raw"]) == 30
    assert int(row["n_group_a_used"]) == 30
    assert int(row["n_group_b_used"]) == 30


def test_effect_iqr_ratio_is_nan_when_global_iqr_is_numerically_unstable():
    rows = []
    a_val = 1.0
    b_val = np.nextafter(1.0, 2.0)
    for _ in range(60):
        rows.append({"label": "A", "lag_mass_median": a_val})
    for _ in range(60):
        rows.append({"label": "B", "lag_mass_median": b_val})
    df = pd.DataFrame(rows)

    cfg = StatsConfig(
        block_size=6,
        n_permutations=80,
        ci_bootstrap_repeats=20,
        alpha=0.2,
        min_effect_iqr_ratio=0.1,
    )
    out = run_pairwise_feature_tests(df, ["lag_mass_median"], cfg)

    assert not out.empty
    row = out.iloc[0]
    assert np.isnan(float(row["effect_iqr_ratio"]))
    assert bool(row["practically_relevant"]) is False
