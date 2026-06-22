from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import StatsConfig


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0

    greater = 0
    lesser = 0
    for xv in x:
        greater += int(np.sum(xv > y))
        lesser += int(np.sum(xv < y))
    return float((greater - lesser) / (x.size * y.size))


def _median_diff(values: np.ndarray, labels: np.ndarray, a: str, b: str) -> float:
    va = values[labels == a]
    vb = values[labels == b]
    return float(np.median(va) - np.median(vb))


def _stable_seed(*parts: str) -> int:
    text = "|".join(parts)
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(text)))


def _effect_iqr_ratio(values: np.ndarray, obs: float) -> tuple[float, float]:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    global_iqr = float(q75 - q25)
    scale = max(float(np.std(values)), float(np.median(np.abs(values))), 1.0)
    min_stable_iqr = 1e-6 * scale

    if not np.isfinite(global_iqr) or global_iqr < min_stable_iqr:
        return global_iqr, float("nan")

    return global_iqr, float(abs(obs) / global_iqr)


def _balanced_pair_values(
    values: np.ndarray,
    labels: np.ndarray,
    group_a: str,
    group_b: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]]:
    pair_mask = (labels == group_a) | (labels == group_b)
    pair_values = values[pair_mask]
    pair_labels = labels[pair_mask]

    idx_a = np.where(pair_labels == group_a)[0]
    idx_b = np.where(pair_labels == group_b)[0]
    n_a_raw = int(idx_a.size)
    n_b_raw = int(idx_b.size)

    if n_a_raw == 0 or n_b_raw == 0:
        return pair_values, pair_labels, {
            "n_group_a_raw": n_a_raw,
            "n_group_b_raw": n_b_raw,
            "n_group_a_used": n_a_raw,
            "n_group_b_used": n_b_raw,
            "balanced_downsampled": False,
        }

    if n_a_raw == n_b_raw:
        return pair_values, pair_labels, {
            "n_group_a_raw": n_a_raw,
            "n_group_b_raw": n_b_raw,
            "n_group_a_used": n_a_raw,
            "n_group_b_used": n_b_raw,
            "balanced_downsampled": False,
        }

    n_target = int(min(n_a_raw, n_b_raw))
    rng = np.random.default_rng(seed)
    use_a = np.sort(rng.choice(idx_a, size=n_target, replace=False))
    use_b = np.sort(rng.choice(idx_b, size=n_target, replace=False))
    use_idx = np.sort(np.concatenate([use_a, use_b]))

    return pair_values[use_idx], pair_labels[use_idx], {
        "n_group_a_raw": n_a_raw,
        "n_group_b_raw": n_b_raw,
        "n_group_a_used": n_target,
        "n_group_b_used": n_target,
        "balanced_downsampled": True,
    }


def _candidate_block_sizes(block_size: int, n: int) -> list[int]:
    # Try nearby sizes first; add 1 as last-resort fallback.
    out: list[int] = []
    for delta in [0, -1, 1, -2, 2, -3, 3]:
        size = block_size + delta
        if size < 1 or size > n:
            continue
        if size not in out:
            out.append(size)
    if 1 not in out:
        out.append(1)
    return out


def _null_distribution_block_permutation(
    values: np.ndarray,
    labels: np.ndarray,
    group_a: str,
    group_b: str,
    block_size: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    block_indices = [
        np.arange(i, min(i + block_size, values.size))
        for i in range(0, values.size, block_size)
    ]

    rng = np.random.default_rng(seed)
    null_values = np.zeros(n_permutations, dtype=float)
    for i in range(n_permutations):
        order = rng.permutation(len(block_indices))
        perm_labels = np.concatenate([labels[block_indices[j]] for j in order])
        null_values[i] = _median_diff(values, perm_labels, group_a, group_b)
    return null_values


def _block_bootstrap_median_diff_ci(
    values: np.ndarray,
    labels: np.ndarray,
    group_a: str,
    group_b: str,
    block_size: int,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    if values.size < 4 or n_bootstrap <= 0:
        return float("nan"), float("nan")

    block_indices = [
        np.arange(i, min(i + block_size, values.size))
        for i in range(0, values.size, block_size)
    ]
    if not block_indices:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    diffs = np.full(int(n_bootstrap), np.nan, dtype=float)

    for i in range(int(n_bootstrap)):
        sampled_idx = np.empty(values.size, dtype=int)
        filled = 0
        while filled < values.size:
            block = block_indices[int(rng.integers(0, len(block_indices)))]
            take = min(len(block), values.size - filled)
            sampled_idx[filled : filled + take] = block[:take]
            filled += take

        boot_values = values[sampled_idx]
        boot_labels = labels[sampled_idx]
        if not np.any(boot_labels == group_a) or not np.any(boot_labels == group_b):
            continue
        diffs[i] = _median_diff(boot_values, boot_labels, group_a, group_b)

    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan")

    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2))
    return lo, hi


def _adaptive_block_null_distribution(
    values: np.ndarray,
    labels: np.ndarray,
    group_a: str,
    group_b: str,
    block_size: int,
    n_permutations: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    candidate_sizes = _candidate_block_sizes(block_size=block_size, n=values.size)

    best_null = np.zeros(n_permutations, dtype=float)
    best_size = int(candidate_sizes[0])
    best_span = -1.0

    for idx, size in enumerate(candidate_sizes):
        null_values = _null_distribution_block_permutation(
            values=values,
            labels=labels,
            group_a=group_a,
            group_b=group_b,
            block_size=int(size),
            n_permutations=n_permutations,
            seed=seed + idx * 1009,
        )
        span = float(np.ptp(null_values))
        if span > best_span:
            best_span = span
            best_null = null_values
            best_size = int(size)
        if span > 1e-12:
            return null_values, int(size)

    return best_null, best_size


def block_permutation_test(
    values: Sequence[float],
    labels: Sequence[str],
    group_a: str,
    group_b: str,
    block_size: int,
    n_permutations: int,
    seed: int = 0,
    return_meta: bool = False,
) -> tuple[float, float] | tuple[float, float, dict[str, float | int | bool]]:
    vals = np.asarray(values, dtype=float)
    labs = np.asarray(labels)

    mask = (labs == group_a) | (labs == group_b)
    vals = vals[mask]
    labs = labs[mask]

    if vals.size < 4:
        if return_meta:
            return 0.0, 1.0, {
                "block_size_requested": int(block_size),
                "block_size_used": int(block_size),
                "block_size_fallback_used": False,
                "null_range": 0.0,
            }
        return 0.0, 1.0

    observed = _median_diff(vals, labs, group_a, group_b)

    null_values, used_block_size = _adaptive_block_null_distribution(
        values=vals,
        labels=labs,
        group_a=group_a,
        group_b=group_b,
        block_size=max(1, int(block_size)),
        n_permutations=int(n_permutations),
        seed=seed,
    )

    p_value = float((np.sum(np.abs(null_values) >= abs(observed)) + 1) / (n_permutations + 1))
    if return_meta:
        return observed, p_value, {
            "block_size_requested": int(block_size),
            "block_size_used": int(used_block_size),
            "block_size_fallback_used": bool(int(used_block_size) != int(block_size)),
            "null_range": float(np.ptp(null_values)),
        }
    return observed, p_value


def benjamini_hochberg(p_values: Sequence[float], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * np.arange(1, m + 1) / m
    passed = ranked <= thresholds

    reject = np.zeros(m, dtype=bool)
    if np.any(passed):
        k = int(np.max(np.where(passed)[0]))
        reject[order[: k + 1]] = True

    q_values = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q = min(prev, ranked[i] * m / rank)
        q_values[i] = q
        prev = q

    q_out = np.empty(m, dtype=float)
    q_out[order] = q_values
    return reject, q_out


def run_pairwise_feature_tests(
    window_summaries: pd.DataFrame,
    feature_columns: Iterable[str],
    config: StatsConfig,
) -> pd.DataFrame:
    if window_summaries.empty:
        return pd.DataFrame()

    labels = sorted(window_summaries["label"].astype(str).unique().tolist())
    pairs = list(combinations(labels, 2))

    rows: List[dict] = []
    for feature in feature_columns:
        values = window_summaries[feature].to_numpy(dtype=float)
        group_labels = window_summaries["label"].astype(str).to_numpy()

        for a, b in pairs:
            if config.balance_labels_for_pairwise:
                pair_values, pair_labels, pair_meta = _balanced_pair_values(
                    values=values,
                    labels=group_labels,
                    group_a=a,
                    group_b=b,
                    seed=int(config.balance_seed + _stable_seed(feature, a, b)),
                )
            else:
                pair_mask = (group_labels == a) | (group_labels == b)
                pair_values = values[pair_mask]
                pair_labels = group_labels[pair_mask]
                n_a = int(np.sum(pair_labels == a))
                n_b = int(np.sum(pair_labels == b))
                pair_meta = {
                    "n_group_a_raw": n_a,
                    "n_group_b_raw": n_b,
                    "n_group_a_used": n_a,
                    "n_group_b_used": n_b,
                    "balanced_downsampled": False,
                }

            obs, p, meta = block_permutation_test(
                values=pair_values,
                labels=pair_labels,
                group_a=a,
                group_b=b,
                block_size=config.block_size,
                n_permutations=config.n_permutations,
                seed=37,
                return_meta=True,
            )

            va = pair_values[pair_labels == a]
            vb = pair_values[pair_labels == b]
            ci_low, ci_high = _block_bootstrap_median_diff_ci(
                values=pair_values,
                labels=pair_labels,
                group_a=a,
                group_b=b,
                block_size=config.block_size,
                n_bootstrap=config.ci_bootstrap_repeats,
                alpha=float(config.ci_alpha),
                seed=int(config.ci_seed + _stable_seed(feature, a, b)),
            )
            global_iqr, effect_iqr_ratio = _effect_iqr_ratio(values, float(obs))

            rows.append(
                {
                    "feature": feature,
                    "group_a": a,
                    "group_b": b,
                    "median_diff": float(obs),
                    "median_diff_ci_low": float(ci_low),
                    "median_diff_ci_high": float(ci_high),
                    "cliffs_delta": float(cliffs_delta(va, vb)),
                    "global_iqr": global_iqr,
                    "effect_iqr_ratio": effect_iqr_ratio,
                    "p_value": float(p),
                    "block_size_requested": int(meta["block_size_requested"]),
                    "block_size_used": int(meta["block_size_used"]),
                    "block_size_fallback_used": bool(meta["block_size_fallback_used"]),
                    "null_range": float(meta["null_range"]),
                    "pairwise_balanced": bool(config.balance_labels_for_pairwise),
                    "balanced_downsampled": bool(pair_meta["balanced_downsampled"]),
                    "n_group_a_raw": int(pair_meta["n_group_a_raw"]),
                    "n_group_b_raw": int(pair_meta["n_group_b_raw"]),
                    "n_group_a_used": int(pair_meta["n_group_a_used"]),
                    "n_group_b_used": int(pair_meta["n_group_b_used"]),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    reject, q_values = benjamini_hochberg(df["p_value"].to_numpy(), config.alpha)
    df["q_value"] = q_values
    df["reject"] = reject
    df["practically_relevant"] = df["effect_iqr_ratio"] >= config.min_effect_iqr_ratio
    df["reject_and_relevant"] = df["reject"] & df["practically_relevant"]
    return df.sort_values(["q_value", "p_value"]).reset_index(drop=True)
