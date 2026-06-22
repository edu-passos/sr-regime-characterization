from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Scenario = Literal[
    "memory_only",
    "nonlinearity_only",
    "interaction_only",
    "parameter_only",
    "amplitude_only",
    "offset_only",
    "noise_only",
    "iid_noise",
]


@dataclass(slots=True)
class SyntheticDataset:
    values: np.ndarray
    labels: np.ndarray
    feature_names: list[str]


def _regime_label(t: int, block: int) -> str:
    return "A" if ((t // block) % 2 == 0) else "B"


def _clip_state(value: float) -> float:
    return float(np.clip(value, -6.0, 6.0))


def _match_first_two_moments(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    reference_array = np.asarray(reference, dtype=float)

    ref_mean = float(np.mean(reference_array))
    ref_std = float(np.std(reference_array))
    value_mean = float(np.mean(values_array))
    value_std = float(np.std(values_array))

    centered = values_array - value_mean
    scaled = centered * (ref_std / max(value_std, 1e-9))
    return scaled + ref_mean


def _simulate_univariate_process(
    innovations: np.ndarray,
    *,
    initial_state: np.ndarray,
    linear_coeffs: dict[int, float],
    tanh_strength: float = 0.0,
    interaction_strength: float = 0.0,
    driver_state: np.ndarray | None = None,
) -> np.ndarray:
    total_length = int(len(innovations))
    max_required_lag = int(max([2, *linear_coeffs.keys()]))
    state = np.zeros(total_length, dtype=float)
    state[:max_required_lag] = np.asarray(initial_state[:max_required_lag], dtype=float)
    driver = state if driver_state is None else np.asarray(driver_state, dtype=float)

    for t in range(max_required_lag, total_length):
        value = float(innovations[t])
        for lag, coeff in linear_coeffs.items():
            value += float(coeff) * float(driver[t - int(lag)])
        if tanh_strength != 0.0:
            nonlinear_residual = float(np.tanh(8.0 * driver[t - 1]) - driver[t - 1])
            value += float(tanh_strength) * nonlinear_residual
        if interaction_strength != 0.0:
            value += float(interaction_strength) * 6.0 * float(driver[t - 1] * driver[t - 2])
        state[t] = _clip_state(value)

    return state


def _base_linear_coeffs() -> dict[int, float]:
    return {1: 0.68, 2: -0.18}


def _memory_pair_coeffs(memory_lag_b: int) -> tuple[dict[int, float], dict[int, float]]:
    lag_b = int(max(2, min(memory_lag_b, 8)))
    coeffs_a = _base_linear_coeffs()
    coeffs_b = dict(coeffs_a)
    coeffs_b[lag_b] = float(coeffs_b.get(lag_b, 0.0) - 0.40)
    return coeffs_a, coeffs_b


def _parameter_pair_coeffs() -> tuple[dict[int, float], dict[int, float]]:
    return {1: 0.68, 2: -0.18}, {1: 0.58, 2: -0.28}


def _paired_regime_series(
    n_steps: int,
    *,
    scenario: Scenario,
    noise_std: float,
    seed: int,
    memory_lag_b: int,
    nonlinearity_strength_b: float,
    interaction_strength_b: float,
    amplitude_scale_b: float,
    offset_b: float,
    noise_scale_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lag_b = int(max(2, min(memory_lag_b, 8)))
    burn_in = max(120, 16 * lag_b)
    if scenario == "memory_only":
        burn_in = max(160, 20 * lag_b)
    total_length = int(n_steps + burn_in)
    initial_state = rng.normal(0.0, max(0.12, float(noise_std)), size=10)
    shared_innovations = rng.normal(0.0, float(noise_std), size=total_length)

    if scenario == "memory_only":
        coeffs_a, coeffs_b = _memory_pair_coeffs(memory_lag_b)
        series_a = _simulate_univariate_process(
            shared_innovations,
            initial_state=initial_state,
            linear_coeffs=coeffs_a,
        )
        series_b = _simulate_univariate_process(
            shared_innovations,
            initial_state=initial_state,
            linear_coeffs=coeffs_b,
        )
    elif scenario == "nonlinearity_only":
        coeffs = _base_linear_coeffs()
        series_a = _simulate_univariate_process(
            shared_innovations,
            initial_state=initial_state,
            linear_coeffs=coeffs,
        )
        # Regime B is a threshold autoregression (SETAR; Tong 1990): the lag-1
        # coefficient switches with the sign of the previous value. Unlike a weak
        # additive curvature -- which a linear AR predicts almost as well, so
        # out-of-sample model selection rejects it -- a SETAR has a genuinely
        # nonlinear one-step map that no linear predictor can capture, so the
        # nonlinearity is strongly detectable out-of-sample. It shares regime A's
        # lag-2 structure; only the sign-dependent lag-1 gain is new.
        #
        # The regime gap |phi_pos - phi_neg| IS the nonlinearity strength: it scales
        # with nonlinearity_strength_b, so the strength sweep genuinely varies the
        # nonlinearity, and strength 0 collapses both regimes to one linear AR (the
        # null). The coefficient pair is drawn at RANDOM per seed (not hand-picked),
        # clipped for stationarity, so no result depends on a chosen coefficient
        # value. A strong nonlinearity of a lag necessarily perturbs the lag readout
        # too (a nonlinear map of a lagged variable is a memory operator -- Volterra),
        # so some lag movement is intrinsic; the gated nonlinearity index is the
        # concept that discriminates this from a (linear) memory change.
        lag2_coeff = float(coeffs.get(2, -0.18))
        gap = float(nonlinearity_strength_b) * float(rng.uniform(3.2, 3.9))
        center = float(rng.uniform(0.0, 0.20))
        phi_pos = float(np.clip(center + 0.5 * gap, -0.95, 0.95))
        phi_neg = float(np.clip(center - 0.5 * gap, -0.95, 0.95))
        series_b = np.zeros_like(series_a)
        series_b[:2] = series_a[:2]
        for t in range(2, len(series_b)):
            phi = phi_pos if series_b[t - 1] >= 0.0 else phi_neg
            series_b[t] = _clip_state(
                phi * series_b[t - 1]
                + lag2_coeff * series_b[t - 2]
                + float(shared_innovations[t])
            )
    elif scenario == "interaction_only":
        coeffs = _base_linear_coeffs()
        series_a = _simulate_univariate_process(
            shared_innovations,
            initial_state=initial_state,
            linear_coeffs=coeffs,
        )
        series_b = _simulate_univariate_process(
            shared_innovations,
            initial_state=initial_state,
            linear_coeffs=coeffs,
            interaction_strength=float(interaction_strength_b),
            driver_state=series_a,
        )
        series_b = _match_first_two_moments(series_b, series_a)
    elif scenario == "parameter_only":
        coeffs_a, coeffs_b = _parameter_pair_coeffs()
        series_a = _simulate_univariate_process(
            shared_innovations,
            initial_state=initial_state,
            linear_coeffs=coeffs_a,
        )
        series_b = _simulate_univariate_process(
            shared_innovations,
            initial_state=initial_state,
            linear_coeffs=coeffs_b,
        )
    elif scenario == "amplitude_only":
        coeffs = _base_linear_coeffs()
        base_series = _simulate_univariate_process(
            shared_innovations,
            initial_state=initial_state,
            linear_coeffs=coeffs,
        )
        series_a = base_series
        series_b = float(amplitude_scale_b) * base_series
    elif scenario == "offset_only":
        coeffs = _base_linear_coeffs()
        base_series = _simulate_univariate_process(
            shared_innovations,
            initial_state=initial_state,
            linear_coeffs=coeffs,
        )
        series_a = base_series
        series_b = base_series + float(offset_b)
    elif scenario == "noise_only":
        coeffs = _base_linear_coeffs()
        scaled_innovations = shared_innovations * float(noise_scale_b)
        series_a = _simulate_univariate_process(
            shared_innovations,
            initial_state=initial_state,
            linear_coeffs=coeffs,
        )
        series_b = _simulate_univariate_process(
            scaled_innovations,
            initial_state=initial_state,
            linear_coeffs=coeffs,
        )
    elif scenario == "iid_noise":
        # AR(0): both regimes are pure iid Gaussian noise with no
        # autoregressive structure. A null scenario for the calibrated
        # NLI: there is no cross-lag signal to recover, no memory
        # difference to detect, no nonlinearity. Any feature firing
        # here is a false positive.
        rng_b = np.random.default_rng(int(seed) + 1)
        innovations_b = rng_b.normal(0.0, float(noise_std), size=total_length)
        series_a = shared_innovations.copy()
        series_b = innovations_b
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return series_a[burn_in:], series_b[burn_in:]


def generate_synthetic_regime_series(
    n_steps: int = 3200,
    scenario: Scenario = "memory_only",
    noise_std: float = 0.16,
    seed: int = 0,
    memory_lag_b: int = 5,
    nonlinearity_strength_b: float = 0.42,
    interaction_strength_b: float = 0.33,
    amplitude_scale_b: float = 1.9,
    offset_b: float = 0.35,
    noise_scale_b: float = 1.75,
    regime_block_size: int | None = None,
) -> SyntheticDataset:
    if n_steps < 100:
        raise ValueError("n_steps should be at least 100")

    block = 200 if regime_block_size is None else int(max(1, regime_block_size))
    regime_a, regime_b = _paired_regime_series(
        n_steps=n_steps,
        scenario=scenario,
        noise_std=noise_std,
        seed=seed,
        memory_lag_b=memory_lag_b,
        nonlinearity_strength_b=nonlinearity_strength_b,
        interaction_strength_b=interaction_strength_b,
        amplitude_scale_b=amplitude_scale_b,
        offset_b=offset_b,
        noise_scale_b=noise_scale_b,
    )

    values = np.zeros((n_steps, 1), dtype=float)
    labels = np.empty(n_steps, dtype="U1")
    regime_offsets = {"A": 0, "B": 0}
    for start in range(0, n_steps, block):
        end = min(start + block, n_steps)
        block_length = int(end - start)
        label = _regime_label(start, block)
        labels[start:end] = label

        regime_start = int(regime_offsets[label])
        regime_end = regime_start + block_length
        if label == "A":
            values[start:end, 0] = regime_a[regime_start:regime_end]
        else:
            values[start:end, 0] = regime_b[regime_start:regime_end]
        regime_offsets[label] = regime_end

    return SyntheticDataset(values=values, labels=labels, feature_names=["y"])
