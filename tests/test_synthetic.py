import numpy as np

from symbolic_structural_profiling.scaling import standardize_window_design
from symbolic_structural_profiling.synthetic import (
    _memory_pair_coeffs,
    _paired_regime_series,
    generate_synthetic_regime_series,
)
from symbolic_structural_profiling.windowing import build_lagged_design


def test_paired_regime_series_is_deterministic_for_a_seed():
    a1, b1 = _paired_regime_series(
        600,
        scenario="memory_only",
        noise_std=0.16,
        seed=7,
        memory_lag_b=6,
        nonlinearity_strength_b=0.42,
        interaction_strength_b=0.33,
        amplitude_scale_b=1.9,
        offset_b=0.35,
        noise_scale_b=1.75,
    )
    a2, b2 = _paired_regime_series(
        600,
        scenario="memory_only",
        noise_std=0.16,
        seed=7,
        memory_lag_b=6,
        nonlinearity_strength_b=0.42,
        interaction_strength_b=0.33,
        amplitude_scale_b=1.9,
        offset_b=0.35,
        noise_scale_b=1.75,
    )

    assert np.allclose(a1, a2)
    assert np.allclose(b1, b2)


def test_amplitude_and_offset_scenarios_are_post_generation_transforms():
    amp_a, amp_b = _paired_regime_series(
        512,
        scenario="amplitude_only",
        noise_std=0.16,
        seed=11,
        memory_lag_b=5,
        nonlinearity_strength_b=0.42,
        interaction_strength_b=0.33,
        amplitude_scale_b=1.8,
        offset_b=0.35,
        noise_scale_b=1.75,
    )
    off_a, off_b = _paired_regime_series(
        512,
        scenario="offset_only",
        noise_std=0.16,
        seed=11,
        memory_lag_b=5,
        nonlinearity_strength_b=0.42,
        interaction_strength_b=0.33,
        amplitude_scale_b=1.8,
        offset_b=0.35,
        noise_scale_b=1.75,
    )

    assert np.allclose(amp_b, 1.8 * amp_a)
    assert np.allclose(off_b - off_a, 0.35)


def test_iid_noise_scenario_has_no_autoregressive_structure():
    data = generate_synthetic_regime_series(n_steps=4000, scenario="iid_noise", seed=3)
    a = data.values[data.labels == "A", 0]
    b = data.values[data.labels == "B", 0]
    assert len(a) > 100 and len(b) > 100
    # AR(0): lag-1 autocorrelation should be near zero (within sampling noise).
    ac_a = float(np.corrcoef(a[:-1], a[1:])[0, 1])
    ac_b = float(np.corrcoef(b[:-1], b[1:])[0, 1])
    assert abs(ac_a) < 0.15, f"A lag-1 autocorr {ac_a:.3f} not ~0"
    assert abs(ac_b) < 0.15, f"B lag-1 autocorr {ac_b:.3f} not ~0"
    # Both regimes should be drawn from independent RNGs but matching scale.
    assert abs(np.std(a) - np.std(b)) < 0.05


def test_generate_synthetic_regime_series_keeps_raw_scale():
    data = generate_synthetic_regime_series(
        n_steps=800,
        scenario="offset_only",
        seed=5,
        offset_b=1.75,
    )

    assert np.std(data.values[:, 0]) > 0.1
    assert abs(float(np.mean(data.values[:, 0]))) > 0.01


def test_generate_synthetic_regime_series_aligns_alternating_blocks_by_regime_slice():
    data = generate_synthetic_regime_series(
        n_steps=800,
        scenario="offset_only",
        seed=5,
        offset_b=1.75,
    )

    block = 200
    first_a = data.values[0:block, 0]
    first_b = data.values[block : 2 * block, 0]
    second_a = data.values[2 * block : 3 * block, 0]
    second_b = data.values[3 * block : 4 * block, 0]

    assert np.all(data.labels[:block] == "A")
    assert np.all(data.labels[block : 2 * block] == "B")
    assert np.allclose(first_b - first_a, 1.75)
    assert np.allclose(second_b - second_a, 1.75)


def test_window_aligned_blocks_preserve_offset_and_amplitude_after_standardization():
    window_size = 120
    regime_block_size = 240

    def _scaled_design(window_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_raw, y_raw, _ = build_lagged_design(window_values, max_lag=6, target_index=0)
        X_scaled, y_scaled, _ = standardize_window_design(X_raw, y_raw, enabled=True)
        return X_scaled, y_scaled

    offset_data = generate_synthetic_regime_series(
        n_steps=960,
        scenario="offset_only",
        seed=5,
        offset_b=1.75,
        regime_block_size=regime_block_size,
    )
    offset_a = offset_data.values[0:window_size]
    offset_b = offset_data.values[regime_block_size : regime_block_size + window_size]
    assert np.allclose(offset_b[:, 0] - offset_a[:, 0], 1.75)
    offset_x_a, offset_y_a = _scaled_design(offset_a)
    offset_x_b, offset_y_b = _scaled_design(offset_b)
    assert np.allclose(offset_x_a, offset_x_b)
    assert np.allclose(offset_y_a, offset_y_b)

    amplitude_data = generate_synthetic_regime_series(
        n_steps=960,
        scenario="amplitude_only",
        seed=5,
        amplitude_scale_b=1.8,
        regime_block_size=regime_block_size,
    )
    amplitude_a = amplitude_data.values[0:window_size]
    amplitude_b = amplitude_data.values[regime_block_size : regime_block_size + window_size]
    assert np.allclose(amplitude_b[:, 0], 1.8 * amplitude_a[:, 0])
    amplitude_x_a, amplitude_y_a = _scaled_design(amplitude_a)
    amplitude_x_b, amplitude_y_b = _scaled_design(amplitude_b)
    assert np.allclose(amplitude_x_a, amplitude_x_b)
    assert np.allclose(amplitude_y_a, amplitude_y_b)


def test_memory_pair_coeffs_share_short_lag_base_and_add_long_lag_only_to_regime_b():
    coeffs_a, coeffs_b = _memory_pair_coeffs(6)

    assert coeffs_a == {1: 0.68, 2: -0.18}
    assert coeffs_b[1] == coeffs_a[1]
    assert coeffs_b[2] == coeffs_a[2]
    assert coeffs_b[6] == -0.40
    assert set(coeffs_a.keys()) == {1, 2}
    assert set(coeffs_b.keys()) == {1, 2, 6}


def test_nonlinearity_only_produces_structurally_different_series():
    series_a, series_b = _paired_regime_series(
        2200,
        scenario="nonlinearity_only",
        noise_std=0.16,
        seed=0,
        memory_lag_b=5,
        nonlinearity_strength_b=0.6,
        interaction_strength_b=0.0,
        amplitude_scale_b=1.9,
        offset_b=0.35,
        noise_scale_b=1.75,
    )

    # Regime B is not moment-matched - it carries the raw nonlinear DGP variance.
    # The structural difference is detected via the Jacobian, not distributional stats.
    assert float(np.mean(np.abs(series_a - series_b))) > 0.05

    # The nonlinear DGP should produce a gradient (tanh(8x) - x) with non-trivial
    # variance at typical AR values, confirming the k=8 coefficient is effective.
    grad_signal = 8.0 / np.cosh(8.0 * series_a) ** 2 - 1.0
    assert float(np.std(grad_signal)) > 1.0


def test_interaction_only_matches_first_two_moments_and_is_materially_stronger_than_old_baseline():
    baseline_a, baseline_b = _paired_regime_series(
        2200,
        scenario="interaction_only",
        noise_std=0.16,
        seed=0,
        memory_lag_b=5,
        nonlinearity_strength_b=0.0,
        interaction_strength_b=0.0,
        amplitude_scale_b=1.9,
        offset_b=0.35,
        noise_scale_b=1.75,
    )
    series_a, series_b = _paired_regime_series(
        2200,
        scenario="interaction_only",
        noise_std=0.16,
        seed=0,
        memory_lag_b=5,
        nonlinearity_strength_b=0.0,
        interaction_strength_b=0.4,
        amplitude_scale_b=1.9,
        offset_b=0.35,
        noise_scale_b=1.75,
    )

    assert abs(float(np.mean(series_a)) - float(np.mean(series_b))) < 0.03
    assert abs(float(np.std(series_a)) - float(np.std(series_b))) < 0.01
    assert float(np.mean(np.abs(baseline_a - baseline_b))) < 1e-9
    assert float(np.mean(np.abs(series_a - series_b))) > 0.05
