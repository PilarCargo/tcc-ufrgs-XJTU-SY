"""Unit tests for time-domain feature definitions and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from xjtu_sy_tcc.features.time_domain import TIME_FEATURE_NAMES, time_features


def test_time_feature_formulas() -> None:
    signal = np.array([-2.0, -1.0, 1.0, 2.0])
    result = time_features(signal)
    assert tuple(result) == TIME_FEATURE_NAMES
    assert result["mean"] == pytest.approx(0.0)
    assert result["variance"] == pytest.approx(2.5)
    assert result["std"] == pytest.approx(np.sqrt(2.5))
    assert result["rms"] == pytest.approx(np.sqrt(2.5))
    assert result["minimum"] == -2.0
    assert result["maximum"] == 2.0
    assert result["max_abs"] == 2.0
    assert result["peak_to_peak"] == 4.0
    assert result["energy"] == 10.0
    assert result["crest_factor"] == pytest.approx(2 / np.sqrt(2.5))
    assert result["shape_factor"] == pytest.approx(np.sqrt(2.5) / 1.5)
    assert result["impulse_factor"] == pytest.approx(2 / 1.5)
    assert result["clearance_factor"] == pytest.approx(2 / np.mean(np.sqrt(np.abs(signal))) ** 2)
    assert result["skewness"] == pytest.approx(0.0, abs=1e-15)
    assert result["kurtosis"] == pytest.approx(1.36)


def test_impulse_signal_has_expected_energy_and_peak() -> None:
    signal = np.zeros(16)
    signal[3] = 4.0
    result = time_features(signal)
    assert result["energy"] == 16.0
    assert result["max_abs"] == 4.0
    assert result["impulse_factor"] == 16.0


def test_zero_signal_preserves_undefined_results_as_nan() -> None:
    result = time_features(np.zeros(16, dtype=np.float32))
    assert result["energy"] == 0.0
    for name in (
        "crest_factor",
        "shape_factor",
        "impulse_factor",
        "clearance_factor",
        "skewness",
        "kurtosis",
    ):
        assert np.isnan(result[name])


def test_constant_nonzero_signal_handles_zero_variance() -> None:
    result = time_features(np.full(16, 3.0))
    assert result["std"] == 0.0
    assert result["rms"] == 3.0
    assert result["crest_factor"] == 1.0
    assert result["shape_factor"] == 1.0
    assert np.isnan(result["skewness"])
    assert np.isnan(result["kurtosis"])


@pytest.mark.parametrize(
    "signal",
    [
        np.array([]),
        np.array([1.0]),
        np.ones((2, 2)),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array(["a", "b"]),
    ],
)
def test_invalid_sample_arrays_are_rejected(signal: np.ndarray) -> None:
    with pytest.raises((ValueError, TypeError)):
        time_features(signal)
