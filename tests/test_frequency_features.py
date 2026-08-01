"""Unit tests for Welch spectral feature definitions."""

from __future__ import annotations

import numpy as np
import pytest

from xjtu_sy_tcc.config.features import FrequencyBand
from xjtu_sy_tcc.features.frequency_domain import (
    FREQUENCY_FEATURE_NAMES,
    compute_welch_psd,
    frequency_features,
)


def test_sinusoid_has_known_dominant_frequency_and_band_power() -> None:
    fs = 1024.0
    time = np.arange(4096) / fs
    signal = np.sin(2 * np.pi * 128.0 * time)
    bands = (FrequencyBand("below", 0.0, 64.0), FrequencyBand("target", 100.0, 150.0))
    result = frequency_features(signal, fs, 1024, 512, bands)
    assert tuple(result) == FREQUENCY_FEATURE_NAMES + ("band_power_below", "band_power_target")
    assert result["dominant_frequency_hz"] == pytest.approx(128.0, abs=1.0)
    assert result["spectral_centroid_hz"] == pytest.approx(128.0, abs=1.0)
    assert result["rms_frequency_hz"] == pytest.approx(128.0, abs=1.0)
    assert result["total_spectral_power"] == pytest.approx(0.5, rel=0.02)
    assert result["band_power_target"] > result["band_power_below"]
    assert 0.0 <= result["spectral_entropy"] <= 1.0


def test_zero_signal_reports_undefined_distribution_without_fabricating_zero() -> None:
    result = frequency_features(np.zeros(64), 64.0, 32, 16)
    assert result["dominant_frequency_hz"] == 0.0
    assert result["dominant_spectral_power"] == 0.0
    assert result["total_spectral_power"] == 0.0
    for name in (
        "spectral_centroid_hz",
        "spectral_spread_hz",
        "rms_frequency_hz",
        "spectral_entropy",
    ):
        assert np.isnan(result[name])


def test_deterministic_frequency_execution() -> None:
    rng = np.random.default_rng(7)
    signal = rng.normal(size=2048).astype(np.float32)
    assert frequency_features(signal, 1000.0, 256, 128) == frequency_features(
        signal, 1000.0, 256, 128
    )


def test_frequency_feature_formulas_match_welch_distribution() -> None:
    rng = np.random.default_rng(12)
    signal = rng.normal(size=1024)
    frequencies, density = compute_welch_psd(signal, 512.0, 256, 128)
    result = frequency_features(signal, 512.0, 256, 128)
    probabilities = density / density.sum()
    centroid = np.sum(frequencies * probabilities)
    dominant = int(np.argmax(density))
    positive = probabilities > 0

    assert result["dominant_frequency_hz"] == frequencies[dominant]
    assert result["dominant_spectral_power"] == density[dominant]
    assert result["spectral_centroid_hz"] == pytest.approx(centroid)
    assert result["spectral_spread_hz"] == pytest.approx(
        np.sqrt(np.sum((frequencies - centroid) ** 2 * probabilities))
    )
    assert result["rms_frequency_hz"] == pytest.approx(
        np.sqrt(np.sum(frequencies**2 * probabilities))
    )
    assert result["spectral_entropy"] == pytest.approx(
        -np.sum(probabilities[positive] * np.log(probabilities[positive]))
        / np.log(probabilities.size)
    )
    assert result["total_spectral_power"] == pytest.approx(np.trapezoid(density, frequencies))


@pytest.mark.parametrize("signal", [np.array([1.0]), np.array([1.0, np.nan]), np.ones((2, 2))])
def test_invalid_frequency_arrays_are_rejected(signal: np.ndarray) -> None:
    with pytest.raises(ValueError):
        frequency_features(signal, 100.0, 16, 8)


def test_band_above_nyquist_is_rejected() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        frequency_features(np.ones(64), 100.0, 16, 8, (FrequencyBand("bad", 40.0, 60.0),))
