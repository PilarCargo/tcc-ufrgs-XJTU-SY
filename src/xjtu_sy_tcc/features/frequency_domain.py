"""Welch-PSD vibration features."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from xjtu_sy_tcc.config.features import FrequencyBand

FREQUENCY_FEATURE_NAMES = (
    "dominant_frequency_hz",
    "dominant_spectral_power",
    "spectral_centroid_hz",
    "spectral_spread_hz",
    "rms_frequency_hz",
    "spectral_entropy",
    "total_spectral_power",
)


def compute_welch_psd(
    signal: np.ndarray, sampling_frequency_hz: float, nperseg: int, overlap: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be one-dimensional with at least two finite samples")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    segment = min(int(nperseg), values.size)
    actual_overlap = min(int(overlap), segment - 1)
    if segment < 2 or nperseg <= 0 or overlap < 0 or overlap >= nperseg:
        raise ValueError("Welch parameters require nperseg >= 2 and 0 <= overlap < nperseg")
    frequencies, density = welch(
        values.astype(np.float32, copy=False),
        fs=sampling_frequency_hz,
        window="hann",
        nperseg=segment,
        noverlap=actual_overlap,
        detrend="constant",
        scaling="density",
        average="mean",
    )
    return frequencies.astype(np.float64), density.astype(np.float64)


def frequency_features(
    signal: np.ndarray,
    sampling_frequency_hz: float,
    nperseg: int,
    overlap: int,
    bands: tuple[FrequencyBand, ...] = (),
) -> dict[str, float]:
    """Calculate Welch-derived descriptors and integrated band powers."""
    frequencies, density = compute_welch_psd(signal, sampling_frequency_hz, nperseg, overlap)
    total = float(np.trapezoid(density, frequencies))
    density_sum = float(np.sum(density, dtype=np.float64))
    dominant = int(np.argmax(density))
    if density_sum == 0.0:
        centroid = spread = rms_frequency = entropy = float("nan")
    else:
        probabilities = density / density_sum
        centroid = float(np.sum(frequencies * probabilities, dtype=np.float64))
        spread = float(
            np.sqrt(np.sum(np.square(frequencies - centroid) * probabilities, dtype=np.float64))
        )
        rms_frequency = float(
            np.sqrt(np.sum(np.square(frequencies) * probabilities, dtype=np.float64))
        )
        positive = probabilities > 0
        normalizer = np.log(probabilities.size) if probabilities.size > 1 else 1.0
        entropy = float(
            -np.sum(probabilities[positive] * np.log(probabilities[positive])) / normalizer
        )
    result = {
        "dominant_frequency_hz": float(frequencies[dominant]),
        "dominant_spectral_power": float(density[dominant]),
        "spectral_centroid_hz": centroid,
        "spectral_spread_hz": spread,
        "rms_frequency_hz": rms_frequency,
        "spectral_entropy": entropy,
        "total_spectral_power": total,
    }
    nyquist = sampling_frequency_hz / 2.0
    for band in bands:
        if band.upper_hz > nyquist:
            raise ValueError(f"Frequency band {band.name!r} exceeds the Nyquist frequency")
        mask = (frequencies >= band.lower_hz) & (frequencies <= band.upper_hz)
        result[f"band_power_{band.name}"] = (
            float(np.trapezoid(density[mask], frequencies[mask]))
            if np.count_nonzero(mask) >= 2
            else 0.0
        )
    return result
