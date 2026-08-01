"""Synthetic PELT candidate filtering and deterministic sensitivity tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xjtu_sy_tcc.config.phase3 import load_phase3_config
from xjtu_sy_tcc.degradation.pelt import (
    DetectionParameters,
    detect_estimated_onset,
    sensitivity_grid,
)


def _bearing(values: np.ndarray) -> pd.DataFrame:
    size = len(values)
    return pd.DataFrame(
        {
            "sequence_index": np.arange(size),
            "elapsed_minutes": np.arange(size, dtype=float),
            "rul_minutes": np.arange(size - 1, -1, -1, dtype=float),
            "baseline_centered_health_indicator": values,
        }
    )


def _parameters(penalty: float = 1.0, effect: float = 0.5) -> DetectionParameters:
    return DetectionParameters("l2", penalty, 5, 1, 3, 5, effect)


def test_single_break_is_detected_after_baseline() -> None:
    values = np.r_[np.zeros(30), np.ones(30) * 4]
    result = detect_estimated_onset(_bearing(values), _parameters(), 10)
    assert result["estimated_onset_status"] == "detected"
    assert 25 <= result["estimated_onset_sequence_index"] <= 35


def test_no_break_returns_not_detected() -> None:
    result = detect_estimated_onset(_bearing(np.zeros(60)), _parameters(100.0), 10)
    assert result["estimated_onset_status"] == "not_detected"


def test_one_sample_impulse_is_rejected_by_persistence() -> None:
    values = np.zeros(60)
    values[30] = 20
    result = detect_estimated_onset(_bearing(values), _parameters(0.1), 10)
    assert result["estimated_onset_status"] == "not_detected"


def test_multiple_breaks_selects_earliest_persistent_deterioration() -> None:
    values = np.r_[np.zeros(20), np.ones(20) * 2, np.ones(20) * 5]
    result = detect_estimated_onset(_bearing(values), _parameters(), 8)
    assert result["estimated_onset_status"] == "detected"
    assert result["estimated_onset_sequence_index"] <= 25


def test_baseline_break_and_minimum_effect_are_rejected() -> None:
    early = np.r_[np.zeros(5), np.ones(55) * 3]
    result = detect_estimated_onset(_bearing(early), _parameters(), 10)
    assert result["estimated_onset_status"] == "not_detected"
    small = np.r_[np.zeros(30), np.ones(30) * 0.01]
    result = detect_estimated_onset(_bearing(small), _parameters(effect=1e15), 10)
    assert result["estimated_onset_status"] == "not_detected"


def test_detection_is_deterministic() -> None:
    bearing = _bearing(np.r_[np.zeros(30), np.ones(30) * 4])
    assert detect_estimated_onset(bearing, _parameters(), 10) == detect_estimated_onset(
        bearing, _parameters(), 10
    )


def test_sensitivity_grid_is_deterministic_and_covers_both_costs() -> None:
    config = load_phase3_config(Path("configs/phase3.yaml"))
    first = sensitivity_grid(config)
    assert first == sensitivity_grid(config)
    assert {item.cost_model for item in first} == {"rbf", "l2"}
    assert len(first) > 1
