"""Exact RUL metric, clipping, stage, and monotonicity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xjtu_sy_tcc.rul.metrics import (
    add_prediction_diagnostics,
    assign_life_stage,
    bearing_metrics,
    monotonicity_violations,
    regression_metrics,
)


def test_mae_rmse_r2_and_signed_error() -> None:
    actual = np.array([3.0, 2.0, 1.0, 0.0])
    predicted = np.array([2.0, 2.0, 2.0, 0.0])
    result = regression_metrics(actual, predicted)
    assert result["mae"] == pytest.approx(0.5)
    assert result["rmse"] == pytest.approx(np.sqrt(0.5))
    assert result["signed_error"] == pytest.approx(0.0)
    assert np.isfinite(result["r2"])
    assert np.isnan(regression_metrics(np.ones(3), np.ones(3))["r2"])


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment": ["e"] * 4,
            "model": ["m"] * 4,
            "fold": [1] * 4,
            "condition_id": [1] * 4,
            "bearing_id": ["B"] * 4,
            "acquisition_number": [1, 2, 3, 4],
            "sequence_index": [0, 1, 2, 3],
            "elapsed_minutes": [0.0, 1.0, 2.0, 3.0],
            "rul_minutes": [3.0, 2.0, 1.0, 0.0],
            "rotation_rpm": [2100.0] * 4,
            "radial_load_kn": [12.0] * 4,
        }
    )


def test_raw_predictions_are_preserved_and_only_lower_clipped() -> None:
    result = add_prediction_diagnostics(_prediction_frame(), np.array([10.0, 2.0, -1.0, -4.0]))
    assert result["prediction_raw_minutes"].tolist() == [10.0, 2.0, -1.0, -4.0]
    assert result["prediction_non_negative_minutes"].tolist() == [10.0, 2.0, 0.0, 0.0]
    assert result["prediction_non_negative_minutes"].max() == 10.0  # no lifetime upper clipping
    metrics = bearing_metrics(result)
    assert metrics.loc[0, "negative_raw_prediction_count"] == 2


def test_monotonicity_violation_count_rate_and_magnitude() -> None:
    result = monotonicity_violations(np.array([5.0, 4.0, 6.0, 3.0, 4.0]))
    assert result["monotonicity_violation_count"] == 2
    assert result["monotonicity_violation_magnitude"] == 3.0
    assert result["monotonicity_violation_rate"] == 0.5


def test_life_stage_assignment_is_retrospective_and_ordered() -> None:
    frame = pd.DataFrame({"bearing_id": ["B"] * 7, "sequence_index": range(7)})
    assert assign_life_stage(frame, (0.33, 0.66)).tolist() == [
        "early",
        "early",
        "intermediate",
        "intermediate",
        "late",
        "late",
        "late",
    ]
