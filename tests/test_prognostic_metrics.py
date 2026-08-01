"""Formula-level tests for Phase 3 prognostic metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xjtu_sy_tcc.degradation.metrics import (
    aggregate_feature_scores,
    bearing_feature_scores,
    prognosability_score,
    trendability_score,
)


def _trajectories(values: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for index, trajectory in enumerate(values, start=1):
        for sequence, value in enumerate(trajectory):
            rows.append(
                {
                    "condition_id": 1,
                    "bearing_id": f"B{index}",
                    "sequence_index": sequence,
                    "feature": float(value),
                }
            )
    return pd.DataFrame(rows)


def test_increasing_decreasing_constant_and_noisy_metrics() -> None:
    frame = _trajectories(
        [
            np.arange(10),
            np.arange(9, -1, -1),
            np.ones(10),
            np.array([0, 2, 1, 3, 2, 4, 3, 5, 4, 6]),
        ]
    )
    scores = bearing_feature_scores(frame, ("feature",)).set_index("bearing_id")
    assert scores.loc["B1", "monotonicity"] == 1.0
    assert scores.loc["B1", "trend_sign"] == 1
    assert scores.loc["B1", "spearman"] == pytest.approx(1.0)
    assert scores.loc["B2", "trend_sign"] == -1
    assert scores.loc["B2", "spearman"] == pytest.approx(-1.0)
    assert scores.loc["B3", "monotonicity"] == 0.0
    assert scores.loc["B3", "spearman"] == 0.0
    assert scores.loc["B4", "monotonicity"] < 1.0


def test_trendability_interpolation_identical_and_opposing() -> None:
    identical = _trajectories([np.arange(5), np.linspace(0, 4, 9)])
    opposing = _trajectories([np.arange(5), np.arange(4, -1, -1)])
    assert trendability_score(identical, "feature", 31) == pytest.approx((1.0, 1))
    assert trendability_score(opposing, "feature", 31) == pytest.approx((1.0, 1))


def test_prognosability_edge_cases_do_not_divide_by_zero() -> None:
    constant = _trajectories([np.ones(10), np.ones(12)])
    assert prognosability_score(constant, "feature", 0.1, 0.1) == 1.0
    inconsistent = _trajectories([np.arange(10), np.arange(10) * 10])
    score = prognosability_score(inconsistent, "feature", 0.2, 0.2)
    assert 0.0 <= score <= 1.0


def test_aggregate_reports_pair_count_and_finite_scores() -> None:
    frame = _trajectories([np.arange(10), np.arange(12), np.arange(8)])
    bearing = bearing_feature_scores(frame, ("feature",))
    result = aggregate_feature_scores(frame, bearing, ("feature",), 21, 0.1, 0.1)
    assert result.loc[0, "trendability_valid_pairs"] == 3
    assert result.loc[0, "finite"]
