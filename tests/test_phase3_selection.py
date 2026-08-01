"""Training-only feature-selection policy tests."""

from __future__ import annotations

import pandas as pd
import pytest

from xjtu_sy_tcc.degradation.selection import ALLOWED_METRICS, rank_features, validate_feature_names


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["horizontal_a", "vertical_a", "horizontal_b"],
            **{metric: [0.5, 0.5, 0.1] for metric in ALLOWED_METRICS},
        }
    )


def test_ranking_has_deterministic_tie_handling_and_count() -> None:
    weights = {metric: 1.0 for metric in ALLOWED_METRICS}
    ranked = rank_features(_scores(), weights, 2)
    assert ranked["feature"].tolist()[:2] == ["horizontal_a", "vertical_a"]
    assert ranked["selected"].sum() == 2


def test_invalid_weights_and_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        rank_features(_scores(), {metric: -1.0 for metric in ALLOWED_METRICS}, 2)
    with pytest.raises(ValueError):
        rank_features(_scores(), {metric: 1.0 for metric in ALLOWED_METRICS}, 4)


def test_metadata_target_and_unknown_features_are_rejected() -> None:
    available = ("horizontal_a", "vertical_a")
    validate_feature_names(available, available)
    for invalid in (("rul_minutes",), ("elapsed_minutes",), ("unknown",), ()):
        with pytest.raises(ValueError):
            validate_feature_names(invalid, available)
