"""Strict retrospective estimated-onset annotation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from xjtu_sy_tcc.rul.posthoc import annotate_posthoc


def _onsets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": index,
                "subset": "test",
                "bearing_id": f"B{index}",
                "estimated_onset_status": "detected",
                "estimated_onset_sequence_index": 2,
            }
            for index in range(1, 16)
        ]
    )


def test_before_and_after_onset_are_assigned_without_input_columns() -> None:
    predictions = pd.DataFrame(
        {
            "fold": [1] * 4,
            "bearing_id": ["B1"] * 4,
            "sequence_index": [0, 1, 2, 3],
            "experiment": ["e"] * 4,
        }
    )
    result = annotate_posthoc(predictions, _onsets(), (0.33, 0.66))
    assert result["estimated_onset_region"].tolist() == [
        "before estimated onset",
        "before estimated onset",
        "after estimated onset",
        "after estimated onset",
    ]
    assert "estimated_onset_sequence_index" not in result.columns


def test_non_test_or_duplicate_onset_scope_is_rejected() -> None:
    predictions = pd.DataFrame(
        {"fold": [1], "bearing_id": ["B1"], "sequence_index": [0], "experiment": ["e"]}
    )
    invalid = _onsets()
    invalid.loc[0, "subset"] = "train"
    with pytest.raises(ValueError, match="15 unique test-onset"):
        annotate_posthoc(predictions, invalid, (0.33, 0.66))
