"""Schema and alignment tests for Phase-2 feature validation."""

from __future__ import annotations

import pandas as pd

from xjtu_sy_tcc.features.validation import validate_feature_table


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition_id": [1, 1],
            "bearing_id": ["B1", "B1"],
            "acquisition_number": [1, 2],
            "sequence_index": [0, 1],
            "elapsed_minutes": [0.0, 1.0],
            "rul_minutes": [1.0, 0.0],
            "sampling_frequency_hz": [100.0, 100.0],
            "horizontal_rms": [1.0, 2.0],
        }
    )


def test_aligned_table_passes() -> None:
    table = _table()
    result = validate_feature_table(table, table.copy(), ("horizontal_rms",))
    assert result.status == "passed"
    assert result.issues == ()


def test_duplicate_acquisition_is_detected() -> None:
    metadata = _table()
    duplicated = pd.concat([metadata.iloc[[0]], metadata.iloc[[0]]], ignore_index=True)
    result = validate_feature_table(metadata, duplicated, ("horizontal_rms",))
    assert result.status == "failed"
    assert "duplicate_acquisitions" in {issue.code for issue in result.issues}
    assert "missing_acquisitions" in {issue.code for issue in result.issues}


def test_missing_acquisition_is_detected() -> None:
    metadata = _table()
    result = validate_feature_table(metadata, metadata.iloc[[0]].copy(), ("horizontal_rms",))
    assert "missing_acquisitions" in {issue.code for issue in result.issues}


def test_missing_feature_column_is_detected() -> None:
    table = _table().drop(columns="horizontal_rms")
    result = validate_feature_table(_table(), table, ("horizontal_rms",))
    assert "missing_columns" in {issue.code for issue in result.issues}
