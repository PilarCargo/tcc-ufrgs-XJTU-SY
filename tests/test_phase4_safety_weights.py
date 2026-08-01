"""Frozen protocol, predictive-input safety, and balanced-weight tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xjtu_sy_tcc.rul.safety import (
    load_frozen_folds,
    load_selected_features,
    phase3_health_predictive_policy,
    validate_input_columns,
)
from xjtu_sy_tcc.rul.weights import bearing_balanced_weights, effective_sample_size


def _features() -> pd.DataFrame:
    rows = []
    for condition in range(1, 4):
        for position in range(1, 6):
            for acquisition in range(position + 1):
                rows.append(
                    {
                        "condition_id": condition,
                        "bearing_id": f"Bearing{condition}_{position}",
                        "acquisition_number": acquisition + 1,
                        "horizontal_rms": float(acquisition),
                    }
                )
    return pd.DataFrame(rows)


def test_frozen_manifest_reuse_and_hash_rejection(tmp_path: Path) -> None:
    source = Path("outputs/splits/folds.json")
    features = _features()
    # Membership, rather than acquisition counts in the manifest, drives overlap validation.
    import hashlib

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = json.loads(source.read_text())
    folds, _ = load_frozen_folds(source, digest, payload["configuration_hash"], features)
    assert len(folds) == 5
    changed = tmp_path / "folds.json"
    changed.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        load_frozen_folds(changed, digest, payload["configuration_hash"], features)


def test_selected_features_match_training_bearings() -> None:
    source = Path("outputs/splits/folds.json")
    payload = json.loads(source.read_text())
    import hashlib

    folds, _ = load_frozen_folds(
        source,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        payload["configuration_hash"],
        _features(),
    )
    selected = load_selected_features(
        Path("outputs/prognostics/selected_features_by_fold.json"),
        folds,
        tuple(pd.read_parquet("outputs/features/features.parquet").columns),
    )
    assert set(selected) == {1, 2, 3, 4, 5}
    assert all(len(value) == 10 for value in selected.values())


@pytest.mark.parametrize(
    "columns",
    [
        ("rul_minutes",),
        ("bearing_id",),
        ("life_fraction",),
        ("estimated_onset_elapsed_minutes",),
        ("smoothed_health_indicator",),
    ],
)
def test_target_future_and_identifier_columns_are_rejected(columns: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        validate_input_columns(columns, columns, allow_time=True)


def test_time_is_allowed_only_for_explicit_time_experiments() -> None:
    with pytest.raises(ValueError):
        validate_input_columns(("elapsed_minutes",), ("elapsed_minutes",), allow_time=False)
    validate_input_columns(("elapsed_minutes",), ("elapsed_minutes",), allow_time=True)


def test_weights_equalize_bearings_and_have_mean_one() -> None:
    training = pd.DataFrame({"bearing_id": ["A"] * 2 + ["B"] * 5 + ["C"] * 9})
    first = bearing_balanced_weights(training)
    second = bearing_balanced_weights(training)
    assert np.array_equal(first, second)
    assert first.mean() == pytest.approx(1.0)
    totals = pd.Series(first).groupby(training["bearing_id"]).sum()
    assert totals.nunique() == 1
    assert 0 < effective_sample_size(first) <= len(first)


def test_health_indicator_policy_rejects_full_trajectory_use() -> None:
    policy = phase3_health_predictive_policy()
    assert policy["smoothing_is_causal"] is True
    assert policy["full_trajectory_prediction_allowed"] is False
    assert policy["earliest_allowed_use"] == "after baseline_acquisitions"
