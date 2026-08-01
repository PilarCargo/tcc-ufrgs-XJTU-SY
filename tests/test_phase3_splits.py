"""Deterministic leakage validation for complete-bearing folds."""

from __future__ import annotations

import pandas as pd

from xjtu_sy_tcc.degradation.splits import assign_subset, build_folds, validate_folds


def _features() -> pd.DataFrame:
    rows = []
    for condition in range(1, 4):
        for position in range(1, 6):
            for acquisition in range(3):
                rows.append(
                    {
                        "condition_id": condition,
                        "bearing_id": f"Bearing{condition}_{position}",
                        "acquisition_number": acquisition + 1,
                    }
                )
    return pd.DataFrame(rows)


def test_five_folds_have_exact_deterministic_membership() -> None:
    features = _features()
    folds = build_folds(features)
    assert len(folds) == 5
    assert folds[0].test_bearings == ("Bearing1_1", "Bearing2_1", "Bearing3_1")
    assert folds[0].validation_bearings == ("Bearing1_2", "Bearing2_2", "Bearing3_2")
    assert len(folds[0].train_bearings) == 9
    assert build_folds(features) == folds


def test_folds_have_no_bearing_or_acquisition_overlap() -> None:
    features = _features()
    folds = build_folds(features)
    validate_folds(folds, features)
    for fold in folds:
        subsets = assign_subset(features, fold)
        keyed = features.assign(subset=subsets)
        assert not keyed.duplicated(["bearing_id", "acquisition_number"]).any()
        assert keyed.groupby("bearing_id")["subset"].nunique().eq(1).all()
        assert set(keyed[keyed["subset"] == "test"]["condition_id"]) == {1, 2, 3}
        assert set(keyed[keyed["subset"] == "validation"]["condition_id"]) == {1, 2, 3}


def test_every_bearing_occurs_once_in_test_and_validation() -> None:
    folds = build_folds(_features())
    test = [bearing for fold in folds for bearing in fold.test_bearings]
    validation = [bearing for fold in folds for bearing in fold.validation_bearings]
    assert len(test) == len(set(test)) == 15
    assert len(validation) == len(set(validation)) == 15
