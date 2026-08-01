"""Deterministic complete-bearing experimental folds."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class Fold:
    fold_id: int
    train_bearings: tuple[str, ...]
    validation_bearings: tuple[str, ...]
    test_bearings: tuple[str, ...]


def build_folds(features: pd.DataFrame) -> tuple[Fold, ...]:
    """Create five folds: position i test, i+1 validation, remaining positions train."""
    condition_bearings = {
        int(condition): tuple(sorted(group["bearing_id"].unique(), key=_bearing_position))
        for condition, group in features.groupby("condition_id", sort=True)
    }
    if len(condition_bearings) != 3 or any(
        len(items) != 5 for items in condition_bearings.values()
    ):
        raise ValueError("Expected exactly three conditions with five bearings each")
    folds = []
    for index in range(5):
        test = tuple(items[index] for items in condition_bearings.values())
        validation = tuple(items[(index + 1) % 5] for items in condition_bearings.values())
        train = tuple(
            bearing
            for items in condition_bearings.values()
            for position, bearing in enumerate(items)
            if position not in {index, (index + 1) % 5}
        )
        folds.append(Fold(index + 1, train, validation, test))
    validate_folds(tuple(folds), features)
    return tuple(folds)


def validate_folds(folds: tuple[Fold, ...], features: pd.DataFrame) -> None:
    """Fail on bearing/acquisition overlap, missing coverage, or condition imbalance."""
    expected = set(features["bearing_id"].unique())
    if len(folds) != 5:
        raise ValueError("Exactly five folds are required")
    test_counts = {bearing: 0 for bearing in expected}
    validation_counts = {bearing: 0 for bearing in expected}
    condition_by_bearing = features.groupby("bearing_id")["condition_id"].first().to_dict()
    for fold in folds:
        subsets = [set(fold.train_bearings), set(fold.validation_bearings), set(fold.test_bearings)]
        if (
            len(fold.train_bearings) != 9
            or len(fold.validation_bearings) != 3
            or len(fold.test_bearings) != 3
        ):
            raise ValueError(f"Fold {fold.fold_id} has invalid subset sizes")
        if any(left & right for i, left in enumerate(subsets) for right in subsets[i + 1 :]):
            raise ValueError(f"Fold {fold.fold_id} contains bearing overlap")
        if set.union(*subsets) != expected:
            raise ValueError(f"Fold {fold.fold_id} does not cover all bearings")
        for subset in subsets:
            if {condition_by_bearing[bearing] for bearing in subset} != {1, 2, 3}:
                raise ValueError(f"Fold {fold.fold_id} lacks condition representation")
        for bearing in fold.test_bearings:
            test_counts[bearing] += 1
        for bearing in fold.validation_bearings:
            validation_counts[bearing] += 1
    if set(test_counts.values()) != {1} or set(validation_counts.values()) != {1}:
        raise ValueError("Each bearing must occur once in test and once in validation")


def assign_subset(features: pd.DataFrame, fold: Fold) -> pd.Series:
    mapping = {bearing: "train" for bearing in fold.train_bearings}
    mapping.update({bearing: "validation" for bearing in fold.validation_bearings})
    mapping.update({bearing: "test" for bearing in fold.test_bearings})
    result = features["bearing_id"].map(mapping)
    if result.isna().any():
        raise ValueError("A feature row could not be assigned to a fold subset")
    return result


def fold_manifest(
    folds: tuple[Fold, ...], features: pd.DataFrame, config_hash: str, created_at: str
) -> dict[str, object]:
    counts = features.groupby("bearing_id").size().to_dict()
    condition = features.groupby("bearing_id")["condition_id"].first().to_dict()
    records = []
    for fold in folds:
        subsets = {}
        for name, bearings in (
            ("train", fold.train_bearings),
            ("validation", fold.validation_bearings),
            ("test", fold.test_bearings),
        ):
            subsets[name] = {
                "bearings": list(bearings),
                "acquisition_count": int(sum(counts[item] for item in bearings)),
                "by_condition": {
                    str(value): [item for item in bearings if condition[item] == value]
                    for value in (1, 2, 3)
                },
            }
        records.append({"fold_id": fold.fold_id, "subsets": subsets})
    return {
        "split_schema_version": 1,
        "created_at_utc": created_at,
        "configuration_hash": config_hash,
        "folds": records,
    }


def _bearing_position(value: str) -> int:
    try:
        return int(value.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid bearing identifier: {value}") from exc
