import pandas as pd
import pytest

from xjtu_sy_tcc.consolidation.harmonization import (
    canonicalize,
    pairwise_support,
    strict_common_support,
)


def data(model, keys=(0, 1, 2)):
    rows = []
    for bearing in [f"B{i}" for i in range(15)]:
        condition = int(bearing[1:]) // 5 + 1
        for k in keys:
            rows.append(
                {
                    "condition_id": condition,
                    "bearing_id": bearing,
                    "acquisition_number": k,
                    "sequence_index": k,
                    "fold": int(bearing[1:]) % 5 + 1,
                    "experiment": model,
                    "model": "x",
                    "rul_minutes": 2 - k,
                    "prediction_raw_minutes": 1.0,
                    "prediction_non_negative_minutes": 1.0,
                    "elapsed_minutes": k,
                    "life_stage": "late",
                    "estimated_onset_region": "after estimated onset",
                }
            )
    return pd.DataFrame(rows)


def test_canonical_key_and_support_policies():
    result = canonicalize(data("a"), data("b", (1, 2)), ("a", "b"))
    assert result.canonical_key.nunique() == 45
    left, right = pairwise_support(result, "a", "b")
    assert len(left) == len(right) == 30
    strict = strict_common_support(result, ("a", "b"))
    assert strict.canonical_key.nunique() == 30


def test_duplicate_and_truth_mismatch_rejected():
    with pytest.raises(ValueError):
        canonicalize(pd.concat([data("a"), data("a")]), data("b"), ("a", "b"))
    changed = data("b")
    changed.loc[0, "rul_minutes"] = 99
    with pytest.raises(ValueError):
        canonicalize(data("a"), changed, ("a", "b"))
