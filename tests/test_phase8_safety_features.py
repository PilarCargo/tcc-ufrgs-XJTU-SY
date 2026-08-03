import numpy as np
import pandas as pd
import pytest

from xjtu_sy_tcc.survival_models.features import (
    balanced_indices,
    filter_and_rank,
    structured_target,
)
from xjtu_sy_tcc.survival_models.safety import (
    EvaluationTruthGuard,
    validate_model_columns,
    validate_outcome,
)


def survival_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bearing_id": ["A"] * 4 + ["B"] * 2,
            "duration_minutes": [4, 3, 2, 1, 2, 1],
            "event_observed": [1, 1, 1, 1, 0, 1],
            "signal": [0, 1, 2, 3, 0, 2],
            "duplicate": [0, 1, 2, 3, 0, 2],
            "constant": [1] * 6,
            "correlated": [0, 2, 4, 6, 0, 4],
        }
    )


def test_evaluation_truth_guard(tmp_path):
    path = tmp_path / "truth.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(path)
    guard = EvaluationTruthGuard(path)
    with pytest.raises(RuntimeError):
        guard.load()
    guard.freeze_predictions()
    assert guard.load().x.tolist() == [1]


def test_forbidden_columns_are_rejected():
    with pytest.raises(ValueError, match="rul_minutes"):
        validate_model_columns(["signal", "rul_minutes"], {"rul_minutes"})


def test_structured_survival_target():
    target = structured_target(survival_frame())
    assert target.dtype.names == ("event", "time")
    assert target["event"].dtype == bool


@pytest.mark.parametrize(("column", "value"), [("duration_minutes", 0), ("event_observed", 2)])
def test_invalid_survival_outcome(column, value):
    data = survival_frame()
    data.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_outcome(data)


def test_balancing_is_equal_and_deterministic():
    data = survival_frame()
    first = balanced_indices(data, 9)
    second = balanced_indices(data, 9)
    assert np.array_equal(first, second)
    assert data.loc[first].groupby("bearing_id").size().nunique() == 1


def test_filtering_removes_constant_duplicate_and_correlation():
    data = survival_frame()
    filtering, groups, ranking, retained = filter_and_rank(
        data,
        ["signal", "duplicate", "constant", "correlated"],
        0.999,
        0.95,
    )
    status = filtering.set_index("feature").status.to_dict()
    assert status["constant"] == "constant"
    assert "exact_duplicate" in status.values()
    assert len(groups) >= 1
    assert ranking["feature"].tolist() == retained


def test_no_test_outcome_is_needed_for_training_ranking():
    train = survival_frame()
    _, _, first, _ = filter_and_rank(train, ["signal"], 0.999, 0.95)
    changed_test = train.copy()
    changed_test["unused_test_truth"] = np.arange(len(train))
    _, _, second, _ = filter_and_rank(changed_test, ["signal"], 0.999, 0.95)
    pd.testing.assert_frame_equal(first, second)
