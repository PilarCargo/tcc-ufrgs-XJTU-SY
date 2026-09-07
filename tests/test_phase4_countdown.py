import numpy as np
import pandas as pd
import pytest

from xjtu_sy_tcc.data.metadata import calculate_rul_minutes
from xjtu_sy_tcc.rul.models import ConditionLifetimeCountdown


def training_frame(lifetimes=(4.0, 6.0, 10.0, 12.0)):
    rows = []
    for position, lifetime in enumerate(lifetimes):
        condition = 1 if position < 2 else 2
        for elapsed in np.arange(0.0, lifetime + 1.0):
            rows.append(
                {
                    "condition_id": condition,
                    "bearing_id": f"B{position}",
                    "elapsed_minutes": elapsed,
                    "rul_minutes": lifetime - elapsed,
                    "total_acquisitions": int(lifetime + 1),
                    "normalized_life_fraction": elapsed / lifetime,
                }
            )
    return pd.DataFrame(rows)


def test_countdown_matches_zero_based_rul_lifetime_convention():
    total, interval = 5, 2.0
    lifetime = (total - 1) * interval
    assert calculate_rul_minutes(total, 0, interval) == lifetime
    assert calculate_rul_minutes(total, total - 1, interval) == 0.0
    training = pd.DataFrame(
        {
            "condition_id": [1] * total,
            "bearing_id": ["A"] * total,
            "elapsed_minutes": np.arange(total) * interval,
            "rul_minutes": [calculate_rul_minutes(total, i, interval) for i in range(total)],
        }
    )
    fitted = ConditionLifetimeCountdown.fit(training)
    assert fitted.median_lifetime_by_condition == {1: lifetime}


def test_countdown_is_train_only_fold_specific_deterministic_and_clipped():
    fold_one = ConditionLifetimeCountdown.fit(training_frame())
    fold_two = ConditionLifetimeCountdown.fit(training_frame((8.0, 10.0, 10.0, 12.0)))
    evaluation = pd.DataFrame(
        {
            "condition_id": [1, 1, 1, 2],
            "elapsed_minutes": [0.0, 5.0, 20.0, 2.0],
            "rul_minutes": [999.0, 888.0, 777.0, 666.0],
            "total_acquisitions": [1000, 1000, 1000, 1000],
            "normalized_life_fraction": [0.0, 0.1, 0.9, 0.2],
        }
    )
    first = fold_one.predict(evaluation)
    changed_forbidden = evaluation.assign(
        rul_minutes=-1.0, total_acquisitions=2, normalized_life_fraction=0.5
    )
    assert np.array_equal(first, fold_one.predict(changed_forbidden))
    assert np.array_equal(first, fold_one.predict(evaluation))
    assert first.tolist() == [5.0, 0.0, 0.0, 9.0]
    assert fold_two.predict(evaluation)[0] == 9.0


def test_countdown_rejects_missing_training_condition_without_global_fallback():
    fitted = ConditionLifetimeCountdown.fit(training_frame())
    with pytest.raises(ValueError, match="No training lifetime"):
        fitted.predict(pd.DataFrame({"condition_id": [3], "elapsed_minutes": [0.0]}))


def test_countdown_rejects_inconsistent_training_endpoint():
    training = training_frame()
    training.loc[training.bearing_id.eq("B0") & training.elapsed_minutes.eq(0), "rul_minutes"] = 99
    with pytest.raises(ValueError, match="inconsistent"):
        ConditionLifetimeCountdown.fit(training)
