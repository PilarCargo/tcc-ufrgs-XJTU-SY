import pandas as pd

from xjtu_sy_tcc.survival.cohorts import fixed_horizon, rate_censor


def full():
    return pd.DataFrame(
        {
            "fold": [1] * 6,
            "subset": ["train"] * 2 + ["validation"] * 2 + ["test"] * 2,
            "bearing_id": ["A", "B", "C", "D", "E", "F"],
            "true_time_to_failure_minutes": [10.0, 100.0, 20.0, 200.0, 30.0, 300.0],
        }
    )


def test_fixed_horizon_censoring():
    result = fixed_horizon(full(), [60])
    assert result.duration_minutes.tolist() == [10, 60, 20, 60, 30, 60]
    assert result.event_observed.tolist() == [1, 0, 1, 0, 1, 0]


def test_rate_horizon_is_training_only_and_reused():
    result, manifest = rate_censor(full(), [0.5])
    horizons = manifest.training_only_horizon_minutes.unique()
    assert len(horizons) == 1
    assert horizons[0] == 55
    assert result.administrative_horizon_minutes.nunique() == 1
