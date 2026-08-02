import numpy as np
import pandas as pd
import pytest

from xjtu_sy_tcc.temporal.sequences import build_sequences, sequence_coverage


def frame():
    rows = []
    for bearing in ("A", "B"):
        for i in range(5):
            rows.append(
                {
                    "bearing_id": bearing,
                    "sequence_index": i,
                    "f": float(i + (bearing == "B") * 10),
                    "elapsed_minutes": i,
                    "rotation_rpm": 1,
                    "radial_load_kn": 2,
                    "rul_minutes": 4 - i,
                }
            )
    return pd.DataFrame(rows)


def test_causal_chronological_sequences_do_not_cross_bearings():
    result = build_sequences(frame(), ["f"], 3)
    assert result.features.shape == (6, 3, 1)
    assert result.targets.tolist() == [2, 1, 0, 2, 1, 0]
    assert result.features[0, :, 0].tolist() == [0, 1, 2]
    assert result.features[3, :, 0].tolist() == [10, 11, 12]


def test_prefix_policy_and_coverage():
    c = sequence_coverage(frame(), 3, 1, "test")
    assert set(c.dropped_prefix) == {2}
    assert set(c.eligible_sequences) == {3}


def test_deterministic_and_balanced_weights():
    a = build_sequences(frame(), ["f"], 2)
    b = build_sequences(frame(), ["f"], 2)
    np.testing.assert_array_equal(a.features, b.features)
    assert a.weights.mean() == pytest.approx(1)
    assert pd.Series(a.weights).groupby(a.metadata.bearing_id).sum().nunique() == 1


def test_invalid_continuity_and_length():
    bad = frame().drop(index=2)
    with pytest.raises(ValueError):
        build_sequences(bad, ["f"], 2)
    with pytest.raises(ValueError):
        build_sequences(frame(), ["f"], 0)
