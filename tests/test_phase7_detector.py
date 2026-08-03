import numpy as np
import pytest

from xjtu_sy_tcc.survival.detector import causal_aggregate, persistent_alarm, robust_normalize
from xjtu_sy_tcc.survival.spectral import kl_divergence, validate_distribution


def test_kl_identity_asymmetry_and_symmetry():
    p = np.array([0.2, 0.8])
    q = np.array([0.5, 0.5])
    assert kl_divergence(p, p) == pytest.approx(0)
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))
    assert kl_divergence(p, q, symmetric=True) == pytest.approx(kl_divergence(q, p, symmetric=True))


def test_distribution_rejects_non_normalized():
    with pytest.raises(ValueError):
        validate_distribution([1, 1])


def test_trailing_aggregation_has_no_future():
    values = np.array([1.0, 2.0, 100.0, 4.0])
    assert causal_aggregate(values, 3, "median")[1] == pytest.approx(1.5)
    changed = values.copy()
    changed[-1] = 999
    np.testing.assert_array_equal(
        causal_aggregate(values, 3)[:-1], causal_aggregate(changed, 3)[:-1]
    )


def test_persistent_alarm_and_impulse_rejection():
    assert (
        persistent_alarm(np.array([0, 0, 0, 5, 5, 5, 0, 0, 0, 0]), 3, 4, 3, 0, 2, 4)["status"]
        == "detected"
    )
    assert (
        persistent_alarm(np.array([0, 0, 0, 9, 0, 0, 0, 0]), 3, 4, 3, 0, 1, 4)["status"]
        == "not_detected"
    )


def test_robust_normalization_uses_prefix_only():
    a = robust_normalize([1, 2, 3, 100], 3)[0]
    b = robust_normalize([1, 2, 3, 999], 3)[0]
    np.testing.assert_array_equal(a[:3], b[:3])
