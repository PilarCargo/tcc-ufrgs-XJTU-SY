import numpy as np
import pandas as pd
import pytest

from xjtu_sy_tcc.consolidation.statistics import (
    bootstrap_difference,
    holm_adjust,
    paired_tests,
    summarize_difference,
)


def differences():
    return pd.DataFrame(
        {
            "condition_id": np.repeat([1, 2, 3], 5),
            "delta_mae": np.arange(-7, 8, dtype=float),
            "delta_rmse": np.arange(-6, 9, dtype=float),
            "exact_status": ["win"] * 7 + ["tie"] + ["loss"] * 7,
        }
    )


def test_summary_and_tests_use_fifteen_bearings():
    table = differences()
    summary = summarize_difference(table)
    assert summary["wins"] == 7 and summary["ties"] == 1 and summary["losses"] == 7
    tests = paired_tests(table, 1e-9)
    assert tests["paired_bearings"] == 15
    assert 0 <= tests["wilcoxon_p_raw"] <= 1
    assert tests["zero_differences"] == 1


def test_stratified_bootstrap_reproducible_and_finite():
    a = bootstrap_difference(differences(), 500, 42, 0.95, True)
    b = bootstrap_difference(differences(), 500, 42, 0.95, True)
    assert a == b
    assert a["effective_replicates"] == 500
    assert a["mean_ci_lower"] <= a["mean_ci_upper"]


def test_holm_monotonic_and_bounded():
    result = holm_adjust(pd.DataFrame({"wilcoxon_p_raw": [0.01, 0.04, 0.2]}))
    assert result.holm_p_adjusted.between(0, 1).all()
    assert result.holm_p_adjusted.iloc[0] == pytest.approx(0.03)
