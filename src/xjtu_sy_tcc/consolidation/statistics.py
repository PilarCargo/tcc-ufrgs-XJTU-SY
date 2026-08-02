"""Bearing-level paired statistics, cluster bootstrap, and multiplicity control."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from xjtu_sy_tcc.rul.metrics import bearing_metrics


def per_bearing(data):
    return bearing_metrics(
        data.rename(columns={"source_phase": "phase"}) if "model" in data else data
    )


def paired_differences(candidate, reference, tie_tolerance, practical_tolerance):
    c = bearing_metrics(candidate)
    r = bearing_metrics(reference)
    keys = ["fold", "condition_id", "bearing_id"]
    merged = c.merge(r, on=keys, suffixes=("_candidate", "_reference"))
    merged["delta_mae"] = merged.mae_candidate - merged.mae_reference
    merged["delta_rmse"] = merged.rmse_candidate - merged.rmse_reference
    merged["exact_status"] = np.select(
        [merged.delta_mae < -tie_tolerance, merged.delta_mae > tie_tolerance],
        ["win", "loss"],
        default="tie",
    )
    merged["practical_status"] = np.select(
        [merged.delta_mae < -practical_tolerance, merged.delta_mae > practical_tolerance],
        ["practical win", "practical loss"],
        default="within practical tolerance",
    )
    return merged


def summarize_difference(table):
    d = table.delta_mae.to_numpy()
    return {
        "mean_delta_mae": float(d.mean()),
        "median_delta_mae": float(np.median(d)),
        "std_delta_mae": float(d.std(ddof=1)),
        "iqr_delta_mae": float(np.quantile(d, 0.75) - np.quantile(d, 0.25)),
        "minimum_delta_mae": float(d.min()),
        "maximum_delta_mae": float(d.max()),
        "wins": int((table.exact_status == "win").sum()),
        "ties": int((table.exact_status == "tie").sum()),
        "losses": int((table.exact_status == "loss").sum()),
        "win_proportion": float((table.exact_status == "win").mean()),
        "mean_delta_rmse": float(table.delta_rmse.mean()),
    }


def bootstrap_difference(table, replicates, seed, confidence, stratified=True):
    rng = np.random.default_rng(seed)
    values = []
    failed = 0
    groups = (
        [g.index.to_numpy() for _, g in table.groupby("condition_id")]
        if stratified
        else [table.index.to_numpy()]
    )
    for _ in range(replicates):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        sample = table.loc[indices]
        d = sample.delta_mae.to_numpy()
        row = (d.mean(), np.median(d), (d < 0).mean())
        if np.isfinite(row).all():
            values.append(row)
        else:
            failed += 1
    a = (1 - confidence) / 2
    array = np.asarray(values)
    return {
        "mean_ci_lower": float(np.quantile(array[:, 0], a)),
        "mean_ci_upper": float(np.quantile(array[:, 0], 1 - a)),
        "median_ci_lower": float(np.quantile(array[:, 1], a)),
        "median_ci_upper": float(np.quantile(array[:, 1], 1 - a)),
        "win_ci_lower": float(np.quantile(array[:, 2], a)),
        "win_ci_upper": float(np.quantile(array[:, 2], 1 - a)),
        "failed_replicates": failed,
        "effective_replicates": len(values),
        "stratified": stratified,
    }


def paired_tests(table, tie_tolerance):
    d = table.delta_mae.to_numpy()
    zeros = int((np.abs(d) <= tie_tolerance).sum())
    nonzero = d[np.abs(d) > tie_tolerance]
    if len(nonzero):
        w = wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
        sign = binomtest(int((nonzero < 0).sum()), len(nonzero), 0.5, alternative="two-sided")
        ranks = pd.Series(np.abs(nonzero)).rank().to_numpy()
        pos = ranks[nonzero > 0].sum()
        neg = ranks[nonzero < 0].sum()
        effect = float((pos - neg) / (pos + neg))
    else:
        w = type("R", (), {"statistic": 0.0, "pvalue": 1.0})()
        sign = type("R", (), {"pvalue": 1.0})()
        effect = 0.0
    return {
        "wilcoxon_statistic": float(w.statistic),
        "wilcoxon_p_raw": float(w.pvalue),
        "wilcoxon_zero_policy": "wilcox; zeros reported and excluded from ranks",
        "zero_differences": zeros,
        "paired_bearings": len(d),
        "sign_test_p_raw": float(sign.pvalue),
        "rank_biserial_effect": effect,
    }


def holm_adjust(table, p_column="wilcoxon_p_raw", alpha=0.05):
    result = table.copy()
    order = result[p_column].sort_values().index
    m = len(result)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * result.loc[index, p_column])
        adjusted[result.index.get_loc(index)] = min(running, 1.0)
    result["holm_p_adjusted"] = adjusted
    result["holm_reject"] = result.holm_p_adjusted < alpha
    result["family_hypothesis_count"] = m
    return result


def bootstrap_model_ranks(bearing_table, models, replicates, seed):
    rng = np.random.default_rng(seed)
    wins = {m: 0.0 for m in models}
    conditions = {c: g.bearing_id.unique() for c, g in bearing_table.groupby("condition_id")}
    pivot = bearing_table.pivot(index="bearing_id", columns="experiment", values="mae")
    for _ in range(replicates):
        sample = np.concatenate(
            [rng.choice(ids, len(ids), replace=True) for ids in conditions.values()]
        )
        scores = pivot.loc[sample, list(models)].mean()
        best = scores.min()
        tied = sorted(scores.index[np.isclose(scores, best)])
        wins[tied[0]] += 1
    return pd.DataFrame(
        {
            "experiment": list(models),
            "bootstrap_rank_first_probability": [wins[m] / replicates for m in models],
        }
    )


def bootstrap_model_metrics(bearing_table, models, replicates, seed, confidence):
    """Bootstrap macro errors with five bearing draws within each condition."""
    rng = np.random.default_rng(seed)
    groups = [group.index.to_numpy() for _, group in bearing_table.groupby("condition_id")]
    rows = []
    alpha = (1 - confidence) / 2
    for model in models:
        scoped = bearing_table[bearing_table.experiment == model]
        groups = [group.index.to_numpy() for _, group in scoped.groupby("condition_id")]
        values = []
        for _ in range(replicates):
            indices = np.concatenate(
                [rng.choice(group, len(group), replace=True) for group in groups]
            )
            sample = scoped.loc[indices]
            values.append((sample.mae.mean(), sample.rmse.mean()))
        array = np.asarray(values)
        rows.append(
            {
                "candidate": model,
                "reference": pd.NA,
                "metric": "model_macro_error",
                "mean_ci_lower": float(np.quantile(array[:, 0], alpha)),
                "mean_ci_upper": float(np.quantile(array[:, 0], 1 - alpha)),
                "median_ci_lower": float(np.quantile(array[:, 1], alpha)),
                "median_ci_upper": float(np.quantile(array[:, 1], 1 - alpha)),
                "failed_replicates": 0,
                "effective_replicates": replicates,
                "stratified": True,
            }
        )
    return pd.DataFrame(rows)
