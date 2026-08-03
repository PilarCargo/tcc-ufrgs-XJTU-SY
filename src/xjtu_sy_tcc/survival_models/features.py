"""Training-only filtering, correlation pruning, and survival ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sksurv.metrics import concordance_index_censored


def balanced_indices(data: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Deterministically downsample each bearing to the smallest bearing count."""
    count = int(data.groupby("bearing_id").size().min())
    rng = np.random.default_rng(seed)
    indices = []
    for _, group in data.groupby("bearing_id", sort=True):
        chosen = rng.choice(group.index.to_numpy(), count, replace=False)
        indices.extend(sorted(chosen))
    return np.asarray(indices)


def structured_target(data: pd.DataFrame) -> np.ndarray:
    return np.array(
        list(
            zip(data.event_observed.astype(bool), data.duration_minutes.astype(float), strict=True)
        ),
        dtype=[("event", "?"), ("time", "<f8")],
    )


def univariate_survival_score(values: pd.Series, outcome: pd.DataFrame) -> float:
    x = values.to_numpy(float)
    if not np.isfinite(x).all() or np.ptp(x) == 0:
        return 0.5
    event = outcome.event_observed.to_numpy(bool)
    time = outcome.duration_minutes.to_numpy(float)
    try:
        direct = concordance_index_censored(event, time, x)[0]
        reverse = concordance_index_censored(event, time, -x)[0]
        return float(max(direct, reverse))
    except ValueError:
        return 0.5


def filter_and_rank(
    train: pd.DataFrame,
    candidates: list[str],
    near_constant_threshold: float,
    correlation_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Filter using training rows only and resolve correlated pairs by survival relevance."""
    records = []
    retained = []
    for name in sorted(candidates):
        reason = "retained"
        if name not in train or not pd.api.types.is_numeric_dtype(train[name]):
            reason = "non_numeric_or_missing"
        elif not np.isfinite(train[name].to_numpy(float)).all():
            reason = "invalid_value"
        elif train[name].nunique(dropna=False) <= 1:
            reason = "constant"
        elif (
            train[name].value_counts(normalize=True, dropna=False).max() >= near_constant_threshold
        ):
            reason = "near_constant"
        if reason == "retained":
            retained.append(name)
        records.append({"feature": name, "status": reason})
    duplicate_of = {}
    fingerprints = {}
    for name in retained:
        fingerprint = pd.util.hash_pandas_object(train[name], index=False).sum()
        key = (str(train[name].dtype), int(fingerprint))
        if key in fingerprints and train[name].equals(train[fingerprints[key]]):
            duplicate_of[name] = fingerprints[key]
        else:
            fingerprints[key] = name
    retained = [name for name in retained if name not in duplicate_of]
    for record in records:
        if record["feature"] in duplicate_of:
            record["status"] = "exact_duplicate"
            record["related_feature"] = duplicate_of[record["feature"]]
    scores = {name: univariate_survival_score(train[name], train) for name in retained}
    order = sorted(retained, key=lambda name: (-scores[name], name))
    kept = []
    groups = []
    for name in order:
        rejected = None
        for selected in kept:
            rho = spearmanr(train[name], train[selected]).statistic
            if np.isfinite(rho) and abs(rho) >= correlation_threshold:
                rejected = selected
                groups.append(
                    {
                        "kept_feature": selected,
                        "removed_feature": name,
                        "absolute_spearman": abs(float(rho)),
                    }
                )
                break
        if rejected is None:
            kept.append(name)
        else:
            for record in records:
                if record["feature"] == name:
                    record["status"] = "correlation_pruned"
                    record["related_feature"] = rejected
    ranking = pd.DataFrame(
        [
            {"feature": name, "survival_relevance": scores[name], "rank": i + 1}
            for i, name in enumerate(kept)
        ]
    )
    return pd.DataFrame(records), pd.DataFrame(groups), ranking, kept
