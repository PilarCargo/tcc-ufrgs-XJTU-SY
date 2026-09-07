"""Post-selection statistics, RUL comparison, and descriptive survival summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sksurv.nonparametric import kaplan_meier_estimator

from xjtu_sy_tcc.consolidation.statistics import holm_adjust

PRIMARY_COMPARISONS = (
    ("cox_context_only", "landmark_km_baseline"),
    ("cox_causal_features", "landmark_km_baseline"),
    ("random_survival_forest", "landmark_km_baseline"),
    ("cox_causal_features", "cox_context_only"),
    ("random_survival_forest", "cox_context_only"),
    ("random_survival_forest", "cox_causal_features"),
)


def cox_ph_diagnostics(bundle, train: pd.DataFrame, permutations: int = 499) -> pd.DataFrame:
    """Cluster-aware Schoenfeld-residual PH diagnostic on the training bearings.

    Event-level residuals are collapsed to one vector per bearing before permutation, so repeated
    landmarks do not inflate the inferential sample size. The test is diagnostic, not a causal
    interpretation of coefficients.
    """
    features = list(bundle.features)
    x = train[features].to_numpy(float)
    if bundle.scaler is not None:
        x = bundle.scaler.transform(train[features])
    time = train.duration_minutes.to_numpy(float)
    event = train.event_observed.to_numpy(bool)
    risk = np.exp(np.clip(bundle.model.predict(x), -50, 50))
    residuals = []
    for index in np.flatnonzero(event):
        at_risk = time >= time[index]
        expected = np.average(x[at_risk], axis=0, weights=risk[at_risk])
        residuals.append(
            {
                "bearing_id": train.iloc[index].bearing_id,
                "log_event_time": np.log(time[index]),
                **{
                    feature: value
                    for feature, value in zip(features, x[index] - expected, strict=True)
                },
            }
        )
    events = pd.DataFrame(residuals)
    clustered = events.groupby("bearing_id", as_index=False).mean(numeric_only=True)
    n = len(clustered)
    rng = np.random.default_rng(1847)

    def absolute_correlation(left, right):
        if np.unique(left).size < 2 or np.unique(right).size < 2:
            return 0.0
        value = pd.Series(left).corr(pd.Series(right), method="spearman")
        return abs(float(value)) if np.isfinite(value) else 0.0

    observed = np.array(
        [absolute_correlation(clustered[f].to_numpy(), clustered.log_event_time) for f in features]
    )
    null = np.empty((permutations, len(features)))
    times = clustered.log_event_time.to_numpy()
    for iteration in range(permutations):
        permuted = rng.permutation(times)
        null[iteration] = [
            absolute_correlation(clustered[f].to_numpy(), permuted) for f in features
        ]
    feature_p = (1 + (null >= observed).sum(axis=0)) / (permutations + 1)
    global_stat = float(np.nanmax(observed))
    global_p = float((1 + (np.nanmax(null, axis=1) >= global_stat).sum()) / (permutations + 1))
    rows = [
        {
            "scope": "global",
            "feature": "__global__",
            "absolute_spearman": global_stat,
            "p_value": global_p,
            "evidence_against_ph": global_p < 0.05,
            "independent_bearings": n,
            "event_landmarks": len(events),
            "resampling_unit": "bearing",
            "method": "max-statistic bearing-level permutation of clustered Schoenfeld residuals",
        }
    ]
    rows.extend(
        {
            "scope": "feature",
            "feature": feature,
            "absolute_spearman": float(statistic),
            "p_value": float(p_value),
            "evidence_against_ph": p_value < 0.05,
            "independent_bearings": n,
            "event_landmarks": len(events),
            "resampling_unit": "bearing",
            "method": "bearing-level permutation of clustered Schoenfeld residuals",
        }
        for feature, statistic, p_value in zip(features, observed, feature_p, strict=True)
    )
    return pd.DataFrame(rows)


def paired_survival_comparisons(bearing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    differences, tests = [], []
    for scenario in sorted(bearing.scenario.unique()):
        scoped = bearing[bearing.scenario == scenario]
        for candidate, reference in PRIMARY_COMPARISONS:
            left = scoped[scoped.model == candidate]
            right = scoped[scoped.model == reference]
            merged = left.merge(
                right,
                on=["fold", "condition_id", "bearing_id"],
                suffixes=("_candidate", "_reference"),
            )
            merged = merged.dropna(
                subset=["integrated_brier_score_candidate", "integrated_brier_score_reference"]
            )
            delta = (
                merged.integrated_brier_score_candidate - merged.integrated_brier_score_reference
            )
            for row, value in zip(merged.itertuples(), delta, strict=True):
                differences.append(
                    {
                        "scenario": scenario,
                        "candidate": candidate,
                        "reference": reference,
                        "fold": row.fold,
                        "condition_id": row.condition_id,
                        "bearing_id": row.bearing_id,
                        "delta_integrated_brier_score": value,
                        "status": "win" if value < 0 else "loss" if value > 0 else "tie",
                    }
                )
            if len(delta):
                result = wilcoxon(delta, zero_method="wilcox", method="auto")
                ranks = pd.Series(np.abs(delta)).rank().to_numpy()
                positive = ranks[delta.to_numpy() > 0].sum()
                negative = ranks[delta.to_numpy() < 0].sum()
                effect = (positive - negative) / (positive + negative) if positive + negative else 0
            else:
                result = type("Result", (), {"statistic": np.nan, "pvalue": 1.0})()
                effect = np.nan
            tests.append(
                {
                    "scenario": scenario,
                    "candidate": candidate,
                    "reference": reference,
                    "paired_bearings": len(delta),
                    "mean_delta_ibs": float(delta.mean()) if len(delta) else np.nan,
                    "median_delta_ibs": float(delta.median()) if len(delta) else np.nan,
                    "wins": int((delta < 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "losses": int((delta > 0).sum()),
                    "wilcoxon_statistic": float(result.statistic),
                    "wilcoxon_p_raw": float(result.pvalue),
                    "rank_biserial_effect": float(effect),
                }
            )
    test_table = pd.DataFrame(tests)
    adjusted = []
    for _, group in test_table.groupby("scenario"):
        adjusted.append(holm_adjust(group.reset_index(drop=True)))
    return pd.DataFrame(differences), pd.concat(adjusted, ignore_index=True)


def kaplan_meier_onset(onset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves, summaries = [], []
    for fold in sorted(onset.fold.unique()):
        train = onset[(onset.fold == fold) & (onset.subset == "train")]
        for condition, group in [("pooled", train), *list(train.groupby("condition_id"))]:
            condition = str(condition)
            times, survival, ci = kaplan_meier_estimator(
                group.event_observed.astype(bool),
                group.duration_minutes,
                conf_type="log-log",
            )
            median_positions = np.flatnonzero(survival <= 0.5)
            summaries.append(
                {
                    "fold": fold,
                    "condition": condition,
                    "bearing_count": group.bearing_id.nunique(),
                    "median_post_onset_survival_minutes": float(times[median_positions[0]])
                    if len(median_positions)
                    else np.nan,
                }
            )
            curves.extend(
                {
                    "fold": fold,
                    "condition": condition,
                    "evaluation_time": time,
                    "survival_probability": probability,
                    "confidence_lower": ci[0, index],
                    "confidence_upper": ci[1, index],
                }
                for index, (time, probability) in enumerate(zip(times, survival, strict=True))
            )
    return pd.DataFrame(curves), pd.DataFrame(summaries)


def matched_rul_comparison(
    survival: pd.DataFrame,
    phase4: pd.DataFrame,
    phase5: pd.DataFrame,
) -> pd.DataFrame:
    survival = survival[survival.scenario == "full_event"].copy()
    existing = pd.concat(
        [
            phase4[
                phase4.experiment.isin(
                    ["dummy_median", "time_only_ridge", "selected_features_plus_time_ridge"]
                )
            ],
            phase5[
                phase5.experiment.isin(
                    ["selected_features_lstm_k1_ablation", "selected_features_plus_time_lstm"]
                )
            ],
        ],
        ignore_index=True,
    )
    keys = ["fold", "bearing_id", "acquisition_number"]
    rows = []
    for survival_model, landmarks in survival.groupby("model"):
        available = landmarks.dropna(subset=["predicted_median_survival_minutes"])
        for experiment, predictions in existing.groupby("experiment"):
            merged = available.merge(
                predictions[keys + ["prediction_non_negative_minutes"]],
                on=keys,
                how="inner",
                validate="one_to_one",
            )
            for bearing, group in merged.groupby("bearing_id"):
                truth = group.true_time_to_failure_minutes.to_numpy(float)
                survival_error = group.predicted_median_survival_minutes.to_numpy(float) - truth
                rul_error = group.prediction_non_negative_minutes.to_numpy(float) - truth
                rows.append(
                    {
                        "survival_model": survival_model,
                        "rul_experiment": experiment,
                        "bearing_id": bearing,
                        "support_size": len(group),
                        "survival_median_mae": float(np.mean(np.abs(survival_error))),
                        "survival_median_rmse": float(np.sqrt(np.mean(survival_error**2))),
                        "existing_rul_mae": float(np.mean(np.abs(rul_error))),
                        "existing_rul_rmse": float(np.sqrt(np.mean(rul_error**2))),
                        "delta_mae": float(
                            np.mean(np.abs(survival_error)) - np.mean(np.abs(rul_error))
                        ),
                    }
                )
    return pd.DataFrame(rows)


def feature_consensus(coefficients: pd.DataFrame) -> pd.DataFrame:
    data = coefficients.copy()
    data["absolute_coefficient"] = data.standardized_coefficient.abs()
    data["feature_family"] = data.feature.str.replace(
        r"^(horizontal|vertical)_", "", regex=True
    ).str.replace(r"__.*$", "_causal_descriptor", regex=True)
    return (
        data.groupby("feature_family", as_index=False)
        .agg(
            fold_occurrences=("fold", "nunique"),
            mean_absolute_coefficient=("absolute_coefficient", "mean"),
            median_hazard_ratio=("hazard_ratio", "median"),
        )
        .sort_values(["fold_occurrences", "mean_absolute_coefficient"], ascending=False)
    )
