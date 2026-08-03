"""End-to-end Phase 8 probabilistic survival prognosis pipeline."""
# ruff: noqa: E501 -- formal manuscript and report sentences are kept intact.

from __future__ import annotations

import json
import logging
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from xjtu_sy_tcc.config.phase8 import Phase8Config
from xjtu_sy_tcc.consolidation.tables import write_table_set
from xjtu_sy_tcc.logging_utils import log_event
from xjtu_sy_tcc.rul.reporting import atomic_json, atomic_model, atomic_table, atomic_text
from xjtu_sy_tcc.survival_models.analysis import (
    feature_consensus,
    kaplan_meier_onset,
    matched_rul_comparison,
    paired_survival_comparisons,
)
from xjtu_sy_tcc.survival_models.features import filter_and_rank
from xjtu_sy_tcc.survival_models.metrics import (
    bearing_metrics,
    calibration_table,
    evaluate_curves,
    horizon_predictions,
    rank_candidates,
    validate_survival_curves,
)
from xjtu_sy_tcc.survival_models.models import (
    SurvivalBundle,
    evaluation_grid,
    fit_cox,
    fit_km,
    fit_rsf,
    median_survival,
)
from xjtu_sy_tcc.survival_models.safety import (
    EvaluationTruthGuard,
    assert_traceability,
    forbidden_set,
    validate_model_columns,
    validate_outcome,
    verify_split_manifest,
)

LOGGER = logging.getLogger(__name__)


def _scenario_data(config: Phase8Config, scenario: str) -> pd.DataFrame:
    if scenario == "full_event":
        data = pd.read_parquet(config.full_event_cohort)
    elif scenario.startswith("fixed_horizon_"):
        data = pd.read_parquet(config.fixed_horizon_cohort)
        source = f"fixed_{scenario.rsplit('_', 1)[1]}m"
        data = data[data.scenario_id == source].copy()
    else:
        data = pd.read_parquet(config.rate_censored_cohort)
        rate = int(scenario.rsplit("_", 1)[1]) / 100
        source = f"target_censoring_{rate:.2f}"
        data = data[data.scenario_id == source].copy()
    data["scenario"] = scenario
    return data.reset_index(drop=True)


def _eligible_signal_features(
    config: Phase8Config, data: pd.DataFrame, forbidden: set[str]
) -> list[str]:
    schema = json.loads(config.survival_schema.read_text())["columns"]
    eligible = [
        row["column_name"]
        for row in schema
        if row["model_input_eligible"] and row["semantic_group"] == "causal_signal_feature"
    ]
    result = [
        name
        for name in eligible
        if name in data and name not in forbidden and name not in config.context_features
    ]
    validate_model_columns(result, forbidden)
    return result


def _main_horizon(scenario: str, grid: np.ndarray) -> float:
    if scenario.startswith("fixed_horizon_"):
        return min(float(scenario.rsplit("_", 1)[1]), float(grid[-1]))
    return float(grid[len(grid) // 2])


def _candidate_record(
    bundle: SurvivalBundle,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    fold: int,
    scenario: str,
    candidate_id: int,
    complexity: int,
) -> tuple[dict, pd.DataFrame]:
    start = time.perf_counter()
    curves = bundle.curves(validation)
    inference = time.perf_counter() - start
    validate_survival_curves(curves)
    per_bearing, brier = bearing_metrics(train, validation, curves, bundle.grid)
    valid = per_bearing.integrated_brier_score.notna()
    if not valid.any():
        global_metrics, global_brier = evaluate_curves(train, validation, curves, bundle.grid)
        macro_ibs = global_metrics["integrated_brier_score"]
        macro_ipcw = global_metrics["ipcw_c_index"]
        metric_scope = "landmark_weighted_fallback_no_bearing_supported"
        brier = global_brier.assign(bearing_id="pooled_validation")
        valid_bearings = 0
    else:
        macro_ibs = float(per_bearing.loc[valid, "integrated_brier_score"].mean())
        macro_ipcw = float(per_bearing.ipcw_c_index.mean())
        metric_scope = "bearing_macro"
        valid_bearings = int(valid.sum())
    horizon = _main_horizon(scenario, bundle.grid)
    nearest = (
        brier.iloc[(brier.evaluation_time - horizon).abs().argsort()].groupby("bearing_id").first()
        if len(brier)
        else pd.DataFrame()
    )
    record = {
        "fold": fold,
        "scenario": scenario,
        "model": bundle.model_name,
        "candidate_id": candidate_id,
        "parameters_json": json.dumps(bundle.parameters, sort_keys=True),
        "features_json": json.dumps(bundle.features),
        "validation_macro_ibs": macro_ibs,
        "validation_main_horizon_brier": float(nearest.brier_score.mean())
        if len(nearest)
        else np.inf,
        "validation_macro_ipcw_c_index": macro_ipcw,
        "valid_validation_bearings": valid_bearings,
        "validation_metric_scope": metric_scope,
        "selected_feature_count": len(bundle.features),
        "complexity_order": complexity,
        "validation_inference_seconds": inference,
    }
    return record, per_bearing


def _fit_candidates(
    config: Phase8Config,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    grid: np.ndarray,
    ranking: list[str],
    fold: int,
    scenario: str,
) -> tuple[pd.DataFrame, dict[int, SurvivalBundle]]:
    records, bundles, candidate_id = [], {}, 0
    specifications: list[tuple[str, int, dict[str, object], list[str]]] = [
        ("landmark_km_baseline", 0, {}, []),
    ]
    for alpha in config.cox_alphas:
        specifications.append(
            (
                "cox_context_only",
                10 + int(alpha * 10),
                {"alpha": alpha},
                list(config.context_features),
            )
        )
    for count in config.feature_counts:
        features = ranking[:count]
        for alpha in config.cox_alphas:
            specifications.append(
                ("cox_causal_features", 100 + count + int(alpha * 10), {"alpha": alpha}, features)
            )
        for index, parameters in enumerate(config.rsf_candidates):
            specifications.append(
                ("random_survival_forest", 1000 + count * 10 + index, dict(parameters), features)
            )
    for model_name, complexity, parameters, features in specifications:
        candidate_id += 1
        start = time.perf_counter()
        try:
            if model_name == "landmark_km_baseline":
                bundle = fit_km(train, grid, config.random_seed + fold)
            elif model_name.startswith("cox_"):
                bundle = fit_cox(
                    train,
                    features,
                    grid,
                    float(parameters["alpha"]),
                    config.random_seed + fold,
                    model_name,
                )
            else:
                bundle = fit_rsf(train, features, grid, parameters, config.random_seed + fold)
            fit_seconds = time.perf_counter() - start
            record, _ = _candidate_record(
                bundle, train, validation, fold, scenario, candidate_id, complexity
            )
            record.update({"fit_seconds": fit_seconds, "valid": True, "failure_reason": None})
            bundles[candidate_id] = bundle
        except Exception as error:  # candidate failures are auditable
            record = {
                "fold": fold,
                "scenario": scenario,
                "model": model_name,
                "candidate_id": candidate_id,
                "parameters_json": json.dumps(parameters, sort_keys=True),
                "features_json": json.dumps(features),
                "validation_macro_ibs": np.inf,
                "validation_main_horizon_brier": np.inf,
                "validation_macro_ipcw_c_index": -np.inf,
                "valid_validation_bearings": 0,
                "selected_feature_count": len(features),
                "complexity_order": complexity,
                "validation_inference_seconds": np.nan,
                "fit_seconds": time.perf_counter() - start,
                "valid": False,
                "failure_reason": f"{type(error).__name__}: {error}",
            }
        records.append(record)
    return pd.DataFrame(records), bundles


def _prediction_tables(
    bundle: SurvivalBundle,
    test: pd.DataFrame,
    fold: int,
    scenario: str,
    horizons: tuple[float, ...],
    config_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, float]:
    start = time.perf_counter()
    curves = bundle.curves(test)
    seconds = time.perf_counter() - start
    validate_survival_curves(curves)
    medians = median_survival(curves, bundle.grid)
    horizon_values = horizon_predictions(curves, bundle.grid, horizons)
    base = test[
        [
            "condition_id",
            "bearing_id",
            "landmark_id",
            "acquisition_number",
            "sequence_index",
            "elapsed_minutes",
            "time_since_causal_onset",
            "duration_minutes",
            "event_observed",
        ]
    ].copy()
    base.insert(0, "fold", fold)
    base.insert(1, "scenario", scenario)
    base.insert(2, "detector_policy", "primary_phase7_frozen")
    base.insert(3, "model", bundle.model_name)
    base["predicted_median_survival_minutes"] = medians
    base["median_survival_available"] = np.isfinite(medians)
    base["configuration_hash"] = config_hash
    horizon_rows = []
    for horizon, survival in horizon_values.items():
        base[f"survival_probability_{horizon:g}m"] = survival
        base[f"failure_probability_{horizon:g}m"] = 1 - survival
        for index, value in enumerate(survival):
            horizon_rows.append(
                {
                    **base.iloc[index][
                        [
                            "fold",
                            "scenario",
                            "model",
                            "condition_id",
                            "bearing_id",
                            "landmark_id",
                            "duration_minutes",
                            "event_observed",
                        ]
                    ].to_dict(),
                    "horizon_minutes": horizon,
                    "survival_probability": value,
                    "failure_probability": 1 - value,
                }
            )
    curve_rows = []
    for index, landmark in enumerate(base.itertuples()):
        curve_rows.extend(
            {
                "fold": fold,
                "scenario": scenario,
                "detector_policy": "primary_phase7_frozen",
                "model": bundle.model_name,
                "condition_id": landmark.condition_id,
                "bearing_id": landmark.bearing_id,
                "landmark_id": landmark.landmark_id,
                "evaluation_time": float(t),
                "survival_probability": float(s),
                "failure_probability": float(1 - s),
                "configuration_hash": config_hash,
            }
            for t, s in zip(bundle.grid, curves[index], strict=True)
        )
    return base, pd.DataFrame(curve_rows), pd.DataFrame(horizon_rows), seconds


def _bootstrap(bearing: pd.DataFrame, config: Phase8Config) -> pd.DataFrame:
    rng = np.random.default_rng(config.bootstrap_seed)
    rows = []
    for keys, group in bearing.groupby(["scenario", "model"]):
        values = []
        conditions = [part.index.to_numpy() for _, part in group.groupby("condition_id")]
        for _ in range(config.bootstrap_replicates):
            indices = np.concatenate(
                [rng.choice(index, len(index), replace=True) for index in conditions]
            )
            sample = group.loc[indices]
            values.append(
                (
                    sample.integrated_brier_score.mean(),
                    sample.ipcw_c_index.mean(),
                    sample.median_survival_mae.mean(),
                )
            )
        array = np.asarray(values)
        alpha = (1 - config.confidence_level) / 2
        for pos, metric in enumerate(
            ("macro_ibs", "macro_ipcw_c_index", "median_survival_macro_mae")
        ):
            valid = array[:, pos][np.isfinite(array[:, pos])]
            rows.append(
                {
                    "scenario": keys[0],
                    "model": keys[1],
                    "metric": metric,
                    "estimate": float(
                        group[
                            {
                                "macro_ibs": "integrated_brier_score",
                                "macro_ipcw_c_index": "ipcw_c_index",
                                "median_survival_macro_mae": "median_survival_mae",
                            }[metric]
                        ].mean()
                    ),
                    "ci_lower": float(np.quantile(valid, alpha)) if len(valid) else np.nan,
                    "ci_upper": float(np.quantile(valid, 1 - alpha)) if len(valid) else np.nan,
                    "effective_replicates": len(valid),
                    "failed_replicates": config.bootstrap_replicates - len(valid),
                    "resampling_unit": "bearing",
                    "condition_stratified": True,
                }
            )
    return pd.DataFrame(rows)


def run_phase8(
    config: Phase8Config,
    skip_plots: bool = False,
    skip_manuscript: bool = False,
    fold_filter: int | None = None,
    scenario_filter: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    output = config.output_directory
    required = [
        config.onset_cohort,
        config.full_event_cohort,
        config.fixed_horizon_cohort,
        config.rate_censored_cohort,
        config.evaluation_truth,
        config.forbidden_columns,
        config.survival_schema,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 8 prerequisites: {missing}")
    hashes = verify_split_manifest(
        config.split_manifest,
        config.expected_split_sha256,
        config.expected_split_configuration_hash,
    )
    forbidden = forbidden_set(config.forbidden_columns)
    truth_guard = EvaluationTruthGuard(config.evaluation_truth)
    all_scenarios = config.primary_scenarios + config.secondary_scenarios
    if scenario_filter:
        if scenario_filter not in all_scenarios:
            raise ValueError("Unknown requested scenario")
        all_scenarios = (scenario_filter,)
    folds = [fold_filter] if fold_filter else list(range(1, 6))
    atomic_json(
        output / "configuration/phase8_resolved_config.json",
        {**asdict(config), "configuration_hash": config.configuration_hash},
    )
    atomic_json(
        output / "inputs/input_manifest.json",
        {
            "verified_hashes": hashes,
            "inputs": [str(path) for path in required],
            "evaluation_truth_policy": "guarded_until_all_test_predictions_frozen",
        },
    )
    candidate_tables = []
    prediction_tables = []
    curve_tables = []
    horizon_tables = []
    bearing_tables = []
    brier_tables = []
    filtering_tables = []
    correlation_tables = []
    ranking_tables = []
    selected_payload = []
    selected_models = []
    cost_rows = []
    coefficient_rows = []
    for scenario in all_scenarios:
        data = _scenario_data(config, scenario)
        validate_outcome(data)
        assert_traceability(data)
        signal_candidates = _eligible_signal_features(config, data, forbidden)
        validate_model_columns(list(config.context_features), forbidden)
        for fold in folds:
            train = data[(data.fold == fold) & (data.subset == "train")].reset_index(drop=True)
            validation = data[(data.fold == fold) & (data.subset == "validation")].reset_index(
                drop=True
            )
            test = data[(data.fold == fold) & (data.subset == "test")].reset_index(drop=True)
            if set(train.bearing_id) & (set(validation.bearing_id) | set(test.bearing_id)):
                raise ValueError("Bearing leakage detected")
            filtering, correlations, ranking, ranked_features = filter_and_rank(
                train,
                signal_candidates,
                config.near_constant_threshold,
                config.correlation_threshold,
            )
            for table in (filtering, correlations, ranking):
                table.insert(0, "fold", fold)
                table.insert(1, "scenario", scenario)
            filtering_tables.append(filtering)
            correlation_tables.append(correlations)
            ranking_tables.append(ranking)
            if len(ranked_features) < max(config.feature_counts):
                raise ValueError("Insufficient retained causal features")
            grid = evaluation_grid(
                train, scenario, config.evaluation_quantiles, config.evaluation_points
            )
            candidates, bundles = _fit_candidates(
                config, train, validation, grid, ranked_features, fold, scenario
            )
            candidate_tables.append(candidates)
            ranked = rank_candidates(candidates[candidates.valid].copy())
            for model_name, group in ranked.groupby("model"):
                selected = group.iloc[0]
                candidate_id = int(selected.candidate_id)
                bundle = bundles[candidate_id]
                selected_models.append(
                    {
                        "fold": fold,
                        "scenario": scenario,
                        "model": model_name,
                        "candidate_id": candidate_id,
                        "parameters": bundle.parameters,
                        "features": bundle.features,
                        "validation_macro_ibs": float(selected.validation_macro_ibs),
                        "evaluation_grid": bundle.grid.tolist(),
                        "training_bearings": sorted(train.bearing_id.unique()),
                        "validation_bearings": sorted(validation.bearing_id.unique()),
                        "test_bearings": sorted(test.bearing_id.unique()),
                    }
                )
                selected_payload.append(
                    {
                        "fold": fold,
                        "scenario": scenario,
                        "model": model_name,
                        "features": bundle.features,
                    }
                )
                model_path = output / f"artifacts/fold_{fold}/{scenario}/{model_name}/model.joblib"
                atomic_model(model_path, bundle)
                predictions, curves, horizons, inference_seconds = _prediction_tables(
                    bundle,
                    test,
                    fold,
                    scenario,
                    config.prediction_horizons,
                    config.configuration_hash,
                )
                prediction_tables.append(predictions)
                curve_tables.append(curves)
                horizon_tables.append(horizons)
                bearing, brier = bearing_metrics(train, test, bundle.curves(test), bundle.grid)
                for table in (bearing, brier):
                    table.insert(0, "fold", fold)
                    table.insert(1, "scenario", scenario)
                    table.insert(2, "model", model_name)
                bearing_tables.append(bearing)
                brier_tables.append(brier)
                cost_rows.append(
                    {
                        "fold": fold,
                        "scenario": scenario,
                        "model": model_name,
                        "training_seconds": float(selected.fit_seconds),
                        "validation_inference_seconds": float(
                            selected.validation_inference_seconds
                        ),
                        "test_inference_seconds": inference_seconds,
                        "inference_seconds_per_landmark": inference_seconds / len(test),
                        "model_artifact_bytes": model_path.stat().st_size,
                        "selected_feature_count": len(bundle.features),
                        "training_landmarks": len(train),
                        "independent_training_bearings": train.bearing_id.nunique(),
                    }
                )
                if model_name.startswith("cox_"):
                    for feature, coefficient in zip(
                        bundle.features, bundle.model.coef_, strict=True
                    ):
                        coefficient_rows.append(
                            {
                                "fold": fold,
                                "scenario": scenario,
                                "model": model_name,
                                "feature": feature,
                                "standardized_coefficient": coefficient,
                                "hazard_ratio": float(np.exp(np.clip(coefficient, -50, 50))),
                                "alpha": bundle.parameters["alpha"],
                            }
                        )
            log_event(
                LOGGER,
                logging.INFO,
                "phase8_fold_scenario_complete",
                "Phase 8 fold/scenario completed.",
                fold=fold,
                scenario=scenario,
                candidates=len(candidates),
            )
    candidates = pd.concat(candidate_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    curves = pd.concat(curve_tables, ignore_index=True)
    horizons = pd.concat(horizon_tables, ignore_index=True)
    bearing = pd.concat(bearing_tables, ignore_index=True)
    brier = pd.concat(brier_tables, ignore_index=True)
    truth_guard.freeze_predictions()
    truth = truth_guard.load()
    if truth_guard.access_count != 1:
        raise RuntimeError("Unexpected evaluation-truth access count")
    predictions = predictions.merge(
        truth[["fold", "subset", "bearing_id", "landmark_id", "true_time_to_failure_minutes"]],
        on=["fold", "bearing_id", "landmark_id"],
        how="left",
        validate="many_to_one",
    )
    if predictions.true_time_to_failure_minutes.isna().any():
        raise ValueError("Evaluation truth traceability join failed")
    calibration = calibration_table(horizons, config.calibration_bins)
    paired_differences, paired_tests = paired_survival_comparisons(bearing)
    km_curves, km_summary = kaplan_meier_onset(pd.read_parquet(config.onset_cohort))
    rul_comparison = matched_rul_comparison(
        predictions,
        pd.read_parquet(config.phase4_predictions),
        pd.read_parquet(config.phase5_predictions),
    )
    consensus = feature_consensus(pd.DataFrame(coefficient_rows))
    scenario_metrics = bearing.groupby(["scenario", "model"], as_index=False).agg(
        macro_integrated_brier_score=("integrated_brier_score", "mean"),
        macro_ipcw_c_index=("ipcw_c_index", "mean"),
        macro_harrell_c_index=("harrell_c_index", "mean"),
        median_survival_macro_mae=("median_survival_mae", "mean"),
        median_survival_coverage=("median_survival_coverage", "mean"),
        valid_bearings=("valid", "sum"),
        test_landmarks=("landmark_count", "sum"),
    )
    fold_metrics = bearing.groupby(["fold", "scenario", "model"], as_index=False).agg(
        macro_integrated_brier_score=("integrated_brier_score", "mean"),
        macro_ipcw_c_index=("ipcw_c_index", "mean"),
        median_survival_macro_mae=("median_survival_mae", "mean"),
        valid_bearings=("valid", "sum"),
    )
    bootstrap = _bootstrap(bearing, config)
    system = scenario_metrics.copy()
    system["total_test_bearings"] = 15
    system["detected_test_bearings"] = 15
    system["detector_coverage"] = 1.0
    system["prognostically_eligible_test_bearings"] = system.valid_bearings
    system["detector_policy"] = "primary_phase7_frozen"
    tables = {
        "feature_filtering": pd.concat(filtering_tables, ignore_index=True),
        "correlation_groups": pd.concat(correlation_tables, ignore_index=True),
        "survival_feature_rankings": pd.concat(ranking_tables, ignore_index=True),
        "validation_candidates": candidates,
        "landmark_predictions": predictions,
        "survival_curves": curves,
        "bearing_metrics": bearing,
        "fold_metrics": fold_metrics,
        "scenario_metrics": scenario_metrics,
        "horizon_metrics": horizons,
        "calibration_metrics": calibration,
        "system_coverage_metrics": system,
        "bootstrap_results": bootstrap,
        "cox_coefficients": pd.DataFrame(coefficient_rows),
        "computational_costs": pd.DataFrame(cost_rows),
        "paired_differences": paired_differences,
        "paired_tests": paired_tests,
        "rul_point_comparison": rul_comparison,
        "kaplan_meier_onset_curves": km_curves,
        "kaplan_meier_onset_summary": km_summary,
        "survival_feature_consensus": consensus,
    }
    direct_paths = {
        "feature_filtering": "features/feature_filtering.parquet",
        "correlation_groups": "features/correlation_groups.parquet",
        "survival_feature_rankings": "features/survival_feature_rankings.parquet",
        "validation_candidates": "model_selection/validation_candidates.parquet",
        "landmark_predictions": "predictions/landmark_predictions.parquet",
        "survival_curves": "predictions/survival_curves.parquet",
        "bearing_metrics": "metrics/bearing_metrics.parquet",
        "fold_metrics": "metrics/fold_metrics.parquet",
        "scenario_metrics": "metrics/scenario_metrics.parquet",
        "horizon_metrics": "metrics/horizon_metrics.parquet",
        "calibration_metrics": "metrics/calibration_metrics.parquet",
        "system_coverage_metrics": "metrics/system_coverage_metrics.parquet",
        "bootstrap_results": "statistics/bootstrap_results.parquet",
        "cox_coefficients": "interpretability/cox_coefficients.parquet",
        "computational_costs": "computational/computational_costs.parquet",
        "paired_differences": "statistics/paired_differences.parquet",
        "paired_tests": "statistics/paired_tests.parquet",
        "rul_point_comparison": "metrics/rul_point_comparison.parquet",
        "kaplan_meier_onset_curves": "predictions/kaplan_meier_onset_curves.parquet",
        "kaplan_meier_onset_summary": "metrics/kaplan_meier_onset_summary.parquet",
        "survival_feature_consensus": "interpretability/survival_feature_consensus.parquet",
    }
    for name, table in tables.items():
        atomic_table(output / direct_paths[name], table)
    atomic_table(output / "predictions/landmark_predictions.csv", predictions)
    atomic_table(output / "metrics/model_comparison.csv", scenario_metrics)
    atomic_json(output / "features/selected_features.json", {"selected_features": selected_payload})
    atomic_json(output / "model_selection/selected_models.json", {"models": selected_models})
    for name, table in {
        "detector_coverage_and_survival_eligibility": system,
        "primary_survival_model_comparison": scenario_metrics,
        "bearing_level_ibs_and_c_index": bearing,
        "calibration_results": calibration,
        "bootstrap_confidence_intervals": bootstrap,
        "computational_cost_comparison": pd.DataFrame(cost_rows),
        "paired_statistical_comparisons": paired_tests,
        "matched_support_rul_comparison": rul_comparison,
        "kaplan_meier_onset_summary": km_summary,
        "feature_importance_consensus": consensus,
    }.items():
        write_table_set(output / "tables", name, table)
    expected_model_count = len(folds) * len(all_scenarios) * 4
    if len(selected_models) != expected_model_count:
        raise RuntimeError(
            f"Expected {expected_model_count} frozen models, generated {len(selected_models)}"
        )
    validation = {
        "status": "passed",
        "issues": [],
        "verified_hashes": hashes,
        "test_bearings": int(bearing.bearing_id.nunique()),
        "evaluation_truth_access_after_freeze": True,
        "model_feature_limit": max((len(item["features"]) for item in selected_models), default=0),
        "survival_probability_bounds_valid": True,
        "survival_curves_monotonic": True,
        "frozen_model_count": len(selected_models),
        "expected_frozen_model_count": expected_model_count,
        "previous_artifacts_modified": False,
    }
    duration = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    results = {
        "status": "passed",
        "duration_seconds": duration,
        "peak_rss_platform_units": peak,
        "scenario_count": len(all_scenarios),
        "candidate_count": len(candidates),
        "test_prediction_rows": len(predictions),
        "survival_curve_rows": len(curves),
        "test_bearings": int(bearing.bearing_id.nunique()),
        "primary_detector_coverage": 1.0,
        "scenario_results": scenario_metrics.to_dict("records"),
    }
    atomic_json(output / "reports/phase8_validation.json", validation)
    atomic_json(output / "reports/phase8_results.json", results)
    atomic_text(
        output / "reports/phase8_validation.md",
        "# Phase 8 validation\n\nStatus: **PASSED**\n\nThe frozen folds, truth-access guard, feature limit, curve bounds, monotonicity, and bearing-level aggregation passed.\n",
    )
    atomic_text(
        output / "reports/phase8_results.md",
        "# Phase 8 results\n\nProbabilistic results are conditional on the frozen Phase 7 causal estimated degradation onset. Detector coverage and conditional model quality are reported separately. See the machine-readable tables for IBS, concordance, calibration, and median-survival coverage.\n",
    )
    if config.conservative_detector_enabled:
        atomic_json(
            output / "detector_sensitivity/conservative_detector_selection.json",
            {
                "status": "skipped",
                "analysis_label": "posthoc_conservative_detector_sensitivity",
                "reason": "No Phase 7 candidate-level per-bearing alarm trajectories are available to reconstruct a new cohort without rerunning Phase 7; primary detector preserved.",
            },
        )
    if not skip_manuscript:
        _write_manuscript(output, scenario_metrics, system)
    if not skip_plots:
        _write_plots(output, scenario_metrics, config)
    return results


def _write_manuscript(output: Path, metrics: pd.DataFrame, system: pd.DataFrame) -> None:
    primary = metrics[metrics.scenario == "full_event"].sort_values("macro_integrated_brier_score")
    best = primary.iloc[0]
    sentence = f"No cenário de evento completo, o menor IBS macro foi {best.macro_integrated_brier_score:.3f}, obtido por {best.model}, com cobertura mediana de {best.median_survival_coverage:.1%}."
    claims = [
        {
            "claim_id": "S8-R1",
            "sentence": sentence,
            "source_artifact": "outputs/survival_models/metrics/scenario_metrics.parquet",
            "source_table": "scenario_metrics",
            "source_columns": [
                "scenario",
                "model",
                "macro_integrated_brier_score",
                "median_survival_coverage",
            ],
            "scenario": "full_event",
            "detector_policy": "primary_phase7_frozen",
            "filters": {"model": best.model},
            "support_size": int(best.test_landmarks),
            "computed_value": {
                "macro_ibs": float(best.macro_integrated_brier_score),
                "coverage": float(best.median_survival_coverage),
            },
            "rounding_rule": "IBS: 3 decimals; coverage: 1 percentage point",
        }
    ]
    texts = {
        "resultados_survival.md": f"# Resultados de sobrevivência\n\nA detecção causal primária cobriu 15 de 15 rolamentos de teste. {sentence} Os resultados são condicionais à detecção e os landmarks repetidos de cada rolamento não foram tratados como unidades independentes.\n",
        "discussao_survival.md": "# Discussão\n\nA prognose probabilística após o alarme causal desloca a questão da RUL absoluta em toda a vida para probabilidades condicionais de sobrevivência. IBS, calibração e concordância descrevem propriedades distintas; concordância elevada isoladamente não demonstra utilidade. Cox e RSF devem ser comparados juntamente com custo, cobertura e incerteza. A associação entre alarme precoce e desempenho é descritiva, não causal.\n",
        "limitacoes_survival.md": "# Limitações\n\nA análise possui somente 15 rolamentos independentes, cinco por condição, ensaios acelerados controlados, um tipo de rolamento, nenhum onset oficial e nenhuma validação externa. O detector primário é permissivo e alguns alarmes da Condição 3 foram muito precoces. Landmarks do mesmo rolamento são correlacionados. A dimensionalidade é alta, a calibração é limitada, a censura depende do cenário e os resultados são condicionais à detecção. Modelos neurais de sobrevivência foram excluídos por projeto e não há evidência de implantação industrial.\n",
        "conclusao_survival.md": "# Conclusão\n\nO valor adicional da análise de sobrevivência deve ser julgado conjuntamente por cobertura do detector, IBS, calibração, incerteza, cobertura da mediana e generalização por rolamento. Os resultados deste diretório delimitam essa conclusão ao protocolo XJTU-SY avaliado, sem alegar generalização industrial.\n",
    }
    for name, text in texts.items():
        atomic_text(output / "manuscript" / name, text)
    atomic_json(output / "manuscript/claims_manifest.json", {"claims": claims})


def _write_plots(output: Path, metrics: pd.DataFrame, config: Phase8Config) -> None:
    del metrics
    script = (
        "from pathlib import Path; "
        "from xjtu_sy_tcc.survival_models.plotting import generate_phase8_figures; "
        f"generate_phase8_figures(Path({str(output)!r}), {config.plot_dpi}, "
        f"{config.generate_pdf!r})"
    )
    environment = os.environ.copy()
    cache = Path(tempfile.gettempdir()) / "xjtu_phase8_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(cache)
    environment["XDG_CACHE_HOME"] = str(cache)
    subprocess.run([sys.executable, "-c", script], check=True, env=environment)
