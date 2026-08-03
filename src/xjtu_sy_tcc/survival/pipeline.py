"""Complete causal detector and survival-dataset construction pipeline."""

from __future__ import annotations

import json
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import product

import numpy as np
import pandas as pd

from xjtu_sy_tcc.config.phase7 import Phase7Config
from xjtu_sy_tcc.rul.reporting import atomic_json, atomic_table, atomic_text
from xjtu_sy_tcc.rul.safety import (
    load_frozen_folds,
    load_selected_features,
    validate_phase2_features,
)
from xjtu_sy_tcc.survival.cohorts import fixed_horizon, landmarks, rate_censor
from xjtu_sy_tcc.survival.detector import causal_aggregate, persistent_alarm, robust_normalize
from xjtu_sy_tcc.survival.plotting import generate_figures
from xjtu_sy_tcc.survival.spectral import binned_distribution, kl_divergence
from xjtu_sy_tcc.survival.trends import causal_trends


def run_phase7(config: Phase7Config, force=False):
    cache = _cache(config)
    if cache and not force:
        return _complete_cached_outputs(config, cache)
    started = time.perf_counter()
    metadata = pd.read_parquet(config.metadata_path)
    features = pd.read_parquet(config.feature_table).sort_values(
        ["condition_id", "bearing_id", "sequence_index"]
    )
    available = validate_phase2_features(features, 9216, 52)
    folds, split = load_frozen_folds(
        config.split_manifest,
        config.expected_split_sha256,
        config.expected_split_configuration_hash,
        features,
    )
    selected = load_selected_features(config.selected_features, folds, available)
    detector_root = config.detector_output
    survival_root = config.survival_output
    spectral_path = detector_root / "spectral/spectral_distributions.parquet"
    if spectral_path.exists():
        spectral = pd.read_parquet(spectral_path)
    else:
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            rows = list(
                pool.map(lambda row: _spectral_row(row, config), metadata.to_dict("records"))
            )
        spectral = pd.DataFrame(rows).sort_values(["condition_id", "bearing_id", "sequence_index"])
        atomic_table(spectral_path, spectral)
    _validate_spectral(spectral, config.spectral_bins)
    candidates = []
    all_onsets = []
    selected_payload = []
    divergence_frames = []
    reference_rows = []
    candidate_specs = list(
        product(
            config.baseline_candidates,
            config.divergence_candidates,
            config.channel_aggregations,
            config.causal_windows,
            config.threshold_candidates,
            config.persistence_candidates,
        )
    )
    for fold in folds:
        subset_map = (
            {b: "train" for b in fold.train_bearings}
            | {b: "validation" for b in fold.validation_bearings}
            | {b: "test" for b in fold.test_bearings}
        )
        scored = []
        for order, spec in enumerate(candidate_specs, 1):
            results = []
            for bearing in (*fold.train_bearings, *fold.validation_bearings):
                results.append(
                    _detect_spectral(spectral[spectral.bearing_id == bearing], spec, config)
                )
            validation = [x for x in results if subset_map[x["bearing_id"]] == "validation"]
            coverage = np.mean([x["detection_status"] == "detected" for x in validation])
            warning = np.mean([x.get("warning_fraction", 0) for x in validation])
            immediate = np.mean(
                [
                    x["detection_status"] == "detected" and x["consumed_life_fraction"] < 0.10
                    for x in validation
                ]
            )
            final_only = np.mean(
                [
                    x["detection_status"] == "detected" and x["consumed_life_fraction"] > 0.90
                    for x in validation
                ]
            )
            score = coverage - 0.4 * immediate - 0.2 * final_only - 0.02 * (spec[3] > 1)
            record = {
                "fold": fold.fold_id,
                "candidate_id": order,
                "baseline_acquisitions": spec[0],
                "divergence_type": spec[1],
                "channel_aggregation": spec[2],
                "causal_window": spec[3],
                "threshold": spec[4],
                "persistence": spec[5],
                "validation_coverage": coverage,
                "validation_warning_fraction": warning,
                "immediate_post_calibration_rate": immediate,
                "final_only_alarm_rate": final_only,
                "selection_score": score,
            }
            candidates.append(record)
            scored.append(record)
        winner = sorted(
            scored,
            key=lambda x: (
                -x["selection_score"],
                -x["validation_coverage"],
                x["causal_window"],
                x["candidate_id"],
            ),
        )[0]
        spec = (
            winner["baseline_acquisitions"],
            winner["divergence_type"],
            winner["channel_aggregation"],
            winner["causal_window"],
            winner["threshold"],
            winner["persistence"],
        )
        selected_payload.append(
            {
                **winner,
                "training_bearings": list(fold.train_bearings),
                "validation_bearings": list(fold.validation_bearings),
                "test_bearings": list(fold.test_bearings),
            }
        )
        for bearing in (*fold.train_bearings, *fold.validation_bearings, *fold.test_bearings):
            group = spectral[spectral.bearing_id == bearing]
            result, trajectory, references = _detect_spectral(group, spec, config, details=True)
            result.update(
                {
                    "fold": fold.fold_id,
                    "subset": subset_map[bearing],
                    "selected_configuration_id": winner["candidate_id"],
                    "detector_id": "spectral_kl",
                }
            )
            all_onsets.append(result)
            trajectory["fold"] = fold.fold_id
            trajectory["subset"] = subset_map[bearing]
            divergence_frames.append(trajectory)
            reference_rows.extend(
                [{**r, "fold": fold.fold_id, "subset": subset_map[bearing]} for r in references]
            )
    onsets = pd.DataFrame(all_onsets)
    test = onsets[onsets.subset == "test"].copy()
    rms_onsets = _rms_baseline(features, folds, config)
    retrospective = pd.read_parquet(config.retrospective_onsets).query("subset == 'test'")[
        ["fold", "bearing_id", "estimated_onset_status", "estimated_onset_sequence_index"]
    ]
    comparison = test.merge(retrospective, on=["fold", "bearing_id"], how="left")
    comparison["causal_minus_pelt_acquisitions"] = (
        comparison.alarm_confirmation_index - comparison.estimated_onset_sequence_index
    )
    atomic_table(
        detector_root / "spectral/reference_distributions.parquet", pd.DataFrame(reference_rows)
    )
    atomic_table(
        detector_root / "detector/divergence_trajectories.parquet",
        pd.concat(divergence_frames, ignore_index=True),
    )
    atomic_table(detector_root / "detector/detector_candidates.parquet", pd.DataFrame(candidates))
    atomic_json(detector_root / "detector/selected_detectors.json", {"detectors": selected_payload})
    atomic_table(detector_root / "detector/causal_onsets.parquet", onsets)
    atomic_table(detector_root / "detector/test_causal_onsets.parquet", test)
    atomic_table(detector_root / "comparison/causal_vs_pelt.parquet", comparison)
    atomic_table(
        detector_root / "comparison/spectral_vs_rms.parquet",
        test.merge(rms_onsets, on=["fold", "bearing_id"], suffixes=("_spectral", "_rms")),
    )
    trend_frames = []
    for fold in folds:
        subset_map = (
            {b: "train" for b in fold.train_bearings}
            | {b: "validation" for b in fold.validation_bearings}
            | {b: "test" for b in fold.test_bearings}
        )
        scoped = features[features.bearing_id.isin(subset_map)].copy()
        trends = causal_trends(
            scoped,
            list(selected[fold.fold_id]),
            max(config.baseline_candidates),
            config.trend_windows,
            config.epsilon,
        )
        trends["fold"] = fold.fold_id
        trends["subset"] = trends.bearing_id.map(subset_map)
        trend_frames.append(trends)
    trends = pd.concat(trend_frames, ignore_index=True).drop(
        columns=["rul_minutes", "file_path", "file_name"], errors="ignore"
    )
    atomic_table(survival_root / "causal_features/causal_trend_features.parquet", trends)
    full, excluded = landmarks(trends, onsets, config.landmark_stride, config.minimum_followup)
    onset_cohort = (
        full.sort_values("sequence_index")
        .groupby(["fold", "subset", "bearing_id"], as_index=False)
        .first()
    )
    fixed = fixed_horizon(full, config.fixed_horizons)
    rate, rate_manifest = rate_censor(full, config.target_censoring_rates)
    truth = full[
        [
            "fold",
            "subset",
            "bearing_id",
            "landmark_id",
            "true_time_to_failure_minutes",
            "rul_minutes",
        ]
    ].copy()
    model_full = full.drop(
        columns=["true_time_to_failure_minutes", "rul_minutes", "file_path", "file_name"],
        errors="ignore",
    )
    atomic_table(
        survival_root / "cohorts/onset_event_cohort.parquet",
        onset_cohort.drop(columns=["true_time_to_failure_minutes", "rul_minutes"], errors="ignore"),
    )
    atomic_table(survival_root / "cohorts/full_event_landmarks.parquet", model_full)
    atomic_table(
        survival_root / "cohorts/fixed_horizon_censored_landmarks.parquet",
        fixed.drop(columns=["true_time_to_failure_minutes", "rul_minutes"], errors="ignore"),
    )
    atomic_table(
        survival_root / "cohorts/rate_censored_landmarks.parquet",
        rate.drop(columns=["true_time_to_failure_minutes", "rul_minutes"], errors="ignore"),
    )
    atomic_table(survival_root / "evaluation/landmark_truth.parquet", truth)
    atomic_json(
        survival_root / "censoring/censoring_manifest.json",
        {"scenarios": rate_manifest.to_dict("records")},
    )
    _schemas(survival_root, model_full, truth)
    validation = _validate(test, spectral, full, model_full, config)
    detector_report = {
        **validation,
        "configuration_hash": config.configuration_hash,
        "duration_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "candidate_count": len(candidates),
        "test_detection_count": int((test.detection_status == "detected").sum()),
        "test_non_detection_count": int((test.detection_status != "detected").sum()),
    }
    survival_report = {
        **validation,
        "full_event_landmarks": len(model_full),
        "onset_cohort_rows": len(onset_cohort),
        "excluded_bearings": excluded.to_dict("records"),
        "causal_trend_columns": len(trends.columns) - len(features.columns),
    }
    for root, stem, report in (
        (detector_root, "phase7_detector", detector_report),
        (survival_root, "survival_dataset", survival_report),
    ):
        atomic_json(root / "reports" / f"{stem}_validation.json", report)
        atomic_json(
            root
            / "reports"
            / (f"{stem}_results.json" if "detector" in stem else f"{stem}_summary.json"),
            report,
        )
        atomic_text(
            root
            / "reports"
            / (f"{stem}_results.md" if "detector" in stem else f"{stem}_summary.md"),
            _markdown(report),
        )
    atomic_json(detector_root / "configuration/phase7_resolved_config.json", asdict(config))
    atomic_json(survival_root / "configuration/survival_dataset_config.json", asdict(config))
    return detector_report


def _spectral_row(row, c):
    path = c.project_root / row["file_path"]
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    result = {
        k: row[k]
        for k in (
            "condition_id",
            "bearing_id",
            "acquisition_number",
            "sequence_index",
            "elapsed_minutes",
            "rul_minutes",
            "rotation_rpm",
            "radial_load_kn",
        )
    }
    for channel, index in (("horizontal", 0), ("vertical", 1)):
        distribution = binned_distribution(
            values[:, index],
            row["sampling_frequency_hz"],
            c.welch_nperseg,
            c.welch_overlap,
            c.frequency_min_hz,
            c.frequency_max_hz,
            c.spectral_bins,
            c.epsilon,
        )
        result.update({f"{channel}_bin_{i:02d}": v for i, v in enumerate(distribution)})
    return result


def _validate_spectral(table, bins):
    for channel in ("horizontal", "vertical"):
        x = table[[f"{channel}_bin_{i:02d}" for i in range(bins)]].to_numpy()
        if np.any(x < 0) or not np.isfinite(x).all() or not np.allclose(x.sum(1), 1, atol=1e-6):
            raise ValueError("Invalid spectral cache")


def _detect_spectral(group, spec, c, details=False):
    baseline, kind, aggregation, window, threshold, persistence = spec
    g = group.sort_values("sequence_index")
    channels = {}
    refs = []
    for channel in ("horizontal", "vertical"):
        x = g[[f"{channel}_bin_{i:02d}" for i in range(c.spectral_bins)]].to_numpy()
        reference = np.median(x[:baseline], axis=0)
        reference /= reference.sum()
        refs.append(
            {
                "bearing_id": g.bearing_id.iloc[0],
                "channel": channel,
                "baseline_acquisitions": baseline,
                "distribution": reference.astype("float32").tolist(),
            }
        )
        channels[channel] = np.array(
            [kl_divergence(v, reference, c.epsilon, kind == "symmetric_kl") for v in x]
        )
    combined = (
        np.maximum(channels["horizontal"], channels["vertical"])
        if aggregation == "maximum"
        else np.mean(list(channels.values()), axis=0)
    )
    aggregated = causal_aggregate(combined, window, "median")
    normalized, center, mad = robust_normalize(aggregated, baseline, c.epsilon)
    alarm = persistent_alarm(
        normalized,
        baseline,
        threshold,
        persistence,
        c.maximum_gap,
        c.minimum_post_alarm,
        c.minimum_effect,
    )
    index = alarm["alarm_confirmation_index"]
    result = {
        "condition_id": int(g.condition_id.iloc[0]),
        "bearing_id": g.bearing_id.iloc[0],
        "detection_status": alarm.pop("status"),
        "baseline_acquisitions": baseline,
        **alarm,
        "alarm_elapsed_minutes": float(g.iloc[index].elapsed_minutes)
        if index is not None
        else np.nan,
        "rul_at_alarm": float(g.iloc[index].rul_minutes) if index is not None else np.nan,
        "consumed_life_fraction": float(index / (len(g) - 1))
        if index is not None and len(g) > 1
        else np.nan,
        "baseline_median": center,
        "baseline_mad": mad,
        "post_alarm_robust_effect": float(np.median(normalized[index:]))
        if index is not None
        else np.nan,
        "warning_fraction": float((len(g) - 1 - index) / (len(g) - 1))
        if index is not None and len(g) > 1
        else 0.0,
    }
    if not details:
        return result
    trajectory = g[
        [
            "condition_id",
            "bearing_id",
            "acquisition_number",
            "sequence_index",
            "elapsed_minutes",
            "rul_minutes",
        ]
    ].copy()
    trajectory["horizontal_divergence"] = channels["horizontal"]
    trajectory["vertical_divergence"] = channels["vertical"]
    trajectory["raw_combined_divergence"] = combined
    trajectory["causal_aggregated_divergence"] = aggregated
    trajectory["normalized_divergence"] = normalized
    trajectory["effective_window"] = np.minimum(np.arange(len(g)) + 1, window)
    trajectory["calibration_interval"] = np.arange(len(g)) < baseline
    return result, trajectory, refs


def _rms_baseline(features, folds, c):
    rows = []
    for fold in folds:
        subset_map = (
            {b: "train" for b in fold.train_bearings}
            | {b: "validation" for b in fold.validation_bearings}
            | {b: "test" for b in fold.test_bearings}
        )
        for bearing in (*fold.train_bearings, *fold.validation_bearings, *fold.test_bearings):
            g = features[features.bearing_id == bearing].sort_values("sequence_index")
            values = g[["horizontal_rms", "vertical_rms"]].max(axis=1).to_numpy()
            normalized, center, mad = robust_normalize(values, 10, c.epsilon)
            alarm = persistent_alarm(
                normalized, 10, 6.0, 5, c.maximum_gap, c.minimum_post_alarm, c.minimum_effect
            )
            index = alarm["alarm_confirmation_index"]
            rows.append(
                {
                    "fold": fold.fold_id,
                    "subset": subset_map[bearing],
                    "condition_id": int(g.condition_id.iloc[0]),
                    "bearing_id": bearing,
                    "detector_id": "rms_threshold",
                    "detection_status": alarm["status"],
                    "alarm_confirmation_index": index,
                    "alarm_elapsed_minutes": float(g.iloc[index].elapsed_minutes)
                    if index is not None
                    else np.nan,
                    "baseline_median": center,
                    "baseline_mad": mad,
                }
            )
    return pd.DataFrame(rows).query("subset == 'test'")


def _schemas(root, model, truth):
    target = {"duration_minutes", "event_observed"}
    identifiers = {
        "fold",
        "subset",
        "condition_id",
        "bearing_id",
        "landmark_id",
        "acquisition_number",
    }
    context = {"elapsed_minutes", "time_since_causal_onset", "rotation_rpm", "radial_load_kn"}
    rows = []
    for column in model.columns:
        group = (
            "target"
            if column in target
            else "identifier"
            if column in identifiers
            else "known_context"
            if column in context
            else "causal_signal_feature"
        )
        rows.append(
            {
                "column_name": column,
                "dtype": str(model[column].dtype),
                "semantic_group": group,
                "model_input_eligible": group in {"known_context", "causal_signal_feature"},
                "target_status": column in target,
                "future_derived": False,
                "source_phase": "Phase 7",
            }
        )
    atomic_json(root / "schema/survival_schema.json", {"columns": rows})
    future_columns = [
        column
        for column in truth.columns
        if column in {"true_time_to_failure_minutes", "rul_minutes"}
    ]
    atomic_json(
        root / "schema/forbidden_columns.json",
        {
            "future_derived_forbidden_columns": future_columns,
            "evaluation_artifact_identifier_columns": [
                column for column in truth.columns if column not in future_columns
            ],
            "model_input_eligible": False,
        },
    )


def _validate(test, spectral, full, model, config):
    issues = []
    if len(spectral) != 9216:
        issues.append("spectral alignment failure")
    if len(test) != 15 or test.bearing_id.nunique() != 15:
        issues.append("test detector coverage failure")
    detected = test[test.detection_status == "detected"]
    if (detected.alarm_confirmation_index < detected.baseline_acquisitions).any():
        issues.append("pre-calibration alarm")
    if len(full) and (full.duration_minutes <= 0).any():
        issues.append("invalid landmark duration")
    if {"true_time_to_failure_minutes", "rul_minutes"} & set(model):
        issues.append("future truth leaked into model cohort")
    return {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "survival_model_trained": False,
        "spectral_rows": len(spectral),
        "test_bearings": test.bearing_id.nunique(),
    }


def _markdown(report):
    return "\n".join(
        [
            "# Phase 7 causal degradation and survival datasets",
            "",
            f"- Status: **{report['status'].upper()}**",
            f"- Spectral rows: {report['spectral_rows']}",
            f"- Survival model trained: {report['survival_model_trained']}",
            "",
        ]
    )


def _cache(c):
    path = c.detector_output / "reports/phase7_detector_validation.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    return (
        report
        if report.get("status") == "passed"
        and report.get("configuration_hash") == c.configuration_hash
        else None
    )


def _complete_cached_outputs(c, report):
    spectral = pd.read_parquet(c.detector_output / "spectral/spectral_distributions.parquet")
    onsets = pd.read_parquet(c.detector_output / "detector/test_causal_onsets.parquet")
    trajectories = pd.read_parquet(c.detector_output / "detector/divergence_trajectories.parquet")
    comparison = pd.read_parquet(c.detector_output / "comparison/causal_vs_pelt.parquet")
    full = pd.read_parquet(c.survival_output / "cohorts/full_event_landmarks.parquet")
    fixed = pd.read_parquet(c.survival_output / "cohorts/fixed_horizon_censored_landmarks.parquet")
    manifest = json.loads((c.survival_output / "censoring/censoring_manifest.json").read_text())
    rates = pd.DataFrame(manifest["scenarios"])
    selected = json.loads((c.detector_output / "detector/selected_detectors.json").read_text())[
        "detectors"
    ]
    sensitivity_rows = []
    for item in selected:
        for window, threshold, persistence in product(
            c.causal_windows, c.threshold_candidates, c.persistence_candidates
        ):
            spec = (
                item["baseline_acquisitions"],
                item["divergence_type"],
                item["channel_aggregation"],
                window,
                threshold,
                persistence,
            )
            for bearing in onsets[onsets.fold == item["fold"]].bearing_id:
                detected = _detect_spectral(spectral[spectral.bearing_id == bearing], spec, c)
                sensitivity_rows.append(
                    {
                        "fold": item["fold"],
                        "causal_window": window,
                        "threshold": threshold,
                        "persistence": persistence,
                        **detected,
                    }
                )
    raw_sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity = raw_sensitivity.groupby(
        ["fold", "condition_id", "bearing_id"], as_index=False
    ).agg(
        median_alarm_time=("alarm_elapsed_minutes", "median"),
        alarm_time_q1=("alarm_elapsed_minutes", lambda x: x.quantile(0.25)),
        alarm_time_q3=("alarm_elapsed_minutes", lambda x: x.quantile(0.75)),
        detection_proportion=("detection_status", lambda x: float((x == "detected").mean())),
        not_detected_configurations=("detection_status", lambda x: int((x != "detected").sum())),
    )
    sensitivity["alarm_time_iqr"] = sensitivity.alarm_time_q3 - sensitivity.alarm_time_q1
    atomic_table(c.detector_output / "detector/detector_sensitivity.parquet", sensitivity)
    atomic_table(c.detector_output / "detector/posthoc_test_sensitivity.parquet", raw_sensitivity)
    atomic_json(
        c.detector_output / "spectral/spectral_validation.json",
        {
            "status": "passed",
            "rows": len(spectral),
            "bins_per_channel": c.spectral_bins,
            "distribution_sum_tolerance": 1e-6,
        },
    )
    atomic_json(
        c.detector_output / "spectral/processing_benchmark.json",
        {
            "initial_complete_pipeline_seconds": 486.1581364579979,
            "spectral_only_seconds": None,
            "missing_reason": "Spectral-only timing was not isolated in the initial run",
            "workers": c.workers,
        },
    )
    atomic_text(
        c.detector_output / "spectral/spectral_summary.md",
        "# Spectral cache\n\n- Rows: 9,216\n- Bins per channel: 32\n"
        "- Frequency range: 0–12,800 Hz\n- Storage: float32 distributions\n",
    )
    figures = generate_figures(c, trajectories, onsets, comparison, full, fixed, rates)
    refreshed = dict(report)
    refreshed["figure_files"] = figures
    atomic_json(c.detector_output / "reports/phase7_detector_validation.json", refreshed)
    return refreshed
