"""Restartable, artifact-oriented orchestration for Phase 3."""

from __future__ import annotations

import logging
import resource
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import psutil

from xjtu_sy_tcc.config.phase3 import Phase3Config
from xjtu_sy_tcc.degradation.health import build_health_indicators
from xjtu_sy_tcc.degradation.metrics import aggregate_feature_scores, bearing_feature_scores
from xjtu_sy_tcc.degradation.pelt import (
    choose_parameters,
    detect_estimated_onset,
    parameters_from_row,
    run_sensitivity,
)
from xjtu_sy_tcc.degradation.plotting import generate_phase3_figures
from xjtu_sy_tcc.degradation.preprocessing import fit_preprocessor
from xjtu_sy_tcc.degradation.reporting import atomic_json, atomic_table, atomic_text
from xjtu_sy_tcc.degradation.selection import rank_features, validate_feature_names
from xjtu_sy_tcc.degradation.splits import build_folds, fold_manifest
from xjtu_sy_tcc.degradation.validation import validate_phase3
from xjtu_sy_tcc.logging_utils import log_event

LOGGER = logging.getLogger(__name__)


def run_phase3(
    config: Phase3Config,
    *,
    generate_figures: bool = True,
    reuse_sensitivity: bool = True,
) -> dict[str, object]:
    """Execute folds, quality analysis, health indicators, PELT, reports, and figures."""
    started = time.perf_counter()
    process = psutil.Process()
    created_at = datetime.now(UTC).isoformat()
    features = pd.read_parquet(config.feature_table)
    feature_columns = tuple(
        name for name in features if name.startswith(("horizontal_", "vertical_"))
    )
    _validate_prerequisites(features, feature_columns, config)
    log_event(
        LOGGER,
        logging.INFO,
        "phase3_prerequisites_validated",
        "Phase 3 input validated.",
        rows=len(features),
        feature_count=len(feature_columns),
    )

    folds = build_folds(features)
    manifest = fold_manifest(folds, features, config.configuration_hash, created_at)
    splits_directory = config.output_root / "splits"
    atomic_json(splits_directory / "folds.json", manifest)
    atomic_text(splits_directory / "folds.md", _fold_markdown(manifest))
    log_event(
        LOGGER,
        logging.INFO,
        "phase3_splits_complete",
        "Deterministic bearing folds created.",
        folds=5,
    )

    by_bearing = bearing_feature_scores(features, feature_columns)
    all_scores = aggregate_feature_scores(
        features,
        by_bearing,
        feature_columns,
        config.interpolation_grid_size,
        config.initial_window_fraction,
        config.final_window_fraction,
    )
    by_condition = pd.concat(
        [
            aggregate_feature_scores(
                features,
                by_bearing,
                feature_columns,
                config.interpolation_grid_size,
                config.initial_window_fraction,
                config.final_window_fraction,
                condition_id=condition,
            ).assign(condition_id=condition)
            for condition in (1, 2, 3)
        ],
        ignore_index=True,
    )
    descriptive = rank_features(all_scores, config.metric_weights, config.selected_feature_count)
    descriptive.insert(0, "scope", "post_hoc_all_dataset_descriptive_only")
    fold_rankings = []
    selected_payload = {"selection_scope": "training_bearings_only", "folds": []}
    for fold in folds:
        scores = aggregate_feature_scores(
            features,
            by_bearing,
            feature_columns,
            config.interpolation_grid_size,
            config.initial_window_fraction,
            config.final_window_fraction,
            bearings=fold.train_bearings,
        )
        ranking = rank_features(scores, config.metric_weights, config.selected_feature_count)
        ranking.insert(0, "fold", fold.fold_id)
        fold_rankings.append(ranking)
        selected = tuple(ranking.loc[ranking["selected"], "feature"])
        validate_feature_names(selected, feature_columns)
        selected_payload["folds"].append(
            {
                "fold": fold.fold_id,
                "training_bearings": list(fold.train_bearings),
                "selected_features": list(selected),
            }
        )
    rankings = pd.concat(fold_rankings, ignore_index=True)
    prognostics_directory = config.output_root / "prognostics"
    atomic_table(prognostics_directory / "feature_scores_by_bearing.parquet", by_bearing)
    atomic_table(prognostics_directory / "feature_scores_by_condition.parquet", by_condition)
    atomic_table(prognostics_directory / "fold_feature_rankings.parquet", rankings)
    atomic_table(prognostics_directory / "feature_ranking_summary.csv", descriptive)
    atomic_json(prognostics_directory / "selected_features_by_fold.json", selected_payload)
    log_event(
        LOGGER,
        logging.INFO,
        "phase3_feature_quality_complete",
        "Prognostic metrics and training-only rankings completed.",
        score_rows=len(by_bearing),
    )

    health_frames = []
    artifacts: dict[int, dict[str, object]] = {}
    degradation_directory = config.output_root / "degradation"
    preprocessing_directory = degradation_directory / "preprocessing"
    for fold in folds:
        ranking = rankings[rankings["fold"] == fold.fold_id]
        selected = tuple(ranking.loc[ranking["selected"], "feature"])
        training = features[features["bearing_id"].isin(fold.train_bearings)]
        fitted = fit_preprocessor(training, selected, config)
        artifact = fitted.artifact()
        artifact["fold"] = fold.fold_id
        artifact["configuration_hash"] = config.configuration_hash
        artifacts[fold.fold_id] = artifact
        atomic_json(preprocessing_directory / f"fold_{fold.fold_id}.json", artifact)
        health_frames.append(
            build_health_indicators(
                features,
                fold,
                fitted,
                config.baseline_fraction,
                config.minimum_baseline_acquisitions,
                config.smoothing_method,
                config.primary_smoothing_window,
            )
        )
    health = pd.concat(health_frames, ignore_index=True)
    atomic_table(degradation_directory / "health_indicators.parquet", health)
    log_event(
        LOGGER,
        logging.INFO,
        "phase3_health_complete",
        "Leakage-free fold health indicators generated.",
        rows=len(health),
    )

    sensitivity_path = degradation_directory / "pelt_sensitivity.parquet"
    prior_report_path = degradation_directory / "onset_validation.json"
    sensitivity = None
    if reuse_sensitivity and sensitivity_path.is_file() and prior_report_path.is_file():
        import json

        prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
        if prior_report.get("configuration_hash") == config.configuration_hash:
            sensitivity = pd.read_parquet(sensitivity_path)
            log_event(
                LOGGER,
                logging.INFO,
                "phase3_sensitivity_reused",
                "Validated PELT sensitivity artifact reused.",
                rows=len(sensitivity),
            )
    if sensitivity is None:
        sensitivity = run_sensitivity(health, config)
    parameter_ranking = choose_parameters(sensitivity)
    selected_parameters = parameter_ranking[parameter_ranking["selected_configuration"]]
    onsets = []
    for fold in folds:
        parameter_row = selected_parameters[selected_parameters["fold"] == fold.fold_id].iloc[0]
        parameters = parameters_from_row(parameter_row)
        fold_health = health[health["fold"] == fold.fold_id]
        for (subset, condition, bearing), group in fold_health.groupby(
            ["subset", "condition_id", "bearing_id"], sort=True
        ):
            detection = detect_estimated_onset(
                group, parameters, int(group["baseline_acquisitions"].iloc[0])
            )
            onsets.append(
                {
                    "fold": fold.fold_id,
                    "subset": subset,
                    "condition_id": int(condition),
                    "bearing_id": bearing,
                    "baseline_acquisitions": int(group["baseline_acquisitions"].iloc[0]),
                    "selected_parameter_id": int(parameter_row["parameter_id"]),
                    **detection,
                }
            )
    onset_table = pd.DataFrame(onsets)
    atomic_table(sensitivity_path, sensitivity)
    atomic_table(degradation_directory / "pelt_parameter_rankings.csv", parameter_ranking)
    atomic_table(degradation_directory / "estimated_onsets.parquet", onset_table)
    atomic_table(degradation_directory / "estimated_onsets.csv", onset_table)
    log_event(
        LOGGER,
        logging.INFO,
        "phase3_pelt_complete",
        "PELT sensitivity and estimated degradation onsets completed.",
        sensitivity_rows=len(sensitivity),
        onset_rows=len(onset_table),
    )

    validation = validate_phase3(
        features,
        feature_columns,
        folds,
        artifacts,
        health,
        onset_table,
        config.expected_rows,
        config.expected_feature_count,
    )
    duration = time.perf_counter() - started
    peak_memory = _peak_rss_bytes(process)
    report = _report_payload(
        config,
        validation,
        features,
        feature_columns,
        rankings,
        artifacts,
        sensitivity,
        parameter_ranking,
        onset_table,
        health,
        duration,
        peak_memory,
    )
    atomic_json(prognostics_directory / "feature_quality_report.json", report["feature_quality"])
    atomic_text(prognostics_directory / "feature_quality_report.md", _quality_markdown(report))
    atomic_json(degradation_directory / "onset_validation.json", report)
    atomic_text(degradation_directory / "phase3_validation.md", _validation_markdown(report))
    if generate_figures and validation.status == "passed":
        figure_count = generate_phase3_figures(
            config, features, rankings, health, onset_table, sensitivity, artifacts
        )
    else:
        figure_count = sum(
            1
            for directory in (
                config.figure_directory / "degradation",
                config.figure_directory / "prognostics",
            )
            if directory.is_dir()
            for path in directory.rglob("*")
            if path.suffix in {".png", ".pdf"}
        )
    report["figure_count"] = figure_count
    report["duration_seconds"] = time.perf_counter() - started
    atomic_json(degradation_directory / "onset_validation.json", report)
    return report


def _validate_prerequisites(
    features: pd.DataFrame, feature_columns: tuple[str, ...], config: Phase3Config
) -> None:
    required = {
        "condition_id",
        "bearing_id",
        "acquisition_number",
        "sequence_index",
        "elapsed_minutes",
        "rul_minutes",
    }
    if required - set(features):
        raise ValueError(f"Feature table is missing columns: {sorted(required - set(features))}")
    if (
        len(features) != config.expected_rows
        or len(feature_columns) != config.expected_feature_count
    ):
        raise ValueError("Phase 2 row or feature count does not match Phase 3 configuration")
    if features.duplicated(["condition_id", "bearing_id", "acquisition_number"]).any():
        raise ValueError("Phase 2 feature table contains duplicate acquisitions")
    if not np.all(np.isfinite(features.loc[:, feature_columns].to_numpy(float))):
        raise ValueError("Phase 2 feature table contains non-finite features")


def _report_payload(
    config,
    validation,
    features,
    feature_columns,
    rankings,
    artifacts,
    sensitivity,
    parameter_ranking,
    onsets,
    health,
    duration,
    peak_memory,
):
    return {
        "phase3_schema_version": 1,
        "status": validation.status,
        "configuration_hash": config.configuration_hash,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": duration,
        "peak_rss_bytes": peak_memory,
        "input_rows": len(features),
        "feature_count": len(feature_columns),
        "health_rows": len(health),
        "sensitivity_rows": len(sensitivity),
        "onset_rows": len(onsets),
        "detected_onsets": int((onsets["estimated_onset_status"] == "detected").sum()),
        "not_detected_onsets": int((onsets["estimated_onset_status"] == "not_detected").sum()),
        "issues": [
            {"severity": issue.severity, "code": issue.code, "message": issue.message}
            for issue in validation.issues
        ],
        "selected_parameters_by_fold": parameter_ranking[
            parameter_ranking["selected_configuration"]
        ].to_dict("records"),
        "selected_features_by_fold": {
            str(fold): group.loc[group["selected"], "feature"].tolist()
            for fold, group in rankings.groupby("fold")
        },
        "pca_explained_variance_by_fold": {
            str(fold): artifact["explained_variance_ratio"] for fold, artifact in artifacts.items()
        },
        "feature_quality": {
            "scope_warning": (
                "All-dataset ranking is post-hoc descriptive only; fold selection uses "
                "training bearings only."
            ),
            "bearing_score_rows": 15 * len(feature_columns),
            "condition_score_rows": 3 * len(feature_columns),
            "fold_ranking_rows": len(rankings),
            "formulas_documented_in": "docs/phase3_methodology.md",
        },
    }


def _fold_markdown(manifest: dict[str, object]) -> str:
    lines = [
        "# Deterministic bearing-level folds",
        "",
        f"Configuration hash: `{manifest['configuration_hash']}`",
        "",
    ]
    for fold in manifest["folds"]:  # type: ignore[index]
        lines.extend([f"## Fold {fold['fold_id']}", ""])  # type: ignore[index]
        for name, values in fold["subsets"].items():  # type: ignore[index]
            lines.append(
                f"- {name}: {', '.join(values['bearings'])} "
                f"({values['acquisition_count']} acquisitions)"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _quality_markdown(report: dict[str, object]) -> str:
    quality = report["feature_quality"]
    return "\n".join(
        [
            "# Phase 3 prognostic feature quality",
            "",
            str(quality["scope_warning"]),
            "",
            f"- Per-bearing score rows: {quality['bearing_score_rows']}",
            f"- Per-condition score rows: {quality['condition_score_rows']}",
            f"- Fold-ranking rows: {quality['fold_ranking_rows']}",
            "",
        ]
    )


def _validation_markdown(report: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Phase 3 validation",
            "",
            f"- Status: **{str(report['status']).upper()}**",
            f"- Input rows: {report['input_rows']}",
            f"- Health-indicator rows: {report['health_rows']}",
            f"- Sensitivity rows: {report['sensitivity_rows']}",
            f"- Detected estimated onsets: {report['detected_onsets']}",
            f"- Not detected: {report['not_detected_onsets']}",
            f"- Duration: {report['duration_seconds']:.3f} s",
            f"- Peak RSS: {report['peak_rss_bytes']} bytes",
            "",
        ]
    )


def _peak_rss_bytes(process: psutil.Process) -> int:
    """Return peak RSS where supported, with current RSS as a conservative fallback."""
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform.startswith(("linux", "freebsd")):
        maximum *= 1024
    return max(maximum, process.memory_info().rss)
