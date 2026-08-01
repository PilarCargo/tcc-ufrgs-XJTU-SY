"""Complete leakage-free classical RUL training, selection, test, and reporting pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import resource
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import psutil

from xjtu_sy_tcc.config.phase4 import Phase4Config
from xjtu_sy_tcc.logging_utils import log_event
from xjtu_sy_tcc.rul.interpretability import permutation_importance, ridge_coefficients
from xjtu_sy_tcc.rul.metrics import (
    add_prediction_diagnostics,
    bearing_metrics,
    fold_metrics,
    grouped_metrics,
)
from xjtu_sy_tcc.rul.models import candidates, experiment_specs, fit_model
from xjtu_sy_tcc.rul.plotting import generate_phase4_figures
from xjtu_sy_tcc.rul.posthoc import annotate_posthoc, life_stage_metrics, onset_region_metrics
from xjtu_sy_tcc.rul.reporting import atomic_json, atomic_model, atomic_table, atomic_text
from xjtu_sy_tcc.rul.safety import (
    TRACE_COLUMNS,
    load_frozen_folds,
    load_selected_features,
    load_test_onsets,
    validate_input_columns,
    validate_phase2_features,
)
from xjtu_sy_tcc.rul.selection import rank_candidates, validation_candidate_record
from xjtu_sy_tcc.rul.validation import validate_phase4
from xjtu_sy_tcc.rul.weights import bearing_balanced_weights, effective_sample_size

LOGGER = logging.getLogger(__name__)


def run_phase4(
    config: Phase4Config,
    *,
    generate_figures: bool = True,
    force_train: bool = False,
) -> dict[str, object]:
    """Execute the complete default experiment matrix and return the validation report."""
    cache = _compatible_cache(config)
    if cache is not None and not force_train:
        if generate_figures and not cache.get("figure_files"):
            cache = _generate_cached_figures(config, cache)
        log_event(LOGGER, logging.INFO, "phase4_cache_reused", "Compatible Phase 4 outputs reused.")
        return cache
    started = time.perf_counter()
    process = psutil.Process()
    features = (
        pd.read_parquet(config.feature_table)
        .sort_values(["condition_id", "bearing_id", "sequence_index"])
        .reset_index(drop=True)
    )
    feature_columns = validate_phase2_features(
        features, config.expected_feature_rows, config.expected_feature_count
    )
    folds, split_payload = load_frozen_folds(
        config.split_manifest,
        config.expected_split_sha256,
        config.expected_split_configuration_hash,
        features,
    )
    selected_by_fold = load_selected_features(
        config.selected_feature_manifest, folds, feature_columns
    )
    test_onsets = load_test_onsets(config.test_onset_table, folds)
    resolved = _resolved_config(config)
    atomic_json(config.output_directory / "configuration" / "phase4_resolved_config.json", resolved)
    log_event(
        LOGGER,
        logging.INFO,
        "phase4_prerequisites_validated",
        "Frozen Phase 4 prerequisites validated.",
        rows=len(features),
        folds=len(folds),
    )

    all_candidate_records = []
    selected_payload = {"configuration_hash": config.configuration_hash, "models": []}
    prediction_frames = []
    coefficient_frames = []
    importance_frames = []
    cost_records = []
    artifact_records = []
    experiment_names = None
    for fold in folds:
        train = features[features["bearing_id"].isin(fold.train_bearings)].copy()
        validation = features[features["bearing_id"].isin(fold.validation_bearings)].copy()
        test = features[features["bearing_id"].isin(fold.test_bearings)].copy()
        weights = bearing_balanced_weights(train)
        specs = experiment_specs(selected_by_fold[fold.fold_id], feature_columns)
        experiment_names = tuple(spec.experiment for spec in specs)
        for spec in specs:
            validate_input_columns(
                spec.input_columns if spec.input_columns else ("condition_id",),
                tuple(features.columns),
                allow_time="time" in spec.experiment,
            )
            records = []
            for candidate_id, parameters in enumerate(candidates(spec, config), start=1):
                bundle = fit_model(spec, parameters, train, weights, config)
                inference_started = time.perf_counter()
                validation_raw = bundle.predict(validation)
                validation_seconds = time.perf_counter() - inference_started
                validation_frame = _prediction_base(validation, fold.fold_id, "validation", spec)
                validation_frame = add_prediction_diagnostics(validation_frame, validation_raw)
                record = validation_candidate_record(
                    bundle, validation_frame, fold.fold_id, candidate_id
                )
                record["validation_inference_seconds"] = validation_seconds
                records.append(record)
            ranked_group = rank_candidates(pd.DataFrame(records))
            all_candidate_records.append(ranked_group)
            chosen = ranked_group[ranked_group["selected"]].iloc[0]
            parameters = json.loads(chosen["parameters_json"])
            frozen = fit_model(spec, parameters, train, weights, config)
            validation_frame = _prediction_base(validation, fold.fold_id, "validation", spec)
            validation_frame = add_prediction_diagnostics(
                validation_frame, frozen.predict(validation)
            )
            test_started = time.perf_counter()
            test_raw = frozen.predict(test)
            test_seconds = time.perf_counter() - test_started
            test_frame = add_prediction_diagnostics(
                _prediction_base(test, fold.fold_id, "test", spec), test_raw
            )
            prediction_frames.append(test_frame)
            coefficient_frames.append(ridge_coefficients(frozen, fold.fold_id))
            importance_frames.append(
                permutation_importance(
                    frozen,
                    validation,
                    validation_frame,
                    fold.fold_id,
                    config.permutation_repeats,
                    config.random_seed,
                )
            )
            artifact_path = (
                config.output_directory
                / "artifacts"
                / f"fold_{fold.fold_id}"
                / spec.experiment
                / "model.joblib"
            )
            atomic_model(artifact_path, frozen)
            artifact_sha = _sha256(artifact_path)
            fitted_bearings = (
                frozen.transformer.fitted_bearings
                if frozen.transformer is not None
                else tuple(sorted(train["bearing_id"].unique()))
            )
            artifact_record = {
                "fold": fold.fold_id,
                "experiment": spec.experiment,
                "model": spec.model_family,
                "parameters": parameters,
                "input_columns": list(spec.input_columns),
                "fitted_bearings": list(fitted_bearings),
                "configuration_hash": config.configuration_hash,
                "artifact_path": artifact_path.relative_to(config.project_root).as_posix(),
                "artifact_sha256": artifact_sha,
            }
            artifact_records.append(artifact_record)
            atomic_json(artifact_path.with_name("metadata.json"), artifact_record)
            selected_payload["models"].append(artifact_record)
            cost_records.append(
                {
                    "fold": fold.fold_id,
                    "experiment": spec.experiment,
                    "model": spec.model_family,
                    "preprocessing_fit_seconds": frozen.preprocessing_fit_seconds,
                    "training_seconds": frozen.training_seconds,
                    "test_inference_seconds": test_seconds,
                    "test_inference_seconds_per_acquisition": test_seconds / len(test),
                    "artifact_size_bytes": artifact_path.stat().st_size,
                    "input_feature_count": len(spec.input_columns),
                    "training_observations": len(train),
                    "effective_weighted_sample_size": effective_sample_size(weights),
                }
            )
        log_event(
            LOGGER,
            logging.INFO,
            "phase4_fold_complete",
            "Frozen fold trained and tested.",
            fold=fold.fold_id,
            test_rows=len(test),
        )

    candidate_table = pd.concat(all_candidate_records, ignore_index=True)
    predictions = annotate_posthoc(
        pd.concat(prediction_frames, ignore_index=True),
        test_onsets,
        config.life_stage_thresholds,
    )
    bearing_table = bearing_metrics(predictions)
    fold_table = fold_metrics(bearing_table)
    acquisition_table = grouped_metrics(predictions, ["experiment", "model", "fold"])
    condition_table = _condition_metrics(predictions, bearing_table)
    life_table = life_stage_metrics(predictions)
    onset_table = onset_region_metrics(predictions)
    comparison = _model_comparison(predictions, bearing_table, fold_table)
    ridge_table = pd.concat(
        [item for item in coefficient_frames if not item.empty], ignore_index=True
    )
    importance_table = pd.concat(
        [item for item in importance_frames if not item.empty], ignore_index=True
    )
    importance_summary = _importance_summary(ridge_table, importance_table)
    costs = pd.DataFrame(cost_records)
    if experiment_names is None:
        raise RuntimeError("No Phase 4 experiments were generated")
    validation = validate_phase4(
        features,
        folds,
        predictions,
        candidate_table,
        artifact_records,
        test_onsets,
        experiment_names,
    )
    _verify_artifact_hashes(config, artifact_records)
    paths = _write_outputs(
        config,
        candidate_table,
        selected_payload,
        predictions,
        acquisition_table,
        bearing_table,
        fold_table,
        condition_table,
        life_table,
        onset_table,
        comparison,
        ridge_table,
        importance_table,
        importance_summary,
        costs,
    )
    figure_count = (
        generate_phase4_figures(
            config,
            predictions,
            comparison,
            condition_table,
            life_table,
            onset_table,
            importance_summary,
            costs,
            test_onsets,
        )
        if generate_figures and validation.status == "passed"
        else 0
    )
    report = _report(
        config,
        validation,
        split_payload,
        feature_columns,
        candidate_table,
        comparison,
        condition_table,
        life_table,
        onset_table,
        costs,
        paths,
        time.perf_counter() - started,
        _peak_rss(process),
        figure_count,
    )
    atomic_json(config.output_directory / "reports" / "phase4_validation.json", report)
    atomic_json(config.output_directory / "reports" / "phase4_results.json", report)
    atomic_text(
        config.output_directory / "reports" / "phase4_results.md",
        _markdown_report(report),
    )
    return report


def _prediction_base(frame, fold, subset, spec):
    result = frame.loc[:, TRACE_COLUMNS].copy()
    result.insert(0, "model", spec.model_family)
    result.insert(0, "experiment", spec.experiment)
    result.insert(0, "subset", subset)
    result.insert(0, "fold", fold)
    return result


def _condition_metrics(predictions, bearing):
    table = grouped_metrics(predictions, ["experiment", "model", "condition_id"])
    macro = bearing.groupby(["experiment", "model", "condition_id"], as_index=False).agg(
        macro_mae=("mae", "mean"), normalized_mae=("normalized_mae", "mean")
    )
    return table.merge(macro, on=["experiment", "model", "condition_id"], how="left")


def _model_comparison(predictions, bearing, folds):
    acquisition = grouped_metrics(predictions, ["experiment", "model"]).rename(
        columns={
            "mae": "global_mae",
            "rmse": "global_rmse",
            "r2": "global_r2",
            "signed_error": "global_signed_error",
        }
    )
    macro = bearing.groupby(["experiment", "model"], as_index=False).agg(
        macro_mae=("mae", "mean"),
        macro_rmse=("rmse", "mean"),
        median_bearing_mae=("mae", "median"),
        bearing_mae_q1=("mae", lambda x: x.quantile(0.25)),
        bearing_mae_q3=("mae", lambda x: x.quantile(0.75)),
        monotonicity_violation_rate=("monotonicity_violation_rate", "mean"),
    )
    fold_summary = folds.groupby(["experiment", "model"], as_index=False).agg(
        fold_macro_mae_mean=("macro_mae", "mean"),
        fold_macro_mae_std=("macro_mae", "std"),
        fold_macro_rmse_mean=("macro_rmse", "mean"),
        fold_macro_rmse_std=("macro_rmse", "std"),
    )
    return (
        acquisition.merge(macro, on=["experiment", "model"])
        .merge(fold_summary, on=["experiment", "model"])
        .sort_values("macro_mae")
    )


def _importance_summary(ridge, permutation):
    ridge_summary = ridge.groupby("feature", as_index=False).agg(
        ridge_mean_absolute_coefficient=("absolute_coefficient", "mean"),
        ridge_fold_count=("fold", "nunique"),
    )
    permutation_summary = permutation.groupby("feature", as_index=False).agg(
        permutation_mean_macro_mae_increase=("importance_mean_macro_mae_increase", "mean"),
        permutation_fold_count=("fold", "nunique"),
    )
    return ridge_summary.merge(permutation_summary, on="feature", how="outer").fillna(0)


def _write_outputs(
    config,
    candidates_table,
    selected_payload,
    predictions,
    acquisition,
    bearing,
    folds,
    condition,
    life,
    onset,
    comparison,
    ridge,
    importance,
    importance_summary,
    costs,
):
    root = config.output_directory
    paths = {
        "validation_candidates": root / "model_selection" / "validation_candidates.parquet",
        "selected_models": root / "model_selection" / "selected_models.json",
        "test_predictions_parquet": root / "predictions" / "test_predictions.parquet",
        "test_predictions_csv": root / "predictions" / "test_predictions.csv",
        "acquisition_metrics": root / "metrics" / "acquisition_metrics.parquet",
        "bearing_metrics": root / "metrics" / "bearing_metrics.parquet",
        "fold_metrics": root / "metrics" / "fold_metrics.parquet",
        "condition_metrics": root / "metrics" / "condition_metrics.parquet",
        "life_stage_metrics": root / "metrics" / "life_stage_metrics.parquet",
        "onset_region_metrics": root / "metrics" / "onset_region_metrics.parquet",
        "model_comparison": root / "metrics" / "model_comparison.csv",
        "ridge_coefficients": root / "interpretability" / "ridge_coefficients.parquet",
        "permutation_importance": root / "interpretability" / "permutation_importance.parquet",
        "importance_summary": root / "interpretability" / "feature_importance_summary.csv",
        "computational_costs": root / "metrics" / "computational_costs.parquet",
    }
    for key, table in (
        ("validation_candidates", candidates_table),
        ("test_predictions_parquet", predictions),
        ("test_predictions_csv", predictions),
        ("acquisition_metrics", acquisition),
        ("bearing_metrics", bearing),
        ("fold_metrics", folds),
        ("condition_metrics", condition),
        ("life_stage_metrics", life),
        ("onset_region_metrics", onset),
        ("model_comparison", comparison),
        ("ridge_coefficients", ridge),
        ("permutation_importance", importance),
        ("importance_summary", importance_summary),
        ("computational_costs", costs),
    ):
        atomic_table(paths[key], table)
    atomic_json(paths["selected_models"], selected_payload)
    return {key: path.relative_to(config.project_root).as_posix() for key, path in paths.items()}


def _resolved_config(config):
    payload = asdict(config)
    payload["configuration_hash"] = config.configuration_hash
    for name in (
        "config_path",
        "project_root",
        "feature_table",
        "split_manifest",
        "selected_feature_manifest",
        "test_onset_table",
        "output_directory",
    ):
        payload[name] = str(payload[name])
    payload["input_hashes"] = {
        "feature_table": _sha256(config.feature_table),
        "split_manifest": _sha256(config.split_manifest),
        "selected_feature_manifest": _sha256(config.selected_feature_manifest),
        "test_onset_table": _sha256(config.test_onset_table),
    }
    payload["health_indicator_experiment"] = (
        "omitted: Phase 3 baseline uses the complete initial calibration interval; "
        "primary full-trajectory comparison would be unfair"
    )
    return payload


def _report(
    config,
    validation,
    split_payload,
    feature_columns,
    candidates_table,
    comparison,
    condition,
    life,
    onset,
    costs,
    paths,
    duration,
    peak,
    figures,
):
    return {
        "phase4_schema_version": 1,
        "status": validation.status,
        "configuration_hash": config.configuration_hash,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "split_manifest_sha256": _sha256(config.split_manifest),
        "split_configuration_hash": split_payload["configuration_hash"],
        "input_hashes": {
            "feature_table": _sha256(config.feature_table),
            "split_manifest": _sha256(config.split_manifest),
            "selected_feature_manifest": _sha256(config.selected_feature_manifest),
            "test_onset_table": _sha256(config.test_onset_table),
        },
        "feature_count": len(feature_columns),
        "target_column": config.target_column,
        "duration_seconds": duration,
        "peak_rss_bytes": peak,
        "figure_files": figures,
        "validation_candidate_count": len(candidates_table),
        "issues": [asdict(issue) for issue in validation.issues],
        "model_comparison": comparison.to_dict("records"),
        "condition_metrics": condition.to_dict("records"),
        "life_stage_metrics": life.to_dict("records"),
        "onset_region_metrics": onset.to_dict("records"),
        "computational_costs": costs.to_dict("records"),
        "artifacts": paths,
        "scientific_scope": {
            "health_indicator": (
                "omitted from prediction; retrospective baseline requires initial calibration"
            ),
            "estimated_onset": "retrospective post-hoc annotation only; never a model input",
            "models": "classical only; no deep learning",
        },
    }


def _markdown_report(report):
    rows = [
        "# Phase 4 classical RUL regression",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Duration: {report['duration_seconds']:.3f} s",
        f"- Peak RSS: {report['peak_rss_bytes']} bytes",
        f"- Validation candidates: {report['validation_candidate_count']}",
        f"- Figure files: {report['figure_files']}",
        "",
        "## Test model comparison",
        "",
        "| Experiment | Macro MAE | Macro RMSE | Global MAE | Global RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["model_comparison"]:
        rows.append(
            f"| {item['experiment']} | {item['macro_mae']:.3f} | "
            f"{item['macro_rmse']:.3f} | {item['global_mae']:.3f} | "
            f"{item['global_rmse']:.3f} |"
        )
    rows.extend(
        [
            "",
            "Estimated degradation onset is used only for retrospective post-hoc evaluation. "
            "It is not ground truth or a model input.",
            "",
        ]
    )
    return "\n".join(rows)


def _compatible_cache(config):
    report_path = config.output_directory / "reports" / "phase4_validation.json"
    predictions = config.output_directory / "predictions" / "test_predictions.parquet"
    if not report_path.is_file() or not predictions.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_inputs = {
        "feature_table": _sha256(config.feature_table),
        "split_manifest": _sha256(config.split_manifest),
        "selected_feature_manifest": _sha256(config.selected_feature_manifest),
        "test_onset_table": _sha256(config.test_onset_table),
    }
    if (
        report.get("status") == "passed"
        and report.get("configuration_hash") == config.configuration_hash
        and report.get("input_hashes") == expected_inputs
    ):
        return report
    return None


def _generate_cached_figures(config: Phase4Config, report: dict[str, object]):
    """Generate missing figures from validated tables without refitting frozen models."""
    root = config.output_directory
    tables = {
        "predictions": pd.read_parquet(root / "predictions" / "test_predictions.parquet"),
        "comparison": pd.read_csv(root / "metrics" / "model_comparison.csv"),
        "condition": pd.read_parquet(root / "metrics" / "condition_metrics.parquet"),
        "life": pd.read_parquet(root / "metrics" / "life_stage_metrics.parquet"),
        "onset": pd.read_parquet(root / "metrics" / "onset_region_metrics.parquet"),
        "importance": pd.read_csv(
            root / "interpretability" / "feature_importance_summary.csv"
        ),
        "costs": pd.read_parquet(root / "metrics" / "computational_costs.parquet"),
    }
    onsets = pd.read_parquet(config.test_onset_table)
    onsets = onsets[onsets["subset"] == "test"].copy()
    figure_count = generate_phase4_figures(
        config,
        tables["predictions"],
        tables["comparison"],
        tables["condition"],
        tables["life"],
        tables["onset"],
        tables["importance"],
        tables["costs"],
        onsets,
    )
    refreshed = dict(report)
    refreshed["figure_files"] = figure_count
    atomic_json(root / "reports" / "phase4_validation.json", refreshed)
    atomic_json(root / "reports" / "phase4_results.json", refreshed)
    atomic_text(root / "reports" / "phase4_results.md", _markdown_report(refreshed))
    return refreshed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_artifact_hashes(config: Phase4Config, records: list[dict[str, object]]) -> None:
    for record in records:
        path = config.project_root / str(record["artifact_path"])
        if _sha256(path) != record["artifact_sha256"]:
            raise ValueError(f"Saved model hash mismatch: {path}")


def _peak_rss(process):
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform.startswith(("linux", "freebsd")):
        maximum *= 1024
    return max(maximum, process.memory_info().rss)
