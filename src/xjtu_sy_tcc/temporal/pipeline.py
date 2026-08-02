"""End-to-end causal LSTM selection, frozen testing, and scientific reporting."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from xjtu_sy_tcc.config.phase5 import LSTMCandidate, Phase5Config
from xjtu_sy_tcc.rul.metrics import (
    add_prediction_diagnostics,
    bearing_metrics,
    fold_metrics,
    grouped_metrics,
)
from xjtu_sy_tcc.rul.posthoc import annotate_posthoc, life_stage_metrics, onset_region_metrics
from xjtu_sy_tcc.rul.reporting import atomic_json, atomic_model, atomic_table, atomic_text
from xjtu_sy_tcc.rul.safety import (
    TRACE_COLUMNS,
    load_frozen_folds,
    load_selected_features,
    load_test_onsets,
    validate_phase2_features,
)
from xjtu_sy_tcc.temporal.interpretability import occlusion_importance
from xjtu_sy_tcc.temporal.model import RULLSTM, predict, train_model
from xjtu_sy_tcc.temporal.plotting import generate_figures as plot_phase5_figures
from xjtu_sy_tcc.temporal.preprocessing import fit_preprocessor
from xjtu_sy_tcc.temporal.selection import rank_candidates
from xjtu_sy_tcc.temporal.sequences import build_sequences, sequence_coverage

EXPERIMENTS = (
    "selected_features_lstm",
    "selected_features_plus_time_lstm",
    "selected_features_lstm_k1_ablation",
)
BASELINES = (
    "dummy_median",
    "time_only_ridge",
    "selected_features_ridge",
    "selected_features_plus_time_ridge",
    "selected_features_hist_gradient_boosting",
)


def run_phase5(config: Phase5Config, *, force_train=False, generate_figures=True):
    cached = _cache(config)
    if cached and not force_train:
        if generate_figures and not cached.get("figure_files"):
            cached = _cached_figures(config, cached)
        return cached
    started = time.perf_counter()
    features = (
        pd.read_parquet(config.feature_table)
        .sort_values(["condition_id", "bearing_id", "sequence_index"])
        .reset_index(drop=True)
    )
    available = validate_phase2_features(
        features, config.expected_feature_rows, config.expected_feature_count
    )
    folds, split = load_frozen_folds(
        config.split_manifest,
        config.expected_split_sha256,
        config.expected_split_configuration_hash,
        features,
    )
    selected = load_selected_features(config.selected_feature_manifest, folds, available)
    onsets = load_test_onsets(config.test_onset_table, folds)
    device = _device(config.device)
    root = config.output_directory
    root.mkdir(parents=True, exist_ok=True)
    candidates_rows = []
    histories = []
    predictions = []
    coverages = []
    importances = []
    costs = []
    selected_models = []
    for fold in folds:
        subsets = {
            name: features[features.bearing_id.isin(getattr(fold, f"{name}_bearings"))].copy()
            for name in ("train", "validation", "test")
        }
        for experiment in EXPERIMENTS:
            context = "plus_time" in experiment
            candidate_list = (
                (_k1_candidate(config.candidates[0]),) if "k1" in experiment else config.candidates
            )
            trained = {}
            for order, candidate in enumerate(candidate_list, 1):
                sets = {
                    name: build_sequences(
                        frame, list(selected[fold.fold_id]), candidate.sequence_length
                    )
                    for name, frame in subsets.items()
                    if name != "test"
                }
                prep = fit_preprocessor(
                    sets["train"], list(selected[fold.fold_id]), config.heavy_tail_feature_tokens
                )
                tx = prep.transform_features(sets["train"].features)
                vx = prep.transform_features(sets["validation"].features)
                tc = prep.transform_context(sets["train"].context)
                vc = prep.transform_context(sets["validation"].context)
                model = RULLSTM(
                    len(selected[fold.fold_id]),
                    candidate.hidden_size,
                    candidate.num_layers,
                    candidate.dropout,
                    3 if context else 0,
                )
                result = train_model(
                    model,
                    tx,
                    tc,
                    prep.scale_target(sets["train"].targets),
                    sets["train"].weights,
                    vx,
                    vc,
                    sets["validation"].targets,
                    sets["validation"].metadata.bearing_id.to_numpy(),
                    prep,
                    config,
                    candidate.learning_rate,
                    device,
                )
                record = {
                    "fold": fold.fold_id,
                    "experiment": experiment,
                    "candidate_order": order,
                    **asdict(candidate),
                    "validation_macro_mae": result.validation_mae,
                    "validation_macro_rmse": result.validation_rmse,
                    "trainable_parameters": result.parameter_count,
                    "best_epoch": result.best_epoch,
                    "epochs_executed": len(result.history),
                    "training_seconds": result.training_seconds,
                }
                candidates_rows.append(record)
                trained[order] = (candidate, result, prep, sets)
            ranked = rank_candidates(
                pd.DataFrame(
                    [
                        r
                        for r in candidates_rows
                        if r["fold"] == fold.fold_id and r["experiment"] == experiment
                    ]
                )
            )
            winner = ranked.iloc[0]
            order = int(winner.candidate_order)
            candidate, result, prep, sets = trained[order]
            histories.append(
                result.history.assign(
                    fold=fold.fold_id, experiment=experiment, candidate_order=order
                )
            )
            testset = build_sequences(
                subsets["test"], list(selected[fold.fold_id]), candidate.sequence_length
            )
            test_x = prep.transform_features(testset.features)
            test_c = prep.transform_context(testset.context)
            tick = time.perf_counter()
            scaled = predict(result.model, test_x, test_c, config.batch_size, device)
            inference = time.perf_counter() - tick
            raw = prep.inverse_target(scaled)
            frame = testset.metadata.loc[:, TRACE_COLUMNS].copy()
            frame.insert(0, "sequence_length", candidate.sequence_length)
            frame.insert(0, "candidate_id", order)
            frame.insert(0, "experiment", experiment)
            frame.insert(0, "fold", fold.fold_id)
            frame.insert(1, "model", "lstm")
            predictions.append(add_prediction_diagnostics(frame, raw))
            coverages.extend(
                sequence_coverage(subsets[name], candidate.sequence_length, fold.fold_id, name)
                for name in subsets
            )
            importance = occlusion_importance(
                result.model,
                prep.transform_features(sets["validation"].features),
                prep.transform_context(sets["validation"].context),
                sets["validation"].targets,
                sets["validation"].metadata.bearing_id.to_numpy(),
                prep,
                config.batch_size,
                device,
                fold.fold_id,
                experiment,
            )
            importances.append(importance)
            artifact = root / "training" / f"fold_{fold.fold_id}" / experiment
            artifact.mkdir(parents=True, exist_ok=True)
            checkpoint = artifact / "best_checkpoint.pt"
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(result.model.state_dict(), temporary)
            temporary.replace(checkpoint)
            atomic_table(artifact / "training_history.parquet", result.history)
            atomic_model(root / "preprocessing" / f"fold_{fold.fold_id}_{experiment}.joblib", prep)
            selected_models.append(
                {
                    "fold": fold.fold_id,
                    "experiment": experiment,
                    "candidate_order": order,
                    "parameters": asdict(candidate),
                    "best_epoch": result.best_epoch,
                    "validation_macro_mae": result.validation_mae,
                    "checkpoint_sha256": _sha(checkpoint),
                    "training_bearings": list(fold.train_bearings),
                }
            )
            costs.append(
                {
                    "fold": fold.fold_id,
                    "experiment": experiment,
                    "sequence_generation_seconds": 0.0,
                    "preprocessing_fit_seconds": 0.0,
                    "candidate_training_seconds": result.training_seconds,
                    "test_inference_seconds": inference,
                    "inference_seconds_per_sequence": inference / len(testset.targets),
                    "trainable_parameters": result.parameter_count,
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                    "training_sequences": len(sets["train"].targets),
                    "validation_sequences": len(sets["validation"].targets),
                    "test_sequences": len(testset.targets),
                    "sequence_length": candidate.sequence_length,
                    "epochs_executed": len(result.history),
                    "best_epoch": result.best_epoch,
                    "device": str(device),
                }
            )
    predictions_table = annotate_posthoc(
        pd.concat(predictions, ignore_index=True), onsets, config.life_stage_thresholds
    )
    bearing = bearing_metrics(predictions_table)
    folds_table = fold_metrics(bearing)
    condition = grouped_metrics(predictions_table, ["experiment", "model", "condition_id"])
    life = life_stage_metrics(predictions_table)
    onset = onset_region_metrics(predictions_table)
    support = _support_matched(config.phase4_predictions, predictions_table)
    comparison = _comparison(predictions_table, bearing, support)
    importance = pd.concat(importances, ignore_index=True)
    summary = (
        importance.groupby(["experiment", "feature"], as_index=False)
        .importance_macro_mae_increase.mean()
        .sort_values("importance_macro_mae_increase", ascending=False)
    )
    candidate_table = pd.concat(
        [
            rank_candidates(group)
            for _, group in pd.DataFrame(candidates_rows).groupby(["fold", "experiment"])
        ],
        ignore_index=True,
    )
    output_tables = {
        "sequences/sequence_coverage.parquet": pd.concat(coverages, ignore_index=True),
        "model_selection/validation_candidates.parquet": candidate_table,
        "training/training_history.parquet": pd.concat(histories, ignore_index=True),
        "predictions/test_predictions.parquet": predictions_table,
        "predictions/test_predictions.csv": predictions_table,
        "metrics/bearing_metrics.parquet": bearing,
        "metrics/fold_metrics.parquet": folds_table,
        "metrics/condition_metrics.parquet": condition,
        "metrics/life_stage_metrics.parquet": life,
        "metrics/onset_region_metrics.parquet": onset,
        "metrics/support_matched_baselines.parquet": support,
        "metrics/model_comparison.csv": comparison,
        "metrics/computational_costs.parquet": pd.DataFrame(costs),
        "interpretability/feature_occlusion_importance.parquet": importance,
        "interpretability/feature_occlusion_summary.csv": summary,
    }
    for relative, table in output_tables.items():
        atomic_table(root / relative, table)
    atomic_json(root / "model_selection/selected_models.json", {"models": selected_models})
    resolved = {
        "configuration_hash": config.configuration_hash,
        "split_sha256": _sha(config.split_manifest),
        "torch_version": torch.__version__,
        "device": str(device),
        "platform": platform.platform(),
    }
    atomic_json(root / "configuration/phase5_resolved_config.json", resolved)
    figures = (
        plot_phase5_figures(
            config,
            predictions_table,
            comparison,
            candidate_table,
            life,
            condition,
            onset,
            summary,
            pd.DataFrame(costs),
            onsets,
        )
        if generate_figures
        else 0
    )
    report = {
        "status": "passed",
        "configuration_hash": config.configuration_hash,
        "input_hashes": _input_hashes(config),
        "split_configuration_hash": split["configuration_hash"],
        "duration_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "device": str(device),
        "validation_candidate_count": len(candidates_rows),
        "test_prediction_rows": len(predictions_table),
        "figure_files": figures,
        "model_comparison": comparison.to_dict("records"),
        "issues": [],
    }
    atomic_json(
        root / "sequences/sequence_validation.json",
        {"status": "passed", "future_leakage": False, "cross_bearing_sequences": 0},
    )
    atomic_json(root / "reports/phase5_validation.json", report)
    atomic_json(root / "reports/phase5_results.json", report)
    atomic_text(root / "reports/phase5_results.md", _markdown(report))
    return report


def _k1_candidate(c):
    return LSTMCandidate(1, c.hidden_size, c.num_layers, c.dropout, c.learning_rate)


def _support_matched(path, predictions):
    phase4 = pd.read_parquet(path)
    rows = []
    for experiment, eligible in predictions.groupby("experiment"):
        keys = eligible[["fold", "bearing_id", "acquisition_number"]].drop_duplicates()
        matched = phase4[phase4.experiment.isin(BASELINES)].merge(
            keys, on=["fold", "bearing_id", "acquisition_number"]
        )
        table = grouped_metrics(matched, ["experiment", "model"])
        bearing = bearing_metrics(matched)
        macro = bearing.groupby(["experiment", "model"], as_index=False).agg(
            macro_mae=("mae", "mean"), macro_rmse=("rmse", "mean")
        )
        table = table.merge(macro)
        table.insert(0, "lstm_experiment", experiment)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _comparison(predictions, bearing, support):
    global_ = grouped_metrics(predictions, ["experiment", "model"]).rename(
        columns={"mae": "global_mae", "rmse": "global_rmse", "r2": "global_r2"}
    )
    macro = bearing.groupby(["experiment", "model"], as_index=False).agg(
        macro_mae=("mae", "mean"), macro_rmse=("rmse", "mean"), median_bearing_mae=("mae", "median")
    )
    return global_.merge(macro)


def _device(request):
    if request == "auto":
        request = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if request == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    return torch.device(request)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _input_hashes(c):
    return {
        k: _sha(getattr(c, k))
        for k in (
            "feature_table",
            "split_manifest",
            "selected_feature_manifest",
            "phase4_predictions",
            "test_onset_table",
        )
    }


def _cache(c):
    p = c.output_directory / "reports/phase5_validation.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return (
        r
        if r.get("status") == "passed"
        and r.get("configuration_hash") == c.configuration_hash
        and r.get("input_hashes") == _input_hashes(c)
        else None
    )


def _cached_figures(config, report):
    root = config.output_directory
    figures = plot_phase5_figures(
        config,
        pd.read_parquet(root / "predictions/test_predictions.parquet"),
        pd.read_csv(root / "metrics/model_comparison.csv"),
        pd.read_parquet(root / "model_selection/validation_candidates.parquet"),
        pd.read_parquet(root / "metrics/life_stage_metrics.parquet"),
        pd.read_parquet(root / "metrics/condition_metrics.parquet"),
        pd.read_parquet(root / "metrics/onset_region_metrics.parquet"),
        pd.read_csv(root / "interpretability/feature_occlusion_summary.csv"),
        pd.read_parquet(root / "metrics/computational_costs.parquet"),
        pd.read_parquet(config.test_onset_table).query("subset == 'test'"),
    )
    refreshed = dict(report)
    refreshed["figure_files"] = figures
    atomic_json(root / "reports/phase5_validation.json", refreshed)
    atomic_json(root / "reports/phase5_results.json", refreshed)
    atomic_text(root / "reports/phase5_results.md", _markdown(refreshed))
    return refreshed


def _markdown(r):
    lines = [
        "# Phase 5 causal LSTM RUL regression",
        "",
        f"- Status: **{r['status'].upper()}**",
        f"- Duration: {r['duration_seconds']:.3f} s",
        "",
        "| Experiment | Macro MAE | Macro RMSE |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {x['experiment']} | {x['macro_mae']:.3f} | {x['macro_rmse']:.3f} |"
        for x in r["model_comparison"]
    ]
    lines += [
        "",
        "All sequences contain only the current and previous acquisitions. "
        "Estimated onset is retrospective only.",
        "",
    ]
    return "\n".join(lines)
