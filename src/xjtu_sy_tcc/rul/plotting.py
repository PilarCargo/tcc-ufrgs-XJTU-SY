"""Publication-ready Phase 4 prediction, error, importance, and cost figures."""

# ruff: noqa: E402 -- configure writable Matplotlib cache before importing it.

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_CACHE = Path(tempfile.gettempdir()) / "xjtu_sy_tcc_matplotlib"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xjtu_sy_tcc.config.phase4 import Phase4Config


def generate_phase4_figures(
    config: Phase4Config,
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    condition: pd.DataFrame,
    life: pd.DataFrame,
    onset: pd.DataFrame,
    importance: pd.DataFrame,
    costs: pd.DataFrame,
    test_onsets: pd.DataFrame,
) -> int:
    count = 0
    directory = config.output_directory / "figures"
    main = (
        "selected_features_ridge",
        "selected_features_random_forest",
        "selected_features_hist_gradient_boosting",
    )
    for bearing in sorted(predictions["bearing_id"].unique()):
        scoped = predictions[
            (predictions["bearing_id"] == bearing) & predictions["experiment"].isin(main)
        ]
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        first = scoped[scoped["experiment"] == main[0]].sort_values("sequence_index")
        axis.plot(
            first["elapsed_minutes"],
            first["rul_minutes"],
            color="black",
            linewidth=1.5,
            label="True RUL",
        )
        for experiment in main:
            group = scoped[scoped["experiment"] == experiment].sort_values("sequence_index")
            axis.plot(
                group["elapsed_minutes"],
                group["prediction_non_negative_minutes"],
                linewidth=0.8,
                label=experiment.replace("selected_features_", ""),
            )
        onset_row = test_onsets[test_onsets["bearing_id"] == bearing].iloc[0]
        if onset_row["estimated_onset_status"] == "detected":
            axis.axvline(
                float(onset_row["estimated_onset_elapsed_minutes"]),
                color="tab:red",
                linestyle="--",
                label="Estimated degradation onset (post-hoc)",
            )
        axis.set(
            title=f"{bearing}: unseen-bearing RUL prediction",
            xlabel="Elapsed time (min)",
            ylabel="RUL (min)",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
        count += _save(figure, directory / "by_bearing" / bearing, config)
    count += _bar(
        comparison,
        "experiment",
        "macro_mae",
        "Test macro MAE",
        "MAE (min)",
        directory / "model_macro_mae",
        config,
    )
    count += _bar(
        comparison,
        "experiment",
        "macro_rmse",
        "Test macro RMSE",
        "RMSE (min)",
        directory / "model_macro_rmse",
        config,
    )
    count += _scatter(predictions, directory / "predicted_vs_true", config, residual=False)
    count += _scatter(predictions, directory / "residuals_vs_true", config, residual=True)
    count += _grouped_error(
        life,
        "life_stage",
        "Error by retrospective life stage",
        directory / "error_by_life_stage",
        config,
    )
    count += _grouped_error(
        condition,
        "condition_id",
        "Error by operating condition",
        directory / "error_by_condition",
        config,
    )
    count += _grouped_error(
        onset,
        "estimated_onset_region",
        "Error around estimated onset (post-hoc)",
        directory / "error_by_estimated_onset",
        config,
    )
    ablation = comparison[
        comparison["experiment"].isin(["selected_features_ridge", "all_features_ridge"])
    ]
    count += _bar(
        ablation,
        "experiment",
        "macro_mae",
        "Selected versus all vibration features",
        "Macro MAE (min)",
        directory / "selected_vs_all",
        config,
    )
    time_vibration = comparison[
        comparison["experiment"].isin(
            ["time_only_ridge", "selected_features_ridge", "selected_features_plus_time_ridge"]
        )
    ]
    count += _bar(
        time_vibration,
        "experiment",
        "macro_mae",
        "Time-only versus vibration experiments",
        "Macro MAE (min)",
        directory / "time_vs_vibration",
        config,
    )
    ranked_importance = importance.assign(
        combined_importance=importance["ridge_mean_absolute_coefficient"]
        + importance["permutation_mean_macro_mae_increase"].clip(lower=0)
    ).nlargest(20, "combined_importance")
    count += _bar(
        ranked_importance,
        "feature",
        "combined_importance",
        "Predictive feature importance (non-causal)",
        "Combined descriptive importance",
        directory / "feature_importance",
        config,
    )
    cost_summary = costs.groupby("experiment", as_index=False).agg(
        training_seconds=("training_seconds", "sum"),
        inference_seconds=("test_inference_seconds", "sum"),
    )
    count += _cost(cost_summary, directory / "training_inference_cost", config)
    return count


def _bar(data, x, y, title, ylabel, path, config):
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.bar(data[x].astype(str), data[y].astype(float), color="tab:blue")
    axis.set(title=title, xlabel=x.replace("_", " ").title(), ylabel=ylabel)
    axis.tick_params(axis="x", rotation=70 if len(data) > 4 else 0)
    axis.grid(axis="y", alpha=0.25)
    return _save(figure, path, config)


def _scatter(predictions, path, config, residual):
    experiments = [
        "time_only_ridge",
        "selected_features_ridge",
        "selected_features_random_forest",
        "selected_features_hist_gradient_boosting",
    ]
    scoped = predictions[predictions["experiment"].isin(experiments)]
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for experiment, group in scoped.groupby("experiment", sort=True):
        y = (
            group["prediction_error_minutes"]
            if residual
            else group["prediction_non_negative_minutes"]
        )
        axis.scatter(group["rul_minutes"], y, s=5, alpha=0.18, label=experiment)
    if residual:
        axis.axhline(0, color="black", linewidth=1)
        ylabel = "Prediction error (min)"
        title = "Residuals versus true RUL"
    else:
        limit = float(scoped["rul_minutes"].max())
        axis.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
        ylabel = "Predicted RUL (min)"
        title = "Predicted versus true RUL"
    axis.set(title=title, xlabel="True RUL (min)", ylabel=ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    return _save(figure, path, config)


def _grouped_error(data, grouping, title, path, config):
    selected = data[
        data["experiment"].isin(
            [
                "time_only_ridge",
                "selected_features_ridge",
                "selected_features_random_forest",
                "selected_features_hist_gradient_boosting",
            ]
        )
    ]
    pivot = selected.pivot(index=grouping, columns="experiment", values="mae")
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    pivot.plot(kind="bar", ax=axis)
    axis.set(title=title, xlabel=grouping.replace("_", " ").title(), ylabel="MAE (min)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7)
    return _save(figure, path, config)


def _cost(data, path, config):
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    positions = np.arange(len(data))
    axis.bar(positions - 0.2, data["training_seconds"], width=0.4, label="Training")
    axis.bar(positions + 0.2, data["inference_seconds"], width=0.4, label="Test inference")
    axis.set(
        title="Training and inference cost",
        xlabel="Experiment",
        ylabel="Seconds",
        xticks=positions,
        xticklabels=data["experiment"].str.replace("_", "\n"),
    )
    axis.tick_params(axis="x", labelsize=7)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    return _save(figure, path, config)


def _save(figure: plt.Figure, path: Path, config: Phase4Config) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = [(path.with_suffix(".png"), {"dpi": config.plot_dpi})]
    if config.generate_pdf:
        outputs.append((path.with_suffix(".pdf"), {}))
    for output, options in outputs:
        temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
        temporary.unlink(missing_ok=True)
        figure.savefig(temporary, **options)
        temporary.replace(output)
    plt.close(figure)
    return len(outputs)
