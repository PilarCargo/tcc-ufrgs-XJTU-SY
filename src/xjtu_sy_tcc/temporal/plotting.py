"""Publication-ready Phase 5 temporal prediction figures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xjtu_phase5_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_figures(
    config,
    predictions,
    comparison,
    candidates,
    life,
    condition,
    onset,
    importance,
    costs,
    test_onsets,
):
    root = config.output_directory / "figures"
    count = 0
    for bearing, group in predictions.groupby("bearing_id", sort=True):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        first = group[group.experiment == "selected_features_lstm"].sort_values("sequence_index")
        ax.plot(first.elapsed_minutes, first.rul_minutes, color="black", label="True RUL")
        for experiment in (
            "selected_features_lstm",
            "selected_features_plus_time_lstm",
            "selected_features_lstm_k1_ablation",
        ):
            g = group[group.experiment == experiment].sort_values("sequence_index")
            ax.plot(
                g.elapsed_minutes,
                g.prediction_non_negative_minutes,
                lw=0.8,
                label=experiment.replace("selected_features_", ""),
            )
        onset_row = test_onsets[test_onsets.bearing_id == bearing].iloc[0]
        if onset_row.estimated_onset_status == "detected":
            ax.axvline(
                onset_row.estimated_onset_elapsed_minutes,
                color="tab:red",
                ls="--",
                label="Estimated degradation onset",
            )
        ax.set(
            title=f"{bearing}: causal temporal RUL", xlabel="Elapsed time (min)", ylabel="RUL (min)"
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        count += _save(fig, root / "by_bearing" / bearing, config)
    count += _bar(
        comparison, "experiment", "macro_mae", "Test macro MAE", root / "macro_mae", config
    )
    count += _bar(
        comparison, "experiment", "macro_rmse", "Test macro RMSE", root / "macro_rmse", config
    )
    count += _bar(
        candidates,
        "sequence_length",
        "validation_macro_mae",
        "Sequence-length validation MAE",
        root / "sequence_length_validation",
        config,
        aggregate=True,
    )
    count += _bar(
        life,
        "life_stage",
        "mae",
        "Error by life stage",
        root / "life_stage",
        config,
        aggregate=True,
    )
    count += _bar(
        condition,
        "condition_id",
        "mae",
        "Error by operating condition",
        root / "condition",
        config,
        aggregate=True,
    )
    count += _bar(
        onset,
        "estimated_onset_region",
        "mae",
        "Error around estimated onset",
        root / "onset_region",
        config,
        aggregate=True,
    )
    count += _bar(
        importance.head(15),
        "feature",
        "importance_macro_mae_increase",
        "Feature occlusion importance",
        root / "feature_occlusion",
        config,
    )
    count += _bar(
        costs,
        "experiment",
        "candidate_training_seconds",
        "Training cost",
        root / "training_cost",
        config,
        aggregate=True,
    )
    for residual, name in ((False, "predicted_vs_true"), (True, "residuals_vs_true")):
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for experiment, g in predictions.groupby("experiment"):
            ax.scatter(
                g.rul_minutes,
                g.prediction_error_minutes if residual else g.prediction_non_negative_minutes,
                s=3,
                alpha=0.25,
                label=experiment,
            )
        ax.set(
            xlabel="True RUL (min)", ylabel="Residual (min)" if residual else "Predicted RUL (min)"
        )
        ax.legend(fontsize=6)
        count += _save(fig, root / name, config)
    return count


def _bar(table, x, y, title, path, config, aggregate=False):
    data = table.groupby(x, as_index=False)[y].mean() if aggregate else table
    if path.name == "feature_occlusion":
        ordered = data.sort_values(y, ascending=True, kind="stable")
        model_labels = {
            "selected_features_lstm": "LSTM com contexto temporal",
            "selected_features_plus_time_lstm": "LSTM com vibração + tempo",
            "selected_features_lstm_k1_ablation": "LSTM, ablação k=1",
        }
        labels = ordered[x].astype(str).str.replace("_", " ")
        if "experiment" in ordered:
            labels = (
                labels
                + " — "
                + ordered["experiment"].map(model_labels).fillna(ordered["experiment"].astype(str))
            )
        fig, ax = plt.subplots(figsize=(10, max(5.0, 0.42 * len(ordered))), constrained_layout=True)
        ax.barh(labels, ordered[y], color="tab:blue")
        ax.set(xlabel="Aumento do macro MAE por oclusão (min)", ylabel="Atributo")
        ax.grid(axis="x", alpha=0.25)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return _save(fig, path, config)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(data[x].astype(str), data[y], color="0.35")
    ax.set(title=title, ylabel=y.replace("_", " "))
    ax.tick_params(axis="x", rotation=35)
    return _save(fig, path, config)


def _save(fig, path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=config.plot_dpi)
    count = 1
    if config.generate_pdf:
        fig.savefig(path.with_suffix(".pdf"))
        count += 1
    plt.close(fig)
    return count
