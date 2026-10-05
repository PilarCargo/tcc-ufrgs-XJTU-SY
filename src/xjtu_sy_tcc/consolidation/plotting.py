"""Deterministic publication figures for final statistical consolidation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xjtu_phase6_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_figures(
    config,
    native,
    strict,
    paired,
    tests,
    bearing,
    condition,
    life,
    onset,
    consensus,
    costs,
    predictions,
):
    root = config.output_directory / "figures"
    count = 0
    manifest = []
    specs = [
        ("performance/native_macro_mae", native, "experiment", "macro_mae"),
        ("performance/strict_common_macro_mae", strict, "experiment", "macro_mae"),
        ("uncertainty/paired_mean_difference", tests, "candidate", "mean_delta_mae"),
        ("bearing/model_minus_dummy", paired, "candidate", "delta_mae"),
        ("condition/error", condition, "condition_id", "mae"),
        ("life_stage/error", life, "life_stage", "mae"),
        ("onset/error", onset, "estimated_onset_region", "mae"),
        ("features/consensus", consensus.head(15), "feature", "mean_normalized_rank"),
        ("computational/performance_training", costs, "training_seconds", "macro_mae"),
    ]
    for name, table, x, y in specs:
        data = table.groupby(x, as_index=False, sort=False)[y].mean()
        if name == "features/consensus":
            count += _save_consensus(data, root / name, config)
            manifest.append({"figure": name, "source_columns": [x, y]})
            continue
        if name in {
            "performance/native_macro_mae",
            "bearing/model_minus_dummy",
        }:
            count += _save_default_comparison(data, x, y, root / name, config)
            count += _save_portuguese_comparison(data, x, y, root / f"{name}_br", config)
            manifest.append({"figure": name, "source_columns": [x, y]})
            continue
        else:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            ax.bar(data[x].astype(str), data[y], color="0.35")
            ax.set(xlabel=str(x).replace("_", " "), ylabel=str(y).replace("_", " "))
            ax.tick_params(axis="x", rotation=35)
        count += _save(fig, root / name, config)
        manifest.append({"figure": name, "source_columns": [x, y]})
    pivot = bearing.pivot(index="bearing_id", columns="experiment", values="mae")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    fig.colorbar(image, ax=ax, label="MAE (min)")
    count += _save(fig, root / "bearing/mae_heatmap", config)
    manifest.append({"figure": "bearing/mae_heatmap", "source": "per_bearing_metrics"})
    wins = paired.groupby(["candidate", "exact_status"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    wins.plot.bar(stacked=True, ax=ax, color=["0.7", "0.3", "0.5"])
    ax.set(ylabel="Bearings")
    count += _save(fig, root / "uncertainty/win_tie_loss", config)
    representatives = []
    dummy = bearing[bearing.experiment == "dummy_median"]
    for _condition_id, g in dummy.groupby("condition_id"):
        representatives.append(g.iloc[(g.mae - g.mae.median()).abs().argmin()].bearing_id)
    representatives += [
        dummy.loc[dummy.mae.idxmin(), "bearing_id"],
        dummy.loc[dummy.mae.idxmax(), "bearing_id"],
    ]
    for bearing_id in dict.fromkeys(representatives):
        group = predictions[predictions.bearing_id == bearing_id]
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for experiment in (
            "dummy_median",
            "time_only_ridge",
            "selected_features_lstm_k1_ablation",
            "selected_features_plus_time_lstm",
        ):
            g = group[group.experiment == experiment].sort_values("sequence_index")
            ax.plot(g.elapsed_minutes, g.prediction_non_negative_minutes, lw=0.8, label=experiment)
        first = group.sort_values("sequence_index").drop_duplicates("canonical_key")
        ax.plot(first.elapsed_minutes, first.rul_minutes, color="black", label="True RUL")
        ax.legend(fontsize=6)
        ax.set(xlabel="Elapsed time (min)", ylabel="RUL (min)")
        count += _save(fig, root / f"performance/representative_{bearing_id}", config)
    (root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return count


def _save_consensus(data, path, config):
    ordered = data.sort_values("mean_normalized_rank", ascending=True, kind="stable")
    figure, axis = plt.subplots(
        figsize=(10, max(6.0, 0.42 * len(ordered))), constrained_layout=True
    )
    labels = ordered["feature"].astype(str).str.replace("_", " ")
    axis.barh(labels, ordered["mean_normalized_rank"], color="tab:blue")
    axis.set(xlabel="Posto normalizado médio", ylabel="Atributo")
    axis.grid(axis="x", alpha=0.25)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return _save(figure, path, config)


def _save_default_comparison(data, x, y, path, config):
    ordered = data.sort_values(y, kind="stable")
    figure, axis = plt.subplots(
        figsize=(10, max(5.0, 0.45 * len(ordered))), constrained_layout=True
    )
    labels = ordered[x].astype(str).str.replace("_", " ")
    axis.barh(labels, ordered[y], color="0.35")
    axis.invert_yaxis()
    axis.set(xlabel=str(y).replace("_", " "), ylabel=str(x).replace("_", " "))
    for spine in axis.spines.values():
        spine.set_visible(False)
    if path.name == "model_minus_dummy":
        axis.axvline(0.0, color="black", linewidth=0.8)
    return _save(figure, path, config)


def _save_portuguese_comparison(data, x, y, path, config):
    """Save the report-ready Portuguese comparison without title or frame."""
    labels = {
        "dummy_median": "Dummy — mediana",
        "selected_features_lstm_k1_ablation": "LSTM — ablação k=1",
        "selected_features_plus_time_lstm": "LSTM — vibração + tempo",
        "time_only_ridge": "Ridge — apenas tempo",
        "selected_features_plus_time_ridge": "Ridge — vibração + tempo",
        "selected_features_ridge": "Ridge — vibração",
        "selected_features_lstm": "LSTM — contexto temporal",
        "all_features_ridge": "Ridge — todos os atributos",
        "selected_features_hist_gradient_boosting": "HistGradientBoosting — selecionados",
        "selected_features_random_forest": "Random Forest — selecionados",
        "all_features_random_forest": "Random Forest — todos",
        "all_features_hist_gradient_boosting": "HistGradientBoosting — todos",
        "condition_median_countdown": "Contagem regressiva — vida mediana por condição",
        "condition_lifetime_countdown": "Contagem regressiva — vida mediana por condição",
    }
    ordered = data.sort_values(y, kind="stable").reset_index(drop=True)
    bar_labels = ordered[x].astype(str).map(labels).fillna(ordered[x].astype(str))
    colors = ["#4C9BD6"] * len(ordered)
    if colors:
        colors[0] = "#0B4F8A"
    figure, axis = plt.subplots(
        figsize=(10, max(5.0, 0.45 * len(ordered))), constrained_layout=True
    )
    axis.barh(bar_labels, ordered[y], color=colors)
    axis.invert_yaxis()
    if path.name == "native_macro_mae_br":
        axis.set(xlabel="Macro MAE por rolamento (min; menor é melhor)", ylabel="Modelo")
    else:
        axis.set(
            xlabel=(
                "Diferença média pareada de MAE (min; menor é melhor; negativo favorece o modelo)"
            ),
            ylabel="Modelo candidato",
        )
        axis.axvline(0.0, color="black", linewidth=0.8)
    axis.grid(axis="x", alpha=0.25)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return _save(figure, path, config)


def _save(fig, path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=config.plot_dpi)
    count = 1
    if config.generate_pdf:
        fig.savefig(path.with_suffix(".pdf"))
        count += 1
    plt.close(fig)
    return count
