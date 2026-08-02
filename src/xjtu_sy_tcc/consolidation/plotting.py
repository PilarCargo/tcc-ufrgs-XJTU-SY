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
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        data = table.groupby(x, as_index=False)[y].mean()
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


def _save(fig, path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=config.plot_dpi)
    count = 1
    if config.generate_pdf:
        fig.savefig(path.with_suffix(".pdf"))
        count += 1
    plt.close(fig)
    return count
