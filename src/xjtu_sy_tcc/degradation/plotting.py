"""Publication-ready Phase 3 figures with explicit estimated-onset terminology."""

# ruff: noqa: E402 -- configure a writable Matplotlib cache before importing it.

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

from xjtu_sy_tcc.config.phase3 import Phase3Config


def generate_phase3_figures(
    config: Phase3Config,
    features: pd.DataFrame,
    rankings: pd.DataFrame,
    health: pd.DataFrame,
    onsets: pd.DataFrame,
    sensitivity: pd.DataFrame,
    artifacts: dict[int, dict[str, object]],
) -> int:
    """Generate one test-fold figure per bearing and six scientific summaries."""
    count = 0
    bearing_directory = config.figure_directory / "degradation" / "by_bearing"
    for bearing in sorted(features["bearing_id"].unique()):
        onset = onsets[(onsets["bearing_id"] == bearing) & (onsets["subset"] == "test")].iloc[0]
        fold = int(onset["fold"])
        trajectory = features[features["bearing_id"] == bearing].sort_values("sequence_index")
        indicator = health[
            (health["fold"] == fold) & (health["bearing_id"] == bearing)
        ].sort_values("sequence_index")
        selected = rankings[(rankings["fold"] == fold) & rankings["selected"]].head(3)["feature"]
        figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
        for feature in selected:
            values = trajectory[feature].to_numpy(float)
            center = np.median(values)
            scale = np.subtract(*np.percentile(values, [75, 25])) or 1.0
            axes[0].plot(
                trajectory["elapsed_minutes"],
                (values - center) / scale,
                linewidth=0.8,
                label=feature,
            )
        axes[0].set(
            ylabel="Robust standardized value (-)",
            title=f"{bearing}: degradation-sensitive features",
        )
        axes[0].legend(fontsize=7)
        axes[1].plot(
            indicator["elapsed_minutes"],
            indicator["baseline_centered_health_indicator"],
            linewidth=0.7,
            alpha=0.55,
            label="Raw health indicator",
        )
        axes[1].plot(
            indicator["elapsed_minutes"],
            indicator["smoothed_health_indicator"],
            linewidth=1.1,
            label=(
                f"Health indicator ({indicator.iloc[0]['smoothing_method']}, "
                f"window={int(indicator.iloc[0]['smoothing_window'])})"
            ),
        )
        baseline_end = float(
            indicator.iloc[int(indicator.iloc[0]["baseline_acquisitions"]) - 1]["elapsed_minutes"]
        )
        axes[1].axvspan(
            0, baseline_end, alpha=0.15, color="gray", label="Initial calibration interval"
        )
        if onset["estimated_onset_status"] == "detected":
            axes[1].axvline(
                float(onset["estimated_onset_elapsed_minutes"]),
                color="tab:red",
                linestyle="--",
                label="Estimated degradation onset",
            )
        axes[1].axvline(
            float(indicator["elapsed_minutes"].iloc[-1]),
            color="black",
            linestyle=":",
            label="Final failure endpoint",
        )
        axes[1].set(xlabel="Elapsed time (min)", ylabel="Health indicator (-)")
        for axis in axes:
            axis.grid(alpha=0.25)
        axes[1].legend(fontsize=7)
        count += _save(figure, bearing_directory / bearing, config)

    summary_directory = config.figure_directory / "degradation" / "summary"
    count += _health_indicator_condition_comparison(
        health,
        summary_directory / "comparacao_health_indicator_condicoes",
        config,
    )
    prognostic_directory = config.figure_directory / "prognostics"
    test_onsets = onsets[onsets["subset"] == "test"].copy()
    count += _bar_figure(
        test_onsets,
        "bearing_id",
        "life_fraction_at_estimated_onset",
        "Estimated onset life fraction",
        "Life fraction (-)",
        summary_directory / "estimated_onset_life_fraction",
        config,
    )
    detected_test = test_onsets[test_onsets["estimated_onset_status"] == "detected"]
    count += _box_figure(
        detected_test,
        "condition_id",
        "rul_at_estimated_onset",
        "Estimated-onset RUL by operating condition",
        "RUL (min)",
        summary_directory / "estimated_onset_rul_by_condition",
        config,
    )
    stability = (
        sensitivity.groupby("bearing_id", sort=True)["onset_iqr_minutes"].median().reset_index()
    )
    count += _bar_figure(
        stability,
        "bearing_id",
        "onset_iqr_minutes",
        "Breakpoint stability by bearing",
        "Median onset IQR (min)",
        summary_directory / "breakpoint_stability",
        config,
    )
    frequency = (
        rankings[rankings["selected"]]
        .groupby("feature")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="fold_count")
    )
    count += _bar_figure(
        frequency,
        "feature",
        "fold_count",
        "Selected-feature frequency",
        "Selected folds",
        prognostic_directory / "selected_feature_frequency",
        config,
    )
    heatmap_features = frequency.head(15)["feature"].tolist()
    heatmap = rankings[rankings["feature"].isin(heatmap_features)].pivot(
        index="feature", columns="fold", values="composite_score"
    )
    count += _heatmap_figure(
        heatmap,
        prognostic_directory / "feature_quality_heatmap",
        config,
    )
    pca = pd.DataFrame(
        [
            {"fold": fold, "pc1_explained_variance": artifact["explained_variance_ratio"][0]}
            for fold, artifact in artifacts.items()
        ]
    )
    count += _bar_figure(
        pca,
        "fold",
        "pc1_explained_variance",
        "PCA first-component explained variance",
        "Explained variance ratio (-)",
        summary_directory / "pca_explained_variance",
        config,
    )
    non_detection = (
        sensitivity.assign(not_detected=sensitivity["estimated_onset_status"].eq("not_detected"))
        .groupby("parameter_id")["not_detected"]
        .sum()
        .reset_index()
    )
    count += _bar_figure(
        non_detection,
        "parameter_id",
        "not_detected",
        "Non-detection count by parameter configuration",
        "Non-detections",
        summary_directory / "non_detection_by_configuration",
        config,
    )
    return count


def _health_indicator_condition_comparison(
    health: pd.DataFrame, path: Path, config: Phase3Config
) -> int:
    """Compare representative bearings while labeling panels by operating condition."""
    representatives = (("Bearing1_1", 1), ("Bearing2_1", 2), ("Bearing3_1", 3))
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
    for index, (axis, (bearing, condition)) in enumerate(zip(axes, representatives, strict=True)):
        indicator = health[
            (health["fold"] == 1)
            & (health["subset"] == "test")
            & (health["bearing_id"] == bearing)
        ].sort_values("sequence_index")
        life_fraction = np.linspace(0.0, 1.0, len(indicator))
        axis.plot(
            life_fraction,
            indicator["baseline_centered_health_indicator"],
            color="#8EC0E8",
            linewidth=0.8,
            alpha=0.8,
            label="HI bruto centralizado",
        )
        axis.plot(
            life_fraction,
            indicator["smoothed_health_indicator"],
            color="#E66100",
            linewidth=1.8,
            label="HI suavizado",
        )
        axis.axvspan(0.8, 1.0, color="#F2D675", alpha=0.25, label="20% finais")
        axis.axvline(
            1.0, color="black", linestyle=":", linewidth=1.4, label="Endpoint experimental"
        )
        axis.set(
            title=f"Condição operacional {condition}",
            xlabel="Fração da vida experimental",
            ylabel="Indicador de saúde (PC1 centralizada)" if index == 0 else None,
        )
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    return _save(figure, path, config)


def _bar_figure(data, x, y, title, ylabel, path, config) -> int:
    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    available = data.dropna(subset=[y])
    axis.bar(available[x].astype(str), available[y].astype(float), color="tab:blue")
    axis.set(title=title, xlabel=x.replace("_", " ").title(), ylabel=ylabel)
    axis.tick_params(axis="x", rotation=75 if len(available) > 10 else 0)
    axis.grid(axis="y", alpha=0.25)
    if path.name == "estimated_onset_life_fraction":
        axis.set(title=None, xlabel="Rolamento", ylabel="Fração da vida consumida (-)")
        axis.set_ylim(0.0, 1.0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    count = _save(figure, path, config)
    if path.name == "estimated_onset_life_fraction":
        report_figure, report_axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
        report_axis.bar(available[x].astype(str), available[y].astype(float), color="#4C9BD6")
        report_axis.set(xlabel="Rolamento", ylabel="Fração da vida consumida (-)")
        report_axis.set_ylim(0.0, 1.0)
        report_axis.tick_params(axis="x", rotation=75 if len(available) > 10 else 0)
        report_axis.grid(axis="y", alpha=0.25)
        for spine in report_axis.spines.values():
            spine.set_visible(False)
        count += _save(report_figure, path.with_name(f"{path.name}_br"), config)
    elif path.name == "pca_explained_variance":
        report_figure, report_axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
        percentages = available[y].astype(float) * 100.0
        bars = report_axis.bar(
            available[x].astype(str), percentages, color="#4C9BD6"
        )
        report_axis.set(xlabel="Partição", ylabel="Variância explicada pela PC1 (%)")
        report_axis.set_ylim(0.0, 100.0)
        report_axis.grid(axis="y", alpha=0.25)
        report_axis.bar_label(bars, fmt="%.2f%%", padding=3)
        for spine in report_axis.spines.values():
            spine.set_visible(False)
        count += _save(report_figure, path.with_name(f"{path.name}_br"), config)
    return count


def _box_figure(data, x, y, title, ylabel, path, config) -> int:
    figure, axis = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    groups = [group[y].dropna().to_numpy(float) for _, group in data.groupby(x, sort=True)]
    labels = [str(name) for name, _ in data.groupby(x, sort=True)]
    if groups:
        axis.boxplot(groups, tick_labels=labels)
    axis.set(title=title, xlabel="Operating condition", ylabel=ylabel)
    axis.grid(axis="y", alpha=0.25)
    return _save(figure, path, config)


def _heatmap_figure(data: pd.DataFrame, path: Path, config: Phase3Config) -> int:
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(data.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set(
        title="Training-only feature quality across folds",
        xlabel="Fold",
        ylabel="Feature",
        xticks=np.arange(len(data.columns)),
        xticklabels=[str(value) for value in data.columns],
        yticks=np.arange(len(data.index)),
        yticklabels=data.index,
    )
    figure.colorbar(image, ax=axis, label="Composite quality score (-)")
    return _save(figure, path, config)


def _save(figure: plt.Figure, path: Path, config: Phase3Config) -> int:
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
