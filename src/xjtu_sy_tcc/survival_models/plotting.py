"""Publication-ready Phase 8 figures generated from frozen tabular results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from xjtu_sy_tcc.rul.reporting import atomic_json  # noqa: E402


def generate_phase8_figures(output: Path, dpi: int = 200, pdf: bool = True) -> int:
    metrics = pd.read_parquet(output / "metrics/scenario_metrics.parquet")
    bearing = pd.read_parquet(output / "metrics/bearing_metrics.parquet")
    calibration = pd.read_parquet(output / "metrics/calibration_metrics.parquet")
    predictions = pd.read_parquet(output / "predictions/landmark_predictions.parquet")
    costs = pd.read_parquet(output / "computational/computational_costs.parquet")
    coefficients = pd.read_parquet(output / "interpretability/cox_coefficients.parquet")
    km = pd.read_parquet(output / "predictions/kaplan_meier_onset_curves.parquet")
    manifest = []

    def save(fig, relative: str, source: str) -> None:
        path = output / "figures" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        if pdf:
            fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        manifest.append({"figure": str(path), "source": source, "selection": "deterministic"})

    for metric, label, name in (
        ("macro_integrated_brier_score", "Macro integrated Brier score", "ibs_by_model"),
        ("macro_ipcw_c_index", "Macro IPCW concordance", "ipcw_concordance_by_model"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        metrics.pivot(index="scenario", columns="model", values=metric).plot.bar(ax=ax)
        ax.set_ylabel(label)
        ax.set_xlabel("Survival scenario")
        ax.legend(title="Model", fontsize=7)
        save(fig, f"discrimination/{name}.png", "metrics/scenario_metrics.parquet")

    fig, ax = plt.subplots(figsize=(7, 6))
    scoped = calibration[calibration.horizon_minutes.isin([60.0, 120.0, 240.0])]
    for horizon, group in scoped.groupby("horizon_minutes"):
        ax.scatter(
            group.mean_predicted_failure_probability,
            group.observed_event_probability,
            s=16,
            alpha=0.7,
            label=f"{horizon:g} min",
        )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Mean predicted failure probability", ylabel="Observed event proportion")
    ax.legend()
    save(fig, "calibration/calibration_horizons.png", "metrics/calibration_metrics.parquet")

    fig, ax = plt.subplots(figsize=(10, 6))
    full = bearing[bearing.scenario == "full_event"]
    heat = full.pivot(index="bearing_id", columns="model", values="integrated_brier_score")
    image = ax.imshow(heat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("Full-event bearing-level integrated Brier score")
    fig.colorbar(image, ax=ax, label="IBS")
    save(fig, "bearing/per_bearing_ibs_heatmap.png", "metrics/bearing_metrics.parquet")

    fig, ax = plt.subplots(figsize=(7, 6))
    full_predictions = predictions[
        (predictions.scenario == "full_event") & predictions.median_survival_available
    ]
    for model, group in full_predictions.groupby("model"):
        ax.scatter(
            group.true_time_to_failure_minutes,
            group.predicted_median_survival_minutes,
            s=6,
            alpha=0.35,
            label=model,
        )
    limit = max(
        full_predictions.true_time_to_failure_minutes.max(),
        full_predictions.predicted_median_survival_minutes.max(),
    )
    ax.plot([0, limit], [0, limit], color="black", linestyle="--")
    ax.set(xlabel="True remaining time (min)", ylabel="Predicted median survival (min)")
    ax.legend(fontsize=7)
    save(
        fig,
        "survival_curves/median_survival_vs_truth.png",
        "predictions/landmark_predictions.parquet",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    pooled = km[km.condition == "pooled"]
    for fold, group in pooled.groupby("fold"):
        ax.step(
            group.evaluation_time,
            group.survival_probability,
            where="post",
            label=f"Partição {fold}",
        )
    ax.set(
        xlabel="Minutos após o início estimado da degradação",
        ylabel="Probabilidade de sobrevivência",
        ylim=(0.0, 1.0),
    )
    ax.legend(title="Partição")
    ax.grid(alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(
        fig,
        "survival_curves/kaplan_meier_onset.png",
        "predictions/kaplan_meier_onset_curves.parquet",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    representative_bearings = ("Bearing1_1", "Bearing2_1", "Bearing3_1")
    rsf_predictions = predictions[
        (predictions.scenario == "full_event")
        & (predictions.model == "random_survival_forest")
        & predictions.bearing_id.isin(representative_bearings)
        & (predictions.time_since_causal_onset == 0)
    ]
    selected_ids = rsf_predictions.set_index("bearing_id").landmark_id.to_dict()
    curves = pd.read_parquet(output / "predictions/survival_curves.parquet")
    for bearing in representative_bearings:
        landmark_id = selected_ids[bearing]
        curve = curves[
            (curves.scenario == "full_event")
            & (curves.model == "random_survival_forest")
            & (curves.landmark_id == landmark_id)
        ].sort_values("evaluation_time")
        condition = int(curve.condition_id.iloc[0])
        line = ax.step(
            curve.evaluation_time,
            curve.survival_probability,
            where="post",
            linewidth=2,
            label=f"{bearing} (condição {condition})",
        )[0]
        observed = float(
            rsf_predictions.loc[
                rsf_predictions.landmark_id == landmark_id, "true_time_to_failure_minutes"
            ].iloc[0]
        )
        ax.axvline(observed, color=line.get_color(), linestyle=":", linewidth=1.5)
    ax.set(
        xlabel="Minutos após o marco temporal no início estimado da degradação",
        ylabel="Probabilidade de sobrevivência",
        ylim=(0.0, 1.0),
    )
    ax.legend(title="Curva; pontilhado = término experimental observado")
    ax.grid(alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(
        fig,
        "survival_curves/curvas_sobrevivencia_rsf.png",
        "predictions/survival_curves.parquet",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    cost_summary = costs.groupby("model", as_index=False).agg(
        training_seconds=("training_seconds", "median"),
        test_inference_seconds=("test_inference_seconds", "median"),
    )
    ax.scatter(cost_summary.training_seconds, cost_summary.test_inference_seconds)
    for row in cost_summary.itertuples():
        ax.annotate(row.model, (row.training_seconds, row.test_inference_seconds), fontsize=7)
    ax.set(xlabel="Median training time (s)", ylabel="Median test inference time (s)")
    save(fig, "computational/performance_vs_cost.png", "computational/computational_costs.parquet")

    fig, ax = plt.subplots(figsize=(9, 6))
    summary = (
        coefficients.assign(abs_coefficient=coefficients.standardized_coefficient.abs())
        .groupby("feature", as_index=False)
        .abs_coefficient.mean()
        .nlargest(15, "abs_coefficient")
        .sort_values("abs_coefficient")
    )
    ax.barh(summary.feature, summary.abs_coefficient)
    ax.set_xlabel("Mean absolute standardized Cox coefficient")
    save(fig, "features/cox_coefficient_summary.png", "interpretability/cox_coefficients.parquet")
    atomic_json(output / "figures/figure_manifest.json", {"figures": manifest})
    return len(manifest)
