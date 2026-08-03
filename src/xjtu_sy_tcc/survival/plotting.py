"""Scientific figures for causal detection and survival cohorts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xjtu_phase7_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_figures(config, trajectories, onsets, comparison, full, fixed, rate_manifest):
    count = 0
    root = config.detector_output / "figures"
    for bearing, group in trajectories[trajectories.subset == "test"].groupby("bearing_id"):
        g = group.sort_values("sequence_index")
        onset = onsets[onsets.bearing_id == bearing].iloc[0]
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot(g.elapsed_minutes, g.raw_combined_divergence, label="Raw divergence", alpha=0.5)
        ax.plot(
            g.elapsed_minutes, g.causal_aggregated_divergence, label="Causal aggregated divergence"
        )
        ax.axvspan(
            0, onset.baseline_acquisitions - 1, color="0.8", label="Fixed calibration interval"
        )
        if onset.detection_status == "detected":
            ax.axvline(
                onset.alarm_elapsed_minutes,
                color="tab:red",
                ls="--",
                label="Causal estimated degradation onset",
            )
        p = comparison[comparison.bearing_id == bearing].iloc[0]
        if p.estimated_onset_status == "detected":
            ax.axvline(
                p.estimated_onset_sequence_index,
                color="tab:purple",
                ls=":",
                label="Retrospective PELT estimated onset",
            )
        ax.set(title=bearing, xlabel="Elapsed time (min)", ylabel="Spectral divergence")
        ax.legend(fontsize=7)
        count += _save(fig, root / "by_bearing" / bearing, config)
    summaries = [
        ("rul_at_alarm", onsets, "bearing_id", "rul_at_alarm"),
        ("consumed_life_at_alarm", onsets, "bearing_id", "consumed_life_fraction"),
        ("landmarks_by_bearing", full, "bearing_id", "duration_minutes"),
        ("fixed_censoring", fixed, "scenario_id", "event_observed"),
        ("target_censoring", rate_manifest, "target_censoring_rate", "achieved_censoring_rate"),
    ]
    for name, table, x, y in summaries:
        data = table.groupby(x, as_index=False)[y].mean()
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.bar(data[x].astype(str), data[y], color="0.35")
        ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
        ax.tick_params(axis="x", rotation=35)
        count += _save(fig, root / "summary" / name, config)
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
