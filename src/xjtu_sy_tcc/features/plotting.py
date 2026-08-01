"""Deterministic exploratory degradation and representative-signal figures."""

# ruff: noqa: E402 -- configure a writable Matplotlib cache before importing it.

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

_MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "xjtu_sy_tcc_matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xjtu_sy_tcc.config.features import FeatureConfig
from xjtu_sy_tcc.features.frequency_domain import compute_welch_psd
from xjtu_sy_tcc.logging_utils import log_event

LOGGER = logging.getLogger(__name__)
_DEGRADATION = (
    ("rms", "RMS", "Vibration amplitude (dataset unit)"),
    ("kurtosis", "Kurtosis", "Kurtosis (-)"),
    ("crest_factor", "Crest factor", "Crest factor (-)"),
    ("energy", "Signal energy", "Energy (dataset unit²)"),
)


def generate_exploratory_plots(config: FeatureConfig, table: pd.DataFrame) -> int:
    """Create unsmoothed bearing trajectories and first/middle/final signal/PSD plots."""
    count = 0
    for (condition, bearing), group in table.groupby(["condition_id", "bearing_id"], sort=True):
        ordered = group.sort_values("sequence_index")
        directory = config.figure_directory / f"condition_{int(condition)}" / str(bearing)
        directory.mkdir(parents=True, exist_ok=True)
        for suffix, title, ylabel in _DEGRADATION:
            figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            for channel, label in (("horizontal", "Horizontal"), ("vertical", "Vertical")):
                axis.plot(
                    ordered["elapsed_minutes"],
                    ordered[f"{channel}_{suffix}"],
                    linewidth=0.9,
                    label=label,
                )
            axis.set(title=f"{bearing}: {title}", xlabel="Elapsed time (min)", ylabel=ylabel)
            axis.grid(alpha=0.25)
            axis.legend()
            count += _save(figure, directory / f"{suffix}_vs_elapsed", config)
        figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        axis.plot(ordered["elapsed_minutes"], ordered["rul_minutes"], linewidth=1.0)
        axis.set(
            title=f"{bearing}: Remaining Useful Life",
            xlabel="Elapsed time (min)",
            ylabel="RUL (min)",
        )
        axis.grid(alpha=0.25)
        count += _save(figure, directory / "rul_vs_elapsed", config)
        positions = {"first": 0, "middle": len(ordered) // 2, "final": len(ordered) - 1}
        for label, position in positions.items():
            row = ordered.iloc[position]
            values = np.loadtxt(
                config.project_root / str(row["file_path"]),
                delimiter=",",
                skiprows=1,
                dtype=np.float32,
                ndmin=2,
            )
            fs = float(row["sampling_frequency_hz"])
            time_seconds = np.arange(values.shape[0], dtype=np.float64) / fs
            figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
            for index, channel in enumerate(("Horizontal", "Vertical")):
                axes[0].plot(time_seconds, values[:, index], linewidth=0.45, label=channel)
                frequencies, density = compute_welch_psd(
                    values[:, index], fs, config.welch_nperseg, config.welch_overlap
                )
                axes[1].semilogy(
                    frequencies,
                    np.maximum(density, np.finfo(float).tiny),
                    linewidth=0.8,
                    label=channel,
                )
            axes[0].set(
                title=f"{bearing}: {label} acquisition (#{int(row['acquisition_number'])})",
                xlabel="Time (s)",
                ylabel="Amplitude (dataset unit)",
            )
            axes[1].set(xlabel="Frequency (Hz)", ylabel="PSD (dataset unit²/Hz)")
            for axis in axes:
                axis.grid(alpha=0.25)
                axis.legend()
            count += _save(figure, directory / f"representative_{label}_signal_psd", config)
        log_event(
            LOGGER,
            logging.INFO,
            "bearing_plots_complete",
            "Exploratory plots generated.",
            condition_id=int(condition),
            bearing_id=str(bearing),
        )
    return count


def _save(figure: plt.Figure, path: Path, config: FeatureConfig) -> int:
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
