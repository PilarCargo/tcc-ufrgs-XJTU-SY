"""Incremental orchestration for Phase-2 feature extraction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from xjtu_sy_tcc.config.features import FeatureConfig
from xjtu_sy_tcc.features.frequency_domain import FREQUENCY_FEATURE_NAMES, frequency_features
from xjtu_sy_tcc.features.time_domain import TIME_FEATURE_NAMES, time_features
from xjtu_sy_tcc.features.validation import FeatureValidation, validate_feature_table
from xjtu_sy_tcc.logging_utils import log_event

LOGGER = logging.getLogger(__name__)
CHANNELS = ("horizontal", "vertical")
PASSTHROUGH_COLUMNS = (
    "condition_id",
    "bearing_id",
    "file_path",
    "file_name",
    "acquisition_number",
    "sequence_index",
    "elapsed_minutes",
    "rul_minutes",
    "rotation_rpm",
    "radial_load_kn",
    "sampling_frequency_hz",
    "sample_count",
    "channel_count",
)


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    table: pd.DataFrame
    validation: FeatureValidation
    duration_seconds: float
    peak_rss_bytes: int
    resumed_rows: int


def feature_column_names(config: FeatureConfig) -> tuple[str, ...]:
    frequency = FREQUENCY_FEATURE_NAMES + tuple(
        f"band_power_{band.name}" for band in config.frequency_bands_hz
    )
    return tuple(
        f"{channel}_{name}" for channel in CHANNELS for name in TIME_FEATURE_NAMES + frequency
    )


def build_features(config: FeatureConfig, *, resume: bool = True) -> FeatureBuildResult:
    """Read each manifested CSV independently and retain only scalar features."""
    started = time.perf_counter()
    metadata = pd.read_parquet(config.metadata_path)
    _validate_manifest(metadata, config)
    checkpoint = config.output_directory / ".features.checkpoint.parquet"
    rows: list[dict[str, object]] = []
    resumed = 0
    if resume and checkpoint.is_file():
        cached = pd.read_parquet(checkpoint)
        if list(cached.columns) == list(PASSTHROUGH_COLUMNS + feature_column_names(config)):
            rows = cached.to_dict("records")
            resumed = len(rows)
            if resumed > len(metadata) or not _checkpoint_matches(metadata, cached):
                rows = []
                resumed = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    for position in range(resumed, len(metadata)):
        manifest_row = metadata.iloc[position]
        path = (config.project_root / str(manifest_row["file_path"])).resolve()
        if config.project_root not in path.parents:
            raise ValueError(f"Manifest path escapes project root: {path}")
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
        if values.shape != (int(manifest_row["sample_count"]), 2):
            raise ValueError(f"Unexpected acquisition shape at {path}: {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite signal values at {path}")
        output = {name: manifest_row[name] for name in PASSTHROUGH_COLUMNS}
        for channel_index, channel in enumerate(CHANNELS):
            signal = values[:, channel_index]
            calculated = time_features(signal)
            calculated.update(
                frequency_features(
                    signal,
                    float(manifest_row["sampling_frequency_hz"]),
                    config.welch_nperseg,
                    config.welch_overlap,
                    config.frequency_bands_hz,
                )
            )
            output.update({f"{channel}_{key}": value for key, value in calculated.items()})
        rows.append(output)
        peak_rss = max(peak_rss, process.memory_info().rss)
        processed = position + 1
        if processed % config.progress_interval == 0 or processed == len(metadata):
            log_event(
                LOGGER,
                logging.INFO,
                "feature_progress",
                "Feature extraction progress.",
                processed=processed,
                total=len(metadata),
            )
        if processed % max(500, config.progress_interval) == 0:
            _atomic_parquet(pd.DataFrame(rows), checkpoint)
    table = pd.DataFrame(rows, columns=PASSTHROUGH_COLUMNS + feature_column_names(config))
    validation = validate_feature_table(metadata, table, feature_column_names(config))
    return FeatureBuildResult(table, validation, time.perf_counter() - started, peak_rss, resumed)


def _validate_manifest(metadata: pd.DataFrame, config: FeatureConfig) -> None:
    missing = sorted(set(PASSTHROUGH_COLUMNS) - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata manifest is missing columns: {', '.join(missing)}")
    if metadata.empty:
        raise ValueError("Metadata manifest is empty")
    if (
        max(band.upper_hz for band in config.frequency_bands_hz)
        > float(metadata["sampling_frequency_hz"].min()) / 2
    ):
        raise ValueError("A configured frequency band exceeds the manifest Nyquist frequency")


def _checkpoint_matches(metadata: pd.DataFrame, cached: pd.DataFrame) -> bool:
    if cached.empty:
        return True
    columns = ["condition_id", "bearing_id", "acquisition_number", "file_path"]
    return (
        metadata.iloc[: len(cached)][columns]
        .reset_index(drop=True)
        .equals(cached[columns].reset_index(drop=True))
    )


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        table.to_parquet(temporary, index=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
