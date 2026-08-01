"""Metadata and Remaining Useful Life calculations."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from xjtu_sy_tcc.config.data import DataConfig
from xjtu_sy_tcc.data.discovery import DiscoveredAcquisition
from xjtu_sy_tcc.data.loader import CsvInspection

METADATA_COLUMNS = (
    "condition_id",
    "bearing_id",
    "file_path",
    "file_name",
    "acquisition_number",
    "sequence_index",
    "elapsed_minutes",
    "total_acquisitions",
    "rul_minutes",
    "rotation_rpm",
    "radial_load_kn",
    "sampling_frequency_hz",
    "sample_count",
    "channel_count",
    "file_size_bytes",
)


def calculate_rul_minutes(
    total_acquisitions: int,
    sequence_index: int,
    acquisition_interval_minutes: float,
) -> float:
    """Calculate absolute RUL for a zero-based acquisition index."""

    if total_acquisitions <= 0:
        raise ValueError("total_acquisitions must be greater than zero")
    if not 0 <= sequence_index < total_acquisitions:
        raise ValueError(
            "sequence_index must be in the closed interval [0, total_acquisitions - 1]"
        )
    if not math.isfinite(acquisition_interval_minutes) or acquisition_interval_minutes <= 0:
        raise ValueError("acquisition_interval_minutes must be finite and greater than zero")
    return (total_acquisitions - 1 - sequence_index) * acquisition_interval_minutes


def build_metadata(
    acquisitions: Sequence[DiscoveredAcquisition],
    inspections: Mapping[Path, CsvInspection],
    config: DataConfig,
) -> pd.DataFrame:
    """Build one deterministic metadata row per discovered acquisition."""

    totals = Counter((item.condition.condition_id, item.bearing_id) for item in acquisitions)
    rows: list[dict[str, object]] = []
    for item in acquisitions:
        key = (item.condition.condition_id, item.bearing_id)
        total_acquisitions = totals[key]
        inspection = inspections.get(item.file_path)
        sample_count = inspection.sample_count if inspection is not None else 0
        channel_count = inspection.channel_count if inspection is not None else 0
        rows.append(
            {
                "condition_id": item.condition.condition_id,
                "bearing_id": item.bearing_id,
                "file_path": item.relative_path.as_posix(),
                "file_name": item.file_path.name,
                "acquisition_number": item.acquisition_number,
                "sequence_index": item.sequence_index,
                "elapsed_minutes": (item.sequence_index * config.acquisition_interval_minutes),
                "total_acquisitions": total_acquisitions,
                "rul_minutes": calculate_rul_minutes(
                    total_acquisitions,
                    item.sequence_index,
                    config.acquisition_interval_minutes,
                ),
                "rotation_rpm": item.condition.rotation_rpm,
                "radial_load_kn": item.condition.radial_load_kn,
                "sampling_frequency_hz": config.sampling_frequency_hz,
                "sample_count": sample_count,
                "channel_count": channel_count,
                "file_size_bytes": item.file_size_bytes,
            }
        )

    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    if metadata.empty:
        return metadata

    integer_types = {
        "condition_id": "int64",
        "acquisition_number": "int64",
        "sequence_index": "int64",
        "total_acquisitions": "int64",
        "sample_count": "int64",
        "channel_count": "int32",
        "file_size_bytes": "int64",
    }
    metadata = metadata.astype(integer_types)
    for column in (
        "elapsed_minutes",
        "rul_minutes",
        "rotation_rpm",
        "radial_load_kn",
        "sampling_frequency_hz",
    ):
        metadata[column] = metadata[column].astype("float64")
    return metadata.sort_values(["condition_id", "bearing_id", "sequence_index"], ignore_index=True)


def build_bearing_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    """Summarize acquisition counts and observed duration by bearing."""

    columns = (
        "condition_id",
        "bearing_id",
        "rotation_rpm",
        "radial_load_kn",
        "total_acquisitions",
        "duration_minutes",
        "initial_rul_minutes",
        "final_rul_minutes",
    )
    if metadata.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        metadata.groupby(["condition_id", "bearing_id"], as_index=False, sort=True)
        .agg(
            rotation_rpm=("rotation_rpm", "first"),
            radial_load_kn=("radial_load_kn", "first"),
            total_acquisitions=("sequence_index", "size"),
            duration_minutes=("elapsed_minutes", "max"),
            initial_rul_minutes=("rul_minutes", "max"),
            final_rul_minutes=("rul_minutes", "min"),
        )
        .loc[:, columns]
    )
    return summary
