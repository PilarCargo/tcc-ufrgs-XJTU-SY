"""Unit tests for metadata construction and Remaining Useful Life calculation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_string_dtype

from xjtu_sy_tcc.config.data import ConditionConfig, DataConfig
from xjtu_sy_tcc.data.loader import CsvInspection
from xjtu_sy_tcc.data.metadata import (
    METADATA_COLUMNS,
    build_bearing_summary,
    build_metadata,
    calculate_rul_minutes,
)
from xjtu_sy_tcc.data.models import DiscoveredAcquisition

CHANNELS = (
    "Horizontal_vibration_signals",
    "Vertical_vibration_signals",
)


def _condition(
    condition_id: int,
    directory_name: str,
    bearing_id: str,
    rotation_hz: float,
    load_kn: float,
) -> ConditionConfig:
    return ConditionConfig(
        condition_id=condition_id,
        directory_name=directory_name,
        bearing_ids=(bearing_id,),
        rotation_hz=rotation_hz,
        rotation_rpm=rotation_hz * 60.0,
        radial_load_kn=load_kn,
    )


def _config(
    tmp_path: Path,
    conditions: tuple[ConditionConfig, ...],
) -> DataConfig:
    return DataConfig(
        config_path=tmp_path / "data.yaml",
        project_root=tmp_path,
        dataset_root=tmp_path / "Data",
        output_directory=tmp_path / "outputs",
        nominal_values_source="unit fixture",
        acquisition_interval_minutes=0.5,
        sampling_frequency_hz=25_600.0,
        expected_sample_count=2,
        expected_channel_names=CHANNELS,
        filename_pattern=r"^(?P<acquisition_number>[1-9][0-9]*)\.csv$",
        detect_duplicate_content=True,
        max_workers=1,
        conditions=conditions,
    )


def _acquisition(
    tmp_path: Path,
    condition: ConditionConfig,
    bearing_id: str,
    acquisition_number: int,
    sequence_index: int,
) -> DiscoveredAcquisition:
    file_path = tmp_path / f"{condition.condition_id}-{bearing_id}-{acquisition_number}.csv"
    return DiscoveredAcquisition(
        condition=condition,
        bearing_id=bearing_id,
        file_path=file_path,
        relative_path=(
            Path("Data") / condition.directory_name / bearing_id / f"{acquisition_number}.csv"
        ),
        acquisition_number=acquisition_number,
        sequence_index=sequence_index,
        file_size_bytes=100 + acquisition_number,
    )


def _inspection(acquisition: DiscoveredAcquisition) -> CsvInspection:
    return CsvInspection(
        file_path=acquisition.file_path,
        channel_names=CHANNELS,
        sample_count=2,
        channel_count=2,
        nan_count=0,
        infinite_count=0,
        parse_error=None,
    )


def test_calculate_rul_minutes_uses_zero_based_formula_and_finishes_at_zero() -> None:
    assert calculate_rul_minutes(5, 0, 1.5) == 6.0
    assert calculate_rul_minutes(5, 2, 1.5) == 3.0
    assert calculate_rul_minutes(5, 4, 1.5) == 0.0


@pytest.mark.parametrize(
    ("total_acquisitions", "sequence_index", "interval"),
    [
        (0, 0, 1.0),
        (-1, 0, 1.0),
        (2, -1, 1.0),
        (2, 2, 1.0),
        (2, 0, 0.0),
        (2, 0, -1.0),
        (2, 0, float("nan")),
        (2, 0, float("inf")),
    ],
)
def test_calculate_rul_minutes_rejects_invalid_arguments(
    total_acquisitions: int,
    sequence_index: int,
    interval: float,
) -> None:
    with pytest.raises(ValueError):
        calculate_rul_minutes(total_acquisitions, sequence_index, interval)


def test_build_metadata_sorts_rows_assigns_counts_and_stable_types(
    tmp_path: Path,
) -> None:
    first_condition = _condition(1, "35Hz12kN", "Bearing1_1", 35.0, 12.0)
    second_condition = _condition(2, "37.5Hz11kN", "Bearing2_1", 37.5, 11.0)
    config = _config(tmp_path, (first_condition, second_condition))
    acquisitions = (
        _acquisition(tmp_path, second_condition, "Bearing2_1", 1, 0),
        _acquisition(tmp_path, first_condition, "Bearing1_1", 2, 1),
        _acquisition(tmp_path, first_condition, "Bearing1_1", 1, 0),
    )
    inspections = {acquisition.file_path: _inspection(acquisition) for acquisition in acquisitions}

    metadata = build_metadata(acquisitions, inspections, config)

    assert tuple(metadata.columns) == METADATA_COLUMNS
    assert metadata[["condition_id", "bearing_id", "sequence_index"]].to_records(
        index=False
    ).tolist() == [(1, "Bearing1_1", 0), (1, "Bearing1_1", 1), (2, "Bearing2_1", 0)]
    assert metadata["file_path"].tolist() == [
        "Data/35Hz12kN/Bearing1_1/1.csv",
        "Data/35Hz12kN/Bearing1_1/2.csv",
        "Data/37.5Hz11kN/Bearing2_1/1.csv",
    ]
    assert metadata["total_acquisitions"].tolist() == [2, 2, 1]
    assert metadata["elapsed_minutes"].tolist() == [0.0, 0.5, 0.0]
    assert metadata["rul_minutes"].tolist() == [0.5, 0.0, 0.0]
    assert metadata["sample_count"].tolist() == [2, 2, 2]
    assert metadata["channel_count"].tolist() == [2, 2, 2]

    expected_dtypes = {
        "condition_id": "int64",
        "acquisition_number": "int64",
        "sequence_index": "int64",
        "elapsed_minutes": "float64",
        "total_acquisitions": "int64",
        "rul_minutes": "float64",
        "rotation_rpm": "float64",
        "radial_load_kn": "float64",
        "sampling_frequency_hz": "float64",
        "sample_count": "int64",
        "channel_count": "int32",
        "file_size_bytes": "int64",
    }
    observed_dtypes = metadata.dtypes.astype(str).to_dict()
    assert {column: observed_dtypes[column] for column in expected_dtypes} == expected_dtypes
    for column in ("bearing_id", "file_path", "file_name"):
        assert is_string_dtype(metadata[column].dtype)


def test_build_bearing_summary_reports_duration_and_endpoint_rul(tmp_path: Path) -> None:
    condition = _condition(1, "35Hz12kN", "Bearing1_1", 35.0, 12.0)
    config = _config(tmp_path, (condition,))
    acquisitions = tuple(
        _acquisition(tmp_path, condition, "Bearing1_1", index + 1, index) for index in range(3)
    )
    inspections = {acquisition.file_path: _inspection(acquisition) for acquisition in acquisitions}
    metadata = build_metadata(acquisitions, inspections, config)

    summary = build_bearing_summary(metadata)

    assert summary.to_dict(orient="records") == [
        {
            "condition_id": 1,
            "bearing_id": "Bearing1_1",
            "rotation_rpm": 2100.0,
            "radial_load_kn": 12.0,
            "total_acquisitions": 3,
            "duration_minutes": 1.0,
            "initial_rul_minutes": 1.0,
            "final_rul_minutes": 0.0,
        }
    ]


def test_empty_metadata_and_summary_preserve_their_schemas(tmp_path: Path) -> None:
    condition = _condition(1, "35Hz12kN", "Bearing1_1", 35.0, 12.0)

    metadata = build_metadata((), {}, _config(tmp_path, (condition,)))
    summary = build_bearing_summary(metadata)

    assert metadata.empty
    assert tuple(metadata.columns) == METADATA_COLUMNS
    assert summary.empty
    assert tuple(summary.columns) == (
        "condition_id",
        "bearing_id",
        "rotation_rpm",
        "radial_load_kn",
        "total_acquisitions",
        "duration_minutes",
        "initial_rul_minutes",
        "final_rul_minutes",
    )


def test_build_metadata_uses_zero_counts_when_inspection_is_missing(tmp_path: Path) -> None:
    condition = _condition(1, "35Hz12kN", "Bearing1_1", 35.0, 12.0)
    acquisition = _acquisition(tmp_path, condition, "Bearing1_1", 1, 0)

    metadata = build_metadata((acquisition,), {}, _config(tmp_path, (condition,)))

    assert metadata.loc[0, "sample_count"] == 0
    assert metadata.loc[0, "channel_count"] == 0
    assert metadata.loc[0, "rul_minutes"] == 0.0
    assert isinstance(metadata, pd.DataFrame)


def test_build_metadata_does_not_narrow_large_condition_identifiers(tmp_path: Path) -> None:
    condition = _condition(40_000, "35Hz12kN", "Bearing40000_1", 35.0, 12.0)
    acquisition = _acquisition(tmp_path, condition, "Bearing40000_1", 1, 0)

    metadata = build_metadata(
        (acquisition,),
        {acquisition.file_path: _inspection(acquisition)},
        _config(tmp_path, (condition,)),
    )

    assert metadata.loc[0, "condition_id"] == 40_000
