"""Unit tests for dataset-level validation and exact duplicate detection."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xjtu_sy_tcc.config.data import ConditionConfig, DataConfig
from xjtu_sy_tcc.data.loader import CsvInspection
from xjtu_sy_tcc.data.metadata import build_metadata
from xjtu_sy_tcc.data.models import DiscoveredAcquisition, Severity
from xjtu_sy_tcc.data.validation import (
    detect_duplicate_content,
    validate_csv_inspections,
    validate_metadata,
)

CHANNELS = (
    "Horizontal_vibration_signals",
    "Vertical_vibration_signals",
)


def _condition() -> ConditionConfig:
    return ConditionConfig(
        condition_id=1,
        directory_name="35Hz12kN",
        bearing_ids=("Bearing1_1",),
        rotation_hz=35.0,
        rotation_rpm=2100.0,
        radial_load_kn=12.0,
    )


def _config(tmp_path: Path, expected_sample_count: int = 2) -> DataConfig:
    return DataConfig(
        config_path=tmp_path / "data.yaml",
        project_root=tmp_path,
        dataset_root=tmp_path / "Data",
        output_directory=tmp_path / "outputs",
        nominal_values_source="unit fixture",
        acquisition_interval_minutes=1.0,
        sampling_frequency_hz=25_600.0,
        expected_sample_count=expected_sample_count,
        expected_channel_names=CHANNELS,
        filename_pattern=r"^(?P<acquisition_number>[1-9][0-9]*)\.csv$",
        detect_duplicate_content=True,
        max_workers=2,
        conditions=(_condition(),),
    )


def _acquisition(
    path: Path,
    acquisition_number: int = 1,
    sequence_index: int = 0,
    *,
    discovered_size: int | None = None,
) -> DiscoveredAcquisition:
    condition = _condition()
    return DiscoveredAcquisition(
        condition=condition,
        bearing_id="Bearing1_1",
        file_path=path,
        relative_path=Path("Data/35Hz12kN/Bearing1_1") / path.name,
        acquisition_number=acquisition_number,
        sequence_index=sequence_index,
        file_size_bytes=path.stat().st_size if discovered_size is None else discovered_size,
    )


def _inspection(
    path: Path,
    *,
    channel_names: tuple[str, ...] = CHANNELS,
    channel_count: int = 2,
    sample_count: int = 2,
    nan_count: int = 0,
    infinite_count: int = 0,
    parse_error: str | None = None,
) -> CsvInspection:
    return CsvInspection(
        file_path=path,
        channel_names=channel_names,
        sample_count=sample_count,
        channel_count=channel_count,
        nan_count=nan_count,
        infinite_count=infinite_count,
        parse_error=parse_error,
    )


def _issue_codes(issues: object) -> set[str]:
    return {issue.code for issue in issues}  # type: ignore[attr-defined]


def _valid_metadata(tmp_path: Path):
    paths = (tmp_path / "1.csv", tmp_path / "2.csv")
    for path in paths:
        path.write_bytes(b"0,0\n")
    acquisitions = tuple(
        _acquisition(path, acquisition_number=index + 1, sequence_index=index)
        for index, path in enumerate(paths)
    )
    inspections = {
        acquisition.file_path: _inspection(acquisition.file_path) for acquisition in acquisitions
    }
    return build_metadata(acquisitions, inspections, _config(tmp_path))


def test_validate_csv_inspections_accepts_matching_finite_data(tmp_path: Path) -> None:
    path = tmp_path / "1.csv"
    path.write_bytes(b"0,0\n0,0\n")
    acquisition = _acquisition(path)

    issues = validate_csv_inspections(
        (acquisition,),
        {path: _inspection(path)},
        _config(tmp_path),
    )

    assert issues == ()


def test_validate_csv_inspections_reports_all_observed_schema_errors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "1.csv"
    path.write_bytes(b"fixture")
    acquisition = _acquisition(path)
    inspection = _inspection(
        path,
        channel_names=("Vertical_vibration_signals",),
        channel_count=1,
        sample_count=1,
        nan_count=1,
        infinite_count=2,
        parse_error="Numeric data could not be parsed.",
    )

    issues = validate_csv_inspections(
        (acquisition,),
        {path: inspection},
        _config(tmp_path),
    )

    assert _issue_codes(issues) == {
        "csv_inspection_error",
        "infinite_values",
        "invalid_channel_count",
        "invalid_channel_names",
        "invalid_sample_count",
        "nan_values",
    }
    assert all(issue.severity is Severity.ERROR for issue in issues)


def test_validate_csv_inspections_reports_missing_inspection(tmp_path: Path) -> None:
    path = tmp_path / "1.csv"
    path.write_bytes(b"fixture")

    issues = validate_csv_inspections(
        (_acquisition(path),),
        {},
        _config(tmp_path),
    )

    assert _issue_codes(issues) == {"missing_csv_inspection"}


def test_validate_csv_inspections_detects_file_size_change(tmp_path: Path) -> None:
    path = tmp_path / "1.csv"
    path.write_bytes(b"fixture")
    acquisition = _acquisition(path, discovered_size=path.stat().st_size + 1)

    issues = validate_csv_inspections(
        (acquisition,),
        {path: _inspection(path)},
        _config(tmp_path),
    )

    assert _issue_codes(issues) == {"acquisition_size_changed"}


def test_validate_csv_inspections_detects_same_size_modification(tmp_path: Path) -> None:
    path = tmp_path / "1.csv"
    path.write_bytes(b"original")
    discovered_stat = path.stat()
    acquisition = replace(
        _acquisition(path),
        modified_time_ns=discovered_stat.st_mtime_ns,
        device_id=discovered_stat.st_dev,
        inode=discovered_stat.st_ino,
    )
    path.write_bytes(b"modified")
    os.utime(
        path,
        ns=(discovered_stat.st_atime_ns, discovered_stat.st_mtime_ns + 1_000_000),
    )

    issues = validate_csv_inspections(
        (acquisition,),
        {path: _inspection(path)},
        _config(tmp_path),
    )

    assert "acquisition_modified_during_audit" in _issue_codes(issues)


def test_validate_metadata_accepts_generated_metadata(tmp_path: Path) -> None:
    assert validate_metadata(_valid_metadata(tmp_path), _config(tmp_path)) == ()


@pytest.mark.parametrize(
    ("column", "row_index", "replacement", "expected_code"),
    [
        ("sequence_index", 1, 0, "invalid_metadata_sequence"),
        ("acquisition_number", 0, 3, "non_monotonic_acquisition_numbers"),
        ("acquisition_number", 1, 1, "non_monotonic_acquisition_numbers"),
        ("elapsed_minutes", 1, 99.0, "invalid_elapsed_minutes"),
        ("rul_minutes", 0, 99.0, "invalid_rul_calculation"),
        ("rul_minutes", 0, -1.0, "invalid_rul_values"),
        ("rul_minutes", 1, 1.0, "nonzero_final_rul"),
        ("total_acquisitions", 0, 3, "invalid_total_acquisitions"),
    ],
)
def test_validate_metadata_reports_bearing_invariant_violations(
    tmp_path: Path,
    column: str,
    row_index: int,
    replacement: int | float,
    expected_code: str,
) -> None:
    metadata = _valid_metadata(tmp_path)
    metadata.loc[row_index, column] = replacement

    issues = validate_metadata(metadata, _config(tmp_path))

    assert expected_code in _issue_codes(issues)


def test_validate_metadata_reports_duplicate_paths(tmp_path: Path) -> None:
    metadata = _valid_metadata(tmp_path)
    metadata.loc[1, "file_path"] = metadata.loc[0, "file_path"]

    issues = validate_metadata(metadata, _config(tmp_path))

    assert "duplicate_metadata_file_paths" in _issue_codes(issues)


@pytest.mark.parametrize("invalid_rul", [np.nan, np.inf, -np.inf])
def test_validate_metadata_reports_nonfinite_rul(
    tmp_path: Path,
    invalid_rul: float,
) -> None:
    metadata = _valid_metadata(tmp_path)
    metadata.loc[0, "rul_minutes"] = invalid_rul

    issues = validate_metadata(metadata, _config(tmp_path))

    assert "invalid_rul_values" in _issue_codes(issues)


def test_validate_metadata_rejects_empty_metadata(tmp_path: Path) -> None:
    metadata = _valid_metadata(tmp_path).iloc[0:0]

    issues = validate_metadata(metadata, _config(tmp_path))

    assert _issue_codes(issues) == {"empty_metadata"}


def test_detect_duplicate_content_confirms_exact_bytes_only(tmp_path: Path) -> None:
    paths = (tmp_path / "1.csv", tmp_path / "2.csv", tmp_path / "3.csv")
    paths[0].write_bytes(b"header\n1,2\n")
    paths[1].write_bytes(b"header\n1,2\n")
    paths[2].write_bytes(b"header\n2,1\n")
    acquisitions = tuple(
        _acquisition(path, acquisition_number=index + 1, sequence_index=index)
        for index, path in enumerate(paths)
    )

    result = detect_duplicate_content(acquisitions, max_workers=2)

    assert result.equal_size_group_count == 1
    assert result.boundary_fingerprint_file_count == 3
    assert result.full_hash_file_count == 2
    assert result.duplicate_groups == ((paths[0], paths[1]),)
    assert _issue_codes(result.issues) == {"duplicate_file_content"}


def test_detect_duplicate_content_rejects_same_size_as_evidence(
    tmp_path: Path,
) -> None:
    first = tmp_path / "1.csv"
    second = tmp_path / "2.csv"
    first.write_bytes(b"equal-size-A")
    second.write_bytes(b"equal-size-B")
    acquisitions = (
        _acquisition(first, acquisition_number=1, sequence_index=0),
        _acquisition(second, acquisition_number=2, sequence_index=1),
    )

    result = detect_duplicate_content(acquisitions, max_workers=2)

    assert result.equal_size_group_count == 1
    assert result.boundary_fingerprint_file_count == 2
    assert result.full_hash_file_count == 0
    assert result.duplicate_groups == ()
    assert result.issues == ()
