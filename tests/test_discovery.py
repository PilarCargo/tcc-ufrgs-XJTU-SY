"""Tests for deterministic, read-only dataset discovery."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from xjtu_sy_tcc.config import ConditionConfig, DataConfig
from xjtu_sy_tcc.data import DiscoveryResult, Severity, discover_dataset, numeric_file_index

_DEFAULT_FILENAME_PATTERN = r"^(?P<acquisition_number>[1-9][0-9]*)\.csv$"
_CSV_CONTENT = "Horizontal_vibration_signals,Vertical_vibration_signals\n0.0,0.0\n"


def _condition(
    condition_id: int = 1,
    directory_name: str = "35Hz12kN",
    bearing_ids: tuple[str, ...] = ("Bearing1_1",),
    expected_acquisition_counts: dict[str, int] | None = None,
) -> ConditionConfig:
    rotation_hz = 35.0 if condition_id == 1 else 37.5
    radial_load_kn = 12.0 if condition_id == 1 else 11.0
    return ConditionConfig(
        condition_id=condition_id,
        directory_name=directory_name,
        bearing_ids=bearing_ids,
        rotation_hz=rotation_hz,
        rotation_rpm=rotation_hz * 60.0,
        radial_load_kn=radial_load_kn,
        expected_acquisition_counts=expected_acquisition_counts or {},
    )


def _config(
    project_root: Path,
    *,
    conditions: tuple[ConditionConfig, ...] | None = None,
    filename_pattern: str = _DEFAULT_FILENAME_PATTERN,
) -> DataConfig:
    nominal_source = project_root / "Data" / "Introduction_to_XJTU-SY_Bearing_Dataset.pdf"
    nominal_source.parent.mkdir(parents=True, exist_ok=True)
    nominal_source.touch(exist_ok=True)
    return DataConfig(
        config_path=project_root / "configs" / "data.yaml",
        project_root=project_root.resolve(),
        dataset_root=(project_root / "Data").resolve(),
        output_directory=(project_root / "outputs").resolve(),
        nominal_values_source="Data/Introduction_to_XJTU-SY_Bearing_Dataset.pdf",
        acquisition_interval_minutes=1.0,
        sampling_frequency_hz=25_600.0,
        expected_sample_count=32_768,
        expected_channel_names=(
            "Horizontal_vibration_signals",
            "Vertical_vibration_signals",
        ),
        filename_pattern=filename_pattern,
        detect_duplicate_content=True,
        max_workers=1,
        conditions=conditions or (_condition(),),
    )


def _write_acquisition(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_CSV_CONTENT, encoding="utf-8")


def _issue_codes(result: DiscoveryResult) -> list[str]:
    return [issue.code for issue in result.issues]


def _symlink_or_skip(target: Path, link: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"Symbolic links are not supported in this test environment: {error}")


def test_discovery_orders_numeric_filenames_as_one_two_ten(tmp_path: Path) -> None:
    bearing_path = tmp_path / "Data" / "35Hz12kN" / "Bearing1_1"
    for filename in ("10.csv", "2.csv", "1.csv"):
        _write_acquisition(bearing_path / filename)

    result = discover_dataset(_config(tmp_path))

    assert [item.acquisition_number for item in result.acquisitions] == [1, 2, 10]
    assert [item.sequence_index for item in result.acquisitions] == [0, 1, 2]
    assert "missing_acquisition_numbers" in _issue_codes(result)


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param("0.csv", id="zero"),
        pytest.param("-1.csv", id="negative"),
        pytest.param("word.csv", id="nonnumeric"),
        pytest.param("1.txt", id="wrong-extension"),
        pytest.param("9223372036854775808.csv", id="larger-than-int64"),
    ],
)
def test_numeric_file_index_rejects_malformed_or_nonpositive_names(filename: str) -> None:
    pattern = re.compile(r"^(?P<acquisition_number>-?[0-9]+)\.csv$")

    assert numeric_file_index(Path(filename), pattern) is None


def test_discovery_builds_deterministic_zero_based_sequences_and_root_warning(
    tmp_path: Path,
) -> None:
    condition = _condition(bearing_ids=("Bearing1_1", "Bearing1_2"))
    data_root = tmp_path / "Data"
    (data_root / ".DS_Store").parent.mkdir(parents=True)
    (data_root / ".DS_Store").write_text("metadata", encoding="utf-8")
    (data_root / "Introduction_to_XJTU-SY_Bearing_Dataset.pdf").write_bytes(b"fixture")
    for filename in ("3.csv", "1.csv", "2.csv"):
        _write_acquisition(data_root / condition.directory_name / "Bearing1_1" / filename)
    for filename in ("2.csv", "1.csv"):
        _write_acquisition(data_root / condition.directory_name / "Bearing1_2" / filename)

    result = discover_dataset(_config(tmp_path, conditions=(condition,)))

    assert [
        (item.bearing_id, item.acquisition_number, item.sequence_index)
        for item in result.acquisitions
    ] == [
        ("Bearing1_1", 1, 0),
        ("Bearing1_1", 2, 1),
        ("Bearing1_1", 3, 2),
        ("Bearing1_2", 1, 0),
        ("Bearing1_2", 2, 1),
    ]
    assert _issue_codes(result) == ["unexpected_root_file"]
    assert result.issues[0].severity is Severity.WARNING
    assert result.unexpected_entries == ("Data/.DS_Store",)
    assert not result.has_errors


def test_discovery_rejects_a_contiguous_but_truncated_sequence(tmp_path: Path) -> None:
    condition = _condition(expected_acquisition_counts={"Bearing1_1": 3})
    bearing_path = tmp_path / "Data" / "35Hz12kN" / "Bearing1_1"
    _write_acquisition(bearing_path / "1.csv")
    _write_acquisition(bearing_path / "2.csv")

    result = discover_dataset(_config(tmp_path, conditions=(condition,)))

    count_issue = next(
        issue for issue in result.issues if issue.code == "unexpected_acquisition_count"
    )
    assert count_issue.details == {
        "expected_count": 3,
        "observed_count": 2,
        "observed_final_number": 2,
    }
    assert result.has_errors


def test_discovery_reports_missing_condition_and_bearing_directories(tmp_path: Path) -> None:
    first_condition = _condition(bearing_ids=("Bearing1_1", "Bearing1_2"))
    second_condition = _condition(
        condition_id=2,
        directory_name="37.5Hz11kN",
        bearing_ids=("Bearing2_1",),
    )
    _write_acquisition(tmp_path / "Data" / "35Hz12kN" / "Bearing1_1" / "1.csv")

    result = discover_dataset(_config(tmp_path, conditions=(first_condition, second_condition)))

    assert "missing_bearing_directory" in _issue_codes(result)
    assert "missing_condition_directory" in _issue_codes(result)
    assert result.has_errors


def test_discovery_rejects_missing_nominal_values_source(tmp_path: Path) -> None:
    bearing_path = tmp_path / "Data" / "35Hz12kN" / "Bearing1_1"
    _write_acquisition(bearing_path / "1.csv")
    config = _config(tmp_path)
    (tmp_path / config.nominal_values_source).unlink()

    result = discover_dataset(config)

    assert "nominal_values_source_missing" in _issue_codes(result)
    assert result.has_errors


def test_discovery_rejects_symlinked_nominal_values_source(tmp_path: Path) -> None:
    bearing_path = tmp_path / "Data" / "35Hz12kN" / "Bearing1_1"
    _write_acquisition(bearing_path / "1.csv")
    config = _config(tmp_path)
    nominal_source = tmp_path / config.nominal_values_source
    nominal_source.unlink()
    outside_source = tmp_path / "outside.pdf"
    outside_source.write_bytes(b"external provenance")
    _symlink_or_skip(outside_source, nominal_source)

    result = discover_dataset(config)

    assert "nominal_values_source_not_regular_file" in _issue_codes(result)
    assert result.has_errors


def test_discovery_reports_gaps_without_destroying_numeric_order(tmp_path: Path) -> None:
    bearing_path = tmp_path / "Data" / "35Hz12kN" / "Bearing1_1"
    _write_acquisition(bearing_path / "3.csv")
    _write_acquisition(bearing_path / "1.csv")

    result = discover_dataset(_config(tmp_path))

    assert [item.acquisition_number for item in result.acquisitions] == [1, 3]
    assert [item.sequence_index for item in result.acquisitions] == [0, 1]
    gap_issue = next(
        issue for issue in result.issues if issue.code == "missing_acquisition_numbers"
    )
    assert gap_issue.details["missing_numbers"] == (2,)
    assert result.has_errors


def test_discovery_reports_unexpected_directories_and_files(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    condition_path = data_root / "35Hz12kN"
    bearing_path = condition_path / "Bearing1_1"
    _write_acquisition(bearing_path / "1.csv")
    (data_root / "UnexpectedCondition").mkdir()
    (data_root / "notes.txt").write_text("unexpected", encoding="utf-8")
    (condition_path / "UnexpectedBearing").mkdir()
    (condition_path / "notes.txt").write_text("unexpected", encoding="utf-8")
    (bearing_path / "nested").mkdir()
    (bearing_path / "malformed.csv").write_text(_CSV_CONTENT, encoding="utf-8")

    result = discover_dataset(_config(tmp_path))

    assert {
        "unexpected_condition_directory",
        "unexpected_root_file",
        "unexpected_bearing_directory",
        "unexpected_condition_file",
        "non_file_acquisition_entry",
        "malformed_acquisition_filename",
    }.issubset(set(_issue_codes(result)))
    assert len(result.unexpected_entries) == 6
    assert result.has_errors


def test_discovery_reports_duplicate_acquisition_number_when_pattern_allows_it(
    tmp_path: Path,
) -> None:
    bearing_path = tmp_path / "Data" / "35Hz12kN" / "Bearing1_1"
    for filename in ("1.csv", "01.csv", "2.csv"):
        _write_acquisition(bearing_path / filename)
    config = _config(
        tmp_path,
        filename_pattern=r"^(?P<acquisition_number>[0-9]+)\.csv$",
    )

    result = discover_dataset(config)

    duplicate_issue = next(
        issue for issue in result.issues if issue.code == "duplicate_acquisition_number"
    )
    assert duplicate_issue.details["acquisition_number"] == 1
    assert len(duplicate_issue.details["paths"]) == 2
    assert [item.acquisition_number for item in result.acquisitions] == [2]
    assert [item.sequence_index for item in result.acquisitions] == [0]


def test_discovery_rejects_configured_bearing_directory_alias(tmp_path: Path) -> None:
    condition = _condition(bearing_ids=("Bearing1_1", "Bearing1_2"))
    condition_path = tmp_path / "Data" / condition.directory_name
    first_bearing = condition_path / "Bearing1_1"
    second_bearing = condition_path / "Bearing1_2"
    _write_acquisition(first_bearing / "1.csv")
    _symlink_or_skip(first_bearing, second_bearing, target_is_directory=True)

    result = discover_dataset(_config(tmp_path, conditions=(condition,)))

    assert "bearing_directory_symlink" in _issue_codes(result)
    assert [(item.bearing_id, item.acquisition_number) for item in result.acquisitions] == [
        ("Bearing1_1", 1)
    ]
    assert "Data/35Hz12kN/Bearing1_2" in result.unexpected_entries
    assert result.has_errors


def test_discovery_rejects_acquisition_symlink_outside_raw_root(tmp_path: Path) -> None:
    outside_file = tmp_path / "outside.csv"
    _write_acquisition(outside_file)
    bearing_path = tmp_path / "Data" / "35Hz12kN" / "Bearing1_1"
    bearing_path.mkdir(parents=True)
    _symlink_or_skip(outside_file, bearing_path / "1.csv")

    result = discover_dataset(_config(tmp_path))

    assert "acquisition_symlink" in _issue_codes(result)
    assert "no_acquisition_files" in _issue_codes(result)
    assert result.acquisitions == ()
    assert result.has_errors


def test_discovery_rejects_unique_acquisition_symlink_inside_raw_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Data" / "source.csv"
    _write_acquisition(source)
    bearing_path = tmp_path / "Data" / "35Hz12kN" / "Bearing1_1"
    bearing_path.mkdir(parents=True)
    _symlink_or_skip(source, bearing_path / "1.csv")

    result = discover_dataset(_config(tmp_path))

    assert "acquisition_symlink" in _issue_codes(result)
    assert result.acquisitions == ()
    assert result.has_errors


def test_gap_detection_limits_reported_numbers_for_a_large_filename(
    tmp_path: Path,
) -> None:
    bearing_path = tmp_path / "Data" / "35Hz12kN" / "Bearing1_1"
    _write_acquisition(bearing_path / "1.csv")
    _write_acquisition(bearing_path / "1000000000.csv")

    result = discover_dataset(_config(tmp_path))

    issue = next(item for item in result.issues if item.code == "missing_acquisition_numbers")
    assert issue.details["missing_number_count"] == 999_999_998
    assert len(issue.details["missing_numbers"]) == 1_000
    assert issue.details["missing_numbers_truncated"] is True
