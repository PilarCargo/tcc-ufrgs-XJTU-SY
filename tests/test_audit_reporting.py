"""Integration tests for Phase-1 audit orchestration and reporting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xjtu_sy_tcc.config.data import ConditionConfig, DataConfig
from xjtu_sy_tcc.data.audit import run_audit
from xjtu_sy_tcc.data.metadata import METADATA_COLUMNS
from xjtu_sy_tcc.data.reporting import audit_output_lock, write_audit_outputs

CHANNEL_NAMES = (
    "Horizontal_vibration_signals",
    "Vertical_vibration_signals",
)


def test_passed_audit_writes_validated_outputs(tmp_path: Path) -> None:
    """A complete tiny dataset should produce every validated artifact."""

    project_root = tmp_path / "passed_project"
    config = _make_config(project_root)
    bearing_directory = config.dataset_root / "35Hz12kN" / "Bearing1_1"
    _write_acquisition(
        bearing_directory / "1.csv",
        (("0.1", "0.2"), ("0.3", "0.4"), ("0.5", "0.6")),
    )
    _write_acquisition(
        bearing_directory / "2.csv",
        (("1.1", "1.2"), ("1.3", "1.4"), ("1.5", "1.6")),
    )

    result = run_audit(config)

    assert result.status == "passed"
    assert result.error_count == 0
    assert result.issues == ()
    assert len(result.metadata) == 2

    config.output_directory.mkdir(parents=True)
    raw_path = bearing_directory / "1.csv"
    raw_contents = raw_path.read_bytes()
    (config.output_directory / "dataset_audit.json").symlink_to(raw_path)
    (config.output_directory / ".dataset_audit.json.tmp").symlink_to(raw_path)

    paths = write_audit_outputs(config, result)

    assert raw_path.read_bytes() == raw_contents
    assert not paths["json_report"].is_symlink()
    assert set(paths) == {
        "json_report",
        "markdown_report",
        "metadata_parquet",
        "bearing_summary_parquet",
        "bearing_summary_csv",
    }
    assert all(path.is_file() for path in paths.values())

    metadata = pd.read_parquet(paths["metadata_parquet"])
    assert list(metadata.columns) == list(METADATA_COLUMNS)
    assert len(metadata) == 2
    assert metadata["sequence_index"].tolist() == [0, 1]
    assert metadata["elapsed_minutes"].tolist() == [0.0, 1.0]
    assert metadata["rul_minutes"].tolist() == [1.0, 0.0]
    assert metadata.iloc[-1]["rul_minutes"] == 0.0
    _assert_metadata_schema(pq.read_schema(paths["metadata_parquet"]))

    summary = pd.read_csv(paths["bearing_summary_csv"])
    assert len(summary) == 1
    assert summary.loc[0, "bearing_id"] == "Bearing1_1"
    assert summary.loc[0, "total_acquisitions"] == 2
    assert summary.loc[0, "duration_minutes"] == 1.0
    assert summary.loc[0, "final_rul_minutes"] == 0.0

    payload = json.loads(paths["json_report"].read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["downstream_processing_allowed"] is True
    assert payload["metadata"]["row_count"] == 2
    assert "metadata_parquet" in payload["paths"]["artifacts"]

    markdown = paths["markdown_report"].read_text(encoding="utf-8")
    assert "Status: **PASSED**" in markdown
    assert "Downstream processing allowed: **yes**" in markdown


def test_failed_audit_with_nan_blocks_validated_tables(tmp_path: Path) -> None:
    """A critical CSV error should emit reports but gate all validated tables."""

    project_root = tmp_path / "failed_project"
    config = _make_config(project_root)
    bearing_directory = config.dataset_root / "35Hz12kN" / "Bearing1_1"
    _write_acquisition(
        bearing_directory / "1.csv",
        (("0.1", "0.2"), ("0.3", "0.4"), ("0.5", "0.6")),
    )
    _write_acquisition(
        bearing_directory / "2.csv",
        (("1.1", "1.2"), ("nan", "1.4"), ("1.5", "1.6")),
    )

    result = run_audit(config)

    assert result.status == "failed"
    assert result.error_count > 0
    assert "nan_values" in {issue.code for issue in result.issues}

    config.output_directory.mkdir(parents=True)
    for stale_name in (
        "metadata.parquet",
        "bearing_summary.parquet",
        "bearing_summary.csv",
    ):
        (config.output_directory / stale_name).write_bytes(b"stale validated output")

    paths = write_audit_outputs(config, result)

    assert set(paths) == {"json_report", "markdown_report"}
    assert all(path.is_file() for path in paths.values())
    assert not (config.output_directory / "metadata.parquet").exists()
    assert not (config.output_directory / "bearing_summary.parquet").exists()
    assert not (config.output_directory / "bearing_summary.csv").exists()

    payload = json.loads(paths["json_report"].read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["downstream_processing_allowed"] is False
    assert "metadata_parquet" not in payload["paths"]["artifacts"]

    markdown = paths["markdown_report"].read_text(encoding="utf-8")
    assert "Status: **FAILED**" in markdown
    assert "Downstream processing allowed: **no**" in markdown


def test_serialization_error_removes_all_known_audit_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "serialization_error_project"
    config = _make_config(project_root)
    bearing_directory = config.dataset_root / "35Hz12kN" / "Bearing1_1"
    for acquisition_number in (1, 2):
        _write_acquisition(
            bearing_directory / f"{acquisition_number}.csv",
            (("0.1", "0.2"), ("0.3", "0.4"), ("0.5", "0.6")),
        )
    result = run_audit(config)
    config.output_directory.mkdir(parents=True)
    known_names = (
        "metadata.parquet",
        "bearing_summary.parquet",
        "bearing_summary.csv",
        "dataset_audit.json",
        "dataset_audit.md",
    )
    for name in known_names:
        (config.output_directory / name).write_bytes(b"stale")

    def fail_serialization(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_serialization)

    with pytest.raises(RuntimeError, match="injected serialization failure"):
        write_audit_outputs(config, result)

    assert not any((config.output_directory / name).exists() for name in known_names)


def test_writer_rejects_output_inside_raw_data_before_creating_it(tmp_path: Path) -> None:
    project_root = tmp_path / "unsafe_output_project"
    config = _make_config(project_root)
    bearing_directory = config.dataset_root / "35Hz12kN" / "Bearing1_1"
    for acquisition_number in (1, 2):
        _write_acquisition(
            bearing_directory / f"{acquisition_number}.csv",
            (("0.1", "0.2"), ("0.3", "0.4"), ("0.5", "0.6")),
        )
    result = run_audit(config)
    unsafe_output = config.dataset_root / "generated"
    unsafe_config = replace(config, output_directory=unsafe_output)

    with pytest.raises(ValueError, match="overlaps the immutable dataset root"):
        write_audit_outputs(unsafe_config, result)

    assert not unsafe_output.exists()


def test_audit_output_lock_rejects_concurrent_execution(tmp_path: Path) -> None:
    config = _make_config(tmp_path / "locked_project")

    with audit_output_lock(config):
        with pytest.raises(RuntimeError, match="Another audit is already using"):
            with audit_output_lock(config):
                pytest.fail("A second audit unexpectedly acquired the same output lock")


def _make_config(project_root: Path) -> DataConfig:
    nominal_source = project_root / "data" / "Introduction_to_XJTU-SY_Bearing_Dataset.pdf"
    nominal_source.parent.mkdir(parents=True, exist_ok=True)
    nominal_source.write_bytes(b"unit-test provenance")
    condition = ConditionConfig(
        condition_id=1,
        directory_name="35Hz12kN",
        bearing_ids=("Bearing1_1",),
        rotation_hz=35.0,
        rotation_rpm=2100.0,
        radial_load_kn=12.0,
        expected_acquisition_counts={"Bearing1_1": 2},
    )
    return DataConfig(
        config_path=project_root / "configs" / "data.yaml",
        project_root=project_root,
        dataset_root=project_root / "data",
        output_directory=project_root / "outputs" / "audits",
        nominal_values_source="data/Introduction_to_XJTU-SY_Bearing_Dataset.pdf",
        acquisition_interval_minutes=1.0,
        sampling_frequency_hz=25_600.0,
        expected_sample_count=3,
        expected_channel_names=CHANNEL_NAMES,
        filename_pattern=r"^(?P<acquisition_number>[1-9][0-9]*)\.csv$",
        detect_duplicate_content=False,
        max_workers=1,
        conditions=(condition,),
    )


def _write_acquisition(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = [",".join(CHANNEL_NAMES)]
    contents.extend(",".join(row) for row in rows)
    path.write_text("\n".join(contents) + "\n", encoding="utf-8")


def _assert_metadata_schema(schema: pa.Schema) -> None:
    assert schema.names == list(METADATA_COLUMNS)
    for column in ("bearing_id", "file_path", "file_name"):
        column_type = schema.field(column).type
        assert pa.types.is_string(column_type) or pa.types.is_large_string(column_type)

    expected_numeric_types = {
        "condition_id": pa.int64(),
        "acquisition_number": pa.int64(),
        "sequence_index": pa.int64(),
        "elapsed_minutes": pa.float64(),
        "total_acquisitions": pa.int64(),
        "rul_minutes": pa.float64(),
        "rotation_rpm": pa.float64(),
        "radial_load_kn": pa.float64(),
        "sampling_frequency_hz": pa.float64(),
        "sample_count": pa.int64(),
        "channel_count": pa.int32(),
        "file_size_bytes": pa.int64(),
    }
    for column, expected_type in expected_numeric_types.items():
        assert schema.field(column).type == expected_type
