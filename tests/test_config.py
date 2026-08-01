"""Tests for strict data-configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from xjtu_sy_tcc.config import load_data_config
from xjtu_sy_tcc.exceptions import ConfigurationError


def _valid_payload() -> dict[str, Any]:
    return {
        "project_root": "..",
        "dataset_root": "Data",
        "output_directory": "outputs",
        "nominal_values_source": "Data/Introduction_to_XJTU-SY_Bearing_Dataset.pdf",
        "acquisition_interval_minutes": 1.0,
        "sampling_frequency_hz": 25_600.0,
        "expected_sample_count": 32_768,
        "expected_channel_names": [
            "Horizontal_vibration_signals",
            "Vertical_vibration_signals",
        ],
        "filename_pattern": r"^(?P<acquisition_number>[1-9][0-9]*)\.csv$",
        "detect_duplicate_content": True,
        "max_workers": 2,
        "conditions": [
            {
                "condition_id": 1,
                "directory_name": "35Hz12kN",
                "bearing_ids": ["Bearing1_1", "Bearing1_2"],
                "expected_acquisition_counts": {
                    "Bearing1_1": 123,
                    "Bearing1_2": 161,
                },
                "rotation_hz": 35.0,
                "rotation_rpm": 2_100.0,
                "radial_load_kn": 12.0,
            }
        ],
    }


def _write_config(tmp_path: Path, payload: dict[str, Any]) -> Path:
    config_path = tmp_path / "configs" / "data.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_load_data_config_accepts_valid_configuration(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _valid_payload())

    config = load_data_config(config_path)

    assert config.config_path == config_path.resolve()
    assert config.project_root == tmp_path.resolve()
    assert config.dataset_root == (tmp_path / "Data").resolve()
    assert config.output_directory == (tmp_path / "outputs").resolve()
    assert config.nominal_values_source == ("Data/Introduction_to_XJTU-SY_Bearing_Dataset.pdf")
    assert config.expected_channel_names == (
        "Horizontal_vibration_signals",
        "Vertical_vibration_signals",
    )
    assert config.expected_channel_count == 2
    assert config.conditions[0].bearing_ids == ("Bearing1_1", "Bearing1_2")
    assert dict(config.conditions[0].expected_acquisition_counts) == {
        "Bearing1_1": 123,
        "Bearing1_2": 161,
    }
    assert config.conditions[0].rotation_rpm == 2_100.0


def test_load_data_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["unknown_setting"] = "unsupported"
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigurationError, match="unexpected keys: unknown_setting"):
        load_data_config(config_path)


def test_load_data_config_rejects_missing_top_level_key(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["sampling_frequency_hz"]
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigurationError, match="missing keys: sampling_frequency_hz"):
        load_data_config(config_path)


@pytest.mark.parametrize(
    "channel_names",
    [
        pytest.param([], id="empty-list"),
        pytest.param("Horizontal_vibration_signals", id="not-a-list"),
        pytest.param(["Horizontal_vibration_signals", 1], id="nonstr-member"),
        pytest.param(
            ["Horizontal_vibration_signals", "Horizontal_vibration_signals"],
            id="duplicate-member",
        ),
    ],
)
def test_load_data_config_rejects_invalid_channel_lists(
    tmp_path: Path, channel_names: object
) -> None:
    payload = _valid_payload()
    payload["expected_channel_names"] = channel_names
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigurationError, match="expected_channel_names"):
        load_data_config(config_path)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        pytest.param("expected_sample_count", True, id="boolean-sample-count"),
        pytest.param("sampling_frequency_hz", "25600", id="string-frequency"),
        pytest.param("detect_duplicate_content", 1, id="integer-boolean"),
        pytest.param("max_workers", 1.5, id="fractional-worker-count"),
        pytest.param("max_workers", 9, id="excessive-worker-count"),
    ],
)
def test_load_data_config_rejects_invalid_scalar_types(
    tmp_path: Path, field_name: str, invalid_value: object
) -> None:
    payload = _valid_payload()
    payload[field_name] = invalid_value
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigurationError, match=field_name):
        load_data_config(config_path)


@pytest.mark.parametrize(
    ("dataset_root", "output_directory"),
    [
        pytest.param("Data", "Data", id="equal"),
        pytest.param("Data", "Data/generated", id="output-inside-raw"),
        pytest.param("generated/raw", "generated", id="raw-inside-output"),
    ],
)
def test_load_data_config_rejects_overlapping_paths(
    tmp_path: Path, dataset_root: str, output_directory: str
) -> None:
    payload = _valid_payload()
    payload["dataset_root"] = dataset_root
    payload["output_directory"] = output_directory
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigurationError, match="must not be equal, nested, or otherwise overlap"):
        load_data_config(config_path)


def test_load_data_config_rejects_rpm_mismatch(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["conditions"][0]["rotation_rpm"] = 2_101.0
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigurationError, match=r"rotation_rpm must equal rotation_hz \* 60"):
        load_data_config(config_path)


def test_load_data_config_rejects_condition_id_outside_int64(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["conditions"][0]["condition_id"] = 2**63
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigurationError, match="condition_id"):
        load_data_config(config_path)


@pytest.mark.parametrize(
    "counts",
    [
        pytest.param({"Bearing1_1": 123}, id="missing-bearing"),
        pytest.param(
            {"Bearing1_1": 123, "Bearing1_2": 161, "Bearing1_3": 10},
            id="unexpected-bearing",
        ),
        pytest.param({"Bearing1_1": 123, "Bearing1_2": 0}, id="nonpositive-count"),
    ],
)
def test_load_data_config_rejects_invalid_expected_acquisition_counts(
    tmp_path: Path,
    counts: dict[str, int],
) -> None:
    payload = _valid_payload()
    payload["conditions"][0]["expected_acquisition_counts"] = counts
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigurationError, match="expected_acquisition_counts"):
        load_data_config(config_path)


@pytest.mark.parametrize("bearing_id", ["../escape", "/absolute", "nested/name", r"nested\name"])
def test_load_data_config_rejects_unsafe_bearing_directory_names(
    tmp_path: Path,
    bearing_id: str,
) -> None:
    payload = _valid_payload()
    payload["conditions"][0]["bearing_ids"] = [bearing_id]
    payload["conditions"][0]["expected_acquisition_counts"] = {bearing_id: 1}
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigurationError, match="single relative directory name"):
        load_data_config(config_path)


@pytest.mark.parametrize(
    "directory_name",
    [
        pytest.param("36Hz12kN", id="rotation-mismatch"),
        pytest.param("35Hz11kN", id="load-mismatch"),
        pytest.param("condition-one", id="malformed-label"),
    ],
)
def test_load_data_config_rejects_condition_folder_mismatch(
    tmp_path: Path, directory_name: str
) -> None:
    payload = _valid_payload()
    payload["conditions"][0]["directory_name"] = directory_name
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigurationError, match="directory_name"):
        load_data_config(config_path)
