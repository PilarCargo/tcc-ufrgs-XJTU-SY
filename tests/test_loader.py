"""Unit tests for strict acquisition CSV inspection."""

from __future__ import annotations

from pathlib import Path

import pytest

from xjtu_sy_tcc.data.loader import inspect_csv

EXPECTED_CHANNELS = (
    "Horizontal_vibration_signals",
    "Vertical_vibration_signals",
)
HEADER = b"Horizontal_vibration_signals,Vertical_vibration_signals"


def _inspect(tmp_path: Path, payload: bytes):
    path = tmp_path / "acquisition.csv"
    path.write_bytes(payload)
    return inspect_csv(path, EXPECTED_CHANNELS)


@pytest.mark.parametrize(
    "payload",
    [
        HEADER + b"\n1.25,-2.5\n0.5,4.0\n",
        HEADER + b"\r\n1.25,-2.5\r\n0.5,4.0\r\n",
        HEADER + b"\n1.25,-2.5\n0.5,4.0",
    ],
    ids=("lf", "crlf", "missing-final-newline"),
)
def test_inspect_csv_accepts_supported_line_endings(
    tmp_path: Path,
    payload: bytes,
) -> None:
    inspection = _inspect(tmp_path, payload)

    assert inspection.file_path == tmp_path / "acquisition.csv"
    assert inspection.channel_names == EXPECTED_CHANNELS
    assert inspection.channel_count == 2
    assert inspection.sample_count == 2
    assert inspection.nan_count == 0
    assert inspection.infinite_count == 0
    assert inspection.parse_error is None


def test_inspect_csv_accepts_an_all_zero_signal(tmp_path: Path) -> None:
    inspection = _inspect(tmp_path, HEADER + b"\n0,0\n-0.0,0e0\n")

    assert inspection.sample_count == 2
    assert inspection.channel_count == 2
    assert inspection.nan_count == 0
    assert inspection.infinite_count == 0
    assert inspection.parse_error is None


@pytest.mark.parametrize(
    ("observed_header", "expected_names"),
    [
        (b"Horizontal_vibration_signals", ("Horizontal_vibration_signals",)),
        (
            b"Horizontal_vibration_signals,Axial_vibration_signals",
            ("Horizontal_vibration_signals", "Axial_vibration_signals"),
        ),
        (
            b"Vertical_vibration_signals,Horizontal_vibration_signals",
            ("Vertical_vibration_signals", "Horizontal_vibration_signals"),
        ),
    ],
    ids=("channel-count", "channel-name", "channel-order"),
)
def test_inspect_csv_rejects_invalid_headers(
    tmp_path: Path,
    observed_header: bytes,
    expected_names: tuple[str, ...],
) -> None:
    inspection = _inspect(tmp_path, observed_header + b"\n0,0\n")

    assert inspection.channel_names == expected_names
    assert inspection.parse_error is not None
    assert "Unexpected channel names or order" in inspection.parse_error


def test_inspect_csv_reports_header_and_data_channel_count_mismatch(
    tmp_path: Path,
) -> None:
    inspection = _inspect(tmp_path, HEADER + b"\n1,2,3\n")

    assert inspection.parse_error is not None
    assert "Numeric channel count does not match the observed header" in inspection.parse_error


@pytest.mark.parametrize(
    ("payload", "error_fragment"),
    [
        (HEADER + b"\nnot-a-number,0\n", "Numeric data could not be parsed"),
        (HEADER + b"\n1,2\n3\n", "Numeric data could not be parsed"),
        (
            HEADER + b"\n1,2\n\n3,4\n",
            "Numeric row count does not match the observed CSV record count",
        ),
    ],
    ids=("nonnumeric", "ragged", "blank-data-line"),
)
def test_inspect_csv_rejects_invalid_data_rows(
    tmp_path: Path,
    payload: bytes,
    error_fragment: str,
) -> None:
    inspection = _inspect(tmp_path, payload)

    assert inspection.parse_error is not None
    assert error_fragment in inspection.parse_error


def test_inspect_csv_rejects_malformed_quoted_csv(tmp_path: Path) -> None:
    inspection = _inspect(
        tmp_path,
        b'"Horizontal_vibration_signals,Vertical_vibration_signals\n1,2\n',
    )

    assert inspection.channel_names == ()
    assert inspection.sample_count == 0
    assert inspection.parse_error is not None
    assert "CSV structure could not be read" in inspection.parse_error


@pytest.mark.parametrize(
    ("payload", "error_fragment"),
    [
        (b"", "empty and has no header"),
        (HEADER + b"\n", "has no data rows"),
    ],
    ids=("empty-file", "header-only"),
)
def test_inspect_csv_rejects_missing_data(
    tmp_path: Path,
    payload: bytes,
    error_fragment: str,
) -> None:
    inspection = _inspect(tmp_path, payload)

    assert inspection.sample_count == 0
    assert inspection.parse_error is not None
    assert error_fragment in inspection.parse_error


def test_inspect_csv_counts_nan_and_infinite_values(tmp_path: Path) -> None:
    inspection = _inspect(tmp_path, HEADER + b"\nnan,inf\n-inf,1\n")

    assert inspection.nan_count == 1
    assert inspection.infinite_count == 2
    assert inspection.parse_error is not None
    assert "Found 1 NaN value(s)" in inspection.parse_error
    assert "Found 2 infinite value(s)" in inspection.parse_error


def test_inspect_csv_rejects_oversized_input_before_loading_it(tmp_path: Path) -> None:
    path = tmp_path / "acquisition.csv"
    with path.open("wb") as file_handle:
        file_handle.truncate(16 * 1024 * 1024 + 1)

    inspection = inspect_csv(path, EXPECTED_CHANNELS)

    assert inspection.sample_count == 0
    assert inspection.parse_error is not None
    assert "16 MiB memory-safety inspection limit" in inspection.parse_error
