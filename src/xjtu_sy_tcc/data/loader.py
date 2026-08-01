"""Strict, memory-bounded inspection of XJTU-SY acquisition CSV files."""

from __future__ import annotations

import csv
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO, TextIOWrapper
from pathlib import Path

import numpy as np

_MAX_CSV_FILE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CsvInspection:
    """Observed structure and numeric validity of one acquisition file."""

    file_path: Path
    channel_names: tuple[str, ...]
    sample_count: int
    channel_count: int
    nan_count: int
    infinite_count: int
    parse_error: str | None


def inspect_csv(
    path: Path,
    expected_channel_names: Sequence[str],
) -> CsvInspection:
    """Inspect one CSV without retaining its vibration samples.

    The acquisition is read into memory once. Its header is decoded and checked
    with :mod:`csv`, while byte-level newline counting records the number of
    physical samples without iterating over every row in Python. NumPy then
    parses the same in-memory bytes and rejects invalid numeric records.
    """

    file_path = Path(path)
    expected_names = tuple(expected_channel_names)
    channel_names: tuple[str, ...] = ()
    sample_count = 0
    errors: list[str] = []
    header_available = False

    try:
        with file_path.open("rb") as file_handle:
            raw_data = file_handle.read(_MAX_CSV_FILE_BYTES + 1)
    except OSError as exc:
        errors.append(f"CSV structure could not be read: {_exception_message(exc)}")
        raw_data = b""
    else:
        if len(raw_data) > _MAX_CSV_FILE_BYTES:
            errors.append("CSV file exceeds the 16 MiB memory-safety inspection limit.")
            raw_data = b""
        elif not raw_data:
            errors.append("The CSV file is empty and has no header.")

    if raw_data:
        physical_line_count = _count_physical_lines(raw_data)
        try:
            with TextIOWrapper(BytesIO(raw_data), encoding="utf-8-sig", newline="") as csv_file:
                reader = csv.reader(csv_file, delimiter=",", strict=True)
                try:
                    header = next(reader)
                except StopIteration:
                    errors.append("The CSV file is empty and has no header.")
                else:
                    header_available = True
                    channel_names = tuple(header)
                    header_line_count = reader.line_num
                    sample_count = max(physical_line_count - header_line_count, 0)
                    _validate_header(
                        channel_names,
                        expected_names,
                        header_line_count,
                        errors,
                    )
                    if sample_count == 0:
                        errors.append("The CSV file has no data rows.")
        except (UnicodeError, csv.Error) as exc:
            errors.append(f"CSV structure could not be read: {_exception_message(exc)}")

    nan_count = 0
    infinite_count = 0
    if header_available and sample_count > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                values = np.loadtxt(
                    BytesIO(raw_data),
                    dtype=np.float32,
                    delimiter=",",
                    comments=None,
                    skiprows=1,
                    ndmin=2,
                )
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"Numeric data could not be parsed: {_exception_message(exc)}")
        else:
            loaded_sample_count = int(values.shape[0])
            loaded_channel_count = int(values.shape[1])
            if loaded_sample_count != sample_count:
                errors.append(
                    "Numeric row count does not match the observed CSV record "
                    f"count: expected {sample_count}, loaded {loaded_sample_count}."
                )
            if loaded_channel_count != len(channel_names):
                errors.append(
                    "Numeric channel count does not match the observed header: "
                    f"header has {len(channel_names)}, data has "
                    f"{loaded_channel_count}."
                )

            nan_count = int(np.count_nonzero(np.isnan(values)))
            infinite_count = int(np.count_nonzero(np.isinf(values)))
            if nan_count:
                errors.append(f"Found {nan_count} NaN value(s).")
            if infinite_count:
                errors.append(f"Found {infinite_count} infinite value(s).")

    return CsvInspection(
        file_path=file_path,
        channel_names=channel_names,
        sample_count=sample_count,
        channel_count=len(channel_names),
        nan_count=nan_count,
        infinite_count=infinite_count,
        parse_error=" ".join(errors) if errors else None,
    )


def _count_physical_lines(raw_data: bytes) -> int:
    """Count LF, CRLF, and CR-delimited physical lines without splitting."""

    if not raw_data:
        return 0

    lf_count = raw_data.count(b"\n")
    cr_count = raw_data.count(b"\r")
    crlf_count = raw_data.count(b"\r\n")
    newline_count = lf_count + cr_count - crlf_count
    has_unterminated_final_line = not raw_data.endswith((b"\n", b"\r"))
    return newline_count + int(has_unterminated_final_line)


def _validate_header(
    observed: tuple[str, ...],
    expected: tuple[str, ...],
    physical_line_count: int,
    errors: list[str],
) -> None:
    if physical_line_count != 1:
        errors.append("The CSV header spans more than one physical line.")
    if not observed:
        errors.append("The CSV header has no channel names.")
    if any(name == "" for name in observed):
        errors.append("The CSV header contains an empty channel name.")
    if len(set(observed)) != len(observed):
        errors.append("The CSV header contains duplicate channel names.")
    if observed != expected:
        errors.append(
            f"Unexpected channel names or order: expected {expected!r}, observed {observed!r}."
        )


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or type(exc).__name__


__all__ = ["CsvInspection", "inspect_csv"]
