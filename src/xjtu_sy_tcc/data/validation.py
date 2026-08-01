"""Dataset-level validation and exact duplicate-content detection."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xjtu_sy_tcc.config.data import DataConfig
from xjtu_sy_tcc.data.discovery import DiscoveredAcquisition
from xjtu_sy_tcc.data.loader import CsvInspection
from xjtu_sy_tcc.data.models import Severity, ValidationIssue

_BOUNDARY_BLOCK_BYTES = 65_536


@dataclass(frozen=True, slots=True)
class DuplicateCheckResult:
    """Summary of an exact duplicate-content scan."""

    duplicate_groups: tuple[tuple[Path, ...], ...]
    equal_size_group_count: int
    boundary_fingerprint_file_count: int
    full_hash_file_count: int
    issues: tuple[ValidationIssue, ...]


def validate_csv_inspections(
    acquisitions: Sequence[DiscoveredAcquisition],
    inspections: Mapping[Path, CsvInspection],
    config: DataConfig,
) -> tuple[ValidationIssue, ...]:
    """Validate every observed CSV against the configured acquisition schema."""

    issues: list[ValidationIssue] = []
    expected_names = config.expected_channel_names
    for acquisition in acquisitions:
        inspection = inspections.get(acquisition.file_path)
        if inspection is None:
            issues.append(
                _issue(
                    "missing_csv_inspection",
                    "No CSV inspection result was produced for a discovered acquisition.",
                    acquisition.file_path,
                )
            )
            continue

        if inspection.parse_error is not None:
            issues.append(
                _issue(
                    "csv_inspection_error",
                    "CSV structure or numeric parsing failed.",
                    acquisition.file_path,
                    parse_error=inspection.parse_error,
                )
            )
        if inspection.channel_names != expected_names:
            issues.append(
                _issue(
                    "invalid_channel_names",
                    "CSV channel names or their order do not match the configuration.",
                    acquisition.file_path,
                    expected=expected_names,
                    observed=inspection.channel_names,
                )
            )
        if inspection.channel_count != config.expected_channel_count:
            issues.append(
                _issue(
                    "invalid_channel_count",
                    "CSV channel count does not match the configuration.",
                    acquisition.file_path,
                    expected=config.expected_channel_count,
                    observed=inspection.channel_count,
                )
            )
        if inspection.sample_count != config.expected_sample_count:
            issues.append(
                _issue(
                    "invalid_sample_count",
                    "CSV sample count does not match the configuration.",
                    acquisition.file_path,
                    expected=config.expected_sample_count,
                    observed=inspection.sample_count,
                )
            )
        if inspection.nan_count:
            issues.append(
                _issue(
                    "nan_values",
                    "CSV contains NaN vibration values.",
                    acquisition.file_path,
                    count=inspection.nan_count,
                )
            )
        if inspection.infinite_count:
            issues.append(
                _issue(
                    "infinite_values",
                    "CSV contains infinite vibration values.",
                    acquisition.file_path,
                    count=inspection.infinite_count,
                )
            )
        try:
            current_stat = acquisition.file_path.stat()
        except OSError as error:
            issues.append(
                _issue(
                    "acquisition_stat_error",
                    f"Cannot inspect the acquisition after parsing: {error}",
                    acquisition.file_path,
                )
            )
        else:
            if current_stat.st_size != acquisition.file_size_bytes:
                issues.append(
                    _issue(
                        "acquisition_size_changed",
                        "Acquisition size changed during the audit.",
                        acquisition.file_path,
                        discovered_size=acquisition.file_size_bytes,
                        current_size=current_stat.st_size,
                    )
                )
            if (
                acquisition.modified_time_ns is not None
                and current_stat.st_mtime_ns != acquisition.modified_time_ns
            ):
                issues.append(
                    _issue(
                        "acquisition_modified_during_audit",
                        "Acquisition modification time changed during the audit.",
                        acquisition.file_path,
                        discovered_modified_time_ns=acquisition.modified_time_ns,
                        current_modified_time_ns=current_stat.st_mtime_ns,
                    )
                )
            if (
                acquisition.device_id is not None
                and acquisition.inode is not None
                and (
                    current_stat.st_dev != acquisition.device_id
                    or current_stat.st_ino != acquisition.inode
                )
            ):
                issues.append(
                    _issue(
                        "acquisition_replaced_during_audit",
                        "Acquisition filesystem identity changed during the audit.",
                        acquisition.file_path,
                    )
                )
    return _sorted_issues(issues)


def validate_metadata(metadata: pd.DataFrame, config: DataConfig) -> tuple[ValidationIssue, ...]:
    """Validate ordering, elapsed-time, and RUL invariants in metadata."""

    issues: list[ValidationIssue] = []
    if metadata.empty:
        return (_issue("empty_metadata", "No acquisition metadata was generated."),)

    duplicated_paths = metadata.loc[metadata["file_path"].duplicated(False), "file_path"]
    if not duplicated_paths.empty:
        issues.append(
            _issue(
                "duplicate_metadata_file_paths",
                "Metadata contains duplicate file paths.",
                paths=tuple(sorted(set(duplicated_paths.astype(str)))),
            )
        )

    grouped = metadata.groupby(["condition_id", "bearing_id"], sort=True)
    for (condition_id, bearing_id), bearing_rows in grouped:
        rows = bearing_rows.sort_values("sequence_index", kind="stable")
        sequence = rows["sequence_index"].to_numpy(dtype=np.int64)
        expected_sequence = np.arange(len(rows), dtype=np.int64)
        context = {"condition_id": int(condition_id), "bearing_id": str(bearing_id)}

        if not np.array_equal(sequence, expected_sequence):
            issues.append(
                _issue(
                    "invalid_metadata_sequence",
                    "Sequence indices are not unique, continuous, and zero-based.",
                    **context,
                )
            )
        acquisition_numbers = rows["acquisition_number"].to_numpy(dtype=np.int64)
        if len(acquisition_numbers) > 1 and np.any(np.diff(acquisition_numbers) <= 0):
            issues.append(
                _issue(
                    "non_monotonic_acquisition_numbers",
                    "Acquisition numbers are not unique and strictly increasing.",
                    **context,
                )
            )

        configured_interval = config.acquisition_interval_minutes
        expected_elapsed = sequence.astype(np.float64) * configured_interval
        elapsed = rows["elapsed_minutes"].to_numpy(dtype=np.float64)
        if not np.allclose(elapsed, expected_elapsed, rtol=0.0, atol=1e-12):
            issues.append(
                _issue(
                    "invalid_elapsed_minutes",
                    "Elapsed time does not match sequence_index times the configured interval.",
                    **context,
                )
            )

        expected_rul = (len(rows) - 1 - sequence.astype(np.float64)) * configured_interval
        observed_rul = rows["rul_minutes"].to_numpy(dtype=np.float64)
        if not np.allclose(observed_rul, expected_rul, rtol=0.0, atol=1e-12):
            issues.append(
                _issue(
                    "invalid_rul_calculation",
                    "RUL does not match the configured zero-based formula.",
                    **context,
                )
            )
        if np.any(observed_rul < 0.0) or not np.all(np.isfinite(observed_rul)):
            issues.append(
                _issue(
                    "invalid_rul_values",
                    "RUL contains a negative or non-finite value.",
                    **context,
                )
            )
        if not math.isclose(float(observed_rul[-1]), 0.0, rel_tol=0.0, abs_tol=1e-12):
            issues.append(
                _issue(
                    "nonzero_final_rul",
                    "The final acquisition does not have RUL equal to zero.",
                    **context,
                    observed=float(observed_rul[-1]),
                )
            )
        if not (rows["total_acquisitions"] == len(rows)).all():
            issues.append(
                _issue(
                    "invalid_total_acquisitions",
                    "total_acquisitions is inconsistent within the bearing.",
                    **context,
                )
            )
    return _sorted_issues(issues)


def detect_duplicate_content(
    acquisitions: Sequence[DiscoveredAcquisition],
    max_workers: int,
) -> DuplicateCheckResult:
    """Find byte-identical files with boundary filtering and full SHA-256 confirmation.

    Exact duplicates must have equal byte sizes and equal boundary fingerprints.
    Full-file hashing is therefore required only for boundary-matching candidates,
    while retaining exact duplicate detection semantics.
    """

    issues: list[ValidationIssue] = []
    by_size: dict[int, list[Path]] = defaultdict(list)
    for acquisition in acquisitions:
        by_size[acquisition.file_size_bytes].append(acquisition.file_path)
    equal_size_groups = [
        tuple(sorted(paths, key=Path.as_posix)) for paths in by_size.values() if len(paths) > 1
    ]
    candidate_paths = tuple(path for group in equal_size_groups for path in group)
    discovered_sizes = {
        acquisition.file_path: acquisition.file_size_bytes for acquisition in acquisitions
    }

    boundary_results = _parallel_digests(
        candidate_paths,
        _boundary_digest,
        max_workers,
    )
    by_boundary: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for path, digest, error in boundary_results:
        if error is not None:
            issues.append(
                _issue(
                    "duplicate_scan_read_error",
                    f"Cannot compute duplicate boundary fingerprint: {error}",
                    path,
                )
            )
            continue
        by_boundary[(discovered_sizes[path], digest)].append(path)

    full_hash_candidates = tuple(
        path
        for paths in by_boundary.values()
        if len(paths) > 1
        for path in sorted(paths, key=Path.as_posix)
    )
    full_results = _parallel_digests(full_hash_candidates, _full_digest, max_workers)
    by_full_hash: dict[str, list[Path]] = defaultdict(list)
    for path, digest, error in full_results:
        if error is not None:
            issues.append(
                _issue(
                    "duplicate_scan_read_error",
                    f"Cannot compute full duplicate hash: {error}",
                    path,
                )
            )
            continue
        by_full_hash[digest].append(path)

    duplicate_groups = tuple(
        tuple(sorted(paths, key=Path.as_posix)) for paths in by_full_hash.values() if len(paths) > 1
    )
    duplicate_groups = tuple(sorted(duplicate_groups, key=lambda paths: paths[0].as_posix()))
    for paths in duplicate_groups:
        issues.append(
            _issue(
                "duplicate_file_content",
                "Multiple acquisition files have byte-identical content.",
                paths[0],
                paths=tuple(path.as_posix() for path in paths),
            )
        )

    return DuplicateCheckResult(
        duplicate_groups=duplicate_groups,
        equal_size_group_count=len(equal_size_groups),
        boundary_fingerprint_file_count=len(candidate_paths),
        full_hash_file_count=len(full_hash_candidates),
        issues=_sorted_issues(issues),
    )


def _parallel_digests(
    paths: Sequence[Path],
    digest_function: Callable[[Path], str],
    max_workers: int,
) -> list[tuple[Path, str, str | None]]:
    if not paths:
        return []

    def calculate(path: Path) -> tuple[Path, str, str | None]:
        try:
            digest = digest_function(path)
        except OSError as error:
            return path, "", str(error)
        return path, str(digest), None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(calculate, paths))


def _boundary_digest(path: Path) -> str:
    size = path.stat().st_size
    with path.open("rb") as file_handle:
        first = file_handle.read(_BOUNDARY_BLOCK_BYTES)
        if size > _BOUNDARY_BLOCK_BYTES:
            file_handle.seek(max(0, size - _BOUNDARY_BLOCK_BYTES))
            last = file_handle.read(_BOUNDARY_BLOCK_BYTES)
        else:
            last = b""
    digest = hashlib.sha256()
    digest.update(size.to_bytes(8, byteorder="big", signed=False))
    digest.update(first)
    digest.update(b"\x00xjtu-boundary\x00")
    digest.update(last)
    return digest.hexdigest()


def _full_digest(path: Path) -> str:
    with path.open("rb") as file_handle:
        return hashlib.file_digest(file_handle, "sha256").hexdigest()


def _issue(
    code: str,
    message: str,
    path: Path | None = None,
    **details: object,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        severity=Severity.ERROR,
        path=path,
        details=details,
    )


def _sorted_issues(issues: Sequence[ValidationIssue]) -> tuple[ValidationIssue, ...]:
    return tuple(
        sorted(
            issues,
            key=lambda issue: (
                issue.path.as_posix() if issue.path is not None else "",
                issue.code,
                issue.message,
            ),
        )
    )
