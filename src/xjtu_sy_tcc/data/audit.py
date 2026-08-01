"""End-to-end Phase-1 audit orchestration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from xjtu_sy_tcc.config.data import DataConfig
from xjtu_sy_tcc.data.discovery import discover_dataset
from xjtu_sy_tcc.data.loader import CsvInspection, inspect_csv
from xjtu_sy_tcc.data.metadata import build_bearing_summary, build_metadata
from xjtu_sy_tcc.data.models import DiscoveredAcquisition, Severity, ValidationIssue
from xjtu_sy_tcc.data.validation import (
    DuplicateCheckResult,
    detect_duplicate_content,
    validate_csv_inspections,
    validate_metadata,
)
from xjtu_sy_tcc.logging_utils import log_event

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AuditResult:
    """Validated metadata, reports inputs, and audit execution statistics."""

    status: str
    started_at_utc: str
    completed_at_utc: str
    duration_seconds: float
    inspection_seconds: float
    duplicate_check_seconds: float
    metadata: pd.DataFrame
    bearing_summary: pd.DataFrame
    issues: tuple[ValidationIssue, ...]
    statistics: dict[str, object]
    duplicate_check: DuplicateCheckResult

    @property
    def error_count(self) -> int:
        """Return the number of critical validation issues."""

        return sum(issue.severity is Severity.ERROR for issue in self.issues)

    @property
    def warning_count(self) -> int:
        """Return the number of non-critical validation warnings."""

        return sum(issue.severity is Severity.WARNING for issue in self.issues)


def run_audit(config: DataConfig) -> AuditResult:
    """Run discovery, exhaustive CSV inspection, metadata, and invariants."""

    started_at = datetime.now(UTC)
    started_timer = perf_counter()
    log_event(
        LOGGER,
        logging.INFO,
        "audit_started",
        "XJTU-SY dataset audit started.",
        dataset_root=config.dataset_root,
        max_workers=config.max_workers,
    )

    discovery = discover_dataset(config)
    log_event(
        LOGGER,
        logging.INFO,
        "discovery_completed",
        "Dataset discovery completed.",
        acquisition_count=len(discovery.acquisitions),
        issue_count=len(discovery.issues),
    )

    inspection_started = perf_counter()
    inspections = _inspect_acquisitions(config, discovery.acquisitions)
    inspection_seconds = perf_counter() - inspection_started

    metadata = build_metadata(discovery.acquisitions, inspections, config)
    metadata_issues = validate_metadata(metadata, config)
    bearing_summary = build_bearing_summary(metadata)

    duplicate_started = perf_counter()
    if config.detect_duplicate_content:
        duplicate_check = detect_duplicate_content(
            discovery.acquisitions,
            config.max_workers,
        )
    else:
        duplicate_check = DuplicateCheckResult(
            duplicate_groups=(),
            equal_size_group_count=0,
            boundary_fingerprint_file_count=0,
            full_hash_file_count=0,
            issues=(),
        )
    duplicate_check_seconds = perf_counter() - duplicate_started
    inspection_issues = validate_csv_inspections(
        discovery.acquisitions,
        inspections,
        config,
    )

    issues = _sort_issues(
        (*discovery.issues, *inspection_issues, *metadata_issues, *duplicate_check.issues)
    )
    status = "failed" if any(issue.severity is Severity.ERROR for issue in issues) else "passed"
    completed_at = datetime.now(UTC)
    duration_seconds = perf_counter() - started_timer
    statistics = _build_statistics(
        config,
        metadata,
        inspections,
        len(discovery.unexpected_entries),
        duplicate_check,
    )
    result = AuditResult(
        status=status,
        started_at_utc=started_at.isoformat(),
        completed_at_utc=completed_at.isoformat(),
        duration_seconds=duration_seconds,
        inspection_seconds=inspection_seconds,
        duplicate_check_seconds=duplicate_check_seconds,
        metadata=metadata,
        bearing_summary=bearing_summary,
        issues=issues,
        statistics=statistics,
        duplicate_check=duplicate_check,
    )
    log_event(
        LOGGER,
        logging.INFO if status == "passed" else logging.ERROR,
        "audit_completed",
        f"XJTU-SY dataset audit {status}.",
        status=status,
        acquisition_count=len(metadata),
        error_count=result.error_count,
        warning_count=result.warning_count,
        duration_seconds=round(duration_seconds, 3),
    )
    return result


def _inspect_acquisitions(
    config: DataConfig,
    acquisitions: tuple[DiscoveredAcquisition, ...],
) -> dict[Path, CsvInspection]:
    expected_names = config.expected_channel_names

    def inspect(acquisition: DiscoveredAcquisition) -> CsvInspection:
        path = acquisition.file_path
        try:
            return inspect_csv(path, expected_names)
        except Exception as error:  # noqa: BLE001 - one bad file must not abort the audit
            return CsvInspection(
                file_path=path,
                channel_names=(),
                sample_count=0,
                channel_count=0,
                nan_count=0,
                infinite_count=0,
                parse_error=f"Unexpected inspection failure: {type(error).__name__}: {error}",
            )

    inspections: dict[Path, CsvInspection] = {}
    if not acquisitions:
        return inspections

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        for completed, inspection in enumerate(executor.map(inspect, acquisitions), start=1):
            inspections[inspection.file_path] = inspection
            if completed % 500 == 0 or completed == len(acquisitions):
                log_event(
                    LOGGER,
                    logging.INFO,
                    "csv_inspection_progress",
                    "CSV inspection progress.",
                    completed=completed,
                    total=len(acquisitions),
                )
    return inspections


def _build_statistics(
    config: DataConfig,
    metadata: pd.DataFrame,
    inspections: Mapping[Path, CsvInspection],
    unexpected_entry_count: int,
    duplicate_check: DuplicateCheckResult,
) -> dict[str, object]:
    observed_headers = sorted(
        {inspection.channel_names for inspection in inspections.values()},
        key=lambda names: tuple(names),
    )
    condition_count = int(metadata["condition_id"].nunique()) if not metadata.empty else 0
    bearing_count = int(metadata["bearing_id"].nunique()) if not metadata.empty else 0
    configured_counts = [
        condition.expected_acquisition_counts.get(bearing_id)
        for condition in config.conditions
        for bearing_id in condition.bearing_ids
    ]
    expected_acquisition_count = (
        sum(count for count in configured_counts if count is not None)
        if all(count is not None for count in configured_counts)
        else None
    )
    return {
        "expected_condition_count": len(config.conditions),
        "observed_condition_count": condition_count,
        "expected_bearing_count": sum(len(item.bearing_ids) for item in config.conditions),
        "observed_bearing_count": bearing_count,
        "expected_acquisition_count": expected_acquisition_count,
        "acquisition_count": len(metadata),
        "total_file_size_bytes": (
            int(metadata["file_size_bytes"].sum()) if not metadata.empty else 0
        ),
        "total_physical_data_rows": sum(item.sample_count for item in inspections.values()),
        "files_with_inspection_errors": sum(
            item.parse_error is not None for item in inspections.values()
        ),
        "observed_channel_headers": [list(names) for names in observed_headers],
        "unexpected_entry_count": unexpected_entry_count,
        "duplicate_detection_enabled": config.detect_duplicate_content,
        "duplicate_detection_method": (
            "equal byte size, SHA-256 boundary fingerprint, then full SHA-256 confirmation"
            if config.detect_duplicate_content
            else "disabled"
        ),
        "equal_size_group_count": duplicate_check.equal_size_group_count,
        "boundary_fingerprint_file_count": duplicate_check.boundary_fingerprint_file_count,
        "full_hash_file_count": duplicate_check.full_hash_file_count,
        "duplicate_content_group_count": len(duplicate_check.duplicate_groups),
    }


def _sort_issues(issues: tuple[ValidationIssue, ...]) -> tuple[ValidationIssue, ...]:
    return tuple(
        sorted(
            issues,
            key=lambda issue: (
                0 if issue.severity is Severity.ERROR else 1,
                issue.path.as_posix() if issue.path is not None else "",
                issue.code,
            ),
        )
    )
