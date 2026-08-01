"""Read-only, deterministic discovery of XJTU-SY acquisition files."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from re import Pattern

from xjtu_sy_tcc.config.data import ConditionConfig, DataConfig
from xjtu_sy_tcc.data.models import (
    DiscoveredAcquisition,
    DiscoveryResult,
    Severity,
    ValidationIssue,
)

_POSITIVE_INTEGER = re.compile(r"[0-9]+")
_MAX_ACQUISITION_NUMBER = 2**63 - 1


@dataclass(frozen=True, slots=True)
class _Candidate:
    condition: ConditionConfig
    bearing_id: str
    source_path: Path
    resolved_path: Path
    relative_path: Path
    acquisition_number: int
    file_size_bytes: int
    modified_time_ns: int
    device_id: int
    inode: int


def numeric_file_index(path: Path, filename_pattern: Pattern[str]) -> int | None:
    """Return a strictly positive numeric index from a matching file name.

    The configured expression should expose an ``acquisition_number`` named group.
    The legacy ``sequence_index`` group remains accepted for programmatic callers.
    A single positional group is also supported for direct programmatic use.
    Patterns without a capture group fall back to the complete file stem.
    A non-match, non-ASCII integer, zero, or negative value returns ``None``.
    """

    match = filename_pattern.fullmatch(path.name)
    if match is None:
        return None

    if "acquisition_number" in filename_pattern.groupindex:
        raw_index = match.group("acquisition_number")
    elif "sequence_index" in filename_pattern.groupindex:
        raw_index = match.group("sequence_index")
    elif filename_pattern.groups == 1:
        raw_index = match.group(1)
    elif filename_pattern.groups == 0:
        raw_index = path.stem
    else:
        return None

    if raw_index is None or _POSITIVE_INTEGER.fullmatch(raw_index) is None:
        return None
    numeric_index = int(raw_index)
    return numeric_index if 0 < numeric_index <= _MAX_ACQUISITION_NUMBER else None


def discover_dataset(config: DataConfig) -> DiscoveryResult:
    """Discover expected acquisitions without modifying or opening their contents."""

    issues: list[ValidationIssue] = []
    unexpected_entries: set[str] = set()
    candidates: list[_Candidate] = []
    dataset_root = config.dataset_root.resolve()
    project_root = config.project_root.resolve()

    if not dataset_root.exists():
        issues.append(
            _issue(
                "dataset_root_missing",
                f"Dataset root does not exist: {dataset_root}",
                Severity.ERROR,
                dataset_root,
            )
        )
        return _result((), issues, unexpected_entries)
    if not dataset_root.is_dir():
        issues.append(
            _issue(
                "dataset_root_not_directory",
                f"Dataset root is not a directory: {dataset_root}",
                Severity.ERROR,
                dataset_root,
            )
        )
        return _result((), issues, unexpected_entries)
    if not dataset_root.is_relative_to(project_root):
        issues.append(
            _issue(
                "dataset_root_outside_project",
                "Dataset root must be inside project_root so acquisition paths can be stored "
                "portably.",
                Severity.ERROR,
                dataset_root,
                project_root=project_root.as_posix(),
            )
        )
        return _result((), issues, unexpected_entries)

    nominal_source_path = _configured_nominal_source_path(config)
    nominal_source_is_valid = True
    if not nominal_source_path.is_relative_to(project_root):
        nominal_source_is_valid = False
        issues.append(
            _issue(
                "nominal_values_source_outside_project",
                "Configured nominal-values provenance must be inside the project root.",
                Severity.ERROR,
                nominal_source_path,
            )
        )
    elif not nominal_source_path.exists():
        nominal_source_is_valid = False
        issues.append(
            _issue(
                "nominal_values_source_missing",
                "Configured nominal-values provenance file does not exist.",
                Severity.ERROR,
                nominal_source_path,
            )
        )
    elif (
        nominal_source_path.is_symlink()
        or nominal_source_path.resolve() != nominal_source_path
        or not nominal_source_path.is_file()
    ):
        nominal_source_is_valid = False
        issues.append(
            _issue(
                "nominal_values_source_not_regular_file",
                "Configured nominal-values provenance must be a regular non-symlink file.",
                Severity.ERROR,
                nominal_source_path,
            )
        )

    try:
        root_entries = _sorted_entries(dataset_root)
    except OSError as error:
        issues.append(_unreadable_directory_issue(dataset_root, error))
        return _result((), issues, unexpected_entries)

    conditions = tuple(
        sorted(config.conditions, key=lambda item: (item.condition_id, item.directory_name))
    )
    expected_condition_names = {condition.directory_name for condition in conditions}
    for entry in root_entries:
        if entry.name in expected_condition_names:
            continue
        if nominal_source_is_valid and entry == nominal_source_path:
            continue
        relative_entry = _portable_path(entry, project_root)
        unexpected_entries.add(relative_entry)
        if entry.is_symlink():
            issues.append(
                _issue(
                    "unexpected_root_symlink",
                    f"Unexpected symbolic link at dataset root: {relative_entry}",
                    Severity.ERROR,
                    entry,
                )
            )
        elif entry.is_dir():
            issues.append(
                _issue(
                    "unexpected_condition_directory",
                    f"Unexpected condition directory: {relative_entry}",
                    Severity.ERROR,
                    entry,
                )
            )
        else:
            issues.append(
                _issue(
                    "unexpected_root_file",
                    f"Ignoring non-condition file at dataset root: {relative_entry}",
                    Severity.WARNING,
                    entry,
                )
            )

    resolved_condition_directories: dict[Path, Path] = {}
    resolved_bearing_directories: dict[Path, Path] = {}
    for condition in conditions:
        condition_path = dataset_root / condition.directory_name
        if condition_path.parent != dataset_root:
            issues.append(
                _issue(
                    "condition_path_outside_dataset_root",
                    "Configured condition name is not a direct child of the dataset root.",
                    Severity.ERROR,
                    condition_path,
                    condition_id=condition.condition_id,
                )
            )
            continue
        if not condition_path.exists():
            issues.append(
                _issue(
                    "missing_condition_directory",
                    f"Expected condition directory is missing: {condition.directory_name}",
                    Severity.ERROR,
                    condition_path,
                    condition_id=condition.condition_id,
                )
            )
            continue
        if condition_path.is_symlink():
            unexpected_entries.add(_portable_path(condition_path, project_root))
            issues.append(
                _issue(
                    "condition_directory_symlink",
                    "Expected condition directories must not be symbolic links.",
                    Severity.ERROR,
                    condition_path,
                    condition_id=condition.condition_id,
                )
            )
            continue
        if not condition_path.is_dir():
            issues.append(
                _issue(
                    "condition_path_not_directory",
                    f"Expected condition path is not a directory: {condition.directory_name}",
                    Severity.ERROR,
                    condition_path,
                    condition_id=condition.condition_id,
                )
            )
            continue

        resolved_condition = condition_path.resolve()
        prior_condition = resolved_condition_directories.get(resolved_condition)
        if prior_condition is not None:
            issues.append(
                _alias_issue(
                    "condition_directory_alias",
                    condition_path,
                    prior_condition,
                    resolved_condition,
                )
            )
            unexpected_entries.add(_portable_path(condition_path, project_root))
            continue
        resolved_condition_directories[resolved_condition] = condition_path

        try:
            condition_entries = _sorted_entries(condition_path)
        except OSError as error:
            issues.append(_unreadable_directory_issue(condition_path, error))
            continue

        expected_bearing_names = set(condition.bearing_ids)
        for entry in condition_entries:
            if entry.name in expected_bearing_names:
                continue
            relative_entry = _portable_path(entry, project_root)
            unexpected_entries.add(relative_entry)
            code = "unexpected_bearing_directory" if entry.is_dir() else "unexpected_condition_file"
            issues.append(
                _issue(
                    code,
                    f"Unexpected entry in condition directory: {relative_entry}",
                    Severity.ERROR,
                    entry,
                    condition_id=condition.condition_id,
                )
            )

        for bearing_id in sorted(condition.bearing_ids):
            bearing_path = condition_path / bearing_id
            if bearing_path.parent != condition_path:
                issues.append(
                    _issue(
                        "bearing_path_outside_condition",
                        "Configured bearing ID is not a direct child directory name.",
                        Severity.ERROR,
                        bearing_path,
                        condition_id=condition.condition_id,
                        bearing_id=bearing_id,
                    )
                )
                continue
            if not bearing_path.exists():
                issues.append(
                    _issue(
                        "missing_bearing_directory",
                        f"Expected bearing directory is missing: {bearing_id}",
                        Severity.ERROR,
                        bearing_path,
                        condition_id=condition.condition_id,
                        bearing_id=bearing_id,
                    )
                )
                continue
            if bearing_path.is_symlink():
                unexpected_entries.add(_portable_path(bearing_path, project_root))
                issues.append(
                    _issue(
                        "bearing_directory_symlink",
                        "Expected bearing directories must not be symbolic links.",
                        Severity.ERROR,
                        bearing_path,
                        condition_id=condition.condition_id,
                        bearing_id=bearing_id,
                    )
                )
                continue
            if not bearing_path.is_dir():
                issues.append(
                    _issue(
                        "bearing_path_not_directory",
                        f"Expected bearing path is not a directory: {bearing_id}",
                        Severity.ERROR,
                        bearing_path,
                        condition_id=condition.condition_id,
                        bearing_id=bearing_id,
                    )
                )
                continue

            resolved_bearing = bearing_path.resolve()
            prior_bearing = resolved_bearing_directories.get(resolved_bearing)
            if prior_bearing is not None:
                issues.append(
                    _alias_issue(
                        "bearing_directory_alias",
                        bearing_path,
                        prior_bearing,
                        resolved_bearing,
                    )
                )
                unexpected_entries.add(_portable_path(bearing_path, project_root))
                continue
            resolved_bearing_directories[resolved_bearing] = bearing_path
            bearing_candidates, bearing_issues, bearing_unexpected = _discover_bearing(
                condition=condition,
                bearing_id=bearing_id,
                bearing_path=bearing_path,
                dataset_root=dataset_root,
                project_root=project_root,
                filename_pattern=re.compile(config.filename_pattern),
            )
            candidates.extend(bearing_candidates)
            issues.extend(bearing_issues)
            unexpected_entries.update(bearing_unexpected)

    aliased_sources: set[Path] = set()
    candidates_by_target: dict[Path, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_target[candidate.resolved_path].append(candidate)
    for resolved_path, aliases in sorted(
        candidates_by_target.items(), key=lambda item: item[0].as_posix()
    ):
        if len(aliases) < 2:
            continue
        aliases.sort(key=lambda item: item.source_path.as_posix())
        aliased_sources.update(alias_.source_path for alias_ in aliases)
        issues.append(
            _issue(
                "duplicate_resolved_file",
                "Multiple acquisition paths resolve to the same file.",
                Severity.ERROR,
                aliases[0].source_path,
                resolved_path=resolved_path.as_posix(),
                aliases=tuple(
                    _portable_path(alias_.source_path, project_root) for alias_ in aliases
                ),
            )
        )

    valid_candidates = [
        candidate for candidate in candidates if candidate.source_path not in aliased_sources
    ]
    valid_candidates.sort(
        key=lambda item: (
            item.condition.condition_id,
            item.bearing_id,
            item.acquisition_number,
            item.relative_path.as_posix(),
        )
    )
    acquisitions: list[DiscoveredAcquisition] = []
    sequence_counters: dict[tuple[int, str], int] = defaultdict(int)
    for candidate in valid_candidates:
        key = (candidate.condition.condition_id, candidate.bearing_id)
        sequence_index = sequence_counters[key]
        sequence_counters[key] += 1
        acquisitions.append(
            DiscoveredAcquisition(
                condition=candidate.condition,
                bearing_id=candidate.bearing_id,
                file_path=candidate.resolved_path,
                relative_path=candidate.relative_path,
                acquisition_number=candidate.acquisition_number,
                sequence_index=sequence_index,
                file_size_bytes=candidate.file_size_bytes,
                modified_time_ns=candidate.modified_time_ns,
                device_id=candidate.device_id,
                inode=candidate.inode,
            )
        )

    return _result(acquisitions, issues, unexpected_entries)


def _discover_bearing(
    *,
    condition: ConditionConfig,
    bearing_id: str,
    bearing_path: Path,
    dataset_root: Path,
    project_root: Path,
    filename_pattern: Pattern[str],
) -> tuple[list[_Candidate], list[ValidationIssue], set[str]]:
    issues: list[ValidationIssue] = []
    unexpected_entries: set[str] = set()
    candidates_by_number: dict[int, list[_Candidate]] = defaultdict(list)
    try:
        entries = _sorted_entries(bearing_path)
    except OSError as error:
        return [], [_unreadable_directory_issue(bearing_path, error)], unexpected_entries

    for entry in entries:
        relative_entry = _portable_path(entry, project_root)
        if entry.is_symlink():
            unexpected_entries.add(relative_entry)
            issues.append(
                _issue(
                    "acquisition_symlink",
                    "Acquisition entries must be regular files, not symbolic links.",
                    Severity.ERROR,
                    entry,
                    condition_id=condition.condition_id,
                    bearing_id=bearing_id,
                )
            )
            continue
        if not entry.is_file():
            unexpected_entries.add(relative_entry)
            issues.append(
                _issue(
                    "non_file_acquisition_entry",
                    f"Bearing directory contains a non-file entry: {relative_entry}",
                    Severity.ERROR,
                    entry,
                    condition_id=condition.condition_id,
                    bearing_id=bearing_id,
                )
            )
            continue

        acquisition_number = numeric_file_index(entry, filename_pattern)
        if acquisition_number is None:
            unexpected_entries.add(relative_entry)
            issues.append(
                _issue(
                    "malformed_acquisition_filename",
                    f"Acquisition file name does not match the configured positive numeric "
                    f"pattern: {entry.name}",
                    Severity.ERROR,
                    entry,
                    condition_id=condition.condition_id,
                    bearing_id=bearing_id,
                )
            )
            continue

        try:
            resolved_path = entry.resolve(strict=True)
            file_stat = resolved_path.stat()
        except OSError as error:
            issues.append(
                _issue(
                    "unreadable_acquisition_file",
                    f"Cannot resolve or inspect acquisition file: {error}",
                    Severity.ERROR,
                    entry,
                    condition_id=condition.condition_id,
                    bearing_id=bearing_id,
                )
            )
            continue
        if not resolved_path.is_relative_to(dataset_root):
            issues.append(
                _issue(
                    "acquisition_outside_dataset_root",
                    "Acquisition symlink resolves outside the immutable dataset root.",
                    Severity.ERROR,
                    entry,
                    resolved_path=resolved_path.as_posix(),
                )
            )
            continue

        candidates_by_number[acquisition_number].append(
            _Candidate(
                condition=condition,
                bearing_id=bearing_id,
                source_path=entry,
                resolved_path=resolved_path,
                relative_path=entry.relative_to(project_root),
                acquisition_number=acquisition_number,
                file_size_bytes=file_stat.st_size,
                modified_time_ns=file_stat.st_mtime_ns,
                device_id=file_stat.st_dev,
                inode=file_stat.st_ino,
            )
        )

    numbers = sorted(candidates_by_number)
    if not numbers:
        issues.append(
            _issue(
                "no_acquisition_files",
                f"Bearing directory contains no valid acquisition files: {bearing_id}",
                Severity.ERROR,
                bearing_path,
                condition_id=condition.condition_id,
                bearing_id=bearing_id,
            )
        )
        return [], issues, unexpected_entries

    if numbers[0] != 1:
        issues.append(
            _issue(
                "acquisition_sequence_not_starting_at_one",
                f"Acquisition numbering for {bearing_id} starts at {numbers[0]}, not 1.",
                Severity.ERROR,
                bearing_path,
                first_acquisition_number=numbers[0],
            )
        )
    missing_numbers, missing_number_count = _summarize_missing_numbers(numbers)
    if missing_number_count:
        issues.append(
            _issue(
                "missing_acquisition_numbers",
                f"Acquisition numbering for {bearing_id} is not continuous.",
                Severity.ERROR,
                bearing_path,
                missing_numbers=missing_numbers,
                missing_number_count=missing_number_count,
                missing_numbers_truncated=missing_number_count > len(missing_numbers),
            )
        )

    expected_count = condition.expected_acquisition_counts.get(bearing_id)
    if expected_count is not None and len(numbers) != expected_count:
        issues.append(
            _issue(
                "unexpected_acquisition_count",
                f"Acquisition count for {bearing_id} does not match the documented expectation.",
                Severity.ERROR,
                bearing_path,
                expected_count=expected_count,
                observed_count=len(numbers),
                observed_final_number=numbers[-1],
            )
        )

    candidates: list[_Candidate] = []
    for acquisition_number in numbers:
        numbered_candidates = sorted(
            candidates_by_number[acquisition_number],
            key=lambda item: item.source_path.as_posix(),
        )
        if len(numbered_candidates) > 1:
            issues.append(
                _issue(
                    "duplicate_acquisition_number",
                    f"Multiple files map to acquisition number {acquisition_number} for "
                    f"{bearing_id}.",
                    Severity.ERROR,
                    bearing_path,
                    acquisition_number=acquisition_number,
                    paths=tuple(
                        _portable_path(candidate.source_path, project_root)
                        for candidate in numbered_candidates
                    ),
                )
            )
            continue
        candidates.append(numbered_candidates[0])
    return candidates, issues, unexpected_entries


def _sorted_entries(directory: Path) -> list[Path]:
    return sorted(directory.iterdir(), key=lambda path: path.name)


def _summarize_missing_numbers(
    sorted_numbers: list[int],
    report_limit: int = 1_000,
) -> tuple[tuple[int, ...], int]:
    """Count numeric gaps without materializing an unbounded integer range."""

    reported: list[int] = []
    missing_count = 0
    for previous, current in zip(sorted_numbers, sorted_numbers[1:], strict=False):
        gap_size = max(0, current - previous - 1)
        missing_count += gap_size
        remaining_capacity = report_limit - len(reported)
        if gap_size and remaining_capacity > 0:
            reported.extend(range(previous + 1, previous + 1 + min(gap_size, remaining_capacity)))
    return tuple(reported), missing_count


def _configured_nominal_source_path(config: DataConfig) -> Path:
    configured_path = Path(config.nominal_values_source).expanduser()
    if not configured_path.is_absolute():
        configured_path = config.project_root / configured_path
    return Path(os.path.abspath(configured_path))


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _issue(
    code: str,
    message: str,
    severity: Severity,
    path: Path | None,
    **details: object,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        severity=severity,
        path=path,
        details=details,
    )


def _alias_issue(code: str, path: Path, prior_path: Path, resolved_path: Path) -> ValidationIssue:
    return _issue(
        code,
        "Multiple configured directories resolve to the same physical directory.",
        Severity.ERROR,
        path,
        prior_path=prior_path.as_posix(),
        resolved_path=resolved_path.as_posix(),
    )


def _unreadable_directory_issue(path: Path, error: OSError) -> ValidationIssue:
    return _issue(
        "unreadable_directory",
        f"Cannot enumerate directory: {error}",
        Severity.ERROR,
        path,
    )


def _result(
    acquisitions: tuple[DiscoveredAcquisition, ...] | list[DiscoveredAcquisition],
    issues: list[ValidationIssue],
    unexpected_entries: set[str],
) -> DiscoveryResult:
    sorted_issues = tuple(
        sorted(
            issues,
            key=lambda issue: (
                issue.path.as_posix() if issue.path is not None else "",
                issue.code,
                issue.message,
            ),
        )
    )
    return DiscoveryResult(
        acquisitions=tuple(acquisitions),
        issues=sorted_issues,
        unexpected_entries=tuple(sorted(unexpected_entries)),
    )
