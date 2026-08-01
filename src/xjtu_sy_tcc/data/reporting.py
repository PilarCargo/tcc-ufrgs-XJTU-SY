"""Machine-readable and human-readable Phase-1 audit outputs."""

from __future__ import annotations

import json
import os
import platform
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

if os.name == "nt":
    import msvcrt
else:
    import fcntl

import pandas as pd

from xjtu_sy_tcc.config.data import DataConfig
from xjtu_sy_tcc.data.audit import AuditResult
from xjtu_sy_tcc.data.models import ValidationIssue

_VALIDATED_TABLE_NAMES = (
    "metadata.parquet",
    "bearing_summary.parquet",
    "bearing_summary.csv",
)
_AUDIT_ARTIFACT_NAMES = (
    *_VALIDATED_TABLE_NAMES,
    "dataset_audit.json",
    "dataset_audit.md",
)


@contextmanager
def audit_output_lock(config: DataConfig) -> Iterator[None]:
    """Hold an exclusive process lock for one complete audit execution."""

    _validate_output_directory(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    _validate_output_directory(config)
    lock_path = config.output_directory / ".audit.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RuntimeError(f"Cannot open secure audit lock {lock_path}: {error}") from error

    locked = False
    try:
        try:
            if os.name == "nt":
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise RuntimeError(
                f"Another audit is already using output directory {config.output_directory}"
            ) from error
        yield
    finally:
        if locked:
            if os.name == "nt":
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def write_audit_outputs(config: DataConfig, result: AuditResult) -> dict[str, Path]:
    """Write audit reports and validated tables without touching raw data."""

    output_directory = config.output_directory
    _validate_output_directory(config)
    output_directory.mkdir(parents=True, exist_ok=True)
    _validate_output_directory(config)
    paths = {
        "json_report": output_directory / "dataset_audit.json",
        "markdown_report": output_directory / "dataset_audit.md",
    }

    try:
        if result.status == "passed":
            paths.update(
                {
                    "metadata_parquet": output_directory / "metadata.parquet",
                    "bearing_summary_parquet": output_directory / "bearing_summary.parquet",
                    "bearing_summary_csv": output_directory / "bearing_summary.csv",
                }
            )
            _atomic_parquet(result.metadata, paths["metadata_parquet"])
            _atomic_parquet(result.bearing_summary, paths["bearing_summary_parquet"])
            _atomic_csv(result.bearing_summary, paths["bearing_summary_csv"])
        else:
            _remove_stale_validated_tables(output_directory)

        payload = build_audit_payload(config, result, paths)
        _atomic_text(
            paths["json_report"],
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        _atomic_text(paths["markdown_report"], render_markdown_report(config, result, paths))
    except BaseException:
        invalidate_audit_outputs(output_directory)
        raise
    return paths


def invalidate_audit_outputs(output_directory: Path) -> None:
    """Remove only known generated artifacts after an incomplete execution."""

    for artifact_name in _AUDIT_ARTIFACT_NAMES:
        artifact = output_directory / artifact_name
        if artifact.is_file() or artifact.is_symlink():
            artifact.unlink(missing_ok=True)


def build_audit_payload(
    config: DataConfig,
    result: AuditResult,
    output_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Build the stable JSON representation of an audit result."""

    paths = output_paths or {}
    expected_conditions = [
        {
            "condition_id": condition.condition_id,
            "directory_name": condition.directory_name,
            "bearing_ids": list(condition.bearing_ids),
            "expected_acquisition_counts": dict(condition.expected_acquisition_counts),
            "rotation_hz": condition.rotation_hz,
            "rotation_rpm": condition.rotation_rpm,
            "radial_load_kn": condition.radial_load_kn,
        }
        for condition in config.conditions
    ]
    return {
        "audit_schema_version": 1,
        "status": result.status,
        "downstream_processing_allowed": result.status == "passed",
        "started_at_utc": result.started_at_utc,
        "completed_at_utc": result.completed_at_utc,
        "execution": {
            "duration_seconds": result.duration_seconds,
            "csv_inspection_seconds": result.inspection_seconds,
            "duplicate_check_seconds": result.duplicate_check_seconds,
            "max_workers": config.max_workers,
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                package: _package_version(package)
                for package in ("numpy", "pandas", "pyarrow", "PyYAML", "xjtu-sy-tcc")
            },
        },
        "paths": {
            "project_root": config.project_root.as_posix(),
            "dataset_root": _portable_path(config.dataset_root, config.project_root),
            "output_directory": _portable_path(config.output_directory, config.project_root),
            "artifacts": {
                name: _portable_path(path, config.project_root) for name, path in paths.items()
            },
        },
        "expected": {
            "conditions": expected_conditions,
            "sampling_frequency_hz": config.sampling_frequency_hz,
            "sample_count": config.expected_sample_count,
            "channel_count": config.expected_channel_count,
            "channel_names": list(config.expected_channel_names),
            "acquisition_interval_minutes": config.acquisition_interval_minutes,
            "filename_pattern": config.filename_pattern,
        },
        "provenance": {
            "nominal_values_source": config.nominal_values_source,
            "csv_observability_limit": (
                "The CSV files contain amplitudes only; sampling frequency, acquisition "
                "interval, vibration units, rotational speed, and radial load are nominal "
                "documented values rather than quantities inferred from CSV fields."
            ),
        },
        "observed": _json_value(result.statistics),
        "metadata": {
            "row_count": len(result.metadata),
            "columns": list(result.metadata.columns),
            "dtypes": {name: str(dtype) for name, dtype in result.metadata.dtypes.items()},
        },
        "bearing_summary": _dataframe_records(result.bearing_summary),
        "validation": {
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "issues": [_issue_payload(issue, config.project_root) for issue in result.issues],
        },
    }


def render_markdown_report(
    config: DataConfig,
    result: AuditResult,
    output_paths: Mapping[str, Path] | None = None,
) -> str:
    """Render a concise audit report suitable for thesis traceability."""

    paths = output_paths or {}
    expected_acquisitions = result.statistics["expected_acquisition_count"]
    acquisition_line = f"- Acquisitions: {result.statistics['acquisition_count']}"
    if expected_acquisitions is not None:
        acquisition_line += f" (expected {expected_acquisitions})"
    lines = [
        "# XJTU-SY dataset audit",
        "",
        f"- Status: **{result.status.upper()}**",
        f"- Downstream processing allowed: **{'yes' if result.status == 'passed' else 'no'}**",
        f"- Dataset root: `{_portable_path(config.dataset_root, config.project_root)}`",
        f"- Completed at: `{result.completed_at_utc}`",
        f"- Duration: {result.duration_seconds:.3f} s",
        f"- Critical errors: {result.error_count}",
        f"- Warnings: {result.warning_count}",
        "",
        "## Observed dataset",
        "",
        f"- Conditions: {result.statistics['observed_condition_count']} "
        f"(expected {result.statistics['expected_condition_count']})",
        f"- Bearings: {result.statistics['observed_bearing_count']} "
        f"(expected {result.statistics['expected_bearing_count']})",
        acquisition_line,
        f"- Physical data rows observed: {result.statistics['total_physical_data_rows']}",
        f"- Raw acquisition bytes: {result.statistics['total_file_size_bytes']}",
        f"- Files with CSV inspection errors: {result.statistics['files_with_inspection_errors']}",
        f"- Exact duplicate-content groups: {result.statistics['duplicate_content_group_count']}",
        "",
        f"Sampling frequency ({config.sampling_frequency_hz:g} Hz), the configured "
        f"{config.acquisition_interval_minutes:g}-minute acquisition interval, vibration "
        "units, rotation, and load cannot be inferred from the amplitude-only CSV schema. "
        f"Their configured provenance is `{config.nominal_values_source}`.",
        "",
        "## Bearing summary",
        "",
    ]
    lines.extend(_markdown_table(result.bearing_summary))
    lines.extend(["", "## Validation issues", ""])
    if not result.issues:
        lines.append("No validation issues were detected.")
    else:
        lines.extend(
            [
                "| Severity | Code | Path | Message |",
                "|---|---|---|---|",
            ]
        )
        for issue in result.issues:
            issue_path = (
                _portable_path(issue.path, config.project_root) if issue.path is not None else ""
            )
            lines.append(
                "| "
                + " | ".join(
                    _escape_markdown(value)
                    for value in (
                        issue.severity.value,
                        issue.code,
                        issue_path,
                        issue.message,
                    )
                )
                + " |"
            )
    lines.extend(["", "## Output artifacts", ""])
    if paths:
        for name, path in sorted(paths.items()):
            lines.append(f"- `{name}`: `{_portable_path(path, config.project_root)}`")
    else:
        lines.append("Artifact paths were not supplied to the report renderer.")
    lines.extend(
        [
            "",
            "Validated Parquet tables are emitted only when the audit has no critical errors. "
            "Feature extraction and model training must remain blocked otherwise.",
            "",
        ]
    )
    return "\n".join(lines)


def _issue_payload(issue: ValidationIssue, project_root: Path) -> dict[str, Any]:
    return {
        "severity": issue.severity.value,
        "code": issue.code,
        "message": issue.message,
        "path": (_portable_path(issue.path, project_root) if issue.path is not None else None),
        "details": _json_value(issue.details),
    }


def _dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in record.items()}
        for record in dataframe.to_dict(orient="records")
    ]


def _json_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _json_value(value.item())  # type: ignore[union-attr]
    return str(value)


def _markdown_table(dataframe: pd.DataFrame) -> list[str]:
    if dataframe.empty:
        return ["No bearing rows were generated."]
    headers = [str(column) for column in dataframe.columns]
    rows = ["| " + " | ".join(_escape_markdown(name) for name in headers) + " |"]
    rows.append("|" + "|".join("---" for _ in headers) + "|")
    for record in dataframe.itertuples(index=False, name=None):
        rows.append(
            "| " + " | ".join(_escape_markdown(_format_cell(value)) for value in record) + " |"
        )
    return rows


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _atomic_text(path: Path, contents: str) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary = Path(file_handle.name)
            file_handle.write(contents)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary = Path(file_handle.name)
            dataframe.to_parquet(file_handle, index=False, engine="pyarrow")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_csv(dataframe: pd.DataFrame, path: Path) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary = Path(file_handle.name)
            dataframe.to_csv(file_handle, index=False)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _remove_stale_validated_tables(output_directory: Path) -> None:
    """Ensure a failed rerun cannot expose tables from an earlier successful audit."""

    for artifact_name in _VALIDATED_TABLE_NAMES:
        (output_directory / artifact_name).unlink(missing_ok=True)


def _validate_output_directory(config: DataConfig) -> None:
    resolved_output = config.output_directory.resolve()
    resolved_dataset = config.dataset_root.resolve(strict=True)
    if (
        resolved_output == resolved_dataset
        or resolved_output.is_relative_to(resolved_dataset)
        or resolved_dataset.is_relative_to(resolved_output)
    ):
        raise ValueError("Resolved audit output directory overlaps the immutable dataset root")
