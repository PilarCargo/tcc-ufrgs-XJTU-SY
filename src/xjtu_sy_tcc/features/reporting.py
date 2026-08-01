"""Atomic Phase-2 tables and structured validation report."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from xjtu_sy_tcc.config.features import FeatureConfig
from xjtu_sy_tcc.features.processing import FeatureBuildResult


@contextmanager
def feature_output_lock(config: FeatureConfig) -> Iterator[None]:
    config.output_directory.mkdir(parents=True, exist_ok=True)
    lock = config.output_directory / ".features.lock"
    try:
        descriptor = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Another feature build is using {config.output_directory}") from exc
    try:
        descriptor.write(str(__import__("os").getpid()))
        descriptor.close()
        yield
    finally:
        lock.unlink(missing_ok=True)


def write_feature_outputs(config: FeatureConfig, result: FeatureBuildResult) -> dict[str, Path]:
    """Publish validated outputs; never publish a failed feature table."""
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "features.parquet"
    report_path = output / "feature_validation.json"
    summary_path = output / "feature_summary.csv"
    checkpoint = output / ".features.checkpoint.parquet"
    feature_columns = list(result.validation.feature_columns)
    summary = result.table[feature_columns].describe().T.reset_index(names="feature")
    payload = {
        "validation_schema_version": 1,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": result.validation.status,
        "expected_rows": result.validation.expected_rows,
        "observed_rows": result.validation.observed_rows,
        "feature_count": len(feature_columns),
        "target_columns": ["rul_minutes"],
        "feature_columns": feature_columns,
        "duration_seconds": result.duration_seconds,
        "peak_rss_bytes": result.peak_rss_bytes,
        "resumed_rows": result.resumed_rows,
        "issues": [
            item.__dict__
            if hasattr(item, "__dict__")
            else {"severity": item.severity, "code": item.code, "message": item.message}
            for item in result.validation.issues
        ],
        "parameters": {
            "welch_nperseg": config.welch_nperseg,
            "welch_overlap": config.welch_overlap,
            "frequency_bands_hz": [
                {"name": band.name, "lower_hz": band.lower_hz, "upper_hz": band.upper_hz}
                for band in config.frequency_bands_hz
            ],
        },
    }
    _atomic_text(report_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if result.validation.status != "passed":
        table_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        return {"validation_report": report_path}
    _atomic_frame(result.table, table_path, "parquet")
    _atomic_frame(summary, summary_path, "csv")
    checkpoint.unlink(missing_ok=True)
    return {"features": table_path, "validation_report": report_path, "summary": summary_path}


def _atomic_text(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_frame(frame: pd.DataFrame, path: Path, kind: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        if kind == "parquet":
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
