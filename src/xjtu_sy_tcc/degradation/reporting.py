"""Atomic Phase 3 artifact publication and exclusive output lock."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


@contextmanager
def phase3_output_lock(output_root: Path) -> Iterator[None]:
    lock_directory = output_root / "degradation"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock = lock_directory / ".phase3.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("Another Phase 3 process is already running") from exc
    try:
        handle.write(str(os.getpid()))
        handle.close()
        yield
    finally:
        lock.unlink(missing_ok=True)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=_json) + "\n")


def atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        if path.suffix == ".parquet":
            table.to_parquet(temporary, index=False)
        elif path.suffix == ".csv":
            table.to_csv(temporary, index=False)
        else:
            raise ValueError(f"Unsupported table extension: {path.suffix}")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _json(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()  # type: ignore[union-attr]
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot serialize {type(value).__name__}")
