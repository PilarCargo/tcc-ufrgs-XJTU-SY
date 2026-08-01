"""Strict configuration for Phase-2 feature extraction."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from xjtu_sy_tcc.exceptions import ConfigurationError

_TOP_KEYS = {
    "project_root",
    "metadata_path",
    "output_directory",
    "figure_directory",
    "welch_nperseg",
    "welch_overlap",
    "frequency_bands_hz",
    "plot_dpi",
    "generate_pdf",
    "progress_interval",
}
_BAND_KEYS = {"name", "lower_hz", "upper_hz"}
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class FrequencyBand:
    name: str
    lower_hz: float
    upper_hz: float


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    config_path: Path
    project_root: Path
    metadata_path: Path
    output_directory: Path
    figure_directory: Path
    welch_nperseg: int
    welch_overlap: int
    frequency_bands_hz: tuple[FrequencyBand, ...]
    plot_dpi: int
    generate_pdf: bool
    progress_interval: int


def load_feature_config(path: Path) -> FeatureConfig:
    """Load and validate a Phase-2 YAML configuration."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    config_path = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load feature configuration {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping) or set(raw) != _TOP_KEYS:
        raise ConfigurationError(
            f"Feature configuration must contain exactly these keys: {sorted(_TOP_KEYS)}"
        )
    root = _resolved(config_path.parent, raw["project_root"], "project_root")
    metadata = _resolved(root, raw["metadata_path"], "metadata_path")
    output = _resolved(root, raw["output_directory"], "output_directory")
    figures = _resolved(root, raw["figure_directory"], "figure_directory")
    nperseg = _positive_int(raw["welch_nperseg"], "welch_nperseg")
    overlap = _nonnegative_int(raw["welch_overlap"], "welch_overlap")
    if overlap >= nperseg:
        raise ConfigurationError("welch_overlap must be smaller than welch_nperseg")
    bands_raw = raw["frequency_bands_hz"]
    if not isinstance(bands_raw, list) or not bands_raw:
        raise ConfigurationError("frequency_bands_hz must be a non-empty list")
    bands: list[FrequencyBand] = []
    for index, item in enumerate(bands_raw):
        if not isinstance(item, Mapping) or set(item) != _BAND_KEYS:
            raise ConfigurationError(f"frequency_bands_hz[{index}] has invalid keys")
        name = item["name"]
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ConfigurationError(f"frequency_bands_hz[{index}].name is invalid")
        lower = _finite_number(item["lower_hz"], f"frequency_bands_hz[{index}].lower_hz")
        upper = _finite_number(item["upper_hz"], f"frequency_bands_hz[{index}].upper_hz")
        if lower < 0 or upper <= lower:
            raise ConfigurationError("Frequency bands require 0 <= lower_hz < upper_hz")
        bands.append(FrequencyBand(name, lower, upper))
    if len({band.name for band in bands}) != len(bands):
        raise ConfigurationError("Frequency-band names must be unique")
    for candidate in (metadata, output, figures):
        if candidate == root or root not in candidate.parents:
            raise ConfigurationError(f"Configured path must be inside project_root: {candidate}")
    return FeatureConfig(
        config_path,
        root,
        metadata,
        output,
        figures,
        nperseg,
        overlap,
        tuple(bands),
        _positive_int(raw["plot_dpi"], "plot_dpi"),
        _boolean(raw["generate_pdf"], "generate_pdf"),
        _positive_int(raw["progress_interval"], "progress_interval"),
    )


def _resolved(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigurationError(f"{name} must be a non-empty path string")
    return (base / value).resolve()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be finite")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value
