"""Strict loading and validation for the XJTU-SY data configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from xjtu_sy_tcc.exceptions import ConfigurationError

_TOP_LEVEL_KEYS = {
    "project_root",
    "dataset_root",
    "output_directory",
    "nominal_values_source",
    "acquisition_interval_minutes",
    "sampling_frequency_hz",
    "expected_sample_count",
    "expected_channel_names",
    "filename_pattern",
    "detect_duplicate_content",
    "max_workers",
    "conditions",
}
_CONDITION_KEYS = {
    "condition_id",
    "directory_name",
    "bearing_ids",
    "expected_acquisition_counts",
    "rotation_hz",
    "rotation_rpm",
    "radial_load_kn",
}
_CONDITION_DIRECTORY_PATTERN = re.compile(
    r"^(?P<rotation_hz>[0-9]+(?:\.[0-9]+)?)Hz"
    r"(?P<radial_load_kn>[0-9]+(?:\.[0-9]+)?)kN$"
)
_MAX_AUDIT_WORKERS = 8
_MAX_INT64 = 2**63 - 1


@dataclass(frozen=True, slots=True)
class ConditionConfig:
    """Nominal operating condition and its bearing identifiers."""

    condition_id: int
    directory_name: str
    bearing_ids: tuple[str, ...]
    rotation_hz: float
    rotation_rpm: float
    radial_load_kn: float
    expected_acquisition_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach expected counts and expose them through a read-only mapping."""

        if (
            isinstance(self.condition_id, bool)
            or not isinstance(self.condition_id, int)
            or not 1 <= self.condition_id <= _MAX_INT64
        ):
            raise ConfigurationError(f"condition_id must be an integer between 1 and {_MAX_INT64}")
        for bearing_id, count in self.expected_acquisition_counts.items():
            if (
                bearing_id not in self.bearing_ids
                or isinstance(count, bool)
                or not isinstance(count, int)
                or not 1 <= count <= _MAX_INT64
            ):
                raise ConfigurationError(
                    "expected_acquisition_counts must contain positive int64 counts "
                    "for configured bearing IDs"
                )
        object.__setattr__(
            self,
            "expected_acquisition_counts",
            MappingProxyType(dict(self.expected_acquisition_counts)),
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Validated paths and nominal properties used by the data pipeline."""

    config_path: Path
    project_root: Path
    dataset_root: Path
    output_directory: Path
    nominal_values_source: str
    acquisition_interval_minutes: float
    sampling_frequency_hz: float
    expected_sample_count: int
    expected_channel_names: tuple[str, ...]
    filename_pattern: str
    detect_duplicate_content: bool
    max_workers: int
    conditions: tuple[ConditionConfig, ...]

    def __post_init__(self) -> None:
        """Enforce the memory-bound worker limit for direct construction too."""

        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or not 1 <= self.max_workers <= _MAX_AUDIT_WORKERS
        ):
            raise ConfigurationError(
                f"max_workers must be an integer between 1 and {_MAX_AUDIT_WORKERS}"
            )

    @property
    def expected_channel_count(self) -> int:
        """Return the number of configured vibration channels."""

        return len(self.expected_channel_names)


def load_data_config(config_path: Path) -> DataConfig:
    """Load a YAML data configuration and reject invalid or ambiguous values.

    The project root is resolved relative to the configuration file. Dataset and
    output paths are then resolved relative to that project root.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A fully validated immutable data configuration.

    Raises:
        ConfigurationError: If the file cannot be read or contains invalid data.
        TypeError: If ``config_path`` is not a :class:`pathlib.Path`.
    """

    if not isinstance(config_path, Path):
        raise TypeError("config_path must be a pathlib.Path")

    resolved_config_path = config_path.expanduser().resolve()
    raw_config = _load_yaml_mapping(resolved_config_path)
    _validate_exact_keys(raw_config, _TOP_LEVEL_KEYS, "data configuration")

    project_root = _resolve_path(
        resolved_config_path.parent,
        _require_non_empty_string(raw_config["project_root"], "project_root"),
    )
    dataset_root = _resolve_path(
        project_root,
        _require_non_empty_string(raw_config["dataset_root"], "dataset_root"),
    )
    output_directory = _resolve_path(
        project_root,
        _require_non_empty_string(raw_config["output_directory"], "output_directory"),
    )
    _validate_non_overlapping_paths(dataset_root, output_directory)

    filename_pattern = _require_non_empty_string(raw_config["filename_pattern"], "filename_pattern")
    _validate_regular_expression(filename_pattern)

    conditions = _parse_conditions(raw_config["conditions"])

    return DataConfig(
        config_path=resolved_config_path,
        project_root=project_root,
        dataset_root=dataset_root,
        output_directory=output_directory,
        nominal_values_source=_require_non_empty_string(
            raw_config["nominal_values_source"], "nominal_values_source"
        ),
        acquisition_interval_minutes=_require_positive_float(
            raw_config["acquisition_interval_minutes"], "acquisition_interval_minutes"
        ),
        sampling_frequency_hz=_require_positive_float(
            raw_config["sampling_frequency_hz"], "sampling_frequency_hz"
        ),
        expected_sample_count=_require_bounded_positive_int(
            raw_config["expected_sample_count"], "expected_sample_count", _MAX_INT64
        ),
        expected_channel_names=_require_unique_string_tuple(
            raw_config["expected_channel_names"], "expected_channel_names"
        ),
        filename_pattern=filename_pattern,
        detect_duplicate_content=_require_bool(
            raw_config["detect_duplicate_content"], "detect_duplicate_content"
        ),
        max_workers=_require_bounded_positive_int(
            raw_config["max_workers"], "max_workers", _MAX_AUDIT_WORKERS
        ),
        conditions=conditions,
    )


def _load_yaml_mapping(config_path: Path) -> Mapping[str, Any]:
    try:
        with config_path.open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
    except FileNotFoundError as error:
        raise ConfigurationError(f"Data configuration does not exist: {config_path}") from error
    except OSError as error:
        raise ConfigurationError(
            f"Cannot read data configuration {config_path}: {error}"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid YAML in data configuration {config_path}: {error}"
        ) from error

    if not isinstance(loaded, Mapping):
        raise ConfigurationError("Data configuration must be a YAML mapping")
    if any(not isinstance(key, str) for key in loaded):
        raise ConfigurationError("Data configuration keys must be strings")
    return loaded


def _validate_exact_keys(values: Mapping[str, Any], expected_keys: set[str], context: str) -> None:
    actual_keys = set(values)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    if not missing_keys and not unexpected_keys:
        return

    details = []
    if missing_keys:
        details.append(f"missing keys: {', '.join(missing_keys)}")
    if unexpected_keys:
        details.append(f"unexpected keys: {', '.join(unexpected_keys)}")
    raise ConfigurationError(f"Invalid {context} ({'; '.join(details)})")


def _require_non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context} must be a non-empty string")
    if "\x00" in value:
        raise ConfigurationError(f"{context} must not contain null characters")
    return value


def _require_positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{context} must be an integer")
    if value <= 0:
        raise ConfigurationError(f"{context} must be greater than zero")
    return value


def _require_bounded_positive_int(value: Any, context: str, maximum: int) -> int:
    parsed = _require_positive_int(value, context)
    if parsed > maximum:
        raise ConfigurationError(f"{context} must not exceed {maximum}")
    return parsed


def _require_positive_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{context} must be numeric")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ConfigurationError(f"{context} must be finite and greater than zero")
    return numeric_value


def _require_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{context} must be a boolean")
    return value


def _require_unique_string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{context} must be a non-empty YAML list")

    parsed_values = tuple(
        _require_non_empty_string(item, f"{context}[{index}]") for index, item in enumerate(value)
    )
    duplicates = sorted({item for item in parsed_values if parsed_values.count(item) > 1})
    if duplicates:
        raise ConfigurationError(f"{context} contains duplicates: {', '.join(duplicates)}")
    return parsed_values


def _resolve_path(base_directory: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _validate_non_overlapping_paths(dataset_root: Path, output_directory: Path) -> None:
    if (
        dataset_root == output_directory
        or dataset_root.is_relative_to(output_directory)
        or output_directory.is_relative_to(dataset_root)
    ):
        raise ConfigurationError(
            "dataset_root and output_directory must not be equal, nested, or otherwise overlap"
        )


def _validate_regular_expression(filename_pattern: str) -> None:
    try:
        re.compile(filename_pattern)
    except re.error as error:
        raise ConfigurationError(
            f"filename_pattern is not a valid regular expression: {error}"
        ) from error


def _parse_conditions(value: Any) -> tuple[ConditionConfig, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError("conditions must be a non-empty YAML list")

    conditions = tuple(_parse_condition(item, index) for index, item in enumerate(value))
    _validate_unique_condition_values(conditions)
    return conditions


def _parse_condition(value: Any, index: int) -> ConditionConfig:
    context = f"conditions[{index}]"
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{context} must be a YAML mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigurationError(f"{context} keys must be strings")
    _validate_exact_keys(value, _CONDITION_KEYS, context)

    bearing_ids = _require_unique_string_tuple(value["bearing_ids"], f"{context}.bearing_ids")
    _validate_path_components(bearing_ids, f"{context}.bearing_ids")
    condition = ConditionConfig(
        condition_id=_require_bounded_positive_int(
            value["condition_id"], f"{context}.condition_id", _MAX_INT64
        ),
        directory_name=_require_non_empty_string(
            value["directory_name"], f"{context}.directory_name"
        ),
        bearing_ids=bearing_ids,
        rotation_hz=_require_positive_float(value["rotation_hz"], f"{context}.rotation_hz"),
        rotation_rpm=_require_positive_float(value["rotation_rpm"], f"{context}.rotation_rpm"),
        radial_load_kn=_require_positive_float(
            value["radial_load_kn"], f"{context}.radial_load_kn"
        ),
        expected_acquisition_counts=_require_acquisition_counts(
            value["expected_acquisition_counts"],
            bearing_ids,
            f"{context}.expected_acquisition_counts",
        ),
    )
    _validate_condition_relationships(condition, context)
    return condition


def _require_acquisition_counts(
    value: Any,
    bearing_ids: tuple[str, ...],
    context: str,
) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{context} must be a YAML mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigurationError(f"{context} keys must be bearing identifier strings")

    expected_keys = set(bearing_ids)
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        details: list[str] = []
        if missing:
            details.append(f"missing bearing IDs: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected bearing IDs: {', '.join(unexpected)}")
        raise ConfigurationError(f"Invalid {context} ({'; '.join(details)})")

    return {
        bearing_id: _require_bounded_positive_int(
            value[bearing_id], f"{context}.{bearing_id}", _MAX_INT64
        )
        for bearing_id in bearing_ids
    }


def _validate_path_components(values: tuple[str, ...], context: str) -> None:
    for index, value in enumerate(values):
        if value in {".", ".."} or Path(value).is_absolute() or "/" in value or "\\" in value:
            raise ConfigurationError(f"{context}[{index}] must be a single relative directory name")


def _validate_condition_relationships(condition: ConditionConfig, context: str) -> None:
    expected_rpm = condition.rotation_hz * 60.0
    if not math.isclose(condition.rotation_rpm, expected_rpm, rel_tol=1e-9, abs_tol=1e-9):
        raise ConfigurationError(
            f"{context}.rotation_rpm must equal rotation_hz * 60 ({expected_rpm:g} expected)"
        )

    directory_match = _CONDITION_DIRECTORY_PATTERN.fullmatch(condition.directory_name)
    if directory_match is None:
        raise ConfigurationError(f"{context}.directory_name must use the '<Hz>Hz<kN>kN' format")

    directory_hz = float(directory_match.group("rotation_hz"))
    directory_load_kn = float(directory_match.group("radial_load_kn"))
    if not math.isclose(directory_hz, condition.rotation_hz, rel_tol=1e-9, abs_tol=1e-9):
        raise ConfigurationError(
            f"{context}.directory_name rotation label does not match rotation_hz"
        )
    if not math.isclose(directory_load_kn, condition.radial_load_kn, rel_tol=1e-9, abs_tol=1e-9):
        raise ConfigurationError(
            f"{context}.directory_name load label does not match radial_load_kn"
        )


def _validate_unique_condition_values(conditions: tuple[ConditionConfig, ...]) -> None:
    _validate_unique_values((condition.condition_id for condition in conditions), "condition IDs")
    _validate_unique_values(
        (condition.directory_name for condition in conditions), "condition directory names"
    )
    _validate_unique_values(
        (bearing_id for condition in conditions for bearing_id in condition.bearing_ids),
        "bearing IDs across conditions",
    )


def _validate_unique_values(values: Iterable[object], context: str) -> None:
    seen: set[object] = set()
    duplicates: set[object] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        duplicate_list = ", ".join(str(value) for value in sorted(duplicates, key=str))
        raise ConfigurationError(f"Duplicate {context}: {duplicate_list}")
