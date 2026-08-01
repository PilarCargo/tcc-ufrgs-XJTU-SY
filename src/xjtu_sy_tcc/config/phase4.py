"""Strict configuration for leakage-free classical RUL regression."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from xjtu_sy_tcc.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class RandomForestCandidate:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: str | float


@dataclass(frozen=True, slots=True)
class HistGradientBoostingCandidate:
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


@dataclass(frozen=True, slots=True)
class Phase4Config:
    config_path: Path
    project_root: Path
    feature_table: Path
    split_manifest: Path
    selected_feature_manifest: Path
    test_onset_table: Path
    output_directory: Path
    expected_feature_rows: int
    expected_feature_count: int
    expected_split_sha256: str
    expected_split_configuration_hash: str
    target_column: str
    sample_weight_policy: str
    random_seed: int
    primary_metric: str
    prediction_policy: str
    heavy_tail_feature_tokens: tuple[str, ...]
    ridge_alphas: tuple[float, ...]
    random_forest_candidates: tuple[RandomForestCandidate, ...]
    hist_gradient_boosting_candidates: tuple[HistGradientBoostingCandidate, ...]
    permutation_repeats: int
    life_stage_thresholds: tuple[float, float]
    plot_dpi: int
    generate_pdf: bool

    @property
    def configuration_hash(self) -> str:
        payload = asdict(self)
        payload.pop("config_path")
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def load_phase4_config(path: Path) -> Phase4Config:
    """Load Phase 4 YAML and reject unknown keys, unsafe outputs, and invalid candidates."""
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load Phase 4 configuration: {exc}") from exc
    keys = {
        "project_root",
        "feature_table",
        "split_manifest",
        "selected_feature_manifest",
        "test_onset_table",
        "output_directory",
        "expected_feature_rows",
        "expected_feature_count",
        "expected_split_sha256",
        "expected_split_configuration_hash",
        "target_column",
        "sample_weight_policy",
        "random_seed",
        "primary_metric",
        "prediction_policy",
        "heavy_tail_feature_tokens",
        "ridge_alphas",
        "random_forest_candidates",
        "hist_gradient_boosting_candidates",
        "permutation_repeats",
        "life_stage_thresholds",
        "plot_dpi",
        "generate_pdf",
    }
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise ConfigurationError(f"Phase 4 configuration must contain exactly: {sorted(keys)}")
    root = _path(config_path.parent, raw["project_root"], "project_root")
    inputs = {
        name: _path(root, raw[name], name)
        for name in (
            "feature_table",
            "split_manifest",
            "selected_feature_manifest",
            "test_onset_table",
        )
    }
    output = _path(root, raw["output_directory"], "output_directory")
    protected = [
        root / "outputs" / name
        for name in ("audits", "features", "splits", "prognostics", "degradation")
    ]
    if any(output == item or item in output.parents for item in protected):
        raise ConfigurationError("Phase 4 output must not overwrite an earlier phase")
    sha = _string(raw["expected_split_sha256"], "expected_split_sha256")
    logical_hash = _string(
        raw["expected_split_configuration_hash"], "expected_split_configuration_hash"
    )
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in (sha, logical_hash)
    ):
        raise ConfigurationError("Expected split hashes must be lowercase SHA-256 strings")
    if raw["target_column"] != "rul_minutes":
        raise ConfigurationError("The Phase 4 target must be rul_minutes")
    if raw["sample_weight_policy"] != "inverse_bearing_frequency":
        raise ConfigurationError("Unknown sample-weight policy")
    if raw["primary_metric"] != "validation_macro_mae":
        raise ConfigurationError("Primary selection metric must be validation_macro_mae")
    if raw["prediction_policy"] != "non_negative_primary_preserve_raw":
        raise ConfigurationError("Unknown prediction policy")
    thresholds = _numeric_list(raw["life_stage_thresholds"], "life_stage_thresholds")
    if len(thresholds) != 2 or not 0 < thresholds[0] < thresholds[1] < 1:
        raise ConfigurationError("life_stage_thresholds must contain two increasing fractions")
    ridge = _numeric_list(raw["ridge_alphas"], "ridge_alphas", positive=True)
    forests = tuple(
        _forest(item, index)
        for index, item in enumerate(
            _list(raw["random_forest_candidates"], "random_forest_candidates")
        )
    )
    gradients = tuple(
        _gradient(item, index)
        for index, item in enumerate(
            _list(raw["hist_gradient_boosting_candidates"], "hist_gradient_boosting_candidates")
        )
    )
    seed = raw["random_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigurationError("random_seed must be a non-negative integer")
    return Phase4Config(
        config_path=config_path,
        project_root=root,
        output_directory=output,
        expected_feature_rows=_positive_int(raw["expected_feature_rows"], "expected_feature_rows"),
        expected_feature_count=_positive_int(
            raw["expected_feature_count"], "expected_feature_count"
        ),
        expected_split_sha256=sha,
        expected_split_configuration_hash=logical_hash,
        target_column="rul_minutes",
        sample_weight_policy="inverse_bearing_frequency",
        random_seed=seed,
        primary_metric="validation_macro_mae",
        prediction_policy="non_negative_primary_preserve_raw",
        heavy_tail_feature_tokens=_strings(
            raw["heavy_tail_feature_tokens"], "heavy_tail_feature_tokens"
        ),
        ridge_alphas=ridge,
        random_forest_candidates=forests,
        hist_gradient_boosting_candidates=gradients,
        permutation_repeats=_positive_int(raw["permutation_repeats"], "permutation_repeats"),
        life_stage_thresholds=(thresholds[0], thresholds[1]),
        plot_dpi=_positive_int(raw["plot_dpi"], "plot_dpi"),
        generate_pdf=_boolean(raw["generate_pdf"], "generate_pdf"),
        **inputs,
    )


def _forest(value: Any, index: int) -> RandomForestCandidate:
    keys = {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ConfigurationError(f"random_forest_candidates[{index}] has invalid keys")
    depth = value["max_depth"]
    if depth is not None:
        depth = _positive_int(depth, f"random_forest_candidates[{index}].max_depth")
    maximum = value["max_features"]
    if maximum != "sqrt":
        maximum = _fraction_or_one(maximum, f"random_forest_candidates[{index}].max_features")
    return RandomForestCandidate(
        _positive_int(value["n_estimators"], "n_estimators"),
        depth,
        _positive_int(value["min_samples_leaf"], "min_samples_leaf"),
        maximum,
    )


def _gradient(value: Any, index: int) -> HistGradientBoostingCandidate:
    keys = {"learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ConfigurationError(f"hist_gradient_boosting_candidates[{index}] has invalid keys")
    return HistGradientBoostingCandidate(
        _positive_float(value["learning_rate"], "learning_rate"),
        _positive_int(value["max_iter"], "max_iter"),
        _positive_int(value["max_leaf_nodes"], "max_leaf_nodes"),
        _positive_int(value["min_samples_leaf"], "min_samples_leaf"),
        _nonnegative_float(value["l2_regularization"], "l2_regularization"),
    )


def _path(base: Path, value: Any, name: str) -> Path:
    return (base / _string(value, name)).resolve()


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty list")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    values = tuple(_string(item, name) for item in _list(value, name))
    if len(set(values)) != len(values):
        raise ConfigurationError(f"{name} contains duplicates")
    return values


def _numeric_list(value: Any, name: str, positive: bool = False) -> tuple[float, ...]:
    values = tuple(
        (_positive_float if positive else _finite_float)(item, name) for item in _list(value, name)
    )
    if len(set(values)) != len(values):
        raise ConfigurationError(f"{name} contains duplicates")
    return values


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ConfigurationError(f"{name} must be finite numeric")
    return float(value)


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0:
        raise ConfigurationError(f"{name} must be non-negative")
    return result


def _fraction_or_one(value: Any, name: str) -> float:
    result = _positive_float(value, name)
    if result > 1:
        raise ConfigurationError(f"{name} must not exceed one")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be boolean")
    return value
