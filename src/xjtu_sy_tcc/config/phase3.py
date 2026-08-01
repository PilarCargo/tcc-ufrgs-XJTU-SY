"""Strict declarative configuration for leakage-free Phase 3 analysis."""

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
class TransformationConfig:
    mode: str
    feature_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PeltConfig:
    primary_cost_model: str
    primary_penalty: float
    primary_minimum_segment_size: int
    primary_jump: int
    primary_persistence_window: int
    minimum_effect_size: float
    cost_models: tuple[str, ...]
    penalties: tuple[float, ...]
    smoothing_windows: tuple[int, ...]
    minimum_segment_sizes: tuple[int, ...]
    persistence_windows: tuple[int, ...]
    agreement_minutes: float
    agreement_life_fraction: float


@dataclass(frozen=True, slots=True)
class Phase3Config:
    config_path: Path
    project_root: Path
    feature_table: Path
    output_root: Path
    figure_directory: Path
    expected_rows: int
    expected_feature_count: int
    interpolation_grid_size: int
    initial_window_fraction: float
    final_window_fraction: float
    selected_feature_count: int
    metric_weights: Mapping[str, float]
    transformation: TransformationConfig
    balance_points_per_bearing: int
    pca_components: int
    baseline_fraction: float
    minimum_baseline_acquisitions: int
    smoothing_method: str
    primary_smoothing_window: int
    pelt: PeltConfig
    plot_dpi: int
    generate_pdf: bool

    @property
    def configuration_hash(self) -> str:
        payload = asdict(self)
        payload.pop("config_path")
        encoded = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_phase3_config(path: Path) -> Phase3Config:
    """Load Phase 3 YAML and reject unsafe paths or incoherent parameters."""
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load Phase 3 configuration: {exc}") from exc
    expected = {
        "project_root",
        "feature_table",
        "output_root",
        "figure_directory",
        "expected_rows",
        "expected_feature_count",
        "interpolation_grid_size",
        "initial_window_fraction",
        "final_window_fraction",
        "selected_feature_count",
        "metric_weights",
        "transformation",
        "balance_points_per_bearing",
        "pca_components",
        "baseline_fraction",
        "minimum_baseline_acquisitions",
        "smoothing_method",
        "primary_smoothing_window",
        "pelt",
        "plot_dpi",
        "generate_pdf",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ConfigurationError(f"Phase 3 configuration must contain exactly: {sorted(expected)}")
    root = _path(config_path.parent, raw["project_root"], "project_root")
    features = _path(root, raw["feature_table"], "feature_table")
    output = _path(root, raw["output_root"], "output_root")
    figures = _path(root, raw["figure_directory"], "figure_directory")
    protected = {root / "outputs" / "audits", root / "outputs" / "features"}
    if output in protected or any(item in output.parents for item in protected):
        raise ConfigurationError(
            "Phase 3 output_root must not overwrite Phase 1 or Phase 2 outputs"
        )
    weights_raw = raw["metric_weights"]
    metric_names = {"monotonicity", "trendability", "prognosability", "spearman_abs", "stability"}
    if not isinstance(weights_raw, Mapping) or set(weights_raw) != metric_names:
        raise ConfigurationError(f"metric_weights must contain exactly {sorted(metric_names)}")
    weights = {
        name: _nonnegative_float(value, f"metric_weights.{name}")
        for name, value in weights_raw.items()
    }
    if sum(weights.values()) <= 0:
        raise ConfigurationError("At least one metric weight must be positive")
    weights = {name: value / sum(weights.values()) for name, value in weights.items()}
    transform_raw = raw["transformation"]
    if not isinstance(transform_raw, Mapping) or set(transform_raw) != {"mode", "feature_tokens"}:
        raise ConfigurationError("transformation has invalid keys")
    mode = transform_raw["mode"]
    if mode not in {"none", "auto_log1p_nonnegative", "yeo_johnson"}:
        raise ConfigurationError("Unknown transformation mode")
    tokens = _strings(
        transform_raw["feature_tokens"], "transformation.feature_tokens", allow_empty=mode == "none"
    )
    pelt = _pelt(raw["pelt"])
    smoothing_method = raw["smoothing_method"]
    if smoothing_method not in {"median", "mean"}:
        raise ConfigurationError("smoothing_method must be median or mean")
    config = Phase3Config(
        config_path=config_path,
        project_root=root,
        feature_table=features,
        output_root=output,
        figure_directory=figures,
        expected_rows=_positive_int(raw["expected_rows"], "expected_rows"),
        expected_feature_count=_positive_int(
            raw["expected_feature_count"], "expected_feature_count"
        ),
        interpolation_grid_size=_minimum_int(
            raw["interpolation_grid_size"], 3, "interpolation_grid_size"
        ),
        initial_window_fraction=_fraction(
            raw["initial_window_fraction"], "initial_window_fraction"
        ),
        final_window_fraction=_fraction(raw["final_window_fraction"], "final_window_fraction"),
        selected_feature_count=_positive_int(
            raw["selected_feature_count"], "selected_feature_count"
        ),
        metric_weights=weights,
        transformation=TransformationConfig(mode, tokens),
        balance_points_per_bearing=_minimum_int(
            raw["balance_points_per_bearing"], 3, "balance_points_per_bearing"
        ),
        pca_components=_positive_int(raw["pca_components"], "pca_components"),
        baseline_fraction=_fraction(raw["baseline_fraction"], "baseline_fraction"),
        minimum_baseline_acquisitions=_minimum_int(
            raw["minimum_baseline_acquisitions"], 2, "minimum_baseline_acquisitions"
        ),
        smoothing_method=smoothing_method,
        primary_smoothing_window=_odd_int(
            raw["primary_smoothing_window"], "primary_smoothing_window"
        ),
        pelt=pelt,
        plot_dpi=_positive_int(raw["plot_dpi"], "plot_dpi"),
        generate_pdf=_boolean(raw["generate_pdf"], "generate_pdf"),
    )
    if config.selected_feature_count < config.pca_components:
        raise ConfigurationError("selected_feature_count must be at least pca_components")
    return config


def _pelt(value: Any) -> PeltConfig:
    keys = {
        "primary_cost_model",
        "primary_penalty",
        "primary_minimum_segment_size",
        "primary_jump",
        "primary_persistence_window",
        "minimum_effect_size",
        "cost_models",
        "penalties",
        "smoothing_windows",
        "minimum_segment_sizes",
        "persistence_windows",
        "agreement_minutes",
        "agreement_life_fraction",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ConfigurationError("pelt has invalid keys")
    costs = _strings(value["cost_models"], "pelt.cost_models")
    if set(costs) - {"rbf", "l2"} or value["primary_cost_model"] not in costs:
        raise ConfigurationError("PELT cost models must be rbf or l2 and include the primary model")
    penalties = tuple(
        _positive_float(item, "pelt.penalties")
        for item in _list(value["penalties"], "pelt.penalties")
    )
    smooth = tuple(
        _odd_int(item, "pelt.smoothing_windows")
        for item in _list(value["smoothing_windows"], "pelt.smoothing_windows")
    )
    minimum = tuple(
        _minimum_int(item, 2, "pelt.minimum_segment_sizes")
        for item in _list(value["minimum_segment_sizes"], "pelt.minimum_segment_sizes")
    )
    persistence = tuple(
        _minimum_int(item, 2, "pelt.persistence_windows")
        for item in _list(value["persistence_windows"], "pelt.persistence_windows")
    )
    primary_penalty = _positive_float(value["primary_penalty"], "pelt.primary_penalty")
    if primary_penalty not in penalties:
        raise ConfigurationError("Primary PELT penalty must occur in the sensitivity grid")
    return PeltConfig(
        str(value["primary_cost_model"]),
        primary_penalty,
        _minimum_int(value["primary_minimum_segment_size"], 2, "pelt.primary_minimum_segment_size"),
        _positive_int(value["primary_jump"], "pelt.primary_jump"),
        _minimum_int(value["primary_persistence_window"], 2, "pelt.primary_persistence_window"),
        _nonnegative_float(value["minimum_effect_size"], "pelt.minimum_effect_size"),
        costs,
        penalties,
        smooth,
        minimum,
        persistence,
        _positive_float(value["agreement_minutes"], "pelt.agreement_minutes"),
        _fraction(value["agreement_life_fraction"], "pelt.agreement_life_fraction"),
    )


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigurationError(f"{name} must be a non-empty path string")
    return (base / value).resolve()


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a non-empty list")
    return value


def _strings(value: Any, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(not isinstance(x, str) or not x for x in value)
    ):
        raise ConfigurationError(f"{name} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ConfigurationError(f"{name} contains duplicates")
    return tuple(value)


def _positive_int(value: Any, name: str) -> int:
    return _minimum_int(value, 1, name)


def _minimum_int(value: Any, minimum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _odd_int(value: Any, name: str) -> int:
    result = _positive_int(value, name)
    if result % 2 == 0:
        raise ConfigurationError(f"{name} must be odd")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _number(value, name)
    if result < 0:
        raise ConfigurationError(f"{name} must be non-negative")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return result


def _fraction(value: Any, name: str) -> float:
    result = _number(value, name)
    if not 0 < result < 1:
        raise ConfigurationError(f"{name} must be strictly between zero and one")
    return result


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ConfigurationError(f"{name} must be finite numeric")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be boolean")
    return value
