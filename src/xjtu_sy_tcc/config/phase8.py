"""Strict configuration for probabilistic post-detection survival prognosis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from xjtu_sy_tcc.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class Phase8Config:
    config_path: Path
    project_root: Path
    split_manifest: Path
    onset_cohort: Path
    full_event_cohort: Path
    fixed_horizon_cohort: Path
    rate_censored_cohort: Path
    evaluation_truth: Path
    survival_schema: Path
    forbidden_columns: Path
    censoring_manifest: Path
    primary_onsets: Path
    detector_candidates: Path
    selected_detectors: Path
    phase3_rankings: Path
    phase4_predictions: Path
    phase5_predictions: Path
    phase6_consensus: Path
    output_directory: Path
    expected_split_sha256: str
    expected_split_configuration_hash: str
    primary_scenarios: tuple[str, ...]
    secondary_scenarios: tuple[str, ...]
    context_features: tuple[str, ...]
    correlation_threshold: float
    near_constant_threshold: float
    feature_counts: tuple[int, ...]
    cox_alphas: tuple[float, ...]
    rsf_candidates: tuple[dict[str, object], ...]
    evaluation_quantiles: tuple[float, float]
    evaluation_points: int
    prediction_horizons: tuple[float, ...]
    calibration_bins: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    random_seed: int
    generate_pdf: bool
    plot_dpi: int
    conservative_detector_enabled: bool
    conservative_min_threshold_delta: float
    conservative_min_persistence_delta: int

    @property
    def configuration_hash(self) -> str:
        data = asdict(self)
        data.pop("config_path")
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def load_phase8_config(path: Path) -> Phase8Config:
    p = path.resolve()
    raw = yaml.safe_load(p.read_text())
    root = (p.parent / raw.pop("project_root")).resolve()
    path_keys = (
        "split_manifest",
        "onset_cohort",
        "full_event_cohort",
        "fixed_horizon_cohort",
        "rate_censored_cohort",
        "evaluation_truth",
        "survival_schema",
        "forbidden_columns",
        "censoring_manifest",
        "primary_onsets",
        "detector_candidates",
        "selected_detectors",
        "phase3_rankings",
        "phase4_predictions",
        "phase5_predictions",
        "phase6_consensus",
        "output_directory",
    )
    paths = {key: (root / raw.pop(key)).resolve() for key in path_keys}
    tuple_keys = (
        "primary_scenarios",
        "secondary_scenarios",
        "context_features",
        "feature_counts",
        "cox_alphas",
        "prediction_horizons",
    )
    for key in tuple_keys:
        raw[key] = tuple(raw[key])
    raw["evaluation_quantiles"] = tuple(raw["evaluation_quantiles"])
    raw["rsf_candidates"] = tuple(raw["rsf_candidates"])
    allowed = {
        "full_event",
        "fixed_horizon_30",
        "fixed_horizon_60",
        "fixed_horizon_120",
        "fixed_horizon_240",
        "target_censoring_25",
        "target_censoring_50",
        "target_censoring_75",
    }
    scenarios = set(raw["primary_scenarios"] + raw["secondary_scenarios"])
    if scenarios - allowed or len(scenarios) != len(
        raw["primary_scenarios"] + raw["secondary_scenarios"]
    ):
        raise ConfigurationError("Unknown or duplicate survival scenario")
    if max(raw["feature_counts"], default=0) > 20 or min(raw["feature_counts"], default=0) < 1:
        raise ConfigurationError("Signal feature counts must be between 1 and 20")
    if not 0 < raw["correlation_threshold"] < 1 or not 0 < raw["near_constant_threshold"] <= 1:
        raise ConfigurationError("Invalid feature-filtering threshold")
    if raw["bootstrap_replicates"] < 100 or raw["evaluation_points"] < 2:
        raise ConfigurationError("Insufficient statistical or grid configuration")
    if paths["output_directory"].name != "survival_models":
        raise ConfigurationError("Phase 8 outputs must be isolated from previous phases")
    return Phase8Config(config_path=p, project_root=root, **paths, **raw)
