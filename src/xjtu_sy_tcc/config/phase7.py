"""Strict configuration for causal detection and survival-ready datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from xjtu_sy_tcc.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class Phase7Config:
    config_path: Path
    project_root: Path
    metadata_path: Path
    feature_table: Path
    split_manifest: Path
    selected_features: Path
    retrospective_onsets: Path
    detector_output: Path
    survival_output: Path
    expected_split_sha256: str
    expected_split_configuration_hash: str
    welch_nperseg: int
    welch_overlap: int
    frequency_min_hz: float
    frequency_max_hz: float
    spectral_bins: int
    epsilon: float
    workers: int
    baseline_candidates: tuple[int, ...]
    divergence_candidates: tuple[str, ...]
    channel_aggregations: tuple[str, ...]
    causal_windows: tuple[int, ...]
    threshold_candidates: tuple[float, ...]
    persistence_candidates: tuple[int, ...]
    minimum_post_alarm: int
    minimum_effect: float
    maximum_gap: int
    trend_windows: tuple[int, ...]
    landmark_stride: int
    minimum_followup: int
    fixed_horizons: tuple[float, ...]
    target_censoring_rates: tuple[float, ...]
    random_seed: int
    plot_dpi: int
    generate_pdf: bool

    @property
    def configuration_hash(self):
        d = asdict(self)
        d.pop("config_path")
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def load_phase7_config(path: Path):
    p = path.resolve()
    raw = yaml.safe_load(p.read_text())
    root = (p.parent / raw.pop("project_root")).resolve()
    path_keys = (
        "metadata_path",
        "feature_table",
        "split_manifest",
        "selected_features",
        "retrospective_onsets",
        "detector_output",
        "survival_output",
    )
    paths = {k: (root / raw.pop(k)).resolve() for k in path_keys}
    tuple_keys = (
        "baseline_candidates",
        "divergence_candidates",
        "channel_aggregations",
        "causal_windows",
        "threshold_candidates",
        "persistence_candidates",
        "trend_windows",
        "fixed_horizons",
        "target_censoring_rates",
    )
    for k in tuple_keys:
        raw[k] = tuple(raw[k])
    if (
        raw["epsilon"] <= 0
        or raw["spectral_bins"] < 2
        or not 0 <= raw["frequency_min_hz"] < raw["frequency_max_hz"] <= 12800
    ):
        raise ConfigurationError("Invalid spectral configuration")
    if any(
        x < 1
        for x in raw["baseline_candidates"] + raw["causal_windows"] + raw["persistence_candidates"]
    ):
        raise ConfigurationError("Causal counts must be positive")
    if set(raw["divergence_candidates"]) - {"kl", "symmetric_kl"} or set(
        raw["channel_aggregations"]
    ) - {"mean", "maximum"}:
        raise ConfigurationError("Unknown detector option")
    if (
        paths["detector_output"].name != "causal_degradation"
        or paths["survival_output"].name != "survival"
    ):
        raise ConfigurationError("Phase 7 outputs must be isolated")
    return Phase7Config(config_path=p, project_root=root, **paths, **raw)
