"""Strict pre-specified Phase 6 statistical configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from xjtu_sy_tcc.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class Phase6Config:
    config_path: Path
    project_root: Path
    split_manifest: Path
    phase4_predictions: Path
    phase5_predictions: Path
    phase3_rankings: Path
    phase4_importance: Path
    phase5_importance: Path
    phase4_costs: Path
    phase5_costs: Path
    test_onsets: Path
    output_directory: Path
    expected_split_sha256: str
    expected_split_configuration_hash: str
    primary_models: tuple[str, ...]
    secondary_models: tuple[str, ...]
    primary_contrasts: tuple[tuple[str, str], ...]
    tie_tolerance_minutes: float
    practical_tolerance_minutes: float
    bootstrap_seed: int
    bootstrap_replicates: int
    confidence_level: float
    alpha: float
    multiple_comparison_method: str
    resampling_unit: str
    stratified_bootstrap: bool
    decimal_precision: int
    plot_dpi: int
    generate_pdf: bool

    @property
    def configuration_hash(self):
        data = asdict(self)
        data.pop("config_path")
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def load_phase6_config(path: Path) -> Phase6Config:
    p = path.resolve()
    raw = yaml.safe_load(p.read_text())
    root = (p.parent / raw.pop("project_root")).resolve()
    path_keys = (
        "split_manifest",
        "phase4_predictions",
        "phase5_predictions",
        "phase3_rankings",
        "phase4_importance",
        "phase5_importance",
        "phase4_costs",
        "phase5_costs",
        "test_onsets",
        "output_directory",
    )
    paths = {k: (root / raw.pop(k)).resolve() for k in path_keys}
    models = tuple(raw.pop("primary_models"))
    secondary = tuple(raw.pop("secondary_models"))
    contrasts = tuple(tuple(x) for x in raw.pop("primary_contrasts"))
    if (
        len(set(models)) != len(models)
        or not models
        or any(len(x) != 2 for x in contrasts)
        or len(set(contrasts)) != len(contrasts)
    ):
        raise ConfigurationError("Invalid model or contrast declaration")
    if any(a not in models or b not in models for a, b in contrasts):
        raise ConfigurationError("Primary contrasts must use primary models")
    if (
        raw["bootstrap_replicates"] < 100
        or not 0 < raw["confidence_level"] < 1
        or not 0 < raw["alpha"] < 1
    ):
        raise ConfigurationError("Invalid statistical configuration")
    if raw["multiple_comparison_method"] != "holm" or raw["resampling_unit"] != "bearing":
        raise ConfigurationError("Only Holm and bearing resampling are supported")
    if paths["output_directory"].name != "consolidation":
        raise ConfigurationError("Phase 6 output must be isolated")
    return Phase6Config(
        config_path=p,
        project_root=root,
        primary_models=models,
        secondary_models=secondary,
        primary_contrasts=contrasts,
        **paths,
        **raw,
    )
