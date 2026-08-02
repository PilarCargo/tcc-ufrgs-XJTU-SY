"""Strict configuration for causal temporal RUL experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from xjtu_sy_tcc.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class LSTMCandidate:
    sequence_length: int
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float


@dataclass(frozen=True, slots=True)
class Phase5Config:
    config_path: Path
    project_root: Path
    feature_table: Path
    split_manifest: Path
    selected_feature_manifest: Path
    phase4_predictions: Path
    phase4_comparison: Path
    test_onset_table: Path
    output_directory: Path
    expected_feature_rows: int
    expected_feature_count: int
    expected_split_sha256: str
    expected_split_configuration_hash: str
    device: str
    random_seed: int
    heavy_tail_feature_tokens: tuple[str, ...]
    candidates: tuple[LSTMCandidate, ...]
    max_epochs: int
    patience: int
    minimum_delta: float
    batch_size: int
    weight_decay: float
    huber_beta: float
    gradient_clip_norm: float
    life_stage_thresholds: tuple[float, float]
    plot_dpi: int
    generate_pdf: bool

    @property
    def configuration_hash(self):
        data = asdict(self)
        data.pop("config_path")
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def load_phase5_config(path: Path) -> Phase5Config:
    p = path.resolve()
    raw = yaml.safe_load(p.read_text())
    root = (p.parent / raw.pop("project_root")).resolve()
    paths = (
        "feature_table",
        "split_manifest",
        "selected_feature_manifest",
        "phase4_predictions",
        "phase4_comparison",
        "test_onset_table",
        "output_directory",
    )
    resolved = {k: (root / raw.pop(k)).resolve() for k in paths}
    candidates = tuple(LSTMCandidate(**x) for x in raw.pop("candidates"))
    if not candidates or any(
        c.sequence_length < 1 or c.hidden_size < 1 or c.num_layers < 1 or c.learning_rate <= 0
        for c in candidates
    ):
        raise ConfigurationError("Invalid LSTM candidate")
    if any(c.dropout and c.num_layers == 1 for c in candidates):
        raise ConfigurationError("One-layer LSTM dropout must be zero")
    if raw["device"] not in {"cpu", "cuda", "mps", "auto"}:
        raise ConfigurationError("Invalid device")
    if raw["max_epochs"] < 1 or raw["patience"] < 1 or raw["patience"] > raw["max_epochs"]:
        raise ConfigurationError("Invalid epochs/patience")
    t = tuple(raw.pop("life_stage_thresholds"))
    if len(t) != 2 or not 0 < t[0] < t[1] < 1:
        raise ConfigurationError("Invalid life-stage thresholds")
    output = resolved["output_directory"]
    if output == root / "outputs" or output.name in {
        "audits",
        "features",
        "splits",
        "prognostics",
        "degradation",
        "rul",
    }:
        raise ConfigurationError("Phase 5 output overlaps an earlier phase")
    return Phase5Config(
        config_path=p,
        project_root=root,
        candidates=candidates,
        heavy_tail_feature_tokens=tuple(raw.pop("heavy_tail_feature_tokens")),
        life_stage_thresholds=t,
        **resolved,
        **raw,
    )
