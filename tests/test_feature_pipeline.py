"""Small end-to-end tests for incremental feature processing and output schemas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xjtu_sy_tcc.config.features import FeatureConfig, FrequencyBand
from xjtu_sy_tcc.features.processing import (
    PASSTHROUGH_COLUMNS,
    build_features,
    feature_column_names,
)
from xjtu_sy_tcc.features.reporting import write_feature_outputs


def test_pipeline_preserves_manifest_alignment_and_writes_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    data_directory = tmp_path / "Data" / "condition" / "B1"
    metadata_rows = []
    for sequence in range(3):
        path = data_directory / f"{sequence + 1}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        time = np.arange(128) / 128.0
        values = np.column_stack(
            [np.sin(2 * np.pi * (8 + sequence) * time), np.cos(2 * np.pi * 16 * time)]
        )
        np.savetxt(
            path,
            values,
            delimiter=",",
            header="Horizontal_vibration_signals,Vertical_vibration_signals",
            comments="",
        )
        metadata_rows.append(
            {
                "condition_id": 1,
                "bearing_id": "B1",
                "file_path": path.relative_to(tmp_path).as_posix(),
                "file_name": path.name,
                "acquisition_number": sequence + 1,
                "sequence_index": sequence,
                "elapsed_minutes": float(sequence),
                "rul_minutes": float(2 - sequence),
                "rotation_rpm": 2100.0,
                "radial_load_kn": 12.0,
                "sampling_frequency_hz": 128.0,
                "sample_count": 128,
                "channel_count": 2,
            }
        )
    pd.DataFrame(metadata_rows).to_parquet(config.metadata_path, index=False)

    result = build_features(config)
    paths = write_feature_outputs(config, result)

    assert result.validation.status == "passed"
    assert len(result.table) == 3
    assert list(result.table.columns) == list(PASSTHROUGH_COLUMNS + feature_column_names(config))
    assert result.table["acquisition_number"].tolist() == [1, 2, 3]
    assert result.table["rul_minutes"].tolist() == [2.0, 1.0, 0.0]
    assert set(paths) == {"features", "validation_report", "summary"}
    persisted = pd.read_parquet(paths["features"])
    pd.testing.assert_frame_equal(persisted, result.table)
    report = json.loads(paths["validation_report"].read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["observed_rows"] == 3
    assert report["target_columns"] == ["rul_minutes"]
    assert report["feature_count"] == 46


def _config(root: Path) -> FeatureConfig:
    output = root / "outputs" / "features"
    output.mkdir(parents=True)
    return FeatureConfig(
        config_path=root / "configs" / "features.yaml",
        project_root=root,
        metadata_path=root / "metadata.parquet",
        output_directory=output,
        figure_directory=root / "outputs" / "figures",
        welch_nperseg=64,
        welch_overlap=32,
        frequency_bands_hz=(FrequencyBand("low", 0.0, 32.0),),
        plot_dpi=100,
        generate_pdf=False,
        progress_interval=2,
    )
