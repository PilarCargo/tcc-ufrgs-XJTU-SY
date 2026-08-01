"""CLI for Phase-2 vibration features and exploratory figures."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from xjtu_sy_tcc.config.features import load_feature_config
from xjtu_sy_tcc.features.plotting import generate_exploratory_plots
from xjtu_sy_tcc.features.processing import build_features
from xjtu_sy_tcc.features.reporting import feature_output_lock, write_feature_outputs
from xjtu_sy_tcc.logging_utils import configure_logging, log_event

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build validated XJTU-SY Phase-2 vibration features."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to the Phase-2 YAML configuration."
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore a resumable extraction checkpoint."
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Build tables without exploratory figures."
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)
    try:
        config = load_feature_config(arguments.config)
        with feature_output_lock(config):
            result = build_features(config, resume=not arguments.force)
            artifacts = write_feature_outputs(config, result)
            plot_files = 0
            if result.validation.status == "passed" and not arguments.skip_plots:
                plot_files = generate_exploratory_plots(config, result.table)
            log_event(
                LOGGER,
                logging.INFO,
                "feature_build_complete",
                "Phase-2 feature build completed.",
                status=result.validation.status,
                rows=len(result.table),
                duration_seconds=result.duration_seconds,
                peak_rss_bytes=result.peak_rss_bytes,
                plot_files=plot_files,
                artifacts={name: path.as_posix() for name, path in artifacts.items()},
            )
        return 0 if result.validation.status == "passed" else 1
    except Exception as error:  # noqa: BLE001
        log_event(
            LOGGER,
            logging.ERROR,
            "feature_build_failed",
            "Phase-2 feature build could not complete.",
            error_type=type(error).__name__,
            error=str(error),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
