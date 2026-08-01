"""Unified restartable CLI for leakage-free Phase 3 analysis."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from xjtu_sy_tcc.config.phase3 import load_phase3_config
from xjtu_sy_tcc.degradation.pipeline import run_phase3
from xjtu_sy_tcc.degradation.reporting import phase3_output_lock
from xjtu_sy_tcc.logging_utils import configure_logging, log_event

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run leakage-free XJTU-SY Phase 3 analysis.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to Phase 3 YAML configuration."
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Run tables and validation without figures."
    )
    parser.add_argument(
        "--force-sensitivity",
        action="store_true",
        help="Recompute the PELT grid instead of reusing a matching validated artifact.",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)
    try:
        config = load_phase3_config(arguments.config)
        with phase3_output_lock(config.output_root):
            report = run_phase3(
                config,
                generate_figures=not arguments.skip_plots,
                reuse_sensitivity=not arguments.force_sensitivity,
            )
        log_event(
            LOGGER,
            logging.INFO,
            "phase3_complete",
            "Phase 3 completed.",
            status=report["status"],
            rows=report["input_rows"],
            duration_seconds=report["duration_seconds"],
            peak_rss_bytes=report["peak_rss_bytes"],
            figures=report.get("figure_count", 0),
        )
        return 0 if report["status"] == "passed" else 1
    except Exception as error:  # noqa: BLE001
        log_event(
            LOGGER,
            logging.ERROR,
            "phase3_failed",
            "Phase 3 could not complete.",
            error_type=type(error).__name__,
            error=str(error),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
