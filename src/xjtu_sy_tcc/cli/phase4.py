"""CLI for leakage-free classical RUL regression."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from xjtu_sy_tcc.config.phase4 import load_phase4_config
from xjtu_sy_tcc.logging_utils import configure_logging, log_event
from xjtu_sy_tcc.rul.pipeline import run_phase4
from xjtu_sy_tcc.rul.reporting import phase4_output_lock

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run classical leakage-free XJTU-SY RUL regression."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)
    try:
        config = load_phase4_config(arguments.config)
        with phase4_output_lock(config.output_directory):
            report = run_phase4(
                config,
                generate_figures=not arguments.skip_plots,
                force_train=arguments.force_train,
            )
        log_event(
            LOGGER,
            logging.INFO,
            "phase4_complete",
            "Phase 4 completed.",
            status=report["status"],
            duration_seconds=report["duration_seconds"],
            peak_rss_bytes=report["peak_rss_bytes"],
            figures=report["figure_files"],
        )
        return 0 if report["status"] == "passed" else 1
    except Exception as error:  # noqa: BLE001
        log_event(
            LOGGER,
            logging.ERROR,
            "phase4_failed",
            "Phase 4 could not complete.",
            error_type=type(error).__name__,
            error=str(error),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
