"""CLI for no-training statistical consolidation and thesis reporting."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from xjtu_sy_tcc.config.phase6 import load_phase6_config
from xjtu_sy_tcc.consolidation.pipeline import run_phase6
from xjtu_sy_tcc.logging_utils import configure_logging, log_event
from xjtu_sy_tcc.rul.reporting import phase4_output_lock

LOGGER = logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Consolidate validated thesis statistics without model training"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-manuscript", action="store_true")
    parser.add_argument("--force-statistics", action="store_true")
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_phase6_config(args.config)
        with phase4_output_lock(config.output_directory):
            report = run_phase6(
                config,
                generate_plots=not args.skip_plots,
                generate_manuscript=not args.skip_manuscript,
                force=args.force_statistics or args.force_all,
            )
        log_event(
            LOGGER,
            logging.INFO,
            "phase6_complete",
            "Phase 6 completed without model training.",
            status=report["status"],
            duration_seconds=report["duration_seconds"],
            peak_rss_bytes=report["peak_rss_bytes"],
        )
        return 0 if report["status"] == "passed" else 1
    except Exception as error:  # noqa: BLE001
        log_event(
            LOGGER,
            logging.ERROR,
            "phase6_failed",
            "Phase 6 failed.",
            error_type=type(error).__name__,
            error=str(error),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
