"""CLI for causal degradation detection and survival-ready datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from xjtu_sy_tcc.config.phase7 import load_phase7_config
from xjtu_sy_tcc.logging_utils import configure_logging, log_event
from xjtu_sy_tcc.rul.reporting import phase4_output_lock
from xjtu_sy_tcc.survival.pipeline import run_phase7

LOGGER = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-spectral-cache", action="store_true")
    parser.add_argument("--force-detector-search", action="store_true")
    parser.add_argument("--force-cohorts", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_phase7_config(args.config)
        with phase4_output_lock(config.detector_output):
            report = run_phase7(config, force=args.force_detector_search or args.force_cohorts)
        log_event(
            LOGGER,
            logging.INFO,
            "phase7_complete",
            "Phase 7 completed.",
            status=report["status"],
            duration_seconds=report["duration_seconds"],
            peak_rss_bytes=report["peak_rss_bytes"],
        )
        return 0 if report["status"] == "passed" else 1
    except Exception as error:  # noqa: BLE001
        log_event(
            LOGGER,
            logging.ERROR,
            "phase7_failed",
            "Phase 7 failed.",
            error_type=type(error).__name__,
            error=str(error),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
