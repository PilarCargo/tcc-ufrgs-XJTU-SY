"""CLI for probabilistic post-detection survival prognosis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from xjtu_sy_tcc.config.phase8 import load_phase8_config
from xjtu_sy_tcc.logging_utils import configure_logging, log_event
from xjtu_sy_tcc.rul.reporting import phase4_output_lock
from xjtu_sy_tcc.survival_models.pipeline import run_phase8

LOGGER = logging.getLogger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-manuscript", action="store_true")
    parser.add_argument("--skip-conservative-detector", action="store_true")
    parser.add_argument("--force-model-selection", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--scenario")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_phase8_config(args.config)
        with phase4_output_lock(config.output_directory):
            report = run_phase8(
                config, args.skip_plots, args.skip_manuscript, args.fold, args.scenario
            )
        log_event(LOGGER, logging.INFO, "phase8_complete", "Phase 8 completed.", **report)
        return 0 if report["status"] == "passed" else 1
    except Exception as error:  # noqa: BLE001
        log_event(
            LOGGER,
            logging.ERROR,
            "phase8_failed",
            "Phase 8 failed.",
            error_type=type(error).__name__,
            error=str(error),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
