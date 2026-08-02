"""CLI for causal temporal RUL regression."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from xjtu_sy_tcc.config.phase5 import load_phase5_config
from xjtu_sy_tcc.logging_utils import configure_logging, log_event
from xjtu_sy_tcc.rul.reporting import phase4_output_lock
from xjtu_sy_tcc.temporal.pipeline import run_phase5

LOGGER = logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(description="Run leakage-free causal LSTM RUL experiments")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_phase5_config(args.config)
        if args.device:
            config = type(config)(
                **{
                    **{name: getattr(config, name) for name in config.__dataclass_fields__},
                    "device": args.device,
                }
            )
        with phase4_output_lock(config.output_directory):
            report = run_phase5(
                config, force_train=args.force_train, generate_figures=not args.skip_plots
            )
        log_event(
            LOGGER,
            logging.INFO,
            "phase5_complete",
            "Phase 5 completed.",
            status=report["status"],
            duration_seconds=report["duration_seconds"],
            peak_rss_bytes=report["peak_rss_bytes"],
        )
        return 0
    except Exception as error:  # noqa: BLE001
        log_event(
            LOGGER,
            logging.ERROR,
            "phase5_failed",
            "Phase 5 failed.",
            error_type=type(error).__name__,
            error=str(error),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
