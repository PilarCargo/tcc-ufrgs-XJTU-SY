"""Command-line interface for the exhaustive Phase-1 dataset audit."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from xjtu_sy_tcc.config.data import load_data_config
from xjtu_sy_tcc.data.audit import run_audit
from xjtu_sy_tcc.data.reporting import (
    audit_output_lock,
    invalidate_audit_outputs,
    write_audit_outputs,
)
from xjtu_sy_tcc.logging_utils import configure_logging, log_event

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the audit command-line parser."""

    parser = argparse.ArgumentParser(
        description="Validate XJTU-SY acquisition files and generate Phase-1 metadata."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Phase-1 data YAML configuration.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Structured log threshold (default: INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and return a process exit code."""

    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)
    try:
        config = load_data_config(arguments.config)
        with audit_output_lock(config):
            invalidate_audit_outputs(config.output_directory)
            result = run_audit(config)
            output_paths = write_audit_outputs(config, result)
            log_event(
                LOGGER,
                logging.INFO,
                "audit_artifacts_written",
                "Dataset audit artifacts were written.",
                artifacts={name: path.as_posix() for name, path in output_paths.items()},
            )
    except Exception as error:  # noqa: BLE001 - the CLI is the process error boundary
        log_event(
            LOGGER,
            logging.ERROR,
            "audit_execution_failed",
            "Dataset audit could not complete.",
            error_type=type(error).__name__,
            error=str(error),
        )
        return 2
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
