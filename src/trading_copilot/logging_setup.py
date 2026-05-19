"""Structured logging via structlog.

Console output is human-readable; the same events are also written as JSON to
`<log_dir>/copilot.log` so we can grep / parse later (Phase 9+ will lean on this
for backtest reproducibility).
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import structlog

from trading_copilot.config import Settings, get_settings


def configure_logging(settings: Settings | None = None) -> structlog.stdlib.BoundLogger:
    settings = settings or get_settings()
    log_dir: Path = settings.paths.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "copilot.log"

    level = getattr(logging, settings.app.log_level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.ExceptionRenderer(),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Mirror everything as JSON to a rotating file. We attach a separate handler
    # on the root stdlib logger so third-party libs are captured too.
    root = logging.getLogger()
    root.setLevel(level)

    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter('{"ts":"%(asctime)s","level":"%(levelname)s","msg":%(message)r}')
        )
        root.addHandler(file_handler)

    return structlog.get_logger("trading_copilot")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Cheap accessor — callers don't need to know configure_logging exists."""
    return structlog.get_logger(name or "trading_copilot")
