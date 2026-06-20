"""structlog JSON logging setup. Call configure() once at startup."""

from __future__ import annotations

import logging

import structlog


def configure(level: str = "info") -> None:
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(*args: object, **kw: object) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(*args, **kw)
