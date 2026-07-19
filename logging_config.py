"""Structured logging configuration for ForecastBench."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog

_run_id_var: ContextVar[str] = ContextVar("run_id", default="")
_configured = False


def generate_run_id() -> str:
    run_id = uuid4().hex[:12]
    _run_id_var.set(run_id)
    return run_id


def get_run_id() -> str:
    return _run_id_var.get()


def _add_run_id(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    run_id = _run_id_var.get()
    if run_id:
        event_dict["run_id"] = run_id
    return event_dict


def configure_logging() -> None:
    global _configured
    if _configured:
        return

    json_mode = os.getenv("FORECASTBENCH_LOG_FORMAT", "").lower() == "json"

    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        _add_run_id,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_mode:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    if not _configured:
        configure_logging()
    return structlog.get_logger(logger_name=name)  # type: ignore[no-any-return]


def reset_logging() -> None:
    """Reset configuration state (for testing only)."""
    global _configured
    _configured = False
    structlog.reset_defaults()
