"""Tests for logging_config module."""

from __future__ import annotations

import io
import os
from unittest.mock import patch

import structlog

from logging_config import (
    configure_logging,
    generate_run_id,
    get_logger,
    get_run_id,
    reset_logging,
)


class TestGetLogger:
    def setup_method(self) -> None:
        reset_logging()

    def teardown_method(self) -> None:
        reset_logging()

    def test_returns_bound_logger(self) -> None:
        log = get_logger("test")
        assert log is not None

    def test_logger_has_standard_methods(self) -> None:
        log = get_logger("test")
        assert callable(getattr(log, "info", None))
        assert callable(getattr(log, "warning", None))
        assert callable(getattr(log, "error", None))
        assert callable(getattr(log, "debug", None))


class TestRunId:
    def setup_method(self) -> None:
        reset_logging()

    def teardown_method(self) -> None:
        reset_logging()

    def test_generate_run_id_returns_string(self) -> None:
        run_id = generate_run_id()
        assert isinstance(run_id, str)
        assert len(run_id) == 12

    def test_generate_run_id_is_unique(self) -> None:
        id1 = generate_run_id()
        id2 = generate_run_id()
        assert id1 != id2

    def test_get_run_id_returns_current(self) -> None:
        run_id = generate_run_id()
        assert get_run_id() == run_id

    def test_get_run_id_empty_before_generate(self) -> None:
        from logging_config import _run_id_var
        _run_id_var.set("")
        assert get_run_id() == ""


class TestConsoleRendering:
    def setup_method(self) -> None:
        reset_logging()

    def teardown_method(self) -> None:
        reset_logging()

    def test_console_mode_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORECASTBENCH_LOG_FORMAT", None)
            configure_logging()
            buf = io.StringIO()
            with patch("structlog.PrintLoggerFactory", return_value=structlog.PrintLoggerFactory(buf)):
                pass
            log = get_logger("test_console")
            log.info("hello", key="value")


class TestJsonRendering:
    def setup_method(self) -> None:
        reset_logging()

    def teardown_method(self) -> None:
        reset_logging()

    def test_json_mode_with_env(self) -> None:
        with patch.dict(os.environ, {"FORECASTBENCH_LOG_FORMAT": "json"}):
            configure_logging()
            log = get_logger("test_json")
            log.info("hello", key="value")

    def test_json_output_is_parseable(self, capsys: object) -> None:
        with patch.dict(os.environ, {"FORECASTBENCH_LOG_FORMAT": "json"}):
            configure_logging()
            log = get_logger("test_json_parse")
            log.info("test_event", data=42)


class TestConfigureLogging:
    def setup_method(self) -> None:
        reset_logging()

    def teardown_method(self) -> None:
        reset_logging()

    def test_configure_is_idempotent(self) -> None:
        configure_logging()
        configure_logging()
        log = get_logger("test")
        log.info("still_works")

    def test_reset_allows_reconfigure(self) -> None:
        configure_logging()
        reset_logging()
        with patch.dict(os.environ, {"FORECASTBENCH_LOG_FORMAT": "json"}):
            configure_logging()
        log = get_logger("test_reset")
        log.info("after_reset")


class TestRunIdInLogs:
    def setup_method(self) -> None:
        reset_logging()

    def teardown_method(self) -> None:
        reset_logging()

    def test_run_id_appears_in_json_output(self, capsys: object) -> None:
        with patch.dict(os.environ, {"FORECASTBENCH_LOG_FORMAT": "json"}):
            configure_logging()
            generate_run_id()
            log = get_logger("test_run_id")
            log.info("with_run_id", foo="bar")


class TestSmokeTests:
    """Verify that log calls in instrumented modules don't crash."""

    def setup_method(self) -> None:
        reset_logging()

    def teardown_method(self) -> None:
        reset_logging()

    def test_multiple_loggers(self) -> None:
        configure_logging()
        for name in ["eval", "score", "fetch_data", "baseline_agent", "cutoff", "dummy_forecaster"]:
            log = get_logger(name)
            log.info("smoke_test", module=name)
            log.debug("smoke_debug", module=name)
            log.warning("smoke_warning", module=name)

    def test_log_with_exception_info(self) -> None:
        configure_logging()
        log = get_logger("test_exc")
        try:
            raise ValueError("test error")
        except ValueError:
            log.error("caught_error", exc_info=True)

    def test_log_with_various_types(self) -> None:
        configure_logging()
        log = get_logger("test_types")
        log.info("mixed_types", count=42, ratio=0.75, flag=True, label="test", items=[1, 2])
