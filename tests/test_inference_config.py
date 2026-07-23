"""Tests for inference configuration: extended thinking, scratchpad prompts, freeze-value defaults, multi-horizon defaults."""

from __future__ import annotations

import argparse
import os
from unittest.mock import patch

from fetch_data import Question


def _make_market_question(
    freeze_value: float | None = None,
    freeze_datetime: str | None = "2024-06-15",
) -> Question:
    return Question(
        id="mq1",
        source="metaculus",
        question="Will X happen?",
        background="Some background",
        resolution_criteria="Resolves YES if X.",
        freeze_datetime=freeze_datetime,
        freeze_datetime_value=freeze_value,
        market_info_close_datetime="2024-12-31",
    )


def _make_dataset_question(
    freeze_value: float | None = 42.5,
    freeze_value_explanation: str | None = "Current GDP index",
) -> Question:
    return Question(
        id="dq1",
        source="acled",
        question="Will GDP exceed threshold?",
        background="Economic data question",
        resolution_criteria="Resolves YES if GDP > 50.",
        freeze_datetime="2024-06-05",
        forecast_due_date="2024-06-15",
        freeze_datetime_value=freeze_value,
        freeze_datetime_value_explanation=freeze_value_explanation,
        resolution_dates=["2024-07-01", "2024-08-01", "2024-09-01"],
    )


class TestExtendedThinking:
    def test_thinking_enabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORECAST_THINKING", None)
            import importlib
            import baseline_agent
            importlib.reload(baseline_agent)
            assert baseline_agent.THINKING_ENABLED is True

    def test_thinking_disabled_via_env(self) -> None:
        with patch.dict(os.environ, {"FORECAST_THINKING": "false"}):
            import importlib
            import baseline_agent
            importlib.reload(baseline_agent)
            assert baseline_agent.THINKING_ENABLED is False

    def test_forecast_kwargs_includes_thinking_when_enabled(self) -> None:
        import baseline_agent
        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            messages = [{"role": "user", "content": "test"}]
            kwargs = baseline_agent._forecast_kwargs(messages)
            assert "thinking" in kwargs
            assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": baseline_agent.MAX_TOKENS // 2}
            assert "temperature" not in kwargs

    def test_forecast_kwargs_excludes_thinking_when_disabled(self) -> None:
        import baseline_agent
        with patch.object(baseline_agent, "THINKING_ENABLED", False):
            messages = [{"role": "user", "content": "test"}]
            kwargs = baseline_agent._forecast_kwargs(messages)
            assert "thinking" not in kwargs
            assert kwargs["temperature"] == 0.3


class TestMaxTokens:
    def test_default_max_tokens(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORECAST_MAX_TOKENS", None)
            import importlib
            import baseline_agent
            importlib.reload(baseline_agent)
            assert baseline_agent.MAX_TOKENS == 16384

    def test_custom_max_tokens(self) -> None:
        with patch.dict(os.environ, {"FORECAST_MAX_TOKENS": "8192"}):
            import importlib
            import baseline_agent
            importlib.reload(baseline_agent)
            assert baseline_agent.MAX_TOKENS == 8192

    def test_forecast_kwargs_sets_max_tokens(self) -> None:
        import baseline_agent
        with patch.object(baseline_agent, "MAX_TOKENS", 32768), \
             patch.object(baseline_agent, "THINKING_ENABLED", False):
            kwargs = baseline_agent._forecast_kwargs([{"role": "user", "content": "test"}])
            assert kwargs["max_tokens"] == 32768

    def test_forecast_kwargs_timeout_default(self) -> None:
        import baseline_agent
        with patch.object(baseline_agent, "THINKING_ENABLED", False):
            kwargs = baseline_agent._forecast_kwargs([{"role": "user", "content": "test"}])
            assert kwargs["timeout"] == 180


class TestDatasetPromptRouting:
    def test_dataset_uses_scratchpad_by_default(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_dataset_question()
        prompt = _build_prompt(q, source="acled", prompt_variant="default")
        assert "reasoning steps" in prompt.lower() or "Reasoning:" in prompt

    def test_dataset_zero_shot_uses_original(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_dataset_question()
        prompt = _build_prompt(q, source="acled", prompt_variant="zero-shot")
        assert "Reasoning:" not in prompt
        assert "reasoning steps" not in prompt.lower()

    def test_scratchpad_prompt_has_star_p_format(self) -> None:
        from baseline_agent import SCRATCHPAD_DATASET_PROMPT
        assert "*p*" in SCRATCHPAD_DATASET_PROMPT

    def test_scratchpad_prompt_has_five_reasoning_steps(self) -> None:
        from baseline_agent import SCRATCHPAD_DATASET_PROMPT
        for step_num in range(1, 6):
            assert f"{step_num}." in SCRATCHPAD_DATASET_PROMPT


class TestMarketFreezeValueDefault:
    def test_market_with_freeze_value_uses_freeze_prompt_by_default(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_market_question(freeze_value=0.75)
        prompt = _build_prompt(q, source="metaculus", prompt_variant="default")
        assert "Market value on" in prompt

    def test_market_without_freeze_value_uses_basic_prompt(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_market_question(freeze_value=None)
        prompt = _build_prompt(q, source="metaculus", prompt_variant="default")
        assert "Market value on" not in prompt

    def test_zero_shot_no_fv_forces_basic_market_prompt(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_market_question(freeze_value=0.75)
        prompt = _build_prompt(q, source="metaculus", prompt_variant="zero-shot-no-fv")
        assert "Market value on" not in prompt

    def test_market_freeze_value_with_zero_shot_fv_variant(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_market_question(freeze_value=0.75)
        prompt = _build_prompt(q, source="metaculus", prompt_variant="zero-shot-fv")
        assert "Market value on" in prompt


def _make_timeseries_question(source: str = "fred") -> Question:
    return Question(
        id="tsq1",
        source=source,
        question="Will value exceed threshold by {resolution_date}?",
        background="Time series data question",
        resolution_criteria="Resolves YES if value > 100.",
        freeze_datetime="2024-06-05",
        forecast_due_date="2024-06-15",
        freeze_datetime_value=95.3,
        freeze_datetime_value_explanation="Current index value",
        resolution_dates=["2024-07-01", "2024-08-01", "2024-09-01"],
    )


class TestSourceAwarePromptRouting:
    def test_timeseries_source_uses_zero_shot_by_default(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_timeseries_question(source="fred")
        prompt = _build_prompt(q, source="fred", prompt_variant="default")
        assert "Reasoning:" not in prompt
        assert "reasoning steps" not in prompt.lower()

    def test_event_source_uses_scratchpad_by_default(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_dataset_question()
        prompt = _build_prompt(q, source="acled", prompt_variant="default")
        assert "Reasoning:" in prompt

    def test_dbnomics_uses_zero_shot_by_default(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_timeseries_question(source="dbnomics")
        prompt = _build_prompt(q, source="dbnomics", prompt_variant="default")
        assert "Reasoning:" not in prompt

    def test_yfinance_uses_zero_shot_by_default(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_timeseries_question(source="yfinance")
        prompt = _build_prompt(q, source="yfinance", prompt_variant="default")
        assert "Reasoning:" not in prompt

    def test_explicit_zero_shot_still_works_for_event_source(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_dataset_question()
        prompt = _build_prompt(q, source="acled", prompt_variant="zero-shot")
        assert "Reasoning:" not in prompt


class TestSourceAwareThinking:
    def test_timeseries_source_disables_thinking(self) -> None:
        import baseline_agent
        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            messages = [{"role": "user", "content": "test"}]
            kwargs = baseline_agent._forecast_kwargs(messages, source="fred")
            assert "thinking" not in kwargs
            assert kwargs["temperature"] == 0.3

    def test_timeseries_source_dbnomics(self) -> None:
        import baseline_agent
        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            messages = [{"role": "user", "content": "test"}]
            kwargs = baseline_agent._forecast_kwargs(messages, source="dbnomics")
            assert "thinking" not in kwargs
            assert kwargs["temperature"] == 0.3

    def test_timeseries_source_yfinance(self) -> None:
        import baseline_agent
        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            messages = [{"role": "user", "content": "test"}]
            kwargs = baseline_agent._forecast_kwargs(messages, source="yfinance")
            assert "thinking" not in kwargs
            assert kwargs["temperature"] == 0.3

    def test_market_source_disables_thinking(self) -> None:
        import baseline_agent
        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            messages = [{"role": "user", "content": "test"}]
            kwargs = baseline_agent._forecast_kwargs(messages, source="metaculus")
            assert "thinking" not in kwargs
            assert kwargs["temperature"] == 0.3

    def test_event_source_keeps_thinking(self) -> None:
        import baseline_agent
        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            messages = [{"role": "user", "content": "test"}]
            kwargs = baseline_agent._forecast_kwargs(messages, source="acled")
            assert "thinking" in kwargs
            assert "temperature" not in kwargs

    def test_none_source_keeps_thinking(self) -> None:
        import baseline_agent
        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            messages = [{"role": "user", "content": "test"}]
            kwargs = baseline_agent._forecast_kwargs(messages, source=None)
            assert "thinking" in kwargs
            assert "temperature" not in kwargs


class TestMultiHorizonDefault:
    def test_multi_horizon_default_is_true(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--multi-horizon", action="store_true", dest="multi_horizon", default=True)
        parser.add_argument("--per-date", action="store_false", dest="multi_horizon")
        args = parser.parse_args([])
        assert args.multi_horizon is True

    def test_per_date_disables_multi_horizon(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--multi-horizon", action="store_true", dest="multi_horizon", default=True)
        parser.add_argument("--per-date", action="store_false", dest="multi_horizon")
        args = parser.parse_args(["--per-date"])
        assert args.multi_horizon is False

    def test_eval_argparse_prompt_choices_include_new_variants(self) -> None:
        """Verify the eval.py argparse accepts the new prompt variants."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--prompt",
            choices=["default", "zero-shot", "zero-shot-fv", "zero-shot-no-fv", "dataset"],
            default="default",
        )
        for variant in ["default", "zero-shot", "zero-shot-fv", "zero-shot-no-fv", "dataset"]:
            args = parser.parse_args(["--prompt", variant])
            assert args.prompt == variant

    def test_eval_argparse_prompt_default_is_default(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--prompt",
            choices=["default", "zero-shot", "zero-shot-fv", "zero-shot-no-fv", "dataset"],
            default="default",
        )
        args = parser.parse_args([])
        assert args.prompt == "default"
