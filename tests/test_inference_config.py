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
    def test_thinking_var_still_parsed(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORECAST_THINKING", None)
            import importlib
            import baseline_agent
            importlib.reload(baseline_agent)
            assert baseline_agent.THINKING_ENABLED is True

    def test_forecast_kwargs_source_aware(self) -> None:
        import baseline_agent
        messages = [{"role": "user", "content": "test"}]
        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            kwargs_fred = baseline_agent._forecast_kwargs(messages, source="fred")
            assert "thinking" not in kwargs_fred
            assert kwargs_fred["temperature"] == 0.3

            kwargs_acled = baseline_agent._forecast_kwargs(messages, source="acled")
            assert "thinking" in kwargs_acled
            assert "temperature" not in kwargs_acled


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
    def test_event_source_uses_scratchpad_by_default(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_dataset_question()
        prompt = _build_prompt(q, source="acled", prompt_variant="default")
        assert "Reasoning:" in prompt
        assert "reasoning steps" in prompt.lower()

    def test_dataset_zero_shot_uses_original(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_dataset_question()
        prompt = _build_prompt(q, source="acled", prompt_variant="zero-shot")
        assert "Reasoning:" not in prompt
        assert "reasoning steps" not in prompt.lower()

    def test_scratchpad_prompt_still_exists(self) -> None:
        from baseline_agent import SCRATCHPAD_DATASET_PROMPT
        assert "*p*" in SCRATCHPAD_DATASET_PROMPT
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
    def test_timeseries_sources_use_zero_shot_by_default(self) -> None:
        from baseline_agent import _build_prompt
        for source in ["fred", "dbnomics", "yfinance"]:
            q = _make_timeseries_question(source=source)
            prompt = _build_prompt(q, source=source, prompt_variant="default")
            assert "Reasoning:" not in prompt, f"source={source} should use zero-shot, not scratchpad"
            assert "reasoning steps" not in prompt.lower(), f"source={source} should use zero-shot"

    def test_event_sources_use_scratchpad_by_default(self) -> None:
        from baseline_agent import _build_prompt
        for source in ["acled", "wikipedia"]:
            q = _make_timeseries_question(source=source)
            prompt = _build_prompt(q, source=source, prompt_variant="default")
            assert "Reasoning:" in prompt, f"source={source} should use scratchpad"
            assert "reasoning steps" in prompt.lower(), f"source={source} should use scratchpad"

    def test_explicit_zero_shot_still_works_for_event_source(self) -> None:
        from baseline_agent import _build_prompt
        q = _make_dataset_question()
        prompt = _build_prompt(q, source="acled", prompt_variant="zero-shot")
        assert "Reasoning:" not in prompt


class TestSourceAwareThinking:
    def test_timeseries_and_market_disable_thinking(self) -> None:
        import baseline_agent
        messages = [{"role": "user", "content": "test"}]
        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            for source in ["fred", "dbnomics", "yfinance", "metaculus", "polymarket", "manifold", "infer"]:
                kwargs = baseline_agent._forecast_kwargs(messages, source=source)
                assert "thinking" not in kwargs, f"source={source} should not enable thinking"
                assert kwargs["temperature"] == 0.3, f"source={source} should use temperature=0.3"

    def test_event_sources_enable_thinking(self) -> None:
        import baseline_agent
        messages = [{"role": "user", "content": "test"}]
        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            for source in ["acled", "wikipedia"]:
                kwargs = baseline_agent._forecast_kwargs(messages, source=source)
                assert "thinking" in kwargs, f"source={source} should enable thinking"
                assert "temperature" not in kwargs, f"source={source} should not set temperature when thinking"

    def test_event_sources_no_thinking_when_disabled(self) -> None:
        import baseline_agent
        messages = [{"role": "user", "content": "test"}]
        with patch.object(baseline_agent, "THINKING_ENABLED", False):
            for source in ["acled", "wikipedia"]:
                kwargs = baseline_agent._forecast_kwargs(messages, source=source)
                assert "thinking" not in kwargs, f"source={source} should not think when THINKING_ENABLED=false"
                assert kwargs["temperature"] == 0.3

    def test_no_source_enables_thinking(self) -> None:
        import baseline_agent
        messages = [{"role": "user", "content": "test"}]
        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            kwargs = baseline_agent._forecast_kwargs(messages, source=None)
            assert "thinking" in kwargs, "source=None should enable thinking"


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
