"""Tests verifying prompt templates match upstream ForecastBench exactly."""

from __future__ import annotations

from fetch_data import Question
from baseline_agent import _build_prompt


def _make_question(
    freeze: str | None = "2024-06-15",
    source: str = "metaculus",
    freeze_datetime_value: float | None = None,
    freeze_datetime_value_explanation: str | None = None,
    resolution_dates: list[str] | None = None,
) -> Question:
    return Question(
        id="q1",
        source=source,
        question="Will X happen?",
        background="Some background",
        resolution_criteria="Resolves YES if X.",
        freeze_datetime=freeze,
        freeze_datetime_value=freeze_datetime_value,
        freeze_datetime_value_explanation=freeze_datetime_value_explanation,
        resolution_dates=resolution_dates,
    )


class TestZeroShotMarketPromptAlignment:
    def test_contains_expert_superforecaster(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "expert superforecaster" in prompt

    def test_contains_tetlock(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "Tetlock" in prompt

    def test_contains_question_background_label(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "Question Background:" in prompt

    def test_contains_resolution_criteria_label(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "Resolution Criteria:" in prompt

    def test_contains_todays_date_label(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "Today's Date:" in prompt

    def test_contains_question_resolution_date_singular(self) -> None:
        q = _make_question()
        q = q.model_copy(update={"market_info_close_datetime": "2024-12-31"})
        prompt = _build_prompt(q)
        assert "Question resolution date:" in prompt

    def test_contains_asterisk_instruction(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "asterisk" in prompt

    def test_contains_answer_placeholder(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "Answer:" in prompt


class TestFreezeValueMarketPromptAlignment:
    def test_contains_market_value_on(self) -> None:
        q = _make_question(
            source="metaculus",
            freeze="2024-06-15",
            freeze_datetime_value=0.73,
        )
        prompt = _build_prompt(q, prompt_variant="zero-shot-fv")
        assert "Market value on" in prompt

    def test_contains_freeze_datetime(self) -> None:
        q = _make_question(
            source="metaculus",
            freeze="2024-06-15",
            freeze_datetime_value=0.73,
        )
        prompt = _build_prompt(q, prompt_variant="zero-shot-fv")
        assert "2024-06-15" in prompt

    def test_contains_answer_placeholder(self) -> None:
        q = _make_question(
            source="metaculus",
            freeze="2024-06-15",
            freeze_datetime_value=0.73,
        )
        prompt = _build_prompt(q, prompt_variant="zero-shot-fv")
        assert "Answer:" in prompt


class TestDatasetPromptAlignment:
    def test_contains_at_each_of_resolution_dates(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current GDP value",
            resolution_dates=["2024-07-01", "2024-08-01"],
        )
        prompt = _build_prompt(q, prompt_variant="dataset")
        assert "at each of the resolution dates" in prompt

    def test_contains_current_value_on(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current GDP value",
            resolution_dates=["2024-07-01"],
        )
        prompt = _build_prompt(q, prompt_variant="dataset")
        assert "Current value on" in prompt

    def test_contains_value_explanation_label(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current GDP value",
            resolution_dates=["2024-07-01"],
        )
        prompt = _build_prompt(q, prompt_variant="dataset")
        assert "Value Explanation:" in prompt

    def test_contains_question_resolution_dates_plural(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current GDP value",
            resolution_dates=["2024-07-01", "2024-08-01"],
        )
        prompt = _build_prompt(q, prompt_variant="dataset")
        assert "Question resolution dates:" in prompt

    def test_contains_answer_placeholder(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current GDP value",
            resolution_dates=["2024-07-01"],
        )
        prompt = _build_prompt(q, prompt_variant="dataset")
        assert "Answer:" in prompt


class TestRemovedCustomInstructions:
    def test_no_think_step_by_step_in_market_prompt(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "Think step-by-step" not in prompt

    def test_no_think_step_by_step_in_fv_prompt(self) -> None:
        q = _make_question(
            source="metaculus",
            freeze="2024-06-15",
            freeze_datetime_value=0.73,
        )
        prompt = _build_prompt(q, prompt_variant="zero-shot-fv")
        assert "Think step-by-step" not in prompt

    def test_no_think_step_by_step_in_dataset_prompt(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="val",
            resolution_dates=["2024-07-01"],
        )
        prompt = _build_prompt(q, prompt_variant="dataset")
        assert "Think step-by-step" not in prompt

    def test_no_probability_output_in_market_prompt(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "Probability:" not in prompt

    def test_no_probability_output_in_fv_prompt(self) -> None:
        q = _make_question(
            source="metaculus",
            freeze="2024-06-15",
            freeze_datetime_value=0.73,
        )
        prompt = _build_prompt(q, prompt_variant="zero-shot-fv")
        assert "Probability:" not in prompt

    def test_no_probability_output_in_dataset_prompt(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="val",
            resolution_dates=["2024-07-01"],
        )
        prompt = _build_prompt(q, prompt_variant="dataset")
        assert "Probability:" not in prompt


class TestTodayDateAlignment:
    def test_uses_forecast_due_date_when_available(self) -> None:
        q = _make_question(freeze="2024-06-15")
        q = q.model_copy(update={"forecast_due_date": "2024-06-25"})
        prompt = _build_prompt(q)
        assert "2024-06-25" in prompt
        assert "Today's Date: 2024-06-25" in prompt

    def test_falls_back_to_freeze_datetime(self) -> None:
        q = _make_question(freeze="2024-06-15")
        prompt = _build_prompt(q)
        assert "Today's Date: 2024-06-15" in prompt
