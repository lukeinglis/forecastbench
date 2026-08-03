"""Tests for verify_parity.py."""

from __future__ import annotations

from verify_parity import (
    extract_template,
    check_prompt_templates,
    check_scoring_formula,
    check_missing_forecast_default,
    _strip_enhancements,
    _find_reference_model,
    _clean_model_slug,
)


SAMPLE_UPSTREAM_SOURCE = '''
ZERO_SHOT_MARKET_PROMPT = """
You are an expert superforecaster, familiar with the work of Tetlock and others. \
Make a prediction of the probability that the question will be resolved as true. \
You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. \
If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's Date: {today_date}

Question resolution date: {resolution_date}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
Do not output anything else.
Answer: {{ Insert answer here }}
"""
'''


class TestExtractTemplate:
    def test_extracts_triple_double_quote(self) -> None:
        result = extract_template(SAMPLE_UPSTREAM_SOURCE, "ZERO_SHOT_MARKET_PROMPT")
        assert result is not None
        assert "superforecaster" in result
        assert "{question}" in result

    def test_returns_none_for_missing_var(self) -> None:
        result = extract_template(SAMPLE_UPSTREAM_SOURCE, "NONEXISTENT_PROMPT")
        assert result is None

    def test_extracts_content_between_quotes(self) -> None:
        source = 'MY_VAR = """hello world"""'
        result = extract_template(source, "MY_VAR")
        assert result == "hello world"

    def test_single_quote_extraction(self) -> None:
        source = "MY_VAR = '''single quotes'''"
        result = extract_template(source, "MY_VAR")
        assert result == "single quotes"

    def test_multiline_content(self) -> None:
        source = 'MY_VAR = """line1\nline2\nline3"""'
        result = extract_template(source, "MY_VAR")
        assert result is not None
        assert "line1" in result
        assert "line3" in result


class TestStripEnhancements:
    def test_strips_statistical_baseline(self) -> None:
        text = (
            "Before\n"
            "Statistical baseline (FRED DFF): 42% probability\n"
            "Note: This is a simple statistical estimate. "
            "Use your judgment to adjust.\n"
            "After"
        )
        result = _strip_enhancements(text)
        assert "Statistical baseline" not in result
        assert "Before" in result
        assert "After" in result

    def test_strips_base_rate_hint(self) -> None:
        text = (
            "Before"
            "Historical context: Questions of this type from this data source "
            "have historically resolved to YES approximately 65% of the time. "
            "This base rate should inform your starting estimate, with adjustments "
            "based on the specific details of this question.\n\n"
            "After"
        )
        result = _strip_enhancements(text)
        assert "Historical context" not in result
        assert "Before" in result
        assert "After" in result

    def test_no_enhancement_passthrough(self) -> None:
        text = "Clean text with no enhancements"
        assert _strip_enhancements(text) == text


class TestCheckScoringFormula:
    def test_local_formula_passes(self) -> None:
        ok, msg = check_scoring_formula(None)
        assert ok
        assert "[PASS]" in msg
        assert "local only" in msg

    def test_leaderboard_cross_check_passes(self) -> None:
        rows = [
            {
                "Model": "test-model",
                "Brier Overall": "0.25",
                "Overall": "50.0",
            },
            {
                "Model": "perfect-model",
                "Brier Overall": "0.04",
                "Overall": "80.0",
            },
        ]
        ok, msg = check_scoring_formula(rows)
        assert ok
        assert "2 leaderboard entries" in msg

    def test_leaderboard_mismatch_fails(self) -> None:
        rows = [
            {
                "Model": "bad-model",
                "Brier Overall": "0.25",
                "Overall": "99.0",  # should be 50.0
            },
        ]
        ok, msg = check_scoring_formula(rows)
        assert not ok
        assert "[FAIL]" in msg

    def test_unparseable_rows_skipped(self) -> None:
        rows = [
            {"Model": "no-data", "Brier Overall": "N/A", "Overall": "N/A"},
        ]
        ok, msg = check_scoring_formula(rows)
        assert ok
        assert "local only" in msg


class TestCheckMissingForecastDefault:
    def test_missing_default_passes(self) -> None:
        ok, msg = check_missing_forecast_default()
        assert ok
        assert "[PASS]" in msg
        assert "0.5" in msg


class TestCheckPromptTemplates:
    def test_upstream_unavailable_warns(self) -> None:
        ok, msg = check_prompt_templates(None)
        assert ok
        assert "[WARN]" in msg

    def test_matching_templates_pass(self) -> None:
        import baseline_agent

        source_parts: list[str] = []
        for name in [
            "ZERO_SHOT_MARKET_PROMPT",
            "ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT",
            "ZERO_SHOT_DATASET_PROMPT",
            "FORECAST_EXTRACTION_PROMPT",
        ]:
            local_val = getattr(baseline_agent, name)
            source_parts.append(f'{name} = """\n{local_val}\n"""')

        fake_source = "\n\n".join(source_parts)
        ok, msg = check_prompt_templates(fake_source)
        assert ok
        assert "[PASS]" in msg
        assert "4/4" in msg


class TestCleanModelSlug:
    def test_vertex_ai_slug(self) -> None:
        assert _clean_model_slug("vertex_ai_claude-sonnet-4_20250514") == "claude-sonnet-4"

    def test_openai_slug(self) -> None:
        assert _clean_model_slug("openai_gpt-4o") == "gpt-4o"

    def test_simple_slug(self) -> None:
        assert _clean_model_slug("o3") == "o3"

    def test_at_style_slug(self) -> None:
        assert _clean_model_slug("vertex_ai_claude-sonnet-4@20250514") == "claude-sonnet-4"

    def test_date_style_slug(self) -> None:
        assert _clean_model_slug("openai_o3-2025-04-16") == "o3-2025-04-16"


class TestFindReferenceModel:
    def test_prefers_o3_as_fallback(self) -> None:
        rows = [
            {"Model": "gpt-4o-2024-05", "Overall": "55.0"},
            {"Model": "o3-2025-04-16", "Overall": "60.0"},
            {"Model": "claude-3-opus", "Overall": "52.0"},
        ]
        result = _find_reference_model(rows)
        assert result is not None
        model, score, is_fallback = result
        assert "o3" in model
        assert score == 60.0
        assert is_fallback is True

    def test_model_hint_matches_sonnet4_not_sonnet45(self) -> None:
        rows = [
            {"Model": "claude-sonnet-4-5-20250929-1024", "Overall": "61.3"},
            {"Model": "claude-sonnet-4-20250514", "Overall": "60.1"},
            {"Model": "o3-2025-04-16", "Overall": "62.2"},
        ]
        result = _find_reference_model(rows, model_hint="vertex_ai_claude-sonnet-4_20250514")
        assert result is not None
        model, score, is_fallback = result
        assert model == "claude-sonnet-4-20250514"
        assert score == 60.1
        assert is_fallback is False

    def test_model_hint_matches_gpt4o(self) -> None:
        rows = [
            {"Model": "o3-2025-04-16", "Overall": "60.0"},
            {"Model": "gpt-4o-2024-05-13", "Overall": "55.0"},
        ]
        result = _find_reference_model(rows, model_hint="openai_gpt-4o")
        assert result is not None
        model, score, is_fallback = result
        assert "gpt-4o" in model
        assert score == 55.0
        assert is_fallback is False

    def test_model_hint_no_match_falls_back(self) -> None:
        rows = [
            {"Model": "o3-2025-04-16", "Overall": "60.0"},
            {"Model": "gpt-4o-2024-05", "Overall": "55.0"},
        ]
        result = _find_reference_model(rows, model_hint="vertex_ai_gemini-pro_20250101")
        assert result is not None
        model, score, is_fallback = result
        assert "o3" in model
        assert is_fallback is True

    def test_falls_back_to_gpt4o(self) -> None:
        rows = [
            {"Model": "gpt-4o-2024-05", "Overall": "55.0"},
            {"Model": "llama-3", "Overall": "48.0"},
        ]
        result = _find_reference_model(rows)
        assert result is not None
        assert "gpt-4o" in result[0]
        assert result[2] is True

    def test_falls_back_to_first_parseable(self) -> None:
        rows = [
            {"Model": "some-model", "Overall": "45.0"},
        ]
        result = _find_reference_model(rows)
        assert result is not None
        assert result[0] == "some-model"
        assert result[1] == 45.0
        assert result[2] is True

    def test_empty_leaderboard(self) -> None:
        assert _find_reference_model([]) is None

    def test_all_unparseable(self) -> None:
        rows = [{"Model": "bad", "Overall": "N/A"}]
        assert _find_reference_model(rows) is None
