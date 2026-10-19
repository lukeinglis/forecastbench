"""Regression tests for lab_forecaster prompt routing (H1)."""

from lab_forecaster import _format_question_text


def test_market_question_returned_unchanged() -> None:
    """Market questions bypass formatting entirely, even with format braces."""
    text = "Will BTC exceed {forecast_due_date}? Resolve on {resolution_date}."
    assert _format_question_text(text, "2025-06-01", is_dataset=False) == text


def test_dataset_question_formats_date() -> None:
    """Dataset templates get forecast_due_date substituted."""
    text = "Question as of {forecast_due_date}."
    assert (
        _format_question_text(text, "2025-06-01", is_dataset=True)
        == "Question as of 2025-06-01."
    )


def test_format_failure_falls_back_to_raw_text() -> None:
    """Unexpected placeholders or malformed specs fall back to the raw text."""
    key_error_text = "Question with {unknown_placeholder}."
    assert (
        _format_question_text(key_error_text, "2025-06-01", is_dataset=True)
        == key_error_text
    )

    value_error_text = "Question with bad spec {0:x}."
    assert (
        _format_question_text(value_error_text, "2025-06-01", is_dataset=True)
        == value_error_text
    )
