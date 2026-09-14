"""Tests for the gate's missing-forecast guard (gate/make_manifest.py).

The lab forecaster never raises, so a dead API (expired credentials, exhausted
credits, wrong routing) silently defaults every forecast to 0.5 and the round
scores a plausible ~50 Brier Index. On 2026-09-04 four rounds "passed" this
way with 100% missing forecasts. score_rounds() now fails loudly instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gate.make_manifest import MAX_MISSING_FRACTION  # noqa: E402


class _EvalResult:
    """Minimal stand-in for eval.run_eval's return: only .scoring is used."""

    def __init__(self, scoring: object) -> None:
        self.scoring = scoring


def _scoring(n_dataset: int, n_market: int, n_missing: int) -> Any:
    from score import ScoringResult

    return ScoringResult(
        dataset_brier=0.25,
        dataset_index=50.0,
        market_brier=0.25,
        market_index=50.0,
        overall_brier=0.25,
        overall_index=50.0,
        n_dataset=n_dataset,
        n_market=n_market,
        n_missing=n_missing,
    )


def _score_rounds_with(monkeypatch: pytest.MonkeyPatch, result: _EvalResult) -> Any:
    """Run score_rounds() on one fake round whose eval result is `result`."""
    import gate.make_manifest as mm

    async def fake_run_eval(**kwargs: object) -> _EvalResult:  # noqa: ANN003, ARG001
        return result

    import eval as ev

    monkeypatch.setattr(ev, "run_eval", fake_run_eval)
    return mm.score_rounds(["2026-03-01"])


def test_total_api_failure_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """100% missing (dead API) must raise, not score 50.0."""
    # n_dataset/n_market are scored rows; n_missing is everything that fell
    # through to the 0.5 default.
    result = _EvalResult(_scoring(n_dataset=684, n_market=123, n_missing=807))
    with pytest.raises(SystemExit, match="eval failure"):
        _score_rounds_with(monkeypatch, result)


def test_healthy_missing_fraction_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """<1% missing is normal for healthy runs and must not raise."""
    result = _EvalResult(_scoring(n_dataset=684, n_market=123, n_missing=3))
    mean_index, per_round = _score_rounds_with(monkeypatch, result)
    assert len(per_round) == 1
    assert mean_index == pytest.approx(50.0)


def test_threshold_sits_between_healthy_and_dead() -> None:
    """Healthy runs are <1% missing; total failure is 100%. The guard must
    sit well above healthy and well below total failure."""
    assert 0.02 < MAX_MISSING_FRACTION < 0.9
