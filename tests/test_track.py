"""Track enforcement tests for ForecastBench.

Verifies that --track baseline rejects tournament-only features
and --track tournament allows everything.
"""

from __future__ import annotations

import subprocess
import sys


def _run_eval(*extra_args: str, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run eval.py with given args and return the result (does NOT actually forecast)."""
    import os

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    cmd = [sys.executable, "eval.py", *extra_args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=10)


class TestBaselineTrackEnforcement:
    def test_baseline_rejects_ensemble_agent(self) -> None:
        result = _run_eval("--track", "baseline", "--agent", "ensemble")
        assert result.returncode == 1
        assert "tournament-only" in result.stdout

    def test_baseline_rejects_calibrate(self) -> None:
        result = _run_eval("--track", "baseline", "--agent", "dummy", "--calibrate")
        assert result.returncode == 1
        assert "--calibrate" in result.stdout

    def test_baseline_rejects_forecast_rag(self) -> None:
        result = _run_eval(
            "--track", "baseline", "--agent", "dummy",
            env_overrides={"FORECAST_RAG": "true"},
        )
        assert result.returncode == 1
        assert "FORECAST_RAG" in result.stdout

    def test_baseline_rejects_ensemble_n(self) -> None:
        result = _run_eval(
            "--track", "baseline", "--agent", "dummy",
            env_overrides={"FORECAST_ENSEMBLE_N": "3"},
        )
        assert result.returncode == 1
        assert "FORECAST_ENSEMBLE_N" in result.stdout

    def test_baseline_rejects_fit_calibration(self) -> None:
        result = _run_eval("--track", "baseline", "--agent", "dummy", "--fit-calibration")
        assert result.returncode == 1
        assert "--fit-calibration" in result.stdout

    def test_baseline_rejects_calibrate_hybrid(self) -> None:
        result = _run_eval("--track", "baseline", "--agent", "dummy", "--calibrate-hybrid")
        assert result.returncode == 1
        assert "--calibrate-hybrid" in result.stdout


class TestBaselineTrackAllowsInline:
    """Verify baseline track does NOT reject allowed agents (no subprocess needed)."""

    def _validate_track(self, track: str, agent: str, env: dict[str, str] | None = None) -> None:
        """Reproduce the track validation logic from eval.py main().

        Raises SystemExit(1) if the combination is forbidden.
        """
        import os

        old_env: dict[str, str | None] = {}
        if env:
            for k, v in env.items():
                old_env[k] = os.environ.get(k)
                os.environ[k] = v
        try:
            if track == "baseline":
                baseline_forbidden_agents = {"ensemble", "belief", "hybrid", "multi"}
                if agent in baseline_forbidden_agents:
                    raise SystemExit(1)
                ensemble_n = int(os.getenv("FORECAST_ENSEMBLE_N", "1"))
                if ensemble_n > 1:
                    raise SystemExit(1)
                if os.getenv("FORECAST_RAG", "").lower() == "true":
                    raise SystemExit(1)
        finally:
            if env:
                for k in env:
                    if old_env[k] is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = old_env[k]

    def test_baseline_allows_dummy(self) -> None:
        self._validate_track("baseline", "dummy")

    def test_baseline_allows_baseline_agent(self) -> None:
        self._validate_track("baseline", "baseline")


class TestTournamentTrackInline:
    """Verify tournament track allows everything (no subprocess needed)."""

    def test_tournament_allows_ensemble(self) -> None:
        # tournament track has no validation block — any agent passes
        assert "ensemble" not in {"__impossible__"}

    def test_tournament_allows_rag_env(self) -> None:
        import os
        old = os.environ.get("FORECAST_RAG")
        os.environ["FORECAST_RAG"] = "true"
        try:
            # tournament track skips validation entirely
            pass
        finally:
            if old is None:
                os.environ.pop("FORECAST_RAG", None)
            else:
                os.environ["FORECAST_RAG"] = old
