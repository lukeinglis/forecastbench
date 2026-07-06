# ForecastBench

## Project
- Python 3.11+ backtest harness for ForecastBench forecasting evaluation
- Flat file structure (no package subdirectories) - all modules at project root
- Pydantic v2 for schema validation, requests for HTTP
- pytest + hypothesis for testing, ruff for linting, mypy for type checking

## Commands
- `cd /Users/linglis/forecastbench && uv run pytest` to run tests
- `cd /Users/linglis/forecastbench && uv run ruff check .` to lint
- `cd /Users/linglis/forecastbench && uv run mypy --ignore-missing-imports *.py` to type check
- `cd /Users/linglis/forecastbench && uv run python eval.py` to run full eval pipeline
- `cd /Users/linglis/forecastbench && uv run python dummy_forecaster.py` to run dummy forecaster
- `cd /Users/linglis/forecastbench && uv run python baseline_agent.py` to run baseline LLM agent eval
- `FORECAST_MODEL=openai/gpt-4o uv run python baseline_agent.py` to run with alternate model

## Architecture
- **fetch_data.py** - Fetches question sets and resolutions from forecastbench-datasets GitHub repo
- **score.py** - Brier score/index calculation with dataset/market separation
- **eval.py** - CLI entrypoint with structural held-out split
- **dummy_forecaster.py** - Baseline forecaster (always predicts 0.5)
- **cutoff.py** - Chronological data cutoff enforcement (CutoffEnvironment, CutoffContext)
- **baseline_agent.py** - LLM baseline forecaster using litellm (zero-shot superforecaster prompt)
- **analyze.py** - Error analysis, calibration, and run result reporting
- **tests/** - pytest test suite

## Style
- Flat file layout at project root - NO package subdirectories
- Type hints on all function signatures
- Pydantic v2 models for data schemas
- Standard pyproject.toml (PEP 621), not Poetry

## Key Conventions
- Brier Index formula: (1 - sqrt(mean_brier_score)) * 100, applied AFTER averaging
- Held-out split is strictly temporal: most recent N question sets by forecast_due_date
- Missing forecasts default to 0.5 per ForecastBench rules
- Binary outcomes only: {0, 1}
- Questions classified as "market" (metaculus, polymarket, manifold, infer) vs "dataset"
- FORECAST_MODEL env var selects LLM provider/model (default: anthropic/claude-sonnet-4-20250514)
- Baseline agent always returns valid [0, 1] float, never raises
- Results saved to results/ directory as JSON
