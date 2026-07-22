# ForecastBench

## Project
- Python 3.11+ backtest harness for ForecastBench forecasting evaluation
- Flat file structure (no package subdirectories) - all modules at project root
- Pydantic v2 for schema validation, requests for HTTP
- pytest + hypothesis for testing, ruff for linting, mypy for type checking

## Commands
- `uv run pytest` to run tests
- `uv run ruff check .` to lint
- `uv run mypy --ignore-missing-imports *.py` to type check
- `uv run python eval.py --agent dummy` to run dummy forecaster (default)
- `uv run python eval.py --agent baseline` to run LLM baseline agent
- `uv run python eval.py --agent baseline --raw` to run without difficulty adjustment
- `uv run python eval.py --agent baseline --prompt default` to use scratchpad for dataset, freeze values for market (this is the default)
- `uv run python eval.py --agent baseline --per-date` to disable multi-horizon batching (multi-horizon is default)
- `uv run python dummy_forecaster.py` to run dummy forecaster (shortcut)
- `uv run python baseline_agent.py` to run baseline LLM agent (shortcut)
- `FORECAST_MODEL=vertex_ai/claude-sonnet-4@20250514 uv run python eval.py --agent baseline` to run with Vertex AI
- `FORECAST_MODEL=openai/gpt-4o uv run python eval.py --agent baseline` to run with alternate model
- `uv run python analyze.py --compare` to compare all saved results
- `uv run python submit.py assemble --org ORG --model MODEL --model-org ORG --result results/FILE.json` to build submission
- `uv run python submit.py validate submissions/FILE.json` to validate coverage

## Architecture
- **fetch_data.py** - Fetches question sets and resolutions from forecastbench-datasets GitHub repo
- **score.py** - Brier score/index calculation with dataset/market separation
- **eval.py** - CLI entrypoint with structural held-out split
- **dummy_forecaster.py** - Baseline forecaster (always predicts 0.5)
- **cutoff.py** - Chronological data cutoff enforcement (CutoffEnvironment, CutoffContext)
- **baseline_agent.py** - LLM baseline forecaster using litellm (zero-shot superforecaster prompt)
- **analyze.py** - Error analysis, calibration, bias detection, and results comparison
- **submit.py** - Submission assembly, coverage validation, GCS upload
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
- FORECAST_MODEL env var selects LLM provider/model (default: vertex_ai/claude-sonnet-4@20250514). Vertex AI ADC tokens auto-refresh.
- FORECAST_THINKING env var enables/disables extended thinking (default: true). FORECAST_MAX_TOKENS sets max tokens (default: 16384).
- Multi-horizon forecasting is enabled by default; use --per-date to forecast each resolution date separately
- Vertex AI auth via `gcloud auth application-default login`, project: itpc-gcp-product-all-claude
- Baseline agent always returns valid [0, 1] float, never raises
- Results saved to results/ directory as JSON (auto-persisted after each eval run)
- Difficulty adjustment activates automatically when 2+ results exist in results/
- Use --raw flag to disable difficulty adjustment
- MARKET_SOURCES defined in fetch_data.py, imported by score.py, eval.py, submit.py
- Submissions staged in submissions/ directory with ForecastBench file naming
