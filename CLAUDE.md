# ForecastBench

## Project
- Python 3.11+ backtest harness for ForecastBench forecasting evaluation
- Flat file structure (no package subdirectories) - all modules at project root
- Depends on forecastbench-parity package for scoring, submission, and question handling
- Pydantic v2 for schema validation, requests for HTTP
- pytest + hypothesis for testing, ruff for linting, mypy for type checking

## Commands
- `uv run pytest` to run tests
- `uv run ruff check .` to lint
- `uv run mypy --ignore-missing-imports --disable-error-code=attr-defined *.py` to type check
- `uv run python eval.py --agent dummy` to run dummy forecaster (default)
- `uv run python eval.py --agent lab` to run LLM lab forecaster
- `uv run python eval.py --agent lab --raw` to run without difficulty adjustment
- `uv run python eval.py --agent lab --per-date` to disable multi-horizon batching (multi-horizon is default)
- `uv run python dummy_forecaster.py` to run dummy forecaster (shortcut)
- `uv run python lab_forecaster.py` to run lab forecaster (shortcut)
- `FORECAST_MODEL=vertex_ai/claude-sonnet-4-6 uv run python eval.py --agent lab` to run with Vertex AI
- `FORECAST_MODEL=openai/gpt-4o uv run python eval.py --agent lab` to run with alternate model
- `uv run python analyze.py --compare` to compare all saved results
- `uv run python submit.py assemble --org ORG --model MODEL --model-org ORG --result results/FILE.json` to build submission
- `uv run python submit.py validate submissions/FILE.json` to validate coverage
- `uv run python verify_parity.py` to run structural and behavioral parity checks against upstream ForecastBench
- `uv run python verify_parity.py --score` to also compare scores against leaderboard
- `uv run python verify_parity.py --refresh` to clear cached upstream data first
- `uv run python check_staleness.py` to check if local main is behind remote

## Architecture
- **forecastbench-parity** (external package) - Frozen competition contract: scoring, submission, question handling
- **fetch_data.py** - Re-exports from forecastbench_parity.questions + constants
- **score.py** - Re-exports from forecastbench_parity.score
- **submit.py** - Re-exports from forecastbench_parity.submission + CLI entry point
- **eval.py** - CLI entrypoint with structural held-out split
- **dummy_forecaster.py** - Baseline forecaster (always predicts 0.5)
- **cutoff.py** - Chronological data cutoff enforcement (CutoffEnvironment, CutoffContext)
- **lab_forecaster.py** - LLM lab forecaster using litellm (zero-shot superforecaster prompt)
- **analyze.py** - Error analysis, calibration, bias detection, and results comparison
- **verify_parity.py** - Pipeline parity verifier (fetches live from upstream ForecastBench repo/leaderboard)
- **check_staleness.py** - Git staleness check to warn when local main is behind remote
- **tournament.py** - Tournament analysis with cost tracking, bootstrap comparisons, and round filtering
- **tests/** - pytest test suite
- **archive/** - Archived experiment files (calibrate, ensemble, hybrid, multi-model, belief, statistical, timeseries RAG)

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
- VERTEXAI_LOCATION env var sets the Vertex AI region (default: europe-west1).
- FORECAST_TEMPERATURE env var sets temperature (default: 0). FORECAST_MAX_TOKENS sets max tokens (default: 16384).
- Multi-horizon forecasting is enabled by default for all dataset sources. Use --per-date to force per-date calling for all sources.
- Vertex AI auth via `gcloud auth application-default login`, project: itpc-gcp-product-all-claude
- Lab forecaster always returns valid [0, 1] float, never raises
- Results saved to results/ directory as JSON (auto-persisted after each eval run)
- Difficulty adjustment activates automatically when 2+ results exist in results/
- Use --raw flag to disable difficulty adjustment
- MARKET_SOURCES defined in forecastbench-parity, re-exported via fetch_data.py
- Submissions staged in submissions/ directory with ForecastBench file naming
- Prompts, temperature, and methodology are all customizable per competition rules - only scoring/submission/questions are fixed

## Competition Rules
- Two tracks: baseline (no tools/search/ensemble/RAG/calibration) and tournament (all features allowed)
- Baseline track: zero-shot prompt only, no web access, no ensemble, no calibration, no RAG
- Tournament track: tools, search, ensemble, RAG, calibration, fine-tuning all permitted
- Scoring methodology is fixed — never modify score.py formulas
- Overall score = equal-weight average of dataset and market Brier scores: (dataset + market) / 2
- Missing forecasts default to 0.5 — never change this default
- Resolution pipeline must match upstream — changes to fetch_data.py resolution logic require running verify_parity.py and tests/test_compliance.py
- No data leakage — prompts cannot contain information from after forecast_due_date
- Submission format must pass submit.py validate

## Parity Repo Boundary — INVIOLABLE
- The `forecastbench-parity` repo (lukeinglis/forecastbench-parity) is a **protected external dependency**
- **NEVER** clone, checkout, modify, commit to, push to, tag, or interact with the parity repo in any way
- **NEVER** use `git clone`, `gh repo clone`, or any mechanism to obtain a working copy of the parity repo
- **NEVER** upgrade the forecastbench-parity pin in pyproject.toml without a reviewed PR approved by a human
- If parity bugs are found: document them in an issue on this repo or the parity repo, then stop. The human will fix them.
- The entire point of the separate repo is to prevent automation from touching competition-critical scoring code
- This boundary applies to ALL agents: CEO, Builder, Researcher, and any other factory agent
