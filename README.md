# forecastbench

Backtest harness for [ForecastBench](https://github.com/forecastingresearch/forecastbench-datasets) forecasting evaluation. Fetches question sets, runs forecasters, scores with Brier Index (including difficulty-adjusted scoring via two-way fixed-effects OLS), and assembles submissions.

## Quick start

```bash
uv run python eval.py --agent dummy       # run dummy forecaster (always 0.5)
uv run python eval.py --agent baseline     # run LLM baseline (needs API creds)
uv run python eval.py --agent baseline --raw  # without difficulty adjustment
uv run python analyze.py --compare         # compare all saved results
```

Set `FORECAST_MODEL` to use a different provider/model:
```bash
FORECAST_MODEL=vertex_ai/claude-sonnet-4@20250514 uv run python eval.py --agent baseline
FORECAST_MODEL=openai/gpt-4o uv run python eval.py --agent baseline
```

## What's here

| File | What it does |
|------|-------------|
| `eval.py` | CLI entrypoint. Runs forecasters, applies held-out split, saves results to `results/` |
| `fetch_data.py` | Fetches question sets and resolutions from forecastbench-datasets repo, caches locally |
| `score.py` | Brier score/index calculation, difficulty adjustment (two-way fixed-effects OLS), Murphy decomposition |
| `analyze.py` | Error analysis, calibration (ECE/MCE), bias detection, worst-question analysis, horizon breakdown, paired comparison |
| `baseline_agent.py` | LLM forecaster using litellm (zero-shot superforecaster prompt, sync + async) |
| `dummy_forecaster.py` | Always predicts 0.5 |
| `cutoff.py` | Chronological data cutoff enforcement for honest backtesting |
| `submit.py` | Assembles and validates ForecastBench submissions, optional GCS upload |
| `logging_config.py` | Structured logging (structlog) with run-level trace IDs |

## Analysis tools

```bash
uv run python analyze.py --compare                        # side-by-side result comparison
uv run python analyze.py --worst 10 results/FILE.json     # 10 highest-error questions
uv run python analyze.py --horizons results/FILE.json     # performance by resolution date
uv run python analyze.py --decompose results/FILE.json    # Murphy decomposition + ECE/MCE
uv run python analyze.py --versus results/A.json results/B.json  # paired statistical comparison
```

## Submission

```bash
uv run python submit.py assemble --org ORG --model MODEL --model-org ORG --result results/FILE.json
uv run python submit.py validate submissions/FILE.json
```

## Key design decisions

- **Difficulty adjustment** auto-enables when 2+ prior results exist in `results/`. Use `--raw` to disable. Run 1 is always unadjusted (logged clearly).
- **Held-out split** is strictly temporal: most recent N question sets by `forecast_due_date`. Zero overlap tested.
- **Missing forecasts** default to 0.5 per ForecastBench rules.
- **Per-question caching** avoids re-burning API calls across runs. Cache in `.cache/`.
- **Brier Index** = `(1 - sqrt(mean_brier_score)) * 100`, applied after averaging.
- Questions classified as "market" (metaculus, polymarket, manifold, infer) vs "dataset".

## Tests

```bash
uv run pytest                  # 221 tests
uv run ruff check .            # lint
uv run mypy --ignore-missing-imports --exclude eval/ --exclude tests/ .  # type check
```

## Auth

Vertex AI: `gcloud auth application-default login` (project: `itpc-gcp-product-all-claude`)
