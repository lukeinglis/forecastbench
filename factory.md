# factory.md — ForecastBench

## Goal
Build and evolve a Python backtest harness for ForecastBench forecasting evaluation. The harness fetches question sets from the forecastbench-datasets GitHub repo, runs forecasters (dummy baseline and LLM-based agents), scores predictions using Brier score/index, and produces calibration and error analysis reports. Target: ForecastBench submission compatibility with multi-horizon dataset question support.

## Scope

### Mutable (files the factory MAY modify)
- `fetch_data.py`
- `score.py`
- `eval.py`
- `dummy_forecaster.py`
- `cutoff.py`
- `baseline_agent.py`
- `analyze.py`
- `tests/**`
- `pyproject.toml`

### Read-Only (files the factory MUST NOT modify)
- `CLAUDE.md`
- `factory.md`
- `.github/**`
- `uv.lock`

## Guards
- Do not remove or weaken existing tests
- Do not introduce API keys, tokens, or credentials into the repo
- Do not change the Brier Index formula: `(1 - sqrt(mean_brier_score)) * 100`, applied AFTER averaging
- Do not change the flat file layout — all modules stay at project root, no package subdirectories
- Missing forecasts must default to 0.5 per ForecastBench rules
- Binary outcomes only: `{0, 1}`
- `forecast() -> float` signature must be preserved (no dict/union return types)
- Composite cache keys use `_` separator (not `|`)

## Eval

### Eval Command
```bash
uv run python eval/score.py
```

### Threshold
0.55

### Smoke Test
```bash
uv run pytest -x -q
```

## Eval Spec

### Dimensions

| Dimension     | Command                                    | Weight | Parser    | Source      |
|---------------|--------------------------------------------|--------|-----------|------------|
| tests         | `uv run pytest -v`                         | 0.417  | exit_code | discovered |
| lint          | `uv run ruff check .`                      | 0.250  | exit_code | discovered |
| type_check    | `uv run mypy ./`                           | 0.125  | exit_code | researched |
| coverage      | `uv run pytest --cov= --cov-report=term -q`| 0.125  | exit_code | researched |
| observability | (inline)                                   | 0.083  | json      | researched |

### Tier
discovered (confidence: 0.80, human_reviewed: true)

## Target Branch
main
