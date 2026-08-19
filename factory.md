# factory.md — ForecastBench

## Goal
Build and evolve a Python backtest harness for ForecastBench forecasting evaluation. The harness fetches question sets from the forecastbench-datasets GitHub repo, runs forecasters (dummy baseline and LLM-based agents), scores predictions using Brier score/index, and produces calibration and error analysis reports. Target: ForecastBench submission compatibility with multi-horizon dataset question support.

## Scope

### Mutable (files the factory MAY modify)
- `eval.py`
- `dummy_forecaster.py`
- `cutoff.py`
- `lab_forecaster.py`
- `analyze.py`
- `verify_parity.py`
- `tests/**`
- `pyproject.toml`

### Read-Only (files the factory MUST NOT modify)
- `fetch_data.py` — competition pipeline (data fetching + resolution matching)
- `score.py` — competition scoring (Brier score/index formulas)
- `submit.py` — competition submission format
- `gate/**` — outer-loop fitness function; editing this is score tampering
- `gate_rounds.json` — full rounds the gate scores, one run_eval call each
- `gate_baseline.json` — Brier Index the gate ladder is centered on
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
- Do not edit anything under `gate/`, `gate_rounds.json`, or `gate_baseline.json` — these define how your work is scored
- Do not add or remove questions from the pinned manifest
- `forecast() -> float` signature must be preserved (no dict/union return types)
- Composite cache keys use `_` separator (not `|`)
- **Do not clone, modify, commit, push, tag, or interact with the forecastbench-parity repo** — it is a protected dependency. The factory treats it as a read-only upstream package. If parity changes are needed, open an issue on the backtester describing what needs to change and stop. Do not attempt workarounds (cloning, temporary checkouts, git submodules, or any other mechanism to reach the parity repo).
- **Do not upgrade the forecastbench-parity dependency pin** in pyproject.toml without explicit human approval via a reviewed PR

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

## Project Eval

- name: brier_gate
  command: uv run pytest gate/ -q
  parse: exit_code
  weight: 4.0
  description: Graded Brier Index gate on the pinned question subsample

- name: uv_tests
  command: uv run pytest -q
  parse: exit_code
  weight: 2.0
  description: Run test suite via uv

- name: uv_lint
  command: uv run ruff check .
  parse: exit_code
  weight: 1.0
  description: Run ruff linter via uv

- name: uv_typecheck
  command: uv run mypy --ignore-missing-imports --exclude eval/ --exclude tests/ --exclude gate/ .
  parse: exit_code
  weight: 1.0
  description: Run mypy type checker via uv

## Eval Weights
- hygiene: 0.2
- growth: 0.3
- project: 0.5

## Target Branch
main
