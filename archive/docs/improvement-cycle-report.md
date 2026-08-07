# ForecastBench Improvement Cycle Report

**Date:** July 28-30, 2026
**Target:** 60.0% Brier Index (leaderboard top-5)

## Competition Simulation Results

All configurations scored with the same methodology: Platt scaling calibration fit on 101,625 historical forecasts with the test round (2026-07-05) excluded. This simulates how calibration would work in the live competition — you fit on past data and apply to the new unseen round.

| # | Model(s) | Method | Post-processing | Raw | Best | Improvement | Notes |
|---|----------|--------|----------------|-----|------|-------------|-------|
| 1 | Sonnet4 + GPT-4o + o3-mini + GPT-5-mini | Multi-model router | Platt | 53.7% | **56.9%** | +3.2 | Per-source model routing |
| 2 | o3-mini | Single model | Platt | 49.1% | **56.7%** | +7.6 | Cheapest model, biggest calibration lift |
| 3 | Sonnet 4 | Hybrid (no dbnomics belief) | Hybrid cal | 52.2% | **55.9%** | +3.7 | Belief state for fred/yfinance/acled |
| 4 | GPT-5-mini | Single model | Platt | 48.9% | **55.9%** | +7.0 | Zero-shot prompt |
| 5 | Sonnet 4 | Hybrid (belief+dbnomics) | Hybrid cal | 49.3% | **55.9%** | +6.6 | Adding dbnomics belief hurt raw by 2.9pts |
| 6 | Multi-model | Multi + source prompts | Platt | 52.4% | **55.9%** | +3.5 | Domain prompts hurt raw by 1.3pts vs #1 |
| 7 | o3 | Single model | Platt | 49.4% | **55.7%** | +6.3 | 5x slower than o3-mini, no improvement |
| 8 | GPT-5 | Single model | Platt | 49.7% | **55.2%** | +5.5 | Partial (980/1693), too slow to complete |
| 9 | GPT-4o | Single model | Hybrid cal | 49.5% | **55.0%** | +5.5 | Worst after calibration |

**Target: 60.0% | Gap from #1: 3.1pts**

### Post-processing definitions

- **Platt scaling**: Learned per-source correction `calibrated = logistic(a * logit(p) + b)`, fit on 101K historical forecasts
- **Hybrid cal**: Platt for events/markets + replace timeseries predictions entirely with historical base rate

### Key observations

- **Multi-model routing is the only approach that improves raw scores** (+4.6pts over average single model). Every other technique scores 49-50% raw.
- **Calibration is the dominant lever** — it adds 3-8pts to every configuration. But it can only correct consistent biases, not improve the model's ability to distinguish between questions.
- **o3-mini gets the biggest calibration lift** (+7.6pts) despite the lowest raw score. Its predictions are consistently biased in ways Platt can correct.
- **Multi-model has the smallest calibration lift** (+3.2pts) because its raw predictions are already better calibrated from using different models per source.
- **The gap between #1 and #9 is only 1.9pts** after calibration. Calibration compresses the differences.
- **o3 (full reasoning) does not beat o3-mini** — reasoning doesn't help forecasting but costs 5x more.
- **GPT-5 is impractically slow** — 2024 seconds per question, couldn't complete a single round.

## Multi-Model Routing Table

The multi-model router (`--agent multi`) selects the best LLM for each question source:

| Source | Model | Raw Score | Why This Model Wins |
|--------|-------|-----------|-------------------|
| acled | Claude Sonnet 4 | 70.3% | Extended thinking helps event reasoning |
| wikipedia | o3-mini | 65.0% | Fast reasoning, good on factual questions |
| dbnomics | GPT-4o | 50.6% | Uniquely good at temperature predictions (+12pts over next best) |
| fred | GPT-5-mini | 45.8% | Slight edge on macro indicators, very cheap |
| yfinance | o3-mini | 50.5% | Best on stock price threshold questions |
| manifold | GPT-5-mini | 72.2% | Strong on prediction market questions |
| polymarket | GPT-5-mini | 53.4% | — |
| metaculus/infer | Claude Sonnet 4 | — | Default fallback |

## What We Tried That Didn't Work

Full details in `docs/feature-graveyard.md`. Summary:

| Experiment | Result | Lesson |
|-----------|--------|--------|
| Dbnomics belief state | -10.8pts raw | Iterative reasoning makes temperature predictions worse |
| Source-specific prompts | -1.3pts raw | Domain-expert framing introduces anchoring bias |
| Training base rate hints | Neutral | Prompt-injected hints don't change model behavior |
| Per-horizon calibration | Overfits | 40 params doesn't generalize across rounds |
| Crowd/market adjustment | No impact | Too few market questions per round |
| o3 (full reasoning) | No improvement over o3-mini | Reasoning doesn't help forecasting |
| GPT-5-pro | Too slow | Impractical for evaluation |
| RAG historical data | No value | Questions already have freeze values |
| Ensemble averaging | Neutral after calibration | Calibration absorbs diversity benefit |
| Horizon/confidence dampening | Redundant | Calibration does the same thing |

## What Calibration Does

The model predicts a raw probability (e.g., 0.70 for a fred question). Calibration corrects this based on historical patterns — maybe when the model says 0.70 for fred, the answer is actually YES only 42% of the time. The correction is applied after the prediction is made; the model never sees it.

Platt scaling fits two parameters per source:
- **`a`** — controls extremity (a < 1 = compress toward 0.5, a > 1 = make more extreme)
- **`b`** — shifts the whole distribution up or down

For timeseries sources, the model's raw predictions carry almost no signal about individual questions. Calibration can correct the average bias but can't create discrimination that isn't there. This is why simply predicting the historical base rate ("65% of dbnomics questions resolve YES") performs comparably to any model's calibrated output.

## Gap Analysis

**Current best: 56.9% | Target: 60.0% | Gap: 3.1pts**

The gap is concentrated in timeseries sources (fred 48-53%, dbnomics 49-56%, yfinance 49-50% after calibration) which make up ~60% of all questions. Events (acled 68%, wikipedia 67%) and markets (61-65%) are already competitive.

### What leaderboard leaders do that we can't (in backtesting)

- **Agentic search** — All top teams retrieve real-time news/data. Removing search degrades scores 3.6x. We can't search without leaking future information.
- **Live crowd prices** — Top systems adjust forecasts toward current market consensus. We only have the freeze-date price.
- **Question decomposition** — Break questions into sub-questions and search for each. Requires search.

### What could still help

- **More calibration training data** — Run more rounds to build up calibration history (+0.5-1pt estimated)
- **Model ensembling** — Call 2-3 models per question and aggregate instead of routing to one (+0.5-1pt)
- **Difficulty adjustment** — ForecastBench uses difficulty-adjusted scoring; our backtesting uses raw. This could shift scores by 1-2pts.

## Cost per Round

| Agent | Time | Cost |
|-------|------|------|
| `--agent multi` (current best) | ~20 min | ~$5 |
| `--agent hybrid` (old Sonnet-only) | 2+ hours | ~$15 |
| `--agent baseline` (single model) | ~10 min | ~$3 |

## Available API Keys

- **OpenAI**: gpt-4o, gpt-5, gpt-5-mini, gpt-5-pro, o3, o3-mini, o3-pro, gpt-4.1
- **Vertex AI (GCP)**: Claude Sonnet 4, Gemini 2.5 Pro/Flash
