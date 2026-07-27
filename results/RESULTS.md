# ForecastBench Backtest Results

## All-Rounds Baseline (July 22, 2026)

**Model:** claude-sonnet-4-20250514 (via Vertex AI)
**Parameters:** temperature=0, max_tokens=2000
**Rounds:** 33 (2024-07-21 through 2026-07-19)
**Total questions scored:** 99,768 (95,768 dataset + 4,000 market)

### Scores

| Metric | Our Backtest | Leaderboard (sonnet-4) | Gap |
|--------|-------------|------------------------|-----|
| Overall Index | 52.6% | 60.3% | -7.7 pts |
| Dataset Index | 52.6% | 59.1% | -6.5 pts |
| Market Index | 52.8% | 61.5% | -8.7 pts |
| Brier Overall | 0.225 | 0.141 | +0.084 |
| Brier Dataset | 0.225 | 0.168 | +0.057 |
| Brier Market | 0.222 | 0.148 | +0.074 |

### Per-Source Breakdown

| Source | Type | Brier | Index | N |
|--------|------|-------|-------|---|
| acled | dataset | 0.155 | 60.7% | 20,216 |
| wikipedia | dataset | 0.185 | 57.0% | 18,000 |
| infer | market | 0.185 | 57.0% | 303 |
| metaculus | market | 0.199 | 55.4% | 435 |
| manifold | market | 0.223 | 52.8% | 816 |
| polymarket | market | 0.231 | 51.9% | 2,446 |
| yfinance | dataset | 0.253 | 49.7% | 19,896 |
| dbnomics | dataset | 0.261 | 49.0% | 17,704 |
| fred | dataset | 0.274 | 47.7% | 19,952 |

### Single-Round Comparison (2026-07-19-llm)

| Metric | Single Round | All Rounds | Delta |
|--------|-------------|------------|-------|
| Overall Index | 49.7% | 52.6% | +2.9 pts |
| Dataset Index | 49.6% | 52.6% | +3.0 pts |
| Market Index | 85.7% (N=4) | 52.8% (N=4,000) | -32.9 pts |
| N | 1,721 | 99,768 | +98,047 |

Single-round market score (85.7%) was noise on 4 questions. All-rounds market (52.8%, N=4,000) is the real number.

### Parity Audit Status

Pipeline verified against official ForecastBench source (July 21, 2026):
- Prompt templates: exact match (SHA-256 verified)
- Temperature: 0 (matches official)
- Max tokens: 2000 (matches official)
- Question text substitution: matches official
- Resolution date format: matches official
- Scoring formula: matches official
- Source classification: matches official

Remaining gap (7.7 pts) likely due to:
1. Model-specific options in official private infrastructure (e.g., extended thinking)
2. Extraction model difference (gpt-5-mini vs gpt-4o-mini)
3. Potential differences in how Vertex AI serves Sonnet 4 vs direct Anthropic API

### Result Files

- All-rounds: results/20260722T134347Z_vertex_ai_claude-sonnet-4_20250514.json
- Single-round (2026-07-19): results/20260721T144555Z_unknown_2026-07-19-llm.json
