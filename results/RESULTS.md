# ForecastBench Backtest Results

> **Pipeline fix applied:** PR #102 (August 2, 2026) fixed multi-horizon resolution
> matching. All results below were generated with the corrected pipeline. Results from
> the broken pipeline are in `pre-fix-archived/`.

## Parity Verification (August 2, 2026)

Three models tested against official ForecastBench leaderboard baselines.

| Model | Leaderboard (adjusted) | Our Raw | Delta | Rounds | N |
|-------|----------------------|---------|-------|--------|---|
| Always 0.5 (dummy) | 50.0% | 50.0% | 0.0 | All | 152,073 |
| gpt-4o | 57.2% | 54.4% | -2.8pts | 4 | 6,663 |
| claude-sonnet-4-6 | 59.7% | 61.7% | +2.0pts | 2 | 2,279 |

The gpt-4o gap is explained by difficulty adjustment (leaderboard uses large peer pool;
our raw scoring doesn't adjust). Claude Sonnet 4.6 exceeds its leaderboard baseline
even with raw scoring.

### Per-Round Detail

**gpt-4o (FORECAST_MODEL=openai/gpt-4o)**

| Round | Overall BI | Dataset BI | Market BI | N |
|-------|-----------|------------|-----------|---|
| 2026-03-01 | 55.7% | 51.5% | 63.9% | 1,145 |
| 2025-08-31 | 53.1% | 51.1% | 59.6% | 2,975 |
| 2026-04-12 | 55.7% | 52.2% | 61.7% | 1,134 |
| 2025-11-09 | 55.2% | 48.3% | 77.0% | 1,409 |

**claude-sonnet-4-6 (default Vertex AI model)**

| Round | Overall BI | Dataset BI | Market BI | N |
|-------|-----------|------------|-----------|---|
| 2026-03-01 | 61.2% | 59.3% | 64.7% | 1,145 |
| 2026-04-12 | 62.2% | 61.9% | 62.8% | 1,134 |

## Result Files

### Valid (post-fix)
- `20260802T172918Z_dummy.json` — Always-0.5 baseline, all rounds
- `20260802T184753Z_openai_gpt-4o_2026-03-01-llm.json`
- `20260802T185310Z_openai_gpt-4o_2025-08-31-llm.json`
- `20260802T185341Z_openai_gpt-4o_2026-04-12-llm.json`
- `20260802T185410Z_openai_gpt-4o_2025-11-09-llm.json`
- `20260802T190655Z_unknown_2026-03-01-llm.json` — claude-sonnet-4-6
- `20260802T192429Z_unknown_2026-04-12-llm.json` — claude-sonnet-4-6

### Archived (pre-fix, invalid)
- `pre-fix-archived/` — Results from broken pipeline, see README there

### Archived (parity v0.1.2, invalid)
- `archive/parity-v0.1.2-23rounds/` — 24 result files generated 2026-08-07 with
  forecastbench-parity v0.1.2. Archived because v0.1.2 had broken multi-horizon
  resolution that silently dropped horizons >0, causing data loss on multi-horizon
  questions. Fixed in parity v0.2.0 which corrects `join_resolved_questions` to
  properly expand and match all horizons. These results should not be used for
  analysis or comparison — regenerate with v0.2.0.
