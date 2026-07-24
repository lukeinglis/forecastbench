# ForecastBench Results

## Current Best: 55.9% Overall Index (3-round avg)

Model: claude-sonnet-4.6 on Vertex AI (europe-west1)
Leaderboard target: 60.2%

### Configuration
- Events (acled, wikipedia): scratchpad prompt + thinking enabled
- Market (metaculus, polymarket, manifold, infer): freeze-value prompt, temp=0.3
- Timeseries (fred, dbnomics, yfinance): zero-shot prompt, temp=0.3
- Multi-horizon: enabled | Ensemble: disabled (N=1)

### Per-Source Scores

| Source | Score | Category | Config |
|--------|-------|----------|--------|
| infer | 79.3% | Market | Freeze values |
| acled | 72.6% | Event | Scratchpad+thinking |
| polymarket | 73.5% | Market | Freeze values |
| manifold | 68.3% | Market | Freeze values |
| wikipedia | 66.1% | Event | Scratchpad+thinking |
| metaculus | 58.6% | Market | Freeze values |
| fred | 50.8% | Timeseries | Zero-shot |
| yfinance | 48.1% | Timeseries | Zero-shot |
| dbnomics | 47.1% | Timeseries | Zero-shot |

### Improvement from Baseline

| Milestone | Overall | Key Change |
|-----------|---------|------------|
| Baseline (Sonnet 4@20250514) | 52.6% | Starting point |
| +Thinking, scratchpad, freeze values | 54.7% | PRs #63-65 |
| +Sonnet 4.6, market thinking fix | 55.9% | PRs #68-69, #72 |

### Reproduce
```
uv run python eval.py --agent baseline --round 2026-07-05-llm
```

### Remaining Gap
The 4.3pt gap to leaderboard (60.2%) is in timeseries sources (fred/dbnomics/yfinance at 47-51%). These are binary threshold comparisons where the model has the current value but can't reliably predict short-term direction. See issue #62 for full analysis.
