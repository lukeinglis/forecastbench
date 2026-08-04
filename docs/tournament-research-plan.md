# Tournament Track Research Plan

**Created:** 2026-08-03
**Goal:** Optimize tournament track forecasting performance through systematic research
**Baseline:** Re-establish on the corrected pipeline (post PR #102 scoring/resolution fix)

## Context

The pipeline was corrected in July 2026 (PR #102 fixed resolution data, PR #110 fixed overall score formula). Previous experimental results were collected against the broken pipeline and should not be trusted. This research plan starts clean.

Two competition tracks are now supported (`--track baseline|tournament`):
- **Baseline:** zero-shot prompt only, no tools/ensemble/RAG/calibration
- **Tournament:** all features allowed (search, ensemble, RAG, calibration, multi-model, agentic)

All research targets the tournament track.

---

## Research Streams

### Stream 1: Per-Model Performance Analysis
**Issue:** #114
**Branch:** None (analysis only)
**Phase:** 1 (start immediately)
**Dependencies:** None
**Status:** Not started

Run multiple models through the corrected pipeline. Understand per-source strengths. This informs all downstream streams — which models for which sources, where we're weakest, what to optimize.

**Models to test:**
| Model | Env var | Status | Cost/question |
|-------|---------|--------|--------------|
| Claude Sonnet 4.6 | `vertex_ai/claude-sonnet-4-6` | Current default | |
| GPT-4o | `openai/gpt-4o` | Available | |
| GPT-5-mini | `openai/gpt-5-mini` | Available | |
| o3-mini | `openai/o3-mini` | Available | |
| Gemini | TBD | TBD | |

**Results matrix (Brier Index by model x source):**
| Model | Overall | acled | dbnomics | fred | wikipedia | yfinance | metaculus | polymarket | manifold | infer |
|-------|---------|-------|----------|------|-----------|----------|-----------|------------|----------|-------|
| Sonnet 4.6 | | | | | | | | | | |
| GPT-4o | | | | | | | | | | |
| GPT-5-mini | | | | | | | | | | |
| o3-mini | | | | | | | | | | |

---

### Stream 2: Web Search Integration
**Issues:** #111, #112
**Branch:** `research/web-search`
**Phase:** 1 (start immediately)
**Dependencies:** None
**Status:** Not started

Add web search capability during forecasting. Two evaluation strategies:

| Approach | Issue | Method | Feedback speed |
|----------|-------|--------|---------------|
| Date-restricted backtesting | #111 | Search with date filters ≤ forecast_due_date | Immediate |
| Forward-looking evaluation | #112 | Forecast unresolved rounds, score as they resolve | Weeks/months |

**Key question:** Does web search provide information the model doesn't already have from training?

**Results:**
| Date | Experiment | Brier Score (with) | Brier Score (without) | Delta | Notes |
|------|-----------|-------------------|----------------------|-------|-------|
| | | | | | |

---

### Stream 3: Crowd Forecast Integration
**Issue:** #113
**Branch:** `research/crowd-forecasts`
**Phase:** 1 (start immediately)
**Dependencies:** None (API research prerequisite built into issue)
**Status:** Not started

Anchor market question forecasts on historical crowd/market probabilities from 10 days before forecast_due_date (matching official ForecastBench approach).

**Prerequisite research:** Verify historical point-in-time data availability:
| Platform | API supports historical lookup? | Notes |
|----------|-------------------------------|-------|
| Polymarket | TBD | |
| Metaculus | TBD | |
| Manifold | TBD | |
| Infer | TBD | |

**Results:**
| Date | Experiment | Market Brier (with) | Market Brier (without) | Delta | Notes |
|------|-----------|-------------------|----------------------|-------|-------|
| | | | | | |

---

### Stream 4: RAG Improvements
**Issue:** #119
**Branch:** `research/rag-improvements`
**Phase:** 1 (start immediately)
**Dependencies:** None, but findings feed into Stream 5 Lane B
**Status:** Not started

**Areas to test:**
| Area | Experiment | Result | Notes |
|------|-----------|--------|-------|
| Retrieval relevance | Manual inspection of retrieved data vs question | | |
| Context window size | 30d vs 1y vs 5y historical data | | |
| Context formatting | Table vs summary stats vs trend description | | |
| Per-source RAG toggle | Enable RAG for specific sources only | | |
| Additional data sources | World Bank, IMF, BLS | | |

**Results:**
| Date | Source | RAG Config | Brier Score (with) | Brier Score (without) | Delta | Notes |
|------|--------|-----------|-------------------|----------------------|-------|-------|
| | | | | | | |

---

### Stream 5: Source-Specific Optimization
**Issues:** #115 (market), #116 (timeseries), #117 (ACLED/Wikipedia)
**Branches:** `research/source-optimization-market`, `research/source-optimization-timeseries`, `research/source-optimization-acled-wikipedia`
**Phase:** 2 (after Stream 1 results)
**Dependencies:** Informed by Stream 1 per-model analysis
**Status:** Not started

Three parallel lanes optimizing strategies per source type.

#### Lane A: Market Sources (#115)
Sources: metaculus, polymarket, manifold, infer

| Date | Source | Experiment | Brier Score | Baseline | Delta | Notes |
|------|--------|-----------|-------------|----------|-------|-------|
| | | | | | | |

#### Lane B: Timeseries Dataset Sources (#116)
Sources: fred, dbnomics, yfinance

| Date | Source | Experiment | Brier Score | Baseline | Delta | Notes |
|------|--------|-----------|-------------|----------|-------|-------|
| | | | | | | |

#### Lane C: ACLED & Wikipedia (#117)
Sources: acled, wikipedia

| Date | Source | Experiment | Brier Score | Baseline | Delta | Notes |
|------|--------|-----------|-------------|----------|-------|-------|
| | | | | | | |

---

### Stream 6: Advanced Calibration
**Issue:** #118
**Branch:** `research/calibration`
**Phase:** 3 (after Stream 5 source optimization)
**Dependencies:** Should run AFTER Stream 5 — calibrate optimized forecasts, not raw ones
**Status:** Not started

**Methods to test:**
| Method | Params | Training data needed | Status | Result |
|--------|--------|---------------------|--------|--------|
| Platt scaling (current) | 2 per source | Low | Implemented | |
| Isotonic regression | Non-parametric | High | | |
| Beta calibration | 3 | Medium | | |
| Temperature scaling | 1 | Low | | |
| Per-source calibration | 2 per source | Medium | | |

**Results:**
| Date | Method | Brier Score (calibrated) | Brier Score (raw) | Lift | Notes |
|------|--------|------------------------|-------------------|------|-------|
| | | | | | |

---

### Stream 7: Agentic Forecasting
**Issue:** #120
**Branch:** `research/agentic-forecasting`
**Phase:** 4 (after Streams 2, 3, 4 provide tools)
**Dependencies:** Benefits from Streams 2, 3, 4 (search, crowd data, RAG become agent tools)
**Status:** Not started

**Results:**
| Date | Experiment | Tools used | Brier Score | Single-call baseline | Delta | Cost/question | Notes |
|------|-----------|-----------|-------------|---------------------|-------|--------------|-------|
| | | | | | | | |

---

### Stream 8: Question Decomposition
**Issue:** #121
**Branch:** `research/question-decomposition`
**Phase:** 4 (after Stream 5 identifies which question types benefit)
**Dependencies:** Informed by Stream 5 source-specific research
**Status:** Not started

**Results:**
| Date | Source/Type | Method | Brier Score | Single-call baseline | Delta | Notes |
|------|-----------|--------|-------------|---------------------|-------|-------|
| | | | | | | |

---

## Execution Order

```
Phase 1 (parallel, start immediately):
  Stream 1: Per-model analysis ──────────────────┐
  Stream 2: Web search integration               │
  Stream 3: Crowd forecast API research           │
  Stream 4: RAG improvements                      │
                                                  │
Phase 2 (after Stream 1 results):                 │
  Stream 5: Source-specific optimization ◄────────┘
    Lane A: Market sources
    Lane B: Timeseries sources
    Lane C: ACLED/Wikipedia
                         │
Phase 3 (after Stream 5):│
  Stream 6: Calibration ◄┘
                         │
Phase 4 (after Streams 2,3,4,6):
  Stream 7: Agentic forecasting
  Stream 8: Question decomposition
```

## Cross-Stream Results Tracker

**Overall tournament score progression:**
| Date | Config description | Overall | Dataset | Market | Notes |
|------|-------------------|---------|---------|--------|-------|
| | Baseline (Sonnet 4.6, raw, no features) | | | | Re-establish on corrected pipeline |
| | | | | | |

## Decision Log

Record key decisions and pivots here as research progresses.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-03 | Start clean — do not trust pre-PR#102 experimental results | Pipeline had resolution data bug corrupting 87% of dataset scoring |
| 2026-08-03 | 9 research branches to keep main clean | Experiments should prove value before merging |
| 2026-08-03 | Calibration after source optimization | Calibrating bad forecasts just makes well-calibrated bad forecasts |
| 2026-08-04 | Reordered streams to match execution flow | Stream numbers now correspond to execution priority |
