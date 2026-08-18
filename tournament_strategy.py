"""Tournament track competitive landscape — structured external intelligence.

All data sourced from published papers, official leaderboards, and ForecastBench
documentation. No internal experiment results.
"""

from __future__ import annotations

import argparse
import enum
from datetime import date

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────


class Track(str, enum.Enum):
    BASELINE = "baseline"
    TOURNAMENT = "tournament"


class TechniqueCategory(str, enum.Enum):
    CALIBRATION = "calibration"
    ENSEMBLE = "ensemble"
    AGENTIC = "agentic"
    SEARCH = "search"
    DECOMPOSITION = "decomposition"
    CROWD = "crowd"
    PROMPT = "prompt"


class Complexity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class EvidenceQuality(str, enum.Enum):
    VALIDATED_IN_PAPER = "validated_in_paper"
    PUBLISHED_CLAIM = "published_claim"
    THEORETICAL = "theoretical"


class Severity(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


# ── Models ─────────────────────────────────────────────────────────────────


class CompetitionRules(BaseModel):
    scoring_formula: str = (
        "Brier Index = (1 - sqrt(mean_brier_score)) * 100, applied AFTER averaging"
    )
    missing_forecast_default: float = 0.5
    overall_score_formula: str = (
        "equal-weight average of dataset and market Brier scores: "
        "(dataset + market) / 2"
    )
    round_cadence: str = "bi-weekly"
    baseline_track_rules: str = (
        "Zero-shot prompt only. No tools, web search, ensemble, "
        "calibration, RAG, or fine-tuning."
    )
    tournament_track_rules: str = (
        "All enhancements permitted: tools, fine-tuning, ensembles, "
        "web search, RAG, calibration, multi-agent, external data, "
        "crowd anchoring."
    )


class Technique(BaseModel):
    name: str
    description: str
    category: TechniqueCategory
    expected_brier_delta_low: float = Field(ge=0.0)
    expected_brier_delta_high: float = Field(ge=0.0)
    complexity: Complexity
    cost_multiplier: float = Field(gt=0.0)
    evidence_source: str
    evidence_quality: EvidenceQuality
    github_issue: int | None = None
    tier: int = Field(ge=1, le=4)
    track: Track = Track.TOURNAMENT


class Competitor(BaseModel):
    name: str
    best_known_brier: float | None = None
    best_known_brier_index: float | None = None
    approach_summary: str
    source_url: str
    as_of_date: date


class PerformanceBenchmark(BaseModel):
    description: str
    metric_name: str
    value: float
    source: str


class Pitfall(BaseModel):
    description: str
    evidence_source: str
    severity: Severity


class TechniqueRoadmap(BaseModel):
    tier_1: list[str] = Field(description="Highest ROI, implement first")
    tier_2: list[str] = Field(description="Differentiation techniques")
    tier_3: list[str] = Field(description="Optimization techniques")
    tier_4: list[str] = Field(description="Defer unless needed")


# ── Data ───────────────────────────────────────────────────────────────────


COMPETITION_RULES = CompetitionRules()


TECHNIQUES: list[Technique] = [
    # Tier 1 — highest ROI
    Technique(
        name="Freeze value integration",
        description=(
            "Incorporate crowd forecast from 10 days before forecast_due_date "
            "as an anchor for LLM predictions."
        ),
        category=TechniqueCategory.CROWD,
        expected_brier_delta_low=6.0,
        expected_brier_delta_high=6.5,
        complexity=Complexity.LOW,
        cost_multiplier=1.0,
        evidence_source="arXiv:2409.19839",
        evidence_quality=EvidenceQuality.VALIDATED_IN_PAPER,
        tier=1,
    ),
    Technique(
        name="Ensemble aggregation (3+ models × 3 prompts)",
        description=(
            "Aggregate predictions across multiple models and prompt variants. "
            "Geometric mean of log odds outperforms median and simple average."
        ),
        category=TechniqueCategory.ENSEMBLE,
        expected_brier_delta_low=3.0,
        expected_brier_delta_high=5.0,
        complexity=Complexity.MEDIUM,
        cost_multiplier=3.0,
        evidence_source="arXiv:2409.19839",
        evidence_quality=EvidenceQuality.VALIDATED_IN_PAPER,
        tier=1,
    ),
    Technique(
        name="Beta-Bernoulli Calibrator (BBC)",
        description=(
            "Dual supervision calibrator using binary outcomes and crowd forecasts. "
            "Outperforms Platt scaling, isotonic regression, and fine-tuned models."
        ),
        category=TechniqueCategory.CALIBRATION,
        expected_brier_delta_low=2.0,
        expected_brier_delta_high=4.0,
        complexity=Complexity.MEDIUM,
        cost_multiplier=1.0,
        evidence_source="arXiv:2605.27668",
        evidence_quality=EvidenceQuality.VALIDATED_IN_PAPER,
        tier=1,
    ),
    # Tier 2 — differentiation
    Technique(
        name="BLF-style agentic system",
        description=(
            "Full agentic pipeline: tool-use loop (up to 10 steps), semi-structured "
            "belief state, 5-trial logit-space aggregation with hierarchical shrinkage, "
            "hierarchical Platt scaling with per-source intercepts, web search with "
            "date filtering. Achieves SOTA 73.3 Brier Index on 400 questions."
        ),
        category=TechniqueCategory.AGENTIC,
        expected_brier_delta_low=10.0,
        expected_brier_delta_high=15.0,
        complexity=Complexity.HIGH,
        cost_multiplier=10.0,
        evidence_source="arXiv:2604.18576v4",
        evidence_quality=EvidenceQuality.VALIDATED_IN_PAPER,
        tier=2,
    ),
    Technique(
        name="Web search with date filtering",
        description=(
            "Retrieve current information to address knowledge cutoff. "
            "Date-range filtering is critical for backtesting validity."
        ),
        category=TechniqueCategory.SEARCH,
        expected_brier_delta_low=5.0,
        expected_brier_delta_high=8.0,
        complexity=Complexity.MEDIUM,
        cost_multiplier=1.5,
        evidence_source="arXiv:2604.18576v4",
        evidence_quality=EvidenceQuality.VALIDATED_IN_PAPER,
        tier=2,
    ),
    Technique(
        name="Question decomposition / Fermi-ization",
        description=(
            "Break complex questions into sub-components. Targets LLMs' biggest "
            "weakness: combination questions show 0.054 Brier gap vs humans."
        ),
        category=TechniqueCategory.DECOMPOSITION,
        expected_brier_delta_low=3.0,
        expected_brier_delta_high=5.0,
        complexity=Complexity.MEDIUM,
        cost_multiplier=1.5,
        evidence_source="arXiv:2409.19839",
        evidence_quality=EvidenceQuality.VALIDATED_IN_PAPER,
        tier=2,
    ),
    Technique(
        name="Market-conditioned prompting",
        description=(
            "Treat market probability as Bayesian prior, instruct LLM to update. "
            "Better calibration than standard prompting."
        ),
        category=TechniqueCategory.CROWD,
        expected_brier_delta_low=1.0,
        expected_brier_delta_high=3.0,
        complexity=Complexity.MEDIUM,
        cost_multiplier=1.0,
        evidence_source="arXiv:2602.21229v2",
        evidence_quality=EvidenceQuality.VALIDATED_IN_PAPER,
        tier=2,
    ),
    Technique(
        name="Multi-trial aggregation with hierarchical shrinkage",
        description=(
            "Run multiple prediction trials and aggregate in logit space "
            "with hierarchical shrinkage to tame overconfident predictions."
        ),
        category=TechniqueCategory.ENSEMBLE,
        expected_brier_delta_low=2.0,
        expected_brier_delta_high=3.0,
        complexity=Complexity.MEDIUM,
        cost_multiplier=5.0,
        evidence_source="arXiv:2604.18576v4",
        evidence_quality=EvidenceQuality.VALIDATED_IN_PAPER,
        tier=2,
    ),
    Technique(
        name="Scratchpad / chain-of-thought prompting",
        description=(
            "Step-by-step reasoning before producing a probability estimate. "
            "Significantly improves performance over direct prompting."
        ),
        category=TechniqueCategory.PROMPT,
        expected_brier_delta_low=1.0,
        expected_brier_delta_high=3.0,
        complexity=Complexity.LOW,
        cost_multiplier=1.2,
        evidence_source="arXiv:2409.19839",
        evidence_quality=EvidenceQuality.VALIDATED_IN_PAPER,
        tier=2,
    ),
    Technique(
        name="Supervisor agent reconciliation",
        description=(
            "Pass ensemble results to a supervisor agent for correction and "
            "reconciliation. More effective than simple mean aggregation."
        ),
        category=TechniqueCategory.AGENTIC,
        expected_brier_delta_low=2.0,
        expected_brier_delta_high=3.0,
        complexity=Complexity.HIGH,
        cost_multiplier=2.0,
        evidence_source="arXiv:2511.07678v1",
        evidence_quality=EvidenceQuality.VALIDATED_IN_PAPER,
        tier=2,
    ),
    # Tier 3 — optimization
    Technique(
        name="Source-specific optimization",
        description=(
            "Tailor prompts per question source (FRED timeseries, ACLED "
            "geopolitical, Metaculus crowd, Polymarket event-driven)."
        ),
        category=TechniqueCategory.PROMPT,
        expected_brier_delta_low=1.0,
        expected_brier_delta_high=2.0,
        complexity=Complexity.HIGH,
        cost_multiplier=1.0,
        evidence_source="Project analysis (no published research found)",
        evidence_quality=EvidenceQuality.THEORETICAL,
        tier=3,
    ),
    Technique(
        name="Timeseries RAG",
        description=(
            "Retrieve historical timeseries data for dataset questions. "
            "Only helps approximately one-third of dataset questions."
        ),
        category=TechniqueCategory.SEARCH,
        expected_brier_delta_low=0.5,
        expected_brier_delta_high=1.0,
        complexity=Complexity.HIGH,
        cost_multiplier=1.3,
        evidence_source="Project architecture analysis",
        evidence_quality=EvidenceQuality.THEORETICAL,
        tier=3,
    ),
    Technique(
        name="Crowd forecast Bayesian prior",
        description=(
            "Use crowd forecasts as a Bayesian prior for LLM predictions. "
            "Must add independent value beyond the crowd signal."
        ),
        category=TechniqueCategory.CROWD,
        expected_brier_delta_low=2.0,
        expected_brier_delta_high=4.0,
        complexity=Complexity.MEDIUM,
        cost_multiplier=1.0,
        evidence_source="arXiv:2312.09081v1",
        evidence_quality=EvidenceQuality.PUBLISHED_CLAIM,
        tier=3,
    ),
    # Tier 4 — defer unless needed
    Technique(
        name="Fine-tuning",
        description=(
            "Fine-tune models on forecasting data. Post-hoc calibration "
            "(BBC) may work better with less effort."
        ),
        category=TechniqueCategory.CALIBRATION,
        expected_brier_delta_low=0.0,
        expected_brier_delta_high=0.0,
        complexity=Complexity.VERY_HIGH,
        cost_multiplier=10.0,
        evidence_source="arXiv:2409.19839",
        evidence_quality=EvidenceQuality.THEORETICAL,
        tier=4,
    ),
]

COMPETITORS: list[Competitor] = [
    Competitor(
        name="Superforecasters",
        best_known_brier=0.096,
        best_known_brier_index=67.8,
        approach_summary=(
            "Human expert forecasters. Rank first overall as of April 2026. "
            "Market questions: 80.3 Brier Index. Gap vs LLMs: p<0.001."
        ),
        source_url="https://arxiv.org/abs/2409.19839",
        as_of_date=date(2026, 4, 1),
    ),
    Competitor(
        name="BLF (Bayesian Linguistic Forecaster)",
        best_known_brier=None,
        best_known_brier_index=73.3,
        approach_summary=(
            "SOTA AI system on 400 questions. Tool-use loop, semi-structured "
            "belief state, 5-trial logit-space aggregation, hierarchical Platt "
            "scaling, web search. Only method significantly beating crowd baseline."
        ),
        source_url="https://arxiv.org/abs/2604.18576v4",
        as_of_date=date(2026, 4, 1),
    ),
    Competitor(
        name="Cassi AI",
        approach_summary=(
            "Top tournament system. Specific score undisclosed."
        ),
        source_url="https://www.forecastbench.org/leaderboard",
        as_of_date=date(2026, 4, 1),
    ),
    Competitor(
        name="xAI",
        approach_summary=(
            "Statistically indistinguishable from superforecaster-level accuracy."
        ),
        source_url="https://www.forecastbench.org/leaderboard",
        as_of_date=date(2026, 4, 1),
    ),
    Competitor(
        name="Google DeepMind",
        approach_summary=(
            "Statistically indistinguishable from superforecaster-level accuracy."
        ),
        source_url="https://www.forecastbench.org/leaderboard",
        as_of_date=date(2026, 4, 1),
    ),
    Competitor(
        name="metac-claude-4-5-sonnet+asknews",
        approach_summary=(
            "Ranked 33rd of 1130 humans (top 3%) in Spring 2026 Metaculus Cup."
        ),
        source_url="https://forum.effectivealtruism.org",
        as_of_date=date(2026, 6, 1),
    ),
    Competitor(
        name="Claude-3.5-Sonnet (baseline)",
        best_known_brier=0.123,
        best_known_brier_index=58.6,
        approach_summary=(
            "Baseline zero-shot performance on 1000-question set."
        ),
        source_url="https://arxiv.org/abs/2409.19839",
        as_of_date=date(2025, 9, 1),
    ),
    Competitor(
        name="GPT-4-Turbo (baseline)",
        best_known_brier=0.126,
        best_known_brier_index=57.0,
        approach_summary=(
            "Baseline zero-shot performance on 1000-question set."
        ),
        source_url="https://arxiv.org/abs/2409.19839",
        as_of_date=date(2025, 9, 1),
    ),
]

BENCHMARKS: list[PerformanceBenchmark] = [
    PerformanceBenchmark(
        description="Superforecaster Brier score on 200-question subset",
        metric_name="brier_score",
        value=0.096,
        source="arXiv:2409.19839",
    ),
    PerformanceBenchmark(
        description="Combination question gap: LLMs vs humans",
        metric_name="brier_gap",
        value=0.054,
        source="arXiv:2409.19839",
    ),
    PerformanceBenchmark(
        description="Freeze value impact on Brier score",
        metric_name="brier_improvement",
        value=0.014,
        source="arXiv:2409.19839",
    ),
    PerformanceBenchmark(
        description="BLF Brier Index (SOTA AI)",
        metric_name="brier_index",
        value=73.3,
        source="arXiv:2604.18576v4",
    ),
    PerformanceBenchmark(
        description="Superforecaster market questions Brier Index",
        metric_name="brier_index",
        value=80.3,
        source="arXiv:2409.19839",
    ),
]

PITFALLS: list[Pitfall] = [
    Pitfall(
        description=(
            "LLM overconfidence is systematic — verbalized probabilities skew "
            "confident regardless of actual accuracy."
        ),
        evidence_source="Nature (s41586-026-10549-w)",
        severity=Severity.CRITICAL,
    ),
    Pitfall(
        description=(
            "Temporal data leakage: 2-13% before mitigation, 0.6-3.7% after "
            "(TimeSPEC, TEMPO frameworks)."
        ),
        evidence_source="arXiv:2602.17234, arXiv:2605.18843",
        severity=Severity.CRITICAL,
    ),
    Pitfall(
        description=(
            "News retrieval DECREASED performance in ForecastBench paper. "
            "AIA Forecaster claims success with high-quality sources — "
            "source quality matters."
        ),
        evidence_source="arXiv:2409.19839, arXiv:2511.07678v1",
        severity=Severity.HIGH,
    ),
    Pitfall(
        description=(
            "Simple crowd forecast anchoring — BLF is the only method "
            "significantly beating crowd baseline, suggesting most methods "
            "just regurgitate crowd signals without adding independent value."
        ),
        evidence_source="arXiv:2604.18576v4",
        severity=Severity.HIGH,
    ),
    Pitfall(
        description=(
            "Unstructured context appending leads to context explosion in "
            "agentic loops — use semi-structured belief state instead."
        ),
        evidence_source="arXiv:2604.18576v4",
        severity=Severity.HIGH,
    ),
    Pitfall(
        description=(
            "Cost explosion in agentic workflows: 5-step agent uses 8x-15x "
            "tokens (not 5x) due to growing context."
        ),
        evidence_source="morphllm.com/llm-cost-optimization",
        severity=Severity.MEDIUM,
    ),
    Pitfall(
        description=(
            "Diminishing returns on premium models: 20x cost for ~4% accuracy "
            "improvement."
        ),
        evidence_source="launchdarkly.com/blog/llm-pricing-comparison",
        severity=Severity.MEDIUM,
    ),
]

ROADMAP = TechniqueRoadmap(
    tier_1=[t.name for t in TECHNIQUES if t.tier == 1],
    tier_2=[t.name for t in TECHNIQUES if t.tier == 2],
    tier_3=[t.name for t in TECHNIQUES if t.tier == 3],
    tier_4=[t.name for t in TECHNIQUES if t.tier == 4],
)


# ── Helpers ────────────────────────────────────────────────────────────────


def get_techniques_by_tier(tier: int) -> list[Technique]:
    return [t for t in TECHNIQUES if t.tier == tier]


def get_techniques_for_track(track: str) -> list[Technique]:
    return [t for t in TECHNIQUES if t.track.value == track]


def get_competitor_by_name(name: str) -> Competitor | None:
    for c in COMPETITORS:
        if c.name == name:
            return c
    query = name.lower()
    for c in COMPETITORS:
        if query in c.name.lower():
            return c
    return None


def get_roadmap() -> TechniqueRoadmap:
    return ROADMAP


def get_pitfalls_by_severity(severity: str) -> list[Pitfall]:
    return [p for p in PITFALLS if p.severity.value == severity]


# ── CLI ────────────────────────────────────────────────────────────────────


def _print_summary() -> None:
    print("ForecastBench Tournament Strategy")
    print("=" * 50)
    print(f"\nTechniques: {len(TECHNIQUES)}")
    print(f"Competitors: {len(COMPETITORS)}")
    print(f"Pitfalls: {len(PITFALLS)}")
    print("\nTier breakdown:")
    for tier in range(1, 5):
        techs = get_techniques_by_tier(tier)
        print(f"  Tier {tier}: {len(techs)} techniques")
        for t in techs:
            delta = f"+{t.expected_brier_delta_low}-{t.expected_brier_delta_high} pts"
            if t.expected_brier_delta_low == 0 and t.expected_brier_delta_high == 0:
                delta = "unknown impact"
            print(f"    - {t.name} ({delta}, {t.complexity.value} complexity)")


def _print_techniques_for_tier(tier: int) -> None:
    techs = get_techniques_by_tier(tier)
    if not techs:
        print(f"No techniques found for tier {tier}")
        return
    print(f"Tier {tier} Techniques")
    print("=" * 50)
    for t in techs:
        print(f"\n{t.name}")
        print(f"  Category: {t.category.value}")
        print(f"  Expected delta: +{t.expected_brier_delta_low}-{t.expected_brier_delta_high} pts")
        print(f"  Complexity: {t.complexity.value}, Cost: {t.cost_multiplier}x")
        print(f"  Evidence: {t.evidence_source} ({t.evidence_quality.value})")
        if t.github_issue:
            print(f"  Issue: #{t.github_issue}")


def _print_competitors() -> None:
    print("Competitor Profiles")
    print("=" * 50)
    for c in COMPETITORS:
        bi = f" (Brier Index: {c.best_known_brier_index})" if c.best_known_brier_index else ""
        print(f"\n{c.name}{bi}")
        print(f"  {c.approach_summary}")
        print(f"  Source: {c.source_url}")
        print(f"  As of: {c.as_of_date}")


def _print_pitfalls() -> None:
    print("Known Pitfalls")
    print("=" * 50)
    for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM]:
        pits = get_pitfalls_by_severity(sev.value)
        if pits:
            print(f"\n[{sev.value.upper()}]")
            for p in pits:
                print(f"  - {p.description}")
                print(f"    Source: {p.evidence_source}")


def _print_roadmap() -> None:
    rm = get_roadmap()
    print("Implementation Roadmap")
    print("=" * 50)
    for tier_num, tier_list in [
        (1, rm.tier_1),
        (2, rm.tier_2),
        (3, rm.tier_3),
        (4, rm.tier_4),
    ]:
        label = {
            1: "Highest ROI — implement first",
            2: "Differentiation",
            3: "Optimization",
            4: "Defer unless needed",
        }[tier_num]
        print(f"\nTier {tier_num} ({label}):")
        for name in tier_list:
            print(f"  - {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ForecastBench tournament competitive landscape"
    )
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4],
                        help="List techniques for a specific tier")
    parser.add_argument("--competitors", action="store_true",
                        help="List competitor profiles")
    parser.add_argument("--pitfalls", action="store_true",
                        help="List known pitfalls")
    parser.add_argument("--roadmap", action="store_true",
                        help="Print implementation roadmap")
    args = parser.parse_args()

    if args.tier:
        _print_techniques_for_tier(args.tier)
    elif args.competitors:
        _print_competitors()
    elif args.pitfalls:
        _print_pitfalls()
    elif args.roadmap:
        _print_roadmap()
    else:
        _print_summary()


if __name__ == "__main__":
    main()
