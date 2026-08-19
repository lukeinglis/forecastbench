from __future__ import annotations

import subprocess
import sys

from tournament_strategy import (
    BENCHMARKS,
    COMPETITORS,
    COMPETITION_RULES,
    PITFALLS,
    TECHNIQUES,
    Complexity,
    EvidenceQuality,
    Severity,
    TechniqueCategory,
    get_competitor_by_name,
    get_pitfalls_by_severity,
    get_roadmap,
    get_techniques_by_tier,
    get_techniques_for_track,
)


class TestTechniqueDataIntegrity:
    def test_technique_count(self) -> None:
        assert len(TECHNIQUES) == 14

    def test_all_have_nonempty_descriptions(self) -> None:
        for t in TECHNIQUES:
            assert t.description.strip(), f"{t.name} has empty description"

    def test_all_have_valid_categories(self) -> None:
        for t in TECHNIQUES:
            assert isinstance(t.category, TechniqueCategory)

    def test_delta_ranges_valid(self) -> None:
        for t in TECHNIQUES:
            assert t.expected_brier_delta_low <= t.expected_brier_delta_high, (
                f"{t.name}: low {t.expected_brier_delta_low} > high {t.expected_brier_delta_high}"
            )

    def test_all_have_evidence_source(self) -> None:
        for t in TECHNIQUES:
            assert t.evidence_source.strip(), f"{t.name} has empty evidence_source"

    def test_all_have_valid_evidence_quality(self) -> None:
        for t in TECHNIQUES:
            assert isinstance(t.evidence_quality, EvidenceQuality)

    def test_all_have_valid_complexity(self) -> None:
        for t in TECHNIQUES:
            assert isinstance(t.complexity, Complexity)

    def test_tier_values_in_range(self) -> None:
        for t in TECHNIQUES:
            assert 1 <= t.tier <= 4, f"{t.name} has invalid tier {t.tier}"

    def test_cost_multiplier_positive(self) -> None:
        for t in TECHNIQUES:
            assert t.cost_multiplier > 0, f"{t.name} has non-positive cost"

    def test_unique_names(self) -> None:
        names = [t.name for t in TECHNIQUES]
        assert len(names) == len(set(names))


class TestTierOrdering:
    def test_techniques_ordered_by_tier(self) -> None:
        tiers = [t.tier for t in TECHNIQUES]
        assert tiers == sorted(tiers), "Techniques not ordered by tier"

    def test_each_tier_has_techniques(self) -> None:
        for tier in [1, 2, 3]:
            assert len(get_techniques_by_tier(tier)) > 0, f"Tier {tier} is empty"

    def test_tier_4_exists(self) -> None:
        assert len(get_techniques_by_tier(4)) >= 1

    def test_roadmap_matches_techniques(self) -> None:
        rm = get_roadmap()
        for tier_num, tier_list in [(1, rm.tier_1), (2, rm.tier_2), (3, rm.tier_3), (4, rm.tier_4)]:
            technique_names = [t.name for t in get_techniques_by_tier(tier_num)]
            assert tier_list == technique_names, (
                f"Roadmap tier {tier_num} doesn't match techniques"
            )


class TestCompetitorProfiles:
    def test_competitor_count(self) -> None:
        assert len(COMPETITORS) == 8

    def test_all_have_nonempty_approach_summary(self) -> None:
        for c in COMPETITORS:
            assert c.approach_summary.strip(), f"{c.name} has empty approach_summary"

    def test_all_have_source_url(self) -> None:
        for c in COMPETITORS:
            assert c.source_url.strip(), f"{c.name} has empty source_url"

    def test_all_have_as_of_date(self) -> None:
        for c in COMPETITORS:
            assert c.as_of_date is not None, f"{c.name} has no as_of_date"

    def test_unique_names(self) -> None:
        names = [c.name for c in COMPETITORS]
        assert len(names) == len(set(names))

    def test_known_competitors_exist(self) -> None:
        names = {c.name for c in COMPETITORS}
        for expected in ["Superforecasters", "BLF (Bayesian Linguistic Forecaster)", "Cassi AI"]:
            assert expected in names, f"Missing competitor: {expected}"


class TestPitfalls:
    def test_pitfall_count(self) -> None:
        assert len(PITFALLS) == 7

    def test_all_have_valid_severity(self) -> None:
        for p in PITFALLS:
            assert isinstance(p.severity, Severity)

    def test_all_have_nonempty_description(self) -> None:
        for p in PITFALLS:
            assert p.description.strip()

    def test_all_have_evidence_source(self) -> None:
        for p in PITFALLS:
            assert p.evidence_source.strip()

    def test_critical_pitfalls_exist(self) -> None:
        critical = get_pitfalls_by_severity("critical")
        assert len(critical) >= 2


class TestHelperFunctions:
    def test_get_techniques_by_tier(self) -> None:
        tier1 = get_techniques_by_tier(1)
        assert len(tier1) == 3
        assert all(t.tier == 1 for t in tier1)

    def test_get_techniques_by_tier_empty(self) -> None:
        assert get_techniques_by_tier(99) == []

    def test_get_techniques_for_track(self) -> None:
        tournament = get_techniques_for_track("tournament")
        assert len(tournament) == len(TECHNIQUES)

    def test_get_techniques_for_track_baseline(self) -> None:
        baseline = get_techniques_for_track("baseline")
        assert baseline == []

    def test_get_competitor_by_name_found(self) -> None:
        c = get_competitor_by_name("Superforecasters")
        assert c is not None
        assert c.best_known_brier == 0.096
        blf = get_competitor_by_name("BLF")
        assert blf is not None
        assert blf.name == "BLF (Bayesian Linguistic Forecaster)"

    def test_get_competitor_by_name_not_found(self) -> None:
        assert get_competitor_by_name("Nonexistent") is None

    def test_get_roadmap(self) -> None:
        rm = get_roadmap()
        assert len(rm.tier_1) == 3
        assert len(rm.tier_2) > 0
        assert len(rm.tier_3) > 0
        assert len(rm.tier_4) >= 1

    def test_get_pitfalls_by_severity(self) -> None:
        critical = get_pitfalls_by_severity("critical")
        assert all(p.severity == Severity.CRITICAL for p in critical)

    def test_get_pitfalls_by_severity_invalid(self) -> None:
        assert get_pitfalls_by_severity("nonexistent") == []


class TestCompetitionRules:
    def test_default_rules(self) -> None:
        assert COMPETITION_RULES.missing_forecast_default == 0.5
        assert "Brier Index" in COMPETITION_RULES.scoring_formula
        assert "bi-weekly" in COMPETITION_RULES.round_cadence

    def test_track_distinctions(self) -> None:
        assert "Zero-shot" in COMPETITION_RULES.baseline_track_rules
        assert "ensemble" in COMPETITION_RULES.tournament_track_rules


class TestBenchmarks:
    def test_benchmarks_populated(self) -> None:
        assert len(BENCHMARKS) >= 4

    def test_all_have_source(self) -> None:
        for b in BENCHMARKS:
            assert b.source.strip()


class TestCLI:
    def _run_cli(self, *extra_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "tournament_strategy.py", *extra_args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_default_summary(self) -> None:
        result = self._run_cli()
        assert result.returncode == 0
        assert "Techniques:" in result.stdout
        assert "Competitors:" in result.stdout

    def test_tier_flag(self) -> None:
        result = self._run_cli("--tier", "1")
        assert result.returncode == 0
        assert "Tier 1" in result.stdout

    def test_competitors_flag(self) -> None:
        result = self._run_cli("--competitors")
        assert result.returncode == 0
        assert "Superforecasters" in result.stdout

    def test_pitfalls_flag(self) -> None:
        result = self._run_cli("--pitfalls")
        assert result.returncode == 0
        assert "CRITICAL" in result.stdout

    def test_roadmap_flag(self) -> None:
        result = self._run_cli("--roadmap")
        assert result.returncode == 0
        assert "Tier 1" in result.stdout
        assert "Tier 4" in result.stdout
