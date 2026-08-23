# Outer Loop Retrospective — Field Report

First-run experience using the remote-factory outer loop on a real project (forecastbench). Written as feedback for the outer loop's developer. Covers setup friction, runtime observations, post-run issues, and suggested improvements.

**Project context:** Python backtest harness for ForecastBench forecasting competition. The gate is a pytest-based Brier score ladder with 2 pinned rounds. A full eval run takes ~10 hours and 9,025 LLM calls.

## Timeline

- **2026-08-19** — Multi-benchmark support merged in remote-factory (#1332). Local benchmark config fixed (our PR #162).
- **2026-08-19** — First outer loop run (gen-0). 12 candidates evaluated, ~$206, 5 commits auto-merged.
- **2026-08-20** — Human review of gen-0 results. Discovered auto-merge baseline confounding, parameter overfitting, off-topic work.
- **2026-08-21–23** — Validation eval runs to verify gen-0 improvements. Discovered scoring pipeline bug that was invisible to the gate but catastrophic on the full dataset. Three eval re-runs required due to cascading issues.

## Setup Friction

### 1. Benchmark misconfiguration (silent failure)

The benchmark name in `outer_loop/config.json` said `featurebench` instead of `forecastbench`. No error was raised — the outer loop ran with the wrong config. Had to create a project-local shadow config at `.factory/benchmarks/forecastbench.toml` to override the built-in one (which used `format = "json"` instead of `format = "pytest"`).

**Suggestion:** Validate the benchmark name against available configs before starting. A typo should be a hard error, not a silent misconfiguration that wastes a multi-hour run.

### 2. No `--focus` for sub-CEOs (remote-factory #1338)

Sub-CEOs spawned by `InnerLoop.step()` receive no focus directive. They scan the full backlog and pick whatever they want, including items unrelated to the fitness function.

**What happened:** One sub-CEO picked up an observability backlog item (structured logging, 131 lines across 9 files) that had nothing to do with improving the Brier score. Another picked up CI/CD work.

**Workaround:** We had to manually trim the backlog to a single item before each outer loop run, then restore the full backlog afterward.

**Suggestion:** `InnerLoop.step()` should pass `--focus` to sub-CEOs with the benchmark target or a user-supplied directive. Without this, the outer loop optimizes random backlog items instead of the fitness function.

### 3. Gate timing constraints

Our Brier gate takes ~310s cold. The InnerLoop timeout is 600s. One candidate timed out on the gate and scored 0.0 despite making useful changes (it was the one that added extended thinking, which actually worked).

**Suggestion:** Per-benchmark timeout overrides, or at minimum surface the timeout config so users know to adjust it.

## The Auto-Merge Problem

This was the most consequential issue from gen-0.

### What happened

When a candidate improves the score and auto-merges to main, the next candidate starts from a higher baseline (it inherits the previous candidate's improvements). All 5 successful candidates hit 7/11 gate rungs, but only because earlier candidates' improvements accumulated on main. By the end of the generation, every candidate looked equally good.

### Why it matters

The entire point of the outer loop is topology search — finding which agent configurations produce the best results. But auto-merge during a generation confounds the signal. You can't tell if a 2-node topology is better than a 3-node topology when they're evaluated against different baselines.

### What it caused downstream

Three sub-CEOs independently tuned calibration parameters (market anchor weight, extremity clamping, dataset shrinkage) against the same 2 gate rounds. Each claimed improvement. But they were fitting 4 parameters to 2 data points — zero degrees of freedom. The "improvements" were noise.

Worse: the gate only covered 2 rounds with ~850 questions each. Our project had a latent scoring bug that made all dataset questions score as 0.5 (missing). The gate didn't catch this because its small question set masked the problem. When we ran the full eval (9,025 questions across 33 rounds), the overall Brier Index was 57.5 — significantly worse than the pre-gen-0 baseline of 61+.

**The outer loop spent $206 optimizing calibration parameters on top of a system where half the questions couldn't even be scored.**

### Suggestion

Either:
1. Evaluate all candidates in a generation against the SAME baseline (snapshot at generation start)
2. Don't auto-merge during a generation — collect candidates as PRs, evaluate on the fixed baseline, merge the best one
3. At minimum, offer a `--no-auto-merge` flag

Also consider warning when the gate has very few evaluation points. Our 2-round gate let sub-CEOs overfit without detection.

## Post-Run Burden

### Human review overhead

5 auto-merged commits needed human review. The review cycle took a full day: spawning researchers to check competition legality, validating calibration parameters, tracing the scoring pipeline, and ultimately discovering the dataset scoring bug. This is disproportionate to the 4-hour run.

**Suggestion:** 
- Option to disable auto-merge and collect candidates as PRs for batch review
- Post-generation summary report: what changed, what scores moved, what parameters were tuned
- A "dry run" mode that scores candidates but doesn't merge

### Dirty working tree

The outer loop left staged and unstaged changes in the working tree after completion. This blocked `git pull` when we tried to pick up bugfixes, causing 3 failed attempts before we diagnosed the issue (required `git stash push -u`).

**Suggestion:** The outer loop should leave the working tree clean, or document the expected git state.

### Worktree accumulation

9 worktrees remained in `.factory-worktrees/` after gen-0, each with its own cache directory. No automatic cleanup. Disk usage grows unbounded, and cached data in worktrees is isolated from the main repo (not reusable).

**Suggestion:** Clean up worktrees after candidates are scored. Or share the cache across worktrees via symlink or configurable cache path.

### Finalize verdict silently overridden

The factory's `config.json` had a stale `eval_command` pointing to a non-existent path. When we ran `factory finalize --verdict keep`, the internal precheck failed silently and overrode the verdict to `revert`. The experiment was recorded as reverted despite all QA passing. No error message explained why.

**Suggestion:** `factory finalize` should clearly report WHY a precheck failed (e.g., "eval_command failed: file not found") rather than silently overriding the verdict.

## Validation Gap

After the outer loop auto-merges changes, there's no built-in step to validate the combined effect on real data. Our gate tested 2 rounds (~1,700 questions). The full eval covers 33 rounds (~54,000 scored entries). The gate showed +1.172 Brier Index improvement. The full eval showed -3.7 points — a regression masked by a scoring bug.

This is the fundamental risk of a narrow gate: it can give false positive signals that compound across candidates. Each sub-CEO sees an improvement on the gate, auto-merges, and the next sub-CEO builds on potentially-overfit or buggy changes.

**Suggestion:** 
- Document the expected post-run validation workflow
- Consider supporting a broader "validation round" as a built-in post-generation step
- Warn when gate coverage is a small fraction of the full evaluation surface

## Cost Summary

| Item | Cost | Time |
|------|------|------|
| Gen-0 outer loop (12 candidates) | ~$206 | ~4 hours |
| Validation eval run #1 (crashed) | ~$80 | ~10 hours |
| Validation eval run #2 (cache invalidated) | ~$80 | ~10 hours |
| Validation eval run #3 (with scoring fix) | ~$100 | ~15 hours (in progress) |
| Human review session | — | ~8 hours |
| Debugging (git state, cache, crashes, scoring pipeline) | — | ~6 hours |
| **Total cost of one outer loop generation** | **~$466** | **~53 hours** |

The outer loop itself was ~4 hours. Everything else — review, validation, debugging, re-runs — was ~49 hours of follow-up work. The ratio of "running the loop" to "dealing with the output" was roughly **1:12**.

Most of this overhead was caused by three issues that the outer loop could help prevent:
1. Auto-merge making review mandatory (instead of optional PR review)
2. No focus directive causing off-topic work that needed to be evaluated
3. Narrow gate masking a scoring bug that only surfaced on the full eval

## Improvements Wishlist (prioritized)

### High Priority (would have prevented our worst problems)
1. **Fix auto-merge baseline confounding** — candidates must be evaluated against a common baseline within a generation
2. **Add `--focus` to sub-CEO dispatch** (#1338) — without this, sub-CEOs do random work
3. **`--no-auto-merge` option** — let users review candidates as PRs before merging

### Medium Priority (significant quality-of-life)
4. **Benchmark name validation** — error on unknown benchmark names before starting
5. **Per-benchmark timeout config** — gates with different costs need different timeouts
6. **Post-generation summary report** — what changed, scores, parameters tuned
7. **Gate coverage warning** — flag when gate has very few evaluation points relative to the full eval surface
8. **Finalize failure transparency** — explain precheck failures, don't silently override verdicts

### Nice to Have
9. **Dry run mode** — score candidates without merging
10. **Tune/eval split support** — separate data for parameter tuning vs evaluation
11. **Clean working tree guarantee** — leave no dirty state after run
12. **Worktree cleanup** — remove worktrees after candidates are scored
13. **Eval command validation** — verify eval_command works before allowing experiments

## What Went Right

To be fair to the outer loop:
- Multi-benchmark support (#1332) landed cleanly and worked once configured correctly
- The topology search infrastructure ran without crashes — all 12 candidates were evaluated
- The gate (pytest ladder) worked reliably and gave consistent scores
- Sub-CEOs produced real, working code changes — the extended thinking and market anchoring features are genuinely useful
- Total cost of $206 for 12 candidates is reasonable for the amount of code generated

The core loop works. The issues are around the edges: focus control, baseline confounding, and the gap between gate scores and full-eval scores.

## Open Questions

1. Should we run gen-1 with the same 2-round gate, or expand to more rounds first?
2. Is the topology search providing value with auto-merge confounding, or should we use single-topology improve cycles until #1338 and the baseline fix land?
3. What's the minimum gate size to avoid sub-CEO parameter overfitting? Our experience suggests 2 rounds is far too few.
