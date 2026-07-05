# Disagreement Edge: evaluating whether model-vs-ADP disagreements are trustworthy

Analysis date: 2026-07-05. Proposes a metric for the gap identified in this session: the
matched-population Spearman (ADR-0010) and tier-accuracy (ADR-0011) scoreboards both grade the
model's *overall* agreement with actual outcomes, but a drafter only extracts value from the model
at the moments it **disagrees** with ADP. Neither existing metric isolates whether those
disagreements are, on average, correct.

**Caveat on sourcing**: this session's web search tooling was down (persistent 502s) and the
background research agent that was meant to verify these against primary sources failed on a
session limit before writing anything. What follows is compiled from working knowledge of the
finance and forecast-evaluation literature, not confirmed against original papers this session.
The named results (Grinold's Fundamental Law, Fair-Shiller encompassing regression,
Diebold-Mariano) are well-established enough in their fields that misattribution is unlikely, but
citations (year, journal, exact formula) should be checked against the originals before being
treated as authoritative in an ADR.

## The core idea, borrowed from three fields

**Quant finance — residualizing against the benchmark.** Grinold's "Fundamental Law of Active
Management" (Grinold, 1989, *Journal of Portfolio Management*; expanded in Grinold & Kahn, *Active
Portfolio Management*) evaluates a manager's stock-picking skill via the **Information
Coefficient**: the correlation between a manager's *forecast scores* and *forward returns*. The
version of this relevant here is the "active" decomposition — once you've subtracted out the
benchmark's own return/consensus view from both the forecast and the outcome, the correlation of
what's left (the manager's *active bet* vs. the *active return*) isolates skill from
benchmark-hugging. A manager who mostly holds the index and makes a few small active bets is
scored only on whether those bets paid off, not on the passive agreement.

**Forecast evaluation — encompassing regression.** Chong & Hendry (1986) and Fair & Shiller (1990,
*"Comparing Information in Forecasts from Econometric Models," American Economic Review*) proposed
regressing the realized outcome on two competing forecasts simultaneously:

```
actual = β0 + β1 * forecast_A + β2 * forecast_B + ε
```

If β2 is indistinguishable from zero once forecast_A is in the regression, forecast_B is said to
be "encompassed" by A — it carries no independent information. If β2 is significantly positive,
B adds real information beyond A. This is a direct, well-established test for exactly the question
"does my model's disagreement with the consensus forecast contain real information?"

**Forecast comparison — paired accuracy tests.** Diebold & Mariano (1995, *"Comparing Predictive
Accuracy," Journal of Business & Economic Statistics*) formalized comparing two forecasts'
accuracy via a paired loss-differential test, robust to small samples via a sign-test framing
rather than requiring large-sample normality. This is the right lineage for a "does the model win
more often than ADP, on the cases where they actually differ" framing — a plain win-rate is a
special case of this loss-differential comparison.

## Candidate formulations for this project

Three formulations, from simplest/most interpretable to most rigorous, all computed **per
position, on the Matched Population, against the Answer Key** (same infrastructure ADR-0010
already establishes):

### 1. Rank-residual correlation (interpretable headline number)

For each player `p` in the Matched Population:

```
D_model(p) = model_rank(p) - adp_rank(p)     # negative = model likes player more than the room
D_actual(p) = actual_rank(p) - adp_rank(p)    # negative = player outperformed ADP's implied rank
```

`Disagreement Edge = corr(D_model, D_actual)` (Spearman or Pearson), computed across the pooled
backtest seasons per position (season as a blocking/clustering factor, not pooled naively — see
pitfalls below).

A positive correlation says: when the model diverges from ADP, the direction of that divergence
tends to match the direction actual outcomes diverge from ADP. This is the rank-space analog of
Grinold's active-return correlation — it's easy to explain to a non-technical reader ("when the
model disagrees with the room, is it usually disagreeing in the right direction?") and reuses
exactly the rank machinery `evaluation.py` already has (`spearman_rank_correlation`).

### 2. Encompassing regression (rigorous, quantifies added information)

```
actual_value ~ β0 + β1 * adp_value + β2 * model_value
```

on a common cardinal scale (percentile-of-population is safer than raw rank here, since rank
differences aren't uniformly meaningful — moving from ADP 3 to model-rank 8 is a very different
real bet than moving from ADP 150 to rank 155). `β2`, tested via the project's existing bootstrap
machinery (`percentile_bootstrap_interval` in `evaluation.py`) rather than an asymptotic normal
test given the small season count, answers "does the model add real information beyond ADP" more
rigorously than the rank correlation, at the cost of being harder to explain in a results note.

### 3. Conditional win-rate on large disagreements (matches the project's existing tier framing)

Restrict to player-seasons where `|model_rank - adp_rank|` exceeds a **pre-specified** threshold
(e.g. one full draft round ≈ 12 picks — chosen for draft-actionability, not tuned post-hoc to
maximize the metric). Within that bucket:

```
win_rate = fraction of cases where |actual_rank - model_rank| < |actual_rank - adp_rank|
```

This is the plainest possible version of a Diebold-Mariano-style paired comparison (a sign test on
which forecast's error is smaller), and it mirrors the tier-accuracy metric's own framing
(ADR-0011): a human-legible win/tie/loss verdict, restricted to the cases that actually matter for
a real draft decision — exactly your Cooper Kupp example.

## Pitfalls

- **Threshold selection (metric 3).** Any "meaningful disagreement" cutoff chosen after looking at
  results is overfitting. Pick it from domain reasoning (a full round, or some fixed rank-gap) before
  computing the win rate, and don't retune it if the first number disappoints.
- **Small independent sample size.** ~14 backtest seasons is the real unit of independence here —
  players within a season share season-level shocks (a bad year for RBs broadly, a rule change,
  etc.), so naive player-level standard errors overstate confidence. Use season-block bootstrap
  (resample whole seasons, not individual players) for any confidence interval, consistent with
  the block-resampling already implicit in the walk-forward design.
- **Noise can look like a small positive edge.** A model that's mostly a noisy copy of ADP plus a
  little genuine signal will show a small positive correlation/win-rate that's mostly sampling
  noise rather than real skill. Report a confidence interval, not a point estimate, and treat "edge
  distinguishable from zero" as the bar, not "edge is positive."
- **Selection/regression-to-the-mean on flagged disagreements.** The "top disagreements" surfaced
  for a drafter (per issue #17's plan) are, by construction, extreme draws from whatever noise
  exists in the model. Even a genuinely skilled model's single biggest flagged disagreements will
  look worse out-of-sample than in-sample selection suggests. Report the whole disagreement-bucket
  distribution in the backtest report, not just the headline top-N flagged cases.

## Recommendation

Use **metric 1 (rank-residual correlation)** as the headline "Disagreement Edge" number reported
per position alongside the existing Spearman/tier-accuracy scoreboards — cheapest to compute, reuses
existing rank-correlation code, and is the most direct match for "is the end goal" framing from
this session. Add **metric 3 (conditional win-rate, pre-specified threshold)** as the
human-readable decision verdict, in the same spirit as tier accuracy's beat/tie/lose framing,
since it maps directly onto the real question ("when the model tells you to deviate from the room,
should you listen"). Treat **metric 2 (encompassing regression)** as a secondary, more rigorous
check to run during development but not necessarily surface in the drafter-facing report — it's
the right tool for validating that the correlation in metric 1 isn't an artifact of a confound, but
it's harder to explain than the other two.

All three should be computed with season-block bootstrap confidence intervals using the existing
`percentile_bootstrap_interval` machinery, and none should be computed with a threshold or
weighting chosen after seeing the results.

## Open follow-up

Verify the Grinold (1989), Fair & Shiller (1990), Chong & Hendry (1986), and Diebold & Mariano
(1995) citations against primary sources once web tooling is available again — this document
currently rests on recalled descriptions of well-known results, not confirmed page/section
references.
