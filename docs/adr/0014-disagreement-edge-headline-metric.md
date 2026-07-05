# Disagreement Edge is the true headline metric; matched-population Spearman and tier accuracy are supporting

ADR-0010 and ADR-0011 grade the model's *overall* rank fidelity against ADP — how well the full
list agrees with actual outcomes. But a drafter extracts value from the model only at the moments
it **disagrees** with ADP (e.g. model ranks a player 25th when his ADP says 40th). If model and ADP
mostly agree, overall rank-correlation numbers are dominated by the agreement cases and say nothing
about whether the disagreements — the only decisions the model actually changes — are trustworthy.
This was the gap: no existing or planned metric isolated that question.

We're adding **Disagreement Edge** as the new headline development goal, per position on the
Matched Population, alongside the existing scoreboards:

- **Primary number**: rank-residual correlation, `corr(model_rank − adp_rank, actual_rank − adp_rank)`.
  Positive means that when the model diverges from ADP, actual outcomes tend to diverge from ADP in
  the same direction — i.e. the model's disagreements carry real information, not noise.
- **Decision verdict**: conditional win-rate restricted to large disagreements (threshold fixed in
  advance, e.g. one draft round ≈ 12 rank positions, not tuned post-hoc) — the fraction of those
  cases where the model's rank landed closer to the actual outcome than ADP's did.

  Unlike tier accuracy (ADR-0011), this is **not** a head-to-head contest with ADP: Disagreement
  Edge is inherently a statement about the model's deviations *from* ADP, and there's no analogous
  "ADP's disagreement with itself" to compute and compare against — that's zero by construction. The
  real null hypothesis is that the model's disagreements are pure noise (correlation ≈ 0, win-rate ≈
  50%), not "worse than ADP." The win condition is therefore **the correlation/win-rate is
  significantly different from the null, per position, via the bootstrap CI below** — not "beats
  ADP."
- **Secondary check** (development-only, not necessarily surfaced in the drafter-facing report): an
  encompassing regression, `actual_value ~ β0 + β1·adp_value + β2·model_value`, to confirm the
  primary correlation isn't a confound — a significant β2 means the model adds information beyond
  ADP alone.

All three are computed with season-block bootstrap confidence intervals (resampling whole backtest
seasons, not individual players), since ~14 backtest seasons is the real unit of independence and
naive player-level standard errors would overstate confidence.

Matched-population Spearman (ADR-0010) and tier accuracy (ADR-0011) remain in the report as
supporting metrics — they're still useful for tracking overall model quality during development —
but they are no longer the metric that decides whether the model is actually useful to a drafter.
That's Disagreement Edge.

See `docs/research/disagreement-edge-metric.md` for the methodology this borrows from (finance's
information-coefficient-net-of-benchmark framing, forecast-encompassing regression, and
Diebold-Mariano-style paired forecast comparison) and known pitfalls (threshold selection,
small-sample power, regression-to-the-mean on flagged top disagreements).

## Consequences

- Issue #14 (tier accuracy) is unaffected in scope but demoted from decision metric to supporting
  signal; the PRD's stated win condition now belongs to Disagreement Edge (issue #19), stated as
  "significantly distinguishable from noise per position," not "beats ADP" (see win-condition note
  above).
- Issue #15 (VOR pooled cross-position scoreboard, ADR-0012) is closed as won't-do: it measured the
  same kind of overall rank agreement Disagreement Edge was created to move past, and extending it to
  a pooled Disagreement Edge wasn't judged worth a second implementation of the same computation
  given the per-position scoreboards already cover the decision-relevant signal.
- Feature selection (issue #16) should select on Disagreement Edge, not matched-population Spearman —
  a feature can raise overall Spearman by making the model hug ADP more closely without adding any
  genuine disagreement signal, which is exactly the failure mode this ADR exists to avoid.
- Issue #17's "largest disagreements flagged" display feature now has a metric behind it — the flagged
  cases are exactly what the conditional win-rate scores, so the two should share one code path when
  #17 is built (rather than #17 inventing its own ad hoc definition of "biggest disagreement").
