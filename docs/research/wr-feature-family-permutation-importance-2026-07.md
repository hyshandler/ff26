# WR Backtest Rebuild + Permutation Importance for Round-1 Feature Families (Issue #26)

Analysis date: 2026-07-06. ADR-0015 shipped four new WR feature families (Prior-Season Total
#22, Opportunity Vacuum #23, Own-Team Offensive Environment #24, Per-Touch Efficiency #25)
wired in directly as WR defaults, deliberately skipping the pre-adoption Disagreement Edge
sweep (ADR-0014) the four original `PositionConfig` ablation flags went through. ADR-0015
committed to "evaluate via the standard backtest report after shipping instead of gating
adoption on a pre-sweep" — this is that evaluation. It rebuilds the WR backtest report with
all four families active over the standard 2012-2025 Walk-Forward Backtest, and runs a
permutation-importance sweep (shuffling each family's prediction-time column values, one
family at a time, everything else held fixed) to isolate whether each family is actually
contributing signal or riding along inert.

## Bug fixed along the way

`load_ngs_receiving`'s empty-seasons-requested return path (`nflverse.py`) built its empty
`DataFrame` from bare column names with no explicit dtypes, defaulting every column —
including `avg_yac_above_expectation` — to `object`. That's invisible on any split whose
target seasons include 2016+, but any Walk-Forward split trained entirely on pre-2016 seasons
(the earliest splits in the standard 2012-2025 range) hits this path, and the `object` dtype
survives the merge chain into LightGBM's training matrix, which rejects `object`-dtype columns
outright. Fixed by giving the empty frame the same explicit dtypes as its non-empty return
path. Independent of any feature-family decision below; covered by a new non-network regression
test in `test_nflverse.py`.

## Method

`build_backtest_report("WR", seasons=STANDARD_BACKTEST_SEASONS, permute_columns=...)` run five
times: once unpermuted (baseline, all four families' real values), then once per family with
that family's columns shuffled at prediction time via the new `permute_columns`/
`permute_random_state` plumbing (`position_model.permute_feature_columns`). Disagreement Edge
(ADR-0014: rank-residual correlation, season-block-bootstrap 95% CI, large-disagreement
(>12 rank) conditional win-rate) is the primary metric — a family that's actually contributing
signal should show the *permuted* variant's correlation point estimate degrade outside the
*baseline's* CI. Matched-population Spearman and tier accuracy (ADR-0011) are reported
alongside as supporting sanity checks only, never used for the verdict, per the same standard
ADR-0014 sets and issue #16's re-sweep followed.

`n_matched=502` throughout (Matched Population per ADR-0010, unaffected by which columns are
permuted at prediction time).

## Results

| Family (permuted) | corr | 95% CI | win_rate | n_large | matched Spearman | tier accuracy |
|---|---|---|---|---|---|---|
| **baseline (none permuted)** | **0.244** | (0.135, 0.326) | 0.365 | 181 | 0.303 | 0.464 |
| prior_season_totals | 0.236 | (0.145, 0.331) | 0.328 | 195 | 0.189 | 0.430 |
| opportunity_vacuum | 0.238 | (0.129, 0.323) | 0.358 | 187 | 0.296 | 0.456 |
| team_offensive_environment | 0.245 | (0.132, 0.341) | 0.352 | 176 | 0.322 | 0.462 |
| per_touch_efficiency | 0.240 | (0.139, 0.330) | 0.372 | 183 | 0.294 | 0.456 |

For reference, the pre-round-1 WR baseline (no round-1 families at all, i.e. WR's other
`PositionConfig` knobs — `multi_season_window`, `experience_feature`, `sos_feature`,
`include_depth_chart_competition` — all at their post-issue-#16 simplest settings) from
`assets/feature_sweep_16.json`: corr 0.239 (matched Spearman 0.307, tier accuracy 0.456,
win_rate 0.353).

## Verdict: all four families are riding along inert on Disagreement Edge

Every permuted variant's correlation point estimate falls well inside the baseline's own CI
(0.135, 0.326), and each permuted variant's CI heavily overlaps the baseline's — exactly the
"indistinguishable from noise" bar issue #16's re-sweep used, applied here in reverse (removing
a family's signal, rather than adding a candidate's). Scrambling any one of the four families
does not measurably hurt (or help) the model's edge over ADP. The **round-as-a-whole** comparison
tells the same story: post-round baseline (0.244) vs. pre-round baseline (0.239) is also fully
inside the noise band — adding all four families together didn't move Disagreement Edge either.

Per family:

- **prior_season_totals**: no Disagreement Edge movement, but permuting it visibly drops
  matched-population Spearman (0.303 → 0.189) and tier accuracy (0.464 → 0.430) — the largest
  supporting-metric move of the four. Read together, this looks like exactly the failure mode
  ADR-0014 exists to catch: `prior_season_fantasy_points` (last season's total) is highly
  correlated with what ADP *itself* is built from (last year's output is a dominant ADP input),
  so the feature raises the model's overall rank correlation with outcomes by making the model
  echo ADP's own prior information more closely — not by adding independent signal that beats
  ADP. That's consistent with the family riding along inert on the metric that actually matters.
- **opportunity_vacuum**: no meaningful movement on any metric (Disagreement Edge, matched
  Spearman, or tier accuracy). Riding along inert.
- **team_offensive_environment**: no meaningful movement on any metric — if anything, permuting
  it produced a marginally *higher* point estimate on both Disagreement Edge and matched
  Spearman, well inside noise. Riding along inert.
- **per_touch_efficiency**: no meaningful movement on any metric. Riding along inert. This
  includes `prior_season_yac_above_expectation`, which is NaN for every pre-2016 target season
  by design (ADR-0015) — that gap doesn't appear to be costing anything distinguishable from
  noise either.

**Implication for RB/QB/TE extension**: none of the four families has demonstrated a
Disagreement-Edge contribution on WR, so there's no basis yet for extending any of them to other
positions. This doesn't mean the underlying ideas are wrong — matched-population Spearman still
trails ADP by a wide margin at every position (WR: model 0.303 vs. ADP 0.520 here), so there's
real room for a feature to close that gap — but these four specific formulations haven't done it
for WR. Per ADR-0015's own stated appetite to revisit "if these features turn out to need the
same noise-vs-signal scrutiny the original four did": they do.

## Caveats

- This sweep permutes one family at a time with the other three (and all of WR's other
  defaults) held at their real values — it doesn't test interactions between families, or
  whether removing all four at once behaves differently than removing each individually. A
  joint ablation would need a larger config grid, out of scope here.
- Matched-population Spearman is a supporting signal only, never a selection criterion — per
  ADR-0014's own rationale, a feature that raises Spearman by making the model resemble ADP more
  closely (as `prior_season_totals` may be doing) is exactly the case Disagreement Edge exists
  to catch, not evidence of a real edge.
- Bootstrap CIs on the primary metric span roughly 0.13-0.34 at n_matched=502 — comfortably
  larger than any of the ~0.01 point-estimate differences between baseline and any permuted
  variant. A much larger backtest population would be needed to detect a genuinely small but
  real per-family contribution, if one exists.
