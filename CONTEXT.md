# Fantasy Football Draft Model

Predicts NFL skill-position (QB/RB/WR/TE) player performance for draft prep — producing a reusable projections table that later feeds a cheat sheet or live draft assistant (both out of scope for now).

## Language

**Projection**:
A player's predicted output for the upcoming season, expressed as a PPG Projection and a Games-Played Estimate rather than a single season-total number.
_Avoid_: season total, forecast

**PPG Projection**:
Predicted average fantasy points per game, computed by applying the league's Scoring Formula to the model's predicted raw stats (yards, receptions, TDs, etc.). The model predicts raw stats, not points directly.
_Avoid_: points projection, fantasy score

**Scoring Formula**:
The league-specific function mapping raw predicted stats to fantasy points (e.g. PPR vs. standard). Applied downstream of the model as a final step, never baked into training, so the same underlying projections serve any league's rules.
_Avoid_: scoring settings, league rules

**Games-Played Estimate**:
The availability half of a Projection (PPG Projection × Games-Played Estimate = season total). In v1 this is a heuristic — an age/position-conditioned base rate plus a recent-major-injury flag — not a trained submodel, based on research showing no strong player-level predictive signal for injury/availability exists in the literature.
_Avoid_: availability model, injury model (implies a trained model, which this isn't in v1)

**Depth-Chart Competition Feature**:
A v1 feature flagging whether a player faces new offseason competition for touches — a free-agent signing or high draft pick at the same position on the same team. Narrower than full team-context modeling: it's mechanically derivable from roster composition and doesn't require judgment calls about scheme fit or trade quality.
_Avoid_: context feature (too broad — covers the deferred trade/coaching-change work too), team feature

**Veteran**:
A player with at least a minimum threshold of career games/snaps — in scope for the v1 model. Distinguished from a Rookie (zero prior NFL seasons, excluded from v1) to keep low-sample players from injecting noise into training.
_Avoid_: player (too generic in this context)

**Rookie**:
A player with zero prior NFL seasons. Excluded from the v1 model entirely — no box-score/advanced-metric history exists to build features from. Targeted for a separate v2 submodel using pre-NFL signals (college production, draft capital, combine metrics).

**Walk-Forward Backtest**:
The evaluation method: train on seasons 1..N, predict season N+1, slide forward one season, repeat. Chosen over k-fold/random cross-validation because random splits would leak future seasons into training for the same player.
_Avoid_: cross-validation, k-fold

**TabFM**:
A pretrained tabular foundation model (TabPFN-style), evaluated as a candidate alongside LightGBM per ADR-0009; the evaluation was blocked by TabPFN's 10k-row training cap, so v1/v2 are LightGBM-only.
_Avoid_: TabPFN (the general term is TabFM here since the specific library choice isn't locked in)

**Matched Population**:
the subset of backtest players that have a real ADP for the target season — the players relevant enough that the crowd actually drafted them. All model-vs-ADP comparisons must be computed on this population for both sides (`docs/adr/0010-matched-population-headline-metric.md`); full-population numbers are inflated by trivially ranking stars over bench players.
_Avoid_: full-population comparison (as an edge claim)

**Answer Key**:
the retrospective-optimal order for a season: players ranked by actual season total fantasy points (per position), or by VOR-adjusted actual points for the pooled cross-position ranking (`docs/adr/0012-vor-adjusted-pooled-scoreboard.md`). Both the model's preseason list and ADP's are scored against it.
_Avoid_: ground truth (ambiguous about total-points vs PPG)

**Tier**:
a band of players with similar actual season value; derived per position from each season's actual outcomes by clustering on point gaps (variable-sized, shaped by that season's real scoring distribution), targeting ~6–8 tiers per position — not fixed ~12-player buckets, which would just re-derive ADP's own round structure instead of reading the season's actual shape. Tier accuracy — placing players in their true tier — is the decision metric vs ADP; matched-population Spearman stays the development metric (`docs/adr/0011-tier-accuracy-decision-metric.md`).
_Avoid_: rank buckets, fixed-size tiers

**Replacement Baseline**:
the last-starter player at a position in 12-team PPR (QB12/RB24/WR36/TE12), computed from actual season finishes. Subtracting its points gives VOR (value over replacement), the pooled Answer Key's unit.
_Avoid_: waiver-level (informal)

**ADP Blend**:
a deferred later layer that uses ADP as a model input and learns where to nudge the crowd's ranking. Not cheating (no leakage; blend-over-ADP improvement is purely model-added information) but excluded from v2, which is standalone-only (`docs/adr/0013-standalone-model-first-adp-blend-deferred.md`).
_Avoid_: ensemble (too generic)

**ADP Benchmark**:
The evaluation comparison against historical Average Draft Position (sourced from the Fantasy Football Calculator API), used as a proxy for crowd/expert consensus since historical *point* projections from major platforms aren't reliably available for free. See `docs/research/historical-adp-and-projections-availability.md`.
_Avoid_: expert benchmark, consensus projection (that would be the unbuilt point-projection alternative)
