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
A pretrained tabular foundation model (TabPFN-style), being evaluated as a candidate alongside LightGBM per position before v1 commits to one approach. See `docs/adr/0009-evaluate-tabfm-alongside-lightgbm.md`.
_Avoid_: TabPFN (the general term is TabFM here since the specific library choice isn't locked in)

**ADP Benchmark**:
The evaluation comparison against historical Average Draft Position (sourced from the Fantasy Football Calculator API), used as a proxy for crowd/expert consensus since historical *point* projections from major platforms aren't reliably available for free. See `docs/research/historical-adp-and-projections-availability.md`.
_Avoid_: expert benchmark, consensus projection (that would be the unbuilt point-projection alternative)
