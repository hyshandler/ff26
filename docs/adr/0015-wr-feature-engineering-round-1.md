# WR feature engineering round 1: four new feature families, scoring formula baked into two of them

Issue #16's re-sweep found every existing feature family (multi-season memory, age/experience,
Strength-of-Schedule, Depth-Chart Competition) indistinguishable from noise under Disagreement Edge
(ADR-0014). Closing the ADP gap (ADR-0011's stated v2 goal) needs genuinely new signal, not more
variants of what's already been tried. This ADR scopes the next round to **WR only** — other
positions are deferred until this round is judged, so lessons from a smaller surface can inform how
(or whether) to extend it.

Four new feature families ship for WR:

## 1. Prior-Season Total

Two features computed once from the immediately preceding season's completed totals (not a
multi-season window, not a trailing within-season average): `prior_season_fantasy_points` and
`prior_season_games_played`. Distinct from the existing `trailing_avg_*` features, which are
per-game rates that reset at each season boundary — Prior-Season Total captures the *total*
(games played × rate), which is the one signal the per-game-rate design deliberately excludes.

## 2. Opportunity Vacuum (issue #20)

Two components, fed to the model as separate columns rather than a hand-multiplied composite (this
codebase's existing convention — e.g. `trailing_target_share` and `trailing_snap_pct` are separate
inputs — and LightGBM finds interactions via splits natively):

- `vacated_target_share`: sum of prior-season target share held by teammates who departed the roster
  entirely (traded, released, retired, or otherwise not on this team's roster for the target season).
  Detection is **pure roster departure only** — no injury/IR/return-status reasoning — the same
  judgment-free standard ADR-0004 set for Depth-Chart Competition's arrival side. Reason for departure
  doesn't change the mechanism (the share is vacated either way), and in-season injury status requires
  guessing whether an absence is season-ending, exactly the kind of subjective call ADR-0004 excluded.
- `prior_season_points_per_target`: the receiving player's own prior-season efficiency, computed with
  the `PPR` formula (see below).

## 3. Own-Team Offensive Environment

Three prior-season team-level features: pass attempts per game, points per game, and passing
efficiency (yards per attempt). Distinct from Strength-of-Schedule (opponent-side, already found to
be noise) — this is about how much opportunity the player's *own* offense generates, an angle SOS's
re-sweep never tested.

## 4. Per-Touch Efficiency

Two features: prior-season yards-per-target (from `weekly_stats`, already loaded, full 2006+
coverage), and prior-season `avg_yac_above_expectation` from `nfl_data_py`'s NGS receiving data.

**Known gap**: NGS receiving data only exists from **2016** onward. Seasons before that have this
feature as NaN for every player. Left as-is rather than restricting the backtest window — LightGBM
already handles missingness natively (the same pattern as a player's first week of a season having
no trailing average yet), and narrowing the window would cost backtest seasons without a clear
enough benefit to justify a second, bespoke evaluation range.

## Scoring Formula baked into two training features

CONTEXT.md's Scoring Formula entry states it's "applied downstream of the model's final step, never
baked into training, so the same underlying projections serve any league's rules." `prior_season_fantasy_points`
and `prior_season_points_per_target` break that rule: both apply the `PPR` formula (`scoring.py`,
`passing_tds: 4`, `receptions: 1`, etc.) directly to prior-season raw-stat totals as a training-time
input, not a downstream-only transform of model output.

This is a deliberate, narrow exception, not a reversal of the rule: raw stats (targets, receptions,
yards, TDs) individually don't capture "how big/efficient was last season" as one number the way a
single points total does, and re-deriving that combined signal without a scoring formula would mean
either picking one raw stat as a proxy (losing information) or hand-designing an equivalent weighting
(reinventing the scoring formula anyway). The model's *output* is still scored downstream via
`score_projections` for whichever league's formula the user wants — this exception only affects two
*input* columns, both computed once from historical (already-known) data, not from anything the model
predicts. If a later league needs different scoring weights, these two input features would need
recomputing with that formula — a real but narrow portability cost, worth taking for the signal.

## Process

These four families are wired in directly as WR defaults, not swept under Disagreement Edge before
adoption the way `PositionConfig`'s ablation flags are. This is a deliberate departure from ADR-0014's
"select on Disagreement Edge, not intuition" process for the existing four families — evaluate via the
standard backtest report after shipping instead of gating adoption on a pre-sweep. Revisit this
shortcut if these features turn out to need the same noise-vs-signal scrutiny the original four did.

## Consequences

- WR-only scope for this round; RB/QB/TE extension is separate follow-up work once this round's
  results are in.
- Two new prior-season-scoped raw data dependencies: NGS receiving data (2016+, via `nfl_data_py`'s
  `import_ngs_data`) and roster-departure detection (reusing `load_seasonal_rosters`, same source as
  Depth-Chart Competition).
- `docs/agents/domain.md`'s CONTEXT.md Scoring Formula entry is amended to note this exception rather
  than silently contradicting it.
