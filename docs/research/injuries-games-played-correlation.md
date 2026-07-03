# Injury-Report Volume vs. Actual Games Played: v2-Trigger Correlation Check

Analysis date: 2026-07-03. Per issue #9 — a lightweight, standalone check of whether nflverse's
injury-report data carries real signal about games-played outcomes, logged as a v2-trigger candidate,
not a submodel built now (per ADR-0005).

## Methodology

- **Population**: every RB player-season with at least one game played, seasons 2009–2023 (2009 is
  `INJURY_REPORTS_EARLIEST_SEASON`, the first season nflverse's `load_injuries()`/`import_injuries()`
  covers; `ff_model.nflverse.load_injury_reports` already encodes this floor).
- **Injury signal**: `injury_report_count` — the total number of weekly injury-report rows (any
  `report_status`: Questionable/Doubtful/Out/etc.) for that player in that season. Seasons with no
  report rows get 0, not dropped.
- **Outcome**: `actual_games_played` — count of that player's weekly-stats rows in the same season
  (regular season only, matching the rest of the pipeline).
- **Correlation**: Pearson r (linear) and Spearman ρ (rank/monotonic), both same-season (not
  year-over-year — a deliberate scope choice for this check, see Limitations).

## Result

| | value |
|---|---|
| n (player-seasons) | 2,073 |
| Mean injury_report_count | 2.81 |
| Mean actual_games_played | 9.36 |
| Pearson r | **0.2765** (p = 1.1e-37) |
| Spearman ρ | **0.2909** (p = 1.0e-41) |

Both correlations are statistically significant at this sample size, but weak in magnitude (~0.28),
and **positive** — more injury-report appearances associate with *more* games played that season, not
fewer.

## Interpretation

The positive sign is very likely an **exposure confound**, not a real availability signal: a player
only generates injury-report rows in weeks they're on an active roster and *in the game-week cycle* —
a player who plays 16 games has 16 chances to pick up a "Questionable" tag for a minor tweak, while a
player who's cut, inactive, or on IR for most of the season generates few or no reports simply because
he isn't part of that week's practice-report process. So this same-season, raw-volume framing mostly
measures "was on an active roster a lot," which correlates with playing a lot — not injury severity or
games missed.

This means the check as run **does not cleanly answer** the trigger question either way. A cleaner
test — closer to the PFF year-over-year games-lost study cited in ADR-0005 — would need to either:
normalize by weeks-on-roster, restrict to specific report statuses (e.g. `Out`/`Doubtful` only, which
games_played.py's existing heuristic already does for its recent-injury flag), or run the correlation
year-over-year (season N's injury signal vs. season N+1's games played) instead of within the same
season.

## Conclusion

This specific check doesn't surface a real predictive availability signal, and doesn't cleanly
reaffirm v1's heuristic either, given the exposure confound. It's logged as-is per the issue's
scope (same-season volume vs. games played) rather than expanded into the year-over-year design,
which is a natural v2 follow-up if games-played modeling is revisited. Recommendation: keep v1's
Games-Played Estimate heuristic unchanged; if v2 revisits a real submodel, start with the
year-over-year, status-filtered design rather than this one.
