# Feature-Family Re-Sweep Under Disagreement Edge (Issue #16)

Analysis date: 2026-07-05. ADR-0014 replaced matched-population Spearman with **Disagreement
Edge** as the headline decision metric: a feature can raise overall Spearman by making the model
hug ADP's own ranking more closely, without adding any genuine independent signal. The four
feature families currently baked into `PositionConfig` (multi-season memory window, age/experience
encoding, Strength-of-Schedule encoding, Depth-Chart Competition) were all originally chosen under
the retired full-population/matched-population criterion (`docs/research/multi-season-memory-features.md`
and the RB/QB/WR/TE extension work), so issue #16 re-swept every family, per position, scored on
Disagreement Edge instead.

## What was built

Two pre-existing bugs blocked the re-sweep and are fixed as part of this work, independent of any
feature-family decision:

- `load_red_zone_rush_attempts` (`nflverse.py`) re-decoded the *entire* nflverse play-by-play
  dataset from scratch on every Walk-Forward Backtest split (RB's red-zone-carries share feature
  needs it), instead of memoizing per season like every other loader in the module. This made a
  single RB backtest over 2012-2025 take 10+ minutes and made a 56-config sweep impractical; it's
  now cached per season in-process, matching the existing pattern.
- `crosswalk_adp_to_player_ids` (`adp.py`) crashed with `AttributeError` on a null `player_name`
  in one 2016 roster row (`player_id=00-0031605`), which broke ADP crosswalking -- and therefore
  `build_backtest_report` -- for *any* position whose target seasons included 2016. Fixed by
  treating a missing name as unmatched (consistent with the function's existing "unmatched rows
  get NaN, not dropped" contract) rather than crashing.

`scripts/feature_family_sweep.py` then ran `run_backtest.build_backtest_report` for each position
(RB/QB/WR/TE) x each family x each candidate value (holding the position's other three families at
their current `PositionConfig` defaults), over the full `STANDARD_BACKTEST_SEASONS` range
(2012-2025). Raw results: `assets/feature_sweep_16.json`.

## Selection criterion

Per ADR-0014: the primary number is Disagreement Edge's rank-residual correlation with its
season-block bootstrap 95% CI; the win-rate on large disagreements (>12 ranks) is the decision
verdict. A candidate only replaces a position's current default if it **clearly** beats it --
not a marginally higher point estimate riding a fully-overlapping CI, per the same standard used in
the original multi-season-memory sweep. Matched-population Spearman and tier accuracy are reported
alongside as supporting sanity checks only, never used to override the Disagreement Edge selection
(this is exactly the failure mode ADR-0014 exists to avoid).

## Results

Current `PositionConfig` default marked **bold**; `n_large` is the number of large-disagreement
(>12 rank) player-seasons the win-rate is computed over.

### RB (n_matched=414)

| Family | Variant | corr | 95% CI | win_rate | n_large |
|---|---|---|---|---|---|
| multi_season_window | none | 0.220 | (0.094, 0.405) | 0.323 | 96 |
| multi_season_window | career | 0.236 | (0.139, 0.382) | 0.322 | 90 |
| multi_season_window | last_n | 0.242 | (0.115, 0.413) | 0.352 | 91 |
| multi_season_window | **recency_weighted** | **0.233** | (0.126, 0.388) | 0.344 | 90 |
| experience_feature | **none** | **0.233** | (0.126, 0.388) | 0.344 | 90 |
| experience_feature | age | 0.231 | (0.117, 0.400) | 0.348 | 89 |
| experience_feature | years_in_league | 0.237 | (0.110, 0.392) | 0.360 | 89 |
| experience_feature | career_games | 0.209 | (0.109, 0.354) | 0.344 | 93 |
| experience_feature | career_stage_bucket | 0.226 | (0.107, 0.376) | 0.361 | 83 |
| sos_feature | **none** | **0.233** | (0.126, 0.388) | 0.344 | 90 |
| sos_feature | season_wide | 0.242 | (0.127, 0.389) | 0.368 | 87 |
| sos_feature | actual_games | 0.231 | (0.121, 0.408) | 0.330 | 88 |
| include_depth_chart_competition | **True** | **0.233** | (0.126, 0.388) | 0.344 | 90 |
| include_depth_chart_competition | False | 0.229 | (0.117, 0.380) | 0.353 | 85 |

### QB (n_matched=198)

| Family | Variant | corr | 95% CI | win_rate | n_large |
|---|---|---|---|---|---|
| multi_season_window | none | 0.330 | (0.193, 0.582) | 0.267 | 15 |
| multi_season_window | career | 0.349 | (0.189, 0.496) | 0.214 | 14 |
| multi_season_window | **last_n** | **0.300** | (0.115, 0.570) | 0.200 | 10 |
| multi_season_window | recency_weighted | 0.316 | (0.182, 0.545) | 0.300 | 10 |
| experience_feature | **none** | **0.300** | (0.115, 0.570) | 0.200 | 10 |
| experience_feature | age | 0.282 | (0.114, 0.496) | 0.222 | 9 |
| experience_feature | years_in_league | 0.293 | (0.142, 0.521) | 0.300 | 10 |
| experience_feature | career_games | 0.310 | (0.146, 0.521) | 0.300 | 10 |
| experience_feature | career_stage_bucket | 0.310 | (0.163, 0.548) | 0.231 | 13 |
| sos_feature | **none** | **0.300** | (0.115, 0.570) | 0.200 | 10 |
| sos_feature | season_wide | 0.309 | (0.186, 0.538) | 0.333 | 12 |
| sos_feature | actual_games | 0.302 | (0.125, 0.558) | 0.167 | 12 |
| include_depth_chart_competition | **True** | **0.300** | (0.115, 0.570) | 0.200 | 10 |
| include_depth_chart_competition | False | 0.324 | (0.169, 0.562) | 0.300 | 10 |

### WR (n_matched=502)

| Family | Variant | corr | 95% CI | win_rate | n_large |
|---|---|---|---|---|---|
| multi_season_window | none | 0.229 | (0.134, 0.298) | 0.342 | 187 |
| multi_season_window | career | 0.233 | (0.137, 0.334) | 0.373 | 193 |
| multi_season_window | last_n | 0.238 | (0.156, 0.351) | 0.349 | 189 |
| multi_season_window | **recency_weighted** | **0.235** | (0.123, 0.342) | 0.356 | 177 |
| experience_feature | **none** | **0.235** | (0.123, 0.342) | 0.356 | 177 |
| experience_feature | age | 0.229 | (0.125, 0.340) | 0.327 | 165 |
| experience_feature | years_in_league | 0.225 | (0.118, 0.320) | 0.329 | 167 |
| experience_feature | career_games | 0.232 | (0.112, 0.330) | 0.365 | 167 |
| experience_feature | career_stage_bucket | 0.232 | (0.115, 0.340) | 0.363 | 171 |
| sos_feature | **none** | **0.235** | (0.123, 0.342) | 0.356 | 177 |
| sos_feature | season_wide | 0.237 | (0.120, 0.338) | 0.328 | 186 |
| sos_feature | actual_games | 0.223 | (0.119, 0.330) | 0.330 | 179 |
| include_depth_chart_competition | **True** | **0.235** | (0.123, 0.342) | 0.356 | 177 |
| include_depth_chart_competition | False | 0.239 | (0.130, 0.344) | 0.353 | 184 |

### TE (n_matched=160)

| Family | Variant | corr | 95% CI | win_rate | n_large |
|---|---|---|---|---|---|
| multi_season_window | none | 0.315 | (-0.004, 0.470) | 0.000 | 1 |
| multi_season_window | career | 0.345 | (0.022, 0.544) | 0.500 | 2 |
| multi_season_window | last_n | 0.279 | (-0.098, 0.477) | 0.500 | 2 |
| multi_season_window | **recency_weighted** | **0.313** | (-0.006, 0.503) | 0.500 | 2 |
| experience_feature | **none** | **0.313** | (-0.006, 0.503) | 0.500 | 2 |
| experience_feature | age | 0.280 | (-0.025, 0.529) | 0.000 | 2 |
| experience_feature | years_in_league | 0.306 | (-0.008, 0.558) | 0.500 | 2 |
| experience_feature | career_games | 0.309 | (-0.022, 0.547) | 0.500 | 2 |
| experience_feature | career_stage_bucket | 0.263 | (-0.042, 0.498) | 0.500 | 2 |
| sos_feature | **none** | **0.313** | (-0.006, 0.503) | 0.500 | 2 |
| sos_feature | season_wide | 0.295 | (-0.030, 0.547) | 0.500 | 2 |
| sos_feature | actual_games | 0.271 | (-0.055, 0.489) | 0.000 | 1 |
| include_depth_chart_competition | **True** | **0.313** | (-0.006, 0.503) | 0.500 | 2 |
| include_depth_chart_competition | False | 0.311 | (-0.028, 0.532) | 0.500 | 2 |

## Decision

**No `PositionConfig` defaults changed.** For all four positions and all four families, every
candidate's correlation point estimate falls well inside the current default's bootstrap CI (and
vice versa) -- the point-estimate differences (typically ±0.01-0.02, occasionally ±0.03-0.05 for
QB/TE) are indistinguishable from sampling noise given the CI widths. This holds even where a
candidate's raw number looks locally appealing (e.g. QB `career` at 0.349 vs. the current `last_n`
at 0.300, or RB `season_wide` SOS at 0.242 vs. the current `none` at 0.233) -- none of these clear
the "clearly beats" bar, so re-adopting them would be selecting on noise, exactly the failure mode
ADR-0014 exists to avoid.

Matched-population Spearman and tier accuracy (`assets/feature_sweep_16.json`, not reproduced in
full above) were inspected as sanity checks per position's current-default row and didn't surface
anything that would have overridden this call even if they'd been used as the criterion.

Every position's Disagreement Edge correlation is positive with a CI that excludes zero at its
current default (RB 0.233 (0.126, 0.388), QB 0.300 (0.115, 0.570), WR 0.235 (0.123, 0.342)) --
**except TE**, whose CI straddles zero at every variant tested (best case (-0.004, 0.503)). That's
a sample-size artifact (see Caveats), not evidence the model's TE disagreements are pure noise.

**Backtest report rebuild:** since no `PositionConfig` default changed, rebuilding
`build_backtest_report` for each position with the (unchanged) new defaults is a no-op against
the sweep's own current-default rows above -- those rows *are* the rebuilt report's numbers,
already produced with the red-zone-caching and ADP-crosswalk fixes in place. There is no
pre-sweep-vs-post-sweep movement to report beyond those two bug fixes, which is the expected
result of a "confirmed, not replaced" outcome.

## Follow-up (2026-07-06): simplify instead of confirm

On review, "no candidate clearly beats the current default" is a weak basis for keeping
a family at a *non-trivial* default (RB/WR/TE's `recency_weighted` multi-season window,
QB's `last_n`, Depth-Chart Competition's `True`) — those defaults were carrying complexity
adopted under the retired matched-population criterion, and this sweep gave them no
independent justification under Disagreement Edge either. Rather than confirm them by
default, every `PositionConfig`/pipeline default now takes the simplest variant of each
family: `multi_season_window="none"` for all positions, `include_depth_chart_competition=False`.
`experience_feature` and `sos_feature` were already at their simplest ("none") and are
unchanged. Per the table above, this costs at most ~0.01-0.05 correlation at the point
estimate for any position/family — indistinguishable from noise per this sweep's own
"clearly beats" bar — in exchange for a materially simpler feature set.

## Caveats

- **QB and TE are underpowered for this exercise.** The large-disagreement win-rate is computed
  over only 9-15 (QB) or 1-2 (TE) player-seasons per config -- far too few to distinguish "clearly
  better" from "clearly worse," let alone from noise. QB's `career` multi-season variant and TE's
  `career` variant both show the highest point estimates of any config in this sweep, but with
  samples this thin that's as likely to be luck as signal. Revisit once more backtest seasons
  accumulate.
- Matched-population Spearman still favors ADP over the model at every position (e.g. RB: model
  ≈0.30-0.33 vs. ADP 0.499; QB: model ≈0.07-0.15 vs. ADP 0.398) -- unchanged by this sweep, since
  no swap was adopted. Closing that gap is v2's stated goal (#11), not this issue's.
- This sweep only compares each family's candidates against the position's *other* current
  defaults held fixed (one-at-a-time, per `build_backtest_report`'s documented sweep contract) --
  it doesn't explore interactions between families (e.g. whether QB `career` multi-season window
  paired with a non-default SOS feature would separate more clearly). A joint sweep would need a
  much larger config grid and is out of scope here.
- The red-zone play-by-play caching fix and the ADP-crosswalk null-name fix are general
  performance/correctness fixes surfaced by this work, not scoped to feature selection -- they
  make every future backtest run (not just this sweep) faster and correct for seasons touching the
  affected roster row.
