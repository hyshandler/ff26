from dataclasses import dataclass

import pandas as pd

from ff_model.depth_chart import depth_chart_competition_history
from ff_model.features import (
    MultiSeasonWindow,
    multi_season_career_averages,
    multi_season_last_n_averages,
    multi_season_recency_weighted_averages,
)
from ff_model.games_played import estimate_games_played
from ff_model.nflverse import (
    load_draft_picks,
    load_injury_reports,
    load_offense_snap_pct,
    load_red_zone_rush_attempts,
    load_seasonal_rosters,
    load_weekly_stats,
    pfr_id_crosswalk,
)
from ff_model.position_config import POSITION_CONFIGS
from ff_model.position_model import build_position_model_projections, multi_season_stat_columns
from ff_model.scoring import PPR, QUANTILE_SUFFIXES, score_projections
from ff_model.veterans import veteran_player_ids

MIN_CAREER_GAMES = 16
"""Minimum career games before the rookie season to count as a Veteran.

One full season's worth of games, per ADR-0006 — tunable, not fixed by the PRD.
"""

EARLIEST_SEASON = 2006
"""Earliest season fed into training, across the whole pipeline.

Matches the modern nflverse era used in prior fantasy-projection modeling work
(see docs/research/fantasy-football-projection-modeling.md, 2006-2020 range).
The Walk-Forward Backtest trains on seasons EARLIEST_SEASON..train_through_season
(an expanding window), predicts train_through_season + 1, and slides forward.
"""


@dataclass(frozen=True)
class PositionProjections:
    position: str
    train_through_season: int
    target_season: int
    projections: pd.DataFrame


def build_position_projections(
    position: str,
    train_through_season: int,
    target_season: int,
    include_depth_chart_competition: bool = True,
    multi_season_window: MultiSeasonWindow | None = None,
    multi_season_n_seasons: int | None = None,
    multi_season_decay: float | None = None,
) -> PositionProjections:
    """Ingest -> filter -> predict -> output, for one position and one Walk-Forward split.

    `multi_season_window` opts into a multi-season memory feature alongside the default
    single-season `trailing_avg_*` features: "career" (full career to date, games-weighted),
    "last_n" (most recent `multi_season_n_seasons` prior seasons), or "recency_weighted"
    (exponential decay across prior seasons, rate `multi_season_decay`). Left as None (default),
    each argument falls back to `PositionConfig`'s per-position winner from the multi-season
    memory backtest (see `docs/research/multi-season-memory-features.md`) -- pass "none"
    explicitly to reproduce the original single-season-only baseline.
    """
    if target_season != train_through_season + 1:
        raise ValueError(
            "target_season must be train_through_season + 1 for the Walk-Forward Backtest"
        )
    if position not in POSITION_CONFIGS:
        raise ValueError(f"Unknown position: {position!r}")
    config = POSITION_CONFIGS[position]
    if multi_season_window is None:
        multi_season_window = config.multi_season_window
    if multi_season_n_seasons is None:
        multi_season_n_seasons = config.multi_season_n_seasons
    if multi_season_decay is None:
        multi_season_decay = config.multi_season_decay

    weekly_seasons = list(range(EARLIEST_SEASON, train_through_season + 1))
    weekly_all_positions = load_weekly_stats(weekly_seasons)
    weekly = weekly_all_positions.loc[weekly_all_positions["position"] == position]

    rosters = load_seasonal_rosters(weekly_seasons + [target_season])
    rosters = rosters.loc[rosters["position"] == position]

    eligible = veteran_player_ids(
        rosters, weekly, season=target_season, min_career_games=MIN_CAREER_GAMES
    )

    pfr_id_by_player_id = pfr_id_crosswalk(rosters)
    red_zone_carries = (
        load_red_zone_rush_attempts(weekly_seasons)
        if config.needs_red_zone_data
        else pd.DataFrame(columns=["season", "week", "player_id", "red_zone_carries"])
    )
    snap_pct = load_offense_snap_pct(weekly_seasons, pfr_id_by_player_id)

    draft_picks = load_draft_picks(weekly_seasons + [target_season])
    depth_chart_history = depth_chart_competition_history(
        rosters, draft_picks, weekly_seasons + [target_season]
    )

    multi_season_history = None
    if multi_season_window != "none":
        stat_columns = multi_season_stat_columns(config)
        all_seasons = weekly_seasons + [target_season]
        if multi_season_window == "career":
            multi_season_history = multi_season_career_averages(
                weekly, stat_columns, all_seasons
            )
        elif multi_season_window == "last_n":
            multi_season_history = multi_season_last_n_averages(
                weekly, stat_columns, all_seasons, n_seasons=multi_season_n_seasons
            )
        else:
            multi_season_history = multi_season_recency_weighted_averages(
                weekly, stat_columns, all_seasons, decay=multi_season_decay
            )

    predictions = build_position_model_projections(
        config,
        weekly_all_positions,
        red_zone_carries,
        snap_pct,
        depth_chart_history,
        train_through_season=train_through_season,
        target_season=target_season,
        eligible_player_ids=eligible,
        include_depth_chart_competition=include_depth_chart_competition,
        multi_season_history=multi_season_history,
    )

    injury_reports = load_injury_reports(weekly_seasons)
    games_played_estimate = estimate_games_played(
        rosters,
        weekly,
        injury_reports,
        train_through_season=train_through_season,
        target_season=target_season,
        player_ids=eligible,
    )
    predictions["games_played_estimate"] = predictions["player_id"].map(games_played_estimate)

    predictions = score_projections(predictions, PPR)
    predictions = add_full_projection(predictions)

    player_names = rosters.drop_duplicates("player_id").set_index("player_id")["player_name"]
    predictions.insert(1, "player_name", predictions["player_id"].map(player_names))
    predictions.insert(0, "target_season", target_season)
    predictions.insert(0, "position", position)

    return PositionProjections(
        position=position,
        train_through_season=train_through_season,
        target_season=target_season,
        projections=predictions,
    )


def add_full_projection(projections: pd.DataFrame) -> pd.DataFrame:
    """Combines PPG Projection with Games-Played Estimate into the full Projection.

    Per ADR-0008: full_projection_{p10,p50,p90} = fantasy_points_{p10,p50,p90} *
    games_played_estimate, added alongside both source columns rather than
    replacing them, so the components stay visible separately.
    """
    result = projections.copy()
    for suffix in QUANTILE_SUFFIXES:
        result[f"full_projection_{suffix}"] = (
            result[f"fantasy_points_{suffix}"] * result["games_played_estimate"]
        )
    return result


def combine_position_projections(
    projections: list[PositionProjections],
) -> pd.DataFrame:
    """Concatenate every position's output into one combined dataframe.

    A thin combination step, not a modeling seam: positions predict different raw
    stats (QB's passing_yards_p50 has no RB equivalent), so the combined frame's
    columns are the union across positions, NaN-filled wherever a stat doesn't
    apply to a given row's position.
    """
    return pd.concat([p.projections for p in projections], ignore_index=True)
